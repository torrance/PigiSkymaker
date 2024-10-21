from argparse import ArgumentParser
import math

from astropy.coordinates import AltAz, SkyCoord, EarthLocation
from astropy.io import fits
from astropy.time import  Time
from casacore.tables import table, taql
import matplotlib.pyplot as plt
from mwa_hyperbeam import FEEBeam
from numba import njit, prange, cuda
import numpy as np

DATACOL = "DATA"

parser = ArgumentParser()
parser.add_argument("mset", nargs=1)
parser.add_argument("--gpuid", type=int, default=0)
args = parser.parse_args()
print(f"Selected GPU id = {args.gpuid % len(cuda.gpus)}")
cuda.select_device(args.gpuid % len(cuda.gpus))

def radec_to_lmndash(ra, dec, ra0, dec0):
    l = np.cos(dec) * np.sin(ra - ra0)
    m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(ra - ra0)

    # Determine sign of n, depending on whether (ra, dec) lies within the hemisphere
    # created by (ra0, dec0)
    nsign = np.sign(
        np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0)
    )

    # Calculate ndash = 1 - n
    # where n = sqrt(1 - l^2 - m^2)
    r2 = l**2 + m**2
    r2[r2 > 1] = 1
    ndash = r2 / (1 + nsign * np.sqrt(1 - r2))

    return np.array([l, m, ndash]).T

@njit(parallel=True)
def predict(data, uvws, lmndash, fluxes, jones):
    for i in prange(data.shape[0]):
        for j in range(data.shape[1]):
            u, v, w = uvws[i, j, :]

            for ((l, m, ndash), flux, (j1, j2, j3, j4)) in zip(lmndash, fluxes[j, :], jones):
                # Note v => -v for MWA
                phase = -2 * np.pi * (-u * l - v * m + w * ndash)
                I = flux * (np.cos(phase) + 1j * np.sin(phase))

                # Perform J * I * J^H=
                # The RHS should be transposed, but we simply do the matrix multiplication
                # by the dot product along its rows, avoiding the transpose
                data[i, j, 0] += I * (j1 * np.conj(j1) + j2 * np.conj(j2))
                data[i, j, 1] += I * (j1 * np.conj(j3) + j2 * np.conj(j4))
                data[i, j, 2] += I * (j3 * np.conj(j1) + j4 * np.conj(j2))
                data[i, j, 3] += I * (j3 * np.conj(j3) + j4 * np.conj(j4))

@cuda.jit
def predictcuda(data, uvws, lmndash, fluxes, jones):
    # Accumulation variable
    tmp = cuda.local.array(4, dtype=data.dtype)

    for idx in range(cuda.grid(1), data.shape[0] * data.shape[1], cuda.gridsize(1)):
        i = idx // data.shape[1]
        j = idx % data.shape[1]

        tmp[:] = 0

        u, v, w = uvws[i, j, :]
        for ((l, m, ndash), flux, (j1, j2, j3, j4)) in zip(lmndash, fluxes[j, :], jones):
            # Note v => -v for MWA
            phase = -2 * np.pi * (-u * l - v * m + w * ndash)
            I = flux * (np.cos(phase) + 1j * np.sin(phase))

            # Perform J * I * J^H=
            # The RHS should be transposed, but we simply do the matrix multiplication
            # by the dot product along its rows, avoiding the transpose
            tmp[0] += I * (j1 * j1.conjugate() + j2 * j2.conjugate())
            tmp[1] += I * (j1 * j3.conjugate() + j2 * j4.conjugate())
            tmp[2] += I * (j3 * j1.conjugate() + j4 * j2.conjugate())
            tmp[3] += I * (j3 * j3.conjugate() + j4 * j4.conjugate())

        data[i, j, 0] = tmp[0]
        data[i, j, 1] = tmp[1]
        data[i, j, 2] = tmp[2]
        data[i, j, 3] = tmp[3]

# Get catalogue columns that we need
ggsm = np.load("/home/torrance/skymaker/catalogue-franzen.npy")
ras = np.deg2rad(ggsm["RAJ2000"])
decs = np.deg2rad(ggsm["DEJ2000"])
S200 = np.array(ggsm["S_200"])
alphas = np.array(ggsm["alpha"])

# For rows missing alpha, set to median value
idxs = np.isfinite(alphas)
alpha = np.median(alphas[idxs])
alphas[~idxs] = alpha
print("Substituting alpha =", alpha, "for missing alpha values")

coords = SkyCoord(ras, decs, unit=("rad", "rad"))

# Open mset
mset = table(args.mset, readonly=False)

# Get mset metadata
freqs = mset.SPECTRAL_WINDOW.getcol("CHAN_FREQ")[0]
timesteps = np.unique(mset.getcol("TIME_CENTROID"))
ra0, dec0 = mset.FIELD.getcell("PHASE_DIR", 0)[0]
delays = mset.MWA_TILE_POINTING.getcell("DELAYS", 0)

# Restrict FOV for faint sources
idxs = (
    (coords.separation(SkyCoord(ra0, dec0, unit=("rad", "rad"))).deg < 30) |
    ((coords.separation(SkyCoord(ra0, dec0, unit=("rad", "rad"))).deg < 45) & (S200 > 50e-3)) |
    ((coords.separation(SkyCoord(ra0, dec0, unit=("rad", "rad"))).deg < 90) & (S200 > 100e-3))
)
ras, decs, S200, alphas, coords = ras[idxs], decs[idxs], S200[idxs], alphas[idxs], coords[idxs]

print(f"Phase center: Ra = {np.rad2deg(ra0)} Dec = {np.rad2deg(dec0)}")
print(f"MWA delays: {delays}")

# Precalculate
lmndash = radec_to_lmndash(ras, decs, ra0, dec0)

# Create beam object
beam = FEEBeam()

# MWA Location
mwa = EarthLocation.from_geodetic(
    116. + 40. / 60. + 14.93 / 3600., -(26. + 42. / 60. + 11.95 / 3600.)
)

# freqs = freqs[:48]
chunksize = 6

for timestep in timesteps:
    submset = taql("select * from $mset where TIME = $timestep")
    uvws = submset.getcol("UVW")
    mjd = Time(timestep / (24 * 60 * 60), format="mjd")

    # Process channels chunks
    for colstart in range(0, len(freqs), chunksize):
        colend = min(colstart + chunksize, len(freqs))

        print(f"\rProcessing time {mjd}... {colstart / len(freqs) * 100:.1f}% complete", end="", flush=True)

        ncols = colend - colstart
        midfreq = np.sum(freqs[colstart:colend]) / ncols

        # Calculate AltAz of RaDec objects
        altazs = coords.transform_to(AltAz(obstime=mjd, location=mwa))

        # Filter out low- and below-horizon objects
        idxs = altazs.alt.deg > 5

        # Calculate Jones matrices
        jones = beam.calc_jones_array(
            altazs[idxs].az.rad, np.pi/2 - altazs[idxs].alt.rad, midfreq, delays, [1] * 16,
            True, mwa.lat.rad, False
        )

        fluxes = S200[None, idxs] * (freqs[colstart:colend, None] / 200e6)**alphas[None, idxs]
        lmbda = 299792458 / freqs[colstart:colend]

        # data = np.zeros((submset.nrows(), ncols, 4), dtype=np.complex128)
        # predict(
        #     data,
        #     uvws[:, None, :] / lmbda[None, :, None],
        #     lmndash[idxs],
        #     fluxes,
        #     jones
        # )

        data_h = cuda.device_array((submset.nrows(), ncols, 4), dtype=np.complex64)
        predictcuda[math.ceil(submset.nrows() * ncols / 512), 512](
            data_h,
            cuda.to_device((uvws[:, None, :] / lmbda[None, :, None]).astype(np.float32)),
            cuda.to_device(lmndash[idxs].astype(np.float32)),
            cuda.to_device(fluxes.astype(np.float32)),
            cuda.to_device(jones.astype(np.complex64))
        )
        data = data_h.copy_to_host()

        if np.any(~np.isfinite(data)):
            print("Alert! NaN detected in data")

        submset.putcolslice(DATACOL, data, (colstart, 0), (colend - 1, 3))

    submset.flush()
    print(f"\rProcessing time {mjd}... 100.0% complete")

mset.flush()
mset.close()
