from argparse import ArgumentParser

from astropy.coordinates import AltAz, SkyCoord, EarthLocation
from astropy.time import  Time
from casacore.tables import table
import matplotlib.pyplot as plt
from mwa_hyperbeam import FEEBeam
import numpy as np

parser = ArgumentParser()
parser.add_argument("mset")
args = parser.parse_args()

# Open mset
mset = table(args.mset, readonly=False)
midtimestep = np.mean(np.unique(mset.getcol("TIME_CENTROID")))
midfreq = np.mean(mset.SPECTRAL_WINDOW.getcol("CHAN_FREQ")[0])
mjd = Time(midtimestep / (24 * 60 * 60), format="mjd")
delays = mset.MWA_TILE_POINTING.getcell("DELAYS", 0)

catalogue = np.load("/home/torrance/skymaker/catalogue-franzen.npy")

ras, decs, fluxes = catalogue["RAJ2000"], catalogue["DEJ2000"], catalogue["S_200"]

# Trim to just sources with > 1 Jy true flux
idxs = fluxes > 1
ras, decs, fluxes = ras[idxs], decs[idxs], fluxes[idxs]

# Create beam object
beam = FEEBeam()

# MWA Location
mwa = EarthLocation.from_geodetic(
    116. + 40. / 60. + 14.93 / 3600., -(26. + 42. / 60. + 11.95 / 3600.)
)

# Calculate AltAz of RaDec objects
coords = SkyCoord(ras, decs, unit=("deg", "deg"))
altazs = coords.transform_to(AltAz(obstime=mjd, location=mwa))

# Remove sources < 5 degrees of the horizon
idxs = altazs.alt.deg > 5
altazs = altazs[idxs]
coords = coords[idxs]
fluxes = fluxes[idxs]

# Calculate Jones matrices
jones = beam.calc_jones_array(
    altazs.az.rad, np.pi/2 - altazs.alt.rad, midfreq, delays, [1] * 16,
    True, mwa.lat.rad, False
)
jones = jones.reshape(-1, 2, 2)

fluxes = 0.5 * fluxes * np.real(np.trace(np.matmul(jones, np.conj(np.transpose(jones, [0, 2, 1]))), axis1=1, axis2=2))

idxs = (fluxes > 0.1) # & (coords.dec.deg < 0)

# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# plt.scatter([idxs].az.rad, altazs[idxs].alt.deg, s=fluxes[idxs] * 4, linewidth=0)
plt.scatter(coords[idxs].ra.deg, coords[idxs].dec.deg, s=fluxes[idxs] * 8, c=np.log10(fluxes[idxs]), linewidth=0)
plt.colorbar()
plt.show()
