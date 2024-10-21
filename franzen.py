from astropy.coordinates import SkyCoord
from astropy.io import fits
from numba import njit
import numpy as np
import matplotlib.pyplot as plt

@njit
def _invertcdf(cdf, xs, idxs):
    for i, x in enumerate(xs):
        for j, val in enumerate(cdf):
            if val >= x:
                idxs[i] = j
                break
        else:
            idxs[i] = len(xs) - 1

def invertcdf(cdf, xs):
    idxs = np.empty_like(xs, dtype=int)
    _invertcdf(cdf, xs, idxs)
    return idxs

def dlogNdlogS(logS):
    return (3.52 + (0.307 - 2.5) * logS - 0.388 * logS**2 - 0.0404 * logS**3 + 0.0351 * logS**4 + 0.006 * logS**5)

# Create cumulative distribution function for Franzen
bins = np.linspace(-3, 2, 10000)
midbins = (bins[1:] + bins[:-1]) / 2
binwidth = bins[1] - bins[0]

# Frazen function is at 154MHz, we want to shift to 200MHz assuming alpha = -0.8
counts = 10**dlogNdlogS(midbins - 0.8 * np.log10(154 / 200)) * (10**bins[1:] - 10**bins[:-1])
N = int(np.sum(counts) * 4 * np.pi)
cdf = np.cumsum(counts) / np.sum(counts)

print("Full sky N (1mJy < 100 Jy) = ", N)

# Draw random uniform values
dist = np.random.uniform(size=N)
dist = midbins[invertcdf(cdf, dist)]

# Add some jitter within each bin
dist += np.random.uniform(low=-binwidth/2, high=binwidth/2, size=len(dist))

print(f"Synthetic sources {10**min(dist)} < {10**max(dist)}")

# Compare to GGSM catalogue
ggsm = fits.open("/home/torrance/GGSM.fits")[1].data

ras, decs = ggsm.field("RAJ2000"), ggsm.field("DEJ2000")
fluxes = ggsm.field("S_200")

# Limit to just Southern Hemisphere sources (e.g. 2pi steradians)
coords = SkyCoord(ras, decs, unit=("deg", "deg"))
idxs = SkyCoord(0, -90, unit=("deg", "deg")).separation(coords).deg < 90
fluxes = fluxes[idxs]

# Plot logdn / logdS
bins = np.linspace(-3, 3, 61)
midbins = (bins[:-1] + bins[1:]) / 2
counts, _ = np.histogram(np.log10(fluxes), bins)

# Normalize count based on binwidth, then based on sky area
counts = counts / ((10**bins)[1:] - (10**bins)[:-1])
counts /= 2 * np.pi

distcounts = np.histogram(dist, bins)[0]
distcounts = distcounts / ((10**bins)[1:] - (10**bins)[:-1])
distcounts /= 4 * np.pi

plt.scatter(10**midbins, counts, color="black", marker="x")
plt.plot(10**midbins, 10**dlogNdlogS(midbins - 0.8 * np.log10(154 / 200)))
plt.plot(10**midbins, distcounts)
plt.xscale("log")
plt.yscale("log")
plt.ylim(ymin=1e-9)
plt.show()

# Randomly place synthetic data across the celestial sphere
ras = np.random.uniform(0, 2 * np.pi, size=len(dist))
decs = np.pi / 2 - np.arccos(np.random.uniform(-1, 1, size=len(dist)))

# Set alphas to fixed -0.8
alphas = np.zeros(len(dist), dtype=float)
alphas[:] = -0.8

# Save catalogue
catalogue = np.empty(
    len(dist),
    dtype=[
        ("RAJ2000", float),
        ("DEJ2000", float),
        ("S_200", float),
        ("alpha", float)
    ]
)

catalogue["RAJ2000"][:] = np.rad2deg(ras)
catalogue["DEJ2000"][:] = np.rad2deg(decs)
catalogue["S_200"][:] = 10**dist
catalogue["alpha"][:] = alphas

# np.save("catalogue-franzen.npy", catalogue)