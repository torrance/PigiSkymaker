from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# Filter GLEAM down only to bright objects for now
ggsm = fits.open("/home/torrance/GGSM.fits")[1].data

bins = np.linspace(-0.75, 2.75, int(3.5 * 4 + 1))
binwidth = 0.25

counts, _ = np.histogram(np.log10(ggsm.field("S_200")), bins)
xs = (bins[:-1] + bins[1:]) / 2
m, c = np.polyfit(xs, np.log10(counts / binwidth), 1)

print("m =", m, "c =", c)

# We need to know how sources to draw from the probability distribution
# Integrate 10^(mx + c) between -2 -> 3
integral = lambda x: 10**(m * x + c) / (m * np.log(10))
N = integral(3) - integral(-2)
print("N =", N)

# Draw random dataset, and reflect and shift to match our negatively sloped dataset
data = np.random.power(-m, int(N))
data = -np.log10(data) - 2

# Merge random data sources with existing ggsm data
# First, finely partition GGSM
bins = np.linspace(-2, 3, 5 * 16 + 1)
binwidth = bins[1] - bins[0]
counts, _ = np.histogram(np.log10(ggsm.field("S_200")), bins)

# Next iterate through each data source, and if a corresponding source exists
# in the associated GGSM partition, then remove it and deduct the GGSM source count by one
idxs = np.zeros_like(data, dtype=bool)
for i, x in enumerate(data):
    # Calculate partition index
    j = int((x + 2) / binwidth)
    if j >= 0 and j < len(counts) and counts[j] > 0:
        counts[j] -= 1
    else:
        idxs[i] = True

# Finally, merge the two source sets
data = data[idxs]
merged = np.concatenate([np.log10(ggsm.field("S_200")), data])

# Plot the datasets together to provide visual check
plt.hist(
    [merged, data, np.log10(ggsm.field("S_200"))],
    np.linspace(-2, 4, 6 * 4 + 1),
    label=["Combined", "Synthetic", "GGSM"]
)
plt.legend()
plt.plot(xs, 10**(m * xs + c) * 0.25, linestyle="dashed")
plt.yscale("log")
plt.show()
exit()

# Randomly place synthetic data across the celestial sphere
ras = np.random.uniform(0, 2 * np.pi, size=len(data))
decs = np.pi / 2 - np.arccos(np.random.uniform(-1, 1, size=len(data)))

ras = np.concatenate([ggsm.field("RAJ2000"), np.rad2deg(ras)])
decs = np.concatenate([ggsm.field("DEJ2000"), np.rad2deg(decs)])

# Merge GGSM and synthetic data for alphas
alphas = np.empty(len(data), dtype=float)
alphas[:] = np.NaN
alphas = np.concatenate([ggsm.field("alpha"), alphas])

catalogue = np.empty(
    len(merged),
    dtype=[
        ("RAJ2000", float),
        ("DEJ2000", float),
        ("S_200", float),
        ("alpha", float)
    ]
)

catalogue["RAJ2000"][:] = ras
catalogue["DEJ2000"][:] = decs
catalogue["S_200"][:] = 10**merged
catalogue["alpha"][:] = alphas

# np.save("catalogue.npy", catalogue)