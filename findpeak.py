import glob

from astropy.io import fits
import numpy as np

for f in glob.glob("/home/torrance/MWA/A2877/CHAN121/jointplusfranzen/residual-field??.fits"):
    data = np.abs(fits.open(f)[0].data)
    print(f"{f} max value {data.max() : .3f} ")
