from astropy.io import fits
from astropy.coordinates import SkyCoord
import numpy as np


catalogue = np.load("catalogue.npy")

ras, decs, fluxes = catalogue["RAJ2000"], catalogue["DEJ2000"], catalogue["S_200"]

coords = SkyCoord(ras, decs, unit=("deg", "deg"))

idxs = (
    (SkyCoord(17.42085, -45.78194, unit=("deg", "deg")).separation(coords).deg < 2)
)

ras, decs, fluxes = ras[idxs], decs[idxs], fluxes[idxs]

with open("regions.reg", "w") as f:
    print("# Region file format: DS9 version 4.0", file=f)
    print('global color=green font="helvetica 10 normal roman" edit=1 move=1 delete=1 highlite=1 include=1 wcs=wcs', file=f)
    print("J2000", file=f)

    for ra, dec, flux in zip(ras, decs, fluxes):
        # print(f"point {ra} {dec} # point=circle size={np.log(flux)}", file=f)
        # print(f"circle {ra} {dec} 0.01", file=f)
        print(f"circle {ra} {dec} {(np.log10(flux) + 3) / 100} # fill=1", file=f)
