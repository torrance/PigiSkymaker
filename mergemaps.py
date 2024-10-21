from argparse import ArgumentParser

from astropy.io import fits
import numpy as np

parser = ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("images", nargs="+")
args = parser.parse_args()

hdus = [fits.open(img)[0] for img in args.images]

print(repr(hdus[0].header))
print(repr(hdus[1].header))

# Calculate bounds
xmin, xmax, ymin, ymax = 0, 0, 0, 0
for hdu in hdus:
    x0 = hdu.header["CRPIX1"] - 1  # subtract 1 conversion to 0-indexed
    y0 = hdu.header["CRPIX2"] - 1

    # Compute pixel offset of each bound wrt the reference pixel
    xmin = min(xmin, int(0 - x0))
    xmax = max(xmax, int(hdu.header["NAXIS1"] - x0))
    ymin = min(ymin, int(0 - y0))
    ymax = max(ymax, int(hdu.header["NAXIS2"] - y0))

    print(xmin, xmax, xmax - xmin, ymin, ymax, ymax - ymin)

merged = np.zeros((ymax - ymin, xmax - xmin), dtype=hdus[0].data.dtype)
merged[:] = np.nan

# Add images
for hdu in hdus:
    x0 = hdu.header["CRPIX1"] - 1 + xmin  # subtract 1 conversion to 0-indexed
    y0 = hdu.header["CRPIX2"] - 1 + ymin

    merged[int(0 - y0):int(hdu.header["NAXIS2"] - y0), int(0 - x0):int(hdu.header["NAXIS1"] - x0)] = hdu.data[:, :]

header = hdus[0].header
header["CRPIX1"] = 1 - xmin
header["CRPIX2"] = 1 - ymin
fits.writeto(args.output, data=merged, header=header, overwrite=True)
