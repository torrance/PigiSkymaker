from argparse import ArgumentParser
import pathlib


from astropy.io import fits
import numpy as np

parser = ArgumentParser()
parser.add_argument("directory", type=str, nargs="+")
parser.add_argument("--beamcorrect", type=bool)
parser.add_argument("--prefix", type=str, default="image")
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

print("--beamcorrect", args.beamcorrect)


img = fits.open(pathlib.PurePath(args.directory[0], f"{args.prefix}-field01.fits"))[0]
img.data[:, :] = 0

beam = fits.open(pathlib.PurePath(args.directory[0], "beam-field01.fits"))[0]
beam.data[:, :] = 0

for basedir in args.directory:
    try:
        img.data += fits.open(pathlib.PurePath(basedir, f"{args.prefix}-field01.fits"))[0].data
        if args.beamcorrect:
            beam.data += fits.open(pathlib.PurePath(basedir, "beam-field01.fits"))[0].data
        else:
            beam.data[:] += 1
    except Exception as e:
        print(f"Failed to process directory: {basedir}")

img.data /= beam.data

img.writeto(args.output, overwrite=True)
