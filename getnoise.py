#! /usr/bin/env python3

from argparse import ArgumentParser

from astropy.io import fits
import numpy as np


parser = ArgumentParser()
parser.add_argument("fits")
args = parser.parse_args()

data = fits.open(args.fits)[0].data
median = np.median(data)
madm = np.median(np.abs(data - median))
noise = 1.4826 * madm

print(f"Noise: {noise}")
