#! /usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

from casacore.tables import table
import numpy as np


parser = ArgumentParser()
parser.add_argument("--noise", type=float, required=True)
parser.add_argument("mset")
args = parser.parse_args()

path = Path(args.mset)
tbl = table(str(path))

path = path.parent / Path(f"{path.stem}-noisy{path.suffix}")
tbl.copy(str(path), deep=True)
tbl = table(str(path), readonly=False)

data = tbl.getcol("DATA")

data += args.noise * np.random.randn(*data.shape)
data += args.noise * 1j * np.random.randn(*data.shape)

tbl.putcol("DATA", data)

