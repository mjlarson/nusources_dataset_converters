#!/usr/bin/env python3
from os.path import join, expandvars
from glob import glob
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--indir", type=str, default=[], action='append', required=True)
parser.add_argument("-o", "--outfile", type=str, required=True)

args = parser.parse_args()

filenames = sorted(glob(join(expandvars(indir), "*.npy")) for indir in args.indir)
output = [np.load(f) for f in tqdm(filenames)]
np.save(args.outfile, np.concatenate(output))
