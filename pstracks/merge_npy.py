#!/usr/bin/env python3
from concurrent.futures import ProcessPoolExecutor
from os.path import join, expandvars
from glob import glob
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--indir", type=str, required=True)
parser.add_argument("-o", "--outfile", type=str, required=True)
parser.add_argument("--nfiles", type=int, default=-1)
parser.add_argument("--ncpu", type=int, default=1)

args = parser.parse_args()

filenames = sorted(glob(join(expandvars(args.indir), "*.npy")))

if args.ncpu == 1:
    output = [np.load(f) for f in tqdm(filenames)]
else:
    executor = ProcessPoolExecutor(args.ncpu)
    try:
        output = [_ for _ in tqdm(executor.map(np.load, filenames, chunksize=100), total=len(filenames))]
    except KeyboardInterrupt:
        executor.shutdown(wait=False)
        raise

output = np.concatenate(output)
if 'ow' in output.dtype.names:
    if args.nfiles == -1:
        print("Merging mc, but didn't receive a manual --nfiles argument."
              " Just using the number of observed files.")
        output['ow'] /= len(filenames)
    else:
        output['ow'] /= args.nfiles

np.save(args.outfile, output)
