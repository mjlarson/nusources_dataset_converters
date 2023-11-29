#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/icetray-start
#METAPROJECT icetray/v1.5.1
#!/usr/bin/env python3

import os, sys
from argparse import ArgumentParser
import numpy as np
import numpy.lib.recfunctions as rf

try: from tqdm import tqdm
except: tqdm = lambda x: x

#----------------------------------------------------
# Build our argument parser.
#----------------------------------------------------
parser = ArgumentParser(description="Base script for converting i3 files to NuSources-compatible"
                        " npy files for use in csky and skyllh.")
parser.add_argument("input", metavar="i3file", type=str,
                    nargs='+', default=[])
parser.add_argument("--output", "-o", dest="output", type=str,
                    action = "store", default="", required=True,
                    help = "The output npy filename and path to write to")
                    
args = parser.parse_args()

#----------------------------------------------------
# Read the files into a list. Assume they all have identical
# formatting and columns.
#----------------------------------------------------
output = []
for f in tqdm(sorted(args.input)):
    output.append(np.load(f))

#----------------------------------------------------
# Merge the files all at once
#----------------------------------------------------
output = np.concatenate(output)

#----------------------------------------------------
# And save the merged output to a new file
#----------------------------------------------------
np.save(args.output, output)

