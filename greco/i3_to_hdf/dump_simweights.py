import os, sys, tables
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from icecube import dataio, dataclasses, icetray, simclasses
from simweights import GenieWeighter, NuGenWeighter
import pickle
from argparse import ArgumentParser

parser = ArgumentParser(description="Base script for converting i3 files to NuSources-compatible"
                        " npy files for use in csky and skyllh.")
parser.add_argument(dest="infiles", type=str,
                    nargs='+', default=[])
parser.add_argument("--outfile", "-o", type=str,
                    action = "store", default="", required=True,
                    help = "The output npy filename and path to write to")
parser.add_argument("--nfiles", type=int, default=None, 
                    help = "If given, use this number of files. Otherwise count the number of input files.")
args = parser.parse_args()

# Figure out the nfiles
if args.nfiles is not None:
    nfiles = args.nfiles
else:
    nfiles = len(args.infiles)

#---------------------------------------------
# Prepare to have both genie and nugen weighters
#---------------------------------------------
weighters = {}

#---------------------------------------------
# Create the structure for the output file
#---------------------------------------------
merged = pd.HDFStore('temp_merged_sim.hdf5')

for f in tqdm(sorted(args.infiles)):
    x = tables.open_file(f






# Decide if this is genie or nugen and build the appropriate objects
nugen, genie = False, False
if "nugen" in f.lower() or args.nugen:
    hdf_like = {"I3MCWeightDict": {}}
    nugen = True
    genie = False
else:
    # This is likely genie-reader for now. Assume that it will be and adjust it later once genie-icetray
    # simulation is available.
    hdf_like = {"I3MCWeightDict": {}, "I3GENIEResultDict": {}}
    nugen = False
    genie = True

# Start reading the files
i3file = dataio.I3File(f)
for frame in i3file:
    if not frame.Stop == icetray.I3Frame.Physics: continue

    if genie or nugen:
        for key, val in dict(frame["I3MCWeightDict"]).items():
            try: 
                hdf_like["I3MCWeightDict"][key].append(val)
            except KeyError: 
                print(key)
                hdf_like["I3MCWeightDict"][key] = [val,]

    if genie:
        for key, val in dict(frame['I3GENIEResultDict']).items():
            if type(val) not in (float, int): continue
            try: 
                hdf_like["I3GENIEResultDict"][key].append(val)
            except KeyError: 
                hdf_like["I3GENIEResultDict"][key] = [val,]

# Create a weighter object of the right kind
hdf_like = {key:pd.DataFrame(val) for key, val in hdf_like.items()}
if nugen:
    weighter = NuGenWeighter(hdf_like, nfiles=nfiles)
if genie:
    weighter = GenieWeighter(hdf_like, nfiles=nfiles)

print(f"Writing weighter to {args.outfile}")
pickle.dump(weighter, open(args.outfile, 'wb'), protocol=-1)
