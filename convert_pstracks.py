#!/usr/bin/env python3

import os, sys
from glob import glob
from argparse import ArgumentParser
import numpy as np
import numpy.lib.recfunctions as rf

from icecube import dataclasses, dataio, icetray, simclasses, paraboloid
from icecube.icetray import I3Units as i3u
from icecube.astro import dir_to_equa

try: from tqdm import tqdm
except: tqdm = lambda x: x

#----------------------------------------------------
# Build our argument parser.
#----------------------------------------------------
parser = ArgumentParser(description="Base script for converting i3 files to NuSources-compatible"
                        " npy files for use in csky and skyllh.")
parser.add_argument("input", metavar="i3file", type=str,
                    nargs="+", default=[],
                    help='Directory or file to include')
parser.add_argument("--output", "-o", dest="output", type=str,
                    action = "store", default="", required=True,
                    help = "The output npy filename and path to write to")
parser.add_argument("--nfiles", "-n", type=int, action='store', required=True,
                    help = "Number of files initially processed (including any that had 0 events passing!)")
parser.add_argument("--nu_fraction", type=float, action="store", default=0.5,
                    help="What fraction of this simulation is neutrinos? Note that if there's an S-frame that's"
                    " compatible with simweights or a TypeWeight key in the I3MCWeightDict, this option will be"
                    " ignored in favor of what's in the file.")
parser.add_argument("--test", action='store_true', default=False,
                    help="If given, only process 10 files.")
args = parser.parse_args()


#----------------------------------------------------
# Set up our output
#----------------------------------------------------
output = []
dtype = [('run', np.int64), ('subevent', np.int32), ('event', np.int64),
         ('time', np.float64), ('logE', np.float32),
         ('ra', np.float32), ('dec', np.float32), 
         ('sigma', np.float32), ('sigma_noCorrection', np.float32),
         ('passed_IceTopVeto', bool)]

is_mc = False
mc_output = []
mc_dtype = [('ow', np.float64), ('trueE', np.float32),
            ('trueRa', np.float32), ('trueDec', np.float32),
            ('trueAzi', np.float32), ('trueZen', np.float32)]

#----------------------------------------------------
# Build the file list from the input
#----------------------------------------------------
filelist = []
for path in args.input:
    if os.path.isdir(path):
        filelist += glob(os.path.join(path, "*.i3*"))
    elif os.path.isfile(path):
        filelist.append(path)

if args.test and len(filelist) > 10:
    filelist = filelist[:10]

#----------------------------------------------------
# Start walking through the files
#----------------------------------------------------
for i3filename in tqdm(filelist):
    i3file = dataio.I3File(i3filename, 'r')

    while i3file.more():
        frame = i3file.pop_frame()
        
        #===========================================
        # Wait for P-frames
        #===========================================
        if not frame.Stop == icetray.I3Frame.Physics: continue
            
        #===========================================
        # Get the event information
        #===========================================
        header = frame['I3EventHeader']
        mjd = header.start_time.mod_julian_day_double

        # And the recos
        direction = frame['SplineMPE_l4'].dir
        energy = frame['SplineMPEMuEXDifferential'].energy

        # And the estimated angular uncertainty. 
        # I think this should have a sindec term, but 
        # this matches what was done in the past for now.
        if frame['SplineMPE_l4Paraboloid'].fit_status == 0:
            sigma = np.hypot(frame['SplineMPE_l4ParaboloidFitParams'].pbfErr1, 
                             frame['SplineMPE_l4ParaboloidFitParams'].pbfErr2) / np.sqrt(2)
        else:
            sigma = frame['SplineMPEBootstrapVectStats']['median']
            
        # Convert from local to equatorial coordinates
        ra, dec = dir_to_equa(zenith=direction.zenith,
                              azimuth=direction.azimuth,
                              mjd = mjd)

        # And check the IceTop veto
        if frame.Has("IceTopVeto"): pass_veto = not frame["IceTopVeto"].value
        else:                       pass_veto = True

        #===========================================
        # And write it out. The order here MUST match
        # the order in the `dtype` variable above or
        # else you'll end up with wrong labels.
        #===========================================
        current_event = [header.run_id,          # run
                         header.sub_event_id,    # subevent
                         header.event_id,        # event
                         mjd,                    # time
                         np.log10(energy),       # logE
                         ra,                     # ra
                         dec,                    # dec
                         sigma,                  # sigma
                         sigma,                  # sigma_noCorrection
                         pass_veto]              # passed_icetopVeto
        #===========================================
        # Monte Carlo keys:
        # Handle the MC-only keys if we have any mctree-like
        #===========================================
        mctree_name = [key for key in frame.keys() if key in ['I3MCTree', 'I3MCTree_preMuonProp']]

        if len(mctree_name) > 0:
            is_mc = True

            #===========================================
            # We'll get the primary info from the mctree
            #===========================================
            mctree = frame[mctree_name[0]]
            primary = dataclasses.get_most_energetic_neutrino(mctree)

            #===========================================
            # Oneweight nonsense. This is a placeholder that should
            # work for cases when we have only non-overlapping MC sets.
            #===========================================
            mcweightdict = frame['I3MCWeightDict']
            ow = mcweightdict['OneWeight'] / mcweightdict['NEvents'] / args.nfiles
            if 'TypeWeight' in mcweightdict: 
                ow /= mcweightdict['TypeWeight']
            else:
                if primary.pdg_encoding > 0: ow /= args.nu_fraction
                else: ow /= 1-args.nu_fraction

            # NuSources reports fluxes as the sum of nu and nubar fluxes.
            # To correctly handle this down the road, we need to correct
            # the oneweight values now by dividing by a factor of 2.
            ow /= 2.0

            #===========================================
            # Calculate RA/dec values from the primary
            #===========================================
            truera, truedec = dir_to_equa(zenith=primary.dir.zenith,
                                          azimuth=primary.dir.azimuth,
                                          mjd = mjd)

            #===========================================
            # Write the MC values
            #===========================================
            current_event.extend([ow,                  # ow
                                  primary.energy,      # trueE
                                  truera,              # trueRa
                                  truedec,             # trueDec
                                  primary.dir.azimuth, # trueAzi
                                  primary.dir.zenith]) # trueZen

        output.append(tuple(current_event))

#----------------------------------------------------
# Convert the output list into a numpy array and save it
#----------------------------------------------------
if is_mc: dtype += mc_dtype

output = np.array(output, dtype=dtype)
np.save(args.output, output)
