#!/usr/bin/env python3

import os, sys
from argparse import ArgumentParser
import numpy as np
import numpy.lib.recfunctions as rf

from icecube import dataclasses, dataio, icetray
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
                    nargs="+", action='append', default=[])
parser.add_argument("--output", "-o", dest="output", type=str,
                    action = "store", default="", required=True,
                    help = "The output npy filename and path to write to")
parser.add_argument("--nfiles", "-n", type=int, action='store', required=True,
                    help = "Number of files initially processed (including any that had 0 events passing!)")

parser.add_argument("--direction_reco", type=str,
                    action='store', default='SplineMPE_l4', required=True,
                    help="Name of the I3Particle containing the directional reconstruction to use for these events")
parser.add_argument("--energy_reco", type=str,
                    action='store', default='SplineMPEMuEXDifferential', required=True,
                    help="Name of the I3Particle containing the energy reconstruction to use for these events")
parser.add_argument("--angerr_reco", type=str,
                    action='store', default=None,
                    help="Eventually, name of an object containing angular error estimates for each event."
                         " For now, though, just assuming a fixed angular uncertainty for every event.") ########################### fix me later!~
                    
parser.add_argument("--nu_fraction", type=float, action="store", default=0.5,
                    help="What fraction of this simulation is neutrinos? Note that if there's an S-frame that's"
                    " compatible with simweights or a TypeWeight key in the I3MCWeightDict, this option will be"
                    " ignored in favor of what's in the file.")
args = parser.parse_args()


#----------------------------------------------------
# Set up our output
#----------------------------------------------------
output = []
dtype = [('run', np.int32), ('subevent', np.int32), ('event', np.int64),
         ('time', np.float64), ('ow', np.float64),
         ('trueRa', np.float32), ('trueDec', np.float32),
         ('trueAzi', np.float32), ('trueZen', np.float32),
         ('trueE', np.float32), ('logE', np.float32),
         ('ra', np.float32), ('dec', np.float32), ('sigma', np.float32)]
sframe_weightdict = {}
sframe_weightkey = None
pframe_weightdict = {}
pframe_weightkeys = []
mctype = None

#----------------------------------------------------
# Start walking through the file
#----------------------------------------------------
for i3filename in tqdm(args.input):
    i3file = dataio.I3File(i3filename, 'r')

    while i3file.more():
        frame = i3file.pop_frame()

        #===========================================
        # If we find an S-frame, store stuff.
        #===========================================
        if frame.Stop == icetray.I3Frame.Simulation:
            if mctype is None:
                if ('I3CorsikaWeight' in frame) or ('I3CoriskaInfo' in frame):
                    raise NotImplementedError("CORSIKA simulation is not implemented here!")
                elif 'I3GenieInfo' in frame:
                    sframe_weightkey = 'I3GenieInfo'
                    pframe_weightkey = 'I3GenieResult'
                    mctype = simweights.GenieWeighter
                else:
                    continue

            for obj in weight_objects:
                try: info = frame[obj]
                except: pass

                for k in dir(info):
                    try: sframe_weightdict[k].append(getattr(info, k))
                    except: sframe_weightdict[k] = [getattr(info, k),]
        
        #===========================================
        # Otherwise just wait for P-frames
        #===========================================
        if not frame.Stop == icetray.I3Frame.Physics: continue

        #===========================================
        # Handle the weighting
        #===========================================
        # Still don't have the MC type? Try again now.
        if mctype is None:
            if 'CorsikaWeightMap' in frame:
                raise NotImplementedError("CORSIKA simulation is not implemented here!")
            elif 'I3MCWeightDict' in frame:
                pframe_weightkey = 'I3MCWeightDict'
                mctype = simweights.NuGenWeighter
            else:
                raise TypeError("Cannot match frames from this file to objects listed on"
                                " https://docs.icecube.aq/simweights/main/reading_files.html"
                                " to determine what kind of weighter we need.")

        weightobj = frame[pframe_weightkey]
        for k in dir(weightobj):
            try: pframe_weightdict[k].append(weightobj[k])
            except: pframe_weightdict[k] = [weightobj[k],]
            
        #===========================================
        # Get the event information
        #===========================================
        header = frame['I3EventHeader']
        mjd = header.start_time.mod_julian_day_double
        mcweightdict = frame['I3MCWeightDict']

        # Find the primary
        try: primary = frame['MCPrimary']
        except:
            mctree = sorted([key for key in frame if 'I3MCTree' in key])[0]
            primary = dataclasses.get_most_energetic_neutrino(frame[mctree])

        # And the recos
        direction = frame[args.direction_reco]
        energy = frame[args.energy_reco]

        # And maybe angular error... later
        sigma = 5*i3u.degree
            
        # Convert from local to equatorial coordinates
        truera, truedec = dir_to_equa(zenith=primary.dir.zenith,
                                      azimuth=primary.dir.azimuth,
                                      mjd = mjd)
        ra, dec = dir_to_equa(zenith=direction.dir.zenith,
                              azimuth=direction.dir.azimuth,
                              mjd = mjd)
            
        #===========================================
        # Oneweight nonsense. This is a placeholder that should
        # work for cases when we have only non-overlapping MC sets.
        #===========================================
        ow = mcweightdict['OneWeight'] / mcweightdict['NEvents'] / args.nfiles
        if 'TypeWeight' in mcweightdict: ow /= mcweightdict['TypeWeight']
        else:
            if primary.pdg_encoding > 0: ow /= args.nu_fraction
            else: ow /= 1-args.nu_fraction

        # NuSources reports fluxes as the sum of nu and nubar fluxes.
        # To correctly handle this down the road, we need to correct
        # the oneweight values now by dividing by a factor of 2.
        ow /= 2.0

        #===========================================
        # And write it out. The order here MUST match
        # the order in the `dtype` variable above or
        # else you'll end up with wrong labels.
        #===========================================
        current_event = [header.run_id,         # run
                         header.sub_event_id,   # subevent
                         header.event_id,       # event
                         mjd,                   # time
                         ow,                    # ow
                         truera,                # trueRa
                         truedec,               # trueDec
                         primary.dir.azimuth,   # trueAzi
                         primary.dir.zenith,    # trueZen
                         mcweightdict['PrimaryNeutrinoEnergy'], # trueE
                         np.log10(energy.energy),               # logE
                         ra,                    # ra
                         dec,                   # dec
                         sigma,                 # sigma
        ]
        output.append(current_event)


#----------------------------------------------------
# Convert the output list into a numpy array and save it
#----------------------------------------------------
output = np.array(output, dtype=dtype)
np.save(args.output, output)
