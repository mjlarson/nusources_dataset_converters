#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/icetray-start
#METAPROJECT icetray/v1.5.1
#!/usr/bin/env python3

import os, sys
from argparse import ArgumentParser
import numpy as np
import numpy.lib.recfunctions as rf

from icecube import dataclasses, dataio, icetray
from icecube.icetray import I3Units as i3u
from icecube.astro import dir_to_equa
from icecube import genie_reader

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
parser.add_argument("--nfiles", "-n", type=int, action='store', required=True,
                    help = "Number of files from this dataset that will be converted in total (ie, across all "
                           "jobs running on this script and including any that had 0 events passing!). Note that "
                           "this is the number of files from this specific dataset! So eg 10 files from 140028 "
                           " and 10 files from 141028 does *not* mean you should use --nfiles 20!")
                    
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
         ('ra', np.float32), ('dec', np.float32), ('sigma', np.float32),
         ('atmo_weight', np.float64),]

dtype_for_bdt = [('L4_separation_in_cogs', np.float32),
                 ('SplitInIcePulses_dynedge_v2_PulsesUpgradeHitMultiplicity.n_hit_pmts', np.float32),
                 ('SplitInIcePulses_dynedge_v2_PulsesUpgradeHitMultiplicity.n_hit_strings', np.float32),
                 ('graphnet_dynedge_zenith_reconstruction_zenith_kappa', np.float32),
                 ('graphnet_dynedge_direction_reconstruction_direction_kappa', np.float32),
                 ('graphnet_dynedge_energy_reconstruction_energy_pred', np.float32),
                 ('graphnet_dynedge_track_classification_track_pred', np.float32)]
dtype += dtype_for_bdt

#----------------------------------------------------
# Start walking through the file
#----------------------------------------------------
for i3filename in tqdm(args.input):
    print(i3filename)
    i3file = dataio.I3File(i3filename, 'r')

    while i3file.more():
        frame = i3file.pop_frame()
        
        #===========================================
        # Wait for P-frames
        #===========================================
        if not frame.Stop == icetray.I3Frame.Physics: continue

        #===========================================
        # QUESO requires cuts to be applied. Do that.
        #===========================================
        if not frame['QuesoL3_Bool']: continue
        if not frame['QuesoL4_Bool']: continue
        if not frame['QuesoL3_Vars_cleaned_num_hits_fid_vol'] >= 7: continue
            
        #===========================================
        # Get the event information
        #===========================================
        header = frame['I3EventHeader']
        mjd = header.start_time.mod_julian_day_double
        mcweightdict = frame['I3MCWeightDict']

        # Find the primary
        try: primary = frame['MCInIcePrimary']
        except:
            mctree = sorted([key for key in frame if 'I3MCTree' in key])[0]
            primary = dataclasses.get_most_energetic_neutrino(frame[mctree])

        mjd = 61041 # January 1, 2026
        truera, truedec = dir_to_equa(float(primary.dir.zenith), float(primary.dir.azimuth), float(mjd))
        pdg_encoding = primary.pdg_encoding

        # And the recos
        energy = frame['graphnet_dynedge_energy_reconstruction_energy_pred'].value

        # Directional reco
        x = frame['graphnet_dynedge_direction_reconstruction_dir_x_pred'].value
        y = frame['graphnet_dynedge_direction_reconstruction_dir_y_pred'].value
        z = frame['graphnet_dynedge_direction_reconstruction_dir_z_pred'].value
        rho = np.sqrt(x**2+y**2)
        azimuth = np.mod(np.arctan2(y,x), 2*np.pi)        
        zenith = np.arctan2(rho,z)
            
        #===========================================
        # Convert from local to equatorial coordinates
        #===========================================
        ra, dec = dir_to_equa(zenith=float(zenith),
                              azimuth=float(azimuth),
                              mjd = float(mjd))
        
        #===========================================
        # And maybe angular error... This will probably
        # need to be handled by a BDT regressor or other
        # algorithm. We'll use the value in the files as
        # a first pass, but we'll also add a bunch of 
        # variables that may be useful as predictors to 
        # train a BDT on.
        #===========================================
        sigma = (frame['graphnet_dynedge_direction_reconstruction_direction_kappa'].value)**-0.5
        sigma = np.clip(sigma, 0, np.pi)
        
        bdt_vars = [frame['L4_separation_in_cogs'].value,
                    frame['SplitInIcePulses_dynedge_v2_PulsesUpgradeHitMultiplicity']['n_hit_pmts'],
                    frame['SplitInIcePulses_dynedge_v2_PulsesUpgradeHitMultiplicity']['n_hit_strings'],
                    frame['graphnet_dynedge_zenith_reconstruction_zenith_kappa'].value,
                    frame['graphnet_dynedge_direction_reconstruction_direction_kappa'].value,
                    frame['graphnet_dynedge_energy_reconstruction_energy_pred'].value,
                    frame['graphnet_dynedge_track_classification_track_pred'].value]

        #===========================================
        # Oneweight nonsense. This is a placeholder that should
        # work for cases when we have only non-overlapping MC sets
        # (ie, only pulling files from one set of NuMu). Note that
        # different flavors or nu vs nubar don't count here: as long
        # as this is the only set of eg NuEBar, then we're good.
        #===========================================
        nevents = frame["I3GenieInfo"].n_flux_events
        ow = frame["I3MCWeightDict"]["OneWeight"]
        ow /= (nevents * args.nfiles)

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
                         np.log10(energy),      # logE
                         ra,                    # ra
                         dec,                   # dec
                         sigma,                 # sigma
                         mcweightdict['weight'],# Atmospheric weight
        ]              + bdt_vars             # Extra variables for the bdt training
        output.append(tuple(current_event))


#----------------------------------------------------
# Convert the output list into a numpy array and save it
#----------------------------------------------------
output = np.array(output, dtype=dtype)
np.save(args.output, output)

