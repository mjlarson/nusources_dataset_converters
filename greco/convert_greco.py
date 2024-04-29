#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/icetray-start
#METAPROJECT icetray/v1.5.1
#!/usr/bin/env python3

import os, sys, pickle
from argparse import ArgumentParser
import numpy as np
import numpy.lib.recfunctions as rf

from icecube import dataclasses, dataio, icetray
from icecube.icetray import I3Units as i3u
from icecube.astro import dir_to_equa
from icecube import genie_reader

import healpy as hp

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
                           "this is the number of files after re-merging for this specific dataset/flavor!")
parser.add_argument("--genie-icetray", "-g", type=bool, action='store_true', default=False,
                    help = "If given, assume we're using oscnext/genie-icetray style weighting")
parser.add_argument("--genie-reader", type=bool, action='store_true', default=False,
                    help = "If given, assume we're using upgrade/genie-reader style weighting")
parser.add_argument("--nugen", type=bool, action='store_true', default=False,
                    help = "If given, assume we're using nugen style weighting")
args = parser.parse_args()

#----------------------------------------------------
# Load the BDT regressor for our angular error
# Originally from here:
# https://github.com/apizzuto/Novae/blob/master/random_forest/load_model.py
# Model is identical to the one stored at
# /data/user/apizzuto/Nova/RandomForests/v2.5/GridSearchResults_logSeparation_True_bootstrap_True_minsamples_100
#----------------------------------------------------
model_file = '/data/ana/PointSource/GRECO_online/regressor_logSeparation_True_bootstrap_True_minsamples_100.pckl'
regressor = pickle.load(open(model_file, 'rb').best_estimator_

#----------------------------------------------------
# Set up our output
#----------------------------------------------------
output = []
dtype = [('run', np.int32), ('subevent', np.int32), ('event', np.int64),
         ('time', np.float64), 
         ('logE', np.float32), ('ra', np.float32), ('dec', np.float32), 
         ('angErr', np.float32), ('uncorrected_angErr', np.float32)]

if any([args.genie_icetray, args.genie_reader, arg.nugen]):
    mc_dtypes = [('ow', np.float64),
                 ('uncorrected_ow', np.float64),
                 ('trueE', np.float32), 
                 ('ptype', np.float32), 
                 ('iscc', bool),
                 ('genie_gen_r', np.float32), 
                 ('genie_gen_z', np.float32),
                 ('trueRa', np.float32), 
                 ('trueDec', np.float32),
                 ('trueAzi', np.float32), 
                 ('trueZen', np.float32),
                 ]
    dtype += mc_dtypes

dtype_for_bdt = [('nstring', np.int32), 
                 ('nchannel', np.int32),
                 ('cascade_energy', np.float32),
                 ('monopod_zen', np.float32),
                 ('pidDeltaLLH', np.float32), 
                 ('pidPeglegLLH', np.float32), 
                 ('pidMonopodLLH', np.float32), 
                 ('pidLength', np.float32),
                 ('monopod_pegleg_dpsi', np.float32),
                ]
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
        # Get the event information
        #===========================================
        header = frame['I3EventHeader']
        mjd = header.start_time.mod_julian_day_double

        #===========================================
        # And the recos
        #===========================================
        track = frame['Pegleg_Fit_NestleTrack']
        casc = frame['Pegleg_Fit_NestleHDCasc']

        # The reconstructed (neutrino) energy
        energy = track.energy + casc.energy

        # Directional reco
        azimuth = track.dir.azimuth
        zenith = track.dir.zenith
            
        #===========================================
        # Convert from local to equatorial coordinates
        #===========================================
        ra, dec = dir_to_equa(zenith=float(zenith),
                              azimuth=float(azimuth),
                              mjd = float(mjd))
        
        #===========================================
        # And angular error... This is calculated
        # from Alex Pizzuto's BDT regressor. First,
        # we need to get the features for the regressor.
        #===========================================
        regressor_features = []
        
        # -----------
        # NChannel, nstring
        # -----------
        if 'I3DST' in frame.keys():
            i3dst = frame['I3DST']
            regressor_features.append(nstring)        # nstring
            regressor_features.append(i3dst.ndom)     # nchannel
        else:
            regressor_features.extend([-1, -1])

        # -----------
        # Zenith and logE
        # Note that these need to be deleted later to avoid double-counting
        # -----------
        regressor_features.append(zenith)             # (reco) zenith
        regressor_features.append(np.log10(energy))   # (reco) logE
        
        # -----------
        # Cascade energy
        # -----------
        regressor_features.append(casc.energy)        # cascade_energy
        
        # -----------
        # monopod azi, zen
        # -----------
        monopod = frame['Monopod_best']
        regressor_features.append(monopod_dir.zenith) # monopod_zen
        
        # -----------
        # PID LLH values
        # -----------
        pegleg_llh = frame['Pegleg_Fit_NestleFitParams'].logl
        monopod_llh = frame['Monopod_bestFitParams'].logl
        
        regressor_features.append(pegleg_llh - monopod_llh) # pidDeltaLLH
        regressor_features.append(pegleg_llh)               # pidPeglegLLH
        regressor_features.append(monopod_llh)              # pidMonopodLLH
        
        # -----------
        # PID track length
        # -----------
        bdt_values.append(track.length)                     # PIDLength
        
        # -----------
        # And the angle between monopod and pegleg
        # -----------
        monopod_ra, monopod_dec = dir_to_equa(zenith=float(monopod.dir.zenith),
                                              azimuth=float(monopod.dir.azimuth),
                                              mjd = float(mjd))
        monopod_pegleg_dpsi = hp.rotator.angdist(np.rad2deg([ra, dec]),
                                                 np.rad2deg(monopod_ra, monopod_dec),
                                                 lonlat = True)
        regressor_features.append(monopod_pegleg_dpsi)      # monopod_pegleg_dpsi

        #===========================================
        # Actually get the sigma value from the bdt
        #===========================================
        angErr = 10**regressor.predict(regressor_features)

        # And make sure to remove the zen/logE from the regressor features
        # once they're no longer needed there.
        del regressor_features[2]
        del regressor_features[2]
        
        #===========================================
        # Check if we're dealing with MC events
        #===========================================
        mc_values = []
        if 'I3MCWeightDict' in frame:
            #===========================================
            # This is a MC event, so we need to get some
            # extra information from the file.
            #===========================================
            mcweightdict = frame['I3MCWeightDict']
            pdg_encoding = primary.pdg_encoding
            is_cc = (mcweightdict['InteractionType'] == 1)

            # Find the primary
            try: primary = frame['MCInIcePrimary']
            except:
                mctree = sorted([key for key in frame if 'I3MCTree' in key])[0]
                primary = dataclasses.get_most_energetic_neutrino(frame[mctree])

            # Get the direction of the true neutrino
            truera, truedec = dir_to_equa(float(primary.dir.zenith), float(primary.dir.azimuth), float(mjd))

            #===========================================
            # Oneweight nonsense. This is a placeholder that should
            # work for cases when we have only non-overlapping MC sets
            # (ie, only pulling files from one set of NuMu). 
            #===========================================
            ow = mcweightdict["OneWeight"]
            nevents = mcweightdict['NEvents']

            # Handle the nu/nubar generation ratios
            if args.genie_icetray:
                frac = 0.7 if (pdg_encoding>0) else 0.3
            elif args.genie_reader:
                frac = 1.0
            elif args.nugen:
                frac = 0.5
            else:
                print("User didn't specify --genie-reader, --genie-icetray, or --nugen."
                      " I don't know what to do, so I'll die.")
                raise RuntimeError
            
            ow /= (frac * nevents * args.nfiles)

            #===========================================
            # NuSources reports fluxes as the sum of nu and nubar fluxes.
            # To correctly handle this down the road, we need to correct
            # the oneweight values now by dividing by a factor of 2.
            #===========================================
            ow /= 2.0

            mc_values = [ow,                    # ow
                         ow,                    # uncorrected_ow
                         primary.energy,        # trueE
                         pdg_encoding,          # ptype
                         is_cc,                 # iscc
                         genie_vol_r,           # genie_gen_r
                         genie_vol_z,           # genie_gen_z
                         truera,                # trueRa
                         truedec,               # trueDec
                         primary.dir.azimuth,   # trueAzi
                         primary.dir.zenith,    # trueZen
                        ]

        #===========================================
        # And write it out. The order here MUST match
        # the order in the `dtype` variable above or
        # else you'll end up with wrong labels.
        #===========================================
        current_event = [header.run_id,         # run
                         header.sub_event_id,   # subevent
                         header.event_id,       # event
                         mjd,                   # time
                         np.log10(energy),      # (reco) logE
                         ra,                    # (reco) ra
                         dec,                   # (reco) dec
                         angErr,                # angErr
                         angErr,                # uncorrected_angErr
                        ]
         
        # If available, add the MC columns too
        current_event += mc_values
        
        # And the BDT values
        current_event += regressor_features
        
        # Convert to a tuple and move onto the next event.
        output.append(tuple(current_event))

#----------------------------------------------------
# Convert the output list into a numpy array and save it
#----------------------------------------------------
output = np.array(output, dtype=dtype)
np.save(args.output, output)

