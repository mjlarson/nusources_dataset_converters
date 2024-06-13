#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/icetray-start
#METAPROJECT icetray/v1.3.3
#!/usr/bin/env python3

import os, sys, pickle, glob
from argparse import ArgumentParser
import numpy as np
import numpy.lib.recfunctions as rf
import sklearn
import copy

from icecube import dataclasses, dataio, icetray, recclasses, dst, millipede, common_variables, recclasses
from icecube.icetray import I3Units
from icecube.astro import dir_to_equa

try: from tqdm import tqdm
except: tqdm = lambda x: x

from angular_uncertainty_bdt import predict_uncertainty
    
def main():
    #----------------------------------------------------
    # Build our argument parser.
    #----------------------------------------------------
    parser = ArgumentParser(description="Base script for converting i3 files to NuSources-compatible"
                            " npy files for use in csky and skyllh.")
    parser.add_argument("input", metavar="i3file", type=str,
                        nargs='+', default=[])
    parser.add_argument("--gcd", dest="gcd", type=str,
                        action = "store", default="/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz",
                        help = "The GCD file associated with these events.")
    parser.add_argument("--output", "-o", dest="output", type=str,
                        action = "store", default="", required=True,
                        help = "The output npy filename and path to write to")
    parser.add_argument("--nfiles", "-n", type=int, action='store', default=-1,
                        help = "Number of files from this dataset that will be converted in total (ie, across all "
                               "jobs running on this script and including any that had 0 events passing!). Note that "
                               "this is the number of files after re-merging for this specific dataset/flavor! If "
                               "this is not specified, we will simply count the number of input files.")
    parser.add_argument("--genie-icetray", action='store_true', default=False,
                        help = "If given, assume we're using oscnext/genie-icetray style weighting")
    parser.add_argument("--genie-reader", action='store_true', default=False,
                        help = "If given, assume we're using upgrade/genie-reader style weighting")
    parser.add_argument("--nugen", action='store_true', default=False,
                        help = "If given, assume we're using nugen style weighting")
    parser.add_argument("--test", action="store_true", default=False,
                        help = "If given, only process the first 10 files.")
    parser.add_argument('--lowen_bound', help='Keep events above this energy', default=1, dtype=float)
    parser.add_argument('--highen_bound', help='Keep events below this energy', default=10**3.5, dtype=float)
    args = parser.parse_args()

    if args.nfiles < 0:
        print("No number of files specified. I'll fall back on just counting the number of input files instead. "
             "This is fine for GENIE events where every file is expected to give "
             "at least some events, but is almost certainly wrong for nugen!")
        args.nfiles = len(args.input)

    assert args.highen_bound > args.lowen_bound, (f'Upper energy bound ({args.highen_bound}) cannot be '
                                                  f'less than/equal to lower energy bound ({args.})!')

    #----------------------------------------------------
    # Set up our output
    #----------------------------------------------------
    output = []
    dtype = [('run', np.int32), ('subevent', np.int32), ('event', np.int64),
             ('time', np.float64), 
             ('logE', np.float32), ('ra', np.float32), ('dec', np.float32), 
             ('angErr', np.float32), ('angErr_noCorrection', np.float32)]
    
    if any([args.genie_icetray, args.genie_reader, args.nugen]):
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
                     ('angle_to_secondary', np.float32),
                     ]
        dtype += mc_dtypes

    dtype_for_bdt = [('nstring', np.int32), 
                     ('nchannel', np.int32),
                     ('azi', np.float32),
                     ('zen', np.float32),
                     ('cascade_energy', np.float32),
                     ('monopod_azi', np.float32),
                     ('monopod_zen', np.float32),
                     ('pidDeltaLLH', np.float32), 
                     ('pidPeglegLLH', np.float32), 
                     ('pidMonopodLLH', np.float32), 
                     ('pidLength', np.float32),
                     ('monopod_ra', np.float32),
                     ('monopod_dec', np.float32),
                     ('monopod_pegleg_dpsi', np.float32),
                     ('rho', np.float32),
                     ('z', np.float32),
                     ('qdir', np.float32),
                    ]
    dtype += dtype_for_bdt

    #----------------------------------------------------
    # Start walking through the file
    #----------------------------------------------------
    filenames = []
    for indir in args.input:
        if os.path.isdir(indir):
            filenames +=  glob.glob(os.path.join(indir, "*"))
        else:
            filenames += [indir, ]
        print(indir, len(filenames))
    if args.test:
        filenames = filenames[:10]

    for i3filename in tqdm(sorted(filenames)):
        print(i3filename)
        i3file = dataio.I3FrameSequence([args.gcd, i3filename,])

        while i3file.more(): # save the geometry, used for QDirPulses calculation later
            frame = i3file.pop_frame()

            if frame.Stop == icetray.I3Frame.Geometry:
                geometry = frame['I3Geometry']
                break

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
                
            if (energy < args.lowen_bound) or (energy > args.highen_bound): continue

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
                regressor_features.append(i3dst.n_string) # nstring
                regressor_features.append(i3dst.ndom)     # nchannel
            else:
                regressor_features.extend([-1, -1])

            # -----------
            # Zenith and logE
            # -----------
            regressor_features.append(azimuth)            # (reco) azi
            regressor_features.append(zenith)             # (reco) zen

            # -----------
            # Cascade energy
            # -----------
            regressor_features.append(casc.energy)        # cascade_energy

            # -----------
            # monopod zen
            # -----------
            monopod = frame['Monopod_best']
            regressor_features.append(monopod.dir.azimuth) # monopod_azi
            regressor_features.append(monopod.dir.zenith)  # monopod_zen

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
            regressor_features.append(track.length)             # PIDLength

            # -----------
            # And the angle between monopod and pegleg
            # -----------
            monopod_ra, monopod_dec = dir_to_equa(zenith=float(monopod.dir.zenith),
                                                  azimuth=float(monopod.dir.azimuth),
                                                  mjd = float(mjd))
            monopod_pegleg_dpsi = monopod.dir.angle(track.dir)
            regressor_features.append(monopod_ra)               # monopod_ra
            regressor_features.append(monopod_dec)              # monopod_dec
            regressor_features.append(monopod_pegleg_dpsi)      # monopod_pegleg_dpsi

            #===========================================
            # Andrew has requested the interaction position
            #===========================================
            rho = np.sqrt((casc.pos.x-46.29)**2 + (casc.pos.y+34.88)**2)
            regressor_features.append(rho)
            regressor_features.append(casc.pos.z)

            #===========================================
            # And some measure of the direct charge
            # We'll include the definition from pstracks here,
            # although it doesn't get used...?
            #===========================================
            definitions = common_variables.direct_hits.get_default_definitions()
            definitions.append(common_variables.direct_hits.I3DirectHitsDefinition("E",-15.,250.)) 
            
            pulsemapmask_name = 'SRTInIcePulses'
            pulsemap = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, pulsemapmask_name)
            particle = frame['Pegleg_Fit_Nestle']
            ndir_map = common_variables.direct_hits.calculate_direct_hits(definitions, geometry, pulsemap, particle)
            qdir = ndir_map['A'].q_dir_pulses # [-15ns; +15ns] time window
            regressor_features.append(qdir)

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
                is_cc = (mcweightdict['InteractionType'] == 1)

                # Find the primary
                if 'I3MCTree_preMuonProp' in frame:
                    mctree = frame['I3MCTree_preMuonProp']
                else:
                    mctree = frame['I3MCTree']
                primary = dataclasses.get_most_energetic_neutrino(mctree)
                pdg_encoding = primary.pdg_encoding

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

                #===========================================
                # We also store the r/z of the interaction 
                # position relative to the genie-reader 
                # generation cylinder. This can be used
                # to merge nugen/genie if we're not using
                # simweights like normal people.
                #===========================================
                genie_vol_r = 0
                genie_vol_z = 0

                #===========================================
                # Andrew has requested the angle to the
                # most energetic secondary.
                #===========================================
                daughters = mctree.get_daughters(primary)
                most_energetic = daughters[0]
                for p in daughters:
                    if p.energy > most_energetic.energy:
                        most_energetic = p
                angle_to_secondary = primary.dir.angle(most_energetic.dir)

                #===========================================
                # Store it all in a single list
                #===========================================
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
                             angle_to_secondary,    # angle_to_secondary
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
                             0,                     # angErr
                             0,                     # angErr_noCorrection
                            ]

            # If available, add the MC columns too
            current_event += mc_values

            # And the BDT values
            current_event += regressor_features

            if np.any(~np.isfinite(current_event)): 
                print("Found something with non-finite values:", current_event)
                continue
            # Convert to a tuple and move onto the next event.
            output.append(tuple(current_event))

    #----------------------------------------------------
    # Convert the output list into a numpy array
    #----------------------------------------------------
    output = np.array(output, dtype=dtype)
    if len(output) == 0:
        raise ValueError('Output array has zero elements! Panic!')

    #-----------------------------------------------------
    # Calculate the angular uncertainty from Alex Pizzuto's BDT
    #-----------------------------------------------------
    output['angErr'] = predict_uncertainty(output)
    output['angErr_noCorrection'] = copy.deepcopy(output['angErr'])

    # And apply the original pull corrections
    south_spline = pickle.load(open("pull_correction_splines/greco_north_e-3.pckl", 'rb'), encoding='latin1')
    north_spline = pickle.load(open("pull_correction_splines/greco_south_e-3.pckl", 'rb'), encoding='latin1')

    south = (output['dec'] <= np.radians(-5))
    north = ~south
    
    output['angErr'][south] = output['angErr_noCorrection'][south] / south_spline(output['logE'][south])
    output['angErr'][north] = output['angErr_noCorrection'][north] / north_spline(output['logE'][north])
    #----------------------------------------------------
    # and finally save it
    #----------------------------------------------------
    np.save(args.output, output)


if __name__ == "__main__":
    main()
