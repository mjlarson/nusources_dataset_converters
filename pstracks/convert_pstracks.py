#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/icetray-start
#METAPROJECT icetray/v1.11.1
#!/usr/bin/env python3
import os, sys, pickle
from glob import glob
from argparse import ArgumentParser
import numpy as np
import numpy.lib.recfunctions as rf
import scipy
from scipy import interpolate

from icecube import dataclasses, dataio, icetray, simclasses, paraboloid
from icecube.icetray import I3Units
from icecube.astro import dir_to_equa

try: from tqdm import tqdm
except: tqdm = lambda x: x

#----------------------------------------------------
# Write a function to process a single file
#----------------------------------------------------
def process_file(upgoing_filename,
                 downgoing_filename,
                 outdir,
                 upgoing_spline, 
                 downgoing_spline, 
                 zen_up_down_split = 85*I3Units.degree, 
                 nu_fraction = 0.5):
    output, ids = [], []
    is_mc = False
    i3file = dataio.I3FrameSequence([upgoing_filename, downgoing_filename])

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

        # Check if we've already seen this event to avoid duplicates
        event_id = (header.run_id, header.sub_event_id, header.event_id)
        if event_id in ids: continue
        ids.append(event_id)

        mjd = header.start_time.mod_julian_day_double

        # If it's a leap second run, fix that now.
        if (120398 <= header.run_id) and (header.run_id <= 126377):
            mjd += 1. / 86400.

        #===========================================
        # Check the upgoing BDT scores
        #===========================================
        bdt_score = np.inf
        mu_score = np.inf
        casc_score = np.inf
        if 'BDTscore' in frame.keys():
            bdt_score = frame['BDTscore'].value
            mu_score = frame['MuScore'].value
            casc_score = frame['CascScore'].value

            # Rescale these if we're using v5, since that included 
            if 'version-005' in upgoing_filename:
                bdt_score = (bdt_score + 1) / 2.0
                mu_score = (mu_score + 1) / 2.0
                casc_score = (casc_score + 1) / 2.0
                
            #if bdt_score <= 0.7 or mu_score >= 0.2:
            #    continue
        elif "BDT_DG_mean" in frame.keys():
            bdt_score = frame['BDT_DG_mean'].value + 10
            mu_score = frame['BDT_DGzen_mean'].value + 10
        elif "BDT_UG_soft" in frame.keys():
            bdt_score = frame['BDT_UG_soft'].value + 20 
            mu_score = frame['BDT_UG_hard_mean'].value + 20
        else:
            print(frame)
            continue

        #===========================================
        # And the recos
        #===========================================
        direction = frame['SplineMPE_l4'].dir
        energy = frame['SplineMPEMuEXDifferential'].energy

        # Kill off events that reco below 1 GeV equivalent energy
        if energy <= 1: 
            continue

        try: truncatedE = frame['SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Muon'].energy
        except: truncatedE = np.nan

        # Convert from local to equatorial coordinates
        ra, dec = dir_to_equa(zenith=direction.zenith,
                              azimuth=direction.azimuth,
                              mjd = mjd)

        #===========================================
        # And the estimated angular uncertainty. 
        # I think this should have a sindec term, but 
        # this matches what was done in the past for now.
        #===========================================
        if frame['SplineMPE_l4Paraboloid'].fit_status == 0:
            uncorrected_angErr = np.hypot(frame['SplineMPE_l4ParaboloidFitParams'].pbfErr1, 
                                         frame['SplineMPE_l4ParaboloidFitParams'].pbfErr2) / np.sqrt(2)
        else:
            uncorrected_angErr = frame['SplineMPEBootstrapVectStats']['median']
            
        if direction.zenith > zen_up_down_split:
            spline = upgoing_spline
        else: 
            spline = downgoing_spline

        correction = scipy.interpolate.splev(np.log10(energy), spline) / 1.1774
        angErr = uncorrected_angErr * correction

        # Ensure events have positive angular uncertainty
        #if angErr < 0: continue

        # Remove poorly reconstructed events from the southern sky
        # and from the northern sky
        if   (dec < np.deg2rad(-5))  and (angErr >= np.deg2rad(5)): continue
        elif (dec >= np.deg2rad(-5)) and (angErr >= np.deg2rad(15)): continue

        #===========================================
        # And check the IceTop veto
        #===========================================
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
                         np.log10(truncatedE),   # logTE
                         ra,                     # ra
                         dec,                    # dec
                         angErr,                  # angErr
                         uncorrected_angErr,      # angErr_noCorrection
                         pass_veto,              # passed_icetopVeto
                         bdt_score,              # bdt_score
                         mu_score,               # mu_score
                         casc_score]             # casc_score

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
            ow = mcweightdict['OneWeight'] / mcweightdict['NEvents']
            if 'TypeWeight' in mcweightdict: 
                ow /= mcweightdict['TypeWeight']
            else:
                if primary.pdg_encoding > 0: ow /= nu_fraction
                else: ow /= 1-nu_fraction

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
    # Generate the output filename
    #----------------------------------------------------
    outfile_name = os.path.basename(i3filename)
    if is_mc:
        outfile_name = outfile_name.replace("Level3_","PSTracks_")
        outfile_name = ".".join(outfile_name.split(".")[:3])
        outfile = os.path.join(outdir, outfile_name)
    else:
        outfile_name = outfile_name.split(".")[0]
        outfile = os.path.join(outdir, outfile_name + ".npy")
            
    #----------------------------------------------------
    # Convert the output list into a numpy array with the appropriate datatype
    #----------------------------------------------------
    dtype = [('run', np.int64), ('subevent', np.int32), ('event', np.int64),
             ('time', np.float64), ('logE', np.float32), ('logTE', np.float32),
             ('ra', np.float32), ('dec', np.float32), 
             ('angErr', np.float32), ('angErr_noCorrection', np.float32),
             ('passed_IceTopVeto', bool),
             ('bdt_score', np.float32), ('mu_score', np.float32), ('casc_score', np.float32)]
    if is_mc:
        dtype += [('ow', np.float64), ('trueE', np.float32),
                  ('trueRa', np.float32), ('trueDec', np.float32),
                  ('trueAzi', np.float32), ('trueZen', np.float32)]

    output = np.array(output, dtype=dtype)

    #----------------------------------------------------
    # Save it
    #----------------------------------------------------
    np.save(outfile, output)
    return


#----------------------------------------------------
# Build a main function
#----------------------------------------------------
if __name__ == "__main__":
    #----------------------------------------------------
    # Build our argument parser.
    #----------------------------------------------------
    parser = ArgumentParser(description="Base script for converting i3 files to NuSources-compatible"
                            " npy files for use in csky and skyllh.")
    parser.add_argument("upgoing_input", metavar="i3file", type=str,
                        nargs="+", default=[],
                        help='Directory or file to include')
    parser.add_argument("--outdir", "-o", dest="outdir", type=str,
                        action = "store", default="", required=True,
                        help = "The path to write output npy files to")
    parser.add_argument("--nu_fraction", type=float, action="store", default=0.5,
                        help="What fraction of this simulation is neutrinos? Note that if there's an S-frame that's"
                        " compatible with simweights or a TypeWeight key in the I3MCWeightDict, this option will be"
                        " ignored in favor of what's in the file.")
    parser.add_argument("--pull-correction-dir", type=str, 
                        default="/cvmfs/icecube.opensciencegrid.org/users/NeutrinoSources/pstracks/pstracks_icetray_v01.10.0_py3v4.3.0.RHEL_7_x86_64/pstracks/resources/pull/",
                        help=("Path to the location of the pull correction spline files. We will read"
                              " upgoing_spline_gamma2.0.npy and downgoing_spline_gamma2.0.npy from this location."))
    parser.add_argument("--test", action='store_true', default=False,
                        help="If given, only process 10 files.")

    args = parser.parse_args()
    
    #----------------------------------------------------
    # Build the file list from the input
    #----------------------------------------------------
    upgoing_filelist = []
    for path in args.upgoing_input:
        if os.path.isdir(path):
            upgoing_filelist += glob(os.path.join(path, "*upgoing*.i3*"))
        elif os.path.isfile(path) and 'upgoing' in os.path.basename(path):
            upgoing_filelist.append(path)

    if args.test and len(upgoing_filelist) > 10:
        print("Test run requested: shortening infiles list to a maximum of 10 files")
        upgoing_filelist = upgoing_filelist[:10]

    #----------------------------------------------------
    # Read the pull correction splines. If reco_zen above 85 degrees, then is "upgoing"
    #----------------------------------------------------
    upgoing_pull_file = os.path.join(args.pull_correction_dir, "upgoing_spline_gamma2.0.npy")
    upgoing_spline = np.load(upgoing_pull_file, allow_pickle=True, encoding='latin1')
    downgoing_pull_file = os.path.join(args.pull_correction_dir, "downgoing_spline_gamma2.0.npy")
    downgoing_spline = np.load(downgoing_pull_file, allow_pickle=True, encoding='latin1')


    #----------------------------------------------------
    # Start walking through the files
    #----------------------------------------------------
    for i3filename in tqdm(upgoing_filelist):
        process_file(upgoing_filename = i3filename, 
                     downgoing_filename = i3filename.replace("upgoing", "downgoing"), 
                     outdir = args.outdir,
                     upgoing_spline = upgoing_spline, 
                     downgoing_spline = downgoing_spline, 
                     zen_up_down_split = 85*I3Units.degree, 
                     nu_fraction = args.nu_fraction)
