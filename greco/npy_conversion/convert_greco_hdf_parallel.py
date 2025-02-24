#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/icetray-start 
#METAPROJECT icetray/v1.9.2
#!/mnt/ceph1-npx/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/venv/bin/python

import os, sys, pickle
from glob import glob
import pandas as pd
from os.path import dirname, join
from argparse import ArgumentParser
import numpy as np
import numpy.lib.recfunctions as rf
import copy

from icecube import dataclasses, dataio, icetray, recclasses, dst, millipede, common_variables, recclasses
from icecube.icetray import I3Units
from icecube.astro import dir_to_equa, angular_distance

try: from tqdm import tqdm
except: tqdm = lambda x: x

from concurrent.futures import ProcessPoolExecutor

#from angular_uncertainty_bdt import predict_uncertainty
from simweights import NuGenWeighter, GenieWeighter

def convert_single(filename):
    #print(f"Converting file {filename} from {nfiles} merged files.")
    #----------------------------------------------------
    # Set up our output
    #----------------------------------------------------
    dtype = [('run', np.int32), ('subevent', np.int32), ('event', np.int64),
             ('time', np.float64), 
             ('logE', np.float32), ('ra', np.float32), ('dec', np.float32), 
             ('angErr', np.float32), ('angErr_noCorrection', np.float32)]

    mc = False
    mc_dtypes = [('ow', np.float64),
        ('uncorrected_ow', np.float64),
        ('trueE', np.float32), 
        ('ptype', np.float32), 
        ('iscc', bool),
        ('trueRa', np.float32), 
        ('trueDec', np.float32),
        ('trueAzi', np.float32), 
        ('trueZen', np.float32),
        ('angle_to_secondary', np.float32),
        ('genie_nugen_overlap', bool),
    ]

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

    #----------------------------------------------------
    # Start walking through the files
    #----------------------------------------------------
    hdf_file = pd.HDFStore(filename)
    output = []

    #===========================================
    # Get the event information
    #===========================================
    header = hdf_file['I3EventHeader']
    mjd = hdf_file['I3EventHeader']['time_start_mjd'].values

    #===========================================
    # And the recos
    #===========================================
    track = hdf_file['Pegleg_Fit_NestleTrack']
    casc = hdf_file['Pegleg_Fit_NestleHDCasc']

    # The reconstructed (neutrino) energy
    energy = track["energy"].values + casc["energy"].values
        
    # Directional reco
    azimuth = track["azimuth"].values
    zenith = track["zenith"].values
    
    #===========================================
    # Convert from local to equatorial coordinates
    #===========================================
    #print("Converting pegleg reconstruction from zenith/azimuth to ra/dec.")
    ra, dec = dir_to_equa(zenith=zenith,
                          azimuth=azimuth,
                          mjd = mjd)

    #===========================================
    # And angular error... This is calculated
    # from Alex Pizzuto's BDT regressor. First,
    # we need to get the features for the regressor.
    #===========================================
    regressor_features = []
    
    # -----------
    # NChannel, nstring
    # -----------
    regressor_features.append(hdf_file["n_string"]["value"].values) # nstring
    regressor_features.append(hdf_file["n_dom"]["value"].values)    # nchannel

    # -----------
    # Zenith and logE
    # -----------
    regressor_features.append(azimuth)                  # (reco) azi
    regressor_features.append(zenith)                   # (reco) zen
    
    # -----------
    # Cascade energy
    # -----------
    regressor_features.append(casc["energy"].values)         # cascade_energy
    
    # -----------
    # monopod zen
    # -----------
    monopod = hdf_file["Monopod_best"]
    regressor_features.append(monopod["azimuth"].values)     # monopod_azi
    regressor_features.append(monopod["zenith"].values)      # monopod_zen

    # -----------
    # PID LLH values
    # -----------
    pegleg_llh = hdf_file["Pegleg_llh"]["value"].values
    monopod_llh = hdf_file["Monopod_llh"]["value"].values

    regressor_features.append(pegleg_llh - monopod_llh) # pidDeltaLLH
    regressor_features.append(pegleg_llh)               # pidPeglegLLH
    regressor_features.append(monopod_llh)              # pidMonopodLLH
    
    # -----------
    # PID track length
    # -----------
    regressor_features.append(track["length"].values)       # PIDLength
    
    # -----------
    # And the angle between monopod and pegleg
    # -----------
    #print("Converting monopod reconstruction from zenith/azimuth to ra/dec.")
    monopod_ra, monopod_dec = dir_to_equa(monopod["zenith"].values, monopod["azimuth"].values, mjd)
    monopod_pegleg_dpsi = angular_distance(monopod_ra, monopod_dec, ra, dec)
    regressor_features.append(monopod_ra)               # monopod_ra
    regressor_features.append(monopod_dec)              # monopod_dec
    regressor_features.append(monopod_pegleg_dpsi)      # monopod_pegleg_dpsi
    
    #===========================================
    # Andrew has requested the interaction position
    #===========================================
    rho = np.sqrt((casc["x"].values-46.29)**2 + (casc["y"].values+34.88)**2)
    regressor_features.append(rho)
    regressor_features.append(casc["z"].values)

    #===========================================
    # And some measure of the direct charge
    # We'll include the definition from pstracks here,
    # although it doesn't get used...?
    #===========================================
    qdir = hdf_file["qdir"]["value"].values # [-15ns; +15ns] time window
    regressor_features.append(qdir)

    #===========================================
    # Check if we're dealing with MC events
    #===========================================
    mc_values = []
    if 'I3MCWeightDict' in dir(hdf_file.root):
        mc = True
        #===========================================
        # This is a MC event, so we need to get some
        # extra information from the file.
        #===========================================
        mcweightdict = hdf_file["I3MCWeightDict"]
        is_cc = (mcweightdict["InteractionType"].values == 1)

        # Find the primary
        primary = hdf_file["MCPrimary"]
        pdg_encoding = primary["pdg_encoding"].values
        trueE = primary["energy"].values

        # Get the direction of the true neutrino
        #print("Converting primary direction from zenith/azimuth to ra/dec.")
        truera, truedec = dir_to_equa(primary["zenith"].values, primary["azimuth"].values, mjd)

        #===========================================
        # Oneweight nonsense. This is a placeholder that should
        # work for cases when we have only non-overlapping MC sets
        # (ie, only pulling files from one set of NuMu). 
        #===========================================
        ow = mcweightdict['OneWeight'].values / mcweightdict['NEvents'].values
        if "I3GENIEResultDict" in dir(hdf_file.root):
            ow[pdg_encoding>0] /= 0.7
            ow[pdg_encoding<0] /= 0.3
        else:
            ow /= 0.5

        #===========================================
        # NuSources reports fluxes as the sum of nu and nubar fluxes.
        # To correctly handle this down the road, we need to correct
        # the oneweight values now by dividing by a factor of 2.
        #===========================================
        ow /= 2.0
        
        #===========================================
        # Andrew has requested the angle to the
        # most energetic secondary.
        #===========================================
        #print("Converting secondary direction from zenith/azimuth to ra/dec.")
        secondary = hdf_file["MCSecondary"]
        secondary_ra, secondary_dec = dir_to_equa(zenith=secondary["zenith"].values,
                                                  azimuth=secondary["azimuth"].values,
                                                  mjd = mjd)
        angle_to_secondary = angular_distance(secondary_ra, secondary_dec, truera, truedec)

        #===========================================
        # Store it all in a single list
        #===========================================
        mc_values = [ow,                    # ow
                     ow,                    # uncorrected_ow
                     trueE,                 # trueE
                     pdg_encoding,          # ptype
                     is_cc,                 # iscc
                     truera,                # trueRa
                     truedec,               # trueDec
                     primary["azimuth"].values,# trueAzi
                     primary["zenith"].values, # trueZen
                     angle_to_secondary,    # angle_to_secondary
                     np.zeros_like(is_cc),  # genie_nugen_overlap
        ]
        
    #===========================================
    # And write it out. The order here MUST match
    # the order in the `dtype` variable above or
    # else you'll end up with wrong labels.
    #===========================================
    current_event = [header["Run"].values,      # run
                     header["SubEvent"].values, # subevent
                     header["Event"].values,    # event
                     mjd,                   # time
                     np.log10(energy),      # (reco) logE
                     ra,                    # (reco) ra
                     dec,                   # (reco) dec
                     np.zeros_like(mjd),    # angErr
                     np.zeros_like(mjd),    # angErr_noCorrection
                 ]

    # If available, add the MC columns too
    if mc: 
        current_event += mc_values
        dtype += mc_dtypes
    
    # And the BDT values
    current_event += regressor_features
    dtype += dtype_for_bdt

    output = list(zip(*current_event))

    #=============================================
    # Also write out a weighting object from simweights
    # if applicable. Note that pytables uses weakref
    # internally, which prevents us from saving a weighter
    # object with pytables-formatted data. Because of this,
    # we have to convert to a different format first.
    #=============================================
    weighter = None
    if mc:
        #print("Building weighter object.")
        nfiles = np.unique(hdf_file['nfiles']['value'].values)[0]

        hdf_like = {"I3MCWeightDict":pd.DataFrame(), "I3GENIEResultDict":pd.DataFrame()}
        for key in hdf_file.root.I3MCWeightDict.colnames:
            hdf_like["I3MCWeightDict"][key] = hdf_file["I3MCWeightDict"][key].values[:]
        hdf_like["I3MCWeightDict"] = hdf_like["I3MCWeightDict"].to_records()
        if "I3GENIEResultDict" in dir(hdf_file.root):
            for key in hdf_file.root.I3GENIEResultDict.colnames:
                hdf_like["I3GENIEResultDict"][key] = hdf_file["I3GENIEResultDict"][key].values[:]

            hdf_like["I3GENIEResultDict"] = hdf_like["I3GENIEResultDict"].to_records()
            weighter = GenieWeighter(hdf_like, nfiles)
        else:
            weighter = NuGenWeighter(hdf_like, nfiles)
        
    hdf_file.close()
    return np.array(output, dtype=dtype), weighter, nfiles

#######################################################
# Run it all in parallel?
#######################################################
def run_parallel(filenames, ncpus=20):
    data, weighters, nf = [], [], []
    with ProcessPoolExecutor(ncpus) as executor:
        jobs = [executor.submit(convert_single, filename) for filename in filenames]
        
        try:
            complete = sum([job.done() for job in jobs])
            pbar = tqdm(total=len(jobs))
            while(complete) < len(jobs):
                current = sum([job.done() for job in jobs])
                pbar.update(current-complete)
                complete = current
                
            for job in jobs:
                d, w, n = job.result()
                data.append(d)
                weighters.append(w)
                nf.append(n)
        except KeyboardInterrupt:
            [job.cancel() for job in jobs]
            executor.shutdown(wait=True)
            raise

    if weighters[0] != None:
        return np.concatenate(data), sum(weighters), sum(nf)
    else:
        return np.concatenate(data), None, 0

    
def main():
    #----------------------------------------------------
    # Build our argument parser.
    #----------------------------------------------------
    parser = ArgumentParser(description="Base script for converting i3 files to NuSources-compatible"
                            " npy files for use in csky and skyllh.")
    parser.add_argument("-i", "--input", type=str,
                        default="/data/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/output_hdf/")
    parser.add_argument("--output", "-o", dest="output", type=str,
                        action = "store", default="", required=True,
                        help = "The output npy filename and path to write to")
    parser.add_argument("--test", type=int, default=None, 
                        help = "Maximum number of files to use in case of testing.")
    parser.add_argument("--ncpus", type=int, default=20)
    args = parser.parse_args()

    #----------------------------------------------------
    # Convert the output list into a numpy array
    #----------------------------------------------------
    results = []
    for flavor in ['nue', 'numu', 'nutau']:
        geniefiles = sorted(glob(f"../../../output_hdf/{flavor}/*"))
        if args.test is not None: geniefiles=geniefiles[:args.test]
        genie, gweighter, gnfiles = run_parallel(geniefiles, ncpus=args.ncpus)
        gweighter.add_weight_column("event_weight", gweighter.weight_cols["wght"])

        nugenfiles = sorted(glob(f"../../../output_hdf/nugen_{flavor}/*"))
        if args.test is not None: nugenfiles=nugenfiles[:args.test]
        nugen, nweighter, nnfiles = run_parallel(nugenfiles, ncpus=args.ncpus)
        nweighter.add_weight_column("wght", nweighter.weight_cols["event_weight"])

        genie['ow'] /= gnfiles
        nugen['ow'] /= nnfiles

        #----------------------------------------------------
        # Build the joint weighter objects?
        # Using 2.0 here in order to force the weighting values 
        # to match the conventions for NuSources, but need to
        # verify that this is correct.
        # While we're here, verify that we're getting the same
        # values that we would have gotten directly from the files
        # without merging.
        #----------------------------------------------------
        gsurface = gweighter.surface
        nsurface = nweighter.surface
        joint_surface = gsurface + nsurface
        genie['uncorrected_ow'] = gweighter.get_weights(1) / 2.0
        nugen['uncorrected_ow'] = nweighter.get_weights(1) / 2.0

        # Ensure the values are all correct
        print('genie', np.unique((genie['uncorrected_ow']-genie['ow'])/genie['ow']))
        print('nugen', np.unique((nugen['uncorrected_ow']-nugen['ow'])/nugen['ow']))

        assert np.all(np.isclose(genie['uncorrected_ow'], genie['ow'], 1e-12))
        assert np.all(np.isclose(nugen['uncorrected_ow'], nugen['ow'], 1e-12))

        # Calculate the joint weights
        gweighter.surface = joint_surface
        nweighter.surface = joint_surface

        genie['ow'] = gweighter.get_weights(1) / 2.0
        nugen['ow'] = nweighter.get_weights(1) / 2.0

        # We know GENIE does not include coincident muons and that the cross
        # sections are potentially different, although the shapes of the cross
        # sections have been shown to be similar. The former suggests that we 
        # should trust the nugen weights better, but genie actually gives slightly 
        # lower rates than nugen in the overlap and so is technically more conservative. 
        # To fix this, let's rescale the nugen to match the GENIE rate from some 
        # part of the overlapping phase space. We'll just use an arbitrary E^-2 
        # here and match the total to genie in [100,150] GeV, which will already provide 
        # ~100k genie events and at least a few thousand nugen events.
        gweighter.surface, nweighter.surface = nsurface, gsurface
        genie['genie_nugen_overlap'] = (gweighter.get_weights(1) > 0)
        nugen['genie_nugen_overlap'] = (nweighter.get_weights(1) > 0)
        
        gw = (genie['uncorrected_ow'] * 1e-12 * genie['trueE']**-2)
        gw *= (genie['genie_nugen_overlap'])
        gw *= (100<genie['trueE']) & (genie['trueE']<150)

        # Merge them!
        output = np.concatenate([genie, nugen])
        
        w = (output['ow'] * 1e-12 * output['trueE']**-2)
        w *= (output['genie_nugen_overlap'])
        w *= (100<output['trueE']) & (output['trueE']<150)
        
        ratio = gw.sum() / w.sum()
        print(f"Ratio of joint to genie in [100,150] GeV is {ratio}")
        output['ow'][output['run']>10000] *= ratio

        if len(output) == 0:
            raise ValueError('Output array has zero elements! Panic!')
            
        results.append(output)

    results = np.concatenate(results)
    #----------------------------------------------------
    # Apply an energy mask
    #----------------------------------------------------
    mask = (results['logE'] > 0) & (results['logE'] < 3.5)
    results = results[mask]

    #----------------------------------------------------
    # and finally save it
    #----------------------------------------------------
    np.save(args.output, results)
    
if __name__ == "__main__":
    main()
