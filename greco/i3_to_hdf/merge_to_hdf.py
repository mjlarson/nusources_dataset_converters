#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/icetray-start
#METAPROJECT /data/user/mlarson/icetray/build/
##!/usr/bin/env python3
import os, shutil
import numpy as np
from scipy import optimize
from copy import deepcopy
from glob import glob
from icecube import dataclasses, dataio, icetray, simclasses, recclasses, gulliver, millipede, common_variables
from icecube.hdfwriter import I3HDFWriter
from icecube.icetray import I3Tray

def mip_energy(length):
    return length * 0.222 * icetray.I3Units.GeV/icetray.I3Units.m

def hadronic_to_em(had_energy):
    # https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/sandbox/mleuermann/Pegleg_cleanUp/trunk/private/pegleg/MillipedeSolver.cxx#L419
    E0 = 0.18791678
    m = 0.16267529
    f0 = 0.30974123
    e = 2.71828183
    en = e if (e>had_energy) else had_energy
    
    # This gives EM energy/hadronic energy!
    HF = 1-np.power(en/E0, -m) * (1-f0)
    return HF * had_energy
    
def em_to_hadronic(em_energy):
    # https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/sandbox/mleuermann/Pegleg_cleanUp/trunk/private/pegleg/MillipedeSolver.cxx#L430
    precision = 1e-2
    if em_energy < 2.71828183: return hadronic_to_em(em_energy)

    def f(e): 
        delta =  em_energy - hadronic_to_em(e)
        return delta        
    result = optimize.root_scalar(f, 
                                  bracket=[0.5*em_energy, 2*em_energy],
                                  method='brentq', 
                                  #rtol=precision,
                                  xtol=precision,
                                  )
    return result.root
    
def check_seeds(frame):
    seeds = ["MPEFitMuEX", "grecoGRECO_FirstHit", "grecoGRECO_SPEFit11",]
    required = ["SRTInIcePulses",]#"OfflinePulsesTimeRange"]
    for seed in seeds:
        if seed not in frame: 
            print(f"Seed {seed} is missing from frame.")
            return False
        if frame[seed].fit_status != dataclasses.I3Particle.OK: 
            print(f"Seed {seed} is not okay.")
            return False
    for key in required:
        if key not in frame: 
            print(f"Key {key} is missing from the frame")
            return False
    return True

def make_particles(frame):
    if 'Pegleg_Fit_NestleParticles' not in frame: return False
    particles = frame['Pegleg_Fit_NestleParticles']
    casc = particles[0]
    casc.energy = em_to_hadronic(casc.energy)

    track = deepcopy(casc)
    track.length = 0
    track.energy = 0
    for p in particles[1:]:
        track.energy += mip_energy(p.length)
        track.length += p.length
        
    if 'Pegleg_Fit_NestleHDCasc' in frame:
        del frame['Pegleg_Fit_NestleHDCasc']
    if 'Pegleg_Fit_NestleTrack' in frame:
        del frame['Pegleg_Fit_NestleTrack']
        
    frame['Pegleg_Fit_NestleHDCasc'] = casc
    frame['Pegleg_Fit_NestleTrack'] = track
    return

def extract_dst(frame):
    frame['n_string'] = icetray.I3Int(frame['I3DST'].n_string)
    frame['n_dom'] = icetray.I3Int(frame['I3DST'].ndom)

def extract_llh(frame):
    try:
        frame['Monopod_llh'] = dataclasses.I3Double(frame['Monopod_bestFitParams'].logl)
        frame['Pegleg_llh'] = dataclasses.I3Double(frame['Pegleg_Fit_NestleFitParams'].logl)
    except KeyError:
        print(frame)
        raise

def direct_hits(frame):
    definitions = common_variables.direct_hits.get_default_definitions()
    definitions.append(common_variables.direct_hits.I3DirectHitsDefinition("E",-15.,250.)) 
    
    pulsemapmask_name = 'SRTInIcePulses'
    pulsemap = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, pulsemapmask_name)
    particle = frame['Pegleg_Fit_Nestle']
    ndir_map = common_variables.direct_hits.calculate_direct_hits(definitions, frame['I3Geometry'], pulsemap, particle)
    qdir = ndir_map['A'].q_dir_pulses # [-15ns; +15ns] time window
    frame['qdir'] = dataclasses.I3Double(qdir)

def get_primary(frame):
    if 'I3MCTree_preMuonProp' in frame:
        mctree = frame['I3MCTree_preMuonProp']
    elif 'I3MCTree' in frame:
        mctree = frame['I3MCTree']
    else: 
        return
    frame['MCPrimary'] = mctree[0]
    frame['MCSecondary'] = mctree[1]

def append_id(frame, keylist):
    event = frame['I3EventHeader'].event_id
    keylist.append(event)
    return

def get_event_ids(recofiles, check_pegleg=False):
    event_list = []
    tray = I3Tray()
    tray.Add("I3Reader", FilenameList=recofiles)
    tray.Add(check_seeds)
    if check_pegleg: tray.Add(lambda frame: 'Pegleg_Fit_NestleParticles' in frame)
    tray.Add(append_id, keylist=event_list)
    tray.Execute()
    return event_list

def write_nfiles(frame, recofiles):
    frame['nfiles'] = icetray.I3Int(len(recofiles))
def dump_to_hdf(gcd, recofiles, output_filename):
    tray = I3Tray()
    tray.Add("I3Reader", FilenameList=[gcd, *recofiles])
    tray.Add(check_seeds)
    tray.Add(lambda frame: 'Pegleg_Fit_NestleParticles' in frame)
    tray.Add(write_nfiles, recofiles=recofiles)
    tray.Add(get_primary)
    tray.Add(extract_dst)
    tray.Add(extract_llh)
    tray.Add(make_particles)
    tray.Add(direct_hits)
    tray.Add(I3HDFWriter, 
             SubEventStreams=["InIceSplit",],
             output=output_filename,
             keys=['I3EventHeader',
                   'Pegleg_Fit_NestleTrack',
                   'Pegleg_Fit_NestleHDCasc', 
                   'Monopod_best',
                   'Pegleg_llh',
                   'Monopod_llh',
                   'qdir',
                   'n_string',
                   'n_dom',
                   'I3MCWeightDict',
                   'I3GENIEResultDict',
                   'MCPrimary',
                   'MCSecondary',
                   'nfiles'])
    tray.Execute()
    
    return


if __name__ == "__main__":
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcd", type=str, default="", action='store', required=True)
    parser.add_argument("--pre", type=str, default=[], action='append')
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tol", type=int, default=0.01)
    parser.add_argument("post", type=str, default=[], nargs='+')
    
    args = parser.parse_args()
    
    print("Reading the following pre-reco files.")
    for f in sorted(args.pre):
        print("\t", f)
    print("Reading the following post-reco files")
    for f in sorted(args.post):
        print("\t", f)
        
    print(f"Will accept differences up to {args.tol}%.")
    
    start = time.time()

    #######################
    # Get the event IDs before/after reco
    #######################
    events_prereco = get_event_ids(args.pre)
    events_postreco = get_event_ids(args.post, check_pegleg=True)

    if len(events_prereco)==0:
        print("WTF? Why are there zero events pre-reco??")
        raise AssertionError

    #######################
    # Ensure they match
    #######################
    overlap = list(set(events_prereco).intersection(events_postreco))
    difference = (len(events_prereco)-len(overlap))/len(events_prereco) - 1
    if (difference > 0) and (difference < args.tol):
        print(f"Found a difference of {difference*100}%, but that's less than tolerance. Passing.")
    elif difference > args.tol:
        print(f"Difference {difference*100}% >= tolerance. Failing file.")
        raise AssertionError

    #######################
    # Write out the hdf file
    #######################
    dump_to_hdf(args.gcd, args.post, args.output)

    proc_time = time.time() - start        
    print("Finished. Processing took {:4.3f} seconds".format(proc_time))
