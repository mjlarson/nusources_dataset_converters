#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/icetray-start
#METAPROJECT /data/user/mlarson/icetray/build/
##!/usr/bin/env python3
import os, shutil
import numpy as np
from scipy import optimize
from copy import deepcopy
from glob import glob
from icecube import dataclasses, dataio, icetray, simclasses, recclasses, gulliver, millipede

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
    
    if em_energy < 2.71828183:
        return hadronic_to_em(em_energy)

    def f(e): 
        delta =  em_energy - hadronic_to_em(e)
        return delta
        
    result = optimize.root_scalar(f, 
                                  bracket=[0.5*em_energy, 2*em_energy],
                                  method='brentq', 
                                  #rtol=precision,
                                  xtol=precision,
                                  )

    #print(em_energy, result.root, em_energy > result.root)
    return result.root
    

def make_particles(frame):
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

def read(prereco, postreco, output_filename, tolerance):
    output = dataio.I3File(output_filename, 'w')
    all_good = True
    
    events_passing_prereco = []

    for filename in sorted(prereco):
        print('Pre-reco file:', filename)
        i3file = dataio.I3File(filename)
        for frame in i3file:
            if frame.Stop != icetray.I3Frame.Physics:
                continue
            event = frame['I3EventHeader'].event_id
            events_passing_prereco.append(event)
        i3file.close()
    print('pre', len(events_passing_prereco))
            
    events_passing_postreco = []
    for i, filename in enumerate(sorted(postreco)):
        print('Post-reco file:', filename)
        if not all_good: 
            print("Found at least one bad file at i={}. Stopping.".format(i))
            break
        i3file = dataio.I3File(filename)
        while i3file.more():
            try: frame = i3file.pop_frame()
            except:
                all_good = False
                break
            if not frame.Stop == icetray.I3Frame.Physics:
                output.push(frame)
                continue
            
            if 'Pegleg_Fit_NestleParticles' not in frame:
                print("Not reconstructed")
                #all_good = False
                continue
                #break
            make_particles(frame)
            
            if 'Pegleg_Fit_NestleHDCasc' not in frame:
                print("No HDCasc.")
                all_good = False
                break
            if 'Pegleg_Fit_NestleTrack' not in frame:
                print("No track")
                all_good = False
                break
                
            event = frame['I3EventHeader'].event_id
            events_passing_postreco.append(event)
            output.push(frame)
        i3file.close()        
    output.close()

    print('post', len(events_passing_postreco))
    if not all_good:
        os.remove(output_filename)
        return False
    elif len(events_passing_prereco) - len(events_passing_postreco) < tolerance:
        print("Found a different number of events pre ({}) vs post ({}) reco.".format(len(events_passing_prereco),
                                                                                      len(events_passing_postreco)))
        print(f"But it's close (< {tolerance} events), so I'll pass it.")
    else:
        print("Found a different number of events pre ({}) vs post ({}) reco.".format(len(events_passing_prereco),
                                                                                      len(events_passing_postreco)))
        print(f"It's not close (> {tolerance} events difference), so this file fails.")
        os.remove(output_filename)
        return False
    return True

if __name__ == "__main__":
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=str, default=[], action='append', required=True)
    parser.add_argument("--post", type=str, default=[], action='append', required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tol", type=int, default=20)
    
    args = parser.parse_args()
    
    print("Reading the following pre-reco files.")
    for f in sorted(args.pre):
        print("\t", f)
    print("Reading the following post-reco files")
    for f in sorted(args.post):
        print("\t", f)
        
    print("Will accept differences up to {} events.".format(args.tol))
    
    start = time.time()
    read(args.pre, args.post, args.output, tolerance=args.tol,)
    proc_time = time.time() - start
    
    print("Finished. Processing took {:4.3f} seconds".format(proc_time))
