#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/icetray-start
#METAPROJECT icetray/v1.5.1
#!/usr/bin/env python3

import os, sys, glob
from argparse import ArgumentParser
import numpy as np

from icecube import dataclasses, dataio, icetray
from icecube.icetray import I3Units
from icecube.astro import dir_to_equa

try: from tqdm import tqdm
except: tqdm = lambda x: x

#----------------------------------------------------
# Build our argument parser.
#----------------------------------------------------
parser = ArgumentParser(description="Base script for converting i3 files to NuSources-compatible"
                        " npy files for use in csky and skyllh.")
parser.add_argument("input", metavar="i3file", type=str, nargs="+")
parser.add_argument("--outfile", "-o", type=str,
                    action = "store", default="", required=True,
                    help = "The output npy filename and path to write to")
parser.add_argument("--nfiles", "-n", type=int, default=2000)
parser.add_argument("--data", default=False, action="store_true")
args = parser.parse_args()

#----------------------------------------------------
# Set up our output
#----------------------------------------------------
output = []
if args.data:
    dtype = [('run', np.int32), ('subevent', np.int32), ('event', np.int64),
             ('time', np.float64), ('logE', np.float32),
             ('ra', np.float32), ('dec', np.float32), ('sigma', np.float32)]
else:
    dtype = [('run', np.int32), ('subevent', np.int32), ('event', np.int64),
             ('time', np.float64), ('ow', np.float64),
             ('trueRa', np.float32), ('trueDec', np.float32),
             ('trueAzi', np.float32), ('trueZen', np.float32),
             ('trueE', np.float32), ('logE', np.float32),
             ('ra', np.float32), ('dec', np.float32), ('sigma', np.float32)]


#----------------------------------------------------
# Get the oneweight values 
#----------------------------------------------------
class OneWeighter:
    def __init__(self, i3file):
        npz_filename = os.path.basename(i3file)
        npz_filename = npz_filename.replace("Level5", "oneweight_information")
        npz_filename = npz_filename.replace("i3.zst", "npz")
        npz_filename = i3file.split("/")[-3] + "/" + npz_filename
        
        path_to_npz = f"/data/i3store/users/ssclafani/ehe_stuff/oneweights/{npz_filename}"
        self.npz = np.load(path_to_npz)
        self.index = 0

        self.propweights = self.npz['propagated_weights']
        hist, bins = np.histogram(self.npz['energy_bin_centers'],
                                  bins=self.npz['energy_bins'],
                                  weights=self.npz['ehe_energy_weights'])
        self.bin_midpoints = bins[:-1] + np.diff(bins)/2
        cdf = np.cumsum(hist)
        self.cdf = cdf/cdf[-1]

    def power_law_flux(self, e_in_gev):
        return 1E-8 * (e_in_gev**-2.)

    def get(self):
        value_bins = np.searchsorted(self.cdf, np.random.uniform())
        true_e = self.bin_midpoints[value_bins]
        return_values = (true_e, self.propweights[self.index]/self.power_law_flux(true_e))
        self.index += 1
        return return_values
    
#----------------------------------------------------
# Pick the energy and direction...
# Returning (direction reco, energy reco, sigma)
#----------------------------------------------------
def particle_picker(frame):
    try:
        return (frame['EHE_Monopod'], frame['Homogenized_QTot'].value, 10*I3Units.degree)
    except:
        return (frame['EHE_SplineMPE'], frame['Homogenized_QTot'].value, 1*I3Units.degree)

#----------------------------------------------------
# Start walking through the file
#----------------------------------------------------
for i3filename in tqdm(args.input):
    i3file = dataio.I3File(i3filename, 'r')
    if not args.data:
        ower = OneWeighter(i3filename)
    
    while i3file.more():
        frame = i3file.pop_frame()

        #===========================================
        # Otherwise just wait for P-frames
        #===========================================
        if not frame.Stop == icetray.I3Frame.Physics: continue

        #===========================================
        # Get the event information
        #===========================================
        header = frame['I3EventHeader']
        mjd = header.start_time.mod_julian_day_double

        if not args.data:
            # Find the primary
            try: primary = frame['MCPrimary']
            except:
                mctree = sorted([key for key in frame if 'I3MCTree' in key])[0]
                primary = dataclasses.get_most_energetic_neutrino(frame[mctree])
            
            truera, truedec = dir_to_equa(zenith=primary.dir.zenith,
                                          azimuth=primary.dir.azimuth,
                                          mjd = mjd)
            #===========================================
            # Oneweight nonsense.
            #===========================================
            # NuSources reports fluxes as the sum of nu and nubar fluxes.
            # To correctly handle this down the road, we need to correct
            # the oneweight values now by dividing by a factor of 2.
            true_e, ow = ower.get()
            ow /= args.nfiles
            ow /= 2.0
            
        #===========================================
        # And the recos
        #===========================================
        direction, energy, sigma = particle_picker(frame)
            
        # Convert from local to equatorial coordinates
        ra, dec = dir_to_equa(zenith=direction.dir.zenith,
                              azimuth=direction.dir.azimuth,
                              mjd = mjd)

        #===========================================
        # And write it out. The order here MUST match
        # the order in the `dtype` variable above or
        # else you'll end up with wrong labels.
        #===========================================
        if args.data:
            current_event = [header.run_id,         # run
                             header.sub_event_id,   # subevent
                             header.event_id,       # event
                             mjd,                   # time
                             np.log10(energy),      # logE
                             ra,                    # ra
                             dec,                   # dec
                             sigma,                 # sigma
            ]
        else:
            current_event = [header.run_id,         # run
                             header.sub_event_id,   # subevent
                             header.event_id,       # event
                             mjd,                   # time
                             ow,                    # ow
                             truera,                # trueRa
                             truedec,               # trueDec
                             primary.dir.azimuth,   # trueAzi
                             primary.dir.zenith,    # trueZen
                             true_e,                # trueE
                             np.log10(energy),      # logE
                             ra,                    # ra
                             dec,                   # dec
                             sigma,                 # sigma
            ]
        output.append(tuple(current_event))


#----------------------------------------------------
# Convert the output list into a numpy array and save it
#----------------------------------------------------
output = np.array(output, dtype=dtype)
np.save(args.outfile, output)
