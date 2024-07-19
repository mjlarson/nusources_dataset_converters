import os, sys, numpy as np
import numpy.lib.recfunctions as rf
import json
import argparse
import copy
from shutil import copyfile
from glob import glob
from os.path import basename, join
from astropy.time import Time
from tqdm import tqdm


def main(args):
    # ----------------------------------------------
    # Define a few useful functions...
    # ----------------------------------------------
    def load_npy(flist):
        output = []
        for f in sorted(flist):
            if len(output) == 0:
                output = np.load(f)
            else:
                output = np.append(output, np.load(f))
        return output

    def copy_npy(flist, new_path):
        for f in sorted(flist): 
            copyfile(f, join(new_path, basename(f)))
        return

    def blank_grl(i3live_json):
        grl_from_live = json.load(open(i3live_json, 'r'))['runs']
        dtype = np.dtype([('run', int), 
                          ('start', np.float64), ('stop', np.float64),
                          ('livetime', np.float32), ('events', int)])

        output = []
        for entry in tqdm(grl_from_live):
            if not entry['good_i3']: continue
            start = Time(entry['good_tstart']).to_value('mjd')
            stop = Time(entry['good_tstop']).to_value('mjd')
            output.append((entry['run'], start, stop, stop-start, 0))

        output = np.array(output, dtype=dtype)
        return output[np.argsort(output['run'])]

    # ----------------------------------------------
    # Initial steps: copy the old GRL. We don't want
    # to change those files.
    # ----------------------------------------------
    copy_npy(glob(join(args.previous, "GRL/*")), 
             join(args.outdir, "GRL/"))

    # ----------------------------------------------
    # Load the current data and the I3Live GRL
    # ----------------------------------------------
    data = {}
    for f in sorted(glob(join(args.indir, "*data*.npy"))):
        data[basename(f)] = np.load(f)

    new_grl = blank_grl(args.grl_json)
    grl_dtype = new_grl.dtype

    # ----------------------------------------------
    # And prepare the output GRL 
    # ----------------------------------------------
    for filename, events in data.items():

        # ------------------------------------------
        # Basic info: print the file we're working on
        # ------------------------------------------
        print(filename)

        if not args.update_old:
            if os.path.exists(join(args.outdir, "GRL", basename(filename))):
                print("\t Already exists. Do nothing.")
                continue

        # ------------------------------------------
        # Find only unique events by looking for their
        # uniquely defined (run, id, sub id) key
        # and remove duplicates
        # ------------------------------------------
        event_key = rf.drop_fields(events, 
                                   [_ for _ in events.dtype.names
                                    if not _ in ['run', 'event', 'subevent']])
        event_key, indices = np.unique(event_key, return_index=True)
        nevents = events.shape
        events = events[indices]

        # ------------------------------------------
        # Get the times for the events and use them
        # to double check the run numbers
        # ------------------------------------------
        event_time = events['time']
        run_indices = np.searchsorted(new_grl['start'], event_time) - 1
        events['run'] = new_grl['run'][run_indices]

        # ------------------------------------------
        # Cut out the first two minutes to handle a bug
        # in the 2012/2013 where strings turned on 
        # sequentially over about 2 minutes, resulting in
        # an unrecorded partial detector runtime.
        # ------------------------------------------
        if any([year in f for year in ["2012", "2013"]]):
            print("Removing partial detector runtime...")
            dt = events['time'] - new_grl['start'][run_indices]
            keep = dt > 120./3600./24.
            events = events[keep]
            print("\tTotal of {}/ events removed".format((~keep).sum(), len(keep)))

            event_time = events['time']
            run_indices = np.searchsorted(new_grl['start'], event_time) - 1

            new_grls[f]['start'] += 120./3600./24.
            new_grls[f]['livetime'] -= 120./3600./24.

        # ------------------------------------------
        # Every event must be in a run
        # ------------------------------------------
        good_mask = np.zeros(len(events), dtype=bool)
        for run in np.unique(events['run'].astype(int)):
            grl_mask = new_grl[(new_grl['run'].astype(int) == run)]
            mask = np.zeros(len(events), dtype=bool)
            for grl_entry in grl_mask:
                submask = (events['time'] >= grl_entry['start'])
                submask &= (events['time'] <= grl_entry['stop'])
                mask |= submask

            mask &= (events['run'].astype(int) == run)
            good_mask |= mask
            
        print("Total of {} good events and {} bad ones?".format(good_mask.sum(), (~good_mask).sum()))
        events = events[good_mask]
        event_time = events['time']
        run_indices = np.searchsorted(new_grl['start'], event_time) - 1
        
        # ------------------------------------------
        # Assign the numbers of events for each run in the new grl
        # ------------------------------------------
        grl = copy.deepcopy(new_grl)
        index, count = np.unique(run_indices, return_counts=True)
        grl['events'][index] = count
        grl = np.unique(grl[grl['events'] > 0])
                
        # ------------------------------------------
        # save it
        # ------------------------------------------
        print("Saving to {}".format(args.outdir))
        np.save(join(args.outdir, basename(filename)), events)
        np.save(join(args.outdir, "GRL/", basename(filename)), grl)
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--previous', type=str, required=True, help='Path to the previous version of GRECO')
    parser.add_argument('--indir', type=str, required=True, help='Numpy data file path to read in')
    parser.add_argument('--outdir', type=str, required=True, help='Numpy data file path to write to')
    parser.add_argument('--grl_json', type=str, required=True, help='Json file containing I3Live GRL information')
    parser.add_argument("--update_old", default=False, action="store_true", help="Update old GRL files?")
    args = parser.parse_args()
    main(args)
