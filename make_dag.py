#!/usr/env python3
import os, sys
from glob import glob
from tqdm import tqdm
import numpy as np

#########################################
# Going to do this in a braindead way.
# Feel free to spruce your own script
# up to make it more easily usable.
########################################

########################################
# Set up your paths and variables that you want to use
########################################
indir_base = "/data/ana/PointSource/PS/version-005-p00/"
indirs = []

indirs = sorted(glob(os.path.join(indir_base, "IC86*/i3/")))
ngroup = 100

indirs += sorted(glob(os.path.join(indir_base, "IC86.2014/mc/*/i3/")))
ngroup = 100

current = os.path.expandvars("$PWD")
script = os.path.join(current,"convert_pstracks.py")
submit_file = os.path.join(current,"submit.sub")

dag_name = "pstracks_npy.dag"

logdir = os.path.join(current, "logs_pstracks/")

########################################
# Make a string where we can store our file contents
########################################
dag_contents = ""

########################################
# Start looping over things 
########################################
for dataset in indirs:
    # group the files
    filenames = sorted(glob(os.path.join(dataset, "*upgoing*.i3*")))
    nfiles = len(filenames)
    
    # every job needs a unique name.
    run_folder = dataset.split("/")[-3]

    # Group by ngroup
    groups = []
    while len(filenames) > 0:
        group, filenames = filenames[:ngroup], filenames[ngroup:]
        groups.append(group)

    print(f"Adding {run_folder} with {nfiles} files in {len(groups)} groups")

    printed = False
    for group in groups:
        job_name = f"PSTracks_{group[0]}_group{ngroup}".replace(".","_")
    
        outdir = os.path.dirname(group[0])
        outdir = outdir.replace("/i3", "/npy")
        if not printed:
            print(outdir)
            printed = True

        # Write out the command that you want condor to run
        cmd = "{} ".format(script)
        cmd += f" --outdir {outdir}"
        for filename in group:
            cmd += f" {filename}" 
            
        # Now we start writing Condor stuff.
        # We'll start with the basic stuff that's always here
        # no matter what you're trying to run
        dag_contents += f"JOB {job_name} {submit_file}\n"
        dag_contents += f"VARS {job_name} "
        dag_contents += f" job_name=\"{job_name}\" "
        dag_contents += f" log_dir=\"{logdir}\" "
        dag_contents += f" cmd=\"{cmd}\" "
        dag_contents += "\n"
        
########################################
# Write the dag file out
########################################
open(dag_name, 'w').write(dag_contents)
