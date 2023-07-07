#!/usr/env python3
import os, sys, glob
import numpy as np

#########################################
# Going to do this in a braindead way.
# Feel free to spruce your own script
# up to make it more easily usable.
########################################

########################################
# Set up your paths and variables that you want to use
########################################
indir = "/data/user/brianclark/IceCube/EHE/data_full/level5/"

n_per_group = 1000
data = True

basedir = "/data/user/mlarson/ehe/"
script = basedir + "convert_ehe.py"
submit_file = basedir + "submit.sub"

dag_name = "ehe.dag"

outdir = basedir + "output/"
logdir = basedir + "logs/"

########################################
# Make a string where we can store our file contents
########################################
dag_contents = ""

########################################
# Start looping over things 
########################################
for dataset in glob.glob(os.path.join(indir, "*")):
    if "old" in dataset.lower(): continue

    # group the files
    filenames = sorted(glob.glob(os.path.join(dataset, "*/*.i3*")))
    filename_groups = [[],]
    for i, f in enumerate(filenames):
        if len(filename_groups[-1]) >= n_per_group:
            filename_groups.append([])
        filename_groups[-1].append(f)
        
    # every job needs a unique name.
    base_job_name = "ehe_{}".format(os.path.basename(dataset))
    print(base_job_name, len(filenames), len(filename_groups))

    for groupnum, group in enumerate(filename_groups):
        if len(group) == 0: continue
        job_name = f"{base_job_name}_{groupnum}"
    
        # And an output file name
        outfile = os.path.join(outdir, job_name+".npy")
        
        # Write out the command that you want condor to run
        cmd = "{} ".format(script)
        cmd += "--outfile {} ".format(outfile)
        cmd += f"-n {len(filenames)} "
        if data: cmd += "--data "
        for f in group:
            cmd += "{} ".format(f)
            
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
