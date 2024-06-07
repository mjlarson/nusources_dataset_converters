#!/usr/bin/python
import os, sys
from glob import glob
from os.path import basename, join

indir = "/data/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/output/"
outdir = "./output"

dagname  = 'greco_npy.dag'
dag_contents = ''

# Start.
for flavor in ['nue', 'numu', 'nutau']:
    print(flavor)
    
    filenames = glob(join(indir, flavor, "*"))
    filenames = sorted(filenames)

    for f in filenames:
        base = basename(f)
        base = base[:base.index(".i3")]

        cmd = "/data/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/nusources_dataset_converters/greco/npy_conversion/convert_greco_andrew.py"
        cmd += " --output {}".format(join(outdir, flavor, basename(f)))
        cmd += " --nfiles {}".format(len(filenames))
        cmd += " --genie-icetray"
        cmd += " {}".format(f)

        job = base.replace(".", "_")
        dag_contents += "JOB %s submit.sub\n" % (job)
        dag_contents += "VARS %s JOBNAME=\"%s\"" % (job, base)
        dag_contents += " cmd=\"%s\"" % (cmd)
        dag_contents += "\n"
            
if dag_contents != "":
    dagman = open(dagname, "w")
    dagman.write(dag_contents)
    dagman.close()

