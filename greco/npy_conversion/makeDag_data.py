#!/usr/bin/python
import os, sys
from glob import glob
from os.path import basename, join

#indir = "/data/ana/PointSource/GRECO_online/version-002-p12/"
indir = "/data/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/output/"
outdir_base = "/data/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/nusources_dataset_converters/greco/npy_conversion/output"

dagname  = 'greco_data_npy.dag'
dag_contents = ''

for season in sorted(os.listdir(indir)):
    if 'IC86' not in season: continue
    if 'backup' in season: continue
    base = season
    print(season)

    outdir = os.path.join(outdir_base, season)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    filenames = glob(join(indir, season, "*"))
    filenames = sorted(filenames)
    if len(filenames) == 0: continue
    cmd = "/data/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/nusources_dataset_converters/greco/npy_conversion/convert_greco.py"
    cmd += " --output {}".format(join(outdir, season + ".npy"))
    cmd += " --nfiles {}".format(len(filenames))
    cmd += " {}".format(join(indir, season))
        
    job = base.replace(".", "_")
    dag_contents += "JOB %s submit.sub\n" % (job)
    dag_contents += "VARS %s JOBNAME=\"%s\"" % (job, base)
    dag_contents += " cmd=\"%s\"" % (cmd)
    dag_contents += "\n"
            
if dag_contents != "":
    dagman = open(dagname, "w")
    dagman.write(dag_contents)
    dagman.close()




