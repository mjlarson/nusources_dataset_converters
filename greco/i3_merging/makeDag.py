#!/usr/bin/python
import os, sys
from glob import glob
from os.path import basename, join

predir = "/data/ana/LE/legacy/NBI_nutau_appearance/online/filtered/"
postdir_base = "/data/ana/PointSource/GRECO_online/version-002-p12/"
outdir_base = '/data/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/output/'

# get the prefiles
all_prefiles = {'nue':glob(join(predir, "nue", "120000", "*")),
                'numu':glob(join(predir, "numu", "140000", "*")),
                'nutau':glob(join(predir, "nutau", "160000", "*"))}

dagname  = 'greco_merging.dag'
dag_contents = ''

# Start.
for flavor in ['nue', 'numu', 'nutau']:
    print(flavor)

    prefiles = all_prefiles[flavor]
    postdir = join(postdir_base, flavor)
    outdir = join(outdir_base, flavor)

    for prefile in sorted(prefiles):
        base = basename(prefile)
        base = base[:base.index(".i3")]
        postfiles = glob(join(postdir, base + "*"))
        if len(postfiles) == 0:
            continue

        cmd = "/data/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/merge_i3.py"
        cmd += " --pre {}".format(prefile)
        print('pre:', prefile)
        print('post:')
        for f in postfiles:
            print('\t', f)
            cmd += " --post {}".format(f)
        cmd += " --output {}".format(join(outdir, basename(prefile)))

        job = base.replace(".", "_")
        dag_contents += "JOB %s submit.sub\n" % (job)
        dag_contents += "VARS %s JOBNAME=\"%s\"" % (job, base)
        dag_contents += " cmd=\"%s\"" % (cmd)
        dag_contents += "\n"
            
if dag_contents != "":
    dagman = open(dagname, "w")
    dagman.write(dag_contents)
    dagman.close()

