#!/usr/bin/python
import os, sys
from glob import glob
from os.path import basename, join

## GENIE files
#predir = "/data/ana/LE/legacy/NBI_nutau_appearance/online/filtered/"
#postdir_base = "/data/ana/PointSource/GRECO_online/version-002-p12/"
#outdir_base = '/data/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/output/'
#prefiles = {'nue':glob(join(predir, "nue", "120000", "*")),
#                'numu':glob(join(predir, "numu", "140000", "*")),
#                'nutau':glob(join(predir, "nutau", "160000", "*"))}
#postfiles = {'nue':glob(join(postdir, 'nue', '*')),
#             'numu':glob(join(postdir, 'numu', '*')),
#             'nutau':glob(join(postdir, 'nutau', '*'))}

## NuGen files
predir = "/data/ana/LE/legacy/NBI_nutau_appearance/online/filtered/"
postdir = "/data/user/mlarson/combo_r129072/scripts/file_shuffler/complete_nugen/"
outdir_base = "/data/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/output/"

prefiles = {'nugen_nue':glob(join(predir, 'nue', '20885', '*')),
            'nugen_numu':glob(join(predir, 'numu', '20878', '*')),
            'nugen_nutau':glob(join(predir, 'nutau', '20895', '*'))}
postfiles = {'nugen_nue':glob(join(postdir, '20885', '*')),
             'nugen_numu':glob(join(postdir, '20878', '*')),
             'nugen_nutau':glob(join(postdir, '20895', '*'))}

gcd = "/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"


dagname  = 'greco_merging.dag'
dag_contents = ''

# Start.
for flavor in prefiles.keys():
    print(flavor)
    outdir = join(outdir_base, flavor)

    for prefile in sorted(prefiles[flavor]):
        base = basename(prefile)
        base = base[:base.index(".i3")]
        current_post = [_ for _ in postfiles[flavor] if base in _]
        if len(current_post) == 0:
            continue

        cmd = "/data/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/nusources_dataset_converters/greco/i3_merging/merge_i3.py"
        cmd += " --gcd {}".format(gcd)
        cmd += " --pre {}".format(prefile)
        print('pre:', prefile)
        print('post:')
        for f in current_post:
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

