#!/usr/bin/python
import os, sys
from glob import glob
from os.path import basename, join

## GENIE files
'''predir = "/data/ana/LE/legacy/NBI_nutau_appearance/online/filtered/"
postdir = "/data/ana/PointSource/GRECO_online/version-002-p13/"
prefiles = {'nue':glob(join(predir, "nue", "120000", "*")),
            'numu':glob(join(predir, "numu", "140000", "*")),
            'nutau':glob(join(predir, "nutau", "160000", "*"))}
postfiles = {'nue':glob(join(postdir, 'nue', '*')),
             'numu':glob(join(postdir, 'numu', '*')),
             'nutau':glob(join(postdir, 'nutau', '*'))}
dagname  = 'greco_hdf_genie.dag'
magnitude_to_merge = 2 # This is orders of magnitude, so 2 means merge 100 files.
'''

## NuGen files
predir = "/data/ana/LE/legacy/NBI_nutau_appearance/online/filtered/"
postdir = "/data/user/mlarson/combo_r129072/scripts/file_shuffler/complete_nugen/"

prefiles = {'nugen_nue':glob(join(predir, 'nue', '20885', '*')),
            'nugen_numu':glob(join(predir, 'numu', '20878', '*')),
            'nugen_nutau':glob(join(predir, 'nutau', '20895', '*'))}
postfiles = {'nugen_nue':glob(join(postdir, '20885', '*')),
             'nugen_numu':glob(join(postdir, '20878', '*')),
             'nugen_nutau':glob(join(postdir, '20895', '*'))}
dagname  = 'greco_hdf_nugen.dag'
magnitude_to_merge = 2 # This is orders of magnitude, so 2 means merge 100 files.


outdir_base = "/data/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/output_hdf/"
gcd = "/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"


dag_contents = ''

# Start.
for flavor in prefiles.keys():
    print(flavor)
    outdir = join(outdir_base, flavor)

    bases = []
    for prefile in sorted(prefiles[flavor]):
        base = basename(prefile)
        base = base[:base.index(".i3")]
        if 'level2' in base: 
            base = base[:base.index("_level2")]
        base = base[:-magnitude_to_merge]
        bases.append(base)
    bases = set(bases)
    
    print(f"Found {len(bases)} jobs to run for this flavor")

    for base in sorted(bases):
        current_pre = [_ for _ in prefiles[flavor] if base in _]
        current_post = [_ for _ in postfiles[flavor] if base in _]
        if len(current_post) == 0:
            continue

        cmd = os.path.expandvars("$PWD/merge_to_hdf.py")
        cmd += " --gcd {}".format(gcd)
        for f in current_pre:
            cmd += " --pre {}".format(f)

        cmd += " --output {}".format(join(outdir, base))
        for i in range(magnitude_to_merge):
            cmd += "X"
        cmd += ".hdf"

        for f in current_post:
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

