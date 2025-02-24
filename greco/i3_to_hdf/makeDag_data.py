#!/usr/bin/python
import os, sys
import functools
from glob import glob
from os.path import basename, join

@functools.lru_cache(maxsize=2)
def gcds_for_year(year, filterdir):
    gcds = (glob(f"/data/exp/IceCube/{year}/filtered/{filterdir}/*/*/*GCD*")
            + glob(f"/data/exp/IceCube/{year+1}/filtered/{filterdir}/*/*/*GCD*"))
    gcd_map = {}
    for gcd in gcds:
        try:
            run = gcd.split("/")[8]
            run = run.split("_")[0]
            run = int(run[3:])
        except: continue
        gcd_map[run] = gcd
        
    return gcd_map
        
def find_gcd(path):
    filename = basename(path)
    season = filename.split("_")[1]
    year = int(season.split(".")[-1])
    filterdir = filename.split("_")[0].lower()
    run = int(filename.split("_")[3][3:])

    gcd_map = gcds_for_year(year, filterdir)
    if run not in gcd_map:
        print("No gcd file found for {filename}. Checked these directories:\n",
              f"\t /data/exp/IceCube/{year}/filtered/{filterdir}/*/{run}/*GCD*\n"
              f"\t /data/exp/IceCube/{year+1}/filtered/{filterdir}/*/{run}/*GCD*\n")
        return None

    return gcd_map[run]
    
predir = "/data/ana/LE/legacy/NBI_nutau_appearance/online/filtered/data/"
postdir_base = "/data/ana/PointSource/GRECO_online/version-002-p13/"
outdir_base = '/data/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/output/'

check_exists = False

# get the prefiles
all_prefiles = {}
for season in os.listdir(predir):
    if 'old' in season: continue
    all_prefiles[season] = glob(join(predir, season, "*"))

dagname  = 'greco_data_merging.dag'
dag_contents = ''

# Start.
for season in sorted(all_prefiles.keys()):

    prefiles = all_prefiles[season]
    postdir = join(postdir_base, season)
    outdir = join(outdir_base, season)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for prefile in sorted(prefiles):
        base = basename(prefile)
        base = base[:base.index(".i3")]
        postfiles = glob(join(postdir, base + "*"))
        if len(postfiles) == 0:
            continue
        if len(postfiles) > 1:
            print("... found too many postfiles?")
            for f in postfiles:
                print(f)
            sys.exit()

        outfile = join(outdir, basename(prefile))
        if check_exists and os.path.exists(outfile):
            continue

        gcd = find_gcd(prefile)

        cmd = "/data/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/nusources_dataset_converters/greco/i3_merging/merge_i3.py"
        cmd += " --pre {}".format(prefile)
        cmd += " --gcd {}".format(gcd)
        print('pre:', prefile)
        print('post:')
        for f in postfiles:
            print('\t', f)
            cmd += " --post {}".format(f)
        cmd += " --output {}".format(outfile)

        job = base.replace(".", "_")
        dag_contents += "JOB %s submit.sub\n" % (job)
        dag_contents += "VARS %s JOBNAME=\"%s\"" % (job, base)
        dag_contents += " cmd=\"%s\"" % (cmd)
        dag_contents += "\n"
            
if dag_contents != "":
    dagman = open(dagname, "w")
    dagman.write(dag_contents)
    dagman.close()

