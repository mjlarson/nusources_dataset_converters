# GRECO-specific stuff
GRECO processing is... special. Because we currently rely on an incredibly slow reconstruction (pegleg), there are complications during processing.

1. A single MC file has ~3000-4000 events, each of which take between 30-120 seconds (on average), working out to ~3 days or more of continuous CPU usage per file
2. We need a huge amount of CPU power to process the MC, meaning we need to use the grid.
3. Grid jobs have a hard limit of ~4 hours/job. Going over this causes issues with the grid sites.

In order to get around this, we need to split up each MC file into smaller bite-sized sub-files containing O(200) events. This still takes too long per job, however, since there are long tails to the reco time per event. This leads to a situation where the "simplest" solution is to stop processing when we approach 3.5 hours of processing time, write the file back to NPX, then re-submit the file to the grid as a new job. 

This complex scheme works, but leads to additional complications.
1. Files can become corrupted due to read/write errors or other issues
2. We do not a apriori know when a file is done without actively checking all frames.
3. We end up with ~100k small sub-files instead of ~2k MC files.

In order to manage this mess, we need to re-merge the sub-files, including checks along the way to handle missing events. We do this using the `merge_i3.py` script here. To use it, load a modern version of icetray and run 

```
python3 /data/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/merge_i3.py \
        --output /data/user/mlarson/combo_r129072/scripts/greco_online/skylab_dataset/v2.5/simplified/output/nue/NuE_120000_000000_level2.i3.bz2 \
        --pre /data/ana/LE/legacy/NBI_nutau_appearance/online/filtered/nue/120000/NuE_120000_000000_level2.i3.bz2 \
        --post /data/ana/PointSource/GRECO_online/version-002-p12/nue/NuE_120000_000000_level2.0000.i3.bz2 \
        --post /data/ana/PointSource/GRECO_online/version-002-p12/nue/NuE_120000_000000_level2.0001.i3.bz2 \
        ...
``` 

The command should include every sub-file created. This script will check that the vast majority of events in the pre-file have passed. In some cases, you'll see fewer events post-reconstruction than you had pre-reconstruction. This is due to various fit status checks during the processing and should only make up a small fraction of the events of O(1%).

