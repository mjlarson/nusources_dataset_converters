sleep $1
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2011/npy/ -o ./version-005-p00/IC86_2021_exp.npy --ncpu=$2
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2012/npy/ -o ./version-005-p00/IC86_2012_exp.npy --ncpu=$2
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2013/npy/ -o ./version-005-p00/IC86_2013_exp.npy --ncpu=$2
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2014/npy/ -o ./version-005-p00/IC86_2014_exp.npy --ncpu=$2
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2015/npy/ -o ./version-005-p00/IC86_2015_exp.npy --ncpu=$2
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2016/npy/ -o ./version-005-p00/IC86_2016_exp.npy --ncpu=$2
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2017/npy/ -o ./version-005-p00/IC86_2017_exp.npy --ncpu=$2
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2018/npy/ -o ./version-005-p00/IC86_2018_exp.npy --ncpu=$2
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2019/npy/ -o ./version-005-p00/IC86_2019_exp.npy --ncpu=$2
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2020/npy/ -o ./version-005-p00/IC86_2020_exp.npy --ncpu=$2
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2021/npy/ -o ./version-005-p00/IC86_2021_exp.npy --ncpu=$2
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2022/npy/ -o ./version-005-p00/IC86_2022_exp.npy --ncpu=$2


python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2014/mc/22644/npy/ -o ./mc/IC86_mc_22644.npy --ncpu=$2
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2014/mc/22645/npy/ -o ./mc/IC86_mc_22645.npy --ncpu=$2
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2014/mc/22646/npy/ -o ./mc/IC86_mc_22646.npy --ncpu=$2
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2014/mc/22666/npy/ -o ./mc/IC86_mc_22666.npy --ncpu=$2
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2014/mc/22667/npy/ -o ./mc/IC86_mc_22667.npy --ncpu=$2
python merge_npy.py -i /data/ana/PointSource/PS/version-005-p00/IC86.2014/mc/22668/npy/ -o ./mc/IC86_mc_22668.npy --ncpu=$2
python merge_npy.py -i ./mc/ -o ./version-005-p00/IC86_mc.npy --nfiles 1
