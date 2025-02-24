#!/usr/bin/env python
import copy
import numpy as np
from angular_uncertainty_bdt import predict_uncertainty

from argparse import ArgumentParser

#----------------------------------------------------
# Build our argument parser.
#----------------------------------------------------
parser = ArgumentParser(description="Base script for converting i3 files to NuSources-compatible"
                        " npy files for use in csky and skyllh.")
parser.add_argument("--input", metavar="input", type=str,
                    default="", required=True,
                    help = "")
args = parser.parse_args()


x = np.load(args.input)
x['angErr'] = predict_uncertainty(x)
x['angErr_noCorrection'] = copy.deepcopy(x['angErr'])
np.save(args.input, x)
