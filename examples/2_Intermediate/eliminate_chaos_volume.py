#!/usr/bin/env python

import logging
import numpy as np
from simsopt.util.mpi import MpiPartition, log
from simsopt.mhd.spec import Spec, ChaosVolume
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.solve.mpi import least_squares_mpi_solve
import os

"""
In this example, we show how the shape of a boundary magnetic
surface can be adjusted to the chaos volume inside it,
considering a vacuum field. For this example we will use the SPEC code
with a single radial domain. The geometry comes from a quasi-helically
symmetric configuration developed at the University of Wisconsin. We
will attempt to eliminate the chaos by minimizing an objective function
representing the chaos volume.
"""

log()

mpi = MpiPartition()
mpi.write()

# Initialze a Spec object from a standard SPEC input file:
s = Spec(os.path.join(os.path.dirname(__file__), 'inputs', 'Input_test.sp'), mpi=mpi)

s.inputlist.nppts = 1000
s.inputlist.lfindzero=0
s.keep_all_files = True

chaos_volume = ChaosVolume(spec=s, critical_dim=1.3, boxcount_method=False, write_chaos_vol_to_file=True)
initial_chaos_vol = chaos_volume.J()

chaos_volume.boxcount_method = True
initial_chaos_vol_boxcount = chaos_volume.J()

print(initial_chaos_vol)
# approx 0.066
print(initial_chaos_vol_boxcount)
# approx 0.088
