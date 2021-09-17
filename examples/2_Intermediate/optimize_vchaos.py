#!/usr/bin/env python

import logging
import numpy as np
from simsopt.util.mpi import MpiPartition, log
from simsopt.mhd.spec import Spec, ChaosVolume
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.solve.serial import least_squares_serial_solve
import os

"""
In this example, we show how the shape of a boundary magnetic
surface can be adjusted to eliminate magnetic islands inside it,
considering a vacuum field. For this example we will use the SPEC code
with a single radial domain. The geometry comes from a quasi-helically
symmetric configuration developed at the University of Wisconsin. We
will eliminate the islands by minimizing an objective function
involving Greene's residue for several O-points and X-points, similar
to the approach of Hanson and Cary (1984).
"""

log()

mpi = MpiPartition()
mpi.write()

# Initialze a Spec object from a standard SPEC input file:
s = Spec(os.path.join(os.path.dirname(__file__), 'inputs', 'Input_test.sp'), mpi=mpi)

s.keep_all_files = True

# To make this example run relatively quickly, we will optimize in a
# small parameter space. Here we pick out just 2 Fourier modes to vary
# in the optimization:
s.boundary.all_fixed()
s.boundary.set_fixed('rc(2,3)', False)
s.boundary.set_fixed('rc(2,2)', False)
s.boundary.set_fixed('zs(2,3)', False)
s.boundary.set_fixed('zs(2,2)', False)

print(s.get_dofs())


# Objective function is Vchaos**2
vchaos = ChaosVolume(spec=s, critical_dim=1.3, nppts=1e3)
prob = LeastSquaresProblem([(vchaos, 0, 1)])

# Solve the optimization problem:
least_squares_serial_solve(prob, grad=True)
