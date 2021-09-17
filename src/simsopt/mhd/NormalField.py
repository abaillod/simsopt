# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a class used to optimize the B.n harmonics on SPEC free-boundary
equilibria.
"""

# These next 2 lines Use double precision:
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jacrev, jit

import numpy as np
import logging
from mpi4py import MPI
from simsopt._core.util import isbool
from simsopt._core.optimizable import Optimizable

logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

# Here I have NormalField subclass optimizable
class NormalField(Optimizable):

    def __init__(self, nfp=1, stellsym=True, mpol=1, ntor=0):
        if not isinstance(nfp, int):
            raise TypeError('nfp must be an integer')
        if not isbool(stellsym):
            raise TypeError('stellsym must be a bool')
        self.nfp = nfp
        self.stellsym = stellsym

        Optimizable.__init__(self)

        mpol = int(mpol)
        ntor = int(ntor)
        # Perform some validation.
        if mpol < 1:
            raise ValueError("mpol must be at least 1")
        if ntor < 0:
            raise ValueError("ntor must be at least 0")
        self.mpol = mpol
        self.ntor = ntor

        self.allocate()

    def allocate(self):
        logger.info("Allocating SurfaceRZFourier")
        self.mdim = self.mpol + 1
        self.ndim = 2 * self.ntor + 1
        myshape = (self.mdim, self.ndim)

        self.vs = np.zeros(myshape)
        self.names = self.make_names('vns', False)
        
        if not self.stellsym:
            self.vc = np.zeros(myshape)
            self.names += self.make_names('vnc', True)

    def change_resolution(self, mpol, ntor):
        """
        Change the values of mpol and ntor. Any new Fourier amplitudes
        will have a magnitude of zero.  Any previous nonzero Fourier
        amplitudes that are not within the new range will be
        discarded.
        """
        old_mpol = self.mpol
        old_ntor = self.ntor
        old_vs = self.vs
        if not self.stellsym:
            old_vc = self.vc
        self.mpol = mpol
        self.ntor = ntor
        self.allocate()
        if mpol < old_mpol or ntor < old_ntor:
            # Don't need to recalculate if we only add zeros
            self.recalculate = True
            self.recalculate_derivs = True
            
        min_mpol = np.min((mpol, old_mpol))
        min_ntor = np.min((ntor, old_ntor))
        for m in range(min_mpol + 1):
            for n in range(-min_ntor, min_ntor + 1):
                self.vs[m, n + ntor] = old_vs[m, n + old_ntor]
                if not self.stellsym:
                    self.vc[m, n + ntor] = old_vc[m, n + old_ntor]

    
    def make_names(self, prefix, include0):
        """
        Form a list of names of the vc, vs array elements.
        """
        if include0:
            names = [prefix + "(0,0)"]
        else:
            names = []

        names += [prefix + '(0,' + str(n) + ')' for n in range(1, self.ntor + 1)]
        for m in range(1, self.mpol + 1):
            names += [prefix + '(' + str(m) + ',' + str(n) + ')' for n in range(-self.ntor, self.ntor + 1)]
        return names


    def _validate_mn(self, m, n):
        """
        Check whether m and n are in the allowed range.
        """
        if m < 0:
            raise ValueError('m must be >= 0')
        if m > self.mpol:
            raise ValueError('m must be <= mpol')
        if n > self.ntor:
            raise ValueError('n must be <= ntor')
        if n < -self.ntor:
            raise ValueError('n must be >= -ntor')
    
    def get_vc(self, m, n):
        """
        Return a particular vc Parameter.
        """
        if self.stellsym:
            return ValueError( \
                'vc does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        return self.vc[m, n + self.ntor]

    def get_vs(self, m, n):
        """
        Return a particular vs Parameter.
        """
        self._validate_mn(m, n)
        return self.vs[m, n + self.ntor]

    def set_vs(self, m, n, val):
        """
        Set a particular vs Parameter.
        """
        self._validate_mn(m, n)
        self.vs[m, n + self.ntor] = val
        self.recalculate = True
        self.recalculate_derivs = True

    def set_vc(self, m, n, val):
        """
        Set a particular vc Parameter.
        """
        if self.stellsym:
            return ValueError( \
                'vc does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        self.vc[m, n + self.ntor] = val
        self.recalculate = True
        self.recalculate_derivs = True
    

    def get_dofs(self):
        """
        Return a 1D numpy array with all the degrees of freedom.
        """
        mpol = self.mpol
        ntor = self.ntor
        if self.stellsym:
            return np.concatenate( \
                (self.vs[0, ntor + 1:], \
                 self.vs[1:, :].reshape(mpol * (ntor * 2 + 1))))
        else:
            return np.concatenate( \
                (self.vc[0, ntor:], \
                 self.vc[1:, :].reshape(mpol * (ntor * 2 + 1)), \
                 self.vs[0, ntor + 1:], \
                 self.vs[1:, :].reshape(mpol * (ntor * 2 + 1))))

    def set_dofs(self, v):
        """
        Set the shape coefficients from a 1D list/array
        """

        n = len(self.get_dofs())
        if len(v) != n:
            raise ValueError('Input vector should have ' + str(n) + \
                             ' elements but instead has ' + str(len(v)))
        
        # Check whether any elements actually change:
        if np.all(np.abs(self.get_dofs() - np.array(v)) == 0):
            logger.info('set_dofs called, but no dofs actually changed')
            return

        logger.info('set_dofs called, and at least one dof changed')
        self.recalculate = True
        self.recalculate_derivs = True
        
        mpol = self.mpol # Shorthand
        ntor = self.ntor
        if self.stellsym:
            self.vs[0, ntor+1:2*ntor+1] = v[0:ntor]
            self.vs[1:,:] = np.array(v[ntor:ntor+mpol*(ntor * 2 + 1)]).reshape(mpol, ntor * 2 + 1)

        else:
            self.vc[0, ntor:2*ntor+1] = v[0:ntor+1]
            self.vc[1:, :] = np.array(v[ntor+1:ntor+1+mpol*(2*ntor+1)]).reshape(mpol, ntor * 2 + 1)
            self.vs[0, ntor+1:2*ntor+1] = v[ntor+1+mpol*(2*ntor+1):2*ntor+1+mpol*(2*ntor+1)]
            self.vs[1:, :] = np.array(v[2*ntor+1+mpol*(2*ntor+1):2*ntor+1+2*mpol*(2*ntor+1)]).reshape(mpol, 2*ntor+1)

    def fixed_range(self, mmin, mmax, nmin, nmax, fixed=True):
        """
        Set the 'fixed' property for a range of m and n values.

        All modes with m in the interval [mmin, mmax] and n in the
        interval [nmin, nmax] will have their fixed property set to
        the value of the 'fixed' parameter. Note that mmax and nmax
        are included (unlike the upper bound in python's range(min,
        max).)
        """
        for m in range(mmin, mmax + 1):
            for n in range(nmin, nmax + 1):
                self.set_fixed('vns({},{})'.format(m, n), fixed)
                if not self.stellsym:
                    self.set_fixed('vnc({},{})'.format(m, n), fixed)