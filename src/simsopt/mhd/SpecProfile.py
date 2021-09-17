

# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a class used to represent any kind of profile for a SPEC equilibrium
"""

# These next 2 lines Use double precision:
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jacrev, jit

import numpy as np
import logging
from mpi4py import MPI
from simsopt._core.util import  isbool
from simsopt._core.optimizable import Optimizable

logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

# Here I have BdotN subclass optimizable
class SpecProfile(Optimizable):

    def __init__(self, Nvol, Lfreebound, ProfileName, Cumulative=False):
        if not isinstance(Nvol, int):
            raise TypeError('nfp must be an integer')
        if not isinstance(Lfreebound, int):
            raise TypeError('Lfreebound must be an integer')
        if not isinstance(ProfileName, str):
            raise TypeError('ProfileName must be a string')

        if (Lfreebound<0 or Lfreebound>1):
            raise ValueError('Lfreebound must be either 0 or 1')

        self.Lfreebound = Lfreebound
        self.ProfileName = ProfileName
        self.length = Nvol + Lfreebound
        self.Cumulative = Cumulative

        Optimizable.__init__(self)
        
        self.allocate()

        # Define fixed attribute
        self.fixed = np.full(len(self.get_dofs()), True)

    def allocate(self):
        myshape = (1,self.length)

        self.values = np.zeros(myshape)
        self.values = self.values[0]
        self.names = self.make_names(self.ProfileName)

    
    def make_names(self, prefix):
        """
        Form a list of names of the profile.
        """

        names = [prefix + '(' + str(n) + ')' for n in range(1, self.length+1)]
        return names


    def _validate_mn(self, ivol):
        """
        Check whether ivol is in the allowed range.
        """
        if ivol < 1:
            raise ValueError('ivol must be >= 1')
        if ivol > self.length:
            raise ValueError('ivol must be <= ', self.length)
    
    def get_value(self, ivol):
        """
        Return a particular value of profile in a specific volume.
        """
        self._validate_mn( ivol )
        return self.values[ivol]

    def set_value(self, ivol, val):
        """
        Set a particular value of profile in a specific volume.
        """
        self._validate_mn( ivol )
        self.values[ ivol ] = val
        self.recalculate = True
        self.recalculate_derivs = True    

    def get_dofs(self):
        """
        Return a 1D numpy array with all the degrees of freedom.
        """
        return self.values

    def set_dofs(self, v):
        """
        Set the shape coefficients from a 1D list/array
        """

        if len(v) != self.length:
            raise ValueError('Input vector should have ' + str(self.length) + \
                             ' elements but instead has ' + str(len(v)))
        
        # Check whether any elements actually change:
        if np.all(np.abs(self.get_dofs() - np.array(v)) == 0):
            logger.info('set_dofs called, but no dofs actually changed')
            return

        logger.info('set_dofs called, and at least one dof changed')
        self.recalculate = True
        self.recalculate_derivs = True

        if not self.Cumulative:
            self.values = v 
        else:
            tmp = self.values
            self.values[0] = v[0]
            for ii in range(1,self.length):
                if self.fixed[ii]:
                    dv = tmp[ii]-tmp[ii-1]
                    self.values[ii] = dv + self.values[ii-1]
                else:
                    self.values[ii] = v[ii]


    def fixed_range(self, vmin, vmax, fixed=True):
        """
        Set the 'fixed' property for a range of m and n values.

        All elements of the profile between volume vmin and vmax will have the 
        property fixed set to fixed
        """
        for ivol in range(vmin, vmax + 1):
            self.set_fixed(self.ProfileName, '({})'.format(ivol), fixed)