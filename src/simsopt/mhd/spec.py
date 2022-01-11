# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a class that handles the SPEC equilibrium code.
"""

import logging
from typing import Union
import os.path
import traceback
from shutil import copyfile

import numpy as np

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None 
    logger.warning(str(e))

# spec_found = True
try:
    import spec
except ImportError as e:
    spec = None
    logger.warning(str(e))
    # spec_found = False

# py_spec_found = True
try:
    import py_spec
except ImportError as e:
    py_spec = None
    logger.warning(str(e))
    # py_spec_found = False

# pyoculus_found = True
try:
    import pyoculus
except ImportError as e:
    pyoculus = None
    logger.warning(str(e))
    # pyoculus_found = False

from .SpecProfile import SpecProfile
from .NormalField import NormalField
from simsopt._core.optimizable import Optimizable
from simsopt._core.util import ObjectiveFailure
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.util.dev import SimsoptRequires
if MPI is not None:
    from simsopt.util.mpi import MpiPartition
else:
    MpiPartition = None
#from ..util.mpi import MpiPartition


@SimsoptRequires(MPI is not None, "mpi4py needs to be installed for running SPEC")
class Spec(Optimizable):
    """
    This class represents the SPEC equilibrium code.

    Philosophy regarding mpol and ntor: The Spec object keeps track of
    mpol and ntor values that are independent of those for the
    boundary Surface object. If the Surface object has different
    mpol/ntor values, the Surface's rbc/zbs arrays are first copied,
    then truncated or expanded to fit the mpol/ntor values of the Spec
    object to before Spec is run. Therefore, you may sometimes need to
    manually change the mpol and ntor values for the Spec object.

    The default behavior is that all  output files will be
    deleted except for the first and most recent iteration on worker
    group 0. If you wish to keep all the output files, you can set
    ``keep_all_files = True``. If you want to save the output files
    for a certain intermediate iteration, you can set the
    ``files_to_delete`` attribute to ``[]`` after that run of SPEC.

    Args:
        filename: SPEC input file to use for initialization. It should end
          in ``.sp``. Or, if None, default values will be used.
        mpi: A :obj:`simsopt.util.mpi.MpiPartition` instance, from which
          the worker groups will be used for SPEC calculations. If ``None``,
          each MPI process will run SPEC independently.
        verbose: Whether to print SPEC output to stdout.
        keep_all_files: If ``False``, all output files will be deleted
          except for the first and most recent ones from worker group 0. If 
          ``True``, all output files will be kept.
        tolerance: Maximum allowed valued for force error
    """

    def __init__(self,
                 filename: Union[str, None] = None,
                 mpi: Union[MpiPartition, None] = None,
                 verbose: bool = True,
                 keep_all_files: bool = False,
                 tolerance: float = 1e-12):

        #if not spec_found:
        if spec is None:
            raise RuntimeError(
                "Using Spec requires spec python wrapper to be installed.")

        #if not py_spec_found:
        if py_spec is None:
            raise RuntimeError(
                "Using Spec requires py_spec to be installed.")

        self.lib = spec
        # For the most commonly accessed fortran modules, provide a
        # shorthand so ".lib" is not needed:
        modules = [
            "inputlist",
            "allglobal",
        ]
        for key in modules:
            setattr(self, key, getattr(spec, key))

        self.verbose = verbose
        # mute screen output if necessary
        # TODO: relies on /dev/null being accessible (Windows!)
        if not self.verbose:
            self.lib.fileunits.mute(1)

        # python wrapper does not need to write files along the run
        #self.lib.allglobal.skip_write = True

        # If mpi is not specified, use a single worker group:
        if mpi is None:
            self.mpi = MpiPartition(ngroups=1)
        else:
            self.mpi = mpi
        # SPEC will use the "groups" communicator from the MpiPartition:
        self.lib.allglobal.set_mpi_comm(self.mpi.comm_groups.py2f())

        if filename is None:
            # Read default input file, which should be in the same
            # directory as this file:
            filename = os.path.join(os.path.dirname(__file__), 'defaults.sp')
            logger.info("Initializing a SPEC object from defaults in " \
                        + filename)
        else:
            if not filename.endswith('.sp'):
                filename = filename + '.sp'
            logger.info("Initializing a SPEC object from file: " + filename)

        # File to read initial guess from
        self.fname_initial_guess = filename

        if tolerance<=0:
            raise ValueError('Tolerance should be larger than zero!')

        self.tolerance = tolerance

        self.init(filename)
        self.extension = filename[:-3]
        self.keep_all_files = keep_all_files
        self.files_to_delete = []

        # Create a surface object for the boundary:
        si = spec.inputlist  # Shorthand
        stellsym = bool(si.istellsym)
        print("In __init__, si.istellsym=", si.istellsym, " stellsym=", stellsym)
        self.boundary = SurfaceRZFourier(nfp=si.nfp,
                                         stellsym=stellsym,
                                         mpol=si.mpol,
                                         ntor=si.ntor)

                                        

        self.Ivolume = SpecProfile( si.nvol, si.lfreebound, 'Ivolume')

        sivol = si.ivolume
        self.mvol =  (si.nvol + si.lfreebound)
        dsivol = [0] * self.mvol
        dsivol[0] = si.ivolume[0]
        dsivol[1:self.mvol] = [sivol[ii] - sivol[ii-1] for ii in range(1,self.mvol)]
        self.Ivolume.values = dsivol[0:self.mvol]

        # Transfer the boundary shape from fortran to the boundary
        # surface object:
        for m in range(si.mpol + 1):
            for n in range(-si.ntor, si.ntor + 1):
                self.boundary.rc[m, n + si.ntor] = si.rbc[n + si.mntor, m + si.mmpol]
                self.boundary.zs[m, n + si.ntor] = si.zbs[n + si.mntor, m + si.mmpol]
                if not stellsym:
                    self.boundary.rs[m, n + si.ntor] = si.rbs[n + si.mntor, m + si.mmpol]
                    self.boundary.zc[m, n + si.ntor] = si.zbc[n + si.mntor, m + si.mmpol]

        if( si.lfreebound==1 ):
            self.NormalField = NormalField( nfp=si.nfp,
                                            stellsym=stellsym,
                                            mpol=si.mpol,
                                            ntor=si.ntor)

            # Transfer the Vnc, Vns shape from fortran to the NormalField object:
            for m in range(si.mpol + 1):
                for n in range(-si.ntor, si.ntor + 1):
                    self.NormalField.vs[m, n + si.ntor] = si.vns[n + si.mntor, m + si.mmpol]
                    if not stellsym:
                        self.NormalField.vc[m, n + si.ntor] = si.vnc[n + si.mntor, m + si.mmpol]

            self.depends_on = ["boundary", "Ivolume", "NormalField"]
                        
        else:
            self.depends_on = ["boundary", "Ivolume"]


        self.need_to_run_code = True
        self.counter = -1
        self.step_counter = -1

        # By default, all dofs owned by SPEC directly, as opposed to
        # dofs owned by the boundary surface object, are fixed.
        self.fixed = np.full(len(self.get_dofs()), True)
        self.names = ['phiedge', 'curtor']

    def get_dofs(self):
        return np.array([self.inputlist.phiedge,
                         self.inputlist.curtor])

    def set_dofs(self, x):
        self.need_to_run_code = True
        self.inputlist.phiedge = x[0]
        self.inputlist.curtor = x[1]

    def init(self, filename: str):
        """
        Initialize SPEC fortran state from an input file.

        Args:
            filename: Name of the file to load. It should end in ``.sp``.
        """
        logger.debug("Entering init")
        if self.mpi.proc0_groups:
            spec.inputlist.initialize_inputs()
            logger.debug("Done with initialize_inputs")
            self.extension = filename[:-3]  # Remove the ".sp"
            spec.allglobal.ext = self.extension
            spec.allglobal.read_inputlists_from_file()
            logger.debug("Done with read_inputlists_from_file")
            spec.allglobal.check_inputs()

        logger.debug('About to call broadcast_inputs')
        spec.allglobal.broadcast_inputs()
        logger.debug('About to call preset')
        spec.preset()
        logger.debug("Done with init")

    def run(self, fd_bool=False):
        """
        Run SPEC, if needed.
        """
        if not self.need_to_run_code:
            logger.info("run() called but no need to re-run SPEC.")
            return
        logger.info("Preparing to run SPEC.")

        # Read last converged case for correct initial guess
        print("reading initial guess...")
        self.init( self.fname_initial_guess )

        self.counter += 1

        si = self.inputlist  # Shorthand

        # nfp must be consistent between the surface and SPEC. The surface's value trumps.
        si.nfp = self.boundary.nfp
        si.istellsym = int(self.boundary.stellsym)

        # Convert boundary to RZFourier if needed:
        boundary_RZFourier = self.boundary.to_RZFourier()

        # Transfer boundary data to fortran:
        si.rbc[:, :] = 0.0
        si.zbs[:, :] = 0.0
        si.rbs[:, :] = 0.0
        si.zbc[:, :] = 0.0
        mpol_capped = np.min([boundary_RZFourier.mpol, si.mmpol])
        ntor_capped = np.min([boundary_RZFourier.ntor, si.mntor])
        stellsym = bool(si.istellsym)
        print("In run, si.istellsym=", si.istellsym, " stellsym=", stellsym)
        for m in range(mpol_capped + 1):
            for n in range(-ntor_capped, ntor_capped + 1):
                si.rbc[n + si.mntor, m + si.mmpol] = boundary_RZFourier.get_rc(m, n)
                si.zbs[n + si.mntor, m + si.mmpol] = boundary_RZFourier.get_zs(m, n)
                if not stellsym:
                    si.rbs[n + si.mntor, m + si.mmpol] = boundary_RZFourier.get_rs(m, n)
                    si.zbc[n + si.mntor, m + si.mmpol] = boundary_RZFourier.get_zc(m, n)

        # Transfer Vns, Vnc from python to fortran

        if( si.lfreebound==1 ):
            mpol_capped = np.min([self.NormalField.mpol, si.mmpol])
            ntor_capped = np.min([self.NormalField.ntor, si.mntor])
            for m in range(mpol_capped + 1):
                for n in range(-ntor_capped, ntor_capped + 1):
                    si.vns[n + si.mntor, m + si.mmpol] = self.NormalField.get_vs(m, n)
                    if not stellsym:
                        si.vnc[n + si.mntor, m + si.mmpol] = self.NormalField.get_vc(m, n)


        # Set the coordinate axis using the lrzaxis=2 feature:
        si.lrzaxis = 2
        # lrzaxis=2 only seems to work if the axis is not already set
        si.ras[:] = 0.0
        si.rac[:] = 0.0
        si.zas[:] = 0.0
        si.zac[:] = 0.0

        #Set volume current
        divol = self.Ivolume.get_dofs()
        si.ivolume[0] = divol[0]
        for ii in range(1,self.mvol):
            si.ivolume[ii] = si.ivolume[ii-1] + divol[ii]

        si.curtor = si.ivolume[self.mvol-1] + np.sum(si.isurf)

        # Another possible way to initialize the coordinate axis: use
        # the m=0 modes of the boundary.
        # m = 0
        # for n in range(2):
        #     si.rac[n] = si.rbc[n + si.mntor, m + si.mmpol]
        #     si.zas[n] = si.zbs[n + si.mntor, m + si.mmpol]

        filename = self.extension + '_{:03}_{:06}'.format(self.mpi.group, self.counter)
        logger.info("Running SPEC using filename " + filename)
        self.allglobal.ext = filename
        try:
            # Here is where we actually run SPEC:
            if self.mpi.proc0_groups:
                logger.debug('About to call check_inputs')
                spec.allglobal.check_inputs()
            logger.debug('About to call broadcast_inputs')
            spec.allglobal.broadcast_inputs()
            logger.debug('About to call preset')
            spec.preset()
            logger.debug(f'About to call init_outfile')
            spec.sphdf5.init_outfile()
            logger.debug('About to call mirror_input_to_outfile')
            spec.sphdf5.mirror_input_to_outfile()
            if self.mpi.proc0_groups:
                logger.debug('About to call wrtend')
                spec.allglobal.wrtend()
            logger.debug('About to call init_convergence_output')
            spec.sphdf5.init_convergence_output()
            logger.debug(f'About to call spec')
            spec.spec()
            logger.debug('About to call diagnostics')
            spec.final_diagnostics()
            logger.debug('About to call write_grid')
            spec.sphdf5.write_grid()
            if self.mpi.proc0_groups:
                logger.debug('About to call wrtend')
                spec.allglobal.wrtend()
            logger.debug('About to call hdfint')
            spec.sphdf5.hdfint()
            logger.debug('About to call finish_outfile')
            spec.sphdf5.finish_outfile()
            logger.debug('About to call ending')
            spec.ending()

        except:
            if self.verbose:
                traceback.print_exc()
            raise ObjectiveFailure("SPEC did not run successfully.")

        logger.info("SPEC run complete.")
        # Barrier so workers do not try to read the .h5 file before it is finished:
        self.mpi.comm_groups.Barrier()

        try:
            self.results = py_spec.SPECout(filename + '.sp.h5')
        except:
            if self.verbose:
                traceback.print_exc()
            raise ObjectiveFailure("Unable to read results following SPEC execution")

        logger.info("Successfully loaded SPEC results.")

        if self.results.output.ForceErr > self.tolerance:
            raise ObjectiveFailure("SPEC didn't converge")

        if not fd_bool:
            self.step_counter += 1
            fname_step = 'step_tmp.sp'

            if  self.mpi.proc0_world:
                copyfile( filename+'.sp.end', fname_step )
                       
        # Will now read initial guess
        self.fname_initial_guess = fname_step

        self.need_to_run_code = False

        # Group leaders handle deletion of files:
        if self.mpi.proc0_groups:

            # If the worker group is not 0, delete all wout files, unless
            # keep_all_files is True:
            if (not self.keep_all_files) and (self.mpi.group > 0):
                os.remove(filename + '.sp.h5')
                os.remove(filename + '.sp.end')

            # Delete the previous output file, if desired:
            for file_to_delete in self.files_to_delete:
                os.remove(file_to_delete)
            self.files_to_delete = []

            # Record the latest output file to delete if we run again:
            if (self.mpi.group == 0) and (self.counter > 0) and (not self.keep_all_files):
                self.files_to_delete.append(filename + '.sp.h5')
                self.files_to_delete.append(filename + '.sp.end')

    def volume(self):
        """
        Return the volume inside the boundary flux surface.
        """
        self.run()
        return self.results.output.volume * self.results.input.physics.Nfp

    def iota(self):
        """
        Return the rotational transform in the middle of the volume.
        """
        self.run()
        return self.results.transform.fiota[1, 0]

class Residue(Optimizable):
    """
    Greene's residue, evaluated from a Spec equilibrum

    Args:
        spec: a Spec object
        pp, qq: Numerator and denominator for the resonant iota = pp / qq
        vol: Index of the Spec volume to consider
        theta: Spec's theta coordinate at the periodic field line
        s_guess: Guess for the value of Spec's s coordinate at the periodic
                field line
        s_min, s_max: bounds on s for the search
        rtol: the relative tolerance of the integrator
    """

    def __init__(self, spec, pp, qq, vol=1, theta=0, s_guess=None, s_min=-1.0,
                 s_max=1.0, rtol=1e-9):
        # if not spec_found:
        if spec is None:
            raise RuntimeError(
                "Residue requires py_spec package to be installed.")
        # if not pyoculus_found:
        if pyoculus is None:
            raise RuntimeError(
                "Residue requires pyoculus package to be installed.")

        self.spec = spec
        self.pp = pp
        self.qq = qq
        self.vol = vol
        self.theta = theta
        self.rtol = rtol
        if s_guess is None:
            self.s_guess = 0.0
        else:
            self.s_guess = s_guess
        self.s_min = s_min
        self.s_max = s_max
        self.depends_on = ['spec']
        self.need_to_run_code = True
        self.fixed_point = None
        # We may at some point want to allow Residue to use a
        # different MpiPartition than the Spec object it is attached
        # to, but for now we'll use the same MpiPartition for
        # simplicity.
        self.mpi = spec.mpi

    def J(self, fd_bool=False):
        """
        Run Spec if needed, find the periodic field line, and return the residue
        """
        if self.need_to_run_code:
            self.need_to_run_code = False
            self.spec.run(fd_bool)
            if self.mpi.proc0_groups:
                # Only the group leader actually computes the residue.
                specb = pyoculus.problems.SPECBfield(self.spec.results, self.vol)
                # Set nrestart=0 because otherwise the random guesses in
                # pyoculus can cause examples/tests to be
                # non-reproducible.
                fp = pyoculus.solvers.FixedPoint(specb, {'theta': self.theta, 'nrestart': 0},
                                                 integrator_params={'rtol': self.rtol})
                self.fixed_point = fp.compute(self.s_guess,
                                              sbegin=self.s_min,
                                              send=self.s_max,
                                              pp=self.pp, qq=self.qq)
            # Broadcast, so all procs would raise ObjectiveFailure together:
            self.fixed_point = self.mpi.comm_groups.bcast(self.fixed_point, root=0)

        if self.fixed_point is None:
            raise ObjectiveFailure("Residue calculation failed")

        return self.fixed_point.GreenesResidue

    def get_dofs(self):
        return np.array([])

    def set_dofs(self, x):
        self.need_to_run_code = True

class ChaosVolume(Optimizable):
    """ChaosVolume allows to determine a certain chaos volume by separating the chaotic trajectories from the non-chaotic ones using the fractal 
    dimension of the magnetic field lines.

        - It is not expected to have large scale differences between x and y, 
        therefore the coordinates of each trajectory are scaled into a square 
        [0, 1]x[0, 1] when computing the dimension. Point-like trajectories (0-dimensional) should still be detected as 0-dimensional.
        - The box-count computing in __compute_LN_boxcount is the lengthiest part
    """

    def __init__(self, spec, critical_dim, polyfit_deg=6, kmin=1, kmax=11, base_reduction_factor=2, nppts=-1, boxcount_method=False, write_chaos_vol_to_file=False):
        """Sets up the chaos volume getter.

        Args:
            spec: SPEC obect to analyse
            critical_dim: critical dimension used to discriminate between chaotic and non-chaotic trajectories
            polyfit_deg: degree of the polynomial fitted to approximate the fractal dimension
            kmin: minimal exponent used to reduce the size of the boxes in box-count process
            kmax: maximal exponent used to reduce the size of the boxes in box-count process
            base_reduction_factor: base of the exponential factor by which the box sizes are reduced
            nppts: number of points per line used for each field line. Defaults to the total number of points per line present in the input list
            nptrj: number of trajectories used. Defaults to the total number of trajectories present in the input list 
        """
        # if not spec_found:
        if spec is None:
            raise RuntimeError(
                "Residue requires py_spec package to be installed.")
        
        self.spec = spec
        self.need_to_run_code = True
        self.boxcount_method = boxcount_method
        self.write_chaos_vol_to_file = write_chaos_vol_to_file
        self.first_time_writing = True
        # initialize parameters using the spec object
        # chooses default toroidal plane to analyse at index 0
        self.toroidal_plane = 0
        if nppts == -1:
            self.nppts = int(spec.inputlist.nppts)
        else:
            self.nppts = int(nppts)
        # beware there is also a trajectory between each volume
        self.nptrj = None

        # load only necessary data, with the correct format
        self.x = None
        self.y = None
        # do not load the x-y coordinates now

        # flag for successful fractal dimension and chaos volume computation
        self.fractal_dim_successful = False
        self.chaos_vol_successful = False

        # initialize parameters
        self.polyfit_deg = int(polyfit_deg)
        self.kmin = int(kmin)
        self.kmax = int(kmax)
        if self.kmin >= self.kmax:
            raise ValueError("Please take kmin < kmax")
        self.base_reduction_factor = np.float64(base_reduction_factor)
        self.critical_dim = np.float64(critical_dim)

        # initialize the fractal dimension array
        self.fractal_dim = None
        # initialize chaos volume
        self.chaos_vol = np.float64(0)

        # We may at some point want to allow ChaosVolume to use a
        # different MpiPartition than the Spec object it is attached
        # to, but for now we'll use the same MpiPartition for
        # simplicity.
        self.mpi = spec.mpi

        # State dependencies
        self.depends_on = ['spec']

    def is_fractal_dim_successful(self):
        """Returns True if the computation of the fractal dimensions has been successfully completed.

        returns fractal_dim_successful: True if the fractal dimension computation is successfully completed, False otherwise
        """
        return self.fractal_dim_successful

    def is_chaos_vol_successful(self):
        """! Returns True if the computation of the chaos volume has been successfully completed.

        returns chaos_vol_successful: True if the chaos volume computation is successfully completed, False otherwise
        """
        return self.chaos_vol_successful

    def set_critical_dim(self, critical_dim):
        """ Modifies critical_dim, after this one needs to compute the chaos volume again
        """
        self.chaos_vol_successful = False
        self.critical_dim = np.float64(critical_dim)

    def __load_xy(self, filename):
        """Loads the necessary x-y coordinates 
        """
        output = py_spec.SPECout(filename)

        self.nptrj = output.poincare.R.shape[0]

        Igeometry = self.spec.inputlist.igeometry
        # load only necessary data, with the correct format
        if Igeometry == 3:
            self.x = np.array(output.poincare.R[:, :, self.toroidal_plane])
            self.y = np.array(output.poincare.Z[:, :, self.toroidal_plane])
        elif Igeometry == 1:
            self.x = np.array(np.mod(output.poincare.t[:, :, self.toroidal_plane], np.pi * 2))
            self.y = np.array(output.poincare.R[:, :, self.toroidal_plane])
        elif Igeometry == 2:
            self.x = np.array(output.poincare.R[:, :, self.toroidal_plane] * np.cos(
                output.poincare.t[:, :, self.toroidal_plane])
            )
            self.y = np.array(output.poincare.R[:, :, self.toroidal_plane] * np.sin(
                output.poincare.t[:, :, self.toroidal_plane])
            )
        else:
            raise ValueError("Unsupported geometry")

    def __compute_LN_boxcount(self, traj_nbr=0):
        """Computes the number of boxes N and the lengths of the boxes (compared to the maximal length of the line) L parameters necessary for fractal dimension computation using box-counting.

        Args:
            traj_nbr: index of the line for which N and L are computed, between 0 and nptrj

        returns N, number of boxes containing a point of the curve
        returns L, reduction factor of the boxes (a proportional factor does not matter on a logarithmic scale, since it transforms in a constant)
        """
        if traj_nbr >= self.nptrj:
            raise Exception("Trajectory number is too high, must be strictly below {}".format(self.nptrj))
        else:
            traj_nbr = np.int64(traj_nbr)

        x_min = np.min(self.x[traj_nbr, :])
        y_min = np.min(self.y[traj_nbr, :])

        Lmax_x = np.max(self.x[traj_nbr, :]) - x_min
        Lmax_y = np.max(self.y[traj_nbr, :]) - y_min

        if Lmax_x < 1e-8:
            # point-like trajectories are possible
            Lmax_x = 1

        if Lmax_y < 1e-8:
            # point-like trajectories are possible
            Lmax_y = 1

        ks = np.linspace(self.kmin, self.kmax, self.kmax - self.kmin + 1)
        N = np.zeros(self.kmax - self.kmin + 1)
        L = 1 / self.base_reduction_factor ** ks
        s = np.int64(np.ceil(self.base_reduction_factor ** ks))

        for k in range(self.kmin, self.kmax + 1):
            # compute outside the for mm loop for much better performances
            # x-y are normalized on a [0, 1]x[0, 1] square
            # the x-y data has already been correctly imported in __load_xy()
            ii = np.int64(np.floor((self.x[traj_nbr, :] - x_min) / (Lmax_x * L[k-self.kmin])))
            jj = np.int64(np.floor((self.y[traj_nbr, :] - y_min) / (Lmax_y * L[k-self.kmin])))

            # uses the unicity of elements in sets to vectorize the counter
            # ii has already length of nppts
            try:
                counter = {(ii[mm],jj[mm]) for mm in range(len(ii))}
            except:
                raise Exception("Problem in counter for compute_LN_boxcount.")
            N[k-self.kmin] = len(counter)

        return L, N

    def __compute_fractal_dim(self, L, N, remove_plateau=False):
        """Computes the fractal dimension using a polynomial fit on the N, L parameters resulting from __compute_LN_boxcount().

        Args:
            N: number of boxes containing a point of the curve
            L: lengths of the boxes (normalized to the maximal length of the curve)
            remove_plateau: if True removes the saturation plateau before fitting N and L. If set to True, consider using a lower polynomial degree (4 or 5 for instance) to avoid overfitting the data

        returns fract_dim: fractal dimension corresponding to N, L
        """
        plateau_threshold = 0.98 * self.nppts
        if remove_plateau:
            # the idea is to remove the saturation plateau first
            # the saturation may occur when the boxsizes are too thin and pick each point separately
            # in such cases the last values of log(N) saturates near log(nppts)
            # only remove the plateau if there are at least 3 good points
            # (important for point-like trajectories, to have at least some points)
            if N[2] < plateau_threshold:
                L = L[N <= plateau_threshold]
                # we have to mask L before masking N
                N = N[N <= plateau_threshold]

        # fit the data
        p = np.polyfit(-np.log(L), np.log(N), self.polyfit_deg)
        dp = np.polyder(p)
        d2p = np.polyder(dp)

        # 200 logarithmically spaced values between 1/L[0] and 1/L[-1]
        epsilon = np.logspace(-np.log10(L[0]), -np.log10(L[-1]), num=200)

        # compute the curvature
        # don't need to take into account the variations in N direction
        # since we consider linearly spaced points (on a log scale)
        curvature = np.abs(np.polyval(d2p, np.log(epsilon)))/(1+np.polyval(dp, np.log(epsilon)) ** 2) ** (3/2)

        # curvature threshold (ok, but kind of arbitrary for the moment)
        threshold_fit = max(max(curvature)/10, 0.15)

        # find all zones (min, max indices) with curvature below threshold
        idx_max_fit = np.array([ii for ii, value in enumerate(curvature[0:-1]) if value < threshold_fit and curvature[ii+1] >= threshold_fit])
        idx_min_fit = np.array([ii for ii, value in enumerate(curvature[1:]) if value < threshold_fit and curvature[ii] >= threshold_fit])

        if curvature[-1] < threshold_fit:
            # check to avoid empty numpy arrays
            if np.any(idx_max_fit):
                idx_max_fit = np.concatenate((idx_max_fit, [len(curvature)-1]))
            else:
                idx_max_fit = np.array([len(curvature)-1])

        if curvature[0] < threshold_fit:
            # check to avoid empty numpy arrays
            if np.any(idx_min_fit):
                idx_min_fit = np.concatenate(([0], idx_min_fit))
            else:
                idx_min_fit = np.array([0])

        # if multiple zones, choose the largest which is not a plateau
        if len(idx_min_fit) > 1:
            # if the plateau has not already been removed, it is removed now
            if not remove_plateau:
                idx_min_plateau = [ii for ii, value in enumerate(np.exp(np.polyval(p, np.log(epsilon)))) if value >= plateau_threshold]
                # check if idx_min_plateau is not empty
                if not idx_min_plateau:
                    pass
                else:
                    idx_min_plateau = min(idx_min_plateau)
                    while idx_max_fit[-1] >= idx_min_plateau and len(idx_max_fit) > 1:
                        idx_max_fit = idx_max_fit[:-1]
                        idx_min_fit = idx_min_fit[:-1]

            # chooses the indices covering the largest zone (plateau has been removed at this point)
            max_diff = max(idx_max_fit-idx_min_fit)
            # in case of multiple zones with same differences, the zone with the lowest index is chosen
            idx_max_diff = min([ii for ii, value in enumerate(idx_max_fit-idx_min_fit) if value == max_diff])

            # indices corresponding to the linear zone
            idx_max_fit = idx_max_fit[idx_max_diff].item()
            idx_min_fit = idx_min_fit[idx_max_diff].item()
        elif len(idx_min_fit) == 1:
            idx_max_fit = idx_max_fit.item()
            idx_min_fit = idx_min_fit.item()
        else:
            raise ValueError("No minima for the curvature has been found")

        # compute the dimension
        fractal_dim = np.mean(np.polyval(dp, np.log(epsilon[idx_min_fit:idx_max_fit])))

        return fractal_dim

    def get_fractal_dim(self, force_compute_fractal_dim=True):
        """Computes the fractal dimensions for all trajectories in x, y. If is_fractal_dim_successful() is False or force_compute_fractal_dim is True, computes the fractal dimensions before returning it. Otherwise, simply returns the fractal dimensions.

        Args:
            force_compute_fractal_dim: if True forces to compute the fractal dimensions

        returns fractal_dim, fractal dimensions of the trajectories
        """
        if self.need_to_run_code:
            # use correct nbr pts per line before running SPEC
            self.spec.inputlist.nppts = self.nppts
            self.spec.run()
            # since the trajectories changed, the fractal dimensions need to be computed again
            force_compute_fractal_dim = True

        filename = self.spec.extension + '_{:03}_{:06}.sp.h5'.format(self.spec.mpi.group, self.spec.counter)

        self.__load_xy(filename)
        self.fractal_dim = np.zeros(self.nptrj)

        if self.fractal_dim_successful is False or force_compute_fractal_dim is True:
            for line in range(self.nptrj):
                L, N = self.__compute_LN_boxcount(line)
                self.fractal_dim[line] = self.__compute_fractal_dim(L, N, remove_plateau=False)

        # print(self.fractal_dim)
        self.fractal_dim_successful = True
        return self.fractal_dim

    def J(self, force_compute_chaos_vol=True, force_compute_fractal_dim=True):
        """Computes the chaos volume in x, y. If is_chaos_vol_successful() is False or force_compute_chaos_vol is True, computes the chaos volume before returning it. Otherwise, simply returns the chaos volume.
        Run Spec if needed.

        Args:
            force_compute_chaos_vol: if True forces to compute the chaos volume
            force_compute_fractal_dim: if True forces to compute the fractal dimensions

        returns chaos_vol: chaos volume of the trajectories
        """
        fractal_dim = self.get_fractal_dim(force_compute_fractal_dim)

        if self.chaos_vol_successful is False or force_compute_chaos_vol is True:
            if not self.boxcount_method:
                # using sigmoid function to smooth the chaos volume (smoother version of Heaviside function in a sense)
                def smooth_heaviside(x, k=20, D_crit=self.critical_dim, D_0=1.1):
                    #
                    # k: slope is k/4
                    # 
                    #
                    # pure sigmoid:
                    # return 1/(1+np.exp(-k*(x-D_crit)))
                    y = np.zeros_like(x)

                    # quadratic interpolation between D_0 and D_crit
                    # b = k / 2 * (D_crit - D_0)
                    # a = (D_crit-D_0)**(-b)/2
                    # for ii, z in enumerate(x):
                    #     if z >= D_0 and z < D_crit:
                    #         y[ii] = a*(z-D_0)**b
                    #     elif z >= D_crit:
                    #         y[ii] = 1/(1+np.exp(-k*(z-D_crit)))

                    # sinh interpolation between D_0 and D_crit
                    b = k / 4 * np.sinh(2*(D_crit - D_0))
                    a = (np.sinh(D_crit-D_0))**(-b)/2
                    for ii, z in enumerate(x):
                        if z >= D_0 and z < D_crit:
                            y[ii] = a*(np.sinh(z-D_0))**b
                        elif z >= D_crit:
                            y[ii] = 1/(1+np.exp(-k*(z-D_crit)))

                    return y

                #TODO: MAKE IT PROPORTIONAL TO DTFLUX
                traj_chaos_measure = 1 / self.nptrj
                self.chaos_vol = traj_chaos_measure * sum(smooth_heaviside(fractal_dim))

            else:
                # using a sort of box-counting method to evaluate the chaos volume
                # 1. ------------- total volume -------------
                # find the index of the boundary field line
                idx_x_max = list(set(np.where(self.x == np.max(self.x))[0]))
                idx_x_min = list(set(np.where(self.x == np.min(self.x))[0]))
                idx_y_max = list(set(np.where(self.y == np.max(self.y))[0]))
                idx_y_min = list(set(np.where(self.y == np.min(self.y))[0]))

                if len(idx_x_max) > 1 or len(idx_x_min) > 1 or len(idx_y_max) > 1 or len(idx_y_min) > 1:
                    raise RuntimeError("There are multiple trajectories near the boundary.")
                elif idx_x_max[0] != idx_x_min[0] or idx_x_max[0] != idx_y_max[0] or idx_x_max[0] != idx_y_min[0] or idx_x_min[0] != idx_y_max[0] or idx_x_min[0] != idx_y_min[0] or idx_y_max[0] != idx_y_min[0]:
                    raise RuntimeError("The boundary is not the same trajectory all along.")

                idx_boundary_traj = idx_x_max[0]

                # TODO USE SPEC MAGNETIC AXIS ?
                # computes the angle w.r.t. the mean of the boundary trajectory
                x_c = np.mean(self.x[idx_boundary_traj, :])
                y_c = np.mean(self.y[idx_boundary_traj, :])

                # convert array to dictionary for sorting
                angles = dict(enumerate(np.arctan2(self.y[idx_boundary_traj, :] - y_c, self.x[idx_boundary_traj, :] - x_c)))
                sorted_idx = [*dict(sorted(angles.items(), key=lambda x:x[1]))]

                a1, a2 = 0, 0
                #TODO USE SPEC VOLUME OUTPUT? DIVIDE BY 2PI?
                # shoelace area for a polygon
                for ii in range(self.nppts-1):
                    a1 += (self.x[idx_boundary_traj, sorted_idx[ii]] - x_c) * (self.y[idx_boundary_traj, sorted_idx[ii+1]] - x_c)
                    a2 += (self.y[idx_boundary_traj, sorted_idx[ii]] - x_c) * (self.x[idx_boundary_traj, sorted_idx[ii+1]] - x_c)
                a1 += (self.x[idx_boundary_traj, sorted_idx[-1]] - x_c) * (self.y[idx_boundary_traj, sorted_idx[0]]-y_c)
                a2 += (self.y[idx_boundary_traj, sorted_idx[-1]] - y_c) * (self.x[idx_boundary_traj, sorted_idx[0]]-x_c)
                # total volume occupied by the Poincaré section
                V_tot = abs(a1 - a2) / 2

                # 2. ------------- chaos volume -------------
                # load chaotic trajectories
                nbr_chaotic_traj = sum(fractal_dim >= self.critical_dim)
                if nbr_chaotic_traj > 0:
                    # Store all  chaotic points in a single array
                    x = np.zeros(nbr_chaotic_traj * self.nppts)
                    y = np.zeros(nbr_chaotic_traj * self.nppts)
                    traj_chaotic_nbr = 0
                    for traj_nbr, dim in enumerate(fractal_dim):
                        # fiters the chaotic field lines
                        if dim >= self.critical_dim:
                            x[traj_chaotic_nbr*self.nppts:(traj_chaotic_nbr+1)*self.nppts] = self.x[traj_nbr, :]
                            y[traj_chaotic_nbr*self.nppts:(traj_chaotic_nbr+1)*self.nppts] = self.y[traj_nbr, :]
                            traj_chaotic_nbr += 1

                    # determine min, max to use for all chaotic points
                    x_min = np.min(x)
                    y_min = np.min(y)

                    # size of the boxes
                    #TODO: USE BOX SIZE RIGHT BEFORE BOX COUNTING SATURATION - see MATLAB get_fractal_dim
                    z = 16 * 2 ** (np.log10(self.nppts))
                    L_x = (np.max(x) - x_min) / z
                    L_y = (np.max(y) - y_min) / z

                    # x-y indices
                    ii = np.int64(np.floor((x - x_min) / L_x))
                    jj = np.int64(np.floor((y - y_min) / L_y))
                    try:
                        #Use sets to avoid duplicates
                        counter = {(ii[mm],jj[mm]) for mm in range(len(ii))}
                    except:
                        raise Exception("Problem in counter for get_chaos_vol with method 3.")

                    N = len(counter)
                    V_chaos = N * L_x * L_y
                else:
                    V_chaos = 0

                self.chaos_vol = V_chaos / V_tot

        self.need_to_run_code = False
        self.chaos_vol_successful = True

        if self.write_chaos_vol_to_file:
            self.print_to_file()

        return self.chaos_vol

    def print_to_file(self):
        """Prints the current state of the ChaosVolume instance to file
        """
        filename = self.spec.extension + '_Vchaos.csv'
        max_rc_zs = min(self.spec.inputlist.ntor, 6)
        if self.first_time_writing:
            with open(filename, 'w+') as file:
                file.write('Vchaos,')
                for m in range(max_rc_zs+1):
                    for n in range(max_rc_zs+1):
                        file.write('rc({},{}),'.format(m, n))
                for m in range(max_rc_zs+1):
                    for n in range(max_rc_zs+1):
                        file.write('zs({},{}),'.format(m, n))

                file.write('Dcrit,nptrj,nppts,polyfit_degree,base_reduc,kmin,kmax')
                file.write('\n')
                self.first_time_writing = False
        
        if self.chaos_vol_successful:
            with open(filename, 'a+') as file:
                file.write('{},'.format(self.chaos_vol))
                for m in range(max_rc_zs+1):
                    for n in range(max_rc_zs+1):
                        file.write('{},'.format(self.spec.boundary.get_rc(m,n)))
                for m in range(max_rc_zs+1):
                    for n in range(max_rc_zs+1):
                        file.write('{},'.format(self.spec.boundary.get_zs(m,n)))

                file.write('{},{},{},{},{},{},{}'.format(self.critical_dim, self.nptrj, self.nppts, self.polyfit_deg, self.base_reduction_factor, self.kmin, self.kmax))
                file.write('\n')

    def print_current_state(self):
        """Prints the current state of the ChaosVolume instance
        """
        print("Successful fractal dimension computation : {}".format(self.fractal_dim_successful))
        print("Successful chaos volume computation : {}".format(self.chaos_vol_successful))
        if self.chaos_vol_successful:
            print("Chaos volume found : {:%} of the total volume".format(self.chaos_vol))
        print("Critical dimension : {}".format(self.critical_dim))
        print("Nbr of trajectories : {}".format(self.nptrj))
        print("Nbr of pts per line in x,y : {}".format(self.nppts))
        print("Degree of the polynomial fitted : {}".format(self.polyfit_deg))
        print("Base of the reduction factor : {}".format(self.base_reduction_factor))
        print("Maximal exponent used for boxsizes : {}".format(self.kmax))
        print("Minimal exponent used for boxsizes : {}".format(self.kmin))

    def get_dofs(self):
        return np.array([])

    def set_dofs(self, x):
        self.need_to_run_code = True
