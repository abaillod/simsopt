from dataclasses import dataclass

import numpy as np
from sympy import Symbol, lambdify, exp

from .._core.json import GSONable
from .._core.util import RealArray

import simsoptpp as sopp
from simsopt.geo.curve import Curve
import numpy as np
from randomgen import PCG64

__all__ = ['GaussianSampler', 'PerturbationSample', 'CurvePerturbed',
           'perturb_coil_curve', 'perturb_coil_curve_localized',
           'ShapeGradientPerturbationSample', 'perturb_coil_curve_aligned']

@dataclass
class GaussianSampler(GSONable):
    r"""
    Generate a periodic gaussian process on the interval [0, 1] on a given list of quadrature points.
    The process has standard deviation ``sigma`` a correlation length scale ``length_scale``.
    Large values of ``length_scale`` correspond to smooth processes, small values result in highly oscillatory
    functions.
    Also has the ability to sample the derivatives of the function.

    We consider the kernel

    .. math::

        \kappa(d) = \sigma^2 \exp(-d^2/l^2)

    and then consider a Gaussian process with covariance

    .. math::

        Cov(X(s), X(t)) = \sum_{i=-\infty}^\infty \sigma^2 \exp(-(s-t+i)^2/l^2)

    the sum is used to make the kernel periodic and in practice the infinite sum is truncated.

    Args:
        points: the quadrature points along which the perturbation should be computed.
        sigma: standard deviation of the underlying gaussian process
               (measure for the magnitude of the perturbation).
        length_scale: length scale of the underlying gaussian process
                      (measure for the smoothness of the perturbation).
        n_derivs: number of derivatives of the gaussian process to sample.
    """

    points: RealArray
    sigma: float
    length_scale: float
    n_derivs: int = 1

    def __post_init__(self):
        xs = self.points
        n = len(xs)
        cov_mat = np.zeros((n*(self.n_derivs+1), n*(self.n_derivs+1)))

        def kernel(x, y):
            return sum((self.sigma**2)*exp(-(x-y+i)**2/(self.length_scale**2)) for i in range(-5, 6))

        XX, YY = np.meshgrid(xs, xs, indexing='ij')
        x = Symbol("x")
        y = Symbol("y")
        f = kernel(x, y)
        for ii in range(self.n_derivs+1):
            for jj in range(self.n_derivs+1):
                if ii + jj == 0:
                    lam = lambdify((x, y), f, "numpy")
                else:
                    lam = lambdify((x, y), f.diff(*(ii * [x] + jj * [y])), "numpy")
                cov_mat[(ii*n):((ii+1)*n), (jj*n):((jj+1)*n)] = lam(XX, YY)

        # we need to compute the sqrt of the covariance matrix. we used to do this using scipy.linalg.sqrtm,
        # but it seems sometime between scipy 1.11.1 and 1.11.2 that function broke/changed behaviour.
        # So we use a LDLT decomposition instead. See als https://github.com/hiddenSymmetries/simsopt/issues/349
        # from scipy.linalg import sqrtm, ldl
        # self.L = np.real(sqrtm(cov_mat))
        from scipy.linalg import ldl
        lu, d, _ = ldl(cov_mat)
        self.L = lu @ np.sqrt(np.maximum(d, 0))

    def draw_sample(self, randomgen=None):
        """
        Returns a list of ``n_derivs+1`` arrays of size ``(len(points), 3)``, containing the
        perturbation and the derivatives.
        """
        n = len(self.points)
        n_derivs = self.n_derivs
        if randomgen is None:
            randomgen = np.random
        z = randomgen.standard_normal(size=(n*(n_derivs+1), 3))
        curve_and_derivs = self.L@z
        return [curve_and_derivs[(i*n):((i+1)*n), :] for i in range(n_derivs+1)]


class PerturbationSample(GSONable):
    """
    This class represents a single sample of a perturbation.  The point of
    having a dedicated class for this is so that we can apply the same
    perturbation to multipe curves (e.g. in the case of multifilament
    approximations to finite build coils).
    The main way to interact with this class is via the overloaded ``__getitem__``
    (i.e. ``[ ]`` indexing).
    For example::

        sample = PerturbationSample(...)
        g = sample[0] # get the values of the perturbation
        gd = sample[1] # get the first derivative of the perturbation
    """

    def __init__(self, sampler, randomgen=None, sample=None):
        self.sampler = sampler
        self.randomgen = randomgen   # If not None, most likely fail with serialization
        if sample:
            self._sample = sample
        else:
            self.resample()

    def resample(self):
        self._sample = self.sampler.draw_sample(self.randomgen)

    def __getitem__(self, deriv):
        """
        Get the perturbation (if ``deriv=0``) or its ``deriv``-th derivative.
        """
        assert isinstance(deriv, int)
        if deriv >= len(self._sample):
            raise ValueError("""
The sample on has {len(self._sample)-1} derivatives.
Adjust the `n_derivs` parameter of the sampler to access higher derivatives.
""")
        return self._sample[deriv]


class CurvePerturbed(sopp.Curve, Curve):

    """A perturbed curve."""

    def __init__(self, curve, sample):
        r"""
        Perturb a underlying :mod:`simsopt.geo.curve.Curve` object by drawing a perturbation from a
        :obj:`GaussianSampler`.

        Comment:
        Doing anything involving randomness in a reproducible way requires care.
        Even more so, when doing things in parallel.
        Let's say we have a list of :mod:`simsopt.geo.curve.Curve` objects ``curves`` that represent a stellarator,
        and now we want to consider ``N`` perturbed stellarators. Let's also say we have multiple MPI ranks.
        To avoid the same thing happening on the different MPI ranks, we could pick a different seed on each rank.
        However, then we get different results depending on the number of MPI ranks that we run on. Not ideal.
        Instead, we should pick a new seed for each :math:`1\le i\le N`. e.g.

        .. code-block:: python

            from randomgen import SeedSequence, PCG64
            import numpy as np
            curves = ...
            sigma = 0.01
            length_scale = 0.2
            sampler = GaussianSampler(curves[0].quadpoints, sigma, length_scale, n_derivs=1)
            globalseed = 1
            N = 10 # number of perturbed stellarators
            seeds = SeedSequence(globalseed).spawn(N)
            idx_start, idx_end = split_range_between_mpi_rank(N) # e.g. [0, 5) on rank 0, [5, 10) on rank 1
            perturbed_curves = [] # this will be a List[List[Curve]], with perturbed_curves[i] containing the perturbed curves for the i-th stellarator
            for i in range(idx_start, idx_end):
                rg = np.random.Generator(PCG64(seeds_sys[j], inc=0))
                stell = []
                for c in curves:
                    pert = PerturbationSample(sampler_systematic, randomgen=rg)
                    stell.append(CurvePerturbed(c, pert))
                perturbed_curves.append(stell)
        """
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, depends_on=[curve])
        self.sample = sample

    def resample(self):
        self.sample.resample()
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        gamma[:] = self.curve.gamma() + self.sample[0]

    def gammadash_impl(self, gammadash):
        gammadash[:] = self.curve.gammadash() + self.sample[1]

    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:] = self.curve.gammadashdash() + self.sample[2]

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:] = self.curve.gammadashdashdash() + self.sample[3]

    def dgamma_by_dcoeff_vjp(self, v):
        return self.curve.dgamma_by_dcoeff_vjp(v)

    def dgammadash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadash_by_dcoeff_vjp(v)

    def dgammadashdash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadashdash_by_dcoeff_vjp(v)

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadashdashdash_by_dcoeff_vjp(v)


def perturb_coil_curve(original_curve, seed, sigma=0.01, length_scale=0.1):
    rg = np.random.Generator(PCG64(seed, inc=0))
    sampler = GaussianSampler(original_curve.quadpoints, sigma, length_scale, n_derivs=1)
    perturbation_sample = PerturbationSample(sampler, randomgen=rg)
    perturbed_curve = CurvePerturbed(original_curve, perturbation_sample)
    return perturbed_curve


import sympy
from math import comb

def _build_window_and_derivs(points, amplitude, s0, sigma_local, n_derivs):
    """
    Returns a list of length (n_derivs+1), [w^(0)(s), w^(1)(s), ..., w^(n_derivs)(s)]
    where w(s) = amplitude * exp(-0.5*((s - s0)/sigma_local)^2).

    Each w^(d)(s) is shape (n_points,).
    We'll do symbolic differentiation so that the product rule is consistent.
    """
    s = sympy.Symbol("s", real=True)
    w_sym = amplitude * sympy.exp(-0.5 * ((s - s0)/sigma_local)**2)

    w_syms = [w_sym.diff(s, i) for i in range(n_derivs+1)]
    w_funcs = [sympy.lambdify(s, expr, "numpy") for expr in w_syms]

    out = []
    for wf in w_funcs:
        out.append(wf(points))  # shape (len(points),)
    return out


class LocalizedPerturbationSample(PerturbationSample):
    """
    A specialized PerturbationSample that multiplies the base random field
    by a local Gaussian window, including the product rule for derivatives.
    """

    def __init__(self, sampler, randomgen=None, sample=None,
                 s0=0.5, sigma_local=0.01, amplitude=1.0):
        """
        :param sampler: The base GaussianSampler (global random field).
        :param s0: center of the local bump
        :param sigma_local: width of the local Gaussian in parameter space
        :param amplitude: optional scale factor for the bump
        """
        super().__init__(sampler, randomgen, sample)
        self.s0 = s0
        self.sigma_local = sigma_local
        self.amplitude = amplitude

        # Precompute w(s), w'(s), etc. up to sampler.n_derivs
        self._w_derivs = _build_window_and_derivs(
            points=self.sampler.points,
            amplitude=self.amplitude,
            s0=self.s0,
            sigma_local=self.sigma_local,
            n_derivs=self.sampler.n_derivs
        )

    def __getitem__(self, d):
        """
        Return the d-th derivative of [w(s)*g(s)] by the product rule:
          f(s) = w(s)*g(s)
          f^(d)(s) = sum_{k=0..d} comb(d,k) * w^(k)(s)* g^(d-k)(s)
        """
        if d >= len(self._sample):
            raise ValueError(f"Requested derivative {d}, but only have up to {len(self._sample)-1}.")

        out = np.zeros_like(self._sample[0])  # shape (npoints, 3)
        for k in range(d+1):
            w_k = self._w_derivs[k]            # shape (npoints,)
            g_dk = self._sample[d-k]          # shape (npoints, 3)
            c = comb(d, k)
            out += c*(w_k[:, None]*g_dk)
        return out


def perturb_coil_curve_localized(
    original_curve,
    seed=123,
    s0=0.5,
    sigma_local=0.01,
    sigma=0.01,
    length_scale=0.01,
    amplitude=1.0,
    n_derivs=1
):
    """
    Construct a localized random perturbation on `original_curve` by
    multiplying the global random field by a local Gaussian envelope
    around s0 (width sigma_local, scale amplitude).

    1) Make a GaussianSampler
    2) LocalizedPerturbationSample (product rule)
    3) CurvePerturbed
    """
    rg = np.random.Generator(PCG64(seed, inc=0))

    sampler = GaussianSampler(
        points=original_curve.quadpoints,
        sigma=sigma,
        length_scale=length_scale,
        n_derivs=n_derivs
    )

    localized_sample = LocalizedPerturbationSample(
        sampler=sampler,
        randomgen=rg,
        s0=s0,
        sigma_local=sigma_local,
        amplitude=amplitude
    )

    perturbed = CurvePerturbed(original_curve, localized_sample)
    return perturbed



class ShapeGradientPerturbationSample(PerturbationSample):
    """
    A specialized PerturbationSample that uses the shape gradient direction
    instead of random sampling, multiplied by a local Gaussian window.
    """
    def __init__(self, points, shape_gradient, s0=0.5, sigma_local=0.01, amplitude=1.0, n_derivs=1):
        """
        Parameters:
        -----------
        points : array-like
            The quadrature points along the curve
        shape_gradient : array-like
            The shape gradient direction at each point
        s0 : float
            Center of the local bump
        sigma_local : float
            Width of the local Gaussian in parameter space
        amplitude : float
            Scale factor for the bump
        n_derivs : int
            Number of derivatives to compute
        """
        # Create a dummy sampler to satisfy PerturbationSample's requirements
        dummy_sampler = type('DummySampler', (), {'points': points, 'n_derivs': n_derivs})()
        super().__init__(dummy_sampler)
        
        self._sample = [shape_gradient]  # Store shape gradient as the base direction
        self.s0 = s0
        self.sigma_local = sigma_local
        self.amplitude = amplitude
        
        # Precompute w(s), w'(s), etc. up to n_derivs
        self._w_derivs = _build_window_and_derivs(
            points=points,
            amplitude=amplitude,
            s0=s0,
            sigma_local=sigma_local,
            n_derivs=n_derivs
        )
        
        # For derivatives of shape gradient, we'll use zeros since we don't have that info
        for _ in range(n_derivs):
            self._sample.append(np.zeros_like(shape_gradient))
            
    def __getitem__(self, d):
        """
        Return the d-th derivative of [w(s)*g(s)] by the product rule:
          f(s) = w(s)*g(s)
          f^(d)(s) = sum_{k=0..d} comb(d,k) * w^(k)(s)* g^(d-k)(s)
        """
        if d >= len(self._sample):
            raise ValueError(f"Requested derivative {d}, but only have up to {len(self._sample)-1}.")

        out = np.zeros_like(self._sample[0])  # shape (npoints, 3)
        for k in range(d+1):
            w_k = self._w_derivs[k]            # shape (npoints,)
            g_dk = self._sample[d-k]          # shape (npoints, 3)
            c = comb(d, k)
            out += c*(w_k[:, None]*g_dk)
        return out


def perturb_coil_curve_aligned(original_curve, shape_gradient, s0=0.5, 
                             sigma_local=0.01, amplitude=1.0, n_derivs=1):
    """
    Construct a localized perturbation on original_curve that aligns with 
    the shape gradient direction.
    
    Parameters:
    -----------
    original_curve : Curve
        The base curve to perturb
    shape_gradient : array-like
        The shape gradient direction at each point
    s0 : float
        Center of the local perturbation
    sigma_local : float
        Width of the local Gaussian window
    amplitude : float
        Scale factor for the perturbation
    n_derivs : int
        Number of derivatives to compute
        
    Returns:
    --------
    CurvePerturbed
        A new curve with the aligned perturbation applied
    """
    aligned_sample = ShapeGradientPerturbationSample(
        points=original_curve.quadpoints,
        shape_gradient=shape_gradient,
        s0=s0,
        sigma_local=sigma_local,
        amplitude=amplitude,
        n_derivs=n_derivs
    )
    
    perturbed = CurvePerturbed(original_curve, aligned_sample)
    return perturbed