###############################################################################
# curveperturbed.py
###############################################################################
from dataclasses import dataclass

import numpy as np
import sympy
from sympy import Symbol, lambdify, exp
from math import comb
import warnings

from .._core.json import GSONable
from .._core.util import RealArray

import simsoptpp as sopp
from simsopt.geo.curve import Curve
import numpy as np
from randomgen import PCG64

__all__ = [
    "GaussianSampler",
    "PerturbationSample",
    "CurvePerturbed",
    "perturb_coil_curve",
    "perturb_coil_curve_localized",
    "ShapeGradientPerturbationSample",
    "perturb_coil_curve_aligned",
    "LocalizedDirectionalPerturbationSample",
    "perturb_coil_curve_localized_directional",
]


###############################################################################
#                               Gaussian Sampler
###############################################################################
@dataclass
class GaussianSampler(GSONable):
    """
    Generate a periodic Gaussian process on [0,1] using a covariance kernel
    for smooth random perturbations.
    """

    points: RealArray
    sigma: float
    length_scale: float
    n_derivs: int = 1

    def __post_init__(self):
        xs = self.points
        n = len(xs)
        cov_mat = np.zeros((n * (self.n_derivs + 1), n * (self.n_derivs + 1)))

        def kernel(x, y):
            return sum(
                (self.sigma**2) * exp(-((x - y + i) ** 2) / (self.length_scale**2))
                for i in range(-5, 6)
            )

        XX, YY = np.meshgrid(xs, xs, indexing="ij")
        x = Symbol("x", real=True)
        y = Symbol("y", real=True)
        f = kernel(x, y)

        # Build partial derivatives for 0..n_derivs
        for ii in range(self.n_derivs + 1):
            for jj in range(self.n_derivs + 1):
                if ii + jj == 0:
                    lam = lambdify((x, y), f, "numpy")
                else:
                    lam = lambdify((x, y), f.diff(*(ii * [x] + jj * [y])), "numpy")
                cov_mat[
                    (ii * n) : ((ii + 1) * n), (jj * n) : ((jj + 1) * n)
                ] = lam(XX, YY)

        # Use LDL decomposition to handle large matrices
        from scipy.linalg import ldl

        lu, d, _ = ldl(cov_mat)
        self.L = lu @ np.sqrt(np.maximum(d, 0))

    def draw_sample(self, randomgen=None):
        """
        Returns a list of (n_derivs+1) arrays, each shape (len(points), 3),
        containing the position-perturbation and its derivatives.
        """
        n = len(self.points)
        n_derivs = self.n_derivs
        if randomgen is None:
            randomgen = np.random
        z = randomgen.standard_normal(size=(n * (n_derivs + 1), 3))
        # Multiply by Cholesky-like factor
        full = self.L @ z
        return [full[(i * n) : ((i + 1) * n), :] for i in range(n_derivs + 1)]


###############################################################################
#                            PerturbationSample
###############################################################################
class PerturbationSample(GSONable):
    """
    Stores a single realization (sample) of a perturbation,
    to be applied to one or multiple curves.
    """

    def __init__(self, sampler, randomgen=None, sample=None):
        self.sampler = sampler
        self.randomgen = randomgen
        self._sample = sample if sample is not None else self.sampler.draw_sample(self.randomgen)

    def resample(self):
        """Re-draw the random sample from the underlying distribution."""
        self._sample = self.sampler.draw_sample(self.randomgen)

    def __getitem__(self, deriv):
        """
        Return the array for the `deriv`-th derivative
        (0 for position, 1 for first derivative, etc.).
        """
        if deriv >= len(self._sample):
            raise ValueError(
                f"Only {len(self._sample) - 1} derivatives are available. "
                f"Requested derivative {deriv} is out of range."
            )
        return self._sample[deriv]


###############################################################################
#                               CurvePerturbed
###############################################################################
class CurvePerturbed(sopp.Curve, Curve):
    """
    A curve that has been perturbed by a given `PerturbationSample`.
    """

    def __init__(self, curve, sample, zero_mean=False):
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
        Curve.__init__(self, x0=np.asarray([]), depends_on=[curve])
        self.sample = sample
        self.zero_mean = zero_mean

    def resample(self):
        """Resample the perturbation and invalidate the cached geometry."""
        self.sample.resample()
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        pert = self.sample[0].copy()
        gamma[:] = self.curve.gamma() + pert
        if self.zero_mean:
            mean_before = np.mean(self.curve.gamma() * self.curve.incremental_arclength()[:, None], axis=0)
            mean_after = np.mean(gamma * self.incremental_arclength()[:, None], axis=0)
            adj = (mean_after-mean_before)/np.mean(self.incremental_arclength())
            gamma -= adj[None, :]

    def gammadash_impl(self, gammadash):
        gammadash[:] = self.curve.gammadash() + self.sample[1]

    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:] = self.curve.gammadashdash() + self.sample[2]

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:] = self.curve.gammadashdashdash() + self.sample[3]

    def dgamma_by_dcoeff_vjp(self, v):
        res = self.curve.dgamma_by_dcoeff_vjp(v)
        if self.zero_mean:
            pert = self.sample[0]
            n = len(self.quadpoints)
            denom = np.mean(self.incremental_arclength())
            mean_before = np.mean(self.curve.gamma() * self.curve.incremental_arclength()[:, None], axis=0)
            mean_after = np.mean((self.curve.gamma() + self.sample[0]) * self.incremental_arclength()[:, None], axis=0)
            adj = mean_after-mean_before

            # # derivative of pert * self.incremental_arclength()[:, None]
            # res -= (1./denom)*self.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 0])/n))*pert[:, 0])
            # res -= (1./denom)*self.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 1])/n))*pert[:, 1])
            # res -= (1./denom)*self.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 2])/n))*pert[:, 2])
            # # derivative of self.curve.gamma() * self.incremental_arclength()[:, None]
            # res -= (1./denom)*self.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 0])/n))*self.curve.gamma()[:, 0])
            # res -= (1./denom)*self.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 1])/n))*self.curve.gamma()[:, 1])
            # res -= (1./denom)*self.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 2])/n))*self.curve.gamma()[:, 2])
            # res -= (1./denom)*self.curve.dgamma_by_dcoeff_vjp((np.sum(v, axis=0)/n)[None, :]*self.incremental_arclength()[:, None])

            # # derivative of self.curve.gamma() * self.curve.incremental_arclength()[:, None]
            # res += (1./denom)*self.curve.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 0])/n))*self.curve.gamma()[:, 0])
            # res += (1./denom)*self.curve.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 1])/n))*self.curve.gamma()[:, 1])
            # res += (1./denom)*self.curve.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 2])/n))*self.curve.gamma()[:, 2])
            # res += (1./denom)*self.curve.dgamma_by_dcoeff_vjp((np.sum(v, axis=0)/n)[None, :]*self.curve.incremental_arclength()[:, None])

            # # derivative of the denominator
            # res += (adj[0]/denom**2) * self.dincremental_arclength_by_dcoeff_vjp(np.ones((n, ))*float(np.mean(v[:, 0])))
            # res += (adj[1]/denom**2) * self.dincremental_arclength_by_dcoeff_vjp(np.ones((n, ))*float(np.mean(v[:, 1])))
            # res += (adj[2]/denom**2) * self.dincremental_arclength_by_dcoeff_vjp(np.ones((n, ))*float(np.mean(v[:, 2])))

            vmean = np.mean(v, axis=0)
            v0, v1, v2 = vmean
            lhs1 = (1./denom) * (
                - v0*pert[:, 0] - v1*pert[:, 1] - v2*pert[:, 2]
                - v0*self.curve.gamma()[:, 0] - v1*self.curve.gamma()[:, 1] - v2*self.curve.gamma()[:, 2]
            ) + (1./denom**2) * (
                v0*np.ones((n, ))*adj[0] + v1*np.ones((n, ))*adj[1] + v2*np.ones((n, ))*adj[2]
            )
            res += self.dincremental_arclength_by_dcoeff_vjp(lhs1)

            lhs2 = (1./denom) * (
                + v0*self.curve.gamma()[:, 0] + v1*self.curve.gamma()[:, 1] + v2*self.curve.gamma()[:, 2]
            )
            res += self.curve.dincremental_arclength_by_dcoeff_vjp(lhs2)

            lhs3 = (1./denom) * (
                - vmean[None, :]*self.incremental_arclength()[:, None]
                + vmean[None, :]*self.curve.incremental_arclength()[:, None]
            )
            res += self.curve.dgamma_by_dcoeff_vjp(lhs3)

        return res

    def dgammadash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadash_by_dcoeff_vjp(v)

    def dgammadashdash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadashdash_by_dcoeff_vjp(v)

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadashdashdash_by_dcoeff_vjp(v)


###############################################################################
#                  Basic Coil Perturbation (Global Gaussian)
###############################################################################
def perturb_coil_curve(original_curve, seed, sigma=0.01, length_scale=0.1):
    """
    Perturb a given coil curve with a global Gaussian process.
    """
    rg = np.random.Generator(PCG64(seed, inc=0))
    sampler = GaussianSampler(
        original_curve.quadpoints,
        sigma,
        length_scale,
        n_derivs=1,
    )
    perturbation_sample = PerturbationSample(sampler, randomgen=rg)
    return CurvePerturbed(original_curve, perturbation_sample)


###############################################################################
#               Localized Gaussian Perturbation (Random Direction)
###############################################################################
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
    multiplying a global random field by a Gaussian envelope around s0.
    The direction is chosen randomly in R^3 at each point.
    This is mostly for demonstration of a "random localized" bump.
    """

    rg = np.random.Generator(PCG64(seed, inc=0))

    # Build a random global field:
    sampler = GaussianSampler(
        points=original_curve.quadpoints,
        sigma=sigma,
        length_scale=length_scale,
        n_derivs=n_derivs
    )

    # Draw the sample:
    random_sample = sampler.draw_sample(randomgen=rg)
    random_field_0 = random_sample[0]  # shape (nquad, 3)

    # Build local Gaussian envelope w(s):
    s = sympy.Symbol("s", real=True)
    w_sym = amplitude * sympy.exp(-0.5 * ((s - s0) / sigma_local) ** 2)
    w_derivs_syms = [w_sym.diff(s, i) for i in range(n_derivs + 1)]
    w_lams = [lambdify(s, expr, "numpy") for expr in w_derivs_syms]

    # Envelope the random field
    class LocalizedSample(PerturbationSample):
        def __init__(self):
            # We'll create a dummy sampler just to satisfy base class
            dummy_sampler = type("DummySampler", (), {
                "points": original_curve.quadpoints,
                "n_derivs": n_derivs
            })()
            super().__init__(dummy_sampler, sample=None)
            # Precompute all derivatives (product rule), though for random_field_0
            # we only have the 0th derivative effectively:
            self._sample = []
            for d in range(n_derivs + 1):
                out_d = np.zeros_like(random_field_0)
                for k in range(d + 1):
                    # shape(0) derivatives for random_field_0 except d=0
                    if d == 0:
                        out_d += w_lams[0](original_curve.quadpoints)[:, None] * random_field_0
                    else:
                        # We'll just do w^(d)*random_field_0
                        if k == d:
                            w_k = w_lams[d](original_curve.quadpoints)
                            out_d += w_k[:, None] * random_field_0
                self._sample.append(out_d)

        def __getitem__(self, d):
            return self._sample[d]

    localized_sample = LocalizedSample()
    return CurvePerturbed(original_curve, localized_sample)


###############################################################################
#                Helpers for Windowing & Derivatives
###############################################################################
def _build_window_and_derivs(points, amplitude, s0, sigma_local, n_derivs):
    """
    Builds a 1D Gaussian window w(s) = amplitude * exp(-0.5*((s - s0)/sigma_local)^2)
    and its derivatives up to order n_derivs.
    Returns a list [w^(0), w^(1), ..., w^(n_derivs)], each shape (len(points),).
    """
    s = sympy.Symbol("s", real=True)
    w_sym = amplitude * sympy.exp(-0.5 * ((s - s0) / sigma_local) ** 2)
    w_syms = [w_sym.diff(s, i) for i in range(n_derivs + 1)]
    w_lams = [lambdify(s, expr, "numpy") for expr in w_syms]
    return [w_lams[i](points) for i in range(n_derivs + 1)]


###############################################################################
#      ShapeGradientPerturbationSample + perturb_coil_curve_aligned
###############################################################################
class ShapeGradientPerturbationSample(PerturbationSample):
    """
    A perturbation that aligns with a user-supplied shape-gradient vector
    field, modulated by a Gaussian envelope around s0.

    If you want it to be localized only around s0, pick a small sigma_local.
    """

    def __init__(self, points, shape_gradient, s0=0.5, sigma_local=0.01,
                 amplitude=1.0, n_derivs=1):
        # We create a minimal "sampler" just to satisfy the parent constructor
        dummy_sampler = type("DummySampler", (), {
            "points": points,
            "n_derivs": n_derivs
        })()
        super().__init__(dummy_sampler)

        self._sample = [shape_gradient]  # 0th derivative => shape gradient itself
        self.s0 = s0
        self.sigma_local = sigma_local
        self.amplitude = amplitude
        self.n_derivs = n_derivs

        # Build the local Gaussian window w(s):
        self._w_derivs = _build_window_and_derivs(
            points=points,
            amplitude=amplitude,
            s0=s0,
            sigma_local=sigma_local,
            n_derivs=n_derivs
        )

        # shape_gradient is constant => derivatives=0 in s
        for _ in range(n_derivs):
            self._sample.append(np.zeros_like(shape_gradient))

    def __getitem__(self, d):
        if d >= len(self._sample):
            raise ValueError(
                f"Requested derivative {d}, but only have up to {len(self._sample) - 1}."
            )
        # out = sum_{k=0..d} comb(d,k) * w^(k)*grad^(d-k).
        # But grad^(d-k)=0 if d-k>0 => shape gradient is constant => only k=d matters
        w_d = self._w_derivs[d]  # shape (npoints,)
        return w_d[:, None] * self._sample[0]


def perturb_coil_curve_aligned(original_curve, shape_gradient, s0=0.5,
                               sigma_local=0.01, amplitude=1.0, n_derivs=1):
    """
    Construct a localized perturbation on original_curve that aligns with
    the shape gradient direction, using a Gaussian envelope around s0.
    """
    aligned_sample = ShapeGradientPerturbationSample(
        points=original_curve.quadpoints,
        shape_gradient=shape_gradient,
        s0=s0,
        sigma_local=sigma_local,
        amplitude=amplitude,
        n_derivs=n_derivs
    )
    return CurvePerturbed(original_curve, aligned_sample)


###############################################################################
#    LocalizedDirectionalPerturbationSample +
#    perturb_coil_curve_localized_directional
###############################################################################
@dataclass
class LocalizedDirectionalPerturbationSample(PerturbationSample):
    """
    A Gaussian-localized perturbation in an externally supplied direction field.

    Optionally, if auto_locate == "shapegrad", we locate the center s0 from the
    maximum of auxiliary_scalar_field (usually ||shape_gradient||).
    """

    points: np.ndarray
    base_direction: np.ndarray
    sigma_local: float = 0.01
    amplitude: float = 1.0
    n_derivs: int = 1
    s0: float = None
    auto_locate: str = None
    auxiliary_scalar_field: np.ndarray = None

    def __post_init__(self):
        # Possibly pick s0 from maximum of auxiliary_scalar_field if using shapegrad
        if self.auto_locate == "shapegrad":
            if self.auxiliary_scalar_field is None:
                raise ValueError(
                    f"auto_locate='{self.auto_locate}' requires auxiliary_scalar_field."
                )
            idx = np.argmax(self.auxiliary_scalar_field)
            self.s0 = self.points[idx]
        elif self.s0 is None:
            warnings.warn("No s0 provided; defaulting to 0.5")
            self.s0 = 0.5

        # Build local window derivatives
        self._w_derivs = _build_window_and_derivs(
            self.points,
            self.amplitude,
            self.s0,
            self.sigma_local,
            self.n_derivs
        )
        # Direction has no parametric derivatives => derivatives=0
        self._direction_derivs = [
            self.base_direction
        ] + [np.zeros_like(self.base_direction) for _ in range(self.n_derivs)]

        # Precompute sample for each derivative
        local_samples = []
        for d in range(self.n_derivs + 1):
            # Only the k=d term in product rule is nonzero => w^(d)*base_direction
            w_d = self._w_derivs[d]  # shape (npoints,)
            out = w_d[:, None] * self.base_direction
            local_samples.append(out)

        # Build the PerturbationSample with the final arrays
        super().__init__(sampler=None, sample=local_samples)

    def __getitem__(self, d):
        if d > self.n_derivs:
            raise ValueError(
                f"Requested derivative {d}, but only have up to {self.n_derivs}."
            )
        return self._sample[d]


def perturb_coil_curve_localized_directional(
    original_curve,
    base_direction,
    auto_locate=None,
    auxiliary_scalar_field=None,
    s0=0.5,
    sigma_local=0.01,
    amplitude=1.0,
    n_derivs=1,
):
    """
    Create a localized perturbation aligned with an external direction field.
    Optionally auto-locate the Gaussian center s0 by scanning a scalar field,
    e.g. the norm of the shape gradient.

    Example usage to localize around the largest shape-gradient magnitude:
        shape_grad = ... # array of shape (nquad, 3)
        shape_grad_norm = np.linalg.norm(shape_grad, axis=1)
        perturbed_curve = perturb_coil_curve_localized_directional(
            original_curve,
            base_direction=shape_grad,
            auto_locate="shapegrad",
            auxiliary_scalar_field=shape_grad_norm,
            sigma_local=0.01,
            amplitude=0.02,
            n_derivs=1,
        )
    """
    points = original_curve.quadpoints
    if base_direction.shape[0] != len(points):
        raise ValueError(
            "base_direction must have shape (len(quadpoints), 3). "
            f"Got shape {base_direction.shape} vs {len(points)}."
        )

    localized_sample = LocalizedDirectionalPerturbationSample(
        points=points,
        base_direction=base_direction,
        sigma_local=sigma_local,
        amplitude=amplitude,
        n_derivs=n_derivs,
        s0=s0,
        auto_locate=auto_locate,
        auxiliary_scalar_field=auxiliary_scalar_field,
    )

    return CurvePerturbed(original_curve, localized_sample)