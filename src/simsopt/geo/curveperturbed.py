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
    # Removed "LocalizedPerturbationSample" from __all__ since it's not defined or used
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
        cov_mat = np.zeros((n*(self.n_derivs+1), n*(self.n_derivs+1)))

        def kernel(x, y):
            return sum(
                (self.sigma**2) * exp(-(x-y + i)**2 / (self.length_scale**2)) 
                for i in range(-5, 6)
            )

        XX, YY = np.meshgrid(xs, xs, indexing='ij')
        x = Symbol("x", real=True)
        y = Symbol("y", real=True)
        f = kernel(x, y)

        for ii in range(self.n_derivs+1):
            for jj in range(self.n_derivs+1):
                lam = lambdify((x, y), f.diff(*(ii*[x] + jj*[y])), "numpy")
                cov_mat[(ii*n):((ii+1)*n), (jj*n):((jj+1)*n)] = lam(XX, YY)

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
        z = randomgen.standard_normal(size=(n*(n_derivs+1), 3))
        return [self.L @ z[(i*n):((i+1)*n), :] for i in range(n_derivs+1)]


###############################################################################
#                            Perturbation Sample
###############################################################################
class PerturbationSample(GSONable):
    """
    Stores a single realization (sample) of a perturbation, 
    to be applied to one or multiple curves.
    """

    def __init__(self, sampler, randomgen=None, sample=None):
        self.sampler = sampler
        self.randomgen = randomgen
        self._sample = sample if sample else self.sampler.draw_sample(self.randomgen)

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
                f"Only {len(self._sample)-1} derivatives are available. "
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

    def __init__(self, curve, sample):
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, depends_on=[curve])
        self.sample = sample

    def resample(self):
        """Resample the perturbation and invalidate the cached geometry."""
        self.sample.resample()
        self.recompute_bell()

    def gamma_impl(self, gamma, quadpoints):
        """Position = base curve + 0th-derivative of the perturbation."""
        gamma[:] = self.curve.gamma() + self.sample[0]

    def gammadash_impl(self, gammadash):
        """Tangent = base tangent + 1st derivative of the perturbation."""
        gammadash[:] = self.curve.gammadash() + self.sample[1]

    def gammadashdash_impl(self, gammadashdash):
        """2nd derivative = base + 2nd derivative of the perturbation."""
        gammadashdash[:] = self.curve.gammadashdash() + self.sample[2]

    def gammadashdashdash_impl(self, gammadashdashdash):
        """3rd derivative = base + 3rd derivative of the perturbation."""
        gammadashdashdash[:] = self.curve.gammadashdashdash() + self.sample[3]


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
        n_derivs=1
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
    (Uses an arbitrary random direction.)
    """
    rg = np.random.Generator(PCG64(seed, inc=0))

    # Build a random global field:
    sampler = GaussianSampler(
        points=original_curve.quadpoints,
        sigma=sigma,
        length_scale=length_scale,
        n_derivs=n_derivs
    )

    # Then envelope it with a local Gaussian window in direction = random(…)
    localized_sample = LocalizedDirectionalPerturbationSample(
        points=original_curve.quadpoints,
        base_direction=np.random.randn(len(original_curve.quadpoints), 3),
        sigma_local=sigma_local,
        amplitude=amplitude,
        n_derivs=n_derivs,
        s0=s0
    )

    return CurvePerturbed(original_curve, localized_sample)


###############################################################################
#                Helpers for Windowing & Derivatives
###############################################################################
def _build_window_and_derivs(points, amplitude, s0, sigma_local, n_derivs):
    """
    Builds a 1D Gaussian window w(s) = amplitude * exp(-((s - s0)/sigma_local)^2 / 2)
    and its derivatives up to order n_derivs.
    """
    s = sympy.Symbol("s", real=True)
    w_sym = amplitude * sympy.exp(-0.5*((s - s0)/sigma_local)**2)
    w_syms = [w_sym.diff(s, i) for i in range(n_derivs+1)]
    return [sympy.lambdify(s, expr, "numpy")(points) for expr in w_syms]


###############################################################################
#      ShapeGradientPerturbationSample + perturb_coil_curve_aligned
###############################################################################
class ShapeGradientPerturbationSample(PerturbationSample):
    """
    A perturbation that aligns with a user-supplied shape-gradient vector
    field, modulated by a Gaussian envelope around s0.
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

        # Build the local Gaussian window w(s):
        self._w_derivs = _build_window_and_derivs(
            points=points,
            amplitude=amplitude,
            s0=s0,
            sigma_local=sigma_local,
            n_derivs=n_derivs
        )

        # If user wants up to n_derivs, the shape gradient is constant => derivatives=0
        for _ in range(n_derivs):
            self._sample.append(np.zeros_like(shape_gradient))

    def __getitem__(self, d):
        if d >= len(self._sample):
            raise ValueError(
                f"Requested derivative {d}, but only have up to {len(self._sample)-1}."
            )
        out = np.zeros_like(self._sample[0])
        # product rule: w^(k)*g^(d-k)
        for k in range(d+1):
            w_k = self._w_derivs[k]            # shape (npoints,)
            g_dk = self._sample[d-k]          # shape (npoints, 3)
            c = comb(d, k)
            out += c * (w_k[:, None] * g_dk)
        return out


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

    - Optionally auto-locate s0 from the maximum of auxiliary_scalar_field 
      if auto_locate in ("concavity", "shapegrad").
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
        # Possibly pick s0 from maximum of auxiliary_scalar_field
        if self.auto_locate in ("concavity", "shapegrad"):
            if self.auxiliary_scalar_field is None:
                raise ValueError(
                    f"auto_locate={self.auto_locate} requires auxiliary_scalar_field."
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
        # Direction has no parametric derivatives => just zeros
        self._direction_derivs = [
            self.base_direction
        ] + [np.zeros_like(self.base_direction) for _ in range(self.n_derivs)]

        # Precompute sample for each derivative
        self._sample = [self._compute_derivative(d) for d in range(self.n_derivs+1)]
        super().__init__(sampler=None, sample=self._sample)

    def _compute_derivative(self, d):
        out = np.zeros_like(self.base_direction)
        for k in range(d+1):
            out += comb(d, k)*self._w_derivs[k][:, None]*self._direction_derivs[d-k]
        return out

    def __getitem__(self, d):
        if d > self.n_derivs:
            raise ValueError(f"Requested derivative {d}, but only have up to {self.n_derivs}.")
        return self._sample[d]


def perturb_coil_curve_localized_directional(
    original_curve, 
    base_direction, 
    auto_locate=None, 
    auxiliary_scalar_field=None, 
    s0=0.5, 
    sigma_local=0.01, 
    amplitude=1.0, 
    n_derivs=1
):
    """
    Create a localized perturbation aligned with an external direction field.
    - Optionally auto-locate the Gaussian center s0 by scanning a scalar field.
    """
    points = original_curve.quadpoints
    if base_direction.shape[0] != len(points):
        raise ValueError(
            "base_direction must have shape (len(quadpoints), 3)."
        )

    localized_sample = LocalizedDirectionalPerturbationSample(
        points=points, 
        base_direction=base_direction, 
        sigma_local=sigma_local,
        amplitude=amplitude, 
        n_derivs=n_derivs, 
        s0=s0,
        auto_locate=auto_locate, 
        auxiliary_scalar_field=auxiliary_scalar_field
    )

    return CurvePerturbed(original_curve, localized_sample)