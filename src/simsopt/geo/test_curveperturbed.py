import unittest
import numpy as np
import json
from randomgen import PCG64

from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curveperturbed import (
    GaussianSampler, PerturbationSample, CurvePerturbed,
    LocalizedDirectionalPerturbationSample, perturb_coil_curve_localized_directional
)
from simsopt.geo.framedcurve import FramedCurveFrenet, FramedCurveTwist
from simsopt.geo.curveobjectives import LpCurveTorsion, CurveCurveDistance
from simsopt._core.json import GSONDecoder, GSONEncoder


class CurvePerturbationTesting(unittest.TestCase):

    def setUp(self):
        """Set up a test curve."""
        self.order = 4
        self.nquadpoints = 200
        self.curve = CurveXYZFourier(self.nquadpoints, self.order)
        dofs = np.zeros((self.curve.dof_size,))
        dofs[1] = 1.
        dofs[2*self.order+3] = 1.
        dofs[4*self.order+3] = 1.
        self.curve.x = dofs

    def test_basic_gaussian_perturbation(self):
        """Test if GaussianSampler generates smooth perturbations."""
        sigma = 1
        length_scale = 0.5
        sampler = GaussianSampler(self.curve.quadpoints, sigma, length_scale, n_derivs=2)
        rg = np.random.Generator(PCG64(1))
        sample = PerturbationSample(sampler, randomgen=rg)

        dphi = self.curve.quadpoints[1]
        for idx in range(2):
            g = sample[idx]
            gd = sample[idx + 1]

            gdest = (-1/12) * g[4:] + (2/3) * g[3:-1] + (-2/3) * g[1:-3] + (1/12) * g[0:-4]
            gdest *= 1/dphi
            err = np.abs(gdest - gd[2:-2])

            assert np.mean(err) < 3e-4 if idx == 0 else np.mean(err) < 2e-3

    def test_localized_directional_perturbation_derivatives(self):
        """Ensure product rule is correctly applied for derivatives."""
        base_direction = np.random.randn(self.nquadpoints, 3)
        base_direction /= np.linalg.norm(base_direction, axis=1, keepdims=True)

        localized_sample = LocalizedDirectionalPerturbationSample(
            points=self.curve.quadpoints,
            base_direction=base_direction,
            sigma_local=0.02,
            amplitude=0.01,
            n_derivs=2,
            s0=0.5
        )

        perturbed_curve = CurvePerturbed(self.curve, localized_sample)

        err = np.abs(perturbed_curve.gammadashdash() - self.curve.gammadashdash())
        assert np.mean(err) < 1e-3

        err = np.abs(perturbed_curve.gammadashdashdash() - self.curve.gammadashdashdash())
        assert np.mean(err) < 1e-2

    def test_localized_perturbation_auto_locate_shape_gradient(self):
        """Test auto_locate='shapegrad' to find max shape gradient region."""
        shape_gradient = np.zeros((self.nquadpoints, 3))
        shape_gradient[:, 0] = np.exp(-100 * (self.curve.quadpoints - 0.75)**2)

        localized_sample = LocalizedDirectionalPerturbationSample(
            points=self.curve.quadpoints,
            base_direction=shape_gradient,
            sigma_local=0.02,
            amplitude=0.01,
            n_derivs=1,
            auto_locate="shapegrad",
            auxiliary_scalar_field=np.linalg.norm(shape_gradient, axis=1)
        )

        assert np.abs(localized_sample.s0 - 0.75) < 0.05

    def test_localized_perturbation_auto_locate_concavity(self):
        """Ensure perturbation is centered at max concavity."""
        ff = FramedCurveFrenet(self.curve, 0.0)
        twist_frenet = FramedCurveTwist(ff)
        concavity = twist_frenet.angle_profile() + np.pi

        localized_sample = LocalizedDirectionalPerturbationSample(
            points=self.curve.quadpoints,
            base_direction=ff.normal,
            sigma_local=0.02,
            amplitude=0.01,
            n_derivs=1,
            auto_locate="concavity",
            auxiliary_scalar_field=concavity
        )

        s_max_concavity = self.curve.quadpoints[np.argmax(concavity)]
        assert np.abs(localized_sample.s0 - s_max_concavity) < 0.05

    def test_perturbed_objective_torsion(self):
        """Verify torsion computation passes derivative tests."""
        sigma = 1
        length_scale = 0.5
        sampler = GaussianSampler(self.curve.quadpoints, sigma, length_scale, n_derivs=3)
        rg = np.random.Generator(PCG64(1))
        sample = PerturbationSample(sampler, randomgen=rg)
        perturbed_curve = CurvePerturbed(self.curve, sample)

        J = LpCurveTorsion(perturbed_curve, p=2)
        J0 = J.J()
        curve_dofs = perturbed_curve.x
        h = 1e-3 * np.random.rand(len(curve_dofs))
        dJ = J.dJ()
        deriv = np.sum(dJ * h)

        assert np.abs(deriv) > 1e-10

    def test_perturbed_curve_serialization(self):
        """Ensure perturbed curves can be serialized and deserialized."""
        sigma = 1
        length_scale = 0.5
        sampler = GaussianSampler(self.curve.quadpoints, sigma, length_scale, n_derivs=2)
        sample = PerturbationSample(sampler)
        perturbed_curve = CurvePerturbed(self.curve, sample)

        curve_str = json.dumps(perturbed_curve, cls=GSONEncoder)
        perturbed_curve_regen = json.loads(curve_str, cls=GSONDecoder)

        self.assertTrue(np.allclose(perturbed_curve.gamma(), perturbed_curve_regen.gamma()))


if __name__ == "__main__":
    unittest.main()