import unittest

import torch

from biot.e2e.bspline import bspline_basis_1d, open_uniform_knots
from biot.e2e.surfaces import DifferentiablePALSurface, SurfaceDomain
from biot.e2e.zernike import DEFAULT_MODES, zernike_basis


class TestE2ESurfaces(unittest.TestCase):
    def make_surface(self):
        return DifferentiablePALSurface.from_quadratic_base(
            control_shape=(5, 5),
            domain=SurfaceDomain(x_range_mm=(-4.0, 4.0), y_range_mm=(-4.0, 4.0)),
            curvature_x_inv_mm=1.0e-4,
            curvature_y_inv_mm=-0.5e-4,
            dtype=torch.float64,
            device="cpu",
        )

    def test_bspline_basis_has_expected_shape_and_partition(self):
        knots = open_uniform_knots(5, 3, (-1.0, 1.0), dtype=torch.float64)
        x = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)

        basis = bspline_basis_1d(x, knots, degree=3)

        self.assertEqual(basis.shape, (9, 5))
        self.assertTrue(torch.all(torch.isfinite(basis)))
        self.assertTrue(torch.allclose(basis.sum(dim=-1), torch.ones_like(x), atol=1e-12))

    def test_theta_zero_matches_base_sag(self):
        surface = self.make_surface()
        x = torch.linspace(-2.0, 2.0, 5, dtype=torch.float64)
        y = torch.linspace(-2.0, 2.0, 5, dtype=torch.float64)
        xx, yy = torch.meshgrid(x, y, indexing="ij")

        sag = surface.sag(xx, yy)
        base = surface.base_sag(xx, yy)

        self.assertTrue(torch.allclose(sag, base, atol=1e-12))
        self.assertTrue(torch.all(torch.isfinite(sag)))

    def test_sag_backward_populates_pal_parameter_gradients(self):
        surface = self.make_surface()
        x = torch.linspace(-3.0, 3.0, 7, dtype=torch.float64)
        y = torch.linspace(-3.0, 3.0, 7, dtype=torch.float64)
        xx, yy = torch.meshgrid(x, y, indexing="ij")

        loss = surface.sag(xx, yy).sum()
        loss.backward()

        self.assertIsNotNone(surface.theta_bspline.grad)
        self.assertTrue(torch.all(torch.isfinite(surface.theta_bspline.grad)))
        self.assertGreater(float(surface.theta_bspline.grad.abs().sum()), 0.0)

    def test_normal_shape_and_unit_length(self):
        surface = self.make_surface()
        with torch.no_grad():
            surface.theta_bspline[2, 2] = 0.01
        x = torch.linspace(-2.0, 2.0, 5, dtype=torch.float64)
        y = torch.linspace(-2.0, 2.0, 5, dtype=torch.float64)
        xx, yy = torch.meshgrid(x, y, indexing="ij")

        normal = surface.normal(xx, yy)
        lengths = torch.sqrt(normal.pow(2).sum(dim=-1))

        self.assertEqual(normal.shape, (5, 5, 3))
        self.assertTrue(torch.all(torch.isfinite(normal)))
        self.assertTrue(torch.allclose(lengths, torch.ones_like(lengths), atol=1e-10))

    def test_smoothness_loss_is_non_negative_and_finite(self):
        surface = self.make_surface()
        with torch.no_grad():
            surface.theta_bspline[1, 2] = 0.01
            surface.theta_zernike[0] = 0.001

        loss = surface.smoothness_loss()

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_zernike_default_modes_exclude_piston_and_tilt(self):
        x = torch.tensor([0.0, 0.5], dtype=torch.float64)
        y = torch.tensor([0.0, 0.25], dtype=torch.float64)

        basis = zernike_basis(x, y, radius_mm=1.0)

        self.assertNotIn("piston", DEFAULT_MODES)
        self.assertNotIn("tilt_x", DEFAULT_MODES)
        self.assertNotIn("tilt_y", DEFAULT_MODES)
        self.assertEqual(basis.shape, (2, len(DEFAULT_MODES)))
        self.assertTrue(torch.all(torch.isfinite(basis)))


if __name__ == "__main__":
    unittest.main()
