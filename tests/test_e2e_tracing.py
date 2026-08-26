import math
import unittest

import torch

from biot.e2e.rays import RayBundle, field_direction, make_pupil_rays, pupil_disk_grid
from biot.e2e.surfaces import DifferentiablePALSurface, SurfaceDomain
from biot.e2e.tracing import (
    PALSurface,
    PlaneSurface,
    SphericalSurface,
    intersect_z_plane,
    snell_refract,
    trace_to_image_plane,
)


class TestE2ETracing(unittest.TestCase):
    def test_pupil_disk_grid_returns_normalized_weights(self):
        points, weights = pupil_disk_grid(5, 2.0, dtype=torch.float64)

        self.assertEqual(points.shape[-1], 2)
        self.assertEqual(weights.shape, (points.shape[0],))
        self.assertTrue(torch.all(points.pow(2).sum(dim=-1) <= 4.0 + 1e-12))
        self.assertTrue(torch.allclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64)))

    def test_plane_propagation_matches_analytic_z_intersection(self):
        origins = torch.tensor([[1.0, -2.0, 0.0]], dtype=torch.float64)
        direction = field_direction(10.0, -5.0, dtype=torch.float64).reshape(1, 3)
        ray = RayBundle(origins, direction, torch.ones(1, dtype=torch.float64), torch.tensor(555.0, dtype=torch.float64))

        points, valid, distance = intersect_z_plane(ray, 10.0)

        expected_x = origins[0, 0] + 10.0 * direction[0, 0] / direction[0, 2]
        expected_y = origins[0, 1] + 10.0 * direction[0, 1] / direction[0, 2]
        self.assertTrue(bool(valid[0]))
        self.assertTrue(torch.allclose(points[0, :2], torch.stack((expected_x, expected_y)), atol=1e-12))
        self.assertGreater(float(distance[0]), 10.0)

    def test_snell_refraction_matches_scalar_law(self):
        theta_i = math.radians(30.0)
        incident = torch.tensor([[math.sin(theta_i), 0.0, math.cos(theta_i)]], dtype=torch.float64)
        normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)

        refracted, valid = snell_refract(incident, normal, 1.0, 1.5)

        theta_t = math.asin(math.sin(theta_i) / 1.5)
        self.assertTrue(bool(valid[0]))
        self.assertAlmostEqual(float(refracted[0, 0]), math.sin(theta_t), places=12)
        self.assertAlmostEqual(float(refracted[0, 2]), math.cos(theta_t), places=12)
        self.assertTrue(torch.allclose(torch.linalg.norm(refracted, dim=-1), torch.ones(1, dtype=torch.float64)))

    def test_snell_marks_total_internal_reflection_invalid(self):
        theta_i = math.radians(50.0)
        incident = torch.tensor([[math.sin(theta_i), 0.0, math.cos(theta_i)]], dtype=torch.float64)
        normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)

        _, valid = snell_refract(incident, normal, 1.5, 1.0)

        self.assertFalse(bool(valid[0]))

    def test_spherical_center_ray_has_no_refraction_deviation(self):
        ray = RayBundle(
            torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            torch.tensor(555.0, dtype=torch.float64),
        )
        surface = SphericalSurface(vertex_z_mm=5.0, radius_mm=20.0, aperture_radius_mm=3.0, n_after=1.5)

        result = trace_to_image_plane(ray, [surface], image_z_mm=10.0)

        self.assertTrue(bool(result.valid[0]))
        self.assertTrue(torch.allclose(result.spots_mm[0], torch.zeros(2, dtype=torch.float64), atol=1e-12))
        self.assertTrue(torch.allclose(result.final_ray.directions[0], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), atol=1e-12))

    def make_pal_interface(self):
        pal = DifferentiablePALSurface.from_quadratic_base(
            control_shape=(5, 5),
            domain=SurfaceDomain(x_range_mm=(-3.0, 3.0), y_range_mm=(-3.0, 3.0)),
            dtype=torch.float64,
            device="cpu",
        )
        return pal, PALSurface(pal, vertex_z_mm=5.0, aperture_radius_mm=3.0, n_after=1.5)

    def test_pal_parameter_perturbation_changes_spot(self):
        ray = RayBundle(
            torch.tensor([[1.0, 0.25, 0.0]], dtype=torch.float64),
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            torch.tensor(555.0, dtype=torch.float64),
        )
        pal, interface = self.make_pal_interface()
        baseline = trace_to_image_plane(ray, [interface], image_z_mm=20.0)

        with torch.no_grad():
            pal.theta_zernike[0] = 0.02
        perturbed = trace_to_image_plane(ray, [interface], image_z_mm=20.0)

        self.assertTrue(bool(baseline.valid[0]))
        self.assertTrue(bool(perturbed.valid[0]))
        self.assertGreater(float((baseline.spots_mm - perturbed.spots_mm).abs().sum()), 1.0e-5)

    def test_spot_loss_backpropagates_to_pal_parameters(self):
        ray = make_pupil_rays(
            sample_count=3,
            pupil_radius_mm=1.0,
            field_x_deg=2.0,
            field_y_deg=0.0,
            pupil_z_mm=0.0,
            dtype=torch.float64,
        )
        pal, interface = self.make_pal_interface()

        result = trace_to_image_plane(ray, [interface], image_z_mm=20.0)
        loss = result.spots_mm[result.valid].pow(2).mean()
        loss.backward()

        self.assertTrue(bool(result.valid.any()))
        self.assertIsNotNone(pal.theta_bspline.grad)
        self.assertTrue(torch.all(torch.isfinite(pal.theta_bspline.grad)))
        self.assertGreater(float(pal.theta_bspline.grad.abs().sum()), 0.0)

    def test_plane_surface_in_sequence_preserves_valid_mask(self):
        ray = make_pupil_rays(sample_count=3, pupil_radius_mm=1.0, dtype=torch.float64)
        surface = PlaneSurface(vertex_z_mm=5.0, aperture_radius_mm=0.25, n_after=1.0)

        result = trace_to_image_plane(ray, [surface], image_z_mm=10.0)

        self.assertGreater(int(result.valid.numel()), int(result.valid.sum()))
        self.assertGreater(int(result.valid.sum()), 0)
        self.assertTrue(torch.all(result.final_ray.weights[~result.valid] == 0))


if __name__ == "__main__":
    unittest.main()
