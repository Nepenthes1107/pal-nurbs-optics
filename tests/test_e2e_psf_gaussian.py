import unittest

import torch

from biot.e2e.psf_gaussian import gaussianized_ray_landing_psf, psf_centroid_mm
from biot.e2e.rays import make_pupil_rays
from biot.e2e.surfaces import DifferentiablePALSurface, SurfaceDomain
from biot.e2e.tracing import PALSurface, trace_to_image_plane


class TestE2EGaussianPSF(unittest.TestCase):
    def test_psf_is_non_negative_finite_and_energy_normalized(self):
        spots = torch.tensor([[0.0, 0.0], [0.01, -0.01]], dtype=torch.float64)
        weights = torch.tensor([0.25, 0.75], dtype=torch.float64)

        psf = gaussianized_ray_landing_psf(
            spots,
            weights,
            psf_size_px=33,
            pixel_pitch_mm=0.002,
            sigma_px=1.5,
        )

        self.assertEqual(psf.shape, (33, 33))
        self.assertTrue(torch.all(torch.isfinite(psf)))
        self.assertTrue(torch.all(psf >= 0))
        self.assertTrue(torch.allclose(psf.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12))

    def test_spot_shift_moves_psf_centroid(self):
        weights = torch.ones(1, dtype=torch.float64)
        psf_left = gaussianized_ray_landing_psf(
            torch.tensor([[0.0, 0.0]], dtype=torch.float64),
            weights,
            psf_size_px=65,
            pixel_pitch_mm=0.002,
            sigma_px=2.0,
        )
        psf_right = gaussianized_ray_landing_psf(
            torch.tensor([[0.012, -0.006]], dtype=torch.float64),
            weights,
            psf_size_px=65,
            pixel_pitch_mm=0.002,
            sigma_px=2.0,
        )

        c_left = psf_centroid_mm(psf_left, 0.002)
        c_right = psf_centroid_mm(psf_right, 0.002)

        self.assertGreater(float(c_right[0] - c_left[0]), 0.009)
        self.assertLess(float(c_right[1] - c_left[1]), -0.004)

    def test_batched_psf_shape_and_energy(self):
        spots = torch.tensor(
            [
                [[0.0, 0.0], [0.004, 0.0]],
                [[-0.004, 0.004], [0.0, -0.004]],
            ],
            dtype=torch.float64,
        )
        weights = torch.tensor([[0.5, 0.5], [0.2, 0.8]], dtype=torch.float64)

        psf = gaussianized_ray_landing_psf(
            spots,
            weights,
            psf_size_px=31,
            pixel_pitch_mm=0.002,
            sigma_px=1.2,
        )

        self.assertEqual(psf.shape, (2, 31, 31))
        self.assertTrue(torch.allclose(psf.sum(dim=(-2, -1)), torch.ones(2, dtype=torch.float64), atol=1e-12))

    def test_chunked_psf_matches_full_gaussian_sum(self):
        spots = torch.linspace(-0.03, 0.03, 20, dtype=torch.float64)
        spots = torch.stack((spots, torch.flip(spots, dims=(0,))), dim=-1)
        weights = torch.linspace(0.1, 1.0, 20, dtype=torch.float64)

        full = gaussianized_ray_landing_psf(
            spots,
            weights,
            psf_size_px=41,
            pixel_pitch_mm=0.002,
            sigma_px=1.7,
        )
        chunked = gaussianized_ray_landing_psf(
            spots,
            weights,
            psf_size_px=41,
            pixel_pitch_mm=0.002,
            sigma_px=1.7,
            ray_chunk_size=6,
        )

        self.assertTrue(torch.allclose(chunked, full, atol=1e-14, rtol=1e-12))
        self.assertTrue(torch.allclose(chunked.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12))

    def test_psf_loss_backpropagates_to_spots(self):
        spots = torch.tensor([[0.006, 0.0]], dtype=torch.float64, requires_grad=True)
        weights = torch.ones(1, dtype=torch.float64)

        psf = gaussianized_ray_landing_psf(
            spots,
            weights,
            psf_size_px=33,
            pixel_pitch_mm=0.002,
            sigma_px=1.5,
        )
        center = psf.shape[-1] // 2
        loss = psf[center, center]
        loss.backward()

        self.assertIsNotNone(spots.grad)
        self.assertTrue(torch.all(torch.isfinite(spots.grad)))
        self.assertGreater(float(spots.grad.abs().sum()), 0.0)

    def test_valid_mask_removes_invalid_ray_energy(self):
        spots = torch.tensor([[0.0, 0.0], [0.02, 0.0]], dtype=torch.float64)
        weights = torch.tensor([0.1, 10.0], dtype=torch.float64)
        valid = torch.tensor([True, False])

        psf = gaussianized_ray_landing_psf(
            spots,
            weights,
            valid=valid,
            psf_size_px=65,
            pixel_pitch_mm=0.002,
            sigma_px=1.5,
        )
        centroid = psf_centroid_mm(psf, 0.002)

        self.assertTrue(torch.allclose(psf.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12))
        self.assertTrue(torch.allclose(centroid, torch.zeros(2, dtype=torch.float64), atol=1e-8))

    def test_psf_loss_backpropagates_to_pal_parameters_through_tracing(self):
        rays = make_pupil_rays(
            sample_count=3,
            pupil_radius_mm=1.0,
            field_x_deg=2.0,
            field_y_deg=0.0,
            dtype=torch.float64,
        )
        pal = DifferentiablePALSurface.from_quadratic_base(
            control_shape=(5, 5),
            domain=SurfaceDomain(x_range_mm=(-3.0, 3.0), y_range_mm=(-3.0, 3.0)),
            dtype=torch.float64,
            device="cpu",
        )
        interface = PALSurface(pal, vertex_z_mm=5.0, aperture_radius_mm=3.0, n_after=1.5)
        trace = trace_to_image_plane(rays, [interface], image_z_mm=20.0)

        psf = gaussianized_ray_landing_psf(
            trace.spots_mm,
            rays.weights,
            valid=trace.valid,
            psf_size_px=65,
            pixel_pitch_mm=0.01,
            sigma_px=1.5,
        )
        centroid = psf_centroid_mm(psf, 0.01)
        loss = centroid.pow(2).sum()
        loss.backward()

        self.assertTrue(bool(trace.valid.any()))
        self.assertIsNotNone(pal.theta_bspline.grad)
        self.assertTrue(torch.all(torch.isfinite(pal.theta_bspline.grad)))
        self.assertGreater(float(pal.theta_bspline.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
