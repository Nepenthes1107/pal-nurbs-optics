import unittest

import torch

from biot.e2e.psf_fft import (
    circular_pupil_mask,
    complex_pupil_from_phase,
    effective_biot_pupil_sample_count,
    standardize_fft_psf_orientation,
    torch_fft_psf_from_phase,
)
from biot.e2e.rays import make_pupil_rays
from biot.e2e.surfaces import DifferentiablePALSurface, SurfaceDomain
from biot.e2e.tracing import PALSurface, trace_to_image_plane


class TestE2ETorchFFTPSF(unittest.TestCase):
    def test_fft_dc_orientation_is_not_bare_vertical_flip(self):
        psf = torch.arange(16, dtype=torch.float64).reshape(4, 4)
        expected = torch.roll(torch.flip(psf, dims=(-2,)), shifts=1, dims=(-2,))
        actual = standardize_fft_psf_orientation(psf)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, torch.flip(psf, dims=(-2,))))

    def test_biot_effective_pupil_count_matches_fft_psf_i_rule(self):
        self.assertEqual(effective_biot_pupil_sample_count(32), 32)
        self.assertEqual(effective_biot_pupil_sample_count(128), 64)
        self.assertEqual(effective_biot_pupil_sample_count(256), 90)
        self.assertEqual(effective_biot_pupil_sample_count(512), 128)

    def test_complex_pupil_shape_and_valid_mask(self):
        sample_count = 9
        aperture = circular_pupil_mask(sample_count, dtype=torch.float64)
        phase = torch.zeros(int(aperture.sum()), dtype=torch.float64)
        valid = torch.ones_like(phase, dtype=torch.bool)
        valid[-1] = False

        pupil, valid_pupil = complex_pupil_from_phase(phase, valid, sample_count=sample_count)

        self.assertEqual(pupil.shape, (sample_count, sample_count))
        self.assertEqual(valid_pupil.shape, (sample_count, sample_count))
        self.assertEqual(int(valid_pupil.sum()), int(valid.sum()))
        self.assertTrue(torch.all(torch.abs(pupil[valid_pupil]) > 0.0))
        self.assertTrue(torch.all(pupil[~aperture] == 0.0))

    def test_fft_psf_is_finite_non_negative_and_energy_normalized(self):
        sample_count = 17
        aperture = circular_pupil_mask(sample_count, dtype=torch.float64)
        coords = torch.linspace(-1.0, 1.0, int(aperture.sum()), dtype=torch.float64)
        phase = 0.25 * coords.pow(2)

        result = torch_fft_psf_from_phase(
            phase,
            torch.ones_like(phase, dtype=torch.bool),
            sample_count=sample_count,
            psf_size_px=65,
        )

        self.assertEqual(result.psf.shape, (65, 65))
        self.assertTrue(torch.all(torch.isfinite(result.psf)))
        self.assertTrue(torch.all(result.psf >= 0.0))
        self.assertTrue(torch.allclose(result.psf.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12))

    def test_complex_pupil_removes_linear_tilt_from_phase(self):
        sample_count = 17
        aperture = circular_pupil_mask(sample_count, dtype=torch.float64)
        coord = torch.linspace(-1.0, 1.0, sample_count, dtype=torch.float64)
        xx, yy = torch.meshgrid(coord, coord, indexing="xy")
        phase = (0.7 + 11.0 * xx - 8.0 * yy).reshape(-1)[aperture.reshape(-1)]
        valid = torch.ones_like(phase, dtype=torch.bool)

        tilted_pupil, tilted_valid = complex_pupil_from_phase(
            phase,
            valid,
            sample_count=sample_count,
            remove_tilt=False,
        )
        corrected_pupil, corrected_valid = complex_pupil_from_phase(
            phase,
            valid,
            sample_count=sample_count,
            remove_tilt=True,
        )

        self.assertTrue(torch.equal(tilted_valid, corrected_valid))
        self.assertGreater(float(torch.std(torch.angle(tilted_pupil[tilted_valid]))), 0.1)
        self.assertTrue(torch.allclose(corrected_pupil[corrected_valid].real, torch.ones_like(phase), atol=1.0e-9))
        self.assertTrue(torch.allclose(corrected_pupil[corrected_valid].imag, torch.zeros_like(phase), atol=1.0e-9))

    def test_complex_pupil_removes_wrapped_linear_tilt_from_phase(self):
        sample_count = 65
        aperture = circular_pupil_mask(sample_count, dtype=torch.float64)
        coord = torch.linspace(-1.0, 1.0, sample_count, dtype=torch.float64)
        xx, yy = torch.meshgrid(coord, coord, indexing="xy")
        phase = (0.7 + 80.0 * xx - 60.0 * yy).reshape(-1)[aperture.reshape(-1)]
        valid = torch.ones_like(phase, dtype=torch.bool)

        corrected_pupil, corrected_valid = complex_pupil_from_phase(
            phase,
            valid,
            sample_count=sample_count,
            remove_tilt=True,
        )

        self.assertTrue(torch.allclose(corrected_pupil[corrected_valid].real, torch.ones_like(phase), atol=1.0e-9))
        self.assertTrue(torch.allclose(corrected_pupil[corrected_valid].imag, torch.zeros_like(phase), atol=1.0e-9))

    def test_fft_psf_linear_tilt_removal_matches_flat_phase(self):
        sample_count = 17
        aperture = circular_pupil_mask(sample_count, dtype=torch.float64)
        coord = torch.linspace(-1.0, 1.0, sample_count, dtype=torch.float64)
        xx, yy = torch.meshgrid(coord, coord, indexing="xy")
        phase = (9.0 * xx + 6.0 * yy).reshape(-1)[aperture.reshape(-1)]
        zero = torch.zeros_like(phase)
        valid = torch.ones_like(phase, dtype=torch.bool)

        corrected = torch_fft_psf_from_phase(
            phase,
            valid,
            sample_count=sample_count,
            psf_size_px=65,
            remove_tilt=True,
        )
        reference = torch_fft_psf_from_phase(
            zero,
            valid,
            sample_count=sample_count,
            psf_size_px=65,
        )

        self.assertTrue(torch.allclose(corrected.psf, reference.psf, atol=1.0e-10))

    def test_fft_psf_loss_backpropagates_to_phase(self):
        sample_count = 15
        aperture = circular_pupil_mask(sample_count, dtype=torch.float64)
        phase = torch.linspace(-0.5, 0.5, int(aperture.sum()), dtype=torch.float64, requires_grad=True)

        result = torch_fft_psf_from_phase(
            phase,
            torch.ones_like(phase, dtype=torch.bool),
            sample_count=sample_count,
            psf_size_px=63,
        )
        center = result.psf.shape[-1] // 2
        loss = result.psf[center - 2, center + 3]
        loss.backward()

        self.assertIsNotNone(phase.grad)
        self.assertTrue(torch.all(torch.isfinite(phase.grad)))
        self.assertGreater(float(phase.grad.abs().sum()), 0.0)

    def test_fft_psf_loss_backpropagates_through_tracing_to_pal(self):
        rays = make_pupil_rays(
            sample_count=5,
            pupil_radius_mm=1.0,
            field_x_deg=1.0,
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
        wavelength_mm = rays.wavelength_nm * 1.0e-6
        phase = 2.0 * torch.pi * trace.optical_path_length_mm / wavelength_mm

        result = torch_fft_psf_from_phase(
            phase,
            trace.valid,
            sample_count=5,
            psf_size_px=33,
        )
        loss = result.psf[12:20, 14:22].sum()
        loss.backward()

        self.assertTrue(bool(trace.valid.any()))
        self.assertIsNotNone(pal.theta_bspline.grad)
        self.assertTrue(torch.all(torch.isfinite(pal.theta_bspline.grad)))
        self.assertGreater(float(pal.theta_bspline.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
