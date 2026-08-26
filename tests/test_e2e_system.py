import unittest
from pathlib import Path

import torch

from biot.e2e.system import (
    FitSpec,
    LocalAsphereSurface,
    LocalBSplineSurface,
    LocalGridSagSurface,
    build_fitted_e2e_system,
    make_aimed_pupil_rays,
    make_aimed_reference_ray,
    trace_system_to_image,
    trace_system_to_image_with_phase,
)


class TestE2ESystemImport(unittest.TestCase):
    def test_builds_fitted_system_with_expected_surface_types(self):
        system, temp_path = build_fitted_e2e_system(
            "eye_image_glass.xlsx",
            object_distance=1000.0,
            field_x_deg=0.0,
            field_y_deg=0.0,
            fit_spec=FitSpec(control_shape=(9, 9), sample_shape=(21, 21)),
            device="cpu",
        )
        try:
            self.assertEqual(temp_path.parent.resolve(), Path("eye_image_glass.xlsx").resolve().parent)
            self.assertEqual(len(system.surfaces), len(system.lens.surfaces))
            self.assertIsInstance(system.front_surface, LocalAsphereSurface)
            self.assertIsInstance(system.back_surface, LocalGridSagSurface)
            self.assertFalse(system.back_surface.control_mm.requires_grad)
            self.assertGreater(system.image_distance_mm, 0.0)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def test_optimization_mode_only_back_surface_is_trainable(self):
        system, temp_path = build_fitted_e2e_system(
            "eye_image_glass.xlsx",
            object_distance=1000.0,
            field_x_deg=0.0,
            field_y_deg=0.0,
            fit_spec=FitSpec(control_shape=(9, 9), sample_shape=(21, 21)),
            device="cpu",
            train_back_surface=True,
        )
        try:
            self.assertIsInstance(system.front_surface, LocalAsphereSurface)
            self.assertIsInstance(system.back_surface, LocalBSplineSurface)
            self.assertTrue(system.back_surface.control_mm.requires_grad)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def test_aimed_pupil_rays_trace_to_image_with_valid_mask(self):
        system, temp_path = build_fitted_e2e_system(
            "eye_image_glass.xlsx",
            object_distance=1000.0,
            field_x_deg=0.0,
            field_y_deg=0.0,
            fit_spec=FitSpec(control_shape=(9, 9), sample_shape=(21, 21)),
            device="cpu",
        )
        try:
            rays = make_aimed_pupil_rays(
                system,
                sample_count=5,
                pupil_radius_mm=2.0,
                field_x_deg=0.0,
                field_y_deg=0.0,
                device="cpu",
                dtype=torch.float64,
            )
            trace = trace_system_to_image(system, rays)

            self.assertEqual(trace.spots_mm.shape[-1], 2)
            self.assertEqual(trace.valid.shape, rays.weights.shape)
            self.assertTrue(torch.all(torch.isfinite(trace.spots_mm)))
            self.assertGreater(float(trace.valid.to(torch.float64).mean()), 0.5)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def test_continuous_reference_opl_has_same_phasor_as_existing_phase(self):
        system, temp_path = build_fitted_e2e_system(
            "eye_image_glass.xlsx",
            object_distance=1000.0,
            field_x_deg=0.0,
            field_y_deg=0.0,
            fit_spec=FitSpec(control_shape=(9, 9), sample_shape=(21, 21)),
            device="cpu",
        )
        try:
            rays = make_aimed_pupil_rays(
                system, sample_count=8, pupil_radius_mm=2.0,
                field_x_deg=0.0, field_y_deg=0.0,
                device="cpu", dtype=torch.float64,
            )
            system.reference_ray = make_aimed_reference_ray(
                system, device="cpu", dtype=torch.float64,
            )
            trace = trace_system_to_image_with_phase(
                system, rays, phase_reference="biot_reference_sphere",
            )
            wavelength_mm = system.wavelength_nm * 1e-6
            opl_phasor = torch.exp(1j * (2.0 * torch.pi * trace.reference_opl_mm / wavelength_mm))
            phase_phasor = torch.exp(1j * trace.phase_rad)
            self.assertLess(
                float((opl_phasor[trace.valid] - phase_phasor[trace.valid]).abs().max()),
                2e-9,
            )
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()


if __name__ == "__main__":
    unittest.main()
