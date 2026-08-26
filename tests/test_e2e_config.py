import unittest
from pathlib import Path

import torch

from biot.e2e import E2EConfig, as_torch_device_dtype, set_random_seed


class TestE2EConfig(unittest.TestCase):
    def test_defaults_reference_existing_project_inputs(self):
        cfg = E2EConfig()
        paths = cfg.existing_input_paths()

        self.assertEqual(paths["excel_path"], Path("eye_image_glass.xlsx"))
        self.assertIsNone(cfg.chart_path)
        self.assertNotIn("chart_path", paths)
        self.assertTrue(paths["excel_path"].is_file())

    def test_round_trip_serialization_preserves_units_and_paths(self):
        cfg = E2EConfig(
            excel_path=Path("eye_image_glass_grad3.xlsx"),
            chart_path=Path("chart_input.xlsx"),
            wavelength_nm=560.0,
            field_x_deg=3.0,
            field_y_deg=-2.0,
            pupil_radius_mm=1.5,
            psf_size_px=32,
            psf_pixel_pitch_mm=0.004,
            dtype="float32",
        )

        restored = E2EConfig.from_dict(cfg.to_dict())

        self.assertEqual(restored.excel_path, Path("eye_image_glass_grad3.xlsx"))
        self.assertEqual(restored.chart_path, Path("chart_input.xlsx"))
        self.assertEqual(restored.wavelength_nm, 560.0)
        self.assertEqual(restored.field_x_deg, 3.0)
        self.assertEqual(restored.field_y_deg, -2.0)
        self.assertEqual(restored.pupil_radius_mm, 1.5)
        self.assertEqual(restored.psf_size_px, 32)
        self.assertEqual(restored.psf_pixel_pitch_mm, 0.004)
        self.assertEqual(restored.dtype, "float32")

    def test_dtype_and_device_parser(self):
        device, dtype = as_torch_device_dtype("cpu", "float64")

        self.assertEqual(device, torch.device("cpu"))
        self.assertEqual(dtype, torch.float64)

    def test_invalid_dtype_raises(self):
        with self.assertRaises(ValueError):
            E2EConfig(dtype="float16")
        with self.assertRaises(ValueError):
            as_torch_device_dtype("cpu", "float16")

    def test_seed_makes_torch_random_reproducible(self):
        set_random_seed(1234)
        first = torch.rand(4)
        set_random_seed(1234)
        second = torch.rand(4)

        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
