import unittest
from pathlib import Path

import numpy as np
import torch

from lens_metrics_core import build_legacy_adapter, compute_trace_power_astigmatism, load_lens


class TestLegacyPower(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.lens = load_lens(repo_root / "eye_image_glass.xlsx", device=torch.device("cpu"))

    def test_default_eyeglass_surface_mapping(self):
        adapter = build_legacy_adapter(self.lens)
        self.assertEqual(adapter.lens_front_index, 1)
        self.assertEqual(adapter.lens_back_index, 2)
        self.assertAlmostEqual(adapter.eye.eye_center_z_mm, 29.3, places=6)
        self.assertAlmostEqual(adapter.eye.pupil_z_mm, 20.92, places=6)

    def test_power_uses_legacy_metadata_and_fields(self):
        result = compute_trace_power_astigmatism(self.lens, fov_deg=2.0, field_num=3)
        meta = result["metadata"]
        self.assertEqual(meta["compatibility_mode"], "EyeGlassSystem.SingleEyeLens")
        self.assertEqual(meta["original_reference_function"], "SingleEyeLens.eval_focal_power")
        self.assertEqual(meta["lens_front_index"], 1)
        self.assertEqual(meta["lens_back_index"], 2)
        self.assertIn("h_main_plane_mm", meta)
        self.assertEqual(meta["valid_count"], 3)

        columns = result["columns"]
        for name in ["fp_sagittal_D", "fp_meridian_D", "fp_mean_D", "astigmatism_D"]:
            self.assertIn(name, columns)
        theta = result["data"][:, columns.index("theta_deg")]
        self.assertEqual(theta[0], 0.0)
        self.assertGreater(meta["trace_first_field_y_deg"], 0.0)
        self.assertTrue(np.isfinite(result["data"][:, columns.index("fp_mean_D")]).all())


if __name__ == "__main__":
    unittest.main()
