import unittest
from pathlib import Path

import numpy as np
import torch

from lens_metrics_core import compute_distortion_grid, load_lens


class TestLegacyDistortionGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.lens = load_lens(repo_root / "eye_image_glass.xlsx", device=torch.device("cpu"))

    def test_trace_grid_is_twice_display_grid(self):
        result = compute_distortion_grid(
            self.lens,
            fov_x_deg=2.0,
            fov_y_deg=2.0,
            field_num=3,
            display_grid_num=5,
            distortion_type="rotating_eye_far",
        )
        self.assertEqual(result["metadata"]["trace_grid_num"], 10)
        self.assertEqual(result["grids"]["regular"].shape, (5, 5, 2))
        self.assertEqual(result["grids"]["magnification"].shape, (5, 5))
        self.assertEqual(result["grids"]["distorted"].shape, (5, 5, 2))
        self.assertEqual(result["metadata"]["grid_reference_mode"], "trace_affine_jacobian")
        matrix = np.asarray(result["metadata"]["grid_reference_matrix_2x2"], dtype=float)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertTrue(np.isfinite(matrix).all())

    def test_near_grid_uses_affine_corrected_coordinates(self):
        result = compute_distortion_grid(
            self.lens,
            fov_x_deg=2.0,
            fov_y_deg=2.0,
            field_num=3,
            display_grid_num=5,
            distortion_type="rotating_eye_near",
        )
        rows = [r for r in result["rows"] if r["valid"]]
        row = rows[0]
        height_obj = np.hypot(float(row["object_x_actual_mm"]), float(row["object_y_actual_mm"]))
        height_corrected = np.hypot(float(row["object_x_affine_corrected"]), float(row["object_y_affine_corrected"]))
        self.assertAlmostEqual(float(row["magnification"]), height_corrected / height_obj, places=8)
        self.assertEqual(result["metadata"]["grid_reference_mode"], "trace_affine_jacobian")
        self.assertIn("affine-corrected", result["metadata"]["magnification_policy"])

    def test_original_grid_axis_bug_is_replicated_by_default(self):
        result = compute_distortion_grid(
            self.lens,
            fov_x_deg=2.0,
            fov_y_deg=2.0,
            field_num=3,
            display_grid_num=5,
            distortion_type="rotating_eye_far",
        )
        self.assertEqual(result["metadata"]["original_grid_axis_bug_policy"], "replicated_ry_for_x_and_y_boundary")


if __name__ == "__main__":
    unittest.main()
