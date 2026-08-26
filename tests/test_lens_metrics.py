import csv
import shutil
import unittest
from pathlib import Path

import numpy as np
import torch

from lens_metrics_core import (
    compute_distortion_curve,
    compute_distortion_grid,
    compute_power_astigmatism,
    extract_eye_positions,
    load_lens,
    sample_field_1d,
)
from lens_metrics import main as lens_metrics_main


class TestLensMetricsCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.lens = load_lens(repo_root / "eye_image_glass.xlsx", device=torch.device("cpu"))

    def test_tan_uniform_field_sampling(self):
        samples = sample_field_1d(10.0, 5, axis="y", tan_uniform=True)
        self.assertEqual(samples["theta_deg"].shape, (5,))
        self.assertAlmostEqual(float(samples["theta_deg"][0]), -10.0, places=7)
        self.assertAlmostEqual(float(samples["theta_deg"][-1]), 10.0, places=7)
        np.testing.assert_allclose(samples["field_x_deg"], 0.0)

    def test_eye_positions_from_cb_and_aperture_vertices(self):
        eye = extract_eye_positions(self.lens)
        self.assertEqual(eye.cb_index, 3)
        self.assertEqual(eye.aperture_index, 7)
        self.assertAlmostEqual(eye.eye_center_z_mm, 29.3, places=6)
        self.assertAlmostEqual(eye.pupil_z_mm, 20.92, places=6)

    def test_power_output_fields_are_finite_and_valid(self):
        result = compute_power_astigmatism(self.lens, fov_deg=2.0, field_num=3)
        columns = result["columns"]
        self.assertIn("local_sagittal_power_D", columns)
        self.assertIn("local_astigmatism_D", columns)
        self.assertAlmostEqual(result["data"][0, columns.index("theta_deg")], 0.0, places=12)
        self.assertEqual(result["metadata"]["initial_view_field_x_deg"], 0.0)
        self.assertEqual(result["metadata"]["initial_view_field_y_deg"], 0.0)
        self.assertEqual(result["metadata"]["power_evaluation_mode"], "averfang_footprint_sampled")
        self.assertIn("trace_result", result)
        valid = result["data"][:, columns.index("valid")]
        self.assertTrue((valid == 1.0).any())
        numeric = result["data"][:, : columns.index("valid")]
        self.assertTrue(np.isfinite(numeric).all())

    def test_distortion_curve_records_reference_index(self):
        result = compute_distortion_curve(self.lens, fov_deg=2.0, field_num=3)
        columns = result["columns"]
        self.assertIn("reference_index", columns)
        self.assertAlmostEqual(result["data"][0, columns.index("theta_deg")], 0.0, places=12)
        self.assertEqual(result["metadata"]["initial_view_field_x_deg"], 0.0)
        self.assertEqual(result["metadata"]["initial_view_field_y_deg"], 0.0)
        ref = result["data"][:, columns.index("reference_index")]
        self.assertTrue((ref == -1.0).all())
        self.assertEqual(result["metadata"]["far_reference_mode"], "trace_paraxial_slope")
        valid = result["data"][:, columns.index("valid")]
        self.assertTrue((valid == 1.0).any())

    def test_distortion_grid_shapes_match_display_grid(self):
        result = compute_distortion_grid(self.lens, fov_x_deg=2.0, fov_y_deg=2.0, field_num=3, display_grid_num=5)
        self.assertEqual(result["grids"]["regular"].shape, (5, 5, 2))
        self.assertEqual(result["grids"]["magnification"].shape, (5, 5))
        self.assertEqual(result["grids"]["distorted"].shape, (5, 5, 2))
        self.assertTrue(np.isfinite(result["grids"]["magnification"]).all())


class TestLensMetricsCLI(unittest.TestCase):
    def test_lens_metrics_all_cli_smoke(self):
        repo_root = Path(__file__).resolve().parents[1]
        output_dir = repo_root / "results" / "test_lens_metrics_all"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        cmd = [
            "lens_metrics.py",
            "eye_image_glass.xlsx",
            "all",
            "--fov",
            "2",
            "--field-num",
            "3",
            "--display-grid-num",
            "5",
            "--device",
            "cpu",
            "--output",
            str(output_dir),
        ]
        old_cwd = Path.cwd()
        try:
            import os

            os.chdir(repo_root)
            result_code = lens_metrics_main(cmd[1:])
        finally:
            os.chdir(old_cwd)
        if result_code != 0:
            self.fail("lens_metrics all failed")

        expected = [
            output_dir / "power" / "power_astigmatism_curve.csv",
            output_dir / "power" / "power_astigmatism_curve.npy",
            output_dir / "power" / "power_astigmatism_curve.png",
            output_dir / "power" / "trace_power_astigmatism_curve.csv",
            output_dir / "power" / "trace_power_astigmatism_curve.npy",
            output_dir / "power" / "trace_power_astigmatism_curve.png",
            output_dir / "distortion_curve" / "distortion_curve.csv",
            output_dir / "distortion_curve" / "distortion_curve.npy",
            output_dir / "distortion_curve" / "distortion_curve.png",
            output_dir / "distortion_grid" / "distortion_grid_samples.csv",
            output_dir / "distortion_grid" / "distortion_grid_regular.npy",
            output_dir / "distortion_grid" / "distortion_grid_magnification.npy",
            output_dir / "distortion_grid" / "distortion_grid_distorted.npy",
            output_dir / "distortion_grid" / "distortion_grid.png",
        ]
        for file_path in expected:
            self.assertTrue(file_path.exists(), f"Missing output file: {file_path}")

        with open(output_dir / "power" / "power_astigmatism_curve.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(any(row["valid"] == "True" for row in rows))


if __name__ == "__main__":
    unittest.main()
