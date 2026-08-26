import shutil
import unittest
from pathlib import Path

import numpy as np

from lens_metrics import main as lens_metrics_main


class TestLensMetricsOutputs(unittest.TestCase):
    def test_cli_all_writes_numeric_outputs_and_metadata(self):
        repo_root = Path(__file__).resolve().parents[1]
        output_dir = repo_root / "results" / "test_lens_metrics_outputs"
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
            "--field-ring-num",
            "8",
            "--pupil-ring-num",
            "8",
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
        self.assertEqual(result_code, 0)

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
            output_dir / "footprint" / "footprint_coverage.npy",
            output_dir / "footprint" / "footprint_coverage_metadata.json",
            output_dir / "footprint" / "footprint_coverage.png",
        ]
        for path in expected:
            self.assertTrue(path.exists(), f"Missing output file: {path}")

        payload = np.load(output_dir / "power" / "power_astigmatism_curve.npy", allow_pickle=True).item()
        self.assertEqual(payload["metadata"]["compatibility_mode"], "EyeGlassSystem.SingleEyeLens")
        self.assertEqual(payload["metadata"]["lens_front_index"], 1)
        self.assertEqual(payload["metadata"]["lens_back_index"], 2)
        self.assertEqual(payload["metadata"]["power_evaluation_mode"], "averfang_footprint_sampled")


if __name__ == "__main__":
    unittest.main()
