import csv
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


class TestCLISmoke(unittest.TestCase):
    def test_multi_rays_cli_smoke_psf_only(self):
        repo_root = Path(__file__).resolve().parents[1]
        output_dir = repo_root / "results" / "test_cli_smoke_psf_only"

        if output_dir.exists():
            shutil.rmtree(output_dir)

        cmd = [
            sys.executable,
            "multi_rays.py",
            "eye_image_glass.xlsx",
            "inf",
            "0",
            "0",
            "--cutoff",
            "100",
            "--np",
            "32",
            "--ni",
            "64",
            "--device",
            "cpu",
            "--output",
            str(output_dir),
        ]

        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )

        if result.returncode != 0:
            self.fail(
                "CLI smoke failed.\n"
                f"stdout:\n{result.stdout}\n\n"
                f"stderr:\n{result.stderr}"
            )

        expected_files = [
            output_dir / "psf_data.npy",
            output_dir / "psf_data.xlsx",
            output_dir / "psf_image.png",
            output_dir / "psf_metrics.csv",
        ]
        for file_path in expected_files:
            self.assertTrue(file_path.exists(), f"Missing output file: {file_path}")

        with open(output_dir / "psf_metrics.csv", newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        self.assertAlmostEqual(float(row["energy_sum"]), 1.0, places=6)
        self.assertEqual(row.get("mtf_enabled"), "False")
        self.assertFalse((output_dir / "mtf_curve.csv").exists())
        self.assertFalse((output_dir / "mtf_curve.xlsx").exists())
        self.assertFalse((output_dir / "mtf_curve.png").exists())
        self.assertFalse((output_dir / "mtf_metrics.csv").exists())

    def test_multi_rays_cli_smoke_with_mtf(self):
        repo_root = Path(__file__).resolve().parents[1]
        output_dir = repo_root / "results" / "test_cli_smoke_with_mtf"

        if output_dir.exists():
            shutil.rmtree(output_dir)

        cmd = [
            sys.executable,
            "multi_rays.py",
            "eye_image_glass.xlsx",
            "inf",
            "0",
            "0",
            "--cutoff",
            "100",
            "--np",
            "32",
            "--ni",
            "64",
            "--device",
            "cpu",
            "--with-mtf",
            "--output",
            str(output_dir),
        ]

        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )

        if result.returncode != 0:
            self.fail(
                "CLI smoke with MTF failed.\n"
                f"stdout:\n{result.stdout}\n\n"
                f"stderr:\n{result.stderr}"
            )

        expected_files = [
            output_dir / "psf_data.npy",
            output_dir / "psf_data.xlsx",
            output_dir / "psf_image.png",
            output_dir / "psf_metrics.csv",
            output_dir / "mtf_curve.csv",
            output_dir / "mtf_curve.xlsx",
            output_dir / "mtf_curve.png",
            output_dir / "mtf_metrics.csv",
        ]
        for file_path in expected_files:
            self.assertTrue(file_path.exists(), f"Missing output file: {file_path}")

        with open(output_dir / "psf_metrics.csv", newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        self.assertAlmostEqual(float(row["energy_sum"]), 1.0, places=6)
        self.assertEqual(row.get("mtf_enabled"), "True")

        with open(output_dir / "mtf_metrics.csv", newline="", encoding="utf-8-sig") as f:
            mtf_row = next(csv.DictReader(f))
        self.assertTrue(float(mtf_row["MTF_Sagittal_At_Cutoff"]) >= 0.0)
        self.assertTrue(float(mtf_row["MTF_Tangential_At_Cutoff"]) >= 0.0)
        self.assertEqual(float(mtf_row["MTF_Export_Max_Frequency_CyclesPerMM"]), 100.0)

        with open(output_dir / "mtf_curve.csv", newline="", encoding="utf-8-sig") as f:
            curve_rows = list(csv.DictReader(f))
        self.assertGreater(len(curve_rows), 0)
        max_freq = max(float(row["frequency_cycles_per_mm"]) for row in curve_rows)
        self.assertLessEqual(max_freq, 100.0)

    def test_multi_rays_cli_legacy_pupil_phase_smoke(self):
        repo_root = Path(__file__).resolve().parents[1]
        output_dir = repo_root / "results" / "test_cli_legacy_pupil_phase"

        if output_dir.exists():
            shutil.rmtree(output_dir)

        cmd = [
            sys.executable,
            "multi_rays.py",
            "eye_image_glass.xlsx",
            "inf",
            "0",
            "0",
            "--cutoff",
            "100",
            "--np",
            "32",
            "--ni",
            "64",
            "--device",
            "cpu",
            "--legacy-pupil-phase",
            "--output",
            str(output_dir),
        ]

        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )

        if result.returncode != 0:
            self.fail(
                "CLI legacy pupil phase failed.\n"
                f"stdout:\n{result.stdout}\n\n"
                f"stderr:\n{result.stderr}"
            )

        self.assertTrue((output_dir / "psf_data.npy").exists())
        with open(output_dir / "psf_metrics.csv", newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        self.assertEqual(row.get("legacy_pupil_phase"), "True")
        self.assertAlmostEqual(float(row["energy_sum"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
