import shutil
import unittest
from pathlib import Path

import numpy as np
from openpyxl import Workbook

from biot.domain import Device, ResultStatus, SweepRequest, SystemConfig
from biot.services import clear_sweep_cache, compute_sweep, generate_sweep_grid


class TestSweepService(unittest.TestCase):
    def setUp(self):
        clear_sweep_cache()

    def test_generate_sweep_grid(self):
        req = SweepRequest(
            system=SystemConfig(excel_path=Path("eye_image_glass.xlsx"), object_distance_mm=float("inf")),
            field_x_min_deg=-5,
            field_x_max_deg=5,
            field_x_step_deg=5,
            field_y_min_deg=0,
            field_y_max_deg=0,
            field_y_step_deg=5,
        )
        self.assertEqual(generate_sweep_grid(req), [(-5.0, 0.0), (0.0, 0.0), (5.0, 0.0)])

    def test_compute_sweep_single_point_and_cache(self):
        repo_root = Path(__file__).resolve().parents[1]
        output_dir = repo_root / "results" / "test_service_sweep"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        chart_path = output_dir / "chart_input.xlsx"
        workbook = Workbook()
        worksheet = workbook.active
        for row in range(1, 9):
            for column in range(1, 9):
                worksheet.cell(row=row, column=column, value=(row + column) % 2)
        workbook.save(chart_path)
        workbook.close()

        req = SweepRequest(
            system=SystemConfig(
                excel_path=repo_root / "eye_image_glass.xlsx",
                object_distance_mm=float("inf"),
                np_pupil=32,
                ni_image=64,
                device=Device.CPU,
            ),
            field_x_min_deg=0.0,
            field_x_max_deg=0.0,
            field_x_step_deg=5.0,
            field_y_min_deg=0.0,
            field_y_max_deg=0.0,
            field_y_step_deg=5.0,
            cutoff_cyc_per_mm=100.0,
            with_mtf=False,
            with_chart_stitch=True,
            with_mtf_grid=True,
            chart_path=chart_path,
            output_dir=output_dir,
            use_cache=True,
        )

        first = compute_sweep(req)
        self.assertEqual(first.status, ResultStatus.SUCCEEDED, first.error)
        self.assertEqual(first.metrics["completed_points"], 1)
        self.assertEqual(first.metrics["failed_points"], 0)
        self.assertTrue(np.isfinite(first.stitched_psf).all())
        self.assertIsNotNone(first.stitched_chart)
        self.assertTrue(np.isfinite(first.stitched_chart).all())
        self.assertIsNotNone(first.mtf_grid)
        self.assertEqual(first.mtf_grid.shape, (1, 1, 2))
        self.assertTrue((output_dir / "sweep_summary.csv").exists())
        self.assertTrue((output_dir / "stitched_chart_preview.npy").exists())
        self.assertTrue((output_dir / "stitched_chart_preview.png").exists())
        self.assertTrue((output_dir / "mtf_cutoff_grid.csv").exists())
        self.assertTrue((output_dir / "mtf_cutoff_grid.png").exists())
        self.assertTrue((output_dir / "manifest.json").exists())

        second = compute_sweep(req)
        self.assertEqual(second.status, ResultStatus.SUCCEEDED, second.error)
        self.assertEqual(second.metrics["cache_hits"], 0)


if __name__ == "__main__":
    unittest.main()
