import unittest
from pathlib import Path

from biot.domain import (
    Device,
    DistortionCurveRequest,
    DistortionCurveResult,
    DistortionGridRequest,
    DistortionGridResult,
    DistortionType,
    LogEvent,
    PowerAstigmatismRequest,
    PowerAstigmatismResult,
    ProgressEvent,
    ResultStatus,
    SingleFieldRequest,
    SingleFieldResult,
    SweepRequest,
    SweepResult,
    SystemConfig,
)


class TestDomainSerialization(unittest.TestCase):
    def test_system_config_round_trip_preserves_units_and_inf(self):
        cfg = SystemConfig(
            excel_path=Path("eye_image_glass.xlsx"),
            object_distance_mm=float("inf"),
            np_pupil=32,
            ni_image=64,
            device=Device.CPU,
            legacy_pupil_phase=True,
        )

        restored = SystemConfig.from_dict(cfg.to_dict())

        self.assertEqual(restored.excel_path, Path("eye_image_glass.xlsx"))
        self.assertEqual(restored.object_distance_mm, float("inf"))
        self.assertEqual(restored.np_pupil, 32)
        self.assertEqual(restored.ni_image, 64)
        self.assertEqual(restored.device, Device.CPU)
        self.assertTrue(restored.legacy_pupil_phase)

    def test_single_field_request_round_trip(self):
        cfg = SystemConfig(excel_path=Path("eye_image_glass.xlsx"), object_distance_mm=1000.0)
        req = SingleFieldRequest(
            system=cfg,
            field_x_deg=10.0,
            field_y_deg=-5.0,
            cutoff_cyc_per_mm=100.0,
            with_mtf=True,
            output_dir=Path("results/test"),
            tag="roundtrip",
        )

        restored = SingleFieldRequest.from_dict(req.to_dict())

        self.assertEqual(restored.system.object_distance_mm, 1000.0)
        self.assertEqual(restored.field_x_deg, 10.0)
        self.assertEqual(restored.field_y_deg, -5.0)
        self.assertTrue(restored.with_mtf)
        self.assertEqual(restored.output_dir, Path("results/test"))
        self.assertEqual(restored.tag, "roundtrip")

    def test_single_field_result_round_trip_excludes_arrays(self):
        req = SingleFieldRequest(
            system=SystemConfig(excel_path=Path("eye_image_glass.xlsx"), object_distance_mm=float("inf")),
            field_x_deg=0.0,
            field_y_deg=0.0,
        )
        result = SingleFieldResult(
            request_id=req.request_id,
            request_snapshot=req.to_dict(),
            status=ResultStatus.SUCCEEDED,
            output_dir=Path("results/test"),
            artifacts={"psf_npy": Path("results/test/psf_data.npy")},
            metrics={"energy_sum": 1.0},
            d_delta_mm=0.001,
        )

        data = result.to_dict()
        restored = SingleFieldResult.from_dict(data)

        self.assertNotIn("psf", data)
        self.assertEqual(restored.status, ResultStatus.SUCCEEDED)
        self.assertEqual(restored.artifacts["psf_npy"], Path("results/test/psf_data.npy"))
        self.assertEqual(restored.metrics["energy_sum"], 1.0)

    def test_sweep_request_and_result_round_trip(self):
        cfg = SystemConfig(excel_path=Path("eye_image_glass.xlsx"), object_distance_mm=float("inf"))
        req = SweepRequest(
            system=cfg,
            field_x_min_deg=-5.0,
            field_x_max_deg=5.0,
            field_x_step_deg=5.0,
            field_y_min_deg=0.0,
            field_y_max_deg=0.0,
            field_y_step_deg=5.0,
            output_dir=Path("results/sweep"),
            use_cache=True,
        )
        restored_req = SweepRequest.from_dict(req.to_dict())
        self.assertEqual(restored_req.field_x_min_deg, -5.0)
        self.assertEqual(restored_req.field_x_max_deg, 5.0)
        self.assertEqual(restored_req.output_dir, Path("results/sweep"))

        result = SweepResult(
            request_id=req.request_id,
            request_snapshot=req.to_dict(),
            status=ResultStatus.SUCCEEDED,
            output_dir=Path("results/sweep"),
            artifacts={"sweep_summary_csv": Path("results/sweep/sweep_summary.csv")},
            metrics={"total_points": 3},
            field_grid=[(-5.0, 0.0), (0.0, 0.0), (5.0, 0.0)],
        )
        data = result.to_dict()
        restored_result = SweepResult.from_dict(data)
        self.assertNotIn("stitched_psf", data)
        self.assertEqual(restored_result.metrics["total_points"], 3)
        self.assertEqual(restored_result.artifacts["sweep_summary_csv"], Path("results/sweep/sweep_summary.csv"))

    def test_lens_metric_requests_and_results_round_trip(self):
        cfg = SystemConfig(excel_path=Path("eye_image_glass.xlsx"), object_distance_mm=float("inf"))
        power_req = PowerAstigmatismRequest(system=cfg, fov_deg=5.0, field_num=5, output_dir=Path("results/power"))
        curve_req = DistortionCurveRequest(
            system=cfg,
            fov_deg=5.0,
            field_num=5,
            distortion_type=DistortionType.ROTATING_EYE_FAR,
            output_dir=Path("results/curve"),
        )
        grid_req = DistortionGridRequest(
            system=cfg,
            fov_x_deg=5.0,
            fov_y_deg=5.0,
            field_num=5,
            display_grid_num=5,
            output_dir=Path("results/grid"),
        )

        self.assertEqual(PowerAstigmatismRequest.from_dict(power_req.to_dict()).field_num, 5)
        self.assertEqual(DistortionCurveRequest.from_dict(curve_req.to_dict()).distortion_type, DistortionType.ROTATING_EYE_FAR)
        self.assertEqual(DistortionGridRequest.from_dict(grid_req.to_dict()).display_grid_num, 5)

        power_result = PowerAstigmatismResult(
            request_id=power_req.request_id,
            request_snapshot=power_req.to_dict(),
            status=ResultStatus.SUCCEEDED,
            artifacts={"power_csv": Path("results/power/power_astigmatism_curve.csv")},
            metadata={"mode": "power"},
        )
        curve_result = DistortionCurveResult(
            request_id=curve_req.request_id,
            request_snapshot=curve_req.to_dict(),
            status=ResultStatus.SUCCEEDED,
            artifacts={"distortion_curve_csv": Path("results/curve/distortion_curve.csv")},
            metadata={"reference_index": -1},
        )
        grid_result = DistortionGridResult(
            request_id=grid_req.request_id,
            request_snapshot=grid_req.to_dict(),
            status=ResultStatus.SUCCEEDED,
            artifacts={"distortion_grid_csv": Path("results/grid/distortion_grid_samples.csv")},
            metadata={"grid_reference_mode": "trace_affine_jacobian"},
        )

        self.assertEqual(PowerAstigmatismResult.from_dict(power_result.to_dict()).metadata["mode"], "power")
        self.assertEqual(DistortionCurveResult.from_dict(curve_result.to_dict()).metadata["reference_index"], -1)
        self.assertEqual(
            DistortionGridResult.from_dict(grid_result.to_dict()).metadata["grid_reference_mode"],
            "trace_affine_jacobian",
        )

    def test_events_round_trip(self):
        progress = ProgressEvent(phase="compute", current=1, total=4, message="running")
        log = LogEvent(level="warning", message="check")

        self.assertEqual(ProgressEvent.from_dict(progress.to_dict()).phase, "compute")
        self.assertEqual(LogEvent.from_dict(log.to_dict()).level, "warning")

    def test_missing_required_field_raises(self):
        with self.assertRaises(ValueError):
            SingleFieldRequest.from_dict({"schema_version": "1.0"})

    def test_schema_mismatch_raises(self):
        cfg = SystemConfig(excel_path=Path("eye_image_glass.xlsx"), object_distance_mm=float("inf")).to_dict()
        cfg["schema_version"] = "0.1"
        with self.assertRaises(ValueError):
            SystemConfig.from_dict(cfg)


if __name__ == "__main__":
    unittest.main()
