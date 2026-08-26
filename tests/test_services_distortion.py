from pathlib import Path

import numpy as np

from biot.domain import (
    Device,
    DistortionCurveRequest,
    DistortionGridRequest,
    DistortionType,
    ResultStatus,
    SystemConfig,
)
from biot.services import compute_distortion_curve, compute_distortion_grid


def test_distortion_curve_service_records_trace_reference():
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "results" / "test_services_distortion_curve"
    cfg = SystemConfig(
        excel_path=repo_root / "eye_image_glass.xlsx",
        object_distance_mm=float("inf"),
        device=Device.CPU,
    )
    req = DistortionCurveRequest(
        system=cfg,
        fov_deg=2.0,
        field_num=3,
        lens_fov_deg=5.0,
        distortion_type=DistortionType.ROTATING_EYE_FAR,
        output_dir=output_dir,
    )

    result = compute_distortion_curve(req)

    assert result.status == ResultStatus.SUCCEEDED, result.error
    assert result.table_data is not None
    assert result.metadata["reference_index"] == -1
    assert result.metadata["far_reference_mode"] == "trace_paraxial_slope"
    assert (output_dir / "distortion_curve.csv").exists()
    assert (output_dir / "distortion_curve.npy").exists()
    assert (output_dir / "distortion_curve.png").exists()


def test_distortion_grid_service_outputs_affine_grids():
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "results" / "test_services_distortion_grid"
    cfg = SystemConfig(
        excel_path=repo_root / "eye_image_glass.xlsx",
        object_distance_mm=float("inf"),
        device=Device.CPU,
    )
    req = DistortionGridRequest(
        system=cfg,
        fov_x_deg=2.0,
        fov_y_deg=2.0,
        field_num=3,
        display_grid_num=3,
        lens_fov_deg=5.0,
        distortion_type=DistortionType.ROTATING_EYE_FAR,
        fix_original_grid_axis_bug=True,
        output_dir=output_dir,
    )

    result = compute_distortion_grid(req)

    assert result.status == ResultStatus.SUCCEEDED, result.error
    assert isinstance(result.regular_grid, np.ndarray)
    assert result.regular_grid.shape == (3, 3, 2)
    assert result.metadata["grid_reference_mode"] == "trace_affine_jacobian"
    assert result.metadata["original_grid_axis_bug_policy"] == "fixed"
    assert (output_dir / "distortion_grid_samples.csv").exists()
    assert (output_dir / "distortion_grid_regular.npy").exists()
    assert (output_dir / "distortion_grid_magnification.npy").exists()
    assert (output_dir / "distortion_grid_distorted.npy").exists()
    assert (output_dir / "distortion_grid.png").exists()
