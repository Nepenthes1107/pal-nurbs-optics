from pathlib import Path

import numpy as np

from biot.domain import Device, PowerAstigmatismRequest, ResultStatus, SystemConfig
from biot.services import compute_power_astigmatism


def test_power_service_generates_cli_named_outputs():
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "results" / "test_services_power"
    cfg = SystemConfig(
        excel_path=repo_root / "eye_image_glass.xlsx",
        object_distance_mm=float("inf"),
        device=Device.CPU,
    )
    req = PowerAstigmatismRequest(
        system=cfg,
        fov_deg=2.0,
        field_num=3,
        lens_fov_deg=5.0,
        output_dir=output_dir,
    )

    result = compute_power_astigmatism(req)

    assert result.status == ResultStatus.SUCCEEDED, result.error
    assert result.table_data is not None
    assert np.isfinite(result.table_data[:, result.table_columns.index("theta_deg")]).all()
    assert result.metadata["power_evaluation_mode"] == "averfang_footprint_sampled"
    assert (output_dir / "power_astigmatism_curve.csv").exists()
    assert (output_dir / "power_astigmatism_curve.npy").exists()
    assert (output_dir / "power_astigmatism_curve.png").exists()
    assert (output_dir / "trace_power_astigmatism_curve.csv").exists()
    assert (output_dir / "trace_power_astigmatism_curve.npy").exists()
    assert (output_dir / "trace_power_astigmatism_curve.png").exists()
    assert (output_dir / "manifest.json").exists()
