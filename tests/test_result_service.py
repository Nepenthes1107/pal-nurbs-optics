import json
from pathlib import Path

from biot.services.result_service import diff_results, export_result, list_results, summarize_result


def _write_manifest(root: Path, name: str, *, metric_value: float, field_x: float = 0.0) -> Path:
    result_dir = root / name
    result_dir.mkdir(parents=True)
    artifact = result_dir / "psf_data.npy"
    artifact.write_bytes(b"npy")
    manifest = {
        "schema_version": "1.0",
        "request_id": name,
        "request_snapshot": {
            "field_x_deg": field_x,
            "field_y_deg": 0.0,
            "cutoff_cyc_per_mm": 100.0,
        },
        "status": "succeeded",
        "output_dir": str(result_dir),
        "finished_at": f"2026-05-09T00:00:0{int(metric_value)}Z",
        "duration_seconds": metric_value,
        "artifacts": {"psf_npy": str(artifact)},
        "metrics": {"energy_sum": metric_value, "finite": True},
        "d_delta_mm": 0.001,
    }
    path = result_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path


def test_result_service_lists_and_summarizes_manifests(tmp_path):
    manifest = _write_manifest(tmp_path, "single_a", metric_value=1.0)

    summaries = list_results(tmp_path)

    assert len(summaries) == 1
    assert summaries[0]["result_type"] == "single_field"
    assert summaries[0]["status"] == "succeeded"
    assert summarize_result(manifest)["artifact_count"] == 1


def test_result_service_diffs_numeric_metrics_and_requests(tmp_path):
    left = _write_manifest(tmp_path, "left", metric_value=1.0, field_x=0.0)
    right = _write_manifest(tmp_path, "right", metric_value=2.0, field_x=10.0)

    diff = diff_results(left, right)

    energy = next(row for row in diff["metrics"] if row["key"] == "energy_sum")
    assert energy["absolute_delta"] == 1.0
    assert any(row["key"] == "field_x_deg" for row in diff["request"])


def test_result_service_exports_manifest_and_artifacts(tmp_path):
    manifest = _write_manifest(tmp_path, "single_a", metric_value=1.0)
    export_root = tmp_path / "exports"

    target = export_result(manifest, export_root)

    assert (target / "manifest.json").exists()
    assert (target / "export_manifest.json").exists()
    copied = list((target / "artifacts").glob("*psf_data.npy"))
    assert len(copied) == 1
