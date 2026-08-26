from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any


def load_result_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load one service result manifest.

    The manifest stores BIOT service metadata only. Physical units are those
    recorded by the originating request/result schema, for example mm, degree,
    nm, diopter, or cycles/mm. This helper is CPU-only, does not create tensors,
    and has no autograd behavior.
    """

    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_manifest_path"] = str(path)
    return data


def infer_result_type(manifest: dict[str, Any]) -> str:
    """Infer a result family from a manifest dictionary."""

    artifacts = manifest.get("artifacts", {}) or {}
    artifact_keys = set(artifacts)
    if "field_grid" in manifest or any(key.startswith("field_") for key in artifact_keys):
        return "sweep"
    if "d_delta_mm" in manifest or "psf_npy" in artifact_keys:
        return "single_field"
    if any(key.startswith("power_") for key in artifact_keys):
        return "power_astigmatism"
    if any(key.startswith("distortion_curve_") for key in artifact_keys):
        return "distortion_curve"
    if any(key.startswith("distortion_grid_") for key in artifact_keys):
        return "distortion_grid"
    return "unknown"


def summarize_result(manifest_path: Path) -> dict[str, Any]:
    """Return a compact, table-friendly summary for one manifest."""

    data = load_result_manifest(manifest_path)
    artifacts = data.get("artifacts", {}) or {}
    metrics = data.get("metrics", {}) or {}
    output_dir = data.get("output_dir") or str(Path(manifest_path).parent)
    return {
        "result_type": infer_result_type(data),
        "status": data.get("status", ""),
        "request_id": data.get("request_id", ""),
        "finished_at": data.get("finished_at", ""),
        "duration_seconds": data.get("duration_seconds", 0.0),
        "output_dir": output_dir,
        "artifact_count": len(artifacts),
        "metric_count": len(metrics),
        "manifest_path": str(Path(manifest_path)),
    }


def list_results(root_dir: Path) -> list[dict[str, Any]]:
    """Find result manifests recursively under `root_dir`.

    Returned paths are filesystem paths only; no artifact arrays are loaded.
    """

    root = Path(root_dir)
    if not root.exists():
        return []
    summaries = [summarize_result(path) for path in sorted(root.rglob("manifest.json"))]
    summaries.sort(key=lambda item: str(item.get("finished_at", "")), reverse=True)
    return summaries


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(child_prefix, child, out)
    elif isinstance(value, list):
        out[prefix] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        out[prefix] = value


def _flatten_dict(value: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    _flatten("", value, out)
    return out


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def diff_results(left_manifest_path: Path, right_manifest_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Compare two saved result manifests.

    Numeric metrics get absolute deltas. Non-numeric request/artifact values
    are reported only when they differ. The function compares metadata and
    scalar metrics; it intentionally does not compare PSF/MTF arrays.
    """

    left = load_result_manifest(left_manifest_path)
    right = load_result_manifest(right_manifest_path)

    left_metrics = _flatten_dict(left.get("metrics", {}) or {})
    right_metrics = _flatten_dict(right.get("metrics", {}) or {})
    metric_rows: list[dict[str, Any]] = []
    for key in sorted(set(left_metrics) | set(right_metrics)):
        left_value = left_metrics.get(key)
        right_value = right_metrics.get(key)
        left_number = _as_float(left_value)
        right_number = _as_float(right_value)
        if left_number is not None and right_number is not None:
            metric_rows.append(
                {
                    "key": key,
                    "left": left_number,
                    "right": right_number,
                    "absolute_delta": abs(left_number - right_number),
                }
            )
        elif left_value != right_value:
            metric_rows.append(
                {
                    "key": key,
                    "left": left_value,
                    "right": right_value,
                    "absolute_delta": "",
                }
            )

    left_request = _flatten_dict(left.get("request_snapshot", {}) or {})
    right_request = _flatten_dict(right.get("request_snapshot", {}) or {})
    request_rows = [
        {"key": key, "left": left_request.get(key), "right": right_request.get(key)}
        for key in sorted(set(left_request) | set(right_request))
        if left_request.get(key) != right_request.get(key)
    ]

    left_artifacts = left.get("artifacts", {}) or {}
    right_artifacts = right.get("artifacts", {}) or {}
    artifact_rows = [
        {"key": key, "left": left_artifacts.get(key), "right": right_artifacts.get(key)}
        for key in sorted(set(left_artifacts) | set(right_artifacts))
        if left_artifacts.get(key) != right_artifacts.get(key)
    ]

    return {
        "metrics": metric_rows,
        "request": request_rows,
        "artifacts": artifact_rows,
    }


def _resolve_artifact_path(raw_path: str, manifest_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = manifest_dir / path
    if candidate.exists():
        return candidate
    return path


def export_result(manifest_path: Path, destination_dir: Path, include_artifacts: bool = True) -> Path:
    """Copy a result manifest and optionally its artifacts into a new folder.

    The export preserves original artifact bytes and writes an `export_manifest`
    file that maps artifact keys to copied relative paths. Directories are copied
    recursively. Existing exported files with the same names are replaced.
    """

    manifest_path = Path(manifest_path)
    manifest = load_result_manifest(manifest_path)
    result_name = manifest_path.parent.name or manifest.get("request_id") or "result"
    target = Path(destination_dir) / result_name
    if target.resolve() == manifest_path.parent.resolve():
        target = Path(destination_dir) / f"{result_name}_export"
    target.mkdir(parents=True, exist_ok=True)

    copied: dict[str, str] = {"manifest_json": "manifest.json"}
    shutil.copy2(manifest_path, target / "manifest.json")

    if include_artifacts:
        artifact_root = target / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        for key, raw_path in (manifest.get("artifacts", {}) or {}).items():
            if key == "manifest_json":
                continue
            source = _resolve_artifact_path(str(raw_path), manifest_path.parent)
            if not source.exists():
                continue
            safe_key = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in key)
            destination = artifact_root / f"{safe_key}_{source.name}"
            if source.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
            copied[key] = str(destination.relative_to(target))

    export_manifest = {
        "source_manifest": str(manifest_path),
        "result_type": infer_result_type(manifest),
        "request_id": manifest.get("request_id", ""),
        "copied_artifacts": copied,
    }
    (target / "export_manifest.json").write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
