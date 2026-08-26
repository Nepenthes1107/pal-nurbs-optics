from __future__ import annotations

import time
import traceback
from typing import Callable

import numpy as np

from biot.domain import (
    CancelToken,
    DistortionCurveRequest,
    DistortionCurveResult,
    DistortionGridRequest,
    DistortionGridResult,
    ProgressEvent,
    ResultStatus,
)
from biot.infra.result_store import save_manifest
from lens_metrics_core import (
    compute_distortion_curve as compute_distortion_curve_core,
    compute_distortion_grid as compute_distortion_grid_core,
    load_lens,
    resolve_device as resolve_lens_metrics_device,
    save_distortion_curve_outputs,
    save_distortion_grid_outputs,
)

ProgressCallback = Callable[[ProgressEvent], None]


def _emit(progress: ProgressCallback | None, request_id: str, phase: str, current: int, total: int, message: str = "") -> None:
    if progress is not None:
        progress(ProgressEvent(phase=phase, current=current, total=total, message=message, request_id=request_id))


def _table_metrics(payload: dict) -> dict:
    metadata = dict(payload.get("metadata", {}))
    rows = list(payload.get("rows", []))
    return {
        "valid_count": int(metadata.get("valid_count", 0)),
        "invalid_count": int(metadata.get("invalid_count", 0)),
        "row_count": int(len(rows)),
        "reference_strategy": metadata.get("magnification_reference_policy", metadata.get("grid_reference_mode", "")),
        "compatibility_deviation": metadata.get("compatibility_deviation", ""),
    }


def _load_request_lens(req: DistortionCurveRequest | DistortionGridRequest):
    device = resolve_lens_metrics_device(req.system.device.value)
    return load_lens(
        req.system.excel_path,
        device=device,
        fov_deg=req.lens_fov_deg,
        aperture_mm=req.aperture_mm,
        wavelength_nm=req.wavelength_nm,
    )


def compute_distortion_curve(
    req: DistortionCurveRequest,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> DistortionCurveResult:
    """Compute a one-dimensional distortion curve through `lens_metrics_core`.

    Inputs use degree for field angles and mm for distances. Returned table
    arrays are CPU NumPy data; ray tracing device support is inherited from
    Lensdata. GUI/file output is detached and does not preserve autograd.
    """

    started = time.perf_counter()
    cancel = cancel or CancelToken()
    result = DistortionCurveResult(
        request_id=req.request_id,
        request_snapshot=req.to_dict(),
        status=ResultStatus.RUNNING,
        output_dir=req.output_dir,
    )
    try:
        if cancel.is_cancelled():
            result.status = ResultStatus.CANCELLED
            return result
        _emit(progress, req.request_id, "load_lens", 0, 3, str(req.system.excel_path))
        lens = _load_request_lens(req)
        if cancel.is_cancelled():
            result.status = ResultStatus.CANCELLED
            return result

        _emit(progress, req.request_id, "compute_distortion_curve", 1, 3, req.distortion_type.value)
        payload = compute_distortion_curve_core(
            lens,
            fov_deg=req.fov_deg,
            field_num=req.field_num,
            axis=req.axis,
            distortion_type=req.distortion_type.value,
            wavelength_nm=req.wavelength_nm,
            near_object_distance_mm=req.near_object_distance_mm,
            pupil_distance_mm=req.pupil_distance_mm,
            lens_front_index=req.system.lens_front_index,
            lens_back_index=req.system.lens_back_index,
        )
        result.table_columns = list(payload["columns"])
        result.table_rows = list(payload["rows"])
        result.metadata = dict(payload["metadata"])
        result.table_data = np.asarray(payload["data"], dtype=np.float64)
        result.metrics = _table_metrics(payload)

        if req.output_dir is not None:
            _emit(progress, req.request_id, "save_distortion_curve", 2, 3, str(req.output_dir))
            paths = save_distortion_curve_outputs(payload, req.output_dir)
            result.artifacts = {f"distortion_curve_{name}": path for name, path in paths.items()}
            manifest_path = save_manifest(result, req.output_dir)
            result.artifacts["manifest_json"] = manifest_path

        result.status = ResultStatus.SUCCEEDED
        _emit(progress, req.request_id, "done", 3, 3)
        return result
    except Exception:
        result.status = ResultStatus.FAILED
        result.error = traceback.format_exc()
        return result
    finally:
        result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result.duration_seconds = float(time.perf_counter() - started)


def compute_distortion_grid(
    req: DistortionGridRequest,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> DistortionGridResult:
    """Compute a two-dimensional distortion grid through `lens_metrics_core`.

    Inputs use degree for field angles and mm for distances. Grid coordinate
    unit is recorded in `metadata["grid_coordinate_unit"]`. Returned arrays are
    CPU NumPy data and do not preserve autograd.
    """

    started = time.perf_counter()
    cancel = cancel or CancelToken()
    result = DistortionGridResult(
        request_id=req.request_id,
        request_snapshot=req.to_dict(),
        status=ResultStatus.RUNNING,
        output_dir=req.output_dir,
    )
    try:
        if cancel.is_cancelled():
            result.status = ResultStatus.CANCELLED
            return result
        _emit(progress, req.request_id, "load_lens", 0, 3, str(req.system.excel_path))
        lens = _load_request_lens(req)
        if cancel.is_cancelled():
            result.status = ResultStatus.CANCELLED
            return result

        _emit(progress, req.request_id, "compute_distortion_grid", 1, 3, req.distortion_type.value)
        payload = compute_distortion_grid_core(
            lens,
            fov_x_deg=req.fov_x_deg,
            fov_y_deg=req.fov_y_deg,
            field_num=req.field_num,
            display_grid_num=req.display_grid_num,
            distortion_type=req.distortion_type.value,
            wavelength_nm=req.wavelength_nm,
            near_object_distance_mm=req.near_object_distance_mm,
            pupil_distance_mm=req.pupil_distance_mm,
            lens_front_index=req.system.lens_front_index,
            lens_back_index=req.system.lens_back_index,
            fix_original_grid_axis_bug=req.fix_original_grid_axis_bug,
        )
        grids = payload["grids"]
        result.table_columns = list(payload["columns"])
        result.table_rows = list(payload["rows"])
        result.metadata = dict(payload["metadata"])
        result.table_data = np.asarray(payload["data"], dtype=np.float64)
        result.regular_grid = np.asarray(grids["regular"], dtype=np.float64)
        result.distorted_grid = np.asarray(grids["distorted"], dtype=np.float64)
        result.magnification_grid = np.asarray(grids["magnification"], dtype=np.float64)
        result.metrics = _table_metrics(payload)

        if req.output_dir is not None:
            _emit(progress, req.request_id, "save_distortion_grid", 2, 3, str(req.output_dir))
            paths = save_distortion_grid_outputs(payload, req.output_dir)
            result.artifacts = {f"distortion_grid_{name}": path for name, path in paths.items()}
            manifest_path = save_manifest(result, req.output_dir)
            result.artifacts["manifest_json"] = manifest_path

        result.status = ResultStatus.SUCCEEDED
        _emit(progress, req.request_id, "done", 3, 3)
        return result
    except Exception:
        result.status = ResultStatus.FAILED
        result.error = traceback.format_exc()
        return result
    finally:
        result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result.duration_seconds = float(time.perf_counter() - started)
