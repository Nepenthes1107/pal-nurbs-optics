from __future__ import annotations

import time
import traceback
from typing import Callable

import numpy as np

from biot.domain import CancelToken, PowerAstigmatismRequest, PowerAstigmatismResult, ProgressEvent, ResultStatus
from biot.infra.result_store import save_manifest
from lens_metrics_core import (
    compute_power_astigmatism as compute_power_astigmatism_core,
    load_lens,
    resolve_device as resolve_lens_metrics_device,
    save_power_outputs,
)

ProgressCallback = Callable[[ProgressEvent], None]


def _emit(
    progress: ProgressCallback | None,
    req: PowerAstigmatismRequest,
    phase: str,
    current: int,
    total: int,
    message: str = "",
) -> None:
    if progress is not None:
        progress(ProgressEvent(phase=phase, current=current, total=total, message=message, request_id=req.request_id))


def _table_metrics(payload: dict) -> dict:
    metadata = dict(payload.get("metadata", {}))
    rows = list(payload.get("rows", []))
    return {
        "valid_count": int(metadata.get("valid_count", 0)),
        "invalid_count": int(metadata.get("invalid_count", 0)),
        "row_count": int(len(rows)),
        "reference_function": metadata.get("original_reference_function", ""),
        "power_evaluation_mode": metadata.get("power_evaluation_mode", ""),
    }


def compute_power_astigmatism(
    req: PowerAstigmatismRequest,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> PowerAstigmatismResult:
    """Compute power/astigmatism curves through `lens_metrics_core`.

    Inputs use mm for lengths, degree for field angles, nm for wavelength, and
    diopter for target focal power. Returned table arrays are CPU NumPy data.
    GPU support is inherited from `lens_metrics_core.load_lens`; returned data
    is detached for GUI/file output and does not preserve autograd.
    """

    started = time.perf_counter()
    cancel = cancel or CancelToken()
    result = PowerAstigmatismResult(
        request_id=req.request_id,
        request_snapshot=req.to_dict(),
        status=ResultStatus.RUNNING,
        output_dir=req.output_dir,
    )
    try:
        if cancel.is_cancelled():
            result.status = ResultStatus.CANCELLED
            return result

        _emit(progress, req, "load_lens", 0, 3, str(req.system.excel_path))
        device = resolve_lens_metrics_device(req.system.device.value)
        lens = load_lens(
            req.system.excel_path,
            device=device,
            fov_deg=req.lens_fov_deg,
            aperture_mm=req.aperture_mm,
            wavelength_nm=req.wavelength_nm,
        )
        if cancel.is_cancelled():
            result.status = ResultStatus.CANCELLED
            return result

        _emit(progress, req, "compute_power", 1, 3, f"field_num={req.field_num}")
        payload = compute_power_astigmatism_core(
            lens,
            fov_deg=req.fov_deg,
            field_num=req.field_num,
            axis=req.axis,
            wavelength_nm=req.wavelength_nm,
            differential_aperture_mm=req.differential_aperture_mm,
            focal_power_D=req.target_focal_power_d,
            crib_diameter_mm=req.averfang_crib_diameter_mm,
        )
        result.table_columns = list(payload["columns"])
        result.table_rows = list(payload["rows"])
        result.metadata = dict(payload["metadata"])
        result.table_data = np.asarray(payload["data"], dtype=np.float64)
        result.metrics = _table_metrics(payload)

        if req.output_dir is not None:
            _emit(progress, req, "save_power", 2, 3, str(req.output_dir))
            paths = save_power_outputs(payload, req.output_dir)
            result.artifacts = {f"power_{name}": path for name, path in paths.items()}
            manifest_path = save_manifest(result, req.output_dir)
            result.artifacts["manifest_json"] = manifest_path

        result.status = ResultStatus.SUCCEEDED
        _emit(progress, req, "done", 3, 3)
        return result
    except Exception:
        result.status = ResultStatus.FAILED
        result.error = traceback.format_exc()
        return result
    finally:
        result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result.duration_seconds = float(time.perf_counter() - started)
