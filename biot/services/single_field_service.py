from __future__ import annotations

import os
import sys
import time
import traceback
from math import isinf
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from openpyxl import load_workbook
from openpyxl.worksheet import _reader as _openpyxl_ws_reader

_openpyxl_ws_reader._cast_number = lambda value: float(value)

from biot.domain import CancelToken, ProgressEvent, ResultStatus, SingleFieldRequest, SingleFieldResult, SystemConfig
from biot.infra.field_mapping import field_angles_to_cb_excel_tilts
from biot.infra.result_store import save_manifest
from mtf_utils import generate_mtf_curve_and_metrics, save_mtf_outputs
from optics import Lensdata

from .visualization_utils import (
    convolve_chart_with_psf,
    default_chart_path,
    load_chart_xlsx,
    save_display_png,
    save_psf_png,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ProgressCallback = Callable[[ProgressEvent], None]


def _is_cuda_runtime_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "cuda" in text or "cublas" in text or "cudnn" in text


def resolve_device(device_pref: str) -> torch.device:
    """Resolve a torch device from `auto/cpu/cuda`.

    Units: none. This function may initialize CUDA but does not create tensors
    used in autograd-enabled optical computations.
    """

    if device_pref == "cpu":
        return torch.device("cpu")

    if device_pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        torch.zeros(1, device="cuda:0")
        return torch.device("cuda:0")

    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda:0")
            return torch.device("cuda:0")
        except Exception as exc:
            print(f"[警告] CUDA 可见但不可用，自动回退 CPU: {exc}")
            return torch.device("cpu")
    return torch.device("cpu")


def modify_excel_config(base_excel, output_path, obj_distance, field_x, field_y):
    """Write a temporary Excel config for one object distance and field.

    Units:
    - obj_distance: mm, or "Infinity".
    - field_x/field_y: degree.
    - Excel H7/I7 are CoordinateBreak tilt_x/tilt_y in degree. The physical
      field axes map crosswise: requested X field is written to I7, requested Y
      field is written to H7.
    This helper preserves the existing CLI behavior and writes only the
    caller-provided output path; it never edits the source Excel in place.

    Uses zipfile + XML editing so that literal ``Infinity`` cells in
    gradient-index models (GRAD3) do not trigger openpyxl's ``_cast_number``
    bug.  Only three cells are touched; the rest of the workbook is copied
    byte-for-byte.
    """

    import re
    import zipfile

    base_excel = Path(base_excel)
    output_path = Path(output_path)

    h7_tilt_x, i7_tilt_y = field_angles_to_cb_excel_tilts(field_x, field_y)

    # --- build replacement cell XML fragments --------------------------------
    if obj_distance == "Infinity" or obj_distance == float("inf"):
        b3_cell = '<c r="B3" t="inlineStr"><is><t>Infinity</t></is></c>'
    else:
        b3_cell = f'<c r="B3"><v>{float(obj_distance):.10g}</v></c>'

    h7_cell = f'<c r="H7"><v>{h7_tilt_x:.10g}</v></c>'
    i7_cell = f'<c r="I7"><v>{i7_tilt_y:.10g}</v></c>'

    edits = {"B3": b3_cell, "H7": h7_cell, "I7": i7_cell}

    # --- copy workbook, patching the sheet XML in-place --------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_excel) as zin:
        names = zin.namelist()
        sheet_name = next(n for n in names if n.startswith("xl/worksheets/sheet"))
        payload = {n: zin.read(n) for n in names}

    xml = payload[sheet_name].decode("utf-8")
    for ref, cell in edits.items():
        pattern = re.compile(
            rf'<c r="{ref}"(?:\s[^>]*)?(?:/>|>.*?</c>)', re.DOTALL
        )
        if pattern.search(xml):
            xml = pattern.sub(cell, xml, count=1)
        else:
            # Cell tag missing (unlikely for B3/H7/I7): insert at row start.
            row_match = re.match(r"[A-Z]+(\d+)", ref)
            if row_match:
                row_num = row_match.group(1)
                row_pat = re.compile(
                    rf'(<row r="{row_num}"(?:\s[^>]*)?>)', re.DOTALL
                )
                xml = row_pat.sub(rf"\1{cell}", xml, count=1)
    payload[sheet_name] = xml.encode("utf-8")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in payload.items():
            zout.writestr(name, data)

    print(f"[配置修改] 物距={obj_distance}, 视场=({field_x}, {field_y})")
    print(f"[CB旋转写入] H7 tilt_x={h7_tilt_x}, I7 tilt_y={i7_tilt_y}")
    print(f"[配置保存] {output_path}")


def _compute_psf_once(
    excel_path: str,
    field_x: float,
    field_y: float,
    cutoff_freq: float,
    n_p: int,
    n_i: int,
    device: torch.device,
    legacy_pupil_phase: bool = False,
    zernike_n_max: int = 5,
):
    """Compute one energy-normalized FFT PSF using the existing Lensdata path.

    Units:
    - excel_path uses the project Excel schema with lengths in mm.
    - field_x/field_y are degree.
    - cutoff_freq is cycles/mm.
    - n_p/n_i are unitless sampling counts.
    - d_delta returned by Lensdata is mm/pixel.
    Supports CPU/GPU through `device`; returned arrays are detached numpy data
    from the existing core path and are not autograd-preserving.
    """

    lens = Lensdata(device=device)
    ## 仅作为光阑索引不可用时的回退：真实缩放半径由 Lensdata.stop_semi_diameter()
    ## 从光阑面读取。这里不能当成物理光瞳半径——GRAD3 光阑半直径是 1.5。
    lens.aperture = 2.0
    lens.view_type = "angle"
    lens.FOV = 10
    lens.wavelengths = torch.tensor([555.0], device=device)
    lens.wavelengths_center = torch.tensor([555.0], device=device)
    lens.aimming = True
    lens.load_file(Path(excel_path), extension=".xlsx")

    hx = field_x / lens.FOV
    hy = field_y / lens.FOV

    pupils, d_delta = lens.fft_psf_i(
        n_p,
        n_i,
        lens.wavelengths_center,
        d_delta=0,
        Hx=hx,
        Hy=hy,
        legacy_pupil_phase=legacy_pupil_phase,
        zernike_n_max=zernike_n_max,
    )
    psf_image = lens._compute_psf(pupils, n_i, d_delta, cutoff_freq, display_size_um=None, methods="fft")
    metrics = dict(getattr(lens, "last_pupil_tilt_metrics", {}))
    metrics.update(getattr(lens, "last_wavefront_zernike_metrics", {}))
    coefficients = list(getattr(lens, "last_wavefront_zernike_coefficients", []))
    return psf_image, d_delta, metrics, coefficients


def get_psf_health_metrics(psf_image: np.ndarray) -> dict:
    """Return finite/non-negative/energy/centroid PSF metrics.

    Input PSF is a 2D intensity array [H, W]. Pixel offsets are reported in
    image pixels. This CPU numpy helper has no autograd support.
    """

    finite = bool(np.isfinite(psf_image).all())
    non_negative = bool((psf_image >= 0).all())
    psf_sum = float(psf_image.sum())
    peak = np.unravel_index(np.argmax(psf_image), psf_image.shape)
    y_indices, x_indices = np.indices(psf_image.shape)
    if psf_sum > 0.0:
        centroid_y = float((y_indices * psf_image).sum() / psf_sum)
        centroid_x = float((x_indices * psf_image).sum() / psf_sum)
    else:
        centroid_y = float("nan")
        centroid_x = float("nan")
    center_y = (psf_image.shape[0] - 1) / 2.0
    center_x = (psf_image.shape[1] - 1) / 2.0
    edge_width = min(8, psf_image.shape[0] // 2, psf_image.shape[1] // 2)
    edge_mask = np.zeros(psf_image.shape, dtype=bool)
    edge_mask[:edge_width, :] = True
    edge_mask[-edge_width:, :] = True
    edge_mask[:, :edge_width] = True
    edge_mask[:, -edge_width:] = True
    edge_energy_fraction = float(psf_image[edge_mask].sum() / psf_sum) if psf_sum > 0.0 else float("nan")
    metrics = {
        "shape_h": int(psf_image.shape[0]),
        "shape_w": int(psf_image.shape[1]),
        "finite": finite,
        "non_negative": non_negative,
        "energy_sum": psf_sum,
        "max": float(psf_image.max()),
        "min": float(psf_image.min()),
        "peak_y": int(peak[0]),
        "peak_x": int(peak[1]),
        "peak_offset_y_px": float(peak[0] - center_y),
        "peak_offset_x_px": float(peak[1] - center_x),
        "centroid_y": centroid_y,
        "centroid_x": centroid_x,
        "centroid_offset_y_px": float(centroid_y - center_y),
        "centroid_offset_x_px": float(centroid_x - center_x),
        "edge_energy_fraction_8px": edge_energy_fraction,
    }
    if not finite or not non_negative:
        raise ValueError(f"PSF 数值异常: finite={finite}, non_negative={non_negative}")
    return metrics


def save_psf_outputs(
    psf_image: np.ndarray,
    d_delta,
    n_i: int,
    output_dir: Path,
    metrics: dict,
    zernike_coefficients: list[dict] | None = None,
    mtf_enabled: bool = False,
    mtf_paths: dict | None = None,
) -> dict[str, Path]:
    """Save PSF arrays, display image, and metrics using legacy file names."""

    output_dir.mkdir(exist_ok=True, parents=True)

    psf_npy_path = output_dir / "psf_data.npy"
    np.save(psf_npy_path, psf_image)

    psf_excel_path = output_dir / "psf_data.xlsx"
    pd.DataFrame(psf_image).to_excel(psf_excel_path, index=False, header=False)

    if torch.is_tensor(d_delta):
        d_delta = d_delta.item()
    d_delta = float(d_delta)

    psf_img_path = output_dir / "psf_image.png"
    save_psf_png(psf_image, d_delta, psf_img_path)

    metrics_path = output_dir / "psf_metrics.csv"
    zernike_path = output_dir / "zernike_coefficients.csv" if zernike_coefficients is not None else None
    zernike_zemax_path = None
    if zernike_path is not None:
        frame = pd.DataFrame(zernike_coefficients)
        # 主文件保持按 ansi_j 升序的既有顺序，不改列名与列序。
        frame.to_csv(zernike_path, index=False)
        # Task 6：按 Zemax 的一基 Noll 序号另存一份，便于与报表逐行对照。
        if "noll_j" in frame.columns:
            zernike_zemax_path = output_dir / "zernike_coefficients_zemax.csv"
            frame.sort_values("noll_j").to_csv(zernike_zemax_path, index=False)
    metrics_with_files = dict(metrics)
    mtf_paths = mtf_paths or {}
    metrics_with_files.update(
        {
            "d_delta_mm": d_delta,
            "output_npy_exists": psf_npy_path.exists(),
            "output_excel_exists": psf_excel_path.exists(),
            "output_png_exists": psf_img_path.exists(),
            "zernike_coefficients_csv_exists": bool(zernike_path and zernike_path.exists()),
            "zernike_coefficients_zemax_csv_exists": bool(
                zernike_zemax_path and zernike_zemax_path.exists()
            ),
            "mtf_enabled": bool(mtf_enabled),
            "mtf_curve_csv_exists": bool(mtf_paths.get("curve_csv") and mtf_paths["curve_csv"].exists()),
            "mtf_curve_xlsx_exists": bool(mtf_paths.get("curve_xlsx") and mtf_paths["curve_xlsx"].exists()),
            "mtf_curve_png_exists": bool(mtf_paths.get("curve_png") and mtf_paths["curve_png"].exists()),
            "mtf_metrics_csv_exists": bool(mtf_paths.get("metrics_csv") and mtf_paths["metrics_csv"].exists()),
        }
    )
    pd.DataFrame([metrics_with_files]).to_csv(metrics_path, index=False)

    paths = {
        "psf_npy": psf_npy_path,
        "psf_xlsx": psf_excel_path,
        "psf_png": psf_img_path,
        "psf_metrics_csv": metrics_path,
    }
    if zernike_path is not None:
        paths["zernike_coefficients_csv"] = zernike_path
    if zernike_zemax_path is not None:
        paths["zernike_coefficients_zemax_csv"] = zernike_zemax_path

    print(f"[已保存] PSF 数据: {psf_npy_path}")
    print(f"[已保存] PSF Excel: {psf_excel_path}")
    print(f"[已保存] PSF 图像: {psf_img_path}")
    if mtf_enabled and mtf_paths:
        print(f"[已保存] MTF 曲线 CSV: {mtf_paths['curve_csv']}")
        print(f"[已保存] MTF 曲线 XLSX: {mtf_paths['curve_xlsx']}")
        print(f"[已保存] MTF 曲线图: {mtf_paths['curve_png']}")
        print(f"[已保存] MTF 指标: {mtf_paths['metrics_csv']}")
    print(f"[已保存] PSF 指标: {metrics_path}")
    if "zernike_coefficients_csv" in paths:
        print(f"[已保存] 波前 Zernike 系数: {paths['zernike_coefficients_csv']}")
    if "zernike_coefficients_zemax_csv" in paths:
        print(f"[已保存] 波前 Zernike 系数(Noll 序): {paths['zernike_coefficients_zemax_csv']}")

    paths.update({f"mtf_{name}": path for name, path in mtf_paths.items()})
    return paths


def _emit(progress: ProgressCallback | None, req: SingleFieldRequest, phase: str, current: int, total: int, message: str = ""):
    if progress is not None:
        progress(ProgressEvent(phase=phase, current=current, total=total, message=message, request_id=req.request_id))


def compute_single_field(
    req: SingleFieldRequest,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> SingleFieldResult:
    """Compute one PSF/MTF result from a `SingleFieldRequest`.

    Units:
    - Field angles are degree.
    - cutoff is cycles/mm.
    - d_delta output is mm/pixel.
    - PSF output shape is [ni_image, ni_image].
    GPU is supported through `req.system.device`; returned numpy arrays are not
    autograd-preserving. Failures are captured in the returned result.
    """

    started = time.perf_counter()
    result = SingleFieldResult(
        request_id=req.request_id,
        request_snapshot=req.to_dict(),
        status=ResultStatus.RUNNING,
        output_dir=req.output_dir,
    )
    cancel = cancel or CancelToken()

    try:
        if cancel.is_cancelled():
            result.status = ResultStatus.CANCELLED
            result.finished_at = result.started_at
            return result

        _emit(progress, req, "resolve_device", 0, 4)
        print(f"Python 解释器: {sys.executable}")
        print(f"请求设备: {req.system.device.value}")
        device = resolve_device(req.system.device.value)
        print(f"初选设备: {device}")

        excel_for_compute = req.system.excel_path
        prepared_config_path: Path | None = None
        if req.system.write_temp_excel and req.output_dir:
            prepared_config_path = Path(req.output_dir) / "compute_config.xlsx"
            prepared_config_path.parent.mkdir(parents=True, exist_ok=True)
            obj_distance = "Infinity" if isinf(float(req.system.object_distance_mm)) else float(req.system.object_distance_mm)
            modify_excel_config(
                req.system.excel_path,
                prepared_config_path,
                obj_distance,
                req.field_x_deg,
                req.field_y_deg,
            )
            excel_for_compute = prepared_config_path

        _emit(progress, req, "compute_psf", 1, 4)
        try:
            psf_image, d_delta, pupil_metrics, zernike_coefficients = _compute_psf_once(
                excel_path=str(excel_for_compute),
                field_x=req.field_x_deg,
                field_y=req.field_y_deg,
                cutoff_freq=req.cutoff_cyc_per_mm,
                n_p=req.system.np_pupil,
                n_i=req.system.ni_image,
                device=device,
                legacy_pupil_phase=req.system.legacy_pupil_phase,
                zernike_n_max=req.system.zernike_n_max,
            )
            actual_device = device
        except RuntimeError as exc:
            if req.system.device.value == "auto" and device.type == "cuda" and _is_cuda_runtime_error(exc):
                print(f"[警告] CUDA 执行失败，自动回退 CPU: {exc}")
                actual_device = torch.device("cpu")
                psf_image, d_delta, pupil_metrics, zernike_coefficients = _compute_psf_once(
                    excel_path=str(excel_for_compute),
                    field_x=req.field_x_deg,
                    field_y=req.field_y_deg,
                    cutoff_freq=req.cutoff_cyc_per_mm,
                    n_p=req.system.np_pupil,
                    n_i=req.system.ni_image,
                    device=actual_device,
                    legacy_pupil_phase=req.system.legacy_pupil_phase,
                    zernike_n_max=req.system.zernike_n_max,
                )
            else:
                raise

        if cancel.is_cancelled():
            result.status = ResultStatus.CANCELLED
            result.psf = psf_image
            result.d_delta_mm = float(d_delta.item()) if torch.is_tensor(d_delta) else float(d_delta)
            return result

        print(f"实际设备: {actual_device}")
        _emit(progress, req, "metrics", 2, 4)
        metrics = get_psf_health_metrics(psf_image)
        metrics.update(pupil_metrics)
        print("[PSF 健康检查]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        d_delta_mm = float(d_delta.item()) if torch.is_tensor(d_delta) else float(d_delta)
        mtf_paths = None
        mtf_metrics = None
        mtf_curve = None
        chart_image = None
        artifacts: dict[str, Path] = {}
        if req.output_dir:
            output_path = Path(req.output_dir)
            _emit(progress, req, "saving", 3, 4)
            if req.with_mtf:
                curve_df, _ = generate_mtf_curve_and_metrics(psf_image, d_delta_mm, req.cutoff_cyc_per_mm)
                mtf_curve = curve_df.to_numpy(dtype=np.float64)
                mtf_metrics, mtf_paths = save_mtf_outputs(
                    psf_image=psf_image,
                    d_delta_mm=d_delta_mm,
                    cutoff_freq=req.cutoff_cyc_per_mm,
                    output_dir=output_path,
                    curve_stem="mtf_curve",
                    metrics_filename="mtf_metrics.csv",
                    title="MTF (Positive Frequencies)",
                )
                print("[MTF 指标]")
                for k, v in mtf_metrics.items():
                    print(f"  {k}: {v}")
                metrics.update(
                    {
                        "mtf_dc": 1.0,
                        "mtf_at_cutoff_sagittal": float(mtf_metrics["MTF_Sagittal_At_Cutoff"]),
                        "mtf_at_cutoff_tangential": float(mtf_metrics["MTF_Tangential_At_Cutoff"]),
                    }
                )
            if req.with_chart_convolution:
                chart_path = req.chart_path or default_chart_path()
                chart = load_chart_xlsx(chart_path, psf_image.shape)
                chart_image = convolve_chart_with_psf(chart, psf_image)
                chart_npy = output_path / "chart_convolved.npy"
                np.save(chart_npy, chart_image)
                chart_artifacts = {
                    "chart_convolved_npy": chart_npy,
                    "chart_convolved_png": save_display_png(
                    chart_image,
                    output_path / "chart_convolved.png",
                    invert=False,
                    ),
                }
            else:
                chart_artifacts = {}
            artifacts = save_psf_outputs(
                psf_image=psf_image,
                d_delta=d_delta,
                n_i=req.system.ni_image,
                output_dir=output_path,
                metrics=metrics,
                zernike_coefficients=zernike_coefficients,
                mtf_enabled=req.with_mtf,
                mtf_paths=mtf_paths,
            )
            artifacts.update(chart_artifacts)
            if prepared_config_path is not None:
                artifacts["compute_config_xlsx"] = prepared_config_path
        elif req.with_mtf:
            print("[警告] 已启用 --with-mtf，但未指定 --output，不会写出 MTF 文件。")

        result.status = ResultStatus.SUCCEEDED
        result.psf = psf_image
        result.mtf_curve = mtf_curve
        result.chart_image = chart_image
        result.d_delta_mm = d_delta_mm
        result.metrics = metrics
        result.mtf_metrics = mtf_metrics
        result.artifacts = artifacts
        result.output_dir = req.output_dir
        if req.output_dir:
            manifest_path = save_manifest(result, Path(req.output_dir))
            result.artifacts["manifest_json"] = manifest_path
        _emit(progress, req, "done", 4, 4)
        return result
    except Exception:
        result.status = ResultStatus.FAILED
        result.error = traceback.format_exc()
        return result
    finally:
        result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result.duration_seconds = float(time.perf_counter() - started)


def calculate_psf(
    excel_path,
    field_x=0,
    field_y=0,
    cutoff_freq=100.0,
    n_p=256,
    n_i=512,
    output_dir=None,
    device_pref="auto",
    with_mtf=False,
    legacy_pupil_phase=False,
    zernike_n_max=5,
):
    """Legacy-compatible convenience wrapper used by `multi_rays.py`."""

    system = SystemConfig(
        excel_path=Path(excel_path),
        object_distance_mm=float("inf"),
        np_pupil=int(n_p),
        ni_image=int(n_i),
        device=device_pref,
        legacy_pupil_phase=bool(legacy_pupil_phase),
        zernike_n_max=int(zernike_n_max),
        write_temp_excel=False,
    )
    req = SingleFieldRequest(
        system=system,
        field_x_deg=float(field_x),
        field_y_deg=float(field_y),
        cutoff_cyc_per_mm=float(cutoff_freq),
        with_mtf=bool(with_mtf),
        output_dir=Path(output_dir) if output_dir else None,
    )
    result = compute_single_field(req)
    if result.status != ResultStatus.SUCCEEDED:
        raise RuntimeError(result.error or f"PSF calculation ended with status {result.status.value}")
    return result.psf, result.d_delta_mm
