from __future__ import annotations

import hashlib
import json
import shutil
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from biot.domain import CancelToken, ProgressEvent, ResultStatus, SingleFieldRequest, SweepRequest, SweepResult
from biot.infra.result_store import save_manifest

from .single_field_service import compute_single_field, save_psf_outputs
from .visualization_utils import (
    convolve_chart_with_psf,
    default_chart_path,
    load_chart_xlsx,
    save_field_stitch_png,
    save_mtf_value_grid_png,
)

ProgressCallback = Callable[[ProgressEvent], None]


def generate_field_values(min_deg: float, max_deg: float, step_deg: float) -> list[float]:
    """Generate inclusive field samples in degree.

    Inputs and outputs are in degree. This helper is NumPy-only and has no GPU
    or autograd behavior.
    """

    min_deg = float(min_deg)
    max_deg = float(max_deg)
    step_deg = float(step_deg)
    if step_deg <= 0.0:
        raise ValueError("field step must be positive.")
    if max_deg < min_deg:
        raise ValueError("field max must be greater than or equal to field min.")

    count = int(round((max_deg - min_deg) / step_deg))
    values = min_deg + np.arange(count + 1, dtype=np.float64) * step_deg
    if not np.isclose(values[-1], max_deg):
        raise ValueError("field range must be exactly divisible by field step.")
    return [float(v) for v in values]


def generate_sweep_grid(req: SweepRequest) -> list[tuple[float, float]]:
    """Generate an X-major/Y-major field grid in degree."""

    x_values = generate_field_values(req.field_x_min_deg, req.field_x_max_deg, req.field_x_step_deg)
    y_values = generate_field_values(req.field_y_min_deg, req.field_y_max_deg, req.field_y_step_deg)
    return [(x, y) for y in y_values for x in x_values]


def _cache_root() -> Path:
    return Path.cwd() / ".biot_cache" / "sweep"


def _point_signature(req: SweepRequest, field_x: float, field_y: float) -> str:
    snapshot = req.to_dict()
    snapshot.pop("request_id", None)
    snapshot.pop("created_at", None)
    snapshot["output_dir"] = ""
    snapshot["field_x_deg"] = float(field_x)
    snapshot["field_y_deg"] = float(field_y)
    text = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_cache(cache_dir: Path, point_result) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if point_result.psf is not None:
        np.save(cache_dir / "psf.npy", point_result.psf)
    (cache_dir / "metrics.json").write_text(
        json.dumps(point_result.metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (cache_dir / "d_delta_mm.txt").write_text(str(point_result.d_delta_mm or ""), encoding="utf-8")


def _read_cache(cache_dir: Path) -> tuple[np.ndarray, dict, float | None]:
    psf_path = cache_dir / "psf.npy"
    metrics_path = cache_dir / "metrics.json"
    if not psf_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("cache entry is incomplete")
    psf = np.load(psf_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    d_delta_text = (cache_dir / "d_delta_mm.txt").read_text(encoding="utf-8").strip()
    d_delta = float(d_delta_text) if d_delta_text else None
    return psf, metrics, d_delta


def build_stitched_psf(
    point_images: dict[tuple[float, float], np.ndarray],
    x_values: list[float],
    y_values: list[float],
) -> np.ndarray:
    """Build a display-preview mosaic from per-field PSF arrays.

    The input PSFs are physical energy-normalized arrays. The output is a
    display preview only; each tile is peak-normalized for visibility and must
    not be used for MTF or physical comparison.
    """

    if not point_images:
        return np.zeros((0, 0), dtype=np.float64)
    sample = next(iter(point_images.values()))
    tile_h, tile_w = sample.shape
    mosaic = np.zeros((len(y_values) * tile_h, len(x_values) * tile_w), dtype=np.float64)
    for row, y in enumerate(y_values):
        for col, x in enumerate(x_values):
            psf = point_images.get((float(x), float(y)))
            if psf is None:
                continue
            tile = np.asarray(psf, dtype=np.float64)
            peak = float(np.nanmax(tile))
            if peak > 0.0:
                tile = tile / peak
            r0 = row * tile_h
            c0 = col * tile_w
            mosaic[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile
    return mosaic


def build_stitched_chart(
    point_images: dict[tuple[float, float], np.ndarray],
    x_values: list[float],
    y_values: list[float],
    chart_path: Path,
) -> np.ndarray:
    """Build a display-only chart convolution mosaic from per-field PSFs.

    Field coordinates are in degree. Each tile uses an energy-normalized PSF
    convolved with the same chart image. Output is display-normalized and must
    not be used for MTF or physical PSF metrics.
    """

    if not point_images:
        return np.zeros((0, 0), dtype=np.float64)
    sample = next(iter(point_images.values()))
    tile_h, tile_w = sample.shape
    chart = load_chart_xlsx(chart_path, sample.shape)
    mosaic = np.zeros((len(y_values) * tile_h, len(x_values) * tile_w), dtype=np.float64)
    for row, y in enumerate(y_values):
        for col, x in enumerate(x_values):
            psf = point_images.get((float(x), float(y)))
            if psf is None:
                continue
            tile = convolve_chart_with_psf(chart, psf)
            r0 = row * tile_h
            c0 = col * tile_w
            mosaic[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile
    return mosaic


def build_mtf_grid(
    rows: list[dict],
    x_values: list[float],
    y_values: list[float],
) -> np.ndarray:
    """Build a [Ny, Nx, 2] cutoff MTF grid.

    The last dimension is [sagittal, tangential]. Values are unitless,
    DC-normalized MTF samples at the request cutoff frequency.
    """

    grid = np.full((len(y_values), len(x_values), 2), np.nan, dtype=np.float64)
    x_index = {float(x): i for i, x in enumerate(x_values)}
    y_index = {float(y): i for i, y in enumerate(y_values)}
    for row in rows:
        if row.get("status") != ResultStatus.SUCCEEDED.value:
            continue
        x = float(row["field_x_deg"])
        y = float(row["field_y_deg"])
        if x not in x_index or y not in y_index:
            continue
        sag = row.get("mtf_at_cutoff_sagittal")
        tan = row.get("mtf_at_cutoff_tangential")
        if sag is None or tan is None:
            continue
        grid[y_index[y], x_index[x], 0] = float(sag)
        grid[y_index[y], x_index[x], 1] = float(tan)
    return grid


def _emit(progress: ProgressCallback | None, req: SweepRequest, phase: str, current: int, total: int, message: str = ""):
    if progress is not None:
        progress(ProgressEvent(phase=phase, current=current, total=total, message=message, request_id=req.request_id))


def compute_sweep(
    req: SweepRequest,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> SweepResult:
    """Compute a field-grid PSF/MTF sweep through the single-field service.

    Units:
    - Field angles are degree.
    - cutoff is cycles/mm.
    - Per-point PSFs are energy-normalized 2D intensity arrays.
    GPU support is inherited from `compute_single_field`; returned arrays are
    CPU NumPy data and do not preserve autograd.
    """

    started = time.perf_counter()
    cancel = cancel or CancelToken()
    result = SweepResult(
        request_id=req.request_id,
        request_snapshot=req.to_dict(),
        status=ResultStatus.RUNNING,
        output_dir=req.output_dir,
    )

    try:
        grid = generate_sweep_grid(req)
        x_values = generate_field_values(req.field_x_min_deg, req.field_x_max_deg, req.field_x_step_deg)
        y_values = generate_field_values(req.field_y_min_deg, req.field_y_max_deg, req.field_y_step_deg)
        total = len(grid)
        if total > 2000:
            raise ValueError(f"sweep point count exceeds 2000: {total}")

        result.field_grid = grid
        output_root = Path(req.output_dir) if req.output_dir else None
        if output_root:
            output_root.mkdir(parents=True, exist_ok=True)

        point_images: dict[tuple[float, float], np.ndarray] = {}
        rows: list[dict] = []
        cache_hits = 0
        failures = 0

        for index, (field_x, field_y) in enumerate(grid, start=1):
            if cancel.is_cancelled():
                result.status = ResultStatus.CANCELLED
                break

            _emit(progress, req, "sweep", index - 1, total, f"field=({field_x:g}, {field_y:g})")
            point_dir = output_root / f"field_x_{field_x:g}_y_{field_y:g}" if output_root else None
            signature = _point_signature(req, field_x, field_y)
            cache_dir = _cache_root() / signature

            try:
                cache_loaded = False
                needs_mtf = bool(req.with_mtf or req.with_mtf_grid)
                if req.use_cache and not needs_mtf and cache_dir.exists():
                    try:
                        psf, metrics, d_delta_mm = _read_cache(cache_dir)
                        cache_loaded = True
                        cache_hits += 1
                        if point_dir is not None:
                            artifacts = save_psf_outputs(
                                psf_image=psf,
                                d_delta=d_delta_mm or metrics.get("d_delta_mm", 0.0),
                                n_i=req.system.ni_image,
                                output_dir=point_dir,
                                metrics=metrics,
                                mtf_enabled=False,
                            )
                        else:
                            artifacts = {}
                    except Exception:
                        cache_loaded = False

                if not cache_loaded:
                    point_req = SingleFieldRequest(
                        system=replace(req.system, write_temp_excel=True),
                        field_x_deg=field_x,
                        field_y_deg=field_y,
                        cutoff_cyc_per_mm=req.cutoff_cyc_per_mm,
                        with_mtf=needs_mtf,
                        output_dir=point_dir,
                    )
                    point_result = compute_single_field(point_req, cancel=cancel)
                    if point_result.status != ResultStatus.SUCCEEDED:
                        failures += 1
                        rows.append(
                            {
                                "field_x_deg": field_x,
                                "field_y_deg": field_y,
                                "status": point_result.status.value,
                                "error": point_result.error or "",
                            }
                        )
                        continue
                    psf = np.asarray(point_result.psf, dtype=np.float64)
                    metrics = dict(point_result.metrics)
                    artifacts = dict(point_result.artifacts)
                    d_delta_mm = point_result.d_delta_mm
                    if req.use_cache:
                        _write_cache(cache_dir, point_result)

                point_images[(float(field_x), float(field_y))] = psf
                row = {
                    "field_x_deg": field_x,
                    "field_y_deg": field_y,
                    "status": ResultStatus.SUCCEEDED.value,
                    "cache_hit": bool(cache_loaded),
                    "d_delta_mm": d_delta_mm,
                }
                row.update(metrics)
                rows.append(row)
                result.point_metrics.append(row)
                if point_dir is not None:
                    result.artifacts[f"field_{index}_dir"] = point_dir
                    for name, path in artifacts.items():
                        result.artifacts[f"field_{index}_{name}"] = path
            except Exception as exc:
                failures += 1
                rows.append(
                    {
                        "field_x_deg": field_x,
                        "field_y_deg": field_y,
                        "status": ResultStatus.FAILED.value,
                        "error": str(exc),
                    }
                )

        result.stitched_psf = build_stitched_psf(point_images, x_values, y_values)
        if req.with_chart_stitch:
            result.stitched_chart = build_stitched_chart(
                point_images,
                x_values,
                y_values,
                req.chart_path or default_chart_path(),
            )
        if req.with_mtf_grid:
            result.mtf_grid = build_mtf_grid(rows, x_values, y_values)
        result.metrics = {
            "total_points": int(total),
            "completed_points": int(len(point_images)),
            "failed_points": int(failures),
            "cache_hits": int(cache_hits),
            "x_count": int(len(x_values)),
            "y_count": int(len(y_values)),
            "chart_stitch_enabled": bool(req.with_chart_stitch),
            "mtf_grid_enabled": bool(req.with_mtf_grid),
        }

        if result.status != ResultStatus.CANCELLED:
            result.status = ResultStatus.SUCCEEDED if failures == 0 else ResultStatus.FAILED

        if output_root:
            summary_path = output_root / "sweep_summary.csv"
            pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
            result.artifacts["sweep_summary_csv"] = summary_path
            if result.stitched_psf is not None and result.stitched_psf.size:
                stitched_path = output_root / "stitched_psf_preview.npy"
                np.save(stitched_path, result.stitched_psf)
                result.artifacts["stitched_psf_preview_npy"] = stitched_path
                result.artifacts["stitched_psf_preview_png"] = save_field_stitch_png(
                    result.stitched_psf,
                    np.asarray(x_values, dtype=np.float64),
                    np.asarray(y_values, dtype=np.float64),
                    output_root / "stitched_psf_preview.png",
                    kind="psf",
                )
            if result.stitched_chart is not None and result.stitched_chart.size:
                chart_path = output_root / "stitched_chart_preview.npy"
                np.save(chart_path, result.stitched_chart)
                result.artifacts["stitched_chart_preview_npy"] = chart_path
                result.artifacts["stitched_chart_preview_png"] = save_field_stitch_png(
                    result.stitched_chart,
                    np.asarray(x_values, dtype=np.float64),
                    np.asarray(y_values, dtype=np.float64),
                    output_root / "stitched_chart_preview.png",
                    kind="chart",
                )
            if result.mtf_grid is not None and result.mtf_grid.size:
                mtf_grid_path = output_root / "mtf_cutoff_grid.npy"
                np.save(mtf_grid_path, result.mtf_grid)
                result.artifacts["mtf_cutoff_grid_npy"] = mtf_grid_path
                mtf_rows = []
                for y_i, y in enumerate(y_values):
                    for x_i, x in enumerate(x_values):
                        mtf_rows.append(
                            {
                                "field_x_deg": x,
                                "field_y_deg": y,
                                "mtf_sagittal_at_cutoff": result.mtf_grid[y_i, x_i, 0],
                                "mtf_tangential_at_cutoff": result.mtf_grid[y_i, x_i, 1],
                            }
                        )
                mtf_grid_csv = output_root / "mtf_cutoff_grid.csv"
                pd.DataFrame(mtf_rows).to_csv(mtf_grid_csv, index=False, encoding="utf-8-sig")
                result.artifacts["mtf_cutoff_grid_csv"] = mtf_grid_csv
                result.artifacts["mtf_cutoff_grid_png"] = save_mtf_value_grid_png(
                    result.mtf_grid,
                    np.asarray(x_values, dtype=np.float64),
                    np.asarray(y_values, dtype=np.float64),
                    output_root / "mtf_cutoff_grid.png",
                )
            manifest_path = save_manifest(result, output_root)
            result.artifacts["manifest_json"] = manifest_path

        _emit(progress, req, "done", total, total)
        return result
    except Exception:
        result.status = ResultStatus.FAILED
        result.error = traceback.format_exc()
        return result
    finally:
        result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result.duration_seconds = float(time.perf_counter() - started)


def clear_sweep_cache() -> None:
    """Remove the local sweep cache under .biot_cache/sweep."""

    root = _cache_root()
    if root.exists():
        shutil.rmtree(root)
