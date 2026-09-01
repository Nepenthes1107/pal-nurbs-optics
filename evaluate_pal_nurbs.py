"""Fail-closed evaluation for a completed PAL-NURBS run.

The source run is read-only. Evaluation first seals six resumable HDF5 PSF
databases, then derives weighted-MTF mean maps, PSF stitches, and chart stitches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal import fftconvolve

from optics import compute_dc_normalized_mtf
from biot.e2e import pal_nurbs as pal


EVAL_SCHEMA = 3
PSF_DATABASE_SCHEMA = 1
STAGE_MANIFEST_SCHEMA = 1
FIELD_VALUES = tuple(float(value) for value in np.arange(-40.0, 40.0 + 0.1, 10.0))
FIELD_COUNT = len(FIELD_VALUES) ** 2
RAW_SIZE_PX = 512
RENDER_SIZE_PX = 130
CROP_PHYSICAL_SIZE_MM = 0.184378803949209
TILE_GAP_PX = 5
PSF_DISPLAY_SMOOTH_SIGMA = 2.0
FIGURE_DPI = 160
WEIGHTED_MTF_INTERPOLATED_RESOLUTION = 200
WEIGHTED_MTF_FIELD_INTERPOLATION = "cubic"
COMMON_FREQ = np.linspace(0.0, 100.0, 1000, dtype=np.float64)
CSF_MM_PER_DEG = 0.291
CSF_F0 = 4.1726
CSF_F1 = 1.3625
CSF_A = 0.8493
CSF_P = 0.7786
CSF_GAIN = 373.08


def _progress(**fields: Any) -> None:
    """Emit one immediately visible, machine-readable evaluation progress line."""
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[pal-eval] {payload}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_distance(value: Any) -> float | str:
    value = float(value)
    return "inf" if math.isinf(value) else value


def _load_config(run: Path, device: str) -> Any:
    saved = _json(run / "config.json")
    allowed = set(getattr(pal.MinimalConfig, "__dataclass_fields__", {}))
    values = {key: value for key, value in saved.items() if key in allowed}
    values["output"] = str(run)
    values["device"] = str(device)
    return pal.MinimalConfig(**values)


def _load_checkpoint(
    run: Path, summary: Mapping[str, Any], device: torch.device,
    *, checkpoint_stage: int | None, source_identity_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    if checkpoint_stage is not None:
        if checkpoint_stage not in (7, 11, 19):
            raise ValueError("checkpoint_stage must be one of 7, 11, or 19")
        stage_records = [
            record for record in summary.get("stages", [])
            if isinstance(record, Mapping)
            and int(record.get("control_count", -1)) == checkpoint_stage
        ]
        if len(stage_records) != 1:
            raise ValueError(
                f"summary does not contain exactly one completed {checkpoint_stage}x"
                f"{checkpoint_stage} stage"
            )
        candidate = run / f"stage_{checkpoint_stage}x{checkpoint_stage}" / "final.pt"
    elif hasattr(pal, "DISTANCE_SPECS"):
        candidate = run / "final.pt"
    else:
        count = int(summary.get("final_control_count", 0))
        if count <= 0:
            raise ValueError("summary lacks final_control_count")
        candidate = run / f"stage_{count}x{count}" / "final.pt"
    if not candidate.is_file():
        raise FileNotFoundError(f"final checkpoint not found: {candidate}")
    with candidate.open("rb") as handle:
        payload = torch.load(handle, map_location=device)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"malformed final checkpoint: {candidate}")
    expected_control_count = (
        checkpoint_stage
        if checkpoint_stage is not None
        else int(payload.get("control_count", summary.get("final_control_count", 0)))
    )
    if int(payload.get("control_count", -1)) != expected_control_count:
        raise ValueError(
            f"checkpoint control_count mismatch: expected {expected_control_count}, "
            f"got {payload.get('control_count')!r}"
        )
    if str(payload.get("identity_sha256", "")) != source_identity_sha256:
        raise ValueError("checkpoint identity does not match source run identity")
    return candidate, payload


def _make_module(
    config: Any, device: torch.device, control_count: int | None = None
) -> torch.nn.Module:
    if control_count is None:
        return pal.FixedWeightNURBSPerturbation(device=device, dtype=torch.float64)
    return pal.FixedWeightNURBSPerturbation(control_count, device=device, dtype=torch.float64)


def _distance_cases(config: Any) -> list[tuple[str, float, list[dict[str, Any]]]]:
    if hasattr(pal, "DISTANCE_SPECS"):
        specs = [(str(spec.label), float(spec.object_distance_mm)) for spec in pal.DISTANCE_SPECS]
    else:
        specs = [("D500", 500.0), ("D1000", 1000.0), ("Dinf", float("inf"))]
    result: list[tuple[str, float, list[dict[str, Any]]]] = []
    for label, distance in specs:
        cases: list[dict[str, Any]] = []
        for row, field_y in enumerate(FIELD_VALUES):
            for column, field_x in enumerate(FIELD_VALUES):
                case = {
                    "case_id": f"{label}_r{row:02d}_c{column:02d}",
                    "field_x_deg": field_x,
                    "field_y_deg": field_y,
                }
                if hasattr(pal, "DISTANCE_SPECS"):
                    case["distance_label"] = label
                else:
                    case["distance_mm"] = distance
                cases.append(case)
        result.append((label, distance, cases))
    return result


def _state_map(module: torch.nn.Module, state: Mapping[str, Any]) -> None:
    expected = module.state_dict()
    if set(expected) != set(state):
        raise ValueError("checkpoint state_dict keys do not match the evaluator module")
    module.load_state_dict(dict(state), strict=True)


def _plot_map(path: Path, values: np.ndarray, *, title: str, symmetric: bool = False) -> None:
    array = np.asarray(values, dtype=np.float64)
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    if symmetric:
        limit = max(float(np.nanmax(np.abs(array))), np.finfo(np.float64).eps)
        image = axis.imshow(
            array, origin="upper", extent=[-40, 40, -40, 40], cmap="coolwarm",
            vmin=-limit, vmax=limit,
        )
    else:
        image = axis.imshow(
            array, origin="upper", extent=[-40, 40, -40, 40], cmap="viridis"
        )
    axis.set(title=title, xlabel="field X (deg)", ylabel="field Y (deg)")
    figure.colorbar(image, ax=axis)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _interpolate_weighted_mtf_map(
    values: np.ndarray,
    field_x_deg: Sequence[float],
    field_y_deg: Sequence[float],
    *,
    resolution: int = WEIGHTED_MTF_INTERPOLATED_RESOLUTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cubic display interpolation over the sealed rectangular field grid.

    The returned grid uses ascending Y rows and covers only the native field
    domain. Missing/non-finite native samples are rejected rather than filled.
    """
    array = np.asarray(values, dtype=np.float64)
    x_native = np.asarray(field_x_deg, dtype=np.float64)
    y_native = np.asarray(field_y_deg, dtype=np.float64)
    if x_native.ndim != 1 or y_native.ndim != 1:
        raise ValueError("weighted-MTF field axes must be one-dimensional")
    if array.shape != (y_native.size, x_native.size):
        raise ValueError(
            "weighted-MTF field map shape does not match its axes: "
            f"map={array.shape}, axes={(y_native.size, x_native.size)}"
        )
    if x_native.size < 4 or y_native.size < 4:
        raise ValueError("cubic weighted-MTF field interpolation requires at least 4x4 nodes")
    if not np.isfinite(array).all():
        raise ValueError("weighted-MTF field map contains NaN/Inf; interpolation is forbidden")
    if bool((array < -1.0e-12).any()) or bool((array > 1.0 + 1.0e-12).any()):
        raise ValueError("weighted-MTF field map lies outside the physical [0,1] range")
    if not (np.diff(x_native) > 0.0).all() or not (np.diff(y_native) > 0.0).all():
        raise ValueError("weighted-MTF field axes must be strictly ascending")
    if isinstance(resolution, bool) or int(resolution) < 2:
        raise ValueError("weighted-MTF interpolation resolution must be at least 2")

    interpolator = RegularGridInterpolator(
        (y_native, x_native), array,
        method=WEIGHTED_MTF_FIELD_INTERPOLATION,
        bounds_error=True,
    )
    native_xx, native_yy = np.meshgrid(x_native, y_native)
    recovered = np.asarray(interpolator((native_yy, native_xx)), dtype=np.float64)
    node_error = float(np.max(np.abs(recovered - array)))
    if node_error > 1.0e-12:
        raise RuntimeError(
            f"weighted-MTF interpolation changed native nodes by {node_error:.3e}"
        )

    x_fine = np.linspace(x_native[0], x_native[-1], int(resolution), dtype=np.float64)
    y_fine = np.linspace(y_native[0], y_native[-1], int(resolution), dtype=np.float64)
    fine_xx, fine_yy = np.meshgrid(x_fine, y_fine)
    fine = np.asarray(interpolator((fine_yy, fine_xx)), dtype=np.float64)
    if fine.shape != (int(resolution), int(resolution)) or not np.isfinite(fine).all():
        raise ValueError("weighted-MTF field interpolation produced an invalid fine grid")
    return x_fine, y_fine, fine


def _plot_interpolated_weighted_mtf_map(
    path: Path,
    values: np.ndarray,
    field_x_deg: np.ndarray,
    field_y_deg: np.ndarray,
    *,
    title: str,
    symmetric: bool = False,
) -> None:
    """Render an already interpolated display map without further image resampling."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (field_y_deg.size, field_x_deg.size):
        raise ValueError("interpolated weighted-MTF map shape does not match its axes")
    if not np.isfinite(array).all():
        raise ValueError("interpolated weighted-MTF map contains NaN/Inf")
    figure, axis = plt.subplots(figsize=(8.4, 6.3), constrained_layout=True)
    image_options: dict[str, Any] = {"cmap": "viridis", "vmin": 0.0, "vmax": 1.0}
    colorbar_label = "weighted mean MTF"
    if symmetric:
        limit = max(float(np.max(np.abs(array))), np.finfo(np.float64).eps)
        image_options = {"cmap": "coolwarm", "vmin": -limit, "vmax": limit}
        colorbar_label = "delta weighted mean MTF"
    image = axis.imshow(
        array,
        origin="lower",
        extent=[
            float(field_x_deg[0]), float(field_x_deg[-1]),
            float(field_y_deg[0]), float(field_y_deg[-1]),
        ],
        interpolation="none",
        aspect="auto",
        **image_options,
    )
    x_ticks = np.arange(
        math.ceil(float(field_x_deg[0]) / 5.0) * 5.0,
        math.floor(float(field_x_deg[-1]) / 5.0) * 5.0 + 0.1,
        5.0,
    )
    y_ticks = np.arange(
        math.ceil(float(field_y_deg[0]) / 5.0) * 5.0,
        math.floor(float(field_y_deg[-1]) / 5.0) * 5.0 + 0.1,
        5.0,
    )
    axis.set_xticks(x_ticks)
    axis.set_yticks(y_ticks)
    axis.set_xlabel("field X (Degrees)", fontfamily="Times New Roman", fontsize=14)
    axis.set_ylabel("field Y (Degrees)", fontfamily="Times New Roman", fontsize=14)
    axis.set_title(f"{title} — interpolated", fontsize=13)
    axis.tick_params(direction="in", top=True, right=True, labelsize=11)
    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontfamily("Times New Roman")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label, fontfamily="Times New Roman", fontsize=14)
    colorbar.ax.tick_params(direction="in", labelsize=12)
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)


def _save_sag_and_averfang(
    root: Path, base_sag: torch.Tensor, module: torch.nn.Module, power_config: Any
) -> None:
    sag_dir = root / "sag"
    sag_dir.mkdir(parents=True, exist_ok=True)
    sag = base_sag.detach().cpu().numpy().astype(np.float64)
    coord = torch.linspace(
        -float(power_config.semi_diameter_mm), float(power_config.semi_diameter_mm),
        sag.shape[0], dtype=torch.float64, device=base_sag.device,
    )
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    delta = module.delta_raw(xx, yy).detach().cpu().numpy().astype(np.float64)
    for name, value in (("baseline", sag), ("optimized", sag + delta), ("delta", delta)):
        np.savez_compressed(
            sag_dir / f"{name}.npz", sag_mm=value, x_mm=coord.cpu().numpy(),
            physical_y_mm=coord.cpu().numpy()[::-1],
        )
        _plot_map(
            sag_dir / f"{name}.png", value * (1e6 if name == "delta" else 1.0),
            title=f"{name} sag ({'um' if name == 'delta' else 'mm'})",
            symmetric=name == "delta",
        )
    maps_dir = root / "averfang"
    maps_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for name, value in (
        ("baseline", base_sag),
        ("optimized", base_sag + torch.as_tensor(delta, device=base_sag.device)),
    ):
        computed = pal.torch_averfang_maps(value, power_config)
        outputs[name] = {
            key: computed[key].detach().cpu().numpy()
            for key in ("power_D", "A_D", "astigmatism_D")
        }
        np.savez_compressed(maps_dir / f"{name}.npz", **outputs[name])
    outputs["delta"] = {
        key: outputs["optimized"][key] - outputs["baseline"][key]
        for key in outputs["baseline"]
    }
    np.savez_compressed(maps_dir / "delta.npz", **outputs["delta"])
    for key, label in (("power_D", "power (D)"), ("astigmatism_D", "astigmatism (D)")):
        for name in ("baseline", "optimized", "delta"):
            _plot_map(
                maps_dir / f"{name}_{key}.png", outputs[name][key],
                title=f"{name} {label}", symmetric=name == "delta",
            )
    _write_json(
        maps_dir / "metadata.json",
        {"units": {"power_D": "D", "A_D": "D", "astigmatism_D": "D"},
         "source": "PAL torch_averfang_maps"},
    )


def _normalize_physical_psf(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 2-D array")
    if np.any(array < 0.0):
        raise ValueError(f"{name} contains negative energy")
    energy = float(array.sum())
    if not math.isfinite(energy) or energy <= 0.0:
        raise ValueError(f"{name} has invalid energy: {energy}")
    if abs(energy - 1.0) > 1.0e-10:
        raise ValueError(f"{name} energy is not normalized: {energy}")
    return array


def _normalize_psf_energy(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite, non-negative, and 2-D")
    energy = float(array.sum())
    if not math.isfinite(energy) or energy <= 0.0:
        raise ValueError(f"{name} has invalid energy: {energy}")
    normalized = array / energy
    if abs(float(normalized.sum()) - 1.0) > 1.0e-10:
        raise ValueError(f"{name} energy normalization failed")
    return normalized


def _render_psf(raw_psf: np.ndarray, raw_pixel_pitch_mm: float) -> tuple[np.ndarray, float]:
    raw = _normalize_physical_psf("raw PSF", raw_psf)
    pitch = float(raw_pixel_pitch_mm)
    if not math.isfinite(pitch) or pitch <= 0.0:
        raise ValueError(f"invalid raw pixel pitch: {pitch}")
    crop_size_px = max(1, round(CROP_PHYSICAL_SIZE_MM / pitch))
    height, width = raw.shape
    if crop_size_px > height or crop_size_px > width:
        raise ValueError(
            "raw PSF support is smaller than the required physical crop: "
            f"required={CROP_PHYSICAL_SIZE_MM:.15g} mm, "
            f"available={min(height, width) * pitch:.15g} mm"
        )
    center_x, center_y = (width + 1) / 2.0, (height + 1) / 2.0
    x_start = max(1, math.floor(center_x - crop_size_px / 2.0))
    y_start = max(1, math.floor(center_y - crop_size_px / 2.0))
    x_end = min(width, x_start + crop_size_px - 1)
    y_end = min(height, y_start + crop_size_px - 1)
    x_start = max(1, x_end - crop_size_px + 1)
    y_start = max(1, y_end - crop_size_px + 1)
    crop = _normalize_psf_energy(
        "physical crop", raw[y_start - 1 : y_end, x_start - 1 : x_end]
    )
    if crop.shape != (crop_size_px, crop_size_px):
        raise ValueError(f"physical crop shape is inconsistent: {crop.shape}")
    resized = crop.copy() if crop.shape == (RENDER_SIZE_PX, RENDER_SIZE_PX) else zoom(
        crop, (RENDER_SIZE_PX / crop.shape[0], RENDER_SIZE_PX / crop.shape[1]),
        order=3, mode="nearest", prefilter=True,
    )
    if resized.shape != (RENDER_SIZE_PX, RENDER_SIZE_PX):
        raise ValueError(f"render PSF resize produced wrong shape: {resized.shape}")
    render = _normalize_psf_energy("render PSF", np.maximum(resized, 0.0))
    return render, float(crop_size_px * pitch / RENDER_SIZE_PX)


def _native_psf_batch(
    model: Any, cases: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the mandatory raw-PSF batch contract; no scalar or field-name fallback."""
    method = getattr(model, "raw_psf_batch", None)
    if not callable(method):
        raise TypeError("PAL evaluation model must implement raw_psf_batch(cases)")
    result = method(cases)
    if not isinstance(result.psf, torch.Tensor):
        raise TypeError("raw_psf_batch.psf must be a torch.Tensor")
    if not isinstance(result.pixel_pitch_mm, torch.Tensor):
        raise TypeError("raw_psf_batch.pixel_pitch_mm must be a torch.Tensor")
    if not isinstance(result.valid_fraction, torch.Tensor):
        raise TypeError("raw_psf_batch.valid_fraction must be a torch.Tensor")
    batch_size = len(cases)
    expected_psf_shape = (batch_size, RAW_SIZE_PX, RAW_SIZE_PX)
    if tuple(result.psf.shape) != expected_psf_shape:
        raise ValueError(
            f"raw PSF batch has shape {tuple(result.psf.shape)}, expected {expected_psf_shape}"
        )
    if tuple(result.pixel_pitch_mm.shape) != (batch_size,):
        raise ValueError("raw PSF batch pixel pitches must have shape [B]")
    if tuple(result.valid_fraction.shape) != (batch_size,):
        raise ValueError("raw PSF batch valid fractions must have shape [B]")
    return (
        result.psf.detach().cpu().numpy().astype(np.float64),
        result.pixel_pitch_mm.detach().cpu().numpy().astype(np.float64),
        result.valid_fraction.detach().cpu().numpy().astype(np.float64),
    )


def _node_sha256(
    field_xy: np.ndarray, raw_psf: np.ndarray, render_psf: np.ndarray,
    raw_pitch: float, render_pitch: float, valid_fraction: float,
) -> str:
    digest = hashlib.sha256()
    values = (
        ("field_xy_deg", np.asarray(field_xy, dtype="<f8")),
        ("raw_psf", np.asarray(raw_psf, dtype="<f8")),
        ("render_psf", np.asarray(render_psf, dtype="<f8")),
        ("scalars", np.asarray([raw_pitch, render_pitch, valid_fraction], dtype="<f8")),
    )
    for name, value in values:
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _condition_contract(
    *, label: str, distance: float, state: str,
    identity_sha256: str, checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": PSF_DATABASE_SCHEMA,
        "evaluation_identity_sha256": identity_sha256,
        "source_checkpoint_sha256": checkpoint_sha256,
        "condition": f"{label}_{state}",
        "distance_label": label,
        "object_distance_mm": _json_distance(distance),
        "state": state,
        "field_values_deg": list(FIELD_VALUES),
        "raw_shape": [RAW_SIZE_PX, RAW_SIZE_PX],
        "render_shape": [RENDER_SIZE_PX, RENDER_SIZE_PX],
        "crop_physical_size_mm": CROP_PHYSICAL_SIZE_MM,
        "orientation": "BIOT physical arrays; field rows stored in ascending Y",
    }


def _create_partial_database(path: Path, contract: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = np.asarray(
        [(x, y) for y in FIELD_VALUES for x in FIELD_VALUES], dtype=np.float64
    )
    with h5py.File(path, "x") as handle:
        handle.attrs["contract_json"] = json.dumps(
            contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        handle.attrs["contract_sha256"] = _canonical_sha256(contract)
        handle.attrs["status"] = "running"
        handle.create_dataset("field_xy_deg", data=fields, dtype="f8")
        handle.create_dataset(
            "raw_psf", shape=(FIELD_COUNT, RAW_SIZE_PX, RAW_SIZE_PX), dtype="f8",
            chunks=(1, RAW_SIZE_PX, RAW_SIZE_PX), compression="gzip",
            compression_opts=4, shuffle=True, fillvalue=np.nan,
        )
        handle.create_dataset(
            "render_psf", shape=(FIELD_COUNT, RENDER_SIZE_PX, RENDER_SIZE_PX), dtype="f8",
            chunks=(1, RENDER_SIZE_PX, RENDER_SIZE_PX), compression="gzip",
            compression_opts=4, shuffle=True, fillvalue=np.nan,
        )
        for name in ("raw_pixel_pitch_mm", "render_pixel_pitch_mm", "valid_fraction"):
            handle.create_dataset(name, shape=(FIELD_COUNT,), dtype="f8", fillvalue=np.nan)
        handle.create_dataset("completed", shape=(FIELD_COUNT,), dtype="u1", fillvalue=0)
        handle.create_dataset("node_sha256", shape=(FIELD_COUNT,), dtype="S64")
        handle.flush()


def _validate_contract(handle: h5py.File, contract: Mapping[str, Any]) -> None:
    if str(handle.attrs.get("contract_sha256", "")) != _canonical_sha256(contract):
        raise ValueError(f"PSF database contract mismatch: {handle.filename}")
    if json.loads(str(handle.attrs.get("contract_json", ""))) != dict(contract):
        raise ValueError(f"PSF database contract payload mismatch: {handle.filename}")
    expected = {
        "field_xy_deg": (FIELD_COUNT, 2),
        "raw_psf": (FIELD_COUNT, RAW_SIZE_PX, RAW_SIZE_PX),
        "render_psf": (FIELD_COUNT, RENDER_SIZE_PX, RENDER_SIZE_PX),
        "raw_pixel_pitch_mm": (FIELD_COUNT,), "render_pixel_pitch_mm": (FIELD_COUNT,),
        "valid_fraction": (FIELD_COUNT,), "completed": (FIELD_COUNT,),
        "node_sha256": (FIELD_COUNT,),
    }
    if set(handle.keys()) != set(expected):
        raise ValueError(f"PSF database datasets differ from schema: {handle.filename}")
    for name, shape in expected.items():
        if handle[name].shape != shape:
            raise ValueError(f"PSF database dataset {name} has wrong shape: {handle[name].shape}")


def _validate_database_node(
    handle: h5py.File, index: int, *, verify_render_contract: bool
) -> None:
    if int(handle["completed"][index]) != 1:
        raise ValueError(f"PSF database node {index} is incomplete")
    field_xy = np.asarray(handle["field_xy_deg"][index], dtype=np.float64)
    expected_field = np.asarray(
        (FIELD_VALUES[index % len(FIELD_VALUES)], FIELD_VALUES[index // len(FIELD_VALUES)]),
        dtype=np.float64,
    )
    if not np.array_equal(field_xy, expected_field):
        raise ValueError(f"PSF database node {index} field coordinate mismatch")
    raw = _normalize_physical_psf("stored raw PSF", handle["raw_psf"][index])
    render = _normalize_physical_psf("stored render PSF", handle["render_psf"][index])
    if raw.shape != (RAW_SIZE_PX, RAW_SIZE_PX) or render.shape != (RENDER_SIZE_PX, RENDER_SIZE_PX):
        raise ValueError(f"PSF database node {index} shape mismatch")
    raw_pitch = float(handle["raw_pixel_pitch_mm"][index])
    render_pitch = float(handle["render_pixel_pitch_mm"][index])
    valid_fraction = float(handle["valid_fraction"][index])
    if not math.isfinite(raw_pitch) or raw_pitch <= 0.0:
        raise ValueError(f"PSF database node {index} raw pixel pitch is invalid")
    if not math.isfinite(render_pitch) or render_pitch <= 0.0:
        raise ValueError(f"PSF database node {index} render pixel pitch is invalid")
    if not math.isfinite(valid_fraction) or not 0.0 <= valid_fraction <= 1.0:
        raise ValueError(f"PSF database node {index} valid fraction is invalid")
    if verify_render_contract:
        expected_render, expected_pitch = _render_psf(raw, raw_pitch)
        if not np.allclose(render, expected_render, rtol=1e-12, atol=1e-12):
            raise ValueError(f"PSF database node {index} render PSF contract mismatch")
        if not math.isclose(render_pitch, expected_pitch, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"PSF database node {index} render pitch contract mismatch")
    expected_hash = _node_sha256(field_xy, raw, render, raw_pitch, render_pitch, valid_fraction)
    saved_hash = bytes(handle["node_sha256"][index]).decode("ascii")
    if saved_hash != expected_hash:
        raise ValueError(f"PSF database node {index} SHA-256 mismatch")


def _validate_condition_file(
    path: Path, contract: Mapping[str, Any], *, verify_render_contract: bool
) -> None:
    with h5py.File(path, "r") as handle:
        _validate_contract(handle, contract)
        if str(handle.attrs.get("status", "")) != "complete":
            raise ValueError(f"PSF database condition is not complete: {path}")
        for index in range(FIELD_COUNT):
            _validate_database_node(handle, index, verify_render_contract=verify_render_contract)


def _write_database_state(
    root: Path, *, identity_sha256: str, status: str,
    completed_conditions: int, completed_nodes: int, current_condition: str | None,
) -> None:
    _write_json(
        root / "psf_database_state.json",
        {"schema_version": PSF_DATABASE_SCHEMA, "status": status,
         "identity_sha256": identity_sha256, "completed_conditions": completed_conditions,
         "total_conditions": 6, "completed_nodes": completed_nodes,
         "total_nodes": 6 * FIELD_COUNT, "current_condition": current_condition},
    )


def _build_condition_database(
    root: Path, *, label: str, distance: float, state: str,
    cases: Sequence[Mapping[str, Any]], model: Any, identity_sha256: str,
    checkpoint_sha256: str, resume: bool, completed_conditions: int,
    psf_batch_size: int,
) -> Path:
    if int(psf_batch_size) <= 0:
        raise ValueError("psf_batch_size must be a positive integer")
    final_path = root / f"{label}_{state}.h5"
    partial_path = root / f"{label}_{state}.partial.h5"
    contract = _condition_contract(
        label=label, distance=distance, state=state,
        identity_sha256=identity_sha256, checkpoint_sha256=checkpoint_sha256,
    )
    if final_path.is_file():
        if partial_path.exists():
            raise ValueError(f"both final and partial PSF databases exist for {label}/{state}")
        _validate_condition_file(final_path, contract, verify_render_contract=True)
        _progress(
            phase="psf_database", condition=f"{completed_conditions + 1}/6",
            name=f"{label}_{state}", fields=f"{FIELD_COUNT}/{FIELD_COUNT}", status="SKIP",
        )
        return final_path
    if partial_path.exists() and not resume:
        raise FileExistsError(f"partial PSF database exists; use --resume: {partial_path}")
    if not partial_path.exists():
        _create_partial_database(partial_path, contract)
    with h5py.File(partial_path, "r+") as handle:
        _validate_contract(handle, contract)
        if str(handle.attrs.get("status", "")) != "running":
            raise ValueError(f"partial PSF database has invalid status: {partial_path}")
        pending_indices: list[int] = []
        for index, case in enumerate(cases):
            if int(handle["completed"][index]) == 1:
                _validate_database_node(handle, index, verify_render_contract=True)
                continue
            pending_indices.append(index)
        completed_in_condition = FIELD_COUNT - len(pending_indices)
        batch_count = math.ceil(len(pending_indices) / int(psf_batch_size))
        _progress(
            phase="psf_database", condition=f"{completed_conditions + 1}/6",
            name=f"{label}_{state}", fields=f"{completed_in_condition}/{FIELD_COUNT}",
            pending=len(pending_indices), batch_size=int(psf_batch_size), status="RUNNING",
        )
        for batch_number, start in enumerate(
            range(0, len(pending_indices), int(psf_batch_size)), start=1
        ):
            batch_indices = pending_indices[start : start + int(psf_batch_size)]
            batch_cases = [cases[index] for index in batch_indices]
            with torch.no_grad():
                raw_batch, pitch_batch, valid_batch = _native_psf_batch(model, batch_cases)
            for offset, index in enumerate(batch_indices):
                case = cases[index]
                raw = _normalize_physical_psf("native raw PSF", raw_batch[offset])
                raw_pitch = float(pitch_batch[offset])
                valid_fraction = float(valid_batch[offset])
                if not math.isfinite(valid_fraction) or not 0.0 <= valid_fraction <= 1.0:
                    raise ValueError(
                        f"invalid valid fraction for {case['case_id']}: {valid_fraction!r}"
                    )
                render, render_pitch = _render_psf(raw, raw_pitch)
                field_xy = np.asarray(
                    [float(case["field_x_deg"]), float(case["field_y_deg"])], dtype=np.float64
                )
                digest = _node_sha256(
                    field_xy, raw, render, raw_pitch, render_pitch, valid_fraction
                )
                handle["raw_psf"][index] = raw
                handle["render_psf"][index] = render
                handle["raw_pixel_pitch_mm"][index] = raw_pitch
                handle["render_pixel_pitch_mm"][index] = render_pitch
                handle["valid_fraction"][index] = valid_fraction
                handle["node_sha256"][index] = digest.encode("ascii")
                handle.flush()
                handle["completed"][index] = 1
                handle.flush()
                _write_database_state(
                    root, identity_sha256=identity_sha256, status="running",
                    completed_conditions=completed_conditions,
                    completed_nodes=(
                        completed_conditions * FIELD_COUNT
                        + int(handle["completed"][:].sum())
                    ),
                    current_condition=f"{label}_{state}",
                )
            completed_in_condition += len(batch_indices)
            _progress(
                phase="psf_database", condition=f"{completed_conditions + 1}/6",
                name=f"{label}_{state}", batch=f"{batch_number}/{batch_count}",
                fields=f"{completed_in_condition}/{FIELD_COUNT}",
                total=f"{completed_conditions * FIELD_COUNT + completed_in_condition}/{6 * FIELD_COUNT}",
                status="DONE",
            )
        for index in range(FIELD_COUNT):
            _validate_database_node(handle, index, verify_render_contract=True)
        handle.attrs["status"] = "complete"
        handle.flush()
    os.replace(partial_path, final_path)
    _validate_condition_file(final_path, contract, verify_render_contract=True)
    _progress(
        phase="psf_database", condition=f"{completed_conditions + 1}/6",
        name=f"{label}_{state}", fields=f"{FIELD_COUNT}/{FIELD_COUNT}", status="COMPLETE",
    )
    return final_path


def _database_conditions(config: Any) -> list[tuple[str, float, str, list[dict[str, Any]]]]:
    return [
        (label, distance, state, cases)
        for label, distance, cases in _distance_cases(config)
        for state in ("baseline", "optimized")
    ]


def _build_psf_database(
    evaluation: Path, *, config: Any, modules: Mapping[str, torch.nn.Module],
    identity_sha256: str, checkpoint_sha256: str, resume: bool,
    psf_batch_size: int,
) -> Path:
    root = evaluation / "psf_database"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "psf_database_manifest.json"
    state_path = root / "psf_database_state.json"
    if manifest_path.is_file():
        _validate_psf_database(evaluation, config=config, identity_sha256=identity_sha256)
        _progress(
            phase="psf_database", conditions="6/6",
            fields=f"{6 * FIELD_COUNT}/{6 * FIELD_COUNT}", status="SKIP",
        )
        return manifest_path
    if state_path.is_file() and not resume:
        raise FileExistsError(f"PSF database state exists; use --resume: {state_path}")
    _write_database_state(
        root, identity_sha256=identity_sha256, status="running",
        completed_conditions=0, completed_nodes=0, current_condition=None,
    )
    files: list[dict[str, Any]] = []
    for index, (label, distance, state, cases) in enumerate(_database_conditions(config)):
        model = pal.MinimalOpticalModel(config, modules[state])
        try:
            path = _build_condition_database(
                root, label=label, distance=distance, state=state, cases=cases,
                model=model, identity_sha256=identity_sha256,
                checkpoint_sha256=checkpoint_sha256, resume=resume,
                completed_conditions=index,
                psf_batch_size=psf_batch_size,
            )
        finally:
            model.close()
        files.append(
            {"condition": f"{label}_{state}", "path": path.name,
             "sha256": _sha256(path), "size": path.stat().st_size}
        )
        _write_database_state(
            root, identity_sha256=identity_sha256, status="running",
            completed_conditions=index + 1, completed_nodes=(index + 1) * FIELD_COUNT,
            current_condition=None,
        )
    _write_json(
        manifest_path,
        {"schema_version": PSF_DATABASE_SCHEMA, "status": "complete",
         "identity_sha256": identity_sha256, "condition_count": len(files), "files": files},
    )
    _write_database_state(
        root, identity_sha256=identity_sha256, status="complete",
        completed_conditions=6, completed_nodes=6 * FIELD_COUNT, current_condition=None,
    )
    _validate_psf_database(evaluation, config=config, identity_sha256=identity_sha256)
    _progress(
        phase="psf_database", conditions="6/6",
        fields=f"{6 * FIELD_COUNT}/{6 * FIELD_COUNT}", status="COMPLETE",
    )
    return manifest_path


def _validate_psf_database(
    evaluation: Path, *, config: Any, identity_sha256: str
) -> dict[str, Any]:
    root = evaluation / "psf_database"
    state = _json(root / "psf_database_state.json")
    manifest = _json(root / "psf_database_manifest.json")
    if state.get("status") != "complete" or manifest.get("status") != "complete":
        raise ValueError("PSF database is not complete")
    if state.get("identity_sha256") != identity_sha256 or manifest.get("identity_sha256") != identity_sha256:
        raise ValueError("PSF database identity mismatch")
    expected = _database_conditions(config)
    records = list(manifest.get("files", []))
    if len(records) != len(expected) or int(manifest.get("condition_count", -1)) != 6:
        raise ValueError("PSF database manifest does not contain exactly six conditions")
    by_condition = {str(record.get("condition")): record for record in records}
    checkpoint_sha256 = str(_json(evaluation / "evaluation_identity.json")["checkpoint_sha256"])
    for label, distance, state_name, _ in expected:
        condition = f"{label}_{state_name}"
        if condition not in by_condition:
            raise ValueError(f"PSF database manifest lacks {condition}")
        record = by_condition[condition]
        path = root / str(record["path"])
        if not path.is_file() or _sha256(path) != str(record["sha256"]):
            raise ValueError(f"PSF database file hash mismatch: {condition}")
        contract = _condition_contract(
            label=label, distance=distance, state=state_name,
            identity_sha256=identity_sha256, checkpoint_sha256=checkpoint_sha256,
        )
        _validate_condition_file(path, contract, verify_render_contract=False)
    return manifest


def _weighted_mtf(psf: np.ndarray, pitch_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    psf = _normalize_physical_psf("weighted-MTF PSF", psf)
    mtf = np.asarray(compute_dc_normalized_mtf(psf), dtype=np.float64)
    size, center = int(mtf.shape[0]), int(mtf.shape[0]) // 2
    frequency = (1.0 / ((size + 1) * float(pitch_mm))) * np.arange(center + 1)
    sagittal = mtf[center, center : center + center + 1]
    tangential = mtf[center : center + center + 1, center]
    count = min(frequency.size, sagittal.size, tangential.size)
    frequency = np.asarray(frequency[:count], dtype=np.float64)
    sagittal = np.clip(np.asarray(sagittal[:count], dtype=np.float64), 0.0, 1.0)
    tangential = np.clip(np.asarray(tangential[:count], dtype=np.float64), 0.0, 1.0)
    if frequency[-1] < 100.0:
        raise ValueError(f"native MTF support ends at {frequency[-1]:g} cycles/mm, below 100")
    sagittal_common = CubicSpline(frequency, sagittal, extrapolate=False)(COMMON_FREQ)
    tangential_common = CubicSpline(frequency, tangential, extrapolate=False)(COMMON_FREQ)
    if not np.isfinite(sagittal_common).all() or not np.isfinite(tangential_common).all():
        raise ValueError("MTF interpolation produced non-finite values")
    cycles_per_degree = COMMON_FREQ * CSF_MM_PER_DEG
    sech = lambda value: 1.0 / np.cosh(value)
    weight = np.maximum(
        CSF_GAIN * (sech((cycles_per_degree / CSF_F0) ** CSF_P)
                    - CSF_A * sech(cycles_per_degree / CSF_F1)), 0.0,
    )
    normalization = float(np.trapz(weight, COMMON_FREQ))
    if not math.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("Ahumada CSF normalization is invalid")
    sagittal_score = float(np.trapz(weight * sagittal_common, COMMON_FREQ) / normalization)
    tangential_score = float(np.trapz(weight * tangential_common, COMMON_FREQ) / normalization)
    return (
        np.asarray([sagittal_score, tangential_score,
                    0.5 * (sagittal_score + tangential_score)]),
        sagittal_common, tangential_common,
    )


def _stage_is_complete(path: Path, config: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    manifest = _json(path)
    if manifest.get("status") != "complete" or manifest.get("config_sha256") != _canonical_sha256(config):
        return False
    for record in manifest.get("files", []):
        artifact = path.parent / str(record["path"])
        if not artifact.is_file() or _sha256(artifact) != str(record["sha256"]):
            raise ValueError(f"completed stage artifact hash mismatch: {artifact}")
    return True


def _write_stage_manifest(
    path: Path, *, config: Mapping[str, Any], files: Sequence[Path]
) -> None:
    _write_json(
        path,
        {"schema_version": STAGE_MANIFEST_SCHEMA, "status": "complete",
         "config": dict(config), "config_sha256": _canonical_sha256(config),
         "files": [
             {"path": artifact.relative_to(path.parent).as_posix(),
              "sha256": _sha256(artifact), "size": artifact.stat().st_size}
             for artifact in sorted(files)
         ]},
    )


def _condition_path(evaluation: Path, label: str, state: str) -> Path:
    return evaluation / "psf_database" / f"{label}_{state}.h5"


def _run_weighted_mtf(
    evaluation: Path, *, config: Any, identity_sha256: str, database_sha256: str
) -> None:
    output = evaluation / "weighted_mtf"
    output.mkdir(parents=True, exist_ok=True)
    stage_config = {
        "identity_sha256": identity_sha256,
        "psf_database_manifest_sha256": database_sha256,
        "algorithm": "Ahumada-1D mean of sagittal/tangential",
        "frequency_support_cycles_per_mm": [0.0, 100.0],
        "samples": len(COMMON_FREQ),
        "published_maps": "mean_native_and_interpolated_png",
        "field_map_interpolation": {
            "purpose": "display_only",
            "method": WEIGHTED_MTF_FIELD_INTERPOLATION,
            "resolution": WEIGHTED_MTF_INTERPOLATED_RESOLUTION,
            "domain": [float(FIELD_VALUES[0]), float(FIELD_VALUES[-1])],
            "extrapolation": False,
            "native_nodes_preserved_abs_tolerance": 1.0e-12,
        },
    }
    manifest_path = output / "weighted_mtf_manifest.json"
    if _stage_is_complete(manifest_path, stage_config):
        _progress(phase="weighted_mtf", conditions="6/6", status="SKIP")
        return
    files: list[Path] = []
    completed = 0
    for label, _, _ in _distance_cases(config):
        maps: dict[str, np.ndarray] = {}
        interpolated_maps: dict[str, np.ndarray] = {}
        interpolated_x: np.ndarray | None = None
        interpolated_y: np.ndarray | None = None
        for state in ("baseline", "optimized"):
            _progress(
                phase="weighted_mtf", condition=f"{completed + 1}/6",
                name=f"{label}_{state}", fields=f"0/{FIELD_COUNT}", status="RUNNING",
            )
            with h5py.File(_condition_path(evaluation, label, state), "r") as handle:
                scores = [
                    _weighted_mtf(handle["raw_psf"][index],
                                  float(handle["raw_pixel_pitch_mm"][index]))[0][2]
                    for index in range(FIELD_COUNT)
                ]
            native_ascending_y = np.asarray(scores).reshape(
                len(FIELD_VALUES), len(FIELD_VALUES)
            )
            maps[state] = native_ascending_y[::-1]
            fine_x, fine_y, fine = _interpolate_weighted_mtf_map(
                native_ascending_y,
                FIELD_VALUES,
                FIELD_VALUES,
            )
            interpolated_x, interpolated_y = fine_x, fine_y
            interpolated_maps[state] = np.clip(fine, 0.0, 1.0)
            completed += 1
            _progress(
                phase="weighted_mtf", condition=f"{completed}/6",
                name=f"{label}_{state}", fields=f"{FIELD_COUNT}/{FIELD_COUNT}", status="DONE",
            )
        maps["delta"] = maps["optimized"] - maps["baseline"]
        interpolated_maps["delta"] = (
            interpolated_maps["optimized"] - interpolated_maps["baseline"]
        )
        if interpolated_x is None or interpolated_y is None:
            raise RuntimeError("weighted-MTF interpolation axes were not initialized")
        for state in ("baseline", "optimized", "delta"):
            numeric = output / f"{label}_{state}_mean_map.npz"
            image = output / f"{label}_{state}_mean.png"
            interpolated_image = output / f"{label}_{state}_mean_interpolated.png"
            np.savez_compressed(
                numeric, mean=maps[state], field_x_deg=np.asarray(FIELD_VALUES),
                field_y_deg=np.asarray(FIELD_VALUES[::-1]),
            )
            _plot_map(image, maps[state], title=f"{label} {state} weighted MTF mean",
                      symmetric=state == "delta")
            _plot_interpolated_weighted_mtf_map(
                interpolated_image,
                interpolated_maps[state],
                interpolated_x,
                interpolated_y,
                title=f"{label} {state} CSF-weighted mean MTF",
                symmetric=state == "delta",
            )
            files.extend((numeric, image, interpolated_image))
    _write_stage_manifest(manifest_path, config=stage_config, files=files)
    _progress(phase="weighted_mtf", conditions="6/6", status="COMPLETE")


def _normalize_display(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("display image contains non-finite values")
    minimum, maximum = float(array.min()), float(array.max())
    return np.zeros_like(array) if maximum <= minimum else (array - minimum) / (maximum - minimum)


def _resize_for_blur_control(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D display image, got {array.shape}")
    target_height, target_width = int(target_shape[0]), int(target_shape[1])
    if target_height < 1 or target_width < 1:
        raise ValueError(f"invalid display target shape: {target_shape}")
    if array.shape == (target_height, target_width):
        return array.copy()
    resized = zoom(
        array, (target_height / array.shape[0], target_width / array.shape[1]),
        order=1, mode="nearest", prefilter=False,
    )
    if resized.shape != (target_height, target_width):
        raise ValueError(f"display resize produced wrong shape: {resized.shape}")
    return resized


def _zero_pad_center(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    target_height, target_width = int(target_shape[0]), int(target_shape[1])
    if target_height < array.shape[0] or target_width < array.shape[1]:
        raise ValueError(f"padding target {target_shape} is smaller than {array.shape}")
    padded = np.zeros((target_height, target_width), dtype=np.float64)
    row, column = (target_height - array.shape[0]) // 2, (target_width - array.shape[1]) // 2
    padded[row : row + array.shape[0], column : column + array.shape[1]] = array
    return padded


def _five_degree_ticks(values: np.ndarray) -> np.ndarray:
    low = int(np.ceil(float(values.min()) / 5.0) * 5)
    high = int(np.floor(float(values.max()) / 5.0) * 5)
    ticks = np.arange(low, high + 1, 5, dtype=int)
    return ticks if ticks.size else values


def _style_stitch_axis(axis: Any, fields: np.ndarray) -> None:
    axis.set_xlabel("field X (Degrees)", fontfamily="Times New Roman", fontsize=18)
    axis.set_ylabel("field Y (Degrees)", fontfamily="Times New Roman", fontsize=18)
    axis.set_xticks(_five_degree_ticks(fields))
    axis.set_yticks(_five_degree_ticks(fields))
    axis.tick_params(direction="in", top=True, right=True, labelsize=14)
    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontfamily("Times New Roman")
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)


def _stitch_canvas(tiles: np.ndarray, field_xy: np.ndarray) -> np.ndarray:
    values_x = np.asarray(sorted(set(float(value) for value in field_xy[:, 0])))
    values_y = np.asarray(sorted(set(float(value) for value in field_xy[:, 1])))
    if values_x.size != len(FIELD_VALUES) or values_y.size != len(FIELD_VALUES):
        raise ValueError("stitch field grid is incomplete")
    side = RENDER_SIZE_PX
    canvas = np.zeros(
        (values_y.size * side + (values_y.size - 1) * TILE_GAP_PX,
         values_x.size * side + (values_x.size - 1) * TILE_GAP_PX), dtype=np.float64,
    )
    for index, (field_x, field_y) in enumerate(field_xy):
        row = values_y.size - 1 - int(np.where(values_y == field_y)[0][0])
        column = int(np.where(values_x == field_x)[0][0])
        row_start, column_start = row * (side + TILE_GAP_PX), column * (side + TILE_GAP_PX)
        canvas[row_start : row_start + side, column_start : column_start + side] = tiles[index]
    return canvas


def _save_psf_stitch_figure(path: Path, image: np.ndarray) -> None:
    fields = np.asarray(FIELD_VALUES)
    smoothed = gaussian_filter(image, sigma=PSF_DISPLAY_SMOOTH_SIGMA)
    figure, axis = plt.subplots(figsize=(8.4, 6.3), dpi=FIGURE_DPI)
    shown = axis.imshow(
        smoothed, extent=[fields.min(), fields.max(), fields.min(), fields.max()],
        origin="upper", cmap="jet", vmin=0.0, vmax=float(np.nanmax(smoothed)), aspect="auto",
    )
    _style_stitch_axis(axis, fields)
    colorbar = figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(direction="in", labelsize=12)
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def _save_chart_stitch_figure(path: Path, image: np.ndarray) -> None:
    fields = np.asarray(FIELD_VALUES)
    display = 1.0 - _normalize_display(image)
    figure, axis = plt.subplots(figsize=(8.4, 6.3), dpi=FIGURE_DPI)
    shown = axis.imshow(
        display, extent=[fields.min(), fields.max(), fields.min(), fields.max()],
        origin="upper", cmap="gray_r", vmin=0.0, vmax=1.0, aspect="auto",
    )
    _style_stitch_axis(axis, fields)
    colorbar = figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.ax.invert_yaxis()
    colorbar.ax.tick_params(direction="in", labelsize=12)
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def _run_psf_stitches(
    evaluation: Path, *, config: Any, identity_sha256: str, database_sha256: str
) -> None:
    output = evaluation / "stitches" / "psf"
    output.mkdir(parents=True, exist_ok=True)
    stage_config = {
        "identity_sha256": identity_sha256,
        "psf_database_manifest_sha256": database_sha256,
        "source_dataset": "render_psf", "tile_gap_px": TILE_GAP_PX,
        "display_smooth_sigma": PSF_DISPLAY_SMOOTH_SIGMA, "origin": "upper",
        "dpi": FIGURE_DPI,
    }
    manifest_path = output / "psf_stitch_manifest.json"
    if _stage_is_complete(manifest_path, stage_config):
        _progress(phase="psf_stitch", conditions="6/6", status="SKIP")
        return
    files: list[Path] = []
    completed = 0
    for label, _, _ in _distance_cases(config):
        for state in ("baseline", "optimized"):
            _progress(
                phase="psf_stitch", condition=f"{completed + 1}/6",
                name=f"{label}_{state}", status="RUNNING",
            )
            with h5py.File(_condition_path(evaluation, label, state), "r") as handle:
                canvas = _stitch_canvas(handle["render_psf"][:], handle["field_xy_deg"][:])
            numeric, image = (
                output / f"{label}_{state}_psf_stitch.npz",
                output / f"{label}_{state}_psf_stitch.png",
            )
            np.savez_compressed(numeric, image=canvas)
            _save_psf_stitch_figure(image, canvas)
            files.extend((numeric, image))
            completed += 1
            _progress(
                phase="psf_stitch", condition=f"{completed}/6",
                name=f"{label}_{state}", status="DONE",
            )
    _write_stage_manifest(manifest_path, config=stage_config, files=files)
    _progress(phase="psf_stitch", conditions="6/6", status="COMPLETE")


def _chart_tile(chart: np.ndarray, render_psf: np.ndarray, blur_scale: float) -> np.ndarray:
    psf = _normalize_physical_psf("chart render PSF", render_psf)
    if blur_scale == 1.0:
        simulated = fftconvolve(chart, psf, mode="same")
    else:
        scaled_shape = (
            max(chart.shape[0], int(round(chart.shape[0] * blur_scale))),
            max(chart.shape[1], int(round(chart.shape[1] * blur_scale))),
        )
        simulated = _resize_for_blur_control(
            fftconvolve(_resize_for_blur_control(chart, scaled_shape),
                        _zero_pad_center(psf, scaled_shape), mode="same"), chart.shape,
        )
    return _normalize_display(simulated)


def _run_chart_stitches(
    evaluation: Path, *, config: Any, identity_sha256: str, database_sha256: str,
    target_path: Path, blur_scale: float,
) -> None:
    if not math.isfinite(blur_scale) or blur_scale < 1.0:
        raise ValueError(f"blur_scale must be finite and >= 1, got {blur_scale!r}")
    target = pd.read_excel(target_path, header=None, engine="openpyxl").to_numpy(dtype=np.float64)
    if target.shape != (RENDER_SIZE_PX, RENDER_SIZE_PX):
        raise ValueError(f"E1.xlsx must be {RENDER_SIZE_PX}x{RENDER_SIZE_PX}, got {target.shape}")
    target = _normalize_display(target)
    output = evaluation / "stitches" / "chart"
    output.mkdir(parents=True, exist_ok=True)
    stage_config = {
        "identity_sha256": identity_sha256,
        "psf_database_manifest_sha256": database_sha256,
        "source_dataset": "render_psf", "target_path": target_path.name,
        "target_sha256": _sha256(target_path), "blur_scale": float(blur_scale),
        "algorithm": "linear chart upsample + centered PSF zero-pad + fftconvolve + linear downsample",
        "tile_gap_px": TILE_GAP_PX, "origin": "upper", "dpi": FIGURE_DPI,
    }
    manifest_path = output / "chart_stitch_manifest.json"
    if _stage_is_complete(manifest_path, stage_config):
        _progress(phase="chart_stitch", conditions="6/6", status="SKIP")
        return
    files: list[Path] = []
    completed = 0
    for label, _, _ in _distance_cases(config):
        for state in ("baseline", "optimized"):
            _progress(
                phase="chart_stitch", condition=f"{completed + 1}/6",
                name=f"{label}_{state}", fields=f"0/{FIELD_COUNT}", status="RUNNING",
            )
            with h5py.File(_condition_path(evaluation, label, state), "r") as handle:
                render = np.asarray(handle["render_psf"][:])
                fields = np.asarray(handle["field_xy_deg"][:])
            tiles = np.stack([_chart_tile(target, render[index], blur_scale)
                              for index in range(FIELD_COUNT)])
            canvas = _stitch_canvas(tiles, fields)
            numeric, image = (
                output / f"{label}_{state}_chart_stitch.npz",
                output / f"{label}_{state}_chart_stitch.png",
            )
            np.savez_compressed(numeric, image=canvas, blur_scale=float(blur_scale))
            _save_chart_stitch_figure(image, canvas)
            files.extend((numeric, image))
            completed += 1
            _progress(
                phase="chart_stitch", condition=f"{completed}/6",
                name=f"{label}_{state}", fields=f"{FIELD_COUNT}/{FIELD_COUNT}", status="DONE",
            )
    _write_stage_manifest(manifest_path, config=stage_config, files=files)
    _progress(phase="chart_stitch", conditions="6/6", status="COMPLETE")


def _write_evaluation_state(
    evaluation: Path, *, identity_sha256: str, status: str, phase: str
) -> None:
    _write_json(
        evaluation / "evaluation_state.json",
        {"schema_version": EVAL_SCHEMA, "status": status, "phase": phase,
         "identity_sha256": identity_sha256},
    )


def _write_evaluation_manifest(evaluation: Path, *, identity_sha256: str) -> None:
    excluded = {"evaluation_manifest.json", "evaluation_state.json"}
    files = [
        {"path": path.relative_to(evaluation).as_posix(), "sha256": _sha256(path),
         "size": path.stat().st_size}
        for path in sorted(evaluation.rglob("*"))
        if path.is_file() and path.name not in excluded and not path.name.endswith(".tmp")
    ]
    _write_json(
        evaluation / "evaluation_manifest.json",
        {"schema_version": EVAL_SCHEMA, "status": "complete",
         "identity_sha256": identity_sha256, "files": files},
    )


def evaluate(
    run: Path, *, device_name: str, resume: bool, blur_scale: float = 4.0,
    psf_batch_size: int = 8, checkpoint_stage: int | None = None,
) -> Path:
    if not math.isfinite(float(blur_scale)) or float(blur_scale) < 1.0:
        raise ValueError(f"blur_scale must be finite and >= 1, got {blur_scale!r}")
    if int(psf_batch_size) <= 0:
        raise ValueError("psf_batch_size must be a positive integer")
    run = run.resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    summary = _json(run / "summary.json")
    if _json(run / "run_state.json").get("status") != "complete":
        raise ValueError("source PAL-NURBS run is not complete")
    source_identity = _json(run / "run_identity.json")
    source_identity_legacy = False
    if hasattr(pal, "_validate_identity_payload"):
        try:
            pal._validate_identity_payload(source_identity)
        except ValueError as error:
            claimed = str(source_identity.get("identity_sha256", ""))
            body = dict(source_identity)
            body.pop("identity_sha256", None)
            canonical = getattr(pal, "_canonical_json_sha256", None)
            if not claimed or not callable(canonical) or canonical(body) != claimed:
                raise error
            source_identity_legacy = True
    if checkpoint_stage is not None and int(checkpoint_stage) not in (7, 11, 19):
        raise ValueError("checkpoint_stage must be one of 7, 11, or 19")
    evaluation = (
        run / "evaluation"
        if checkpoint_stage is None
        else run / f"evaluation_stage_{int(checkpoint_stage)}x{int(checkpoint_stage)}"
    )
    identity_path = evaluation / "evaluation_identity.json"
    if identity_path.exists() and not resume:
        raise FileExistsError(f"evaluation already exists: {evaluation}; use --resume")
    device = torch.device(device_name)
    config = _load_config(run, device_name)
    checkpoint_path, checkpoint = _load_checkpoint(
        run, summary, device, checkpoint_stage=checkpoint_stage,
        source_identity_sha256=str(source_identity.get("identity_sha256", "")),
    )
    checkpoint_sha256 = _sha256(checkpoint_path)
    state_dict = checkpoint["state_dict"]
    control_count = int(checkpoint.get("control_count", summary.get("final_control_count", 7)))
    identity_body = {
        "schema_version": EVAL_SCHEMA,
        "source_run_identity_sha256": str(source_identity.get("identity_sha256", "")),
        "source_training_distances": [
            _json_distance(config.near_object_distance_mm),
            _json_distance(config.intermediate_object_distance_mm),
            _json_distance(config.far_object_distance_mm),
        ] if not hasattr(pal, "DISTANCE_SPECS") else
        [spec.serialized_distance for spec in pal.DISTANCE_SPECS],
        "evaluation_distances": ["D500", "D1000", "Dinf"],
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_path": str(checkpoint_path.relative_to(run)),
        "source_identity_legacy_schema": source_identity_legacy,
        "aperture_contract": "Newton loose residual mapped to aperture boundary",
        "field_grid_deg": list(FIELD_VALUES),
        "psf_database": {
            "schema_version": PSF_DATABASE_SCHEMA,
            "raw_shape": [RAW_SIZE_PX, RAW_SIZE_PX],
            "render_shape": [RENDER_SIZE_PX, RENDER_SIZE_PX],
            "crop_physical_size_mm": CROP_PHYSICAL_SIZE_MM, "condition_files": 6,
            "batch_size": int(psf_batch_size),
        },
        "mtf": {"frequency_max_cycles_per_mm": 100.0, "samples": len(COMMON_FREQ),
                "csf": "Ahumada-1D", "interpolation": "cubic", "published": "mean_only"},
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "cuda": torch.version.cuda, "numpy": str(np.__version__),
                    "scipy": str(scipy.__version__), "h5py": str(h5py.__version__),
                    "platform": sys.platform},
    }
    if checkpoint_stage is not None:
        identity_body.update({
            "checkpoint_selection": f"stage_{int(checkpoint_stage)}x{int(checkpoint_stage)}",
            "checkpoint_control_count": control_count,
        })
    identity = {**identity_body, "identity_sha256": _canonical_sha256(identity_body)}
    if identity_path.exists() and _json(identity_path).get("identity_sha256") != identity["identity_sha256"]:
        raise ValueError("existing evaluation identity does not match current source/checkpoint")
    _write_json(identity_path, identity)
    identity_sha256 = str(identity["identity_sha256"])
    _progress(
        phase="startup", device=device_name, resume=resume,
        psf_batch_size=int(psf_batch_size),
        checkpoint=(
            "final" if checkpoint_stage is None
            else f"stage_{int(checkpoint_stage)}x{int(checkpoint_stage)}"
        ),
        status="COMPLETE",
    )
    _write_evaluation_state(
        evaluation, identity_sha256=identity_sha256, status="running", phase="psf_database"
    )
    base_sag, power_config, _ = pal.load_pal(config, device)
    module_baseline = _make_module(
        config, device, None if hasattr(pal, "DISTANCE_SPECS") else control_count
    )
    module_optimized = _make_module(
        config, device, None if hasattr(pal, "DISTANCE_SPECS") else control_count
    )
    _state_map(module_optimized, state_dict)
    _progress(phase="sag_averfang", status="RUNNING")
    _save_sag_and_averfang(evaluation, base_sag, module_optimized, power_config)
    _progress(phase="sag_averfang", status="COMPLETE")
    database_manifest_path = _build_psf_database(
        evaluation, config=config,
        modules={"baseline": module_baseline, "optimized": module_optimized},
        identity_sha256=identity_sha256, checkpoint_sha256=checkpoint_sha256, resume=resume,
        psf_batch_size=int(psf_batch_size),
    )
    if _validate_psf_database(
        evaluation, config=config, identity_sha256=identity_sha256
    ).get("status") != "complete":
        raise ValueError("PSF database did not reach complete status")
    database_sha256 = _sha256(database_manifest_path)
    _write_evaluation_state(
        evaluation, identity_sha256=identity_sha256, status="running", phase="weighted_mtf"
    )
    _validate_psf_database(evaluation, config=config, identity_sha256=identity_sha256)
    _run_weighted_mtf(
        evaluation, config=config, identity_sha256=identity_sha256,
        database_sha256=database_sha256,
    )
    _write_evaluation_state(
        evaluation, identity_sha256=identity_sha256, status="running", phase="psf_stitch"
    )
    _validate_psf_database(evaluation, config=config, identity_sha256=identity_sha256)
    _run_psf_stitches(
        evaluation, config=config, identity_sha256=identity_sha256,
        database_sha256=database_sha256,
    )
    target_path = Path(__file__).resolve().parent / "inputs" / "evaluation" / "E1.xlsx"
    if not target_path.is_file():
        raise FileNotFoundError(f"evaluation target is missing: {target_path}")
    _write_evaluation_state(
        evaluation, identity_sha256=identity_sha256, status="running", phase="chart_stitch"
    )
    _validate_psf_database(evaluation, config=config, identity_sha256=identity_sha256)
    _run_chart_stitches(
        evaluation, config=config, identity_sha256=identity_sha256,
        database_sha256=database_sha256, target_path=target_path,
        blur_scale=float(blur_scale),
    )
    _write_json(
        evaluation / "evaluation_summary.json",
        {"status": "complete", "identity_sha256": identity_sha256,
         "psf_database_status": "complete", "psf_condition_count": 6,
         "psf_count": 6 * FIELD_COUNT,
         "psf_batch_size": int(psf_batch_size),
         "checkpoint_selection": (
             "final" if checkpoint_stage is None
             else f"stage_{int(checkpoint_stage)}x{int(checkpoint_stage)}"
         ),
         "checkpoint_control_count": control_count,
         "distance_labels": [item[0] for item in _distance_cases(config)],
         "weighted_mtf_products": "mean_native_and_interpolated_png",
         "weighted_mtf_interpolation": {
             "purpose": "display_only",
             "method": WEIGHTED_MTF_FIELD_INTERPOLATION,
             "resolution": WEIGHTED_MTF_INTERPOLATED_RESOLUTION,
             "extrapolation": False,
         },
         "chart_blur_scale": float(blur_scale),
         "source_run_unchanged": True},
    )
    _write_evaluation_manifest(evaluation, identity_sha256=identity_sha256)
    _write_evaluation_state(
        evaluation, identity_sha256=identity_sha256, status="complete", phase="complete"
    )
    _progress(phase="complete", output=evaluation, status="COMPLETE")
    return evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate completed PAL-NURBS run")
    parser.add_argument("--run", required=True, help="completed run directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--checkpoint-stage", type=int, choices=(7, 11, 19), default=None,
        help=(
            "Evaluate a completed stage checkpoint instead of the run's final checkpoint; "
            "writes to evaluation_stage_NxN without modifying the source run."
        ),
    )
    parser.add_argument(
        "--psf-batch-size", type=int, default=8,
        help="Number of field cases traced together for each raw PSF CUDA batch.",
    )
    parser.add_argument(
        "--blur-scale", type=float, default=4.0,
        help=("Display-only chart scale (finite and >=1). Larger values reduce apparent "
              "chart blur; PSF databases, MTF, and PSF stitches are unchanged."),
    )
    arguments = parser.parse_args()
    evaluate(
        Path(arguments.run), device_name=arguments.device,
        resume=bool(arguments.resume), blur_scale=float(arguments.blur_scale),
        psf_batch_size=int(arguments.psf_batch_size),
        checkpoint_stage=arguments.checkpoint_stage,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
