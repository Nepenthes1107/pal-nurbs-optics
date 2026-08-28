"""Evaluate a completed PAL-NURBS run without modifying the training run.

The evaluator writes all products below ``<run>/evaluation`` and keeps a
separate evaluation identity.  It supports both the main 7->11->19 runner and
the fixed multidistance 7x7 runner by inspecting the checkpoint/config shape.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.signal import fftconvolve

from optics import compute_dc_normalized_mtf
from biot.e2e import pal_nurbs as pal


EVAL_SCHEMA = 1
FIELD_VALUES = tuple(float(v) for v in np.arange(-40.0, 40.0 + 0.1, 10.0))
COMMON_FREQ = np.linspace(0.0, 100.0, 1000, dtype=np.float64)
CSF_MM_PER_DEG = 0.291
CSF_F0 = 4.1726
CSF_F1 = 1.3625
CSF_A = 0.8493
CSF_P = 0.7786
CSF_GAIN = 373.08


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _finite_array(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _load_config(run: Path, device: str) -> Any:
    saved = _json(run / "config.json")
    allowed = set(getattr(pal.MinimalConfig, "__dataclass_fields__", {}))
    values = {key: value for key, value in saved.items() if key in allowed}
    values["output"] = str(run)
    values["device"] = str(device)
    return pal.MinimalConfig(**values)


def _load_checkpoint(run: Path, summary: Mapping[str, Any], device: torch.device) -> tuple[Path, dict[str, Any]]:
    if hasattr(pal, "DISTANCE_SPECS"):
        candidate = run / "final.pt"
    else:
        count = int(summary.get("final_control_count", 0))
        if count <= 0:
            raise ValueError("summary lacks final_control_count")
        candidate = run / f"stage_{count}x{count}" / "final.pt"
    if not candidate.is_file():
        raise FileNotFoundError(f"final checkpoint not found: {candidate}")
    payload = torch.load(candidate, map_location=device)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"malformed final checkpoint: {candidate}")
    return candidate, payload


def _make_module(config: Any, device: torch.device, control_count: int | None = None) -> torch.nn.Module:
    if control_count is None:
        return pal.FixedWeightNURBSPerturbation(device=device, dtype=torch.float64)
    return pal.FixedWeightNURBSPerturbation(control_count, device=device, dtype=torch.float64)


def _distance_cases(config: Any) -> list[tuple[str, float, list[dict[str, Any]]]]:
    if hasattr(pal, "DISTANCE_SPECS"):
        specs = list(pal.DISTANCE_SPECS)
        result = []
        for spec in specs:
            rows = []
            for row, fy in enumerate(FIELD_VALUES):
                for col, fx in enumerate(FIELD_VALUES):
                    rows.append({
                        "case_id": f"{spec.label}_r{row:02d}_c{col:02d}",
                        "distance_label": spec.label,
                        "field_x_deg": fx,
                        "field_y_deg": fy,
                    })
            result.append((str(spec.label), float(spec.object_distance_mm), rows))
        return result
    # Evaluation is deliberately fixed to the new three-distance contract.
    # This also lets a legacy main/run_001 be measured without rewriting its
    # historical training config.
    specs = (("D500", 500.0), ("D1000", 1000.0), ("Dinf", float("inf")))
    result = []
    for label, distance in specs:
        rows = []
        for row, fy in enumerate(FIELD_VALUES):
            for col, fx in enumerate(FIELD_VALUES):
                rows.append({
                    "case_id": f"{label}_r{row:02d}_c{col:02d}",
                    "distance_mm": distance,
                    "field_x_deg": fx,
                    "field_y_deg": fy,
                })
        result.append((label, distance, rows))
    return result


def _state_map(module: torch.nn.Module, state: Mapping[str, Any]) -> None:
    expected = module.state_dict()
    if set(expected) != set(state):
        raise ValueError("checkpoint state_dict keys do not match the evaluator module")
    module.load_state_dict(dict(state), strict=True)


def _plot_map(path: Path, values: np.ndarray, *, title: str, xlabel: str = "field X (deg)", ylabel: str = "field Y (deg)", symmetric: bool = False) -> None:
    arr = np.asarray(values, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    if symmetric:
        limit = float(np.nanmax(np.abs(arr)))
        limit = max(limit, np.finfo(np.float64).eps)
        image = ax.imshow(arr, origin="upper", extent=[-40, 40, -40, 40], cmap="coolwarm", vmin=-limit, vmax=limit)
    else:
        image = ax.imshow(arr, origin="upper", extent=[-40, 40, -40, 40], cmap="viridis")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.colorbar(image, ax=ax)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_sag_and_averfang(root: Path, base_sag: torch.Tensor, module: torch.nn.Module, power_config: Any, zones: Mapping[str, torch.Tensor]) -> None:
    sag_dir = root / "sag"
    sag_dir.mkdir(parents=True, exist_ok=True)
    sag = base_sag.detach().cpu().numpy().astype(np.float64)
    coord = torch.linspace(-float(power_config.semi_diameter_mm), float(power_config.semi_diameter_mm), sag.shape[0], dtype=torch.float64, device=base_sag.device)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    delta = module.delta_raw(xx, yy).detach().cpu().numpy().astype(np.float64)
    optimized = sag + delta
    for name, value in (("baseline", sag), ("optimized", optimized), ("delta", delta)):
        np.savez_compressed(sag_dir / f"{name}.npz", sag_mm=value, x_mm=coord.cpu().numpy(), physical_y_mm=coord.cpu().numpy()[::-1])
        _plot_map(sag_dir / f"{name}.png", value * (1e6 if name == "delta" else 1.0), title=f"{name} sag ({'um' if name == 'delta' else 'mm'})", symmetric=name == "delta")
    maps_dir = root / "averfang"
    maps_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for name, value in (("baseline", base_sag), ("optimized", base_sag + torch.as_tensor(delta, device=base_sag.device))):
        computed = pal.torch_averfang_maps(value, power_config)
        outputs[name] = {key: computed[key].detach().cpu().numpy() for key in ("power_D", "A_D", "astigmatism_D")}
        np.savez_compressed(maps_dir / f"{name}.npz", **outputs[name])
    outputs["delta"] = {key: outputs["optimized"][key] - outputs["baseline"][key] for key in outputs["baseline"]}
    np.savez_compressed(maps_dir / "delta.npz", **outputs["delta"])
    for key, label in (("power_D", "power (D)"), ("astigmatism_D", "astigmatism (D)")):
        for name in ("baseline", "optimized", "delta"):
            _plot_map(maps_dir / f"{name}_{key}.png", outputs[name][key], title=f"{name} {label}", symmetric=name == "delta")
    _write_json(maps_dir / "metadata.json", {"units": {"power_D": "D", "astigmatism_D": "D"}, "source": "PAL torch_averfang_maps"})


def _weighted_mtf(psf: np.ndarray, pitch_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    psf = _finite_array("PSF", psf)
    if np.any(psf < -1e-15) or psf.sum() <= 0.0:
        raise ValueError("PSF must be non-negative with positive energy")
    psf = np.maximum(psf, 0.0)
    psf /= psf.sum()
    mtf = np.asarray(compute_dc_normalized_mtf(psf), dtype=np.float64)
    n = int(mtf.shape[0])
    center = n // 2
    freq = (1.0 / ((n + 1) * float(pitch_mm))) * np.arange(center + 1, dtype=np.float64)
    sag = mtf[center, center : center + center + 1]
    tan = mtf[center : center + center + 1, center]
    if freq[-1] < 100.0:
        raise ValueError(f"native MTF support ends at {freq[-1]:g} cycles/mm, below 100")
    freq = freq[: min(freq.size, sag.size, tan.size)]
    sag = np.clip(np.asarray(sag[: freq.size], dtype=np.float64), 0.0, 1.0)
    tan = np.clip(np.asarray(tan[: freq.size], dtype=np.float64), 0.0, 1.0)
    sag_i = CubicSpline(freq, sag, extrapolate=False)(COMMON_FREQ)
    tan_i = CubicSpline(freq, tan, extrapolate=False)(COMMON_FREQ)
    if not np.isfinite(sag_i).all() or not np.isfinite(tan_i).all():
        raise ValueError("MTF interpolation produced non-finite values")
    cpdeg = COMMON_FREQ * CSF_MM_PER_DEG
    sech = lambda x: 1.0 / np.cosh(x)
    weight = np.maximum(CSF_GAIN * (sech((cpdeg / CSF_F0) ** CSF_P) - CSF_A * sech(cpdeg / CSF_F1)), 0.0)
    norm = float(np.trapz(weight, COMMON_FREQ))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Ahumada CSF normalization is invalid")
    sag_score = float(np.trapz(weight * sag_i, COMMON_FREQ) / norm)
    tan_score = float(np.trapz(weight * tan_i, COMMON_FREQ) / norm)
    return np.asarray([sag_score, tan_score, 0.5 * (sag_score + tan_score)]), sag_i, tan_i


def _save_stitches(root: Path, label: str, records: Sequence[Mapping[str, Any]], target: np.ndarray) -> None:
    out = root / "stitches"
    out.mkdir(parents=True, exist_ok=True)
    for state in ("baseline", "optimized"):
        canvas_psf = np.zeros((9 * 130, 9 * 130), dtype=np.float64)
        canvas_chart = np.zeros_like(canvas_psf)
        for record in records:
            if record["state"] != state:
                continue
            tile = np.asarray(record["psf"], dtype=np.float64)
            chart = fftconvolve(target, tile, mode="same")
            row = int(record["row"])
            col = int(record["col"])
            y0, x0 = row * 130, col * 130
            canvas_psf[y0:y0 + 130, x0:x0 + 130] = tile
            canvas_chart[y0:y0 + 130, x0:x0 + 130] = chart
        for name, image in (("psf_stitch", canvas_psf), ("chart_stitch", canvas_chart)):
            np.savez_compressed(out / f"{label}_{state}_{name}.npz", image=image)
            fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
            ax.imshow(image, origin="upper", cmap="gray")
            ax.set(title=f"{label} {state} {name}")
            fig.savefig(out / f"{label}_{state}_{name}.png", dpi=120)
            plt.close(fig)


def evaluate(run: Path, *, device_name: str, resume: bool) -> Path:
    run = run.resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    summary = _json(run / "summary.json")
    source_identity = _json(run / "run_identity.json")
    if hasattr(pal, "_validate_identity_payload"):
        pal._validate_identity_payload(source_identity)
    evaluation = run / "evaluation"
    identity_path = evaluation / "evaluation_identity.json"
    if identity_path.exists() and not resume:
        raise FileExistsError(f"evaluation already exists: {evaluation}; use --resume only for the same identity")
    device = torch.device(device_name)
    config = _load_config(run, device_name)
    checkpoint_path, checkpoint = _load_checkpoint(run, summary, device)
    state_dict = checkpoint["state_dict"]
    control_count = int(checkpoint.get("control_count", summary.get("final_control_count", 7)))
    eval_identity_body = {
        "schema_version": EVAL_SCHEMA,
        "source_run_identity_sha256": str(source_identity.get("identity_sha256", "")),
        "source_training_distances": [config.near_object_distance_mm, config.intermediate_object_distance_mm, config.far_object_distance_mm] if not hasattr(pal, "DISTANCE_SPECS") else [spec.serialized_distance for spec in pal.DISTANCE_SPECS],
        "evaluation_distances": ["D500", "D1000", "Dinf"],
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_path": str(checkpoint_path.relative_to(run)),
        "aperture_contract": "Newton loose residual mapped to aperture boundary",
        "field_grid_deg": list(FIELD_VALUES),
        "mtf": {"frequency_max_cycles_per_mm": 100.0, "samples": 1000, "csf": "Ahumada-1D", "interpolation": "cubic"},
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda, "platform": sys.platform},
    }
    eval_identity = {**eval_identity_body, "identity_sha256": hashlib.sha256(json.dumps(eval_identity_body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
    if identity_path.exists() and _json(identity_path).get("identity_sha256") != eval_identity["identity_sha256"]:
        raise ValueError("existing evaluation identity does not match current source/checkpoint")
    _write_json(identity_path, eval_identity)
    base_sag, power_config, zones = pal.load_pal(config, device)
    module_baseline = _make_module(config, device, None if hasattr(pal, "DISTANCE_SPECS") else control_count)
    module_optimized = _make_module(config, device, None if hasattr(pal, "DISTANCE_SPECS") else control_count)
    _state_map(module_optimized, state_dict)
    _save_sag_and_averfang(evaluation, base_sag, module_optimized, power_config, zones)
    target_path = Path(__file__).resolve().parent / "inputs" / "evaluation" / "E1.xlsx"
    if not target_path.is_file():
        raise FileNotFoundError(f"evaluation target is missing: {target_path}")
    import pandas as pd
    target = pd.read_excel(target_path, header=None, engine="openpyxl").to_numpy(dtype=np.float64)
    if target.shape != (130, 130):
        raise ValueError(f"E1.xlsx must be 130x130, got {target.shape}")
    all_records: list[dict[str, Any]] = []
    (evaluation / "psf").mkdir(parents=True, exist_ok=True)
    (evaluation / "mtf").mkdir(parents=True, exist_ok=True)
    mtf_scores: dict[str, dict[str, np.ndarray]] = {}
    for label, distance, cases in _distance_cases(config):
        mtf_scores[label] = {}
        for state, module in (("baseline", module_baseline), ("optimized", module_optimized)):
            model = pal.MinimalOpticalModel(config, module)
            scores = []
            try:
                for case in cases:
                    with torch.no_grad():
                        result = model.field(case)
                    psf = result.psf.detach().cpu().numpy().astype(np.float64)
                    pitch = float(result.pixel_pitch_mm)
                    health = {"valid_fraction": float(result.valid_fraction.detach().cpu()), "edge_fraction": float(result.edge_fraction.detach().cpu()), "pixel_pitch_mm": pitch}
                    if not np.isfinite(psf).all() or np.any(psf < 0.0) or abs(float(psf.sum()) - 1.0) > 1e-8:
                        raise ValueError(f"invalid physical PSF for {case['case_id']}")
                    score, sag_curve, tan_curve = _weighted_mtf(psf, pitch)
                    row = int(case["case_id"].split("_r")[1].split("_c")[0])
                    col = int(case["case_id"].split("_c")[1])
                    all_records.append({"state": state, "label": label, "row": row, "col": col, "psf": psf})
                    scores.append(score)
                    np.savez_compressed(evaluation / "psf" / f"{label}_{state}_r{row:02d}_c{col:02d}.npz", psf=psf, **health)
                    with (evaluation / "mtf" / f"{label}_{state}_r{row:02d}_c{col:02d}.csv").open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle); writer.writerow(["frequency_cycles_per_mm", "MTF_Sagittal", "MTF_Tangential"])
                        writer.writerows(zip(COMMON_FREQ.tolist(), sag_curve.tolist(), tan_curve.tolist()))
                mtf_scores[label][state] = np.asarray(scores, dtype=np.float64).reshape(9, 9, 3)
            finally:
                model.close()
        mtf_scores[label]["delta"] = mtf_scores[label]["optimized"] - mtf_scores[label]["baseline"]
        for state in ("baseline", "optimized", "delta"):
            np.savez_compressed(evaluation / "mtf" / f"{label}_{state}_map.npz", **{"sag": mtf_scores[label][state][..., 0], "tan": mtf_scores[label][state][..., 1], "mean": mtf_scores[label][state][..., 2]})
            _plot_map(evaluation / "mtf" / f"{label}_{state}_mean.png", mtf_scores[label][state][..., 2], title=f"{label} {state} weighted MTF mean", symmetric=state == "delta")
        _save_stitches(evaluation, label, [r for r in all_records if r["label"] == label], target)
    _write_json(evaluation / "evaluation_summary.json", {"status": "complete", "identity_sha256": eval_identity["identity_sha256"], "psf_count": len(all_records), "distance_labels": [item[0] for item in _distance_cases(config)], "source_run_unchanged": True})
    return evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate completed PAL-NURBS run")
    parser.add_argument("--run", required=True, help="completed run directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    evaluate(Path(args.run), device_name=args.device, resume=bool(args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
