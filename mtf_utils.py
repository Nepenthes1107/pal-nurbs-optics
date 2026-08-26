from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd

from biot.infra.image_io import write_cv_image
from optics import compute_dc_normalized_mtf, sanitize_and_energy_normalize_psf


MTF_EXPORT_MAX_FREQ_CYCLES_PER_MM = 100.0


def _interp_at_x(freq_axis: np.ndarray, values: np.ndarray, x_target: float) -> float:
    """Linear interpolation on a monotonic frequency axis."""
    if freq_axis.size == 0 or values.size == 0:
        return float("nan")
    if freq_axis.size == 1 or values.size == 1:
        return float(values[0])

    if x_target <= float(freq_axis[0]):
        return float(values[0])
    if x_target >= float(freq_axis[-1]):
        return float(values[-1])

    idx = int(np.searchsorted(freq_axis, x_target, side="right") - 1)
    idx = max(0, min(idx, freq_axis.size - 2))
    x0 = float(freq_axis[idx])
    x1 = float(freq_axis[idx + 1])
    y0 = float(values[idx])
    y1 = float(values[idx + 1])
    if x1 == x0:
        return y0
    t = (x_target - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def generate_mtf_curve_and_metrics(psf_image: np.ndarray, d_delta_mm: float, cutoff_freq: float):
    """
    Build 1D MTF curves and cutoff metrics from a PSF array.

    Units and conventions:
    - Input PSF: 2D intensity array, shape [H, W].
    - d_delta_mm: sampling interval on image plane in mm/pixel.
    - cutoff_freq: cycles/mm.
    - Output frequency axis: cycles/mm, exported in the range 0 to 100.
    - Output curves: DC-normalized MTF (MTF(0) = 1).
    - CPU numpy pipeline only (no autograd).
    """
    psf_norm = sanitize_and_energy_normalize_psf(psf_image)
    mtf_2d = compute_dc_normalized_mtf(psf_norm)

    n = int(mtf_2d.shape[0])
    center = n // 2
    d_delta_mm = float(d_delta_mm)
    if d_delta_mm <= 0.0 or (not np.isfinite(d_delta_mm)):
        raise ValueError(f"Invalid d_delta_mm for MTF generation: {d_delta_mm}")

    freq_step = 1.0 / ((n + 1) * d_delta_mm)
    freq_pos = freq_step * np.arange(center + 1, dtype=np.float64)

    mtf_sag = np.asarray(mtf_2d[center, center : center + center + 1], dtype=np.float64)
    mtf_tan = np.asarray(mtf_2d[center : center + center + 1, center], dtype=np.float64)

    target_len = min(freq_pos.size, mtf_sag.size, mtf_tan.size)
    freq_pos = freq_pos[:target_len]
    mtf_sag = mtf_sag[:target_len]
    mtf_tan = mtf_tan[:target_len]

    sagittal_at_cutoff = _interp_at_x(freq_pos, mtf_sag, float(cutoff_freq))
    tangential_at_cutoff = _interp_at_x(freq_pos, mtf_tan, float(cutoff_freq))

    export_mask = freq_pos <= MTF_EXPORT_MAX_FREQ_CYCLES_PER_MM
    if not np.any(export_mask):
        export_mask[0] = True
    freq_export = freq_pos[export_mask]
    mtf_sag_export = mtf_sag[export_mask]
    mtf_tan_export = mtf_tan[export_mask]

    curve_df = pd.DataFrame(
        {
            "frequency_cycles_per_mm": freq_export,
            "MTF_Sagittal": mtf_sag_export,
            "MTF_Tangential": mtf_tan_export,
        }
    )
    metrics = {
        "MTF_Sagittal_At_Cutoff": float(sagittal_at_cutoff),
        "MTF_Tangential_At_Cutoff": float(tangential_at_cutoff),
        "MTF_Cutoff_CyclesPerMM": float(cutoff_freq),
        "MTF_Export_Max_Frequency_CyclesPerMM": MTF_EXPORT_MAX_FREQ_CYCLES_PER_MM,
    }
    return curve_df, metrics


def save_mtf_outputs(
    psf_image: np.ndarray,
    d_delta_mm: float,
    cutoff_freq: float,
    output_dir: Path,
    curve_stem: str,
    metrics_filename: str,
    title: str,
) -> Tuple[Dict[str, float], Dict[str, Path]]:
    """
    Save 1D MTF curves to CSV/XLSX/PNG and return scalar metrics + file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_df, metrics = generate_mtf_curve_and_metrics(psf_image, d_delta_mm, cutoff_freq)

    curve_csv = output_dir / f"{curve_stem}.csv"
    curve_xlsx = output_dir / f"{curve_stem}.xlsx"
    curve_png = output_dir / f"{curve_stem}.png"
    metrics_csv = output_dir / metrics_filename

    curve_df.to_csv(curve_csv, index=False, encoding="utf-8-sig")
    curve_df.to_excel(curve_xlsx, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    _save_mtf_curve_png(curve_df, float(cutoff_freq), curve_png, title)

    paths = {
        "curve_csv": curve_csv,
        "curve_xlsx": curve_xlsx,
        "curve_png": curve_png,
        "metrics_csv": metrics_csv,
    }
    return metrics, paths


def _save_mtf_curve_png(curve_df: pd.DataFrame, cutoff_freq: float, output_path: Path, title: str):
    """Save a simple MTF curve image using OpenCV for runtime stability."""
    width = 1100
    height = 700
    margin_left = 110
    margin_right = 60
    margin_top = 90
    margin_bottom = 110

    img = np.full((height, width, 3), 255, dtype=np.uint8)

    x0 = margin_left
    y0 = height - margin_bottom
    x1 = width - margin_right
    y1 = margin_top
    plot_w = x1 - x0
    plot_h = y0 - y1

    cv2.rectangle(img, (x0, y1), (x1, y0), (230, 230, 230), 1)

    for t in np.linspace(0.0, 1.0, 6):
        y = int(y0 - t * plot_h)
        cv2.line(img, (x0, y), (x1, y), (245, 245, 245), 1)
        cv2.putText(img, f"{t:.1f}", (x0 - 55, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)

    max_freq = MTF_EXPORT_MAX_FREQ_CYCLES_PER_MM
    for t in np.linspace(0.0, 1.0, 6):
        f = t * max_freq
        x = int(x0 + t * plot_w)
        cv2.line(img, (x, y0), (x, y1), (245, 245, 245), 1)
        cv2.putText(
            img,
            f"{f:.0f}",
            (x - 12, y0 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (90, 90, 90),
            1,
        )

    cv2.line(img, (x0, y0), (x1, y0), (60, 60, 60), 2)
    cv2.line(img, (x0, y0), (x0, y1), (60, 60, 60), 2)

    freq = curve_df["frequency_cycles_per_mm"].to_numpy(dtype=np.float64)
    sag = np.clip(curve_df["MTF_Sagittal"].to_numpy(dtype=np.float64), 0.0, 1.05)
    tan = np.clip(curve_df["MTF_Tangential"].to_numpy(dtype=np.float64), 0.0, 1.05)

    def _to_points(y_values: np.ndarray):
        xs = x0 + np.clip(freq / max_freq, 0.0, 1.0) * plot_w
        ys = y0 - np.clip(y_values / 1.05, 0.0, 1.0) * plot_h
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        return pts.reshape(-1, 1, 2)

    sag_pts = _to_points(sag)
    tan_pts = _to_points(tan)
    cv2.polylines(img, [sag_pts], False, (230, 90, 30), 2, lineType=cv2.LINE_AA)
    cv2.polylines(img, [tan_pts], False, (40, 120, 220), 2, lineType=cv2.LINE_AA)

    if 0.0 <= cutoff_freq <= max_freq:
        cutoff_x = int(x0 + np.clip(cutoff_freq / max_freq, 0.0, 1.0) * plot_w)
        cv2.line(img, (cutoff_x, y0), (cutoff_x, y1), (120, 120, 120), 1, lineType=cv2.LINE_AA)

    cv2.putText(img, title, (x0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
    cv2.putText(img, "Frequency (cycles/mm)", (x0 + 270, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    cv2.putText(img, "MTF", (20, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)

    legend_y = y1 + 20
    cv2.line(img, (x1 - 300, legend_y), (x1 - 250, legend_y), (230, 90, 30), 2)
    cv2.putText(img, "Sagittal", (x1 - 240, legend_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    cv2.line(img, (x1 - 160, legend_y), (x1 - 110, legend_y), (40, 120, 220), 2)
    cv2.putText(img, "Tangential", (x1 - 100, legend_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)

    write_cv_image(output_path, img)
