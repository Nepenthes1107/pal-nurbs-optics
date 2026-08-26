"""AverFang-style local power and astigmatism maps.

This module ports the numeric core of
``D:\\MATLAB\\matlab仿真原始\\AverFang.m`` for BIOT GridSag surfaces.  Lengths
are in mm, refractive index is unitless, and power outputs are in dioptres.
The default ``crib_diameter_mm=80`` uses 81 samples over ``[-40, 40]`` mm.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd


EPS = 1e-12


def load_sag_xlsx(path: str | Path, grid_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Load a GridSag sag matrix from an XLSX file.

    Args:
        path: Sag XLSX path. Values are in mm.
        grid_shape: Optional expected ``(height, width)``. If the XLSX is a
            single vector, it is reshaped to this shape.

    Returns:
        ``float64`` array with shape ``[H, W]``. No GPU/autograd is involved.
    """

    data = pd.read_excel(path, header=None, engine="openpyxl").values.astype(np.float64)
    if grid_shape is not None and data.shape != tuple(grid_shape):
        data = data.reshape(tuple(grid_shape))
    return data


def compute_averfang_maps(
    sag_mm: np.ndarray,
    *,
    semi_dia_mm: float,
    refractive_index: float,
    front_radius_mm: float,
    center_thickness_mm: float,
    crib_diameter_mm: float = 80.0,
) -> dict[str, object]:
    """Compute local SPH and astigmatism maps from a GridSag rear surface.

    Args:
        sag_mm: Rear-surface sag values in mm, shape ``[H, W]``.
        semi_dia_mm: Physical semi-diameter of the sag grid in mm.
        refractive_index: Lens refractive index at the evaluated wavelength.
        front_radius_mm: Front spherical surface radius in mm.
        center_thickness_mm: Center lens thickness in mm.
        crib_diameter_mm: Physical output aperture diameter in mm. The output
            point count is ``round(crib_diameter_mm) + 1``.

    Returns:
        Dict containing ``x_mm``, legacy ``y_mm``, ``physical_y_mm``,
        ``power_D``, ``astigmatism_D``, ``cylinder_raw_D`` and ``metadata``.
        Maps retain raw workbook row order. Workbook row 0 is physical +Y, so
        ``physical_y_mm`` is descending while legacy ``y_mm`` is ascending.
        No GPU/autograd is involved.
    """

    z = np.asarray(sag_mm, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"sag_mm must be a 2-D array, got shape {z.shape}")
    height, width = z.shape
    if height < 3 or width < 3:
        raise ValueError("sag grid must have at least 3 x 3 samples")
    if height != width:
        raise ValueError(f"AverFang port currently expects a square sag grid, got {z.shape}")
    if semi_dia_mm <= 0:
        raise ValueError("semi_dia_mm must be positive")
    if front_radius_mm <= 0:
        raise ValueError("front_radius_mm must be positive")

    pitch_x = 2.0 * float(semi_dia_mm) / float(width - 1)
    pitch_y = 2.0 * float(semi_dia_mm) / float(height - 1)
    if abs(pitch_x - pitch_y) > 1e-9:
        raise ValueError(f"non-square grid pitch is unsupported: pitch_x={pitch_x}, pitch_y={pitch_y}")
    pitch = pitch_x

    y_coord = np.linspace(-float(semi_dia_mm), float(semi_dia_mm), height)
    x_coord = np.linspace(-float(semi_dia_mm), float(semi_dia_mm), width)
    x_grid, y_grid = np.meshgrid(x_coord, y_coord)

    dz_dy, dz_dx = np.gradient(z, pitch, pitch)
    d2z_dyy, d2z_dyx = np.gradient(dz_dy, pitch, pitch)
    d2z_dxy, d2z_dxx = np.gradient(dz_dx, pitch, pitch)
    d2z_dxy = 0.5 * (d2z_dxy + d2z_dyx)

    # First and second fundamental forms for r(x, y) = [x, y, z(x, y)].
    e = 1.0 + dz_dx * dz_dx
    f = dz_dx * dz_dy
    g = 1.0 + dz_dy * dz_dy
    normal_scale = np.sqrt(1.0 + dz_dx * dz_dx + dz_dy * dz_dy)
    l = d2z_dxx / normal_scale
    m = d2z_dxy / normal_scale
    n = d2z_dyy / normal_scale
    denom = e * g - f * f
    gaussian = (l * n - m * m) / np.maximum(denom * denom, EPS)
    mean = (e * n + g * l - 2.0 * f * m) / (2.0 * np.maximum(denom, EPS) ** 1.5)
    disc = np.maximum(mean * mean - gaussian, 0.0)
    pmax = mean + np.sqrt(disc)
    pmin = mean - np.sqrt(disc)

    with np.errstate(divide="ignore", invalid="ignore"):
        rmax = 1.0 / pmin
        rmin = 1.0 / pmax
    # Match AverFang.m's curvature orientation: the rear GridSag radius used by
    # the thick-lens formula is negative for the current concave rear surface.
    rear_radius_avg = -0.5 * (rmin + rmax)
    curvature_diff = np.real(pmax - pmin)

    point_count = int(round(float(crib_diameter_mm))) + 1
    if point_count < 3:
        raise ValueError("crib_diameter_mm must produce at least 3 samples")
    if point_count > min(height, width):
        raise ValueError(
            f"crib_diameter_mm={crib_diameter_mm} requires {point_count} samples, "
            f"but sag grid is only {height} x {width}"
        )
    start_y = (height - point_count) // 2
    start_x = (width - point_count) // 2
    end_y = start_y + point_count
    end_x = start_x + point_count

    rr = rear_radius_avg[start_y:end_y, start_x:end_x]
    cdiff = curvature_diff[start_y:end_y, start_x:end_x]
    z_crop = z[start_y:end_y, start_x:end_x]
    x_crop = x_grid[start_y:end_y, start_x:end_x]
    y_crop = y_grid[start_y:end_y, start_x:end_x]

    front_sag = -np.sqrt(np.maximum(float(front_radius_mm) ** 2 - x_crop * x_crop - y_crop * y_crop, 0.0)) + float(
        front_radius_mm
    )
    local_thickness = z_crop - front_sag + float(center_thickness_mm)
    n_lens = float(refractive_index)
    rq = float(front_radius_mm)

    numerator = (n_lens - 1.0) * 1000.0 * (n_lens * (-rr - rq) + (n_lens - 1.0) * local_thickness)
    denominator = (-n_lens * rr * rq) + (n_lens - 1.0) * local_thickness * rr
    with np.errstate(divide="ignore", invalid="ignore"):
        power = numerator / denominator
    power = np.round(power, 3)
    cylinder_raw = -cdiff * (n_lens - 1.0) * 1000.0
    astigmatism = np.abs(cylinder_raw)

    x_out = x_coord[start_x:end_x]
    y_out = y_coord[start_y:end_y]
    radius = float(crib_diameter_mm) / 2.0
    circular_mask = (x_crop * x_crop + y_crop * y_crop) <= radius * radius + 1e-9
    finite_mask = circular_mask & np.isfinite(power) & np.isfinite(astigmatism)
    power = np.where(finite_mask, power, np.nan)
    cylinder_raw = np.where(finite_mask, cylinder_raw, np.nan)
    astigmatism = np.where(finite_mask, astigmatism, np.nan)

    metadata = {
        "source_formula": r"D:\MATLAB\matlab仿真原始\AverFang.m",
        "crib_diameter_mm": float(crib_diameter_mm),
        "point_count": int(point_count),
        "semi_dia_mm": float(semi_dia_mm),
        "grid_pitch_mm": float(pitch),
        "refractive_index": float(refractive_index),
        "front_radius_mm": float(front_radius_mm),
        "center_thickness_mm": float(center_thickness_mm),
        "coordinate_range_x_mm": [float(x_out[0]), float(x_out[-1])],
        "legacy_raw_grid_y_range_mm": [float(y_out[0]), float(y_out[-1])],
        "physical_y_range_mm": [float(-y_out[-1]), float(-y_out[0])],
        "coordinate_convention": (
            "maps retain raw GridSag workbook row order [row,column]; "
            "columns map to physical X=-semi..+semi, rows map to physical "
            "Y=+semi..-semi because optics.GridSag flips workbook rows"
        ),
        "valid_count": int(np.count_nonzero(finite_mask)),
        "invalid_count": int(finite_mask.size - np.count_nonzero(finite_mask)),
        "astigmatism_sign_policy": "astigmatism_D is abs(AverFang Ass); cylinder_raw_D preserves original signed Ass",
    }
    return {
        "x_mm": x_out,
        "y_mm": y_out,
        "physical_y_mm": -y_out,
        "power_D": power,
        "astigmatism_D": astigmatism,
        "cylinder_raw_D": cylinder_raw,
        "metadata": metadata,
    }


def _trim_map_for_plot(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    trim_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return map arrays cropped by `trim_pixels` on all four display edges."""

    trim = max(0, int(trim_pixels))
    if trim == 0:
        return x, y, data
    if data.shape[0] <= 2 * trim or data.shape[1] <= 2 * trim:
        raise ValueError(f"trim_pixels={trim} is too large for map shape {data.shape}")
    return x[trim:-trim], y[trim:-trim], data[trim:-trim, trim:-trim]


def physical_display_map(
    x_mm: np.ndarray,
    physical_y_mm: np.ndarray,
    data: np.ndarray,
    *,
    trim_pixels: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, y, map)`` ordered for physical X/Y display.

    Raw GridSag workbook rows run from physical +Y to -Y. This helper keeps
    numeric arrays in their engineering/raw order while sorting rows into
    ascending physical Y only at the display boundary.
    """

    x_plot, y_plot, data_plot = _trim_map_for_plot(
        np.asarray(x_mm, dtype=float),
        np.asarray(physical_y_mm, dtype=float),
        np.asarray(data),
        trim_pixels,
    )
    order = np.argsort(y_plot)
    return x_plot, y_plot[order], data_plot[order, :]


def save_averfang_outputs(result: Mapping[str, object], output_dir: str | Path, *, trim_pixels: int = 3) -> dict[str, Path]:
    """Save AverFang maps as PNG, NPY, and metadata JSON."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": output / "averfang_map.png",
        "npy": output / "averfang_map.npy",
        "metadata": output / "averfang_metadata.json",
    }
    saved = dict(result)
    saved["metadata"] = dict(result["metadata"])
    saved["metadata"]["display_trim_pixels"] = int(trim_pixels)
    np.save(paths["npy"], saved)
    with paths["metadata"].open("w", encoding="utf-8") as f:
        json.dump(saved["metadata"], f, ensure_ascii=False, indent=2)
    plot_averfang_map(result, paths["png"], trim_pixels=trim_pixels)
    return paths


def plot_averfang_map(result: Mapping[str, object], path: str | Path, *, trim_pixels: int = 3) -> None:
    """Render a 1 x 2 Power/Astigmatism map PNG."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x = np.asarray(result["x_mm"], dtype=float)
    physical_y = np.asarray(result.get("physical_y_mm", -np.asarray(result["y_mm"], dtype=float)), dtype=float)
    power = np.asarray(result["power_D"], dtype=float)
    astig = np.asarray(result["astigmatism_D"], dtype=float)
    x_plot, y_plot, power_plot = physical_display_map(
        x,
        physical_y,
        power,
        trim_pixels=trim_pixels,
    )
    _, _, astig_plot = physical_display_map(
        x,
        physical_y,
        astig,
        trim_pixels=trim_pixels,
    )

    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavuserif",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), dpi=180, constrained_layout=True)
    panels = [(power_plot, "Power (D)"), (astig_plot, "Astigmatism (D)")]
    cmap = "turbo" if "turbo" in plt.colormaps() else "viridis"
    for ax, (data, label) in zip(axes, panels):
        image = ax.imshow(
            data,
            extent=[float(x_plot[0]), float(x_plot[-1]), float(y_plot[0]), float(y_plot[-1])],
            origin="lower",
            cmap=cmap,
            aspect="equal",
        )
        finite = np.isfinite(data)
        if finite.any():
            levels = np.linspace(float(np.nanmin(data)), float(np.nanmax(data)), 14)
            if np.unique(np.round(levels, 12)).size > 1:
                ax.contour(x_plot, y_plot, data, levels=levels, colors="0.2", linewidths=0.45)
        ax.plot(0.0, 0.0, "k+", markersize=5, linewidth=0.9)
        ax.set_xlabel("X (mm)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Y (mm)", fontsize=12, fontweight="bold")
        ax.tick_params(direction="in", top=True, right=True)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_title(label, fontsize=9, fontweight="bold")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute AverFang-style local power and astigmatism maps")
    parser.add_argument("excel_path", help="BIOT lens Excel file")
    parser.add_argument("--crib", type=float, default=80.0, help="Physical aperture diameter in mm [default: 80]")
    parser.add_argument("--wavelength", type=float, default=555.0, help="Wavelength in nm [default: 555]")
    parser.add_argument("--trim-pixels", type=int, default=3, help="Display crop pixels on each map edge [default: 3]")
    parser.add_argument("--output", default="results/averfang_map", help="Output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    import torch
    from lens_metrics_core import build_legacy_adapter, load_lens, resolve_device
    from optics import GridSag

    lens = load_lens(args.excel_path, device=resolve_device("cpu"), wavelength_nm=args.wavelength)
    adapter = build_legacy_adapter(lens, wavelength_nm=args.wavelength)
    front = lens.surfaces[1]
    back = lens.surfaces[2]
    if not isinstance(back, GridSag):
        raise ValueError("Excel row 6 / surfaces[2] must be a GridSag surface for AverFang maps")
    sag_path = getattr(back, "sag_file_path", None)
    if sag_path is None:
        raise ValueError("GridSag surface does not record sag_file_path")
    sag = load_sag_xlsx(sag_path, grid_shape=back.grid_shape)
    result = compute_averfang_maps(
        sag,
        semi_dia_mm=float(back.semi_dia),
        refractive_index=adapter.n1,
        front_radius_mm=1.0 / adapter.c0,
        center_thickness_mm=adapter.h_glass_mm,
        crib_diameter_mm=args.crib,
    )
    result["metadata"].update(
        {
            "lens_excel_path": str(Path(args.excel_path)),
            "sag_file_path": str(sag_path),
            "front_surface_index": 1,
            "back_surface_index": 2,
            "front_surface_type": type(front).__name__,
            "back_surface_type": type(back).__name__,
        }
    )
    paths = save_averfang_outputs(result, args.output, trim_pixels=args.trim_pixels)
    print(f"[averfang] wrote {paths['png']}")
    _ = torch  # keep import visible for environments that lazy-load torch DLLs
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
