from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from optics import Lensdata, sanitize_and_energy_normalize_psf

from .bspline import bspline_basis_1d, bspline_surface_2d, open_uniform_knots
from .psf_gaussian import psf_centroid_mm
from .surfaces import SurfaceDomain


@dataclass(frozen=True)
class SurfaceFitResult:
    label: str
    row_number: int
    surface_type: str
    semi_diameter_mm: float
    control_shape: tuple[int, int]
    sample_shape: tuple[int, int]
    valid_samples: int
    rmse_mm: float
    mae_mm: float
    max_abs_mm: float
    p95_abs_mm: float
    sag_range_mm: float

    def as_dict(self) -> dict[str, float | int | str]:
        relative_rmse = self.rmse_mm / self.sag_range_mm if self.sag_range_mm > 0 else np.nan
        return {
            "label": self.label,
            "excel_row": self.row_number,
            "surface_type": self.surface_type,
            "semi_diameter_mm": self.semi_diameter_mm,
            "control_shape_x": self.control_shape[0],
            "control_shape_y": self.control_shape[1],
            "sample_shape_x": self.sample_shape[0],
            "sample_shape_y": self.sample_shape[1],
            "valid_samples": self.valid_samples,
            "rmse_mm": self.rmse_mm,
            "rmse_um": self.rmse_mm * 1.0e3,
            "mae_um": self.mae_mm * 1.0e3,
            "max_abs_um": self.max_abs_mm * 1.0e3,
            "p95_abs_um": self.p95_abs_mm * 1.0e3,
            "sag_range_mm": self.sag_range_mm,
            "relative_rmse_to_sag_range": relative_rmse,
        }


@dataclass(frozen=True)
class PSFComparisonResult:
    label: str
    shape_h: int
    shape_w: int
    l1: float
    l2: float
    normalized_cross_correlation: float
    max_abs: float
    centroid_dx_mm: float
    centroid_dy_mm: float
    second_moment_ratio: float
    edge_energy_biot: float
    edge_energy_e2e: float
    biot_energy_sum: float
    e2e_energy_sum: float
    radial_profile_rmse: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "label": self.label,
            "shape_h": self.shape_h,
            "shape_w": self.shape_w,
            "l1": self.l1,
            "l2": self.l2,
            "normalized_cross_correlation": self.normalized_cross_correlation,
            "max_abs": self.max_abs,
            "centroid_dx_mm": self.centroid_dx_mm,
            "centroid_dy_mm": self.centroid_dy_mm,
            "second_moment_ratio": self.second_moment_ratio,
            "edge_energy_biot": self.edge_energy_biot,
            "edge_energy_e2e": self.edge_energy_e2e,
            "biot_energy_sum": self.biot_energy_sum,
            "e2e_energy_sum": self.e2e_energy_sum,
            "radial_profile_rmse": self.radial_profile_rmse,
        }


def sample_biot_surface(
    excel_path: str | Path,
    *,
    surface_index: int,
    sample_shape: tuple[int, int] = (81, 81),
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, object]:
    """Sample a BIOT surface sag on its local square domain.

    ``surface_index`` is zero-based in ``Lensdata.surfaces``. For the default
    Excel file, index 1 is the spectacle front surface and index 2 is the
    GridSag back surface.
    """
    lens = Lensdata(device=torch.device(device))
    lens.load_file(Path(excel_path), extension=".xlsx")
    surface = lens.surfaces[int(surface_index)]
    semi_dia = float(surface.semi_dia)
    nx, ny = int(sample_shape[0]), int(sample_shape[1])
    x = torch.linspace(-semi_dia, semi_dia, nx, dtype=torch.float64, device=torch.device(device))
    y = torch.linspace(-semi_dia, semi_dia, ny, dtype=torch.float64, device=torch.device(device))
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    sag = surface.get_surface(xx, yy)[0].to(dtype=torch.float64)
    mask = xx.pow(2) + yy.pow(2) <= semi_dia**2 + 1.0e-12
    return xx, yy, sag, mask, surface


def fit_bspline_control_grid(
    x_mm: torch.Tensor,
    y_mm: torch.Tensor,
    target_sag_mm: torch.Tensor,
    mask: torch.Tensor,
    *,
    control_shape: tuple[int, int],
    domain: SurfaceDomain,
    degree: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares fit a tensor-product B-spline surface to sampled sag.

    The full finite rectangular sampling grid is used for fitting when
    available, because a circular aperture mask leaves corner control points
    underconstrained. The caller can still evaluate metrics only inside the
    physical aperture mask.
    """
    if x_mm.shape != y_mm.shape or x_mm.shape != target_sag_mm.shape or x_mm.shape != mask.shape:
        raise ValueError("x_mm, y_mm, target_sag_mm, and mask must have identical shape")
    cx, cy = int(control_shape[0]), int(control_shape[1])
    if cx <= degree or cy <= degree:
        raise ValueError("control dimensions must be greater than degree")

    x_flat = x_mm.reshape(-1)
    y_flat = y_mm.reshape(-1)
    target_flat = target_sag_mm.reshape(-1)
    _ = mask
    mask_flat = torch.isfinite(target_flat)

    x_knots = open_uniform_knots(cx, degree, domain.x_range_mm, device=x_mm.device, dtype=x_mm.dtype)
    y_knots = open_uniform_knots(cy, degree, domain.y_range_mm, device=x_mm.device, dtype=x_mm.dtype)
    bx = bspline_basis_1d(x_flat[mask_flat], x_knots, degree)
    by = bspline_basis_1d(y_flat[mask_flat], y_knots, degree)
    design = (bx[:, :, None] * by[:, None, :]).reshape(int(mask_flat.sum()), cx * cy)
    solution = torch.linalg.lstsq(design, target_flat[mask_flat].unsqueeze(-1)).solution.squeeze(-1)
    controls = solution.reshape(cx, cy)
    recon = bspline_surface_2d(
        x_mm,
        y_mm,
        controls,
        x_knots,
        y_knots,
        degree_x=degree,
        degree_y=degree,
    )
    return controls, recon


def evaluate_surface_fit(
    *,
    label: str,
    row_number: int,
    surface_type: str,
    semi_diameter_mm: float,
    target_sag_mm: torch.Tensor,
    reconstructed_sag_mm: torch.Tensor,
    mask: torch.Tensor,
    control_shape: tuple[int, int],
) -> SurfaceFitResult:
    valid = mask & torch.isfinite(target_sag_mm) & torch.isfinite(reconstructed_sag_mm)
    if not torch.any(valid):
        raise ValueError("surface fit has no valid samples")
    error = reconstructed_sag_mm[valid] - target_sag_mm[valid]
    abs_error = error.abs()
    target_valid = target_sag_mm[valid]
    return SurfaceFitResult(
        label=label,
        row_number=int(row_number),
        surface_type=surface_type,
        semi_diameter_mm=float(semi_diameter_mm),
        control_shape=tuple(int(v) for v in control_shape),
        sample_shape=tuple(int(v) for v in target_sag_mm.shape),
        valid_samples=int(valid.sum().item()),
        rmse_mm=float(torch.sqrt(error.pow(2).mean()).item()),
        mae_mm=float(abs_error.mean().item()),
        max_abs_mm=float(abs_error.max().item()),
        p95_abs_mm=float(torch.quantile(abs_error, 0.95).item()),
        sag_range_mm=float((target_valid.max() - target_valid.min()).item()),
    )


def compare_energy_normalized_psfs(
    biot_psf: np.ndarray,
    e2e_psf: np.ndarray,
    *,
    pixel_pitch_mm: float,
    label: str = "psf_compare",
    edge_width_px: int = 8,
) -> PSFComparisonResult:
    """Compare two same-grid energy-normalized PSFs."""
    biot = sanitize_and_energy_normalize_psf(biot_psf)
    e2e = sanitize_and_energy_normalize_psf(e2e_psf)
    if biot.shape != e2e.shape:
        raise ValueError(f"PSF shapes differ: BIOT {biot.shape}, e2e {e2e.shape}")
    h, w = biot.shape
    diff = e2e - biot
    biot_flat = biot.reshape(-1)
    e2e_flat = e2e.reshape(-1)
    biot_centered = biot_flat - biot_flat.mean()
    e2e_centered = e2e_flat - e2e_flat.mean()
    denom = np.linalg.norm(biot_centered) * np.linalg.norm(e2e_centered)
    ncc = float(np.dot(biot_centered, e2e_centered) / denom) if denom > 0 else np.nan

    biot_t = torch.as_tensor(biot, dtype=torch.float64)
    e2e_t = torch.as_tensor(e2e, dtype=torch.float64)
    c_biot = psf_centroid_mm(biot_t, pixel_pitch_mm).detach().cpu().numpy()
    c_e2e = psf_centroid_mm(e2e_t, pixel_pitch_mm).detach().cpu().numpy()
    second_biot = _second_moment_mm2(biot, pixel_pitch_mm, c_biot)
    second_e2e = _second_moment_mm2(e2e, pixel_pitch_mm, c_e2e)
    edge_biot = _edge_energy(biot, edge_width_px)
    edge_e2e = _edge_energy(e2e, edge_width_px)
    _, radial_biot = radial_profile(biot, pixel_pitch_mm)
    _, radial_e2e = radial_profile(e2e, pixel_pitch_mm)
    radial_rmse = float(np.sqrt(np.mean(np.square(radial_e2e - radial_biot))))
    return PSFComparisonResult(
        label=label,
        shape_h=int(h),
        shape_w=int(w),
        l1=float(np.abs(diff).sum()),
        l2=float(np.sqrt(np.square(diff).sum())),
        normalized_cross_correlation=ncc,
        max_abs=float(np.abs(diff).max()),
        centroid_dx_mm=float(c_e2e[0] - c_biot[0]),
        centroid_dy_mm=float(c_e2e[1] - c_biot[1]),
        second_moment_ratio=float(second_e2e / second_biot) if second_biot > 0 else np.nan,
        edge_energy_biot=float(edge_biot),
        edge_energy_e2e=float(edge_e2e),
        biot_energy_sum=float(np.asarray(biot_psf, dtype=np.float64).sum()),
        e2e_energy_sum=float(np.asarray(e2e_psf, dtype=np.float64).sum()),
        radial_profile_rmse=radial_rmse,
    )


def _second_moment_mm2(psf: np.ndarray, pixel_pitch_mm: float, centroid_xy_mm: Iterable[float]) -> float:
    h, w = psf.shape
    coord = (np.arange(h, dtype=np.float64) - (h - 1.0) / 2.0) * float(pixel_pitch_mm)
    yy, xx = np.meshgrid(coord, coord, indexing="ij")
    cx, cy = centroid_xy_mm
    return float(np.sum(psf * ((xx - cx) ** 2 + (yy - cy) ** 2)))


def _edge_energy(psf: np.ndarray, edge_width_px: int) -> float:
    edge = int(edge_width_px)
    if edge <= 0:
        return 0.0
    mask = np.zeros_like(psf, dtype=bool)
    mask[:edge, :] = True
    mask[-edge:, :] = True
    mask[:, :edge] = True
    mask[:, -edge:] = True
    return float(psf[mask].sum())


def radial_profile(psf: np.ndarray, pixel_pitch_mm: float, *, bin_count: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return radial mean profile from a linear, energy-normalized PSF."""
    psf_norm = sanitize_and_energy_normalize_psf(psf)
    h, w = psf_norm.shape
    if h != w:
        raise ValueError("radial profile expects a square PSF")
    bins = int(bin_count or (h // 2))
    coord = (np.arange(h, dtype=np.float64) - (h - 1.0) / 2.0) * float(pixel_pitch_mm)
    yy, xx = np.meshgrid(coord, coord, indexing="ij")
    rr = np.sqrt(xx**2 + yy**2)
    edges = np.linspace(0.0, rr.max(), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profile = np.zeros((bins,), dtype=np.float64)
    for i in range(bins):
        mask = (rr >= edges[i]) & (rr < edges[i + 1])
        profile[i] = float(psf_norm[mask].mean()) if np.any(mask) else 0.0
    return centers, profile


def save_surface_fit_outputs(
    output_dir: str | Path,
    *,
    label: str,
    x_mm: torch.Tensor,
    y_mm: torch.Tensor,
    target_sag_mm: torch.Tensor,
    reconstructed_sag_mm: torch.Tensor,
    mask: torch.Tensor,
    result: SurfaceFitResult,
) -> None:
    """Save numeric and visual surface-fit diagnostics."""
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / label
    target = target_sag_mm.detach().cpu().numpy()
    recon = reconstructed_sag_mm.detach().cpu().numpy()
    valid_mask = mask.detach().cpu().numpy().astype(bool)
    error_um = (recon - target) * 1.0e3
    np.save(prefix.with_name(prefix.name + "_target_sag_mm.npy"), target)
    np.save(prefix.with_name(prefix.name + "_reconstructed_sag_mm.npy"), recon)
    np.save(prefix.with_name(prefix.name + "_error_um.npy"), error_um)

    pd.DataFrame(
        {
            "x_mm": x_mm.detach().cpu().numpy().reshape(-1),
            "y_mm": y_mm.detach().cpu().numpy().reshape(-1),
            "target_sag_mm": target.reshape(-1),
            "reconstructed_sag_mm": recon.reshape(-1),
            "error_um": error_um.reshape(-1),
            "valid_aperture": valid_mask.reshape(-1),
        }
    ).to_csv(prefix.with_name(prefix.name + "_surface_samples.csv"), index=False)

    display_error = np.where(valid_mask, error_um, np.nan)
    display_target = np.where(valid_mask, target, np.nan)
    display_recon = np.where(valid_mask, recon, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, data, title, cmap in [
        (axes[0], display_target, "BIOT target sag (mm)", "viridis"),
        (axes[1], display_recon, "e2e B-spline sag (mm)", "viridis"),
        (axes[2], display_error, "reconstruction error (um)", "coolwarm"),
    ]:
        im = ax.imshow(data.T, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x sample")
        ax.set_ylabel("y sample")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        f"{label}: RMSE={result.rmse_mm * 1.0e3:.3f} um, "
        f"P95={result.p95_abs_mm * 1.0e3:.3f} um"
    )
    fig.savefig(prefix.with_name(prefix.name + "_surface_fit.png"), dpi=160)
    plt.close(fig)


def save_psf_comparison_outputs(
    output_dir: str | Path,
    *,
    biot_psf: np.ndarray,
    e2e_psf: np.ndarray,
    result: PSFComparisonResult,
) -> None:
    """Save numeric and visual PSF comparison diagnostics."""
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    biot = sanitize_and_energy_normalize_psf(biot_psf)
    e2e = sanitize_and_energy_normalize_psf(e2e_psf)
    diff = e2e - biot
    np.save(out / "biot_psf_energy_normalized.npy", biot)
    np.save(out / "e2e_psf_energy_normalized.npy", e2e)
    np.save(out / "psf_difference.npy", diff)

    pd.DataFrame([result.as_dict()]).to_csv(out / "psf_similarity_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = [
        (biot, "BIOT FFT PSF linear", "magma"),
        (e2e, "e2e Gaussian ray PSF linear", "magma"),
        (diff, "e2e - BIOT", "coolwarm"),
    ]
    for ax, (data, title, cmap) in zip(axes, panels):
        im = ax.imshow(data, origin="lower", cmap=cmap)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        f"NCC={result.normalized_cross_correlation:.4f}, "
        f"L1={result.l1:.4f}, centroid dx/dy="
        f"({result.centroid_dx_mm:.4g}, {result.centroid_dy_mm:.4g}) mm"
    )
    fig.savefig(out / "psf_similarity.png", dpi=160)
    plt.close(fig)

    radius_mm, biot_profile = radial_profile(biot, 1.0)
    _, e2e_profile = radial_profile(e2e, 1.0)
    pd.DataFrame(
        {
            "radius_px_equivalent": radius_mm,
            "biot_radial_mean": biot_profile,
            "e2e_radial_mean": e2e_profile,
            "difference": e2e_profile - biot_profile,
        }
    ).to_csv(out / "radial_profile.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(radius_mm, biot_profile, label="BIOT FFT")
    ax.plot(radius_mm, e2e_profile, label="e2e Gaussian")
    ax.set_xlabel("radius (px equivalent)")
    ax.set_ylabel("linear radial mean")
    ax.legend()
    fig.savefig(out / "radial_profile_linear.png", dpi=160)
    plt.close(fig)
