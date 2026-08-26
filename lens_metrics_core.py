"""Lens geometric evaluation helpers for power, astigmatism, and distortion.

All public functions use millimetres for length, degrees for field angle,
nanometres for wavelength input, dioptres for power output, and relative
unitless values for distortion. Ray tracing runs on the supplied Lensdata
device. Core tensor calculations keep autograd where the underlying Lensdata
trace supports it; file saving and plotting detach tensors to NumPy.
"""

from __future__ import annotations

import csv
import json
import os
import struct
import warnings
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull

from averfang import _trim_map_for_plot, compute_averfang_maps, load_sag_xlsx
from basics import Ray
from optics import CoordinateBreak, GridSag, Lensdata


EPS = 1e-12
LEGACY_OPTICAL_DIFF = 1e-6
DEFAULT_LENS_FRONT_INDEX = 1
DEFAULT_LENS_BACK_INDEX = 2
DEFAULT_LENS_FRONT_EXCEL_ROW = 5
DEFAULT_LENS_BACK_EXCEL_ROW = 6
DEFAULT_LEGACY_H0_MM = 10.0


@dataclass
class EyePositions:
    """Eye model reference planes in the Lensdata global coordinate system.

    Attributes:
        eye_center_z_mm: Vertex z position of the Excel CB surface, in mm.
        pupil_z_mm: Vertex z position of the Excel aperture A surface, in mm.
        cb_index: Surface index of the coordinate break.
        aperture_index: Surface index of the aperture surface.
    """

    eye_center_z_mm: float
    pupil_z_mm: float
    cb_index: int
    aperture_index: int


@dataclass
class LegacyLensAdapter:
    """EyeGlassSystem-compatible eyeglass surface mapping.

    All distances are in mm and surface indices are Python `Lensdata.surfaces`
    indices. The adapter intentionally traces only through the eyeglass front
    and back surfaces, matching `SingleEyeLens` rather than the full eye model.
    """

    lens: Lensdata
    eye: EyePositions
    lens_front_index: int
    lens_back_index: int
    h_glass_mm: float
    n0: float
    n1: float
    n2: float
    c0: float
    c1: float
    h0_mm: float = DEFAULT_LEGACY_H0_MM

    @property
    def metadata(self) -> Dict[str, object]:
        return {
            "compatibility_mode": "EyeGlassSystem.SingleEyeLens",
            "lens_front_excel_row": DEFAULT_LENS_FRONT_EXCEL_ROW,
            "lens_back_excel_row": DEFAULT_LENS_BACK_EXCEL_ROW,
            "lens_front_index": int(self.lens_front_index),
            "lens_back_index": int(self.lens_back_index),
            "h_eye_center_mm": float(self.eye.eye_center_z_mm),
            "h_pupil_mm": float(self.eye.pupil_z_mm),
            "cb_index": int(self.eye.cb_index),
            "aperture_index": int(self.eye.aperture_index),
            "h_glass_mm": float(self.h_glass_mm),
            "n0": float(self.n0),
            "n1": float(self.n1),
            "n2": float(self.n2),
            "surf0_center_curvature_1_per_mm": float(self.c0),
            "surf1_center_curvature_1_per_mm": float(self.c1),
            "legacy_h0_mm": float(self.h0_mm),
            "cb_tilt_policy": "legacy_ray_direction_only_no_cli_field_tilt",
        }


def resolve_device(device: str) -> torch.device:
    """Return a torch device for CLI input `auto`, `cpu`, or `cuda`."""

    if device == "auto":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return torch.device("cuda:0")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device}")


def load_lens(
    excel_path: str | Path,
    *,
    device: torch.device,
    fov_deg: float = 25.0,
    aperture_mm: float = 2.0,
    wavelength_nm: float = 555.0,
) -> Lensdata:
    """Load a lens for geometric evaluation.

    Args:
        excel_path: Lens Excel path.
        device: Torch device used by ray tracing.
        fov_deg: Maximum field angle used by Lensdata, in degree.
        aperture_mm: System aperture radius used by Lensdata, in mm.
        wavelength_nm: Center wavelength, in nm.

    Returns:
        Loaded Lensdata object. GPU is supported through `device`. Loading and
        ray tracing use Lensdata tensor logic; no explicit detach is performed.
    """

    lens = Lensdata(device=device)
    lens.aperture = aperture_mm
    lens.view_type = "angle"
    lens.FOV = fov_deg
    wavelength = torch.tensor([float(wavelength_nm)], dtype=torch.float64, device=device)
    lens.wavelengths = wavelength
    lens.wavelengths_center = wavelength
    lens.aimming = True
    lens.load_file(Path(excel_path), extension=".xlsx")
    return lens


def extract_eye_positions(lens: Lensdata) -> EyePositions:
    """Extract CB and aperture vertex positions from a loaded Lensdata object.

    Coordinates are Lensdata global z coordinates in mm. The CB surface is
    interpreted as the eye rotation centre. The aperture surface is interpreted
    as the pupil plane. GPU/autograd are not relevant for this metadata helper.
    """

    cb_index = None
    for idx, surface in enumerate(lens.surfaces):
        if isinstance(surface, CoordinateBreak) or getattr(surface, "type", None) == "CB":
            cb_index = idx
            break
    if cb_index is None:
        raise ValueError("No CoordinateBreak (CB) surface found for eye rotation center")
    if lens.aperture_ind is None:
        raise ValueError("No aperture (A) surface found for pupil plane")

    return EyePositions(
        eye_center_z_mm=float(_scalar(lens.surfaces[cb_index].position)),
        pupil_z_mm=float(_scalar(lens.surfaces[lens.aperture_ind].position)),
        cb_index=int(cb_index),
        aperture_index=int(lens.aperture_ind),
    )


def _effective_center_curvature(surface: object) -> float:
    """Estimate the vertex curvature of a lens surface.

    GridSag sets ``self.c = 0`` (:obj:`optics.GridSag`) even though the
    actual sag may have non-zero curvature at the centre.  For such surfaces
    we numerically differentiate the B-spline at ``(x, y) = (0, 0)``.  Under
    the paraxial approximation ``z ≈ c·(x² + y²)/2`` the second derivatives
    satisfy ``∂²z/∂x² = ∂²z/∂y² = c`` at the vertex.

    For all other surface types the built-in ``.c`` attribute is used.
    """
    if isinstance(surface, GridSag):
        d2z_dx2 = float(surface.spline(0, 0, dx=2, dy=0, grid=False))
        d2z_dy2 = float(surface.spline(0, 0, dx=0, dy=2, grid=False))
        return (d2z_dx2 + d2z_dy2) / 2.0
    return float(_scalar(getattr(surface, "c", 0.0)))


def build_legacy_adapter(
    lens: Lensdata,
    *,
    lens_front_index: Optional[int] = None,
    lens_back_index: Optional[int] = None,
    wavelength_nm: float = 555.0,
) -> LegacyLensAdapter:
    """Build the EyeGlassSystem-compatible eyeglass adapter.

    Args:
        lens: Loaded `Lensdata`.
        lens_front_index: Python index of the eyeglass front surface. Defaults
            to Excel row 5, `surfaces[1]`, for `eye_image_glass.xlsx`.
        lens_back_index: Python index of the eyeglass back surface. Defaults to
            Excel row 6, `surfaces[2]`, for `eye_image_glass.xlsx`.
        wavelength_nm: Wavelength used to evaluate material refractive index.

    Returns:
        Adapter with mm coordinates, refractive indices, and curvature metadata.
        GPU/autograd are preserved in ray tracing; metadata is detached.
    """

    front = DEFAULT_LENS_FRONT_INDEX if lens_front_index is None else int(lens_front_index)
    back = DEFAULT_LENS_BACK_INDEX if lens_back_index is None else int(lens_back_index)
    if front < 0 or back < 0 or front >= len(lens.surfaces) or back >= len(lens.surfaces):
        raise ValueError(f"Invalid eyeglass surface indices: front={front}, back={back}")
    if front >= back:
        raise ValueError(f"lens_front_index must be smaller than lens_back_index: {front}, {back}")
    if back + 1 >= len(lens.materials):
        raise ValueError("Lens materials do not contain media around the requested eyeglass surfaces")

    wavelength = _wavelength_tensor(lens, wavelength_nm)
    front_surface = lens.surfaces[front]
    back_surface = lens.surfaces[back]
    return LegacyLensAdapter(
        lens=lens,
        eye=extract_eye_positions(lens),
        lens_front_index=front,
        lens_back_index=back,
        h_glass_mm=float(_scalar(back_surface.position) - _scalar(front_surface.position)),
        n0=float(_scalar(lens.materials[front].ior(wavelength))),
        n1=float(_scalar(lens.materials[front + 1].ior(wavelength))),
        n2=float(_scalar(lens.materials[back + 1].ior(wavelength))),
        c0=_effective_center_curvature(front_surface),
        c1=_effective_center_curvature(back_surface),
    )


def sample_field_1d(
    fov_deg: float,
    field_num: int,
    *,
    axis: str = "y",
    tan_uniform: bool = False,
) -> Dict[str, np.ndarray]:
    """Generate one-dimensional field samples.

    Args:
        fov_deg: Maximum absolute field angle, in degree.
        field_num: Number of samples.
        axis: `x` or `y`. The other axis remains zero.
        tan_uniform: If True, samples uniformly in tan(theta) and converts back
            to degree; otherwise samples uniformly in degree.

    Returns:
        Dict with `theta_deg`, `field_x_deg`, and `field_y_deg`, each shape
        `[field_num]`. No GPU/autograd behaviour applies.
    """

    if field_num < 2:
        raise ValueError("field_num must be at least 2")
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    if tan_uniform:
        max_tan = np.tan(np.deg2rad(float(fov_deg)))
        theta = np.rad2deg(np.arctan(np.linspace(-max_tan, max_tan, field_num)))
    else:
        theta = np.linspace(-float(fov_deg), float(fov_deg), field_num)
    field_x = theta if axis == "x" else np.zeros_like(theta)
    field_y = theta if axis == "y" else np.zeros_like(theta)
    return {"theta_deg": theta, "field_x_deg": field_x, "field_y_deg": field_y}


def sample_legacy_positive_fields(fov_deg: float, field_num: int, *, tan_uniform: bool) -> np.ndarray:
    """Sample one-sided positive Y fields following `GenerateCheifRays`.

    Args:
        fov_deg: Positive maximum field angle in degree.
        field_num: Number of samples.
        tan_uniform: If True, sample uniformly in tangent space; otherwise
            sample directly in degree.

    Returns:
        NumPy array of shape `[field_num]` in degree. No GPU/autograd behaviour.
    """

    if field_num < 2:
        raise ValueError("field_num must be at least 2")
    if tan_uniform:
        values = np.linspace(LEGACY_OPTICAL_DIFF, np.tan(np.deg2rad(float(fov_deg))), int(field_num))
        return np.rad2deg(np.arctan(values))
    return np.linspace(LEGACY_OPTICAL_DIFF, float(fov_deg), int(field_num))


def display_positive_field_angles(theta_deg: np.ndarray) -> np.ndarray:
    """Return display/report angles with the first one-dimensional field at 0 deg.

    The legacy ray generator traces the first sample at `LEGACY_OPTICAL_DIFF`
    degrees to avoid singular far-field magnification. Reports and plots expose
    that sample as the on-axis default view `(field_x, field_y) = (0, 0)`.
    """

    display = np.asarray(theta_deg, dtype=float).copy()
    if display.size:
        display[0] = 0.0
    return display


def sample_legacy_grid_fields(fov_x_deg: float, fov_y_deg: float, field_num: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample two-dimensional fields as `GenerateCheifRays3D`.

    The x/y directions are uniformly sampled in tangent space over the symmetric
    field range and returned as two `[field_num, field_num]` degree grids.
    """

    if field_num < 1:
        raise ValueError("field_num must be at least 1")
    if int(field_num) == 1:
        tx = np.array([np.tan(np.deg2rad(float(fov_x_deg)))], dtype=float)
        ty = np.array([np.tan(np.deg2rad(float(fov_y_deg)))], dtype=float)
    else:
        tx = np.linspace(-np.tan(np.deg2rad(float(fov_x_deg))), np.tan(np.deg2rad(float(fov_x_deg))), int(field_num))
        ty = np.linspace(-np.tan(np.deg2rad(float(fov_y_deg))), np.tan(np.deg2rad(float(fov_y_deg))), int(field_num))
    return np.meshgrid(np.rad2deg(np.arctan(tx)), np.rad2deg(np.arctan(ty)), indexing="xy")


def compute_trace_power_astigmatism(
    lens: Lensdata,
    *,
    fov_deg: float = 25.0,
    field_num: int = 51,
    axis: str = "y",
    wavelength_nm: float = 555.0,
    differential_aperture_mm: float = 0.01,
    pupil_z_mm: Optional[float] = None,
    focal_power_D: float = 0.0,
    lens_front_index: Optional[int] = None,
    lens_back_index: Optional[int] = None,
) -> Dict[str, object]:
    """Compute legacy trace-based sagittal/meridional power curves.

    Input field angles are degree. Wavelength is nm. Differential ray offsets
    and pupil plane coordinates are mm. Returned arrays are one-dimensional
    NumPy arrays with length `field_num`; power values are dioptres. Ray tracing
    runs on the Lensdata device and keeps tensors until final export conversion.
    """

    if axis != "y":
        raise ValueError("Legacy EyeGlassSystem focal power evaluation only supports axis='y'")
    adapter = build_legacy_adapter(
        lens,
        lens_front_index=lens_front_index,
        lens_back_index=lens_back_index,
        wavelength_nm=wavelength_nm,
    )
    pupil_z = adapter.eye.pupil_z_mm if pupil_z_mm is None else float(pupil_z_mm)
    rows: List[Dict[str, float | bool]] = []
    meta = {
        "mode": "power",
        "axis": axis,
        "fov_deg": float(fov_deg),
        "field_num": int(field_num),
        "wavelength_nm": float(wavelength_nm),
        "differential_aperture_mm": float(differential_aperture_mm),
        "legacy_paraxial_aperture_mm": float(LEGACY_OPTICAL_DIFF / 2.0),
        "pupil_z_mm": float(pupil_z),
        "focal_power_D": float(focal_power_D),
        "original_reference_function": "SingleEyeLens.eval_focal_power",
        "sagittal_meridian_naming": (
            "physically_correct: fp_sagittal uses x-offset (sagittal) focus, "
            "fp_meridian uses y-offset (meridian) focus"
        ),
        "compatibility_deviation": (
            "BIOT deviates from MATLAB naming convention: MATLAB labels "
            "1000/z_min as sagittal and 1000/z_max as meridian (swapped). "
            "BIOT uses physically correct labels."
        ),
        "sampling_policy": "GenerateCheifRays uniform one-sided positive y field",
        "initial_view_field_x_deg": 0.0,
        "initial_view_field_y_deg": 0.0,
        "power_reference_policy": "h_pupil plane plus thick-lens principal plane",
        **adapter.metadata,
    }

    try:
        chiefs, bundle, theta, fields = generate_forward_rays_legacy(
            adapter,
            field_num=field_num,
            fov_deg=fov_deg,
            wavelength_nm=wavelength_nm,
        )
        chiefs_out, valid_chiefs = trace_legacy_lens(adapter, chiefs)
        bundle_out, valid_bundle = trace_legacy_lens(adapter, bundle)
        if not bool(valid_chiefs.detach().cpu().all().item()):
            raise RuntimeError("legacy forward chief trace has invalid rays")

        bundle_local = rotate_about_x_local(bundle_out, -fields, adapter.eye.eye_center_z_mm)
        chiefs_local = rotate_about_x_local(chiefs_out, -theta, adapter.eye.eye_center_z_mm)
        bundle_pupil = Ray(
            intersect_z_plane(bundle_local.o, bundle_local.d, pupil_z),
            bundle_local.d,
            bundle_local.wavelength,
            bundle_local.weight,
            bundle_local.phase,
            device=bundle_local.o.device,
        )
        chiefs_pupil = Ray(
            intersect_z_plane(chiefs_local.o, chiefs_local.d, pupil_z),
            chiefs_local.d,
            chiefs_local.wavelength,
            chiefs_local.weight,
            chiefs_local.phase,
            device=chiefs_local.o.device,
        )
        del chiefs_pupil  # kept for parity with the original flow; rays drive the formulas below.

        o, d = ray_values(bundle_pupil)
        valid_np = valid_bundle.detach().cpu().reshape(-1).numpy().astype(bool)
        denom = d[:, 0] ** 2 + d[:, 1] ** 2
        t = np.full_like(denom, np.nan, dtype=float)
        np.divide(-(o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1]), denom, out=t, where=np.abs(denom) > 1e-30)
        z_cross = o[:, 2] + d[:, 2] * t

        h = adapter.h_glass_mm
        D = (adapter.n1 - adapter.n0) * (adapter.n2 - adapter.n1) * h - adapter.n1 * (
            (adapter.n2 - adapter.n1) / adapter.c0 if abs(adapter.c0) > EPS else 0.0
        ) - adapter.n1 * ((adapter.n1 - adapter.n0) / adapter.c1 if abs(adapter.c1) > EPS else 0.0)
        if abs(D) <= EPS:
            raise RuntimeError("legacy thick-lens principal plane denominator is too small")
        d_main_plane = adapter.n2 * (adapter.n1 - adapter.n0) * adapter.h_glass_mm / adapter.c1 / D
        h_main_plane = adapter.h_glass_mm + d_main_plane
        meta["h_main_plane_mm"] = float(h_main_plane)
        meta["d_main_plane_mm"] = float(d_main_plane)
        meta["trace_first_field_y_deg"] = float(theta[0]) if len(theta) else np.nan
        theta_display = display_positive_field_angles(theta)

        for i, theta_i in enumerate(theta_display):
            idx0 = i * 4
            z_values = z_cross[idx0 : idx0 + 4]
            valid_field = bool(valid_np[idx0 : idx0 + 4].all()) and np.isfinite(z_values[:2]).all()
            z_sagittal = z_values[0] - h_main_plane
            z_meridian = z_values[1] - h_main_plane
            fp_sag = 1000.0 / z_sagittal if valid_field and abs(z_sagittal) > EPS else np.nan
            fp_mer = 1000.0 / z_meridian if valid_field and abs(z_meridian) > EPS else np.nan
            fp_mean = 1000.0 / 0.5 / (z_meridian + z_sagittal) if valid_field and abs(z_meridian + z_sagittal) > EPS else np.nan
            valid_row = bool(valid_field and np.isfinite([fp_sag, fp_mer, fp_mean]).all())
            rows.append(
                {
                    "theta_deg": float(theta_i),
                    "fp_sagittal_D": float(fp_sag),
                    "fp_meridian_D": float(fp_mer),
                    "fp_mean_D": float(fp_mean),
                    "astigmatism_D": float(fp_sag - fp_mer) if valid_row else np.nan,
                    "fp_sagittal_error_D": float(fp_sag + float(focal_power_D)) if valid_row else np.nan,
                    "fp_meridian_error_D": float(fp_mer + float(focal_power_D)) if valid_row else np.nan,
                    "fp_mean_error_D": float(fp_mean + float(focal_power_D)) if valid_row else np.nan,
                    "valid": valid_row,
                }
            )
    except Exception as exc:
        for theta_i in display_positive_field_angles(sample_legacy_positive_fields(fov_deg, field_num, tan_uniform=False)):
            rows.append(
                {
                    "theta_deg": float(theta_i),
                    "fp_sagittal_D": np.nan,
                    "fp_meridian_D": np.nan,
                    "fp_mean_D": np.nan,
                    "astigmatism_D": np.nan,
                    "fp_sagittal_error_D": np.nan,
                    "fp_meridian_error_D": np.nan,
                    "fp_mean_error_D": np.nan,
                    "valid": False,
                    "error": str(exc),
                }
            )

    if not any(bool(r["valid"]) for r in rows):
        raise RuntimeError("Power/astigmatism computation failed for all field samples")
    meta["valid_count"] = int(sum(bool(r["valid"]) for r in rows))
    meta["invalid_count"] = int(len(rows) - meta["valid_count"])
    return table_result(rows, meta)


def compute_power_astigmatism(
    lens: Lensdata,
    *,
    fov_deg: float = 25.0,
    field_num: int = 51,
    axis: str = "y",
    wavelength_nm: float = 555.0,
    differential_aperture_mm: float = 0.01,
    pupil_z_mm: Optional[float] = None,
    focal_power_D: float = 0.0,
    lens_front_index: Optional[int] = None,
    lens_back_index: Optional[int] = None,
    crib_diameter_mm: float = 80.0,
) -> Dict[str, object]:
    """Compute footprint-sampled AverFang local power and astigmatism curves.

    Input field angles are degree. The fixed project eyeglass surfaces are
    Excel row 5 / ``surfaces[1]`` for the front sphere and Excel row 6 /
    ``surfaces[2]`` for the rear GridSag. Footprints are traced to the rear
    GridSag surface and used to linearly interpolate an AverFang-style local
    power map. Returned power values are in dioptres. Ray tracing runs on the
    supplied Lensdata device; exported arrays are detached NumPy data.
    """

    if axis != "y":
        raise ValueError("Footprint-sampled AverFang power evaluation only supports axis='y'")
    if lens_front_index is not None or lens_back_index is not None:
        raise ValueError("AverFang footprint power uses fixed Excel row 5/6 surfaces; lens indices are not supported")
    adapter = build_legacy_adapter(lens, lens_front_index=1, lens_back_index=2, wavelength_nm=wavelength_nm)
    front_surface = lens.surfaces[1]
    back_surface = lens.surfaces[2]
    if not isinstance(back_surface, GridSag):
        raise ValueError("Excel row 6 / surfaces[2] must be a GridSag surface")
    sag_path = getattr(back_surface, "sag_file_path", None)
    if sag_path is None:
        raise ValueError("GridSag surface does not record sag_file_path")
    front_radius = 1.0 / adapter.c0 if abs(adapter.c0) > EPS else np.nan
    if not np.isfinite(front_radius) or front_radius <= 0:
        raise ValueError(f"Excel row 5 front surface must have a positive spherical ROC, got {front_radius}")

    sag = load_sag_xlsx(sag_path, grid_shape=back_surface.grid_shape)
    averfang = compute_averfang_maps(
        sag,
        semi_dia_mm=float(back_surface.semi_dia),
        refractive_index=adapter.n1,
        front_radius_mm=float(front_radius),
        center_thickness_mm=adapter.h_glass_mm,
        crib_diameter_mm=float(crib_diameter_mm),
    )
    x_mm = np.asarray(averfang["x_mm"], dtype=float)
    y_mm = np.asarray(averfang["y_mm"], dtype=float)
    power_map = np.asarray(averfang["power_D"], dtype=float)
    astig_map = np.asarray(averfang["astigmatism_D"], dtype=float)
    cyl_raw_map = np.asarray(averfang["cylinder_raw_D"], dtype=float)
    power_interp = RegularGridInterpolator((y_mm, x_mm), power_map, bounds_error=False, fill_value=np.nan)
    astig_interp = RegularGridInterpolator((y_mm, x_mm), astig_map, bounds_error=False, fill_value=np.nan)
    cyl_interp = RegularGridInterpolator((y_mm, x_mm), cyl_raw_map, bounds_error=False, fill_value=np.nan)

    chief, theta = generate_legacy_chief_rays_1d(
        adapter,
        field_num=field_num,
        fov_deg=fov_deg,
        ray_type="uniform",
        wavelength_nm=wavelength_nm,
        pupil_distance_mm=250.0,
    )
    traced, valid = trace_legacy_lens(adapter, chief)
    footprint = traced.o.detach().cpu().reshape(-1, 3).numpy()
    valid_np = valid.detach().cpu().reshape(-1).numpy().astype(bool)
    theta_display = display_positive_field_angles(theta)

    trace_result = compute_trace_power_astigmatism(
        lens,
        fov_deg=fov_deg,
        field_num=field_num,
        axis=axis,
        wavelength_nm=wavelength_nm,
        differential_aperture_mm=differential_aperture_mm,
        pupil_z_mm=pupil_z_mm,
        focal_power_D=focal_power_D,
        lens_front_index=1,
        lens_back_index=2,
    )
    trace_by_theta = list(trace_result["rows"])

    rows: List[Dict[str, float | bool | str]] = []
    invalid_outside = 0
    invalid_trace = 0
    for i, theta_i in enumerate(theta_display):
        x_fp = float(footprint[i, 0])
        y_fp = float(footprint[i, 1])
        sample = np.array([[y_fp, x_fp]], dtype=float)
        local_mean = float(power_interp(sample)[0])
        local_astig = float(astig_interp(sample)[0])
        local_cyl_raw = float(cyl_interp(sample)[0])
        row_valid = bool(valid_np[i] and np.isfinite([local_mean, local_astig, local_cyl_raw]).all())
        if not valid_np[i]:
            invalid_trace += 1
        elif not row_valid:
            invalid_outside += 1
        if row_valid:
            local_sag = local_mean + 0.5 * local_astig
            local_mer = local_mean - 0.5 * local_astig
            local_mean_error = local_mean + float(focal_power_D)
            local_sag_error = local_sag + float(focal_power_D)
            local_mer_error = local_mer + float(focal_power_D)
        else:
            local_sag = local_mer = local_mean_error = local_sag_error = local_mer_error = np.nan
        trace_row = trace_by_theta[i] if i < len(trace_by_theta) else {}
        trace_mean = float(trace_row.get("fp_mean_D", np.nan))
        rows.append(
            {
                "theta_deg": float(theta_i),
                "footprint_x_mm": x_fp,
                "footprint_y_mm": y_fp,
                "local_sagittal_power_D": float(local_sag),
                "local_meridional_power_D": float(local_mer),
                "local_mean_power_D": float(local_mean) if row_valid else np.nan,
                "local_astigmatism_D": float(local_astig) if row_valid else np.nan,
                "local_sagittal_error_D": float(local_sag_error),
                "local_meridional_error_D": float(local_mer_error),
                "local_mean_error_D": float(local_mean_error),
                "local_cylinder_raw_D": float(local_cyl_raw) if row_valid else np.nan,
                "trace_sagittal_power_D": float(trace_row.get("fp_sagittal_D", np.nan)),
                "trace_meridional_power_D": float(trace_row.get("fp_meridian_D", np.nan)),
                "trace_mean_power_D": trace_mean,
                "trace_astigmatism_D": float(trace_row.get("astigmatism_D", np.nan)),
                "delta_trace_minus_local_mean_D": trace_mean - local_mean if row_valid and np.isfinite(trace_mean) else np.nan,
                "valid": row_valid,
            }
        )

    if not any(bool(r["valid"]) for r in rows):
        raise RuntimeError("AverFang footprint power computation failed for all field samples")

    meta = {
        "mode": "power",
        "power_evaluation_mode": "averfang_footprint_sampled",
        "axis": axis,
        "fov_deg": float(fov_deg),
        "field_num": int(field_num),
        "wavelength_nm": float(wavelength_nm),
        "differential_aperture_mm": float(differential_aperture_mm),
        "pupil_z_mm": float(adapter.eye.pupil_z_mm if pupil_z_mm is None else pupil_z_mm),
        "focal_power_D": float(focal_power_D),
        "original_reference_function": r"D:\MATLAB\matlab仿真原始\AverFang.m",
        "trace_diagnostic_reference_function": "SingleEyeLens.eval_focal_power",
        "power_reference_policy": "chief-ray rear-GridSag footprint samples AverFang local power map",
        "power_split_policy": (
            "AverFang Dss is treated as mean/equivalent sphere; abs(Ass) is split symmetrically into two "
            "nominal principal powers. Sagittal/meridional labels are nominal unless local principal-axis "
            "orientation is evaluated."
        ),
        "astigmatism_sign_policy": "local_astigmatism_D is abs(AverFang Ass); local_cylinder_raw_D preserves signed Ass",
        "initial_view_field_x_deg": 0.0,
        "initial_view_field_y_deg": 0.0,
        "trace_first_field_y_deg": float(theta[0]) if len(theta) else np.nan,
        "averfang_crib_diameter_mm": float(crib_diameter_mm),
        "averfang_sag_file_path": str(sag_path),
        "front_surface_index": 1,
        "back_surface_index": 2,
        "front_surface_type": type(front_surface).__name__,
        "back_surface_type": type(back_surface).__name__,
        "footprint_surface": "rear GridSag, Excel row 6 / surfaces[2]",
        "interpolation": "linear, no extrapolation, circular mask remains NaN",
        "invalid_footprint_outside_averfang_aperture": int(invalid_outside),
        "invalid_trace_count": int(invalid_trace),
        **adapter.metadata,
    }
    meta.update({f"averfang_{k}": v for k, v in dict(averfang["metadata"]).items() if k not in {"valid_count", "invalid_count"}})
    meta["valid_count"] = int(sum(bool(r["valid"]) for r in rows))
    meta["invalid_count"] = int(len(rows) - meta["valid_count"])
    result = table_result(rows, meta)
    result["trace_result"] = trace_result
    result["averfang_map"] = averfang
    return result


def compute_footprint_coverage(
    lens: Lensdata,
    *,
    field_radius_deg: float = 50.0,
    field_sample_count: int = 72,
    pupil_sample_count: int = 72,
    pupil_radius_mm: Optional[float] = None,
    wavelength_nm: float = 555.0,
    crib_diameter_mm: float = 80.0,
) -> Dict[str, object]:
    """Trace a rotating-eye maximum-field pupil footprint on the rear GridSag.

    Args:
        lens: Loaded `Lensdata` system. Ray tracing uses its configured device.
        field_radius_deg: Maximum eyeball rotation angle in degree. This is a
            semi-field angle around the CoordinateBreak eye point.
        field_sample_count: Number of field azimuth samples over 0-360 deg,
            endpoint excluded.
        pupil_sample_count: Number of pupil-boundary samples over 0-360 deg,
            endpoint excluded.
        pupil_radius_mm: Pupil boundary radius in mm. If ``None`` or invalid,
            ``lens.aperture`` is used; if that is unavailable, 2 mm is used.
        wavelength_nm: Wavelength in nm.
        crib_diameter_mm: AverFang power-map physical diameter in mm.

    Returns:
        Dict with field/pupil sample arrays, rear-surface footprint arrays,
        valid mask, convex-hull vertices, AverFang power map and metadata.
        Coordinates are mm on the rear GridSag surface. Exported arrays are CPU
        NumPy data; core tracing keeps the Lensdata tensor device until detach.
    """

    if field_radius_deg <= 0:
        raise ValueError("field_radius_deg must be positive")
    if int(field_sample_count) < 3:
        raise ValueError("field_sample_count must be at least 3")
    if int(pupil_sample_count) < 3:
        raise ValueError("pupil_sample_count must be at least 3")

    adapter = build_legacy_adapter(lens, lens_front_index=1, lens_back_index=2, wavelength_nm=wavelength_nm)
    front_surface = lens.surfaces[1]
    back_surface = lens.surfaces[2]
    if not isinstance(back_surface, GridSag):
        raise ValueError("Excel row 6 / surfaces[2] must be a GridSag surface")
    sag_path = getattr(back_surface, "sag_file_path", None)
    if sag_path is None:
        raise ValueError("GridSag surface does not record sag_file_path")
    front_radius = 1.0 / adapter.c0 if abs(adapter.c0) > EPS else np.nan
    if not np.isfinite(front_radius) or front_radius <= 0:
        raise ValueError(f"Excel row 5 front surface must have a positive spherical ROC, got {front_radius}")

    resolved_pupil_radius = float(pupil_radius_mm) if pupil_radius_mm is not None else np.nan
    pupil_radius_source = "argument"
    if not np.isfinite(resolved_pupil_radius) or resolved_pupil_radius <= 0:
        lens_aperture = getattr(lens, "aperture", np.nan)
        resolved_pupil_radius = float(_scalar(lens_aperture)) if lens_aperture is not None else np.nan
        pupil_radius_source = "Lensdata.aperture"
    if not np.isfinite(resolved_pupil_radius) or resolved_pupil_radius <= 0:
        resolved_pupil_radius = 2.0
        pupil_radius_source = "fallback_2mm"

    sag = load_sag_xlsx(sag_path, grid_shape=back_surface.grid_shape)
    averfang = compute_averfang_maps(
        sag,
        semi_dia_mm=float(back_surface.semi_dia),
        refractive_index=adapter.n1,
        front_radius_mm=float(front_radius),
        center_thickness_mm=adapter.h_glass_mm,
        crib_diameter_mm=float(crib_diameter_mm),
    )
    x_mm = np.asarray(averfang["x_mm"], dtype=float)
    y_mm = np.asarray(averfang["y_mm"], dtype=float)
    power_map = np.asarray(averfang["power_D"], dtype=float)
    power_interp = RegularGridInterpolator((y_mm, x_mm), power_map, bounds_error=False, fill_value=np.nan)

    field_phi = np.linspace(0.0, 2.0 * np.pi, int(field_sample_count), endpoint=False)
    pupil_phi = np.linspace(0.0, 2.0 * np.pi, int(pupil_sample_count), endpoint=False)
    field_angle_rad = np.deg2rad(float(field_radius_deg))
    field_slope_tan = np.tan(field_angle_rad)
    field_slope_x = field_slope_tan * np.cos(field_phi)
    field_slope_y = field_slope_tan * np.sin(field_phi)
    field_x = np.rad2deg(np.arctan(field_slope_x))
    field_y = np.rad2deg(np.arctan(field_slope_y))
    gaze_directions = np.stack(
        (
            np.sin(field_angle_rad) * np.cos(field_phi),
            np.sin(field_angle_rad) * np.sin(field_phi),
            -np.full_like(field_phi, np.cos(field_angle_rad)),
        ),
        axis=1,
    )
    pupil_x = resolved_pupil_radius * np.cos(pupil_phi)
    pupil_y = resolved_pupil_radius * np.sin(pupil_phi)

    origins = np.zeros((int(field_sample_count) * int(pupil_sample_count), 3), dtype=float)
    directions = np.zeros_like(origins)
    eye_center = np.array([0.0, 0.0, adapter.eye.eye_center_z_mm], dtype=float)
    pupil_center = np.array([0.0, 0.0, adapter.eye.pupil_z_mm], dtype=float)
    base_points = pupil_center.reshape(1, 3) + np.stack(
        (pupil_x, pupil_y, np.zeros_like(pupil_x)),
        axis=1,
    )
    base_gaze = np.array([0.0, 0.0, -1.0], dtype=float)
    base_pupil_offset = base_points - eye_center.reshape(1, 3)
    for field_idx, gaze_dir in enumerate(gaze_directions):
        rotation = rotation_from_vectors(base_gaze, gaze_dir)
        start = field_idx * int(pupil_sample_count)
        stop = start + int(pupil_sample_count)
        origins[start:stop] = eye_center.reshape(1, 3) + base_pupil_offset @ rotation.T
        directions[start:stop] = gaze_dir.reshape(1, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    rays0 = make_ray_from_arrays(lens, origins, directions, wavelength_nm)
    traced, trace_valid = trace_legacy_lens(adapter, rays0)
    footprint = traced.o.detach().cpu().reshape(-1, 3).numpy()
    trace_valid_np = trace_valid.detach().cpu().reshape(-1).numpy().astype(bool)
    local_power = power_interp(np.column_stack((footprint[:, 1], footprint[:, 0])))
    valid = trace_valid_np & np.isfinite(footprint[:, 0]) & np.isfinite(footprint[:, 1]) & np.isfinite(local_power)
    valid_points = footprint[valid, :2]
    if valid_points.shape[0] < 3:
        raise RuntimeError(
            f"Footprint convex hull requires at least 3 valid points, got {valid_points.shape[0]}"
        )
    hull = ConvexHull(valid_points)
    hull_vertices_xy = valid_points[hull.vertices]

    total_count = int(valid.size)
    valid_count = int(np.count_nonzero(valid))
    metadata = {
        "mode": "footprint-coverage",
        "footprint_model": "rotating_eye_about_cb",
        "field_sampling": "maximum eyeball rotation cone, azimuth endpoint excluded",
        "field_radial_coordinate_policy": (
            "constant 3D rotation angle: angle(gaze_direction, -Z) = field_radius_deg; "
            "field_x_deg/field_y_deg are component angles derived from the rotated gaze direction"
        ),
        "field_radius_deg": float(field_radius_deg),
        "field_radius_tan": float(field_slope_tan),
        "field_sample_count": int(field_sample_count),
        "field_phi_endpoint_included": False,
        "pupil_sampling": "pupil boundary ring, azimuth endpoint excluded",
        "pupil_radius_mm": float(resolved_pupil_radius),
        "pupil_radius_source": pupil_radius_source,
        "pupil_sample_count": int(pupil_sample_count),
        "pupil_phi_endpoint_included": False,
        "eye_rotation_center_z_mm": float(adapter.eye.eye_center_z_mm),
        "pupil_plane_z_mm": float(adapter.eye.pupil_z_mm),
        "pupil_motion_policy": "pupil center and pupil boundary basis rotate rigidly around the CB eye point",
        "ray_direction_policy": "all rays for one field are parallel to the rotated gaze direction",
        "footprint_surface": "rear GridSag, Excel row 6 / surfaces[2]",
        "front_surface_index": 1,
        "back_surface_index": 2,
        "background_map": "AverFang Power (D)",
        "averfang_crib_diameter_mm": float(crib_diameter_mm),
        "averfang_sag_file_path": str(sag_path),
        "coordinate_range_x_mm": [float(x_mm[0]), float(x_mm[-1])],
        "coordinate_range_y_mm": [float(y_mm[0]), float(y_mm[-1])],
        "total_ray_count": total_count,
        "valid_ray_count": valid_count,
        "invalid_ray_count": int(total_count - valid_count),
        "invalid_ray_fraction": float((total_count - valid_count) / total_count),
        "invalid_trace_count": int(np.count_nonzero(~trace_valid_np)),
        "invalid_outside_averfang_power_map_count": int(np.count_nonzero(trace_valid_np & ~np.isfinite(local_power))),
        "convex_hull_vertex_count": int(hull_vertices_xy.shape[0]),
        "convex_hull_area_mm2": float(hull.volume),
        "convex_hull_perimeter_mm": float(hull.area),
        **adapter.metadata,
    }
    return {
        "field_phi_deg": np.rad2deg(field_phi),
        "field_x_deg": field_x,
        "field_y_deg": field_y,
        "field_slope_x": field_slope_x,
        "field_slope_y": field_slope_y,
        "gaze_direction": gaze_directions,
        "pupil_phi_deg": np.rad2deg(pupil_phi),
        "pupil_x_mm": pupil_x,
        "pupil_y_mm": pupil_y,
        "footprint_x_mm": footprint[:, 0].reshape(int(field_sample_count), int(pupil_sample_count)),
        "footprint_y_mm": footprint[:, 1].reshape(int(field_sample_count), int(pupil_sample_count)),
        "footprint_z_mm": footprint[:, 2].reshape(int(field_sample_count), int(pupil_sample_count)),
        "local_power_D": local_power.reshape(int(field_sample_count), int(pupil_sample_count)),
        "trace_valid": trace_valid_np.reshape(int(field_sample_count), int(pupil_sample_count)),
        "valid": valid.reshape(int(field_sample_count), int(pupil_sample_count)),
        "convex_hull_xy_mm": hull_vertices_xy,
        "averfang_map": averfang,
        "metadata": metadata,
    }


def compute_distortion_curve(
    lens: Lensdata,
    *,
    fov_deg: float = 25.0,
    field_num: int = 51,
    axis: str = "y",
    distortion_type: str = "rotating_eye_far",
    wavelength_nm: float = 555.0,
    near_object_distance_mm: float = 250.0,
    pupil_distance_mm: float = 250.0,
    lens_front_index: Optional[int] = None,
    lens_back_index: Optional[int] = None,
) -> Dict[str, object]:
    """Compute one-dimensional magnification and relative distortion.

    Angles are degree, distances are mm. Output arrays have shape
    `[field_num]`. Far modes use tangent angle ratio; near modes use object
    plane height ratio at z = -near_object_distance_mm. Ray tracing runs on the
    Lensdata device; final table values are detached NumPy arrays.
    """

    if axis != "y":
        raise ValueError("Legacy EyeGlassSystem distortion curve only supports axis='y'")
    legacy_type = normalize_distortion_type(distortion_type)
    if legacy_type not in {
        "rotate_eye_far",
        "fix_eye_far",
        "rotate_eye_near",
        "fix_eye_near",
        "dist_in_hand",
        "standard",
    }:
        raise ValueError(f"Unsupported distortion_type: {distortion_type}")
    adapter = build_legacy_adapter(
        lens,
        lens_front_index=lens_front_index,
        lens_back_index=lens_back_index,
        wavelength_nm=wavelength_nm,
    )
    is_far = legacy_type.endswith("_far") or legacy_type == "standard"
    use_far_slope_reference = legacy_type in {"rotate_eye_far", "fix_eye_far"}
    use_near_slope_reference = legacy_type in {"rotate_eye_near", "fix_eye_near", "dist_in_hand"}
    use_trace_slope_reference = use_far_slope_reference or use_near_slope_reference
    ray_type = "dist_" + legacy_type if legacy_type != "standard" else "dist_standard"
    rays0, theta = generate_legacy_chief_rays_1d(
        adapter,
        field_num=field_num,
        fov_deg=fov_deg,
        ray_type=ray_type,
        wavelength_nm=wavelength_nm,
        pupil_distance_mm=pupil_distance_mm,
    )
    theta_display = display_positive_field_angles(theta)
    rows: List[Dict[str, float | bool]] = []
    try:
        rays, valid = trace_legacy_lens(adapter, rays0)
        o, d = ray_values(rays)
        o0, d0 = ray_values(rays0)
        valid_np = valid.detach().cpu().reshape(-1).numpy().astype(bool)
        theta_in = np.rad2deg(np.arctan(np.abs(d[:, 1] / d[:, 2])))
        actual = np.full(int(field_num), np.nan, dtype=float)
        ideal = np.full(int(field_num), np.nan, dtype=float)
        magnif = np.full(int(field_num), np.nan, dtype=float)
        if legacy_type == "standard":
            fp1, _, fp, delta_h = center_focal_power(adapter)
            height_ideal = -np.tan(np.deg2rad(theta_in)) / fp / (1.0 - delta_h / adapter.n1 * fp1)
            height_actual = -(
                1.0 - (adapter.eye.eye_center_z_mm - adapter.h_glass_mm) * fp
            ) / fp * np.tan(np.deg2rad(theta))
            actual = height_actual
            ideal = height_ideal
            magnif = height_actual / height_ideal
        elif is_far:
            magnif = np.divide(
                np.tan(np.deg2rad(theta_in)),
                np.tan(np.deg2rad(theta)),
                out=np.full(int(field_num), np.nan, dtype=float),
                where=np.abs(theta) > 1e-8,
            )
        else:
            actual_p = intersect_z_plane(
                torch.as_tensor(o, dtype=torch.float64, device=lens.device).reshape(-1, 1, 3),
                torch.as_tensor(d, dtype=torch.float64, device=lens.device).reshape(-1, 1, 3),
                -near_object_distance_mm,
            )
            ideal_p = intersect_z_plane(
                torch.as_tensor(o0, dtype=torch.float64, device=lens.device).reshape(-1, 1, 3),
                torch.as_tensor(d0, dtype=torch.float64, device=lens.device).reshape(-1, 1, 3),
                -near_object_distance_mm,
            )
            actual = actual_p.detach().cpu().reshape(-1, 3).numpy()[:, 1]
            ideal = ideal_p.detach().cpu().reshape(-1, 3).numpy()[:, 1]
            denom = actual.copy()
            denom[theta == 0.0] = actual[0]
            magnif = ideal / denom

        reference_index = 0
        reference_fit_count = 0
        reference_intercept = np.nan
        reference_object_axis_tan = np.nan
        reference_actual_axis_mm = np.nan
        if use_far_slope_reference:
            # 标准符号约定：tan(θ_out) = M_ref × tan(θ_in) + b
            reference_mag, reference_fit_count, reference_intercept = trace_paraxial_slope_reference(
                theta_in,
                theta,
                valid_np,
                sample_count=5,
            )
            u_img = np.tan(np.deg2rad(theta_in))   # tan(出射角)
            u_obj = np.tan(np.deg2rad(theta))       # tan(入射角)
            # 放大率 M = (tan(θ_out) - b) / tan(θ_in)
            magnif = np.full(int(field_num), np.nan, dtype=float)
            np.divide(u_img - reference_intercept, u_obj, out=magnif, where=np.abs(u_obj) > EPS)
            if field_num > 0 and valid_np[0]:
                magnif[0] = reference_mag
            reference_index = -1
        elif use_near_slope_reference:
            reference_mag, reference_fit_count, reference_intercept = trace_height_slope_reference(
                ideal,
                actual,
                valid_np,
                sample_count=5,
            )
            reference_actual_axis_mm = -reference_intercept / reference_mag
            denom = actual - reference_actual_axis_mm
            magnif = np.full(int(field_num), np.nan, dtype=float)
            np.divide(ideal, denom, out=magnif, where=np.abs(denom) > EPS)
            if field_num > 0 and valid_np[0]:
                magnif[0] = reference_mag
            reference_index = -1
        else:
            if not (bool(valid_np[0]) and np.isfinite(magnif[0]) and abs(float(magnif[0])) > EPS):
                raise RuntimeError("Legacy distortion reference sample at index 0 is invalid")
            reference_mag = float(magnif[0])
        for idx, theta_i in enumerate(theta_display):
            valid_row = bool(valid_np[idx] and np.isfinite(magnif[idx]))
            distortion = float((magnif[idx] - reference_mag) / reference_mag) if valid_row else np.nan
            rows.append(
                {
                    "theta_deg": float(theta_i),
                    "theta_in_deg": float(theta_in[idx]) if np.isfinite(theta_in[idx]) else np.nan,
                    "magnification": float(magnif[idx]) if valid_row else np.nan,
                    "magnification_reference": reference_mag,
                    "distortion": distortion,
                    "distortion_percent": 100.0 * distortion if valid_row else np.nan,
                    "actual_height_mm": float(actual[idx]) if np.isfinite(actual[idx]) else np.nan,
                    "ideal_height_mm": float(ideal[idx]) if np.isfinite(ideal[idx]) else np.nan,
                    "reference_index": reference_index,
                    "valid": valid_row,
                }
            )
    except Exception as exc:
        raise RuntimeError(f"Distortion curve failed: {exc}") from exc

    meta = {
        "mode": "distortion-curve",
        "axis": axis,
        "fov_deg": float(fov_deg),
        "field_num": int(field_num),
        "distortion_type": distortion_type,
        "legacy_distortion_type": legacy_type,
        "wavelength_nm": float(wavelength_nm),
        "near_object_distance_mm": float(near_object_distance_mm),
        "pupil_distance_mm": float(pupil_distance_mm),
        "reference_index": -1 if use_trace_slope_reference else 0,
        "original_reference_function": "SingleEyeLens.eval_distortion",
        "sampling_policy": "GenerateCheifRays one-sided positive tan(theta) field",
        "initial_view_field_x_deg": 0.0,
        "initial_view_field_y_deg": 0.0,
        "trace_first_field_y_deg": float(theta[0]) if len(theta) else np.nan,
        "magnification_reference_policy": (
            "trace-based paraxial slope fit of tan(theta_img)=M*tan(theta_obj)+b"
            if use_far_slope_reference
            else "trace-based height slope fit of height_ideal=M*height_actual+b"
            if use_near_slope_reference
            else "first sample magnif(1)"
        ),
        "far_reference_mode": "trace_paraxial_slope" if use_far_slope_reference else "not_applicable",
        "far_reference_fit_sample_count": int(reference_fit_count) if use_far_slope_reference else 0,
        "far_reference_fit_intercept": float(reference_intercept) if use_far_slope_reference else np.nan,
        "far_reference_object_axis_tan": float(reference_object_axis_tan) if use_far_slope_reference else np.nan,
        "near_reference_mode": "trace_height_slope" if use_near_slope_reference else "not_applicable",
        "near_reference_fit_sample_count": int(reference_fit_count) if use_near_slope_reference else 0,
        "near_reference_fit_intercept_mm": float(reference_intercept) if use_near_slope_reference else np.nan,
        "near_reference_actual_axis_mm": float(reference_actual_axis_mm) if use_near_slope_reference else np.nan,
        "compatibility_deviation": (
            "Far-field distortion reference intentionally deviates from MATLAB magnif(1); "
            "uses trace-based paraxial slope and fitted object-axis offset for physical distortion."
            if use_far_slope_reference
            else "Near-field distortion reference intentionally deviates from MATLAB magnif(1); "
            "uses trace-based height slope and fitted actual-height axis for physical distortion."
            if use_near_slope_reference
            else "none"
        ),
        "magnification_policy": (
            "far: (tan(theta_in)-b)/tan(theta); near: height_ideal/height_actual"
            if use_far_slope_reference
            else "far: tan(theta_in)/tan(theta); near: height_ideal/(height_actual-fitted_actual_axis)"
            if use_near_slope_reference
            else "far: tan(theta_in)/tan(theta); near: height_ideal/height_actual"
        ),
        **adapter.metadata,
    }
    meta["valid_count"] = int(sum(bool(r["valid"]) for r in rows))
    meta["invalid_count"] = int(len(rows) - meta["valid_count"])
    return table_result(rows, meta)


def compute_distortion_grid(
    lens: Lensdata,
    *,
    fov_x_deg: float = 25.0,
    fov_y_deg: float = 25.0,
    field_num: int = 51,
    display_grid_num: int = 21,
    distortion_type: str = "rotating_eye_far",
    wavelength_nm: float = 555.0,
    near_object_distance_mm: float = 250.0,
    pupil_distance_mm: float = 250.0,
    lens_front_index: Optional[int] = None,
    lens_back_index: Optional[int] = None,
    fix_original_grid_axis_bug: bool = False,
) -> Dict[str, object]:
    """Compute two-dimensional distortion samples and display grids.

    Input field angles are degree. Object coordinates in the CSV are mm on the
    near object plane for both far and near modes, using the near distance as a
    reporting plane for far angular data. Grid arrays have shape
    `[display_grid_num, display_grid_num, 2]` or `[display_grid_num,
    display_grid_num]`. SciPy griddata is used only for display interpolation.
    """

    if field_num < 2 or display_grid_num < 2:
        raise ValueError("field_num and display_grid_num must be at least 2")
    legacy_type = normalize_distortion_type(distortion_type)
    if legacy_type not in {
        "rotate_eye_far",
        "fix_eye_far",
        "rotate_eye_near",
        "fix_eye_near",
        "dist_in_hand",
        "standard",
    }:
        raise ValueError(f"Unsupported distortion_type: {distortion_type}")
    adapter = build_legacy_adapter(
        lens,
        lens_front_index=lens_front_index,
        lens_back_index=lens_back_index,
        wavelength_nm=wavelength_nm,
    )
    is_far = legacy_type.endswith("_far") or legacy_type == "standard"
    ray_type = "dist_" + legacy_type if legacy_type != "standard" else "dist_standard"
    trace_grid_num = 2 * int(display_grid_num)
    rows: List[Dict[str, float | bool]] = []

    ray_single, single_tx, single_ty = boundary_grid_ray(
        adapter,
        fov_x_deg=fov_x_deg,
        fov_y_deg=fov_y_deg,
        ray_type=ray_type,
        wavelength_nm=wavelength_nm,
        pupil_distance_mm=pupil_distance_mm,
    )
    ray_single_traced, _ = trace_legacy_lens(adapter, ray_single)
    _, single_d = ray_values(ray_single_traced)
    ray_single_thetax_in = np.rad2deg(
        np.arctan(single_d[0, 0] / abs(single_d[0, 2]))
    ) if fix_original_grid_axis_bug else np.rad2deg(np.arctan(single_d[0, 1] / abs(single_d[0, 2])))
    ray_single_thetay_in = np.rad2deg(np.arctan(single_d[0, 1] / abs(single_d[0, 2])))

    rays0, thetax_grid, thetay_grid = generate_legacy_chief_rays_3d(
        adapter,
        field_num=trace_grid_num,
        fov_x_deg=fov_x_deg,
        fov_y_deg=fov_y_deg,
        ray_type=ray_type,
        wavelength_nm=wavelength_nm,
        pupil_distance_mm=pupil_distance_mm,
    )
    rays, valid = trace_legacy_lens(adapter, rays0)
    o, d = ray_values(rays)
    o0, d0 = ray_values(rays0)
    valid_np = valid.detach().cpu().reshape(-1).numpy().astype(bool)
    tx_flat = thetax_grid.reshape(-1)
    ty_flat = thetay_grid.reshape(-1)
    magnif_y = None  # set in is_far path; curve-consistent y-direction magnification

    if legacy_type == "standard":
        grid_reference_mode = "legacy_radial"
        affine_matrix = np.eye(2, dtype=float)
        affine_offset = np.zeros(2, dtype=float)
        fp1, _, fp, delta_h = center_focal_power(adapter)
        theta_in = np.rad2deg(np.arctan(np.sqrt((d[:, 0] / np.abs(d[:, 2])) ** 2 + (d[:, 1] / np.abs(d[:, 2])) ** 2)))
        objx_in = d[:, 0] / np.abs(d[:, 2])
        objy_in = d[:, 1] / np.abs(d[:, 2])
        theta = np.rad2deg(np.arctan(np.sqrt(np.tan(np.deg2rad(tx_flat)) ** 2 + np.tan(np.deg2rad(ty_flat)) ** 2)))
        height_ideal = -np.tan(np.deg2rad(theta_in)) / fp / (1.0 - delta_h / adapter.n1 * fp1)
        height_actual = -(1.0 - adapter.eye.eye_center_z_mm * fp) / fp * np.tan(np.deg2rad(theta))
        magnif = height_actual / height_ideal
        x_obj = np.linspace(-np.tan(np.deg2rad(ray_single_thetax_in)), np.tan(np.deg2rad(ray_single_thetax_in)), int(display_grid_num))
        y_obj = np.linspace(-np.tan(np.deg2rad(ray_single_thetay_in)), np.tan(np.deg2rad(ray_single_thetay_in)), int(display_grid_num))
    elif is_far:
        grid_reference_mode = "trace_affine_jacobian"
        thetax_in = np.rad2deg(np.arctan(d[:, 0] / np.abs(d[:, 2])))
        thetay_in = np.rad2deg(np.arctan(d[:, 1] / np.abs(d[:, 2])))
        objx_in = np.tan(np.deg2rad(thetax_in))
        objy_in = np.tan(np.deg2rad(thetay_in))
        ideal_np = np.stack((np.tan(np.deg2rad(tx_flat)), np.tan(np.deg2rad(ty_flat))), axis=1)
        source_np = np.stack((objx_in, objy_in), axis=1)
        affine_matrix, affine_offset, affine_fit_count = trace_affine_reference(source_np, ideal_np, valid_np)
        corrected_np = apply_inverse_affine(ideal_np, affine_matrix, affine_offset)
        magnif = radial_ratio(corrected_np, source_np)
        # 统一参考：从 2D 仿射矩阵的 y 分量提取参考放大率，使网格与曲线使用相同主轴和参考
        grid_ref_mag = float(affine_matrix[1, 1])
        grid_ref_offset_y = float(affine_offset[1])
        grid_reference_object_axis_tan = -grid_ref_offset_y / grid_ref_mag if abs(grid_ref_mag) > EPS else 0.0
        # 与曲线一致的一维 y 方向放大率（用于和畸变曲线直接对比）
        u_img_y = np.tan(np.deg2rad(ty_flat))
        u_obj_y = objy_in
        denom_y = np.abs(u_obj_y) - grid_reference_object_axis_tan
        magnif_y = np.full(tx_flat.size, np.nan, dtype=float)
        np.divide(np.abs(u_img_y), denom_y, out=magnif_y, where=np.abs(denom_y) > EPS)
        magnif_y[np.abs(denom_y) <= EPS] = grid_ref_mag
        x_obj = np.linspace(-np.tan(np.deg2rad(ray_single_thetax_in)), np.tan(np.deg2rad(ray_single_thetax_in)), int(display_grid_num))
        y_obj = np.linspace(-np.tan(np.deg2rad(ray_single_thetay_in)), np.tan(np.deg2rad(ray_single_thetay_in)), int(display_grid_num))
    else:
        grid_reference_mode = "trace_affine_jacobian"
        actual_p = intersect_z_plane(
            torch.as_tensor(o, dtype=torch.float64, device=lens.device).reshape(-1, 1, 3),
            torch.as_tensor(d, dtype=torch.float64, device=lens.device).reshape(-1, 1, 3),
            -near_object_distance_mm,
        )
        ideal_p = intersect_z_plane(
            torch.as_tensor(o0, dtype=torch.float64, device=lens.device).reshape(-1, 1, 3),
            torch.as_tensor(d0, dtype=torch.float64, device=lens.device).reshape(-1, 1, 3),
            -near_object_distance_mm,
        )
        actual_np = actual_p.detach().cpu().reshape(-1, 3).numpy()
        ideal_np = ideal_p.detach().cpu().reshape(-1, 3).numpy()
        objx_in = actual_np[:, 0]
        objy_in = actual_np[:, 1]
        source_np = np.stack((objx_in, objy_in), axis=1)
        affine_matrix, affine_offset, affine_fit_count = trace_affine_reference(source_np, ideal_np[:, :2], valid_np)
        corrected_np = apply_inverse_affine(ideal_np[:, :2], affine_matrix, affine_offset)
        magnif = radial_ratio(corrected_np, source_np)
        single_p = intersect_z_plane(ray_single_traced.o, ray_single_traced.d, -near_object_distance_mm)
        single_np = single_p.detach().cpu().reshape(-1, 3).numpy()[0]
        x_obj = np.linspace(-single_np[0], single_np[0], int(display_grid_num))
        y_obj = np.linspace(-single_np[1], single_np[1], int(display_grid_num))

    for idx in range(tx_flat.size):
        valid_row = bool(valid_np[idx] and np.isfinite(magnif[idx]))
        corrected_x = float(corrected_np[idx, 0]) if "corrected_np" in locals() and np.isfinite(corrected_np[idx, 0]) else np.nan
        corrected_y = float(corrected_np[idx, 1]) if "corrected_np" in locals() and np.isfinite(corrected_np[idx, 1]) else np.nan
        rows.append(
            {
                "theta_x_deg": float(tx_flat[idx]),
                "theta_y_deg": float(ty_flat[idx]),
                "object_x_actual_mm": float(objx_in[idx]) if np.isfinite(objx_in[idx]) else np.nan,
                "object_y_actual_mm": float(objy_in[idx]) if np.isfinite(objy_in[idx]) else np.nan,
                "object_x_ideal_mm": float(np.tan(np.deg2rad(tx_flat[idx])) if is_far else ideal_np[idx, 0]) if "ideal_np" in locals() else float(np.tan(np.deg2rad(tx_flat[idx]))),
                "object_y_ideal_mm": float(np.tan(np.deg2rad(ty_flat[idx])) if is_far else ideal_np[idx, 1]) if "ideal_np" in locals() else float(np.tan(np.deg2rad(ty_flat[idx]))),
                "object_x_affine_corrected": corrected_x,
                "object_y_affine_corrected": corrected_y,
                "distortion_x": corrected_x - float(objx_in[idx]) if np.isfinite(corrected_x) and np.isfinite(objx_in[idx]) else np.nan,
                "distortion_y": corrected_y - float(objy_in[idx]) if np.isfinite(corrected_y) and np.isfinite(objy_in[idx]) else np.nan,
                "magnification": float(magnif[idx]) if valid_row else np.nan,
                "magnification_unified": float(magnif_y[idx]) if valid_row and magnif_y is not None else float(magnif[idx]) if valid_row else np.nan,
                "valid": valid_row,
            }
        )

    if grid_reference_mode == "trace_affine_jacobian":
        regular, mag_grid, distorted, mag_display = make_distortion_display_grids_vector_from_arrays(
            objx_in,
            objy_in,
            corrected_np,
            x_obj,
            y_obj,
        )
    else:
        affine_fit_count = 0
        regular, mag_grid, distorted, mag_display = make_distortion_display_grids_from_arrays(
            objx_in,
            objy_in,
            magnif,
            x_obj,
            y_obj,
        )
    meta = {
        "mode": "distortion-grid",
        "fov_x_deg": float(fov_x_deg),
        "fov_y_deg": float(fov_y_deg),
        "field_num": int(field_num),
        "display_grid_num": int(display_grid_num),
        "trace_grid_num": int(trace_grid_num),
        "distortion_type": distortion_type,
        "legacy_distortion_type": legacy_type,
        "wavelength_nm": float(wavelength_nm),
        "near_object_distance_mm": float(near_object_distance_mm),
        "pupil_distance_mm": float(pupil_distance_mm),
        "original_reference_function": "SingleEyeLens.plot_distortion3D",
        "sampling_policy": "GenerateCheifRays3D symmetric tan(theta), trace grid = 2 * display_grid_num",
        "magnification_reference_policy": (
            "trace-based 2x2 affine Jacobian; y-reference A[1,1] unified with distortion curve"
            if grid_reference_mode == "trace_affine_jacobian"
            else "grid radial ratio, no scalar reference"
        ),
        "magnification_policy": (
            "magnification: radial norm of affine-corrected over actual; "
            "magnification_unified: 1D y-axis formula consistent with curve"
            if grid_reference_mode == "trace_affine_jacobian"
            else "far: h_img/h_obj; near: height_img/height_obj"
        ),
        "grid_reference_mode": grid_reference_mode,
        "grid_reference_matrix_2x2": affine_matrix.tolist(),
        "grid_reference_offset": affine_offset.tolist(),
        "grid_reference_fit_sample_count": int(affine_fit_count),
        "grid_reference_y_magnification": (
            float(affine_matrix[1, 1])
            if grid_reference_mode == "trace_affine_jacobian"
            else float(affine_matrix[0, 0])
        ),
        "grid_reference_y_object_axis_tan": (
            grid_reference_object_axis_tan
            if is_far and grid_reference_mode == "trace_affine_jacobian"
            else 0.0
        ),
        "compatibility_deviation": (
            "Grid magnification_unified column matches distortion curve definition "
            "(y-axis 1D tan-ratio formula with shared A[1,1] reference). "
            "magnification column uses radial 2D affine ratio."
            if grid_reference_mode == "trace_affine_jacobian"
            else "none"
        ),
        "griddata_policy": "linear griddata saved with original NaN; nearest only for display fallback",
        "original_grid_axis_bug_policy": "fixed" if fix_original_grid_axis_bug else "replicated_ry_for_x_and_y_boundary",
        **adapter.metadata,
    }
    meta["valid_count"] = int(sum(bool(r["valid"]) for r in rows))
    meta["invalid_count"] = int(len(rows) - meta["valid_count"])

    # 将 far/standard 模式的网格从 tan(angle) 转换为角度 (°)
    if is_far:
        # 对 trace_affine_jacobian 模式：用近轴局部仿射逆映射隔离纯畸变
        # 红色网格显示：对于追迹得到的出射角 θ_out，线性（无畸变）系统需要
        # 什么入射角 θ_in_ref 才能产生这个出射角。蓝色网格为均匀入射角参考。
        # 两者差异即为纯非线性畸变分量。
        if grid_reference_mode == "trace_affine_jacobian":
            # 用近轴光线（视场角 < 5°）独立拟合出射角 vs 入射角的线性关系
            r_sq = tx_flat**2 + ty_flat**2
            near = (r_sq < 25.0) & valid_np  # 5° 以内近轴区
            if near.sum() >= 4:
                u_in_x = np.tan(np.deg2rad(tx_flat[near]))
                u_in_y = np.tan(np.deg2rad(ty_flat[near]))
                u_out_x = np.tan(np.deg2rad(thetax_in.ravel()[near]))
                u_out_y = np.tan(np.deg2rad(thetay_in.ravel()[near]))
                # 独立最小二乘拟合 tan(θ_out) = M * tan(θ_in) + b
                A_x = np.column_stack([u_in_x, np.ones_like(u_in_x)])
                Mx, bx = np.linalg.lstsq(A_x, u_out_x, rcond=None)[0]
                A_y = np.column_stack([u_in_y, np.ones_like(u_in_y)])
                My, by = np.linalg.lstsq(A_y, u_out_y, rcond=None)[0]
            else:
                # 回退到全局仿射
                Mx = My = 1.0
                bx = by = 0.0
            theta_inc_ref_x = np.rad2deg(np.arctan(
                (np.tan(np.deg2rad(thetax_in.ravel())) - bx) / Mx
            ))
            theta_inc_ref_y = np.rad2deg(np.arctan(
                (np.tan(np.deg2rad(thetay_in.ravel())) - by) / My
            ))
            theta_inc_ref = np.column_stack([theta_inc_ref_x, theta_inc_ref_y])
            theta_inc_src = np.stack((tx_flat, ty_flat), axis=1)  # 采样入射角 (deg)
            valid_interp = valid_np & np.isfinite(theta_inc_ref).all(axis=1)
            n_display = int(display_grid_num)
            if valid_interp.sum() >= 3:
                gx_1d = np.linspace(-fov_x_deg, fov_x_deg, n_display)
                gy_1d = np.linspace(-fov_y_deg, fov_y_deg, n_display)
                gx_grid, gy_grid = np.meshgrid(gx_1d, gy_1d, indexing="xy")
                src = theta_inc_src[valid_interp]
                val_x = theta_inc_ref[valid_interp, 0]
                val_y = theta_inc_ref[valid_interp, 1]
                # 插入中心强制点 (0,0)，确保显示网格中心插值为 0
                src = np.vstack([src, [0.0, 0.0]])
                val_x = np.concatenate([val_x, [0.0]])
                val_y = np.concatenate([val_y, [0.0]])
                cx = griddata(src, val_x, (gx_grid, gy_grid), method="linear")
                cy = griddata(src, val_y, (gx_grid, gy_grid), method="linear")
                nx = griddata(src, val_x, (gx_grid, gy_grid), method="nearest")
                ny = griddata(src, val_y, (gx_grid, gy_grid), method="nearest")
                display_x = np.where(np.isfinite(cx), cx, nx)
                display_y = np.where(np.isfinite(cy), cy, ny)
                regular = np.stack((gx_grid, gy_grid), axis=-1)  # 均匀入射角
                distorted = np.stack((display_x, display_y), axis=-1)  # 仿射逆映射后等效入射角
                # 重新计算径向放大率
                r_reg = np.sqrt(regular[..., 0]**2 + regular[..., 1]**2)
                r_dist = np.sqrt(distorted[..., 0]**2 + distorted[..., 1]**2)
                mag_display = np.divide(r_dist, r_reg, out=np.full_like(r_reg, np.nan), where=r_reg > 1e-8)
                mag_display[r_reg <= 1e-8] = 1.0
                mag_grid = mag_display.copy()
                meta["grid_coordinate_reference"] = "incident_angle"
            else:
                regular = np.rad2deg(np.arctan(regular))
                distorted = np.rad2deg(np.arctan(distorted))
                meta["grid_coordinate_reference"] = "exit_angle"
        else:
            regular = np.rad2deg(np.arctan(regular))
            distorted = np.rad2deg(np.arctan(distorted))
            meta["grid_coordinate_reference"] = "exit_angle"
    else:
        meta["grid_coordinate_reference"] = "exit_angle"
    meta["grid_coordinate_unit"] = "deg" if is_far else "mm"

    result = table_result(rows, meta)
    result["grids"] = {
        "regular": regular,
        "magnification": mag_grid,
        "magnification_display": mag_display,
        "distorted": distorted,
    }
    return result


def save_power_outputs(result: Mapping[str, object], output_dir: str | Path) -> Dict[str, Path]:
    """Save power/astigmatism CSV, NPY dict, and PNG outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": output / "power_astigmatism_curve.csv",
        "npy": output / "power_astigmatism_curve.npy",
        "png": output / "power_astigmatism_curve.png",
    }
    write_csv(paths["csv"], result["rows"])
    np.save(paths["npy"], {"columns": result["columns"], "data": result["data"], "metadata": result["metadata"]})
    plot_power(result, paths["png"])
    if "trace_result" in result:
        trace = result["trace_result"]
        paths.update(
            {
                "trace_csv": output / "trace_power_astigmatism_curve.csv",
                "trace_npy": output / "trace_power_astigmatism_curve.npy",
                "trace_png": output / "trace_power_astigmatism_curve.png",
            }
        )
        write_csv(paths["trace_csv"], trace["rows"])
        np.save(paths["trace_npy"], {"columns": trace["columns"], "data": trace["data"], "metadata": trace["metadata"]})
        plot_power(trace, paths["trace_png"])
    return paths


def save_distortion_curve_outputs(result: Mapping[str, object], output_dir: str | Path) -> Dict[str, Path]:
    """Save distortion curve CSV, NPY dict, and PNG outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": output / "distortion_curve.csv",
        "npy": output / "distortion_curve.npy",
        "png": output / "distortion_curve.png",
    }
    write_csv(paths["csv"], result["rows"])
    np.save(paths["npy"], {"columns": result["columns"], "data": result["data"], "metadata": result["metadata"]})
    plot_distortion_curve(result, paths["png"])
    return paths


def save_distortion_grid_outputs(result: Mapping[str, object], output_dir: str | Path) -> Dict[str, Path]:
    """Save distortion grid CSV, NPY arrays, and PNG outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grids = result["grids"]
    paths = {
        "regular": output / "distortion_grid_regular.npy",
        "magnification": output / "distortion_grid_magnification.npy",
        "distorted": output / "distortion_grid_distorted.npy",
        "csv": output / "distortion_grid_samples.csv",
        "png": output / "distortion_grid.png",
    }
    np.save(paths["regular"], grids["regular"])
    np.save(paths["magnification"], grids["magnification"])
    np.save(paths["distorted"], grids["distorted"])
    write_csv(paths["csv"], result["rows"])
    plot_distortion_grid(result, paths["png"])
    return paths


def save_footprint_coverage_outputs(
    result: Mapping[str, object],
    output_dir: str | Path,
    *,
    trim_pixels: int = 3,
) -> Dict[str, Path]:
    """Save footprint coverage NPY, metadata JSON, and PNG outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "npy": output / "footprint_coverage.npy",
        "metadata": output / "footprint_coverage_metadata.json",
        "png": output / "footprint_coverage.png",
    }
    metadata = dict(result["metadata"])
    metadata["display_trim_pixels"] = int(trim_pixels)
    np.save(
        paths["npy"],
        {
            "field_phi_deg": result["field_phi_deg"],
            "field_x_deg": result["field_x_deg"],
            "field_y_deg": result["field_y_deg"],
            "field_slope_x": result["field_slope_x"],
            "field_slope_y": result["field_slope_y"],
            "gaze_direction": result["gaze_direction"],
            "pupil_phi_deg": result["pupil_phi_deg"],
            "pupil_x_mm": result["pupil_x_mm"],
            "pupil_y_mm": result["pupil_y_mm"],
            "footprint_x_mm": result["footprint_x_mm"],
            "footprint_y_mm": result["footprint_y_mm"],
            "footprint_z_mm": result["footprint_z_mm"],
            "local_power_D": result["local_power_D"],
            "trace_valid": result["trace_valid"],
            "valid": result["valid"],
            "convex_hull_xy_mm": result["convex_hull_xy_mm"],
            "metadata": metadata,
        },
    )
    with paths["metadata"].open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    plot_footprint_coverage(result, paths["png"], trim_pixels=trim_pixels)
    return paths


def lens_fov_float(lens: Lensdata) -> float:
    """Return Lensdata FOV as a Python float in degree."""

    return float(_scalar(lens.FOV))


def make_eye_ray(
    lens: Lensdata,
    *,
    origin_z_mm: float,
    field_x_deg: float,
    field_y_deg: float,
    wavelength_nm: float,
    direction_sign: float,
) -> Ray:
    """Create a ray in global coordinates from an eye/pupil reference point.

    The ray origin is `[0, 0, origin_z_mm]` in mm. Direction uses tangent field
    angles and `direction_sign=-1` for eye-to-object tracing or `+1` for
    object-to-eye tracing. Returned shape is `[1, 1, 3]`.
    """

    dtype = torch.float64
    device = lens.device
    tx = torch.tan(torch.deg2rad(torch.tensor(float(field_x_deg), dtype=dtype, device=device)))
    ty = torch.tan(torch.deg2rad(torch.tensor(float(field_y_deg), dtype=dtype, device=device)))
    z = torch.tensor(float(direction_sign), dtype=dtype, device=device)
    d = _normalize(torch.stack((tx, ty, z)).reshape(1, 1, 3))
    o = torch.tensor([[[0.0, 0.0, float(origin_z_mm)]]], dtype=dtype, device=device)
    return Ray(o, d, _wavelength_tensor(lens, wavelength_nm), device=device)


def make_ray_from_arrays(
    lens: Lensdata,
    origins: np.ndarray,
    directions: np.ndarray,
    wavelength_nm: float,
) -> Ray:
    """Create a batch ray from `[N, 3]` mm origins and unitless directions."""

    device = lens.device
    o = torch.as_tensor(origins, dtype=torch.float64, device=device).reshape(-1, 1, 3)
    d = _normalize(torch.as_tensor(directions, dtype=torch.float64, device=device).reshape(-1, 1, 3))
    return Ray(o, d, _wavelength_tensor(lens, wavelength_nm), device=device)


def generate_legacy_chief_rays_1d(
    adapter: LegacyLensAdapter,
    *,
    field_num: int,
    fov_deg: float,
    ray_type: str,
    wavelength_nm: float,
    pupil_distance_mm: float,
) -> Tuple[Ray, np.ndarray]:
    """Generate one-dimensional chief rays matching `GenerateCheifRays`.

    Output ray origins are in global mm coordinates and directions are unit
    vectors. Only the Y field is sampled. `ray_type` uses MATLAB-style names:
    `uniform`, `dist_rotate_eye`, `dist_fix_eye`, `dist_in_hand`,
    `dist_standard`.
    """

    if "dist_in_hand" in ray_type:
        z0 = float(pupil_distance_mm) + adapter.h_glass_mm
        max_tan = 0.9 * adapter.lens.surfaces[adapter.lens_back_index].semi_dia / 2.0 / z0
        theta = np.rad2deg(np.arctan(np.linspace(LEGACY_OPTICAL_DIFF, max_tan, int(field_num))))
    elif "dist_fix_eye" in ray_type:
        z0 = adapter.eye.pupil_z_mm
        theta = sample_legacy_positive_fields(fov_deg, field_num, tan_uniform=True)
    elif "dist_rotate_eye" in ray_type or "dist_standard" in ray_type:
        z0 = adapter.eye.eye_center_z_mm
        theta = sample_legacy_positive_fields(fov_deg, field_num, tan_uniform=True)
    else:
        z0 = adapter.eye.eye_center_z_mm
        theta = sample_legacy_positive_fields(fov_deg, field_num, tan_uniform=False)

    origins = np.zeros((int(field_num), 3), dtype=float)
    origins[:, 2] = z0
    directions = np.zeros_like(origins)
    directions[:, 1] = np.sin(np.deg2rad(theta))
    directions[:, 2] = -np.cos(np.deg2rad(theta))
    return make_ray_from_arrays(adapter.lens, origins, directions, wavelength_nm), theta


def generate_legacy_chief_rays_3d(
    adapter: LegacyLensAdapter,
    *,
    field_num: int,
    fov_x_deg: float,
    fov_y_deg: float,
    ray_type: str,
    wavelength_nm: float,
    pupil_distance_mm: float,
) -> Tuple[Ray, np.ndarray, np.ndarray]:
    """Generate two-dimensional chief rays matching `GenerateCheifRays3D`."""

    if "dist_in_hand" in ray_type:
        z0 = float(pupil_distance_mm) + adapter.h_glass_mm
        factor = 0.60
        max_tan = factor * adapter.lens.surfaces[adapter.lens_back_index].semi_dia / 2.0 / z0
        if int(field_num) == 1:
            tx = np.array([max_tan], dtype=float)
            ty = np.array([max_tan], dtype=float)
        else:
            tx = np.linspace(-max_tan, max_tan, int(field_num))
            ty = np.linspace(-max_tan, max_tan, int(field_num))
        thetax, thetay = np.meshgrid(np.rad2deg(np.arctan(tx)), np.rad2deg(np.arctan(ty)), indexing="xy")
    elif "dist_fix_eye" in ray_type:
        z0 = adapter.eye.pupil_z_mm
        thetax, thetay = sample_legacy_grid_fields(fov_x_deg, fov_y_deg, field_num)
    elif "dist_rotate_eye" in ray_type or "dist_standard" in ray_type:
        z0 = adapter.eye.eye_center_z_mm
        thetax, thetay = sample_legacy_grid_fields(fov_x_deg, fov_y_deg, field_num)
    else:
        z0 = adapter.eye.eye_center_z_mm
        if int(field_num) == 1:
            x = np.array([float(fov_x_deg)], dtype=float)
            y = np.array([float(fov_y_deg)], dtype=float)
        else:
            x = np.linspace(-float(fov_x_deg), float(fov_x_deg), int(field_num))
            y = np.linspace(-float(fov_y_deg), float(fov_y_deg), int(field_num))
        thetax, thetay = np.meshgrid(x, y, indexing="xy")

    origins = np.zeros((int(field_num) * int(field_num), 3), dtype=float)
    origins[:, 2] = z0
    rx = np.tan(np.deg2rad(thetax)).reshape(-1)
    ry = np.tan(np.deg2rad(thetay)).reshape(-1)
    directions = np.stack((rx, ry, -np.ones_like(rx)), axis=1)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return make_ray_from_arrays(adapter.lens, origins, directions, wavelength_nm), thetax, thetay


def trace_legacy_lens(adapter: LegacyLensAdapter, ray: Ray) -> Tuple[Ray, torch.Tensor]:
    """Trace only through the eyeglass front/back surfaces."""

    return adapter.lens.trace(ray, stop_ind=adapter.lens_back_index, is_fixed=True)


def generate_forward_rays_legacy(
    adapter: LegacyLensAdapter,
    *,
    field_num: int,
    fov_deg: float,
    wavelength_nm: float,
) -> Tuple[Ray, Ray, np.ndarray, np.ndarray]:
    """Generate forward chief and paraxial differential rays.

    This ports `SingleEyeLens.generate_forward_rays()` for the paraxial
    `eval_focal_power()` path. It first traces eye-side chief rays backward
    through the eyeglass, extends them to `z=-h0`, flips direction, then creates
    a four-arm local differential bundle around each chief ray.
    """

    chief_backward, theta = generate_legacy_chief_rays_1d(
        adapter,
        field_num=field_num,
        fov_deg=fov_deg,
        ray_type="uniform",
        wavelength_nm=wavelength_nm,
        pupil_distance_mm=250.0,
    )
    traced, valid = trace_legacy_lens(adapter, chief_backward)
    if not bool(valid.detach().cpu().all().item()):
        raise RuntimeError("legacy chief backward trace has invalid rays")
    front_plane = intersect_z_plane(traced.o, traced.d, -adapter.h0_mm)
    chief_forward = Ray(front_plane.clone(), _normalize(-traced.d.clone()), traced.wavelength, device=adapter.lens.device)

    ape = LEGACY_OPTICAL_DIFF / 2.0
    phi = np.linspace(0.0, 2.0 * np.pi * (1.0 - 1.0 / 4.0), 4)
    local_x = ape * np.cos(phi)
    local_y = ape * np.sin(phi)
    origins: List[torch.Tensor] = []
    directions: List[torch.Tensor] = []
    for i in range(int(field_num)):
        base_o = chief_forward.o[i : i + 1]
        base_d = chief_forward.d[i : i + 1]
        ry = base_d[..., 1]
        rz = base_d[..., 2]
        theta_y = -torch.atan(ry / rz)
        lx = torch.as_tensor(local_x, dtype=base_o.dtype, device=base_o.device).reshape(-1, 1)
        ly = torch.as_tensor(local_y, dtype=base_o.dtype, device=base_o.device).reshape(-1, 1)
        lz = torch.zeros_like(ly)
        y0 = torch.cos(theta_y) * ly - torch.sin(theta_y) * lz
        z0 = torch.sin(theta_y) * ly + torch.cos(theta_y) * lz
        offset = torch.stack((lx.reshape(-1), y0.reshape(-1), z0.reshape(-1)), dim=-1).reshape(-1, 1, 3)
        origins.append(base_o + offset)
        directions.append(base_d.repeat(4, 1, 1))

    bundle = Ray(
        torch.cat(origins, dim=0),
        _normalize(torch.cat(directions, dim=0)),
        _wavelength_tensor(adapter.lens, wavelength_nm),
        device=adapter.lens.device,
    )
    fields = np.repeat(theta, 4)
    return chief_forward, bundle, theta, fields


def rotate_about_x_local(ray: Ray, theta_y_deg: np.ndarray, h_eye_center_mm: float) -> Ray:
    """Rotate ray positions/directions about X through the eye centre."""

    theta = torch.as_tensor(theta_y_deg, dtype=ray.o.dtype, device=ray.o.device).reshape(-1, 1)
    c = torch.cos(torch.deg2rad(theta))
    s = torch.sin(torch.deg2rad(theta))
    o = ray.o.clone()
    d = ray.d.clone()
    y = o[..., 1]
    z = o[..., 2] - float(h_eye_center_mm)
    o[..., 1] = c * y - s * z
    o[..., 2] = s * y + c * z + float(h_eye_center_mm)
    ry = d[..., 1]
    rz = d[..., 2]
    d[..., 1] = c * ry - s * rz
    d[..., 2] = s * ry + c * rz
    return Ray(o, _normalize(d), ray.wavelength, ray.weight, ray.phase, device=o.device)


def ray_values(ray: Ray) -> Tuple[np.ndarray, np.ndarray]:
    """Return detached origin and direction arrays shaped `[N, 3]`."""

    return (
        ray.o.detach().cpu().reshape(-1, 3).numpy(),
        ray.d.detach().cpu().reshape(-1, 3).numpy(),
    )


def trace_paraxial_slope_reference(
    theta_img_deg: np.ndarray,
    theta_obj_deg: np.ndarray,
    valid_mask: np.ndarray,
    *,
    sample_count: int = 5,
) -> Tuple[float, int, float]:
    """Estimate far-field paraxial angular magnification from traced rays.

    Input angles are in degree. The fit uses the first valid small-angle
    samples and solves `tan(theta_img) = M * tan(theta_obj) + b`; the returned
    reference magnification is the slope `M`. This keeps the reference tied to
    the actual ray trace and avoids requiring a Gaussian-optics power `P` or
    eye-centre distance `d_ec`. GPU/autograd are not involved because this is
    a post-processing metric reference.
    """

    theta_img = np.asarray(theta_img_deg, dtype=float).reshape(-1)
    theta_obj = np.asarray(theta_obj_deg, dtype=float).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    usable = valid & np.isfinite(theta_img) & np.isfinite(theta_obj)
    indices = np.flatnonzero(usable)
    if indices.size < 2:
        raise RuntimeError("Need at least two valid small-angle samples for trace-based paraxial reference")
    indices = indices[: max(2, int(sample_count))]
    u_img = np.tan(np.deg2rad(theta_img[indices]))
    u_obj = np.tan(np.deg2rad(theta_obj[indices]))
    if np.nanmax(u_obj) - np.nanmin(u_obj) <= EPS:
        raise RuntimeError("Object-angle samples are too close for trace-based paraxial reference")
    x_mean = float(np.mean(u_obj))
    y_mean = float(np.mean(u_img))
    x_centered = u_obj - x_mean
    denom = float(np.sum(x_centered**2))
    if denom <= EPS:
        raise RuntimeError("Object-angle samples are too close for trace-based paraxial reference")
    slope = float(np.sum(x_centered * (u_img - y_mean)) / denom)
    intercept = y_mean - slope * x_mean
    if not (np.isfinite(slope) and abs(float(slope)) > EPS):
        raise RuntimeError("Trace-based paraxial reference produced an invalid slope")
    return float(slope), int(indices.size), float(intercept)


def trace_height_slope_reference(
    height_ideal_mm: np.ndarray,
    height_actual_mm: np.ndarray,
    valid_mask: np.ndarray,
    *,
    sample_count: int = 5,
) -> Tuple[float, int, float]:
    """Estimate near-field paraxial magnification from traced object heights.

    Input heights are in mm on the near object plane. The fit uses the first
    valid small-field samples and solves
    `height_ideal = M * height_actual + b`; the returned reference
    magnification is the slope `M`. This is post-processing only: GPU/autograd
    are not preserved.
    """

    ideal = np.asarray(height_ideal_mm, dtype=float).reshape(-1)
    actual = np.asarray(height_actual_mm, dtype=float).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    usable = valid & np.isfinite(ideal) & np.isfinite(actual)
    indices = np.flatnonzero(usable)
    if indices.size < 2:
        raise RuntimeError("Need at least two valid small-field samples for trace-based near reference")
    indices = indices[: max(2, int(sample_count))]
    x = actual[indices]
    y = ideal[indices]
    if np.nanmax(x) - np.nanmin(x) <= EPS:
        raise RuntimeError("Actual-height samples are too close for trace-based near reference")
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    denom = float(np.sum(x_centered**2))
    if denom <= EPS:
        raise RuntimeError("Actual-height samples are too close for trace-based near reference")
    slope = float(np.sum(x_centered * (y - y_mean)) / denom)
    intercept = y_mean - slope * x_mean
    if not (np.isfinite(slope) and abs(float(slope)) > EPS):
        raise RuntimeError("Trace-based near reference produced an invalid slope")
    return float(slope), int(indices.size), float(intercept)


def trace_affine_reference(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    valid_mask: np.ndarray,
    *,
    sample_count: int = 25,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Fit a local 2D affine reference `target = A @ source + b`.

    Coordinates may be angular tangents for far-field grids or mm object-plane
    heights for near-field grids. The fit uses the valid samples nearest the
    source origin. Returns `A` as a `[2, 2]` matrix and `b` as `[2]`.
    """

    source = np.asarray(source_xy, dtype=float).reshape(-1, 2)
    target = np.asarray(target_xy, dtype=float).reshape(-1, 2)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    usable = valid & np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    indices = np.flatnonzero(usable)
    if indices.size < 3:
        raise RuntimeError("Need at least three valid points for trace-based affine grid reference")
    order = np.argsort(np.sum(source[indices] ** 2, axis=1))
    indices = indices[order[: max(3, int(sample_count))]]
    x = source[indices, 0]
    y = source[indices, 1]
    ones = np.ones_like(x)
    normal = np.array(
        [
            [np.sum(x * x), np.sum(x * y), np.sum(x)],
            [np.sum(x * y), np.sum(y * y), np.sum(y)],
            [np.sum(x), np.sum(y), np.sum(ones)],
        ],
        dtype=float,
    )
    target_fit = target[indices]
    rhs = np.array(
        [
            [np.sum(x * target_fit[:, 0]), np.sum(x * target_fit[:, 1])],
            [np.sum(y * target_fit[:, 0]), np.sum(y * target_fit[:, 1])],
            [np.sum(target_fit[:, 0]), np.sum(target_fit[:, 1])],
        ],
        dtype=float,
    )
    coeff_x = solve_linear_system(normal, rhs[:, 0])
    coeff_y = solve_linear_system(normal, rhs[:, 1])
    matrix = np.array([[coeff_x[0], coeff_x[1]], [coeff_y[0], coeff_y[1]]], dtype=float)
    offset = np.array([coeff_x[2], coeff_y[2]], dtype=float)
    det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if not (np.isfinite(matrix).all() and np.isfinite(offset).all() and abs(float(det)) > EPS):
        raise RuntimeError("Trace-based affine grid reference is singular or invalid")
    return matrix, offset, int(indices.size)


def solve_linear_system(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Solve a small dense linear system without relying on NumPy LAPACK."""

    a = np.asarray(matrix, dtype=float).copy()
    b = np.asarray(vector, dtype=float).copy()
    n = int(b.size)
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(a[col:, col])))
        if abs(float(a[pivot, col])) <= EPS:
            raise RuntimeError("Singular least-squares normal matrix")
        if pivot != col:
            a[[col, pivot], :] = a[[pivot, col], :]
            b[[col, pivot]] = b[[pivot, col]]
        factor = a[col, col]
        a[col, :] /= factor
        b[col] /= factor
        for row in range(n):
            if row == col:
                continue
            row_factor = a[row, col]
            a[row, :] -= row_factor * a[col, :]
            b[row] -= row_factor * b[col]
    return b


def apply_inverse_affine(points_xy: np.ndarray, matrix: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Apply `inv(matrix) @ (points - offset)` for `[N, 2]` coordinates."""

    points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    a = np.asarray(matrix, dtype=float).reshape(2, 2)
    b = np.asarray(offset, dtype=float).reshape(2)
    det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    if abs(float(det)) <= EPS:
        raise RuntimeError("Affine reference matrix is singular")
    shifted = points - b.reshape(1, 2)
    out = np.empty_like(shifted)
    out[:, 0] = (a[1, 1] * shifted[:, 0] - a[0, 1] * shifted[:, 1]) / det
    out[:, 1] = (-a[1, 0] * shifted[:, 0] + a[0, 0] * shifted[:, 1]) / det
    return out


def radial_ratio(target_xy: np.ndarray, source_xy: np.ndarray) -> np.ndarray:
    """Return `||target|| / ||source||`, with the origin normalized to one."""

    target = np.asarray(target_xy, dtype=float).reshape(-1, 2)
    source = np.asarray(source_xy, dtype=float).reshape(-1, 2)
    numerator = np.sqrt(np.sum(target**2, axis=1))
    denominator = np.sqrt(np.sum(source**2, axis=1))
    ratio = np.full(target.shape[0], np.nan, dtype=float)
    np.divide(numerator, denominator, out=ratio, where=denominator > EPS)
    origin = denominator <= EPS
    ratio[origin & np.isfinite(numerator)] = 1.0
    return ratio


def rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a 3 x 3 rotation matrix that maps unit vector `source` to `target`.

    Inputs are unitless 3D vectors. The returned matrix is CPU NumPy data. This
    helper is used for rigid eye/pupil geometry and has no GPU/autograd role.
    """

    a = np.asarray(source, dtype=float).reshape(3)
    b = np.asarray(target, dtype=float).reshape(3)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm <= EPS or b_norm <= EPS:
        raise ValueError("rotation_from_vectors requires non-zero vectors")
    a = a / a_norm
    b = b / b_norm
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3, dtype=float)
    if c < -1.0 + 1e-12:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=float))
        if np.linalg.norm(axis) <= EPS:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=float))
        axis = axis / np.linalg.norm(axis)
        return -np.eye(3, dtype=float) + 2.0 * np.outer(axis, axis)
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3, dtype=float) + vx + vx @ vx * (1.0 / (1.0 + c))


def compute_affine_reference_2d(
    adapter: LegacyLensAdapter,
    ray_type: str,
    wavelength_nm: float,
    pupil_distance_mm: float,
    half_range_deg: float = 2.0,
    grid_n: int = 3,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Generate a small 2D chief ray grid near center and fit the 2D affine reference.

    The returned affine maps *source* (actual traced tan-space positions) to
    *ideal* (input tan-space positions).  ``A[1, 1]`` is the y-direction reference
    magnification used by both distortion-curve and distortion-grid for a unified
    reference definition.  Returns ``(A, b, fit_count, valid_np)``.  Only the
    central ``grid_n × grid_n`` rays are traced (default 9 rays at ±2°).
    """

    rays, thetax, thetay = generate_legacy_chief_rays_3d(
        adapter,
        field_num=grid_n,
        fov_x_deg=half_range_deg,
        fov_y_deg=half_range_deg,
        ray_type=ray_type,
        wavelength_nm=wavelength_nm,
        pupil_distance_mm=pupil_distance_mm,
    )
    rays_traced, valid = trace_legacy_lens(adapter, rays)
    _, d = ray_values(rays_traced)
    valid_np = valid.detach().cpu().reshape(-1).numpy().astype(bool)
    tx_flat = thetax.reshape(-1)
    ty_flat = thetay.reshape(-1)
    thetax_in = np.rad2deg(np.arctan(d[:, 0] / np.abs(d[:, 2])))
    thetay_in = np.rad2deg(np.arctan(d[:, 1] / np.abs(d[:, 2])))
    source = np.stack((np.tan(np.deg2rad(thetax_in)), np.tan(np.deg2rad(thetay_in))), axis=1)
    ideal = np.stack((np.tan(np.deg2rad(tx_flat)), np.tan(np.deg2rad(ty_flat))), axis=1)
    A, b, n = trace_affine_reference(source, ideal, valid_np)
    return A, b, n, valid_np


def normalize_distortion_type(distortion_type: str) -> str:
    """Map Python CLI distortion names to MATLAB `SingleEyeLens` names."""

    mapping = {
        "rotating_eye_far": "rotate_eye_far",
        "rotating_eye_near": "rotate_eye_near",
        "fixed_eye_far": "fix_eye_far",
        "fixed_eye_near": "fix_eye_near",
        "handheld_near": "dist_in_hand",
    }
    return mapping.get(str(distortion_type), str(distortion_type))


def center_focal_power(adapter: LegacyLensAdapter) -> Tuple[float, float, float, float]:
    """Return original `get_center_focal_power()` values.

    The returned powers use the original centre-curvature convention and are in
    inverse mm, as in the MATLAB implementation.
    """

    fp1 = (adapter.n1 - adapter.n0) / adapter.n0 * adapter.c0
    fp2 = (adapter.n2 - adapter.n1) / adapter.n2 * adapter.c1
    delta_h = adapter.h_glass_mm
    fp = fp1 + fp2 - delta_h * fp1 * fp2 / adapter.n1
    if abs(fp) <= EPS:
        raise RuntimeError("center focal power is too small for standard distortion")
    return fp1, fp2, fp, delta_h


def boundary_grid_ray(
    adapter: LegacyLensAdapter,
    *,
    fov_x_deg: float,
    fov_y_deg: float,
    ray_type: str,
    wavelength_nm: float,
    pupil_distance_mm: float,
) -> Tuple[Ray, np.ndarray, np.ndarray]:
    """Generate the single boundary ray used by original grid plotting."""

    fov_max = max(float(fov_x_deg), float(fov_y_deg))
    tan_vec = np.array([np.tan(np.deg2rad(float(fov_x_deg))), np.tan(np.deg2rad(float(fov_y_deg)))], dtype=float)
    denom = np.sqrt(np.sum(tan_vec**2))
    if denom <= EPS:
        effective = np.array([0.0, 0.0], dtype=float)
    else:
        effective = np.rad2deg(np.arctan(tan_vec / denom * np.tan(np.deg2rad(fov_max))))
    return generate_legacy_chief_rays_3d(
        adapter,
        field_num=1,
        fov_x_deg=float(effective[0]),
        fov_y_deg=float(effective[1]),
        ray_type=ray_type,
        wavelength_nm=wavelength_nm,
        pupil_distance_mm=pupil_distance_mm,
    )


@contextmanager
def temporary_cb_tilt(lens: Lensdata, field_x_deg: float, field_y_deg: float):
    """Temporarily set CB tilt to match an Excel H/I field angle in memory."""

    eye = extract_eye_positions(lens)
    cb = lens.surfaces[eye.cb_index]
    old = (cb.tilt_x.clone(), cb.tilt_y.clone(), cb.tilt_z.clone())
    try:
        cb.tilt_x = torch.as_tensor(-float(field_x_deg), dtype=old[0].dtype, device=old[0].device)
        cb.tilt_y = torch.as_tensor(-float(field_y_deg), dtype=old[1].dtype, device=old[1].device)
        cb.tilt_z = torch.as_tensor(float(old[2]), dtype=old[2].dtype, device=old[2].device)
        yield
    finally:
        cb.tilt_x, cb.tilt_y, cb.tilt_z = old


def local_sagittal_meridional_basis(chief_direction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return local sagittal and meridional unit vectors perpendicular to chief ray."""

    d = _normalize(chief_direction)
    device = d.device
    dtype = d.dtype
    z_axis = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=dtype, device=device)
    sagittal = torch.cross(d, z_axis, dim=-1)
    if torch.linalg.norm(sagittal.reshape(-1)) < 1e-9:
        sagittal = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=dtype, device=device)
    sagittal = _normalize(sagittal)
    meridional = _normalize(torch.cross(sagittal, d, dim=-1))
    return sagittal, meridional


def differential_power_D(
    lens: Lensdata,
    chief_forward: Ray,
    chief_out: Ray,
    offset_basis: torch.Tensor,
    offset_mm: float,
) -> torch.Tensor:
    """Estimate local power from +/- differential rays around a chief ray."""

    distances = []
    for sign in (-1.0, 1.0):
        ray = Ray(
            chief_forward.o + float(sign) * float(offset_mm) * offset_basis,
            chief_forward.d.clone(),
            chief_forward.wavelength,
            device=lens.device,
        )
        ray_out, valid = lens.trace(ray, is_fixed=True)
        if not bool(valid.detach().cpu().reshape(-1)[0].item()):
            raise RuntimeError("differential ray trace is invalid")
        t = closest_parameter_on_first_ray(chief_out.o, chief_out.d, ray_out.o, ray_out.d)
        distances.append(t)
    focus_mm = torch.stack(distances).mean()
    if torch.abs(focus_mm) <= EPS:
        raise RuntimeError("differential focus distance is too small")
    return 1000.0 / focus_mm


def closest_parameter_on_first_ray(
    p0: torch.Tensor,
    d0: torch.Tensor,
    p1: torch.Tensor,
    d1: torch.Tensor,
) -> torch.Tensor:
    """Return closest-line parameter t on ray 0 for two 3D rays."""

    d0 = _normalize(d0)
    d1 = _normalize(d1)
    w0 = p0 - p1
    b = torch.sum(d0 * d1, dim=-1)
    d = torch.sum(d0 * w0, dim=-1)
    e = torch.sum(d1 * w0, dim=-1)
    denom = 1.0 - b * b
    if torch.abs(denom).reshape(-1)[0] < 1e-12:
        raise RuntimeError("differential rays are nearly parallel after tracing")
    return ((b * e - d) / denom).reshape(-1)[0]


def trace_distortion_sample(
    lens: Lensdata,
    *,
    field_x_deg: float,
    field_y_deg: float,
    axis: str,
    origin_kind: str,
    wavelength_nm: float,
    near_object_distance_mm: float,
) -> Dict[str, float]:
    """Trace one distortion sample and return angle/object-height metrics."""

    eye = extract_eye_positions(lens)
    origin_z = eye.eye_center_z_mm if origin_kind == "eye" else eye.pupil_z_mm
    with temporary_cb_tilt(lens, field_x_deg, field_y_deg):
        ray = make_eye_ray(
            lens,
            origin_z_mm=origin_z,
            field_x_deg=field_x_deg,
            field_y_deg=field_y_deg,
            wavelength_nm=wavelength_nm,
            direction_sign=-1.0,
        )
        traced, valid = lens.trace(ray, is_fixed=True)
    if not bool(valid.detach().cpu().reshape(-1)[0].item()):
        raise RuntimeError("distortion ray trace is invalid")

    axis_idx = 0 if axis == "x" else 1
    d = traced.d.reshape(-1, 3)[0]
    dz = d[2]
    theta_in = torch.rad2deg(torch.atan2(d[axis_idx], -dz))
    actual_point = intersect_z_plane(traced.o, traced.d, -float(near_object_distance_mm))
    ideal_point = intersect_z_plane(ray.o, ray.d, -float(near_object_distance_mm))
    return {
        "theta_in_deg": float(theta_in.detach().cpu().item()),
        "actual_height_mm": float(actual_point.reshape(-1, 3)[0, axis_idx].detach().cpu().item()),
        "ideal_height_mm": float(ideal_point.reshape(-1, 3)[0, axis_idx].detach().cpu().item()),
    }


def intersect_z_plane(o: torch.Tensor, d: torch.Tensor, z_mm: float) -> torch.Tensor:
    """Intersect ray(s) with a constant global-z plane, all units in mm."""

    z = torch.as_tensor(float(z_mm), dtype=o.dtype, device=o.device)
    denom = d[..., 2]
    if torch.any(torch.abs(denom) <= EPS):
        raise RuntimeError("ray is parallel to target z plane")
    t = (z - o[..., 2]) / denom
    return o + t[..., None] * d


def make_distortion_display_grids(rows: Sequence[Mapping[str, object]], display_grid_num: int):
    """Interpolate valid sample magnification onto a regular object grid."""

    valid = [r for r in rows if bool(r["valid"]) and np.isfinite(float(r["magnification"]))]
    if not valid:
        raise RuntimeError("No valid distortion grid rows for interpolation")
    points = np.array(
        [[float(r["object_x_ideal_mm"]), float(r["object_y_ideal_mm"])] for r in valid],
        dtype=float,
    )
    values = np.array([float(r["magnification"]) for r in valid], dtype=float)
    x_min, y_min = np.nanmin(points, axis=0)
    x_max, y_max = np.nanmax(points, axis=0)
    if abs(x_max - x_min) <= EPS:
        x_min -= 1.0
        x_max += 1.0
    if abs(y_max - y_min) <= EPS:
        y_min -= 1.0
        y_max += 1.0
    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, display_grid_num),
        np.linspace(y_min, y_max, display_grid_num),
        indexing="xy",
    )
    if len(valid) >= 4:
        mag = griddata(points, values, (gx, gy), method="linear")
    else:
        mag = np.full_like(gx, np.nan, dtype=float)
    nearest = griddata(points, values, (gx, gy), method="nearest")
    mag = np.where(np.isfinite(mag), mag, nearest)
    regular = np.stack((gx, gy), axis=-1)
    distorted = np.stack((gx * mag, gy * mag), axis=-1)
    return regular, mag, distorted


def make_distortion_display_grids_from_arrays(
    objx_in: np.ndarray,
    objy_in: np.ndarray,
    magnification: np.ndarray,
    x_obj: np.ndarray,
    y_obj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply original `griddata` then `magnification * grid` semantics."""

    gx, gy = np.meshgrid(np.asarray(x_obj, dtype=float), np.asarray(y_obj, dtype=float), indexing="xy")
    points = np.stack((np.asarray(objx_in, dtype=float).reshape(-1), np.asarray(objy_in, dtype=float).reshape(-1)), axis=1)
    values = np.asarray(magnification, dtype=float).reshape(-1)
    mask = np.isfinite(points).all(axis=1) & np.isfinite(values)
    if mask.sum() < 3:
        mag = np.full_like(gx, np.nan, dtype=float)
        display = np.full_like(gx, np.nan, dtype=float)
    else:
        mag = griddata(points[mask], values[mask], (gx, gy), method="linear")
        nearest = griddata(points[mask], values[mask], (gx, gy), method="nearest")
        display = np.where(np.isfinite(mag), mag, nearest)
    regular = np.stack((gx, gy), axis=-1)
    distorted = np.stack((gx * display, gy * display), axis=-1)
    return regular, mag, distorted, display


def make_distortion_display_grids_vector_from_arrays(
    objx_in: np.ndarray,
    objy_in: np.ndarray,
    corrected_xy: np.ndarray,
    x_obj: np.ndarray,
    y_obj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate affine-corrected 2D coordinates onto a regular object grid."""

    gx, gy = np.meshgrid(np.asarray(x_obj, dtype=float), np.asarray(y_obj, dtype=float), indexing="xy")
    points = np.stack((np.asarray(objx_in, dtype=float).reshape(-1), np.asarray(objy_in, dtype=float).reshape(-1)), axis=1)
    corrected = np.asarray(corrected_xy, dtype=float).reshape(-1, 2)
    mask = np.isfinite(points).all(axis=1) & np.isfinite(corrected).all(axis=1)
    if mask.sum() < 3:
        cx = np.full_like(gx, np.nan, dtype=float)
        cy = np.full_like(gy, np.nan, dtype=float)
        display_x = np.full_like(gx, np.nan, dtype=float)
        display_y = np.full_like(gy, np.nan, dtype=float)
    else:
        cx = griddata(points[mask], corrected[mask, 0], (gx, gy), method="linear")
        cy = griddata(points[mask], corrected[mask, 1], (gx, gy), method="linear")
        nearest_x = griddata(points[mask], corrected[mask, 0], (gx, gy), method="nearest")
        nearest_y = griddata(points[mask], corrected[mask, 1], (gx, gy), method="nearest")
        display_x = np.where(np.isfinite(cx), cx, nearest_x)
        display_y = np.where(np.isfinite(cy), cy, nearest_y)
    regular = np.stack((gx, gy), axis=-1)
    distorted = np.stack((display_x, display_y), axis=-1)
    mag = radial_ratio(np.stack((cx, cy), axis=-1).reshape(-1, 2), regular.reshape(-1, 2)).reshape(gx.shape)
    display = radial_ratio(distorted.reshape(-1, 2), regular.reshape(-1, 2)).reshape(gx.shape)
    return regular, mag, distorted, display


def distortion_empty_row(theta_deg: float) -> Dict[str, float | bool]:
    """Return a CSV row with all distortion curve columns initialized."""

    return {
        "theta_deg": float(theta_deg),
        "theta_in_deg": np.nan,
        "magnification": np.nan,
        "magnification_reference": np.nan,
        "distortion": np.nan,
        "distortion_percent": np.nan,
        "actual_height_mm": np.nan,
        "ideal_height_mm": np.nan,
        "reference_index": -1,
        "valid": False,
    }


def table_result(rows: Sequence[Mapping[str, object]], metadata: Mapping[str, object]) -> Dict[str, object]:
    """Convert row dicts into a result with stable columns and numeric array."""

    columns: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns and key != "error":
                columns.append(key)
    data = []
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, np.nan)
            if isinstance(value, bool):
                values.append(1.0 if value else 0.0)
            else:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    values.append(np.nan)
        data.append(values)
    return {
        "columns": columns,
        "data": np.asarray(data, dtype=float),
        "rows": list(rows),
        "metadata": dict(metadata),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write row dictionaries to UTF-8 CSV."""

    columns: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_power(result: Mapping[str, object], path: Path) -> None:
    """Plot power error and astigmatism curves to PNG."""

    rows = result["rows"]
    x = np.array([float(r["theta_deg"]) for r in rows], dtype=float)
    if rows and "local_sagittal_error_D" in rows[0]:
        sagittal = np.array([float(r["local_sagittal_error_D"]) for r in rows], dtype=float)
        meridian = np.array([float(r["local_meridional_error_D"]) for r in rows], dtype=float)
        mean = np.array([float(r["local_mean_error_D"]) for r in rows], dtype=float)
        astigmatism = np.array([float(r["local_astigmatism_D"]) for r in rows], dtype=float)
        labels = ["local sagittal error", "local meridional error", "local mean error", "local astigmatism"]
    else:
        sagittal = np.array([float(r["fp_sagittal_error_D"]) for r in rows], dtype=float)
        meridian = np.array([float(r["fp_meridian_error_D"]) for r in rows], dtype=float)
        mean = np.array([float(r["fp_mean_error_D"]) for r in rows], dtype=float)
        astigmatism = np.array([float(r["astigmatism_D"]) for r in rows], dtype=float)
        labels = ["trace sagittal error", "trace meridional error", "trace mean error", "trace astigmatism"]
    if _plot_power_matplotlib(path, x, sagittal, meridian, mean, astigmatism, labels=labels):
        return
    _write_line_plot_png(
        path,
        x,
        [sagittal, meridian, mean, astigmatism],
        [(255, 0, 0), (0, 0, 255), (40, 40, 40), (255, 0, 0)],
        labels=labels,
        x_label="Angle(deg)",
        y_label="Error(D)",
        dashed=[False, False, False, True],
    )


def plot_distortion_curve(result: Mapping[str, object], path: Path) -> None:
    """Plot magnification and relative distortion curve to PNG."""

    rows = result["rows"]
    x = np.array([float(r["theta_deg"]) for r in rows], dtype=float)
    mag = np.array([float(r["magnification"]) for r in rows], dtype=float)
    dist_percent = np.array([float(r["distortion_percent"]) for r in rows], dtype=float)
    if _plot_distortion_curve_matplotlib(path, x, mag, dist_percent):
        return
    _write_two_panel_line_plot_png(path, x, mag, dist_percent, (255, 0, 0), (255, 0, 0))


def plot_distortion_grid(result: Mapping[str, object], path: Path) -> None:
    """Plot regular and distorted grids to PNG."""

    grids = result["grids"]
    regular = np.asarray(grids["regular"], dtype=float)
    distorted = grids["distorted"]
    meta = dict(result.get("metadata", {}))
    unit = meta.get("grid_coordinate_unit", "mm")
    coord_ref = meta.get("grid_coordinate_reference", "exit_angle")
    if unit == "deg":
        if coord_ref == "incident_angle":
            xlabel, ylabel = r"$\theta_{x,\mathrm{in}}$ (°)", r"$\theta_{y,\mathrm{in}}$ (°)"
        else:
            xlabel, ylabel = r"$\theta_x$ (°)", r"$\theta_y$ (°)"
    else:
        xlabel, ylabel = f"X ({unit})", f"Y ({unit})"
    if _plot_distortion_grid_matplotlib(path, regular, distorted, xlabel=xlabel, ylabel=ylabel):
        return

    img = _new_canvas(800, 800)
    points = np.concatenate([regular.reshape(-1, 2), distorted.reshape(-1, 2)], axis=0)
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        _write_png_rgb(path, img)
        return
    bounds = _bounds(points[finite, 0], points[finite, 1])

    def draw_grid(grid: np.ndarray, color: Tuple[int, int, int], width: int) -> None:
        for i in range(grid.shape[0]):
            _draw_polyline(img, _map_points(grid[i, :, 0], grid[i, :, 1], bounds, img.shape), color, width)
            _draw_polyline(img, _map_points(grid[:, i, 0], grid[:, i, 1], bounds, img.shape), color, width)

    draw_grid(regular, (0, 45, 255), 1)
    draw_grid(distorted, (255, 0, 0), 2)
    _write_png_rgb(path, img)


def plot_footprint_coverage(result: Mapping[str, object], path: Path, *, trim_pixels: int = 3) -> None:
    """Plot rear-surface footprint convex hull over the AverFang Power map."""

    plt = _try_import_pyplot()
    if plt is None:
        raise RuntimeError("matplotlib is required to plot footprint coverage")
    averfang = result["averfang_map"]
    x = np.asarray(averfang["x_mm"], dtype=float)
    y = np.asarray(averfang["y_mm"], dtype=float)
    power = np.asarray(averfang["power_D"], dtype=float)
    x_plot, y_plot, power_plot = _trim_map_for_plot(x, y, power, trim_pixels)
    fx = np.asarray(result["footprint_x_mm"], dtype=float)
    fy = np.asarray(result["footprint_y_mm"], dtype=float)
    valid = np.asarray(result["valid"], dtype=bool)
    hull_xy = np.asarray(result["convex_hull_xy_mm"], dtype=float)
    hull_closed = np.vstack([hull_xy, hull_xy[0:1]]) if hull_xy.size else hull_xy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.rcParams.update(
                {
                    "font.family": ["Times New Roman", "DejaVu Serif"],
                    "axes.unicode_minus": False,
                    "mathtext.fontset": "dejavuserif",
                }
            )
            fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=180, constrained_layout=True)
            cmap = "turbo" if "turbo" in plt.colormaps() else "viridis"
            image = ax.imshow(
                power_plot,
                extent=[float(x_plot[0]), float(x_plot[-1]), float(y_plot[0]), float(y_plot[-1])],
                origin="lower",
                cmap=cmap,
                aspect="equal",
            )
            finite = np.isfinite(power_plot)
            if finite.any():
                levels = np.linspace(float(np.nanmin(power_plot)), float(np.nanmax(power_plot)), 14)
                if np.unique(np.round(levels, 12)).size > 1:
                    ax.contour(x_plot, y_plot, power_plot, levels=levels, colors="0.25", linewidths=0.45)
            ax.scatter(fx[valid], fy[valid], s=3.5, c="white", edgecolors="0.15", linewidths=0.15, alpha=0.75)
            ax.plot(hull_closed[:, 0], hull_closed[:, 1], color="#d62728", linewidth=1.8, label="Footprint hull")
            ax.plot(0.0, 0.0, "k+", markersize=5, linewidth=0.9)
            ax.set_xlim(float(x_plot[0]), float(x_plot[-1]))
            ax.set_ylim(float(y_plot[0]), float(y_plot[-1]))
            ax.set_xlabel("y/mm", fontsize=12, fontweight="bold")
            ax.set_ylabel("x/mm", fontsize=12, fontweight="bold")
            ax.set_title("Footprint Coverage on Power (D)", fontsize=12, fontweight="bold")
            ax.tick_params(direction="in", top=True, right=True)
            ax.legend(loc="upper right", fontsize=8, frameon=True)
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_title("Power (D)", fontsize=9, fontweight="bold")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path)
            plt.close(fig)
    except Exception:
        plt.close("all")
        raise


def _try_import_pyplot():
    """Try importing matplotlib; return plt or None if unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _style_axes(ax) -> None:
    ax.tick_params(direction="in", top=True, right=True, width=0.8, labelsize=12)
    ax.grid(True, linestyle="--", linewidth=0.4, color="0.85")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.35")


def _plot_power_matplotlib(
    path: Path,
    angle_deg: np.ndarray,
    sagittal: np.ndarray,
    meridian: np.ndarray,
    mean: np.ndarray,
    astigmatism: np.ndarray,
    labels: Optional[Sequence[str]] = None,
) -> bool:
    plt = _try_import_pyplot()
    if plt is None:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.rcParams.update(
                {
                    "font.family": ["Times New Roman", "DejaVu Serif"],
                    "axes.unicode_minus": False,
                    "mathtext.fontset": "dejavuserif",
                }
            )
            fig, ax = plt.subplots(figsize=(6.7, 4.4), dpi=180)
            labels = list(labels or ["sagittal", "meridional", "mean", "astigmatism"])
            ax.plot(sagittal, angle_deg, color="#E63946", linewidth=1.5, label=labels[0])
            ax.plot(meridian, angle_deg, color="#457B9D", linewidth=1.5, label=labels[1])
            ax.plot(mean, angle_deg, color="black", linewidth=1.5, label=labels[2], linestyle="--")
            ax.plot(astigmatism, angle_deg, color="#E63946", linewidth=1.5, label=labels[3], linestyle=":")
            ax.set_xlabel("Power / Astigmatism (D)", fontsize=14, fontweight="bold")
            ax.set_ylabel("Angle ($^\\circ$)", fontsize=14, fontweight="bold")
            ax.legend(loc="upper left", frameon=True, framealpha=0.9, facecolor="white", edgecolor="0.8", fontsize=10)
            ax.set_ylim(bottom=0.0)
            _style_axes(ax)
            fig.subplots_adjust(left=0.14, right=0.98, bottom=0.13, top=0.97)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=180)
            plt.close(fig)
            return True
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return False


def _plot_distortion_curve_matplotlib(path: Path, angle_deg: np.ndarray, magnification: np.ndarray, distortion_percent: np.ndarray) -> bool:
    plt = _try_import_pyplot()
    if plt is None:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.rcParams.update(
                {
                    "font.family": ["Times New Roman", "DejaVu Serif"],
                    "axes.unicode_minus": False,
                    "mathtext.fontset": "dejavuserif",
                }
            )
            fig, axes = plt.subplots(2, 1, figsize=(5.8, 6.7), dpi=180, sharey=True)
            axes[0].plot(magnification, angle_deg, color="#457B9D", linewidth=1.5, label="Magnification")
            axes[1].plot(distortion_percent, angle_deg, color="#E63946", linewidth=1.5, label="Distortion")
            axes[0].set_xlabel("Magnification", fontsize=13, fontweight="bold")
            axes[1].set_xlabel("Distortion (%)", fontsize=13, fontweight="bold")
            axes[0].set_ylabel("Angle ($^\\circ$)", fontsize=14, fontweight="bold")
            for ax in axes:
                ax.legend(loc="best", frameon=True, framealpha=0.9, facecolor="white", edgecolor="0.8", fontsize=10)
                ax.set_ylim(bottom=0.0)
                _style_axes(ax)
            fig.subplots_adjust(left=0.16, right=0.98, bottom=0.11, top=0.98, hspace=0.14)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=180)
            plt.close(fig)
            return True
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return False


def _plot_distortion_grid_matplotlib(
    path: Path,
    regular: np.ndarray,
    distorted: np.ndarray,
    *,
    xlabel: str = "X",
    ylabel: str = "Y",
) -> bool:
    plt = _try_import_pyplot()
    if plt is None:
        return False
    try:
        fig, ax = plt.subplots(figsize=(5.1, 5.1), dpi=180)
        for grid, color, linewidth in ((regular, "blue", 0.75), (distorted, "red", 0.85)):
            for i in range(grid.shape[0]):
                ax.plot(grid[i, :, 0], grid[i, :, 1], color=color, linewidth=linewidth)
                ax.plot(grid[:, i, 0], grid[:, i, 1], color=color, linewidth=linewidth)
        points = np.concatenate([regular.reshape(-1, 2), distorted.reshape(-1, 2)], axis=0)
        finite = np.isfinite(points).all(axis=1)
        if finite.any():
            x_min, x_max = float(np.nanmin(points[finite, 0])), float(np.nanmax(points[finite, 0]))
            y_min, y_max = float(np.nanmin(points[finite, 1])), float(np.nanmax(points[finite, 1]))
            span = max(x_max - x_min, y_max - y_min, EPS)
            cx = 0.5 * (x_min + x_max)
            cy = 0.5 * (y_min + y_max)
            margin = 0.04 * span
            ax.set_xlim(cx - 0.5 * span - margin, cx + 0.5 * span + margin)
            ax.set_ylim(cy - 0.5 * span - margin, cy + 0.5 * span + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 刻度对齐蓝色（理想无畸变参考）网格的位置
        n = regular.shape[0]
        step = max(1, (n - 1) // 4)  # ~5 个刻度
        indices = list(range(0, n - 1, step)) + [n - 1]
        ax.set_xticks(regular[0, indices, 0])
        ax.set_yticks(regular[indices, 0, 1])

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return True
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return False


def _write_two_panel_line_plot_png(
    path: Path,
    x: np.ndarray,
    y_top: np.ndarray,
    y_bottom: np.ndarray,
    color_top: Tuple[int, int, int],
    color_bottom: Tuple[int, int, int],
) -> None:
    """Render a two-panel line plot PNG using only NumPy and stdlib zlib."""

    img = _new_canvas(900, 680)
    finite_x = x[np.isfinite(x)]
    if finite_x.size == 0:
        _write_png_rgb(path, img)
        return

    panels = [
        (y_top, color_top, (70, 45, img.shape[1] - 35, 305), "magnification", "Magnification"),
        (y_bottom, color_bottom, (70, 375, img.shape[1] - 35, img.shape[0] - 55), "distortion", "Distortion(%)"),
    ]
    for y, color, panel, label, y_label in panels:
        finite_y = y[np.isfinite(y)]
        if finite_y.size == 0:
            continue
        left, top, right, bottom = panel
        bounds = _bounds(finite_x, finite_y)
        bounds = (0.0, bounds[1], bounds[2], bounds[3])
        _draw_plot_box(img, left, top, right, bottom)
        _draw_text(img, y_label, left + 5, top - 12, (0, 0, 0), 0.52, 1)
        _draw_line(img, left + 35, top + 26, left + 80, top + 26, color, 2)
        _draw_text(img, label, left + 90, top + 31, (0, 0, 0), 0.52, 1)
        plot_shape = (img.shape[0], img.shape[1], left, top, right, bottom)
        _draw_polyline(img, _map_points(x, y, bounds, plot_shape), color, 2)
    _draw_text(img, "Angle(deg)", img.shape[1] // 2 - 55, img.shape[0] - 16, (0, 0, 0), 0.58, 1)
    _write_png_rgb(path, img)


def _write_line_plot_png(
    path: Path,
    x: np.ndarray,
    series: Sequence[np.ndarray],
    colors: Sequence[Tuple[int, int, int]],
    *,
    labels: Optional[Sequence[str]] = None,
    x_label: str = "",
    y_label: str = "",
    dashed: Optional[Sequence[bool]] = None,
) -> None:
    """Render a simple line plot PNG using only NumPy and stdlib zlib."""

    img = _new_canvas(900, 560)
    finite_x = x[np.isfinite(x)]
    finite_y = np.concatenate([y[np.isfinite(y)] for y in series if np.isfinite(y).any()])
    if finite_x.size == 0 or finite_y.size == 0:
        _write_png_rgb(path, img)
        return
    bounds = _bounds(finite_x, finite_y)
    bounds = (0.0, bounds[1], bounds[2], bounds[3])
    left, top, right, bottom = 70, 40, img.shape[1] - 30, img.shape[0] - 55
    dash_flags = list(dashed) if dashed is not None else [False] * len(series)
    _draw_plot_box(img, left, top, right, bottom)
    if y_label:
        _draw_text(img, y_label, left + 5, top - 12, (0, 0, 0), 0.58, 1)
    if x_label:
        _draw_text(img, x_label, (left + right) // 2 - 55, img.shape[0] - 18, (0, 0, 0), 0.58, 1)
    if labels:
        legend_x = left + 35
        legend_y = top + 30
        for idx, (label, color) in enumerate(zip(labels, colors)):
            y0 = legend_y + idx * 24
            if idx < len(dash_flags) and dash_flags[idx]:
                _draw_dashed_line(img, legend_x, y0 - 5, legend_x + 45, y0 - 5, color, 2)
            else:
                _draw_line(img, legend_x, y0 - 5, legend_x + 45, y0 - 5, color, 2)
            _draw_text(img, str(label), legend_x + 55, y0, (0, 0, 0), 0.52, 1)
    plot_shape = (img.shape[0], img.shape[1], left, top, right, bottom)
    for idx, (y, color) in enumerate(zip(series, colors)):
        _draw_polyline(img, _map_points(x, y, bounds, plot_shape), color, 2, dashed=idx < len(dash_flags) and dash_flags[idx])
    _write_png_rgb(path, img)


def _draw_plot_box(img: np.ndarray, left: int, top: int, right: int, bottom: int) -> None:
    _draw_line(img, left, bottom, right, bottom, (80, 80, 80), 1)
    _draw_line(img, left, top, right, top, (80, 80, 80), 1)
    _draw_line(img, left, top, left, bottom, (80, 80, 80), 1)
    _draw_line(img, right, top, right, bottom, (80, 80, 80), 1)
    for frac in np.linspace(0, 1, 5):
        yy = int(round(top + frac * (bottom - top)))
        _draw_line(img, left, yy, right, yy, (235, 235, 235), 1)
    for frac in np.linspace(0, 1, 6):
        xx = int(round(left + frac * (right - left)))
        _draw_line(img, xx, bottom - 4, xx, bottom + 4, (80, 80, 80), 1)
        _draw_line(img, xx, top - 4, xx, top + 4, (80, 80, 80), 1)


def _new_canvas(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _bounds(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    if abs(xmax - xmin) <= EPS:
        xmin -= 1.0
        xmax += 1.0
    if abs(ymax - ymin) <= EPS:
        ymin -= 1.0
        ymax += 1.0
    xpad = 0.05 * (xmax - xmin)
    ypad = 0.08 * (ymax - ymin)
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def _map_points(
    x: np.ndarray,
    y: np.ndarray,
    bounds: Tuple[float, float, float, float],
    shape,
) -> List[Optional[Tuple[int, int]]]:
    if len(shape) == 6:
        height, width, left, top, right, bottom = shape
    else:
        height, width = shape[:2]
        left, top, right, bottom = 40, 40, width - 40, height - 40
    xmin, xmax, ymin, ymax = bounds
    pts: List[Optional[Tuple[int, int]]] = []
    for xx, yy in zip(np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)):
        if not np.isfinite(xx) or not np.isfinite(yy):
            pts.append(None)
            continue
        px = int(round(left + (float(xx) - xmin) / (xmax - xmin) * (right - left)))
        py = int(round(bottom - (float(yy) - ymin) / (ymax - ymin) * (bottom - top)))
        pts.append((px, py))
    return pts


def _draw_polyline(
    img: np.ndarray,
    pts: Sequence[Optional[Tuple[int, int]]],
    color: Tuple[int, int, int],
    width: int,
    *,
    dashed: bool = False,
) -> None:
    prev = None
    for pt in pts:
        if pt is None:
            prev = None
            continue
        if prev is not None:
            if dashed:
                _draw_dashed_line(img, prev[0], prev[1], pt[0], pt[1], color, width)
            else:
                _draw_line(img, prev[0], prev[1], pt[0], pt[1], color, width)
        _draw_dot(img, pt[0], pt[1], color, max(1, width))
        prev = pt


def _draw_dot(img: np.ndarray, x: int, y: int, color: Tuple[int, int, int], radius: int) -> None:
    h, w = img.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    img[y0:y1, x0:x1] = color


def _draw_line(
    img: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
    width: int,
) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        _draw_dot(img, x, y, color, width)


def _draw_dashed_line(
    img: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
    width: int,
) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    dash_len = 10
    gap_len = 7
    for i in range(steps + 1):
        if (i % (dash_len + gap_len)) >= dash_len:
            continue
        t = i / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        _draw_dot(img, x, y, color, width)


def _draw_text(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
    scale: float,
    thickness: int,
) -> None:
    try:
        import cv2

        cv2.putText(
            img,
            str(text),
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(scale),
            tuple(int(c) for c in color),
            int(thickness),
            cv2.LINE_AA,
        )
    except Exception:
        return


def _write_png_rgb(path: Path, img: np.ndarray) -> None:
    """Write an RGB uint8 image as a PNG file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk("IHDR".encode("ascii"), struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk("IDAT".encode("ascii"), zlib.compress(raw, level=6))
    png += chunk("IEND".encode("ascii"), b"")
    path.write_bytes(png)


def _wavelength_tensor(lens: Lensdata, wavelength_nm: float) -> torch.Tensor:
    return torch.tensor([float(wavelength_nm)], dtype=torch.float64, device=lens.device)


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(EPS)


def _scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)
