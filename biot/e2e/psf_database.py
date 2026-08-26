from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage
from scipy.io import loadmat, savemat
from scipy.signal import convolve2d, fftconvolve


DEFAULT_SIZE_REFERENCE_MM = 0.184378803949209
DEFAULT_CROP_SIZE_REFERENCE = 130
DEFAULT_F_EYE_MM = 16.667


@dataclass(frozen=True)
class RenderSimulationConfig:
    """External render-fast compatible simulation metadata.

    Units:
        field ranges and render FOV are in degree; PSF physical sizes are in
        mm. ``crop_size_reference`` is the saved square PSF tile side length in
        pixels.
    """

    x_min: float = -40.0
    x_max: float = 40.0
    y_min: float = -40.0
    y_max: float = 40.0
    step: float = 5.0
    size_reference_mm: float = DEFAULT_SIZE_REFERENCE_MM
    crop_size_reference: int = DEFAULT_CROP_SIZE_REFERENCE
    scale_x: float = 202.4531
    f_eye_mm: float = DEFAULT_F_EYE_MM
    tile_size: int = DEFAULT_CROP_SIZE_REFERENCE
    tile_gap: int = 5
    fov_x_deg: float = 80.0
    fov_y_deg: float = 80.0
    pitch_deg: float = 5.0

    def to_mat_struct(self) -> dict[str, Any]:
        return {
            "field": {
                "xMin": float(self.x_min),
                "xMax": float(self.x_max),
                "yMin": float(self.y_min),
                "yMax": float(self.y_max),
                "step": float(self.step),
            },
            "psf": {
                "sizeReference": float(self.size_reference_mm),
                "cropSizeReference": int(self.crop_size_reference),
                "scaleX": float(self.scale_x),
                "fEye": float(self.f_eye_mm),
            },
            "stitch": {"tileSize": int(self.tile_size), "tileGap": int(self.tile_gap)},
            "render": {
                "fovX": float(self.fov_x_deg),
                "fovY": float(self.fov_y_deg),
                "pitch": float(self.pitch_deg),
            },
        }


@dataclass(frozen=True)
class RenderPsfSample:
    """One PSF sample using the external ``psf_images`` field layout."""

    field_x: float
    field_y: float
    map: np.ndarray
    factor_x_um: float
    map1: np.ndarray
    map_mm: float
    crop_map: np.ndarray


def render_field_grid(config: RenderSimulationConfig) -> np.ndarray:
    """Return external-builder field order: Y descending, X descending."""
    y_values = np.arange(config.y_max, config.y_min - 0.5 * config.step, -config.step)
    x_values = np.arange(config.x_max, config.x_min - 0.5 * config.step, -config.step)
    return np.asarray([(x, y) for y in y_values for x in x_values], dtype=np.float64)


def normalize_psf_energy(psf: np.ndarray) -> np.ndarray:
    arr = np.asarray(psf, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"Invalid PSF energy for normalization: {total}")
    return arr / total


def compute_biot_fft_pixel_pitch_from_f_number(
    f_number: float,
    *,
    biot_np: int,
    wavelength_nm: float,
) -> float:
    """Mirror BIOT ``fft_psf_i`` image-plane pitch formula.

    Returns:
        Pixel pitch in mm/pixel for the raw FFT PSF grid.
    """
    n_p = int(biot_np)
    if n_p <= 0:
        raise ValueError("biot_np must be positive")
    lam_mm = float(wavelength_nm) * 1.0e-6
    f_number = float(f_number)
    if n_p == 32:
        return f_number * lam_mm * (n_p - 2) / 2.0 / n_p
    return (f_number * lam_mm * (n_p - 2) / 2.0 / n_p) * ((32.0 / n_p) ** 0.5)


def compute_physical_fft_pixel_pitch_from_f_number(
    f_number: float,
    *,
    pupil_sample_count: int,
    psf_size_px: int,
    wavelength_nm: float,
) -> float:
    """Return Fraunhofer FFT image-plane pitch for the e2e raw PSF grid.

    The e2e complex pupil is sampled on a square grid spanning the physical
    pupil diameter. With pupil spacing ``du = D / (Np - 1)`` and focal length
    ``f = F# * D``, the FFT image-plane sampling is
    ``dx = wavelength * f / (Nfft * du)``. The pupil diameter cancels, giving
    ``dx = wavelength * F# * (Np - 1) / Nfft``.
    """
    n_pupil = int(pupil_sample_count)
    n_fft = int(psf_size_px)
    if n_pupil <= 1:
        raise ValueError("pupil_sample_count must be greater than 1")
    if n_fft <= 0:
        raise ValueError("psf_size_px must be positive")
    lam_mm = float(wavelength_nm) * 1.0e-6
    return lam_mm * float(f_number) * float(n_pupil - 1) / float(n_fft)


def compute_biot_fft_pixel_pitch_mm(lens: Any, *, biot_np: int, wavelength_nm: float):
    """Compute the BIOT-aligned raw FFT PSF pixel pitch from a loaded Lensdata."""
    import torch

    lam = torch.as_tensor(float(wavelength_nm) * 1.0e-6, device=lens.device, dtype=torch.float64)
    f_number = lens.cal_WFNO(lam)
    if torch.is_tensor(f_number):
        f_number = float(f_number.detach().cpu().reshape(-1)[0].item())
    return compute_biot_fft_pixel_pitch_from_f_number(
        float(f_number),
        biot_np=int(biot_np),
        wavelength_nm=float(wavelength_nm),
    )


def compute_physical_fft_pixel_pitch_mm(
    lens: Any,
    *,
    pupil_sample_count: int,
    psf_size_px: int,
    wavelength_nm: float,
) -> float:
    """Compute the physically scaled raw FFT PSF pixel pitch from Lensdata F#."""
    import torch

    lam = torch.as_tensor(float(wavelength_nm) * 1.0e-6, device=lens.device, dtype=torch.float64)
    f_number = lens.cal_WFNO(lam)
    if torch.is_tensor(f_number):
        f_number = float(f_number.detach().cpu().reshape(-1)[0].item())
    return compute_physical_fft_pixel_pitch_from_f_number(
        float(f_number),
        pupil_sample_count=int(pupil_sample_count),
        psf_size_px=int(psf_size_px),
        wavelength_nm=float(wavelength_nm),
    )


def crop_resize_external_map(
    raw_map: np.ndarray,
    pixel_pitch_mm: float,
    config: RenderSimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(map, crop_map)`` using the external MATLAB-style crop rule.

    ``crop_map`` is the native physical crop from the raw FFT PSF. ``map`` is
    bicubic-resized to ``config.crop_size_reference`` and energy-normalized for
    physical rendering.
    """
    raw = normalize_psf_energy(raw_map)
    pixel_pitch_mm = float(pixel_pitch_mm)
    if not np.isfinite(pixel_pitch_mm) or pixel_pitch_mm <= 0.0:
        raise ValueError(f"Invalid pixel_pitch_mm: {pixel_pitch_mm}")

    crop_size = float(config.size_reference_mm) / pixel_pitch_mm
    half = int(round(crop_size / 2.0))
    center_y = (raw.shape[0] + 1) / 2.0 - 1.0
    center_x = (raw.shape[1] + 1) / 2.0 - 1.0
    y0 = int(round(center_y - half))
    x0 = int(round(center_x - half))
    y1 = y0 + int(round(crop_size)) + 1
    x1 = x0 + int(round(crop_size)) + 1

    if y0 < 0 or x0 < 0 or y1 > raw.shape[0] or x1 > raw.shape[1]:
        zeros = np.zeros((int(config.crop_size_reference), int(config.crop_size_reference)), dtype=np.float64)
        return zeros, zeros

    crop = normalize_psf_energy(raw[y0:y1, x0:x1])
    resized = cv2.resize(
        crop.astype(np.float64, copy=False),
        (int(config.crop_size_reference), int(config.crop_size_reference)),
        interpolation=cv2.INTER_CUBIC,
    )
    return normalize_psf_energy(resized), crop


def build_render_psf_sample(
    *,
    field_x: float,
    field_y: float,
    raw_map: np.ndarray,
    pixel_pitch_mm: float,
    config: RenderSimulationConfig,
) -> RenderPsfSample:
    """Convert a raw BIOT/e2e FFT PSF into external ``psf_images`` fields."""
    raw = normalize_psf_energy(raw_map)
    map_data, crop_map = crop_resize_external_map(raw, pixel_pitch_mm, config)
    factor_x_um = float(pixel_pitch_mm) * 1000.0
    return RenderPsfSample(
        field_x=float(field_x),
        field_y=float(field_y),
        map=map_data,
        factor_x_um=factor_x_um,
        map1=raw,
        map_mm=float(raw.shape[0]) * float(pixel_pitch_mm),
        crop_map=crop_map,
    )


def save_render_psf_mat(path: str | Path, samples: list[RenderPsfSample], config: RenderSimulationConfig) -> None:
    """Save external render-fast compatible MAT with ``psf_images``."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    psf_images = []
    for sample in samples:
        psf_images.append(
            {
                "fieldX": float(sample.field_x),
                "fieldY": float(sample.field_y),
                "factorX": float(sample.factor_x_um),
                "map": np.asarray(sample.map, dtype=np.float64),
                "map1": np.asarray(sample.map1, dtype=np.float64),
                "size": int(np.asarray(sample.map1).shape[0]),
                "mapMM": float(sample.map_mm),
                "crop_map": np.asarray(sample.crop_map, dtype=np.float64),
            }
        )
    savemat(
        output,
        {"psf_images": np.asarray(psf_images, dtype=object), "simulationConfig": config.to_mat_struct()},
        do_compression=True,
        oned_as="row",
    )


def visualize_render_psf_database(
    samples: list[RenderPsfSample],
    config: RenderSimulationConfig,
    output_dir: str | Path,
    *,
    target_path: str | Path | None = "E1.mat",
    psf_display_mode: str = "tile-peak",
    psf_display_flip_y: bool = True,
) -> tuple[Path, Path]:
    """Save external-style PSF and target convolution atlas images.

    This mirrors ``multifocus_render.visualize.visualize_psf_db``: each field
    PSF tile is stitched into a large atlas, and a same-size target tile is
    convolved with that field's PSF for the target atlas. The default target is
    ``E1.mat``; missing target files fail instead of falling back to a built-in
    target. ``psf_display_mode`` and ``psf_display_flip_y`` only affect
    ``psf_map.png`` display values; target convolution always uses the original
    PSF tile normalized by energy.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = _load_visualization_target(target_path, int(config.tile_size))

    cols = int(round((config.x_max - config.x_min) / config.step)) + 1
    rows = int(round((config.y_max - config.y_min) / config.step)) + 1
    canvas_h = int(rows * (config.tile_size + config.tile_gap))
    canvas_w = int(cols * (config.tile_size + config.tile_gap))
    stitched_psf = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    stitched_conv = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for sample in samples:
        tile = np.asarray(sample.map, dtype=np.float32)
        if tile.shape != (int(config.tile_size), int(config.tile_size)):
            tile = cv2.resize(
                tile,
                (int(config.tile_size), int(config.tile_size)),
                interpolation=cv2.INTER_CUBIC,
            )
        kernel = tile.copy()
        total = float(kernel.sum())
        if total > 0.0:
            kernel /= total
        conv = fftconvolve(target, kernel, mode="same")
        display_tile = _prepare_psf_display_tile(tile, psf_display_mode, flip_y=psf_display_flip_y)
        row = int(round((float(sample.field_y) - config.y_min) / config.step))
        col = int(round((float(sample.field_x) - config.x_min) / config.step))
        y0 = row * (int(config.tile_size) + int(config.tile_gap))
        x0 = col * (int(config.tile_size) + int(config.tile_gap))
        stitched_psf[y0 : y0 + int(config.tile_size), x0 : x0 + int(config.tile_size)] = display_tile
        stitched_conv[y0 : y0 + int(config.tile_size), x0 : x0 + int(config.tile_size)] = conv

    psf_img = ndimage.gaussian_filter(stitched_psf, 2.0)
    conv_img = ndimage.gaussian_filter(stitched_conv, 1.0)
    psf_out = output / "psf_map.png"
    target_out = output / "target_map.png"
    _save_atlas(psf_out, psf_img, config, "jet", "field X (Degrees)", "field Y (Degrees)")
    _save_atlas(target_out, conv_img, config, "gray", "field X (Degrees)", "field Y (Degrees)")
    return psf_out, target_out


def _prepare_psf_display_tile(tile: np.ndarray, mode: str, *, flip_y: bool = True) -> np.ndarray:
    image = _psf_display_tile(tile, mode)
    if flip_y:
        return np.flipud(image)
    return image


def _psf_display_tile(tile: np.ndarray, mode: str) -> np.ndarray:
    image = np.asarray(tile, dtype=np.float32)
    mode_key = str(mode).strip().lower().replace("_", "-")
    if mode_key == "raw":
        return image
    if mode_key == "tile-peak":
        peak = float(np.max(image))
        if peak > 0.0:
            return image / peak
        return image
    if mode_key == "log":
        peak = float(np.max(image))
        if peak <= 0.0:
            return image
        return np.log1p(image / peak * 255.0).astype(np.float32) / np.log1p(255.0)
    raise ValueError(f"Unsupported psf display mode: {mode}")


def _load_visualization_target(target_path: str | Path | None, tile_size: int) -> np.ndarray:
    if target_path is None:
        raise FileNotFoundError("visualization target is required; pass --visualize-target E1.mat")
    path = Path(target_path)
    if not path.exists():
        raise FileNotFoundError(f"Visualization target not found: {path}")

    target = None
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        import pandas as pd

        target = pd.read_excel(path, header=None, engine="openpyxl").values.astype(np.float32)
    elif suffix == ".mat":
        try:
            data = loadmat(path, squeeze_me=True, struct_as_record=False)
            for key, value in data.items():
                if not key.startswith("__"):
                    target = np.asarray(value, dtype=np.float32)
                    break
        except NotImplementedError:
            import h5py

            with h5py.File(path, "r") as handle:
                for key in handle.keys():
                    target = np.asarray(handle[key], dtype=np.float32)
                    if target.ndim == 2:
                        target = target.T
                    break
    else:
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            target = image.astype(np.float32)

    if target is None:
        raise ValueError(f"Could not load visualization target data from: {path}")
    target = cv2.resize(np.asarray(target, dtype=np.float32), (int(tile_size), int(tile_size)), interpolation=cv2.INTER_CUBIC)
    target -= float(target.min())
    if float(target.max()) > 0.0:
        target /= float(target.max())
    return target.astype(np.float32)


def _save_atlas(
    path: Path,
    image: np.ndarray,
    config: RenderSimulationConfig,
    cmap: str,
    xlabel: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6.5), facecolor="white")
    ax = fig.add_subplot(111)
    im = ax.imshow(
        image,
        extent=_display_extent(config),
        origin="lower",
        cmap=cmap,
        aspect="equal",
    )
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _display_extent(config: RenderSimulationConfig) -> tuple[float, float, float, float]:
    """Return a non-singular imshow extent for atlas display only."""
    x_min = float(config.x_min)
    x_max = float(config.x_max)
    y_min = float(config.y_min)
    y_max = float(config.y_max)
    half_step = 0.5 * abs(float(config.step)) if float(config.step) != 0.0 else 0.5
    if x_min == x_max:
        x_min -= half_step
        x_max += half_step
    if y_min == y_max:
        y_min -= half_step
        y_max += half_step
    return x_min, x_max, y_min, y_max


def _mat_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _mat_items(value: Any) -> list[Any]:
    arr = np.asarray(value, dtype=object)
    if arr.shape == ():
        return [arr.item()]
    return list(arr.ravel())


def load_render_psf_mat(path: str | Path) -> tuple[list[RenderPsfSample], RenderSimulationConfig | None]:
    """Load a MAT saved by this module or the external multifocus builder."""
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "psf_images" not in data:
        raise ValueError(f"MAT file missing psf_images: {path}")
    samples: list[RenderPsfSample] = []
    for item in _mat_items(data["psf_images"]):
        raw = np.asarray(_mat_field(item, "map1", _mat_field(item, "map")), dtype=np.float64)
        factor_x_um = float(np.asarray(_mat_field(item, "factorX", 0.0)).reshape(-1)[0])
        samples.append(
            RenderPsfSample(
                field_x=float(np.asarray(_mat_field(item, "fieldX")).reshape(-1)[0]),
                field_y=float(np.asarray(_mat_field(item, "fieldY")).reshape(-1)[0]),
                factor_x_um=factor_x_um,
                map=np.asarray(_mat_field(item, "map"), dtype=np.float64),
                map1=raw,
                map_mm=float(np.asarray(_mat_field(item, "mapMM", 0.0)).reshape(-1)[0]),
                crop_map=np.asarray(_mat_field(item, "crop_map", _mat_field(item, "map")), dtype=np.float64),
            )
        )
    config = None
    if "simulationConfig" in data:
        config = _config_from_mat(data["simulationConfig"])
    return samples, config


def _config_from_mat(obj: Any) -> RenderSimulationConfig:
    field = _mat_field(obj, "field")
    psf = _mat_field(obj, "psf")
    stitch = _mat_field(obj, "stitch")
    render = _mat_field(obj, "render")
    return RenderSimulationConfig(
        x_min=float(_mat_field(field, "xMin", -40.0)),
        x_max=float(_mat_field(field, "xMax", 40.0)),
        y_min=float(_mat_field(field, "yMin", -40.0)),
        y_max=float(_mat_field(field, "yMax", 40.0)),
        step=float(_mat_field(field, "step", 5.0)),
        size_reference_mm=float(_mat_field(psf, "sizeReference", DEFAULT_SIZE_REFERENCE_MM)),
        crop_size_reference=int(_mat_field(psf, "cropSizeReference", DEFAULT_CROP_SIZE_REFERENCE)),
        scale_x=float(_mat_field(psf, "scaleX", 202.4531)),
        f_eye_mm=float(_mat_field(psf, "fEye", DEFAULT_F_EYE_MM)),
        tile_size=int(_mat_field(stitch, "tileSize", DEFAULT_CROP_SIZE_REFERENCE)),
        tile_gap=int(_mat_field(stitch, "tileGap", 5)),
        fov_x_deg=float(_mat_field(render, "fovX", 80.0)),
        fov_y_deg=float(_mat_field(render, "fovY", 80.0)),
        pitch_deg=float(_mat_field(render, "pitch", 5.0)),
    )


def render_scene_fast_scipy_compatible(
    rgb: np.ndarray,
    samples: list[RenderPsfSample],
    config: RenderSimulationConfig,
) -> np.ndarray:
    """SciPy version of the external ``render-fast`` algorithm.

    This intentionally follows the external project's node-weighted convolution
    strategy: per pixel field interpolation is converted into four field-node
    weight maps, each node convolves the full image, then the valid crop is
    accumulated.
    """
    image = np.asarray(rgb, dtype=np.float32)
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0
    size_y, size_x = image.shape[:2]
    crop_size = _reference_crop_size(size_x, config)
    field_x_deg, field_y_deg = _field_angle_maps(size_y, size_x, config)
    x_axis = np.sort(np.unique([s.field_x for s in samples])).astype(np.float32)
    y_axis = np.sort(np.unique([s.field_y for s in samples])).astype(np.float32)
    lookup = {(round(s.field_x, 6), round(s.field_y, 6)): np.asarray(s.map, dtype=np.float32) for s in samples}
    x0, x1, wx0, wx1 = _linear_axis_weights(field_x_deg, x_axis)
    y0, y1, wy0, wy1 = _linear_axis_weights(field_y_deg, y_axis)
    terms = ((x0, y0, wx0 * wy0), (x1, y0, wx1 * wy0), (x0, y1, wx0 * wy1), (x1, y1, wx1 * wy1))

    half = crop_size // 2
    output = np.zeros((size_y, size_x, 3), dtype=np.float32)
    for xi, yi, weight in terms:
        for tx in np.unique(xi):
            tx_mask = xi == tx
            for ty in np.unique(yi[tx_mask]):
                node_weight = weight * tx_mask * (yi == ty)
                if float(np.sum(node_weight)) <= 0.0:
                    continue
                kernel = _lookup_render_kernel(lookup, x_axis, y_axis, int(tx), int(ty), crop_size)
                for channel in range(3):
                    full = convolve2d(
                        image[..., channel] * node_weight,
                        kernel,
                        mode="full",
                        boundary="fill",
                        fillvalue=0.0,
                    )
                    output[..., channel] += full[half : half + size_y, half : half + size_x]
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def render_scene_fast_torch_compatible(
    rgb: np.ndarray,
    samples: list[RenderPsfSample],
    config: RenderSimulationConfig,
    *,
    device: str = "cuda",
) -> np.ndarray:
    """Torch version of the external ``render-fast`` algorithm."""
    try:
        import torch
    except ImportError:
        print("[warning] torch is not available; falling back to scipy render-fast.")
        return render_scene_fast_scipy_compatible(rgb, samples, config)

    selected_device = str(device)
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA is not available; falling back to CPU render-fast.")
        selected_device = "cpu"

    image = np.asarray(rgb, dtype=np.float32)
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0
    size_y, size_x = image.shape[:2]
    crop_size = _reference_crop_size(size_x, config)
    field_x_deg, field_y_deg = _field_angle_maps(size_y, size_x, config)
    x_axis = np.sort(np.unique([s.field_x for s in samples])).astype(np.float32)
    y_axis = np.sort(np.unique([s.field_y for s in samples])).astype(np.float32)
    lookup = {(round(s.field_x, 6), round(s.field_y, 6)): np.asarray(s.map, dtype=np.float32) for s in samples}
    x0, x1, wx0, wx1 = _linear_axis_weights(field_x_deg, x_axis)
    y0, y1, wy0, wy1 = _linear_axis_weights(field_y_deg, y_axis)
    terms = ((x0, y0, wx0 * wy0), (x1, y0, wx1 * wy0), (x0, y1, wx0 * wy1), (x1, y1, wx1 * wy1))

    torch_device = torch.device(selected_device)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(torch_device)
    half = crop_size // 2
    output = torch.zeros_like(tensor)
    with torch.no_grad():
        for xi, yi, weight in terms:
            for tx in np.unique(xi):
                tx_mask = xi == tx
                for ty in np.unique(yi[tx_mask]):
                    node_weight = weight * tx_mask * (yi == ty)
                    if float(np.sum(node_weight)) <= 0.0:
                        continue
                    kernel_np = _lookup_render_kernel(lookup, x_axis, y_axis, int(tx), int(ty), crop_size)
                    kernel_np = np.flipud(np.fliplr(kernel_np))
                    kernel = torch.from_numpy(kernel_np.copy()).to(torch_device, dtype=tensor.dtype)
                    kernel = kernel.view(1, 1, *kernel.shape).repeat(3, 1, 1, 1)
                    weight_t = torch.from_numpy(node_weight.astype(np.float32, copy=False)).to(
                        torch_device,
                        dtype=tensor.dtype,
                    )
                    weighted = tensor * weight_t.unsqueeze(0).unsqueeze(0)
                    pad_y = kernel.shape[-2] - 1
                    pad_x = kernel.shape[-1] - 1
                    padded = torch.nn.functional.pad(weighted, (pad_x, pad_x, pad_y, pad_y), mode="constant", value=0.0)
                    full = torch.nn.functional.conv2d(padded, kernel, groups=3)
                    output += full[..., half : half + size_y, half : half + size_x]
    result = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def render_scene_fast_compatible(
    rgb: np.ndarray,
    samples: list[RenderPsfSample],
    config: RenderSimulationConfig,
    *,
    backend: str = "torch",
    device: str = "cuda",
) -> np.ndarray:
    """Render with the external-compatible fast backend."""
    backend_key = str(backend).strip().lower()
    if backend_key == "scipy":
        return render_scene_fast_scipy_compatible(rgb, samples, config)
    if backend_key == "torch":
        return render_scene_fast_torch_compatible(rgb, samples, config, device=device)
    raise ValueError(f"Unsupported render-fast backend: {backend}")


def _field_angle_maps(size_y: int, size_x: int, config: RenderSimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    half_fov_x = config.fov_x_deg / 2.0
    half_fov_y = config.fov_y_deg / 2.0
    field_x, field_y = np.meshgrid(
        np.linspace(-half_fov_x, half_fov_x, size_x),
        np.linspace(half_fov_y, -half_fov_y, size_y),
    )
    return field_x.astype(np.float32), field_y.astype(np.float32)


def _reference_crop_size(size_x: int, config: RenderSimulationConfig) -> int:
    scene_pixel_angle = config.fov_x_deg / size_x
    new_pixel_size_mm = config.size_reference_mm / config.crop_size_reference
    psf_pixel_angle = (new_pixel_size_mm / config.f_eye_mm) * (180.0 / np.pi)
    scale_factor = scene_pixel_angle / psf_pixel_angle
    return max(1, int(round(config.crop_size_reference / scale_factor)))


def _linear_axis_weights(values: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float32)
    if len(axis) == 1:
        index = np.zeros_like(values, dtype=np.int16)
        ones = np.ones_like(values, dtype=np.float32)
        zeros = np.zeros_like(values, dtype=np.float32)
        return index, index, ones, zeros
    clipped = np.clip(values.astype(np.float32), axis[0], axis[-1])
    upper = np.searchsorted(axis, clipped, side="right").astype(np.int16)
    upper = np.clip(upper, 1, len(axis) - 1)
    lower = (upper - 1).astype(np.int16)
    denom = axis[upper] - axis[lower]
    t = np.where(denom > 0, (clipped - axis[lower]) / denom, 0.0).astype(np.float32)
    at_min = clipped <= axis[0]
    at_max = clipped >= axis[-1]
    lower = np.where(at_min, 0, lower).astype(np.int16)
    upper = np.where(at_min, 0, upper).astype(np.int16)
    lower = np.where(at_max, len(axis) - 1, lower).astype(np.int16)
    upper = np.where(at_max, len(axis) - 1, upper).astype(np.int16)
    t = np.where(at_min | at_max, 0.0, t).astype(np.float32)
    return lower, upper, (1.0 - t).astype(np.float32), t


def _lookup_render_kernel(
    lookup: dict[tuple[float, float], np.ndarray],
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    tx: int,
    ty: int,
    crop_size: int,
) -> np.ndarray:
    key = (round(float(x_axis[tx]), 6), round(float(y_axis[ty]), 6))
    kernel = cv2.resize(
        np.flipud(np.asarray(lookup[key], dtype=np.float32)),
        (int(crop_size), int(crop_size)),
        interpolation=cv2.INTER_CUBIC,
    )
    return normalize_psf_energy(kernel).astype(np.float32)
