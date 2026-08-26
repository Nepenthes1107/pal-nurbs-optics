from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.signal import fftconvolve

from biot.infra.image_io import write_cv_image


def normalize_display(image: np.ndarray) -> np.ndarray:
    """Return a finite 0..1 display image.

    This helper is for GUI/display artifacts only. It must not be used for PSF
    energy normalization or MTF calculation.
    """

    arr = np.asarray(image, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax <= vmin:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def default_chart_path() -> Path:
    return Path.cwd() / "E1.xlsx"


def load_chart_xlsx(xlsx_path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    """Load and resize a unitless chart image from XLSX for display convolution.

    Inputs:
    - xlsx_path: chart matrix file.
    - target_shape: target image shape [height, width] in pixels.
    Output is a finite 0..1 NumPy image. No GPU/autograd support.
    """

    chart = pd.read_excel(Path(xlsx_path), header=None, engine="openpyxl").values
    chart = normalize_display(chart)
    if chart.shape == target_shape:
        return chart
    width = int(target_shape[1])
    height = int(target_shape[0])
    return cv2.resize(chart, (width, height), interpolation=cv2.INTER_AREA)


def convolve_chart_with_psf(chart: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Convolve a display chart with an energy-normalized PSF.

    Inputs:
    - chart: unitless 2D display image [H, W].
    - psf: 2D energy-normalized PSF [H, W].
    Output is a finite 0..1 display image. No GPU/autograd support.
    """

    psf_arr = np.asarray(psf, dtype=np.float64)
    psf_arr = np.nan_to_num(psf_arr, nan=0.0, posinf=0.0, neginf=0.0)
    psf_arr = np.maximum(psf_arr, 0.0)
    total = float(psf_arr.sum())
    if total <= 0.0:
        raise ValueError("PSF energy is not positive for chart convolution.")
    psf_arr = psf_arr / total
    convolved = fftconvolve(np.asarray(chart, dtype=np.float64), psf_arr, mode="same")
    return normalize_display(convolved)


def save_display_png(image: np.ndarray, output_path: Path, *, invert: bool = False) -> Path:
    """Save a finite 0..255 grayscale display PNG."""

    display = normalize_display(image)
    if invert:
        display = 1.0 - display
    u8 = np.clip(display * 255.0, 0, 255).astype(np.uint8)
    return write_cv_image(output_path, u8)


def _five_degree_ticks(values: np.ndarray) -> np.ndarray:
    low = int(np.ceil(float(values.min()) / 5.0) * 5)
    high = int(np.floor(float(values.max()) / 5.0) * 5)
    ticks = np.arange(low, high + 1, 5, dtype=int)
    if ticks.size == 0:
        return values
    return ticks


def _style_field_axis(ax, field_x_values: np.ndarray, field_y_values: np.ndarray) -> None:
    ax.set_xlabel("field X (Degrees)", fontfamily="Times New Roman", fontsize=18)
    ax.set_ylabel("field Y (Degrees)", fontfamily="Times New Roman", fontsize=18)
    ax.set_xticks(_five_degree_ticks(field_x_values))
    ax.set_yticks(_five_degree_ticks(field_y_values))
    ax.tick_params(direction="in", top=True, right=True, labelsize=14)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _field_extent(field_x_values: np.ndarray, field_y_values: np.ndarray) -> list[float]:
    x_min = float(np.min(field_x_values))
    x_max = float(np.max(field_x_values))
    y_min = float(np.min(field_y_values))
    y_max = float(np.max(field_y_values))
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    return [x_min, x_max, y_min, y_max]


def save_psf_png(
    psf_image: np.ndarray,
    d_delta_mm: float,
    output_path: Path,
    *,
    dpi: int = 160,
) -> Path:
    """Save a single PSF as a PNG with physical axes (um) and a colorbar.

    Args:
        psf_image: Energy-normalized PSF, shape [n_i, n_i].
        d_delta_mm: Image-plane sampling interval in mm/pixel.
        output_path: Destination PNG path.
        dpi: Output resolution.

    Returns:
        Path: ``output_path``.

    Notes:
        Display only. The colorbar is labeled with the raw normalized intensity
        so the figure stays traceable to ``psf_data.npy``; it must not be used
        as a numerical check, and it is never fed to the MTF path.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(psf_image, dtype=np.float64)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # 像元中心坐标 [um]：零点落在 fftshift 的 DC 索引 n//2 上，与 optics.py 的坐标
    # 网格约定一致；imshow 的 extent 给的是外沿，故各向外扩半个像元。
    #
    # 行序：optics.py 末尾的 standardize_psf_orientation 做过一次 flipud，所以保存
    # 数组的行 0 是 +Y（顶部），与 Zemax 导出 xlsx 的行序一致（该 xlsx 与本数组
    # 逐行配对，无需翻转）。因此必须用 origin="upper" 把行 0 画在顶部：用
    # origin="lower" 会让整幅图关于 x 轴翻转。extent 的 top 对应行 0。
    n_row, n_col = arr.shape
    pitch_um = float(d_delta_mm) * 1e3
    half = 0.5 * pitch_um
    x_left = (0 - (n_col // 2)) * pitch_um - half
    x_right = (n_col - 1 - (n_col // 2)) * pitch_um + half
    y_top = ((n_row // 2) - 0) * pitch_um + half
    y_bottom = ((n_row // 2) - (n_row - 1)) * pitch_um - half

    vmax = float(np.nanmax(arr))
    fig = Figure(figsize=(7.2, 6.0), dpi=dpi, tight_layout=True)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        arr,
        extent=[x_left, x_right, y_bottom, y_top],
        origin="upper",
        cmap="jet",
        vmin=0.0,
        vmax=vmax if vmax > 0 else 1.0,
        aspect="equal",
    )
    ax.set_xlabel("X (um)", fontfamily="Times New Roman", fontsize=18)
    ax.set_ylabel("Y (um)", fontfamily="Times New Roman", fontsize=18)
    ax.tick_params(direction="in", top=True, right=True, labelsize=14)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Intensity", fontfamily="Times New Roman", fontsize=16)
    cbar.ax.tick_params(direction="in", labelsize=12)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def save_field_stitch_png(
    image: np.ndarray,
    field_x_values: np.ndarray,
    field_y_values: np.ndarray,
    output_path: Path,
    *,
    kind: str,
    dpi: int = 160,
) -> Path:
    """Save a reference-style field stitch PNG with field axes."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_x_values = np.asarray(field_x_values, dtype=np.float64)
    field_y_values = np.asarray(field_y_values, dtype=np.float64)
    extent = _field_extent(field_x_values, field_y_values)
    fig = Figure(figsize=(8.4, 6.3), dpi=dpi, tight_layout=True)
    ax = fig.add_subplot(111)
    if kind == "chart":
        display = 1.0 - normalize_display(image)
        im = ax.imshow(display, extent=extent, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.ax.invert_yaxis()
    else:
        arr = np.asarray(image, dtype=np.float64)
        vmax = float(np.nanmax(arr))
        im = ax.imshow(arr, extent=extent, origin="lower", cmap="jet", vmin=0.0, vmax=vmax if vmax > 0 else 1.0, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _style_field_axis(ax, field_x_values, field_y_values)
    cbar.ax.tick_params(direction="in", labelsize=12)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def save_mtf_value_grid_png(
    mtf_grid: np.ndarray,
    field_x_values: np.ndarray,
    field_y_values: np.ndarray,
    output_path: Path,
    dpi: int = 160,
) -> Path:
    """Save a MATLAB-style two-number MTF grid PNG."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(mtf_grid, dtype=np.float64)
    rows, cols, channels = arr.shape
    if channels != 2:
        raise ValueError(f"Expected grid shape [rows, cols, 2], got {arr.shape}.")

    fig = Figure(figsize=(max(7.0, cols * 1.25), max(5.0, rows * 0.9)), dpi=dpi, tight_layout=True)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    for col in range(cols + 1):
        ax.plot([col, col], [0, rows], color="black", linewidth=0.8)
    for row in range(rows + 1):
        ax.plot([0, cols], [row, row], color="black", linewidth=0.8)
    for row in range(rows):
        for col in range(cols):
            cx = col + 0.5
            cy = row + 0.5
            sag = arr[row, col, 0]
            tan = arr[row, col, 1]
            ax.text(cx, cy - 0.14, "nan" if not np.isfinite(sag) else f"{sag:.4f}", ha="center", va="center", color="black", fontsize=10)
            ax.text(cx, cy + 0.18, "nan" if not np.isfinite(tan) else f"{tan:.4f}", ha="center", va="center", color="#d20000", fontsize=10)
    field_x_values = np.asarray(field_x_values, dtype=np.float64)
    field_y_values = np.asarray(field_y_values, dtype=np.float64)
    if field_x_values.size == cols:
        ax.set_xticks(np.arange(cols) + 0.5)
        ax.set_xticklabels([f"{x:g}" for x in field_x_values], fontfamily="Times New Roman")
    if field_y_values.size == rows:
        ax.set_yticks(np.arange(rows) + 0.5)
        ax.set_yticklabels([f"{y:g}" for y in field_y_values], fontfamily="Times New Roman")
    ax.set_xlabel("field X (Degrees)", fontfamily="Times New Roman", fontsize=14)
    ax.set_ylabel("field Y (Degrees)", fontfamily="Times New Roman", fontsize=14)
    ax.set_title("Cutoff MTF: black=Sagittal, red=Tangential", fontsize=13)
    ax.tick_params(direction="in", top=True, right=True, labelsize=11)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path
