import numpy as np
import torch.fft
import pandas as pd
import os
from openpyxl.worksheet import _reader as _openpyxl_ws_reader
_openpyxl_ws_reader._cast_number = lambda value: float(value)
import prysm.polynomials
from prysm.polynomials import zernike_nm, zernike_nm_sequence, zernike_nm_der, lstsq, ansi_j_to_nm
from matplotlib.patches import Patch
from scipy.interpolate import RectBivariateSpline
import cv2


# from .basics import *
from basics import *
import pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from scipy.ndimage import zoom
from scipy.special import jv, gammaln, comb, eval_jacobi
from scipy.optimize import least_squares

torch.set_default_dtype(torch.float64)

_PUPIL_REFERENCE_LATERAL_TOLERANCE_MM = 1e-9


def sanitize_and_energy_normalize_psf(psf_image):
    """
    Sanitize PSF numeric values and normalize total energy to 1.
    """
    psf = np.asarray(psf_image, dtype=np.float64)
    psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)
    psf = np.maximum(psf, 0.0)
    total = float(np.sum(psf))
    if (not np.isfinite(total)) or total <= 0.0:
        raise ValueError(f"Invalid PSF total energy for normalization: {total}")
    return psf / total


def compute_dc_normalized_mtf(psf_image):
    """
    Compute DC-normalized MTF from PSF.
    """
    psf_norm = sanitize_and_energy_normalize_psf(psf_image)
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf_norm)))
    mtf = np.abs(otf)
    cy = mtf.shape[0] // 2
    cx = mtf.shape[1] // 2
    dc = float(mtf[cy, cx])
    if (not np.isfinite(dc)) or dc <= 0.0:
        raise ValueError(f"Invalid OTF DC component for normalization: {dc}")
    return mtf / dc


def _real_zernike_nm(n, m, rho, theta):
    """Evaluate the OSA/ANSI real RMS-normalized Zernike mode ``Z_n^m``.

    Coordinates ``rho`` and ``theta`` are dimensionless, with ``rho <= 1``.
    The ANSI convention uses cosine for positive ``m`` and sine for negative
    ``m``; ``norm=True`` makes the modes unit RMS over the unit disk.  This
    CPU NumPy diagnostic does not preserve autograd.
    """
    if n < 0 or abs(m) > n or (n - abs(m)) % 2:
        raise ValueError(f"Invalid real Zernike indices: n={n}, m={m}")

    return zernike_nm(n, m, np.asarray(rho), np.asarray(theta), norm=True)


def _ansi_nm_to_j(n, m):
    """Return the zero-based OSA/ANSI single-term index for ``(n, m)``."""
    return (n * (n + 2) + m) // 2


def _noll_nm_to_j(n, m):
    """Return the one-based Noll index for ``(n, m)``, as Zemax Zernike Standard uses.

    Within a radial order ``n`` the terms run by increasing ``|m|``; ``m = 0``
    takes one slot and each ``|m| > 0`` takes two consecutive slots.  Of that
    pair the even index is the cosine term (``m > 0``) and the odd index is the
    sine term (``m < 0``).  Orders start at ``j0(n) = n(n+1)/2 + 1``.

    The cosine/sine order therefore flips between adjacent orders: ``(3,-1)``
    is 7 while ``(3,1)`` is 8, but ``(4,2)`` is 12 while ``(4,-2)`` is 13.  Do
    not "tidy" this into a uniform order -- the parity rule is what makes it
    match Zemax.
    """
    if n < 0 or abs(m) > n or (n - abs(m)) % 2:
        raise ValueError(f"Invalid real Zernike indices: n={n}, m={m}")

    order_start = n * (n + 1) // 2 + 1
    abs_m = abs(m)
    if n % 2 == 0:
        offset = 0 if abs_m == 0 else 1 + 2 * (abs_m // 2 - 1)
    else:
        offset = 2 * ((abs_m - 1) // 2)
    j = order_start + offset
    if abs_m == 0:
        return j
    # The pair is (j, j+1); pick whichever parity matches the m sign.
    return j if (j % 2 == 0) == (m > 0) else j + 1


def noll_j_to_nm(j):
    """Return ``(n, m)`` for a one-based Noll index ``j``. Inverse of :func:`_noll_nm_to_j`."""
    if isinstance(j, bool) or int(j) != j or int(j) < 1:
        raise ValueError(f"Noll index must be a positive integer, got {j!r}")
    j = int(j)

    n = 0
    while (n + 1) * (n + 2) // 2 + 1 <= j:
        n += 1
    offset = j - (n * (n + 1) // 2 + 1)
    if n % 2 == 0:
        abs_m = 0 if offset == 0 else 2 * ((offset + 1) // 2)
    else:
        abs_m = 2 * (offset // 2) + 1
    if abs_m == 0:
        return n, 0
    return n, abs_m if j % 2 == 0 else -abs_m


def _zemax_term_label(j):
    """Return the Zemax report label for a Noll index, e.g. ``Z   4`` / ``Z  21``."""
    return f"Z {int(j):3d}"


def _real_zernike_term_name(n, m):
    """Return a compact signed-m name for a real ``(n, m)`` mode, e.g. ``Z(2, -2)``."""
    return f"Z({n}, {m})"


def _solve_full_rank_least_squares(design, values):
    """Solve a small full-rank least-squares system by reorthogonalized QR.

    Zernike fitting has few columns (21 for the default fifth order), while
    some deployed NumPy/MKL builds abort inside ``lstsq``.  This explicit QR
    algorithm avoids that external LAPACK path and rejects rank-deficient
    designs instead of silently regularizing them.
    """
    rows, columns = design.shape
    q = np.empty((rows, columns), dtype=np.float64)
    upper = np.zeros((columns, columns), dtype=np.float64)
    scale = max(float(np.max(np.abs(design))), 1.0)
    rank_tolerance = np.finfo(np.float64).eps * max(rows, columns) * scale

    for column in range(columns):
        vector = design[:, column].copy()
        for _ in range(2):
            for previous in range(column):
                projection = float(np.sum(q[:, previous] * vector))
                upper[previous, column] += projection
                vector -= projection * q[:, previous]
        diagonal = float(np.sqrt(np.sum(vector * vector)))
        if not np.isfinite(diagonal) or diagonal <= rank_tolerance:
            raise ValueError(
                f"Wavefront Zernike design matrix is rank deficient at mode column {column}"
            )
        upper[column, column] = diagonal
        q[:, column] = vector / diagonal

    rhs = np.asarray([np.sum(q[:, column] * values) for column in range(columns)], dtype=np.float64)
    coefficients = np.empty(columns, dtype=np.float64)
    for row in range(columns - 1, -1, -1):
        trailing = np.sum(upper[row, row + 1 :] * coefficients[row + 1 :])
        coefficients[row] = (rhs[row] - trailing) / upper[row, row]
    return coefficients


def fit_wavefront_zernike(opd_mm, x_norm, y_norm, pupil_mask, wavelength_mm, n_max=5):
    """Fit continuous wavefront OPD to real Zernike modes over a unit pupil.

    Args:
        opd_mm: Continuous scalar optical-path difference, shape ``[H, W]`` in
            mm. It must be the unwrapped traced phase converted before forming
            ``exp(1j * phase)``.
        x_norm, y_norm: Normalized pupil-coordinate grids, shape ``[H, W]``.
        pupil_mask: Boolean valid-pupil mask, shape ``[H, W]``.
        wavelength_mm: Positive wavelength in mm.
        n_max: Non-negative highest radial Zernike order.

    Returns:
        ``(coefficients, metrics)`` where coefficients are dictionaries with
        OPD coefficients in mm, um, and waves. This CPU NumPy diagnostic does
        not preserve autograd and never changes the supplied wavefront.
    """
    if isinstance(n_max, bool) or int(n_max) != n_max or int(n_max) < 0:
        raise ValueError(f"zernike_n_max must be a non-negative integer, got {n_max!r}")
    n_max = int(n_max)
    wavelength_mm = float(wavelength_mm)
    if not np.isfinite(wavelength_mm) or wavelength_mm <= 0.0:
        raise ValueError(f"wavelength_mm must be finite and positive, got {wavelength_mm!r}")

    opd = np.asarray(opd_mm, dtype=np.float64)
    x = np.asarray(x_norm, dtype=np.float64)
    y = np.asarray(y_norm, dtype=np.float64)
    mask = np.asarray(pupil_mask, dtype=bool)
    if opd.ndim != 2 or x.shape != opd.shape or y.shape != opd.shape or mask.shape != opd.shape:
        raise ValueError("OPD, normalized pupil grids, and pupil mask must be same-shape 2D arrays")
    if not np.isfinite(opd).all() or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Wavefront OPD and normalized pupil coordinates must be finite")

    rho = np.hypot(x, y)
    valid = mask & (rho <= 1.0 + 1e-12)
    valid_count = int(np.count_nonzero(valid))
    terms = [(n, m) for n in range(n_max + 1) for m in range(-n, n + 1, 2)]
    if valid_count == 0:
        raise ValueError("Cannot fit wavefront Zernike coefficients: pupil mask contains no valid points")
    if valid_count < len(terms):
        raise ValueError(
            f"Cannot fit {len(terms)} Zernike modes with only {valid_count} valid pupil samples"
        )

    theta = np.arctan2(y[valid], x[valid])
    rho_valid = rho[valid]
    design = np.column_stack([_real_zernike_nm(n, m, rho_valid, theta) for n, m in terms])
    coefficients_mm = _solve_full_rank_least_squares(design, opd[valid])

    fitted_opd_mm = np.sum(design * coefficients_mm[np.newaxis, :], axis=1)
    residual_mm = opd[valid] - fitted_opd_mm
    coefficients = []
    for (n, m), coefficient in zip(terms, coefficients_mm):
        waves = float(coefficient / wavelength_mm)
        noll_j = int(_noll_nm_to_j(n, m))
        coefficients.append(
            {
                # Existing six columns keep their names and order: downstream
                # readers index them positionally.
                "ansi_j": int(_ansi_nm_to_j(n, m)),
                "n": int(n),
                "m": int(m),
                "term_name": _real_zernike_term_name(n, m),
                "coefficient_opd_um": float(coefficient * 1e3),
                "coefficient_waves": waves,
                # Task 6 diagnostic columns. The sign flip is output-only; the
                # PSF/MTF phase path is untouched.
                "noll_j": noll_j,
                "coefficient_waves_zemax_convention": -waves,
                "zemax_term_label": _zemax_term_label(noll_j),
                "is_piston": bool(n == 0 and m == 0),
            }
        )
    metrics = {
        "zernike_n_max": n_max,
        "zernike_mode_count": len(terms),
        "zernike_valid_pupil_points": valid_count,
        "zernike_wavelength_mm": wavelength_mm,
        "zernike_wavelength_nm": wavelength_mm * 1e6,
        "zernike_wavefront_opd_rms_mm": float(np.sqrt(np.mean(opd[valid] ** 2))),
        "zernike_residual_rms_mm": float(np.sqrt(np.mean(residual_mm**2))),
        "zernike_residual_rms_um": float(np.sqrt(np.mean(residual_mm**2)) * 1e3),
        "zernike_residual_rms_waves": float(np.sqrt(np.mean(residual_mm**2)) / wavelength_mm),
        "zernike_basis": "osa_ansi_real_zernike_nm_rms_normalized",
        "zernike_indexing": "ansi_j_zero_based",
        "zernike_low_order_policy": "retain_piston_tip_tilt_defocus",
        "zernike_sign_convention": "biot_native",
        "zernike_zemax_sign_relation": "zemax = -biot",
        "zernike_zemax_indexing": "noll_j_one_based",
        "zernike_piston_reference": "chief_reference_ray",
    }
    return coefficients, metrics


def standardize_psf_orientation(psf_image):
    """Standardize PSF row order across all entry points.

    Reverses the row axis so that output row 0 is +Y, matching the row order of
    the Zemax FFT PSF xlsx exports — those pair with the saved array row-by-row,
    no further flip. (An earlier docstring said "horizontal flip", which
    contradicted the code: reversing rows is a *vertical* flip.)

    The reversal is taken **about the fftshift DC index** ``n // 2``, i.e.
    ``r -> (n - r) mod n``, not ``np.flipud``'s ``r -> n - 1 - r``. On an
    even-length axis those differ by exactly one pixel, so a bare ``flipud``
    moved every PSF's row symmetry axis from 256 to 255 and shifted the image
    one pixel off the DC index that ``fft_psf_i``'s coordinate grid puts zero
    on. The column axis, which is never flipped, is the control: it lands on
    256.0 to machine precision.

    Consumers must respect that convention: an ``imshow`` of the returned array
    needs ``origin="upper"``, and ``cv2.imwrite`` already puts row 0 on top.
    """
    arr = np.asarray(psf_image)
    ## r -> (n - r) mod n 等价于先 flipud（r -> n-1-r）再整体下移 1 行。
    return np.roll(np.flipud(arr), 1, axis=0).copy()


def _complex_circular_mean_angle(values, eps=1e-12):
    """
    Return the circular mean angle of complex values in radians.

    Inputs:
        values: Complex torch tensor with arbitrary shape. Values represent
            unitless phasors such as exp(1j * phase).
        eps: Unitless magnitude threshold for invalid phasors.

    Output:
        Scalar torch tensor in radians on the same device as values.

    GPU/autograd:
        Supports CPU/GPU tensors. The returned angle uses torch operations, but
        the circular mean is intended for diagnostic phase-ramp removal rather
        than gradient-sensitive optimization.
    """
    if values.numel() == 0:
        return torch.zeros((), dtype=torch.float64, device=values.device)

    magnitude = torch.abs(values)
    valid = magnitude > eps
    if not bool(valid.any().detach().cpu().item()):
        return torch.zeros((), dtype=torch.float64, device=values.device)

    unit_values = values[valid] / magnitude[valid]
    mean_value = torch.mean(unit_values)
    if float(torch.abs(mean_value).detach().cpu()) <= eps:
        return torch.zeros((), dtype=torch.float64, device=values.device)
    return torch.angle(mean_value)


def _estimate_complex_pupil_tilt(pupil, x_grid, y_grid, mask, eps=1e-12):
    """
    Estimate linear pupil phase ramp from adjacent complex phase differences.

    Inputs:
        pupil: Complex torch tensor, shape (H, W), pupil field exp(1j*phase).
        x_grid, y_grid: Torch tensors, shape (H, W), normalized pupil
            coordinates in [-1, 1], unitless.
        mask: Boolean torch tensor, shape (H, W), valid pupil samples.
        eps: Unitless magnitude threshold.

    Outputs:
        dict of scalar torch tensors:
            tilt_x_rad_per_norm, tilt_y_rad_per_norm: phase slopes in
                radian per normalized pupil coordinate.
            phase_step_x_rad, phase_step_y_rad: adjacent-sample wrapped phase
                increments in radians.

    GPU/autograd:
        Supports CPU/GPU tensors. The estimate is robust to 2*pi wrapped
        absolute phase because it uses complex adjacent phase differences.
    """
    device = pupil.device
    zero = torch.zeros((), dtype=torch.float64, device=device)

    valid_x = (
        mask[:, 1:]
        & mask[:, :-1]
        & (torch.abs(pupil[:, 1:]) > eps)
        & (torch.abs(pupil[:, :-1]) > eps)
    )
    valid_y = (
        mask[1:, :]
        & mask[:-1, :]
        & (torch.abs(pupil[1:, :]) > eps)
        & (torch.abs(pupil[:-1, :]) > eps)
    )

    if bool(valid_x.any().detach().cpu().item()):
        phase_step_x = _complex_circular_mean_angle(
            pupil[:, 1:][valid_x] * torch.conj(pupil[:, :-1][valid_x]),
            eps=eps,
        )
        dx = torch.mean((x_grid[:, 1:] - x_grid[:, :-1])[valid_x])
        tilt_x = phase_step_x / dx
    else:
        phase_step_x = zero
        tilt_x = zero

    if bool(valid_y.any().detach().cpu().item()):
        phase_step_y = _complex_circular_mean_angle(
            pupil[1:, :][valid_y] * torch.conj(pupil[:-1, :][valid_y]),
            eps=eps,
        )
        dy = torch.mean((y_grid[1:, :] - y_grid[:-1, :])[valid_y])
        tilt_y = phase_step_y / dy
    else:
        phase_step_y = zero
        tilt_y = zero

    return {
        "tilt_x_rad_per_norm": tilt_x,
        "tilt_y_rad_per_norm": tilt_y,
        "phase_step_x_rad": phase_step_x,
        "phase_step_y_rad": phase_step_y,
    }


def summarize_complex_pupil_phase(pupil, x_grid, y_grid, mask, prefix, eps=1e-12):
    """
    Summarize piston and linear x/y tilt in a complex pupil field.

    Inputs:
        pupil: Complex torch tensor, shape (H, W), pupil field exp(1j*phase).
        x_grid, y_grid: Torch tensors, shape (H, W), normalized pupil
            coordinates in [-1, 1], unitless.
        mask: Boolean torch tensor, shape (H, W), valid pupil samples.
        prefix: Prefix for returned metric names.
        eps: Unitless magnitude threshold for valid phasors.

    Output:
        dict with detached Python values. Tilt units are radian per normalized
        pupil coordinate; phase-step and piston units are radian.

    GPU/autograd:
        Supports CPU/GPU tensors. Metrics are detached Python values for
        logging and do not preserve autograd.
    """
    if pupil.shape != x_grid.shape or pupil.shape != y_grid.shape or pupil.shape != mask.shape:
        raise ValueError("pupil, x_grid, y_grid, and mask must have identical shapes")

    mask = mask.bool()
    estimate = _estimate_complex_pupil_tilt(pupil, x_grid, y_grid, mask, eps=eps)
    piston = _complex_circular_mean_angle(pupil[mask], eps=eps)

    def scalar(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    return {
        f"{prefix}_piston_rad": scalar(piston),
        f"{prefix}_tilt_x_rad_per_norm": scalar(estimate["tilt_x_rad_per_norm"]),
        f"{prefix}_tilt_y_rad_per_norm": scalar(estimate["tilt_y_rad_per_norm"]),
        f"{prefix}_phase_step_x_rad": scalar(estimate["phase_step_x_rad"]),
        f"{prefix}_phase_step_y_rad": scalar(estimate["phase_step_y_rad"]),
        f"{prefix}_mask_points": int(mask.sum().detach().cpu().item()),
    }


# 关于raise的使用:
# 如果该错误，是一定不允许发生的,发生了则说明代码逻辑存在问题，则应该使用raise ValueError()
# 如果该错误，是可能发生的，不是代码本身问题，比如优化过程中，某个结构下光线无法追迹，这是可能发生的,此时应该使用raise Exception()
class Lensdata():
    """
    The origin of the Lensgroup, which is a collection of multiple optical surfaces, is located at "origin".
    The Lensgroup can rotate freely around the x/y axes, and the rotation angles are defined as "theta_x", "theta_y", and "theta_z" (in degrees).
    
    In the Lensgroup's coordinate system, which is the object frame coordinate system, surfaces are arranged starting from "z = 0".
    There is a small 3D origin shift, called "shift", between the center of the surface (0,0,0) and the mount's origin.
    The sum of the shift and the origin is equal to the Lensgroup's origin.
    
    There are two configurations for ray tracing: forward and backward.
    - In the forward mode, rays begin at the surface with "d = 0" and propagate along the +z axis, e.g. from scene to image plane.
    - In the backward mode, rays begin at the surface with "d = d_max" and propagate along the -z axis, e.g. from image plane to scene.
    """

    def __init__(self, origin=np.zeros(3), shift=np.zeros(3), theta_x=0., theta_y=0., theta_z=0.,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.origin = torch.Tensor(origin).to(device)
        self.shift = torch.Tensor(shift).to(device)
        self.theta_x = torch.Tensor(np.asarray(theta_x)).to(device)
        self.theta_y = torch.Tensor(np.asarray(theta_y)).to(device)
        self.theta_z = torch.Tensor(np.asarray(theta_z)).to(device)
        self.device = device

        # lens surfaces
        self.surfaces = []
        self.materials = []
        self.obj = None  # 物面
        self.img = None  # 像面
        self.aperture_ind = None  # 光阑位置


        # system aperture
        self.enp_dia = None  # 入瞳直径
        self.NA_obj = None  # 物方数值孔径
        self.aperture = None  # 光阑浮动
        self.aperture_mode = None  # 系统孔径模式
        self.enp_pos = None  # 入瞳位置
        self.exp_pos = None  # 出瞳位置
        self.EFL = None  # 有效焦距
        self.image_distance = None  # 像距
        self.Fnum = None  # 工作F数

        # field of view
        self.view_type = None  # ‘angle’,'height'
        self.FOV = None  # 最大视场角[deg]
        self.angle = None  # 最大视场角[rad]

        # wavelength
        self.wavelengths = None
        self.wavelengths_center = None

        # ray_aimming
        self.aimming = False
        self.last_pupil_tilt_metrics = {}

        # sensor
        self.pixel_size = 6.45  # um
        self.filme_size = [640, 480]  # [pixel]

        Endpoint.__init__(self, self._compute_transformation(), device)

        self.mts_prepared = False
    
    def showimfft(self, image_path, psf_data,target_size=(130,130)):
        """
        加载视标图像和PSF数据，通过FFT进行卷积，并显示结果。
        
        Args:
            image_path (str): 视标图像的 .xlsx 文件路径。
            psf_data (np.ndarray): PSF 数据，一个2D numpy数组。
        """
        # --- 1. 加载数据 ---
        # 从 .xlsx 文件加载视标图像
        df_image = pd.read_excel(image_path, header=None, engine='openpyxl')
        image_np = df_image.to_numpy().astype(np.float32)

        # 直接使用传入的PSF数据
        psf_np = psf_data.astype(np.float32)
        psf_np = np.flipud(psf_np)

        # --- 2. 将PSF和视标图像尺寸调整为相同大小 ---
        psf_resized = cv2.resize(psf_np, target_size, interpolation=cv2.INTER_AREA)
        image_resized = cv2.resize(image_np, target_size, interpolation=cv2.INTER_AREA)

        # 将numpy数组转换为PyTorch张量
        image_tensor = torch.from_numpy(image_resized).float().to(self.device)
        psf_tensor = torch.from_numpy(psf_resized).float().to(self.device)

        # --- 3. 卷积 (使用FFT) ---
        # 对图像和PSF进行傅里叶变换
        fft_image = torch.fft.fft2(image_tensor)
        fft_psf = torch.fft.fft2(psf_tensor)

        # 在频域中相乘
        result_fft = fft_image * fft_psf

        # 反变换回空间域，将结果移位，使零频分量位于中心，并取实部
        convolved_image_tensor = torch.fft.fftshift(torch.fft.ifft2(result_fft).real)

        # 将最终结果转回numpy以便显示
        convolved_image = convolved_image_tensor.cpu().numpy()

        # --- 4. 显示结果 ---
        plt.figure(figsize=(12, 6))

        # 显示调整大小后的原始视标
        plt.subplot(1, 3, 1)
        plt.imshow(image_resized, cmap='gray')
        plt.title(f'Resized Optotype ({target_size[0]}x{target_size[1]})')
        plt.axis('off')

        # 显示调整大小后的 PSF
        plt.subplot(1, 3, 2)
        plt.imshow(psf_resized, cmap='gray')
        plt.title(f'Resized PSF ({target_size[0]}x{target_size[1]})')
        plt.axis('off')

        # 显示卷积后的图像
        plt.subplot(1, 3, 3)
        plt.imshow(convolved_image, cmap='gray')
        plt.title('Convolved Image (FFT)')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


    def initial_check(self):
        """
        检查系统参数设置是否正常，并计算定义部分系统参数。
        注意：initial_check该函数在最初设置完系统参数后仅且使用一次。在后续，其他系统参数会被定义，此时不改变初始系统设置
        比如系统孔径中，初始设置是入瞳直径，但后续光阑孔径，像方F数被计算定义，但此时孔径类型依然不变
        """
        # check system aperture
        if self.enp_dia is not None:
            self.aperture_mode = 'enp'  # 入瞳直径
        if self.NA_obj is not None:
            self.aperture_mode = 'NA_obj'  # 物方数值孔径
        if self.aperture is not None:
            self.aperture_mode = 'aperture'  # 光阑半径
        if self.aperture_mode is None:
            raise Exception('system aperture have not be determined')

        self.enp_pos = self.find_enp()
        self.exp_pos = self.find_exp()
        self.enp_dia = self.cal_enp(self.aperture_mode, self.enp_pos)
        self.aperture = self.cal_aperture(self.enp_dia, self.enp_pos)
        print('check entrance pupil position:{} [mm]'.format(self.enp_pos))
        print('check exit pupil position:{} [mm]'.format(self.exp_pos))
        print('check entrance pupil diameter:{} [mm]'.format(self.enp_dia))
        print('check aperture semi-diameter:{} [mm]'.format(self.aperture))
        print('aper_index:{}'.format(self.aperture_ind))


        # check view of field
        if self.view_type is None or self.FOV is None:
            raise Exception('field of view miss some parameters')
        self.FOV = torch.tensor(self.FOV)
        self.angle = torch.deg2rad(self.FOV)
        # check wavelengths
        if self.wavelengths is None or self.wavelengths_center is None:
            raise Exception('wavelengths have not be determined')

        self.EFL, self.image_distance = self.cal_EFL(self.enp_dia)
        print('check EFL :{} [mm]'.format((self.EFL).item()))
        print('check image distance:{} [mm]'.format(self.image_distance))
        print('Whether the ray aiming function is enabled:', self.aimming)
        print(self.aperture)
        ray = self.find_chief_ray(Hx=torch.zeros([1, 1]), Hy=torch.ones([1, 1]), wavelength=None)
        p, _ = self.trace_eyesensor(ray)
        print('核对中心波长最大视场主光线追迹，x:{:.10f}, y:{:.10f}'.format(p[..., 0].item(), p[..., 1].item()))

        # 初步检查时，计算孔径

    def iter_update(self, parameters_name=None):
        """
        当镜头结构参数变换后，更新系统参数。
        比如与系统孔径相关的参数，以及其他通过傍轴光线追迹获得的系统参数
        输入：
            parameters_name:一个字符串列表，其中是额外需要更新的参数名。默认为无额外需要更新的参数
        """
        # JNS：暂且把能更新的都更新了，或许看情况做出修改
        with torch.no_grad():
            self.enp_pos = self.find_enp()
            self.enp_dia = self.cal_enp(self.aperture_mode, self.enp_pos)
            self.aperture = self.cal_aperture(self.enp_dia, self.enp_pos)
            self.EFL, self.image_distance = self.cal_EFL(self.enp_dia)



        # 更新孔径，半直径的快速计算，但是可能不是特别需要，先空着。
        # 理论上这里更新完孔径后，之后的追迹过程中，孔径都是固定的

    # ------------------------------------------------------------------------------------
    # lens editor 
    # ------------------------------------------------------------------------------------

    def load_file(self, filename: pathlib.Path, extension='.xlsx'):
        if extension == '.txt':
            self.surfaces, self.obj, self.img, self.materials, self.aperture_ind = self.read_lensfile_txt(str(filename))
        elif extension == '.xlsx':
            self.surfaces, self.obj, self.img, self.materials, self.aperture_ind = self.read_lensfile_xlsx(
                str(filename), device=self.device)
        else:
            raise ValueError('无法读取该类型文件')
        self._sync()

    def load(self, surfaces: list, materials: list):
        return NotImplementedError()

    def _sync(self):
        for i in range(len(self.surfaces)):
            self.surfaces[i].to(self.device)

    def transform_ray(self, ray, _x, _y, _z, inverse=False):
        """按 CoordinateBreak 的 tilt 旋转光线的局部坐标系。

        Args:
            ray: 待变换的 Ray。
            _x, _y, _z: 该 CB 面的 tilt_x / tilt_y / tilt_z，单位 degree。
            inverse: False 时施加正向旋转 ``R = Rx @ Ry @ Rz``；True 时施加它的
                严格逆变换 ``R.T``。

        Notes:
            旋转矩阵不可交换，``Rx(-x) @ Ry(-y) @ Rz(-z)`` **不是**
            ``Rx(x) @ Ry(y) @ Rz(z)`` 的逆（次序也要反转）。反向追迹必须走
            ``inverse=True``，用正向角构造同一个 R 再取转置，才对任意
            tilt 组合都严格闭合。历史实现在反向追迹里传 ``(-tilt_x, -tilt_y,
            +tilt_z)`` 而不反转次序：单一 tilt 非零时恰好等价，tilt_x 与
            tilt_y 同时非零时闭合误差可达 mm 量级（2026-08-01 修复）。
        """
        ## 构建旋转矩阵。Zemax CoordinateBreak 的 order=0 语义是"先绕 x、再绕 y、
        ## 最后绕 z"的内旋链，作为被动变换施加到向量上就是 Rz @ Ry @ Rx。
        Rx = rodrigues_rotation_matrix(torch.Tensor([1, 0, 0]).to(self.device), torch.deg2rad(self.theta_x + _x))
        Ry = rodrigues_rotation_matrix(torch.Tensor([0, 1, 0]).to(self.device), torch.deg2rad(self.theta_y + _y))
        Rz = rodrigues_rotation_matrix(torch.Tensor([0, 0, 1]).to(self.device), torch.deg2rad(self.theta_z + _z))
        R = Rz @ Ry @ Rx
        if inverse:
            ## 正交矩阵的逆等于转置；纯重排，不引入新的浮点运算。
            R = R.transpose(-1, -2)
        if torch.is_tensor(R):
            self.R = R
        else:
            self.R = torch.Tensor(R)
        o = ray.o
        d = ray.d
        o = torch.squeeze(self.R @ o[..., None]) #旋转后的新坐标向量[N,3]
        if len(o.shape) == 2:
            o = torch.unsqueeze(o, dim=1)
        if len(o.shape) == 1:
            o = torch.unsqueeze(o, dim=0)
            o = torch.unsqueeze(o, dim=0)
        d = torch.squeeze(self.R @ d[..., None]) #旋转后的新方向[N,3]
        if len(d.shape) == 2:
            d = torch.unsqueeze(d, dim=1)
        if len(d.shape) == 1:
            d = torch.unsqueeze(d, dim=0)
            d = torch.unsqueeze(d, dim=0)
        if o.is_cuda:
            return Ray(o, d, ray.wavelength, ray.weight, ray.phase, device=torch.device('cuda'), )
        else:
            return Ray(o, d, ray.wavelength, ray.weight, ray.phase)

    def update(self, _x=0.0, _y=0.0):
        self.to_world = self._compute_transformation(_x, _y)
        self.to_object = self.to_world.inverse()
        # 旋转后更新系统参数
        self.update_system_params()

    def _compute_transformation(self, _x=0.0, _y=0.0, _z=0.0):
        """
        计算正确的旋转变换矩阵
        关键：使用光学中心作为旋转中心，确保旋转后主光线仍指向像面中心
        """
        # 1. 计算旋转中心（光学系统的几何中心）
        rotation_center = self._calculate_optical_center()
        
        # 2. 构建旋转矩阵 (Z -> Y -> X 顺序)
        Rx = rodrigues_rotation_matrix(torch.Tensor([1, 0, 0]).to(self.device), torch.deg2rad(self.theta_x + _x))
        Ry = rodrigues_rotation_matrix(torch.Tensor([0, 1, 0]).to(self.device), torch.deg2rad(self.theta_y + _y))
        Rz = rodrigues_rotation_matrix(torch.Tensor([0, 0, 1]).to(self.device), torch.deg2rad(self.theta_z + _z))
        
        # 组合旋转矩阵
        R = Rx @ Ry @ Rz
        
        # 3. 实现"平移-旋转-平移"变换
        # 先平移到旋转中心，然后旋转，最后平移回来
        # T = T(rotation_center) * R * T(-rotation_center)
        t = rotation_center - R @ rotation_center + self.origin + self.shift
        
        return Transformation(R, t)

    def _find_aperture(self):
        # JNS:觉得不合适所以删了，可能以后还需要另写，源码中是用来确定光阑的位置
        # 目前我在self.read_lensfile时,也会输出光阑面的索引
        return NotImplementedError()

    def _calculate_optical_center(self):
        """
        计算光学系统的真实几何中心，用作旋转中心
        关键是要找到系统的光学中心，而不是简单的几何中心
        """
        try:
            # 方法1：使用像面位置作为参考计算旋转中心
            # 对于光学系统，旋转中心应该在像面附近，这样旋转不会影响像面上的成像位置
            if hasattr(self, 'surfaces') and len(self.surfaces) > 0:
                # 计算系统的总长度
                total_thickness = 0
                for surface in self.surfaces[:-1]:  # 排除最后一个像面
                    if hasattr(surface, 'thickness'):
                        total_thickness += surface.thickness
                
                # 旋转中心设置在系统中点偏向像面的位置
                # 这样可以最小化旋转对像面成像的影响
                z_center = total_thickness * 0.7  # 偏向像面70%的位置
                center = torch.tensor([0.0, 0.0, z_center], device=self.device)
            else:
                # 默认使用原点
                center = torch.zeros(3, device=self.device)
            
            return center
        except Exception:
            # 出错时返回原点
            return torch.zeros(3, device=self.device)

    def update_system_params(self):
        """
        旋转后重新计算系统关键参数
        包括入瞳位置、EFL等参数的更新
        """
        try:
            # 重新计算入瞳位置
            if hasattr(self, 'find_enp'):
                self.enp_pos = self.find_enp()
            
            # 重新计算有效焦距
            if hasattr(self, 'cal_EFL') and hasattr(self, 'enp_dia'):
                self.EFL = self.cal_EFL(self.enp_dia)
                
        except Exception as e:
            print(f"Warning: Failed to update system parameters: {e}")

    @staticmethod
    def _resolve_gridsag_path(sag_file_path, lens_file_path):
        """
        Resolve GridSag source path with fallbacks.

        Resolution order:
        1) original path in Excel
        2) path relative to Excel file directory
        3) basename under project root (directory containing optics.py)
        """
        raw_path = '' if sag_file_path is None else str(sag_file_path).strip()
        if not raw_path:
            raise FileNotFoundError(
                "GridSag sag file path is empty. Please provide a valid XLSX path in lens Excel."
            )

        lens_dir = pathlib.Path(lens_file_path).resolve().parent
        project_root = pathlib.Path(__file__).resolve().parent
        base_name = pathlib.Path(raw_path).name

        candidate_paths = [
            pathlib.Path(raw_path),
            lens_dir / raw_path,
            lens_dir / base_name,
            project_root / base_name,
        ]

        checked = []
        seen = set()
        for candidate in candidate_paths:
            try:
                resolved = candidate.expanduser()
                if not resolved.is_absolute():
                    resolved = (pathlib.Path.cwd() / resolved).resolve()
                else:
                    resolved = resolved.resolve()
            except Exception:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            checked.append(key)
            if pathlib.Path(resolved).exists():
                return key

        checked_lines = '\n'.join(f"  - {p}" for p in checked)
        raise FileNotFoundError(
            f"GridSag sag file not found.\n"
            f"Original value: {raw_path}\n"
            f"Checked candidates:\n{checked_lines}"
        )

    @staticmethod
    def read_lensfile_xlsx(filename, device=None):
        surfaces = []
        materials = []
        df = pd.read_excel(filename, header=None, keep_default_na=False, engine='openpyxl')
        position = 0.
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i in df.index:
            if i < 2:  # first two lines are comments; ignore them
                continue
            else:
                sl = df.iloc[i].tolist()
                while sl and (sl[-1] is None or (isinstance(sl[-1], str) and not sl[-1].strip())):
                    sl.pop()
                excel_row = int(i) + 1
                if len(sl) < 5:
                    raise ValueError(f"Excel row {excel_row}: expected type, thickness, roc, semi_diameter, material")
                surface_type = str(sl[0]).strip()
                th = sl[1]
                raw_roc = sl[2]
                raw_semi_dia = sl[3]
                material_name = str(sl[4]).strip()
                if not surface_type or not material_name:
                    raise ValueError(f"Excel row {excel_row}: missing required type or material")
                if raw_semi_dia is None or (isinstance(raw_semi_dia, str) and not raw_semi_dia.strip()):
                    if surface_type in {'O', 'CB'}:
                        semi_dia = float('inf')
                    else:
                        raise ValueError(f"Excel row {excel_row}: missing required semi_diameter")
                else:
                    semi_dia = float(raw_semi_dia)
                # Excel 中写为 Infinity 的半口径对任何面型都表示无孔径约束。
                # pandas(engine='openpyxl') 配合 _cast_number 补丁会把
                # '<v>Infinity</v>' 解析为 float('inf')，这里统一处理。
                if not np.isfinite(semi_dia):
                    semi_dia = float('inf')
                if isinstance(raw_roc, str) and raw_roc.strip().lower() in {'infinity', 'inf'}:
                    roc = 0.0
                else:
                    try:
                        radius = float(raw_roc)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Excel row {excel_row}: invalid roc value {raw_roc!r}") from exc
                    if not np.isfinite(radius):
                        raise ValueError(f"Excel row {excel_row}: roc must be finite or Infinity, got {raw_roc!r}")
                    roc = 0.0 if radius == 0.0 else 1.0 / radius
                if surface_type == 'GRIN3':
                    raise ValueError(
                        f"Excel row {excel_row}: GRIN3 is the internal surface type; "
                        "the prescription keyword must be GRAD3"
                    )
                if surface_type == 'GRAD3' and material_name.lower() != 'grada':
                    raise ValueError(
                        f"Excel row {excel_row}: GRAD3 requires the supported gradient "
                        f"material 'grada', got {material_name!r}"
                    )
                materials.append(Material(material_name))
                # 无穷远物距在本代码库统一用 thickness == 0 标记。
                # Excel 里可能写成字符串 'Infinity'，也可能被读成浮点 inf，两者都要归一化，
                # 否则 inf 会流进 t_back = thickness / d_z，把物点推到 ±inf 并产生 NaN 方向。
                if isinstance(th, str) and th.strip().lower() in ("infinity", "inf"):
                    th = 0
                else:
                    th = float(th)
                    if not np.isfinite(th):
                        th = 0.0
                    else:
                        position += th
                if surface_type == 'O':  # object
                    position = 0.  # 全局坐标系从第一透镜面开始，这里不计
                    obj = Asphere(semi_dia, position, th, c=roc, device=device)
                elif surface_type == 'X':  # XY-polynomial
                    coeff_list = []
                    for ac in range(5, len(sl)):
                        if ac == 5:
                            conic = float(sl[5])
                        elif ac == 6:
                            order = int(sl[6])
                        elif ac == 7:
                            # 归一化半径是后面才引入的
                            # 而之前的工作没有总结和更新，因此这里用if语句使之前的工作也能运行
                            ## 不懂
                            if int(sl[7]) % 10 == 0:
                                norm_radiu = int(sl[7])
                            else:
                                coeff_list.append(float(sl[ac]))
                                norm_radiu = 1
                        else:
                            coeff_list.append(float(sl[ac]))
                    surfaces.append(
                        XYPolynomial(semi_dia, position - th, th, c=roc, k=conic, coeff_list=coeff_list, J=order,
                                     norm_radiu=norm_radiu, device=device)
                    )
                    # JNS: 现在是无对称，记得根据需要修改
                elif surface_type == 'P':  # Phase_Plate
                    coeff_list = []
                    for ac in range(5, len(sl)):
                        if ac == 5:
                            conic = float(sl[5])
                        elif ac == 6:
                            order = int(sl[6])
                        elif ac == 7:
                            # 归一化半径是后面才引入的
                            # 而之前的工作没有总结和更新，因此这里用if语句使之前的工作也能运行
                            if int(sl[7]) % 10 == 0:
                                norm_radiu = int(sl[7])
                            else:
                                coeff_list.append(float(sl[ac]))
                                norm_radiu = 1
                        else:
                            coeff_list.append(float(sl[ac]))
                    surfaces.append(
                        Phase_Plate(semi_dia, position - th, th, c=roc, k=conic, coeff_list=coeff_list, J=order,
                                    norm_radiu=norm_radiu, device=device)
                    )
                    # JNS: 现在是无对称，记得根据需要修改
                elif surface_type == 'B':  # B-spline
                    raise NotImplementedError()
                elif surface_type == 'M':  # mixed-type of X and B
                    raise NotImplementedError()
                elif surface_type == 'S':  # asphere surface
                    if len(sl) <= 5:
                        surfaces.append(Asphere(semi_dia, position - th, th, c=roc, device=device))
                    else:
                        coeff_list = []
                        for ac in range(5, len(sl)):
                            if ac == 5:
                                conic = float(sl[5])
                            else:
                                coeff_list.append(float(sl[ac]))
                        surfaces.append(Asphere(semi_dia, position - th, th, c=roc, k=conic, coeff_list=coeff_list,
                                                device=device))
                elif surface_type == 'A':  # aperture
                    surfaces.append(Asphere(semi_dia, position - th, th, c=roc, device=device))
                    aperture_ind = len(surfaces) - 1
                    surfaces[-1].isaperture = True

                elif surface_type == 'CB':  # 坐标间断面
                    if len(sl) < 10:
                        raise ValueError(
                            f"Excel row {excel_row}: CB requires 10 columns, got {len(sl)}"
                        )
                    
                    dec_x = float(sl[5]) if len(sl) > 5 else 0.0
                    dec_y = float(sl[6]) if len(sl) > 6 else 0.0
                    tilt_x = -float(sl[7]) if len(sl) > 7 else 0.0
                    tilt_y = -float(sl[8]) if len(sl) > 8 else 0.0
                    tilt_z = float(sl[9]) if len(sl) > 9 else 0.0

                    surfaces.append(CoordinateBreak(
                        semi_dia, position - th, th, c=roc, dec_x=dec_x, dec_y=dec_y,
                        tilt_x=tilt_x, tilt_y=tilt_y, tilt_z=tilt_z, device=device
                    ))
                elif surface_type == 'G':  # GridSag surface
                    if len(sl) < 9:
                        raise ValueError(
                            f"Excel row {excel_row}: GridSag requires 9 columns, got {len(sl)}"
                        )
                    
                    grid_height = int(sl[6])
                    grid_width = int(sl[7])
                    sag_file_path = Lensdata._resolve_gridsag_path(sl[8], filename)
                    surfaces.append(
                        GridSag(
                            semi_dia, position - th, th,
                            sag_file_path=sag_file_path,
                            grid_shape=(grid_height, grid_width),
                            device=device
                        )
                    )

                elif surface_type == 'GRAD3':  # Gradient_3 surface
                    if len(sl) < 14:
                        raise ValueError(
                            f"Excel row {excel_row}: GRAD3 requires 14 columns, got {len(sl)}"
                        )
                    
                    # 解析非球面和梯度折射率参数
                    # Zemax "Gradient 3" 的参数列顺序（与 Excel 模板一致）：
                    #   sl[5]=conic k
                    #   sl[6]=Δt 梯度积分步长 (mm)
                    #   sl[7]=n0, sl[8]=Nr2, sl[9]=Nr4, sl[10]=Nr6
                    #   sl[11]=Nz1, sl[12]=Nz2, sl[13]=Nz3
                    # 注意：GRAD3 行不承载非球面高次项，Nz* 不能被当成 coeff_list，
                    # 否则求交会被切到 'implicit' 分支并引入伪非球面形变。
                    def _grad3_param(idx, default=0.0):
                        if len(sl) > idx and str(sl[idx]).strip() != '':
                            return float(sl[idx])
                        return default

                    k = _grad3_param(5, 0.0)
                    delta_t = _grad3_param(6, 0.0)
                    n0 = _grad3_param(7, 1.0)
                    Nr2 = _grad3_param(8)
                    Nr4 = _grad3_param(9)
                    Nr6 = _grad3_param(10)
                    Nz1 = _grad3_param(11)
                    Nz2 = _grad3_param(12)
                    Nz3 = _grad3_param(13)

                    surfaces.append(
                        Gradient_3(semi_dia, position - th, th, c=roc, k=k,
                                   n0=n0, Nr2=Nr2, Nr4=Nr4, Nr6=Nr6,
                                   Nz1=Nz1, Nz2=Nz2, Nz3=Nz3, delta_t=delta_t,
                                   coeff_list=None,
                                   material_name=material_name,
                                   device=device)
                    )

                elif surface_type == 'I':  # sensor
                    conic = float(sl[5])
                    img = Asphere(semi_dia, position - th, th, c=roc, k=conic, device=device)

                else:
                    raise ValueError(
                        f"Excel row {excel_row}: unsupported surface type {surface_type!r}"
                    )


        return surfaces, obj, img, materials, aperture_ind

    def save_lensdata(self, path=None):
        savepath = path
        surfaces = self.surfaces
        lensdata = []
        # object surface
        str1 = 'O infinite 0 0 AIR'
        str1 = str1.split()
        str1[1] = self.obj.thickness
        lensdata.append(str1)
        # len surface
        for i in range(len(surfaces)):
            attribute = []
            # 判断面型
            if surfaces[i].type == 'S':
                surfacetype = surfaces[i].type
                if surfaces[i].coeff is not None:
                    conic_coeff = str(float(surfaces[i].k))
                    coeff = ' '.join([str(x) for x in surfaces[i].coeff.tolist()])
                    coeff = conic_coeff + ' ' + coeff
                else:
                    coeff = ''
            elif surfaces[i].type == 'X':
                surfacetype = surfaces[i].type
                conic_coeff = str(float(surfaces[i].k))
                order = str(surfaces[i].J)
                norm_radiu = str(surfaces[i].norm_radiu)
                temp = surfaces[i].coeff.flatten()
                coeff = ' '.join([str(x) for x in temp.tolist()])
                # JNS: XY多形式在保存时与读取时，在text中文件的格式不同，保存了整个系数数组，不只有个别
                coeff = conic_coeff + ' ' + order + ' ' + norm_radiu + ' ' + coeff
            elif surfaces[i].type == 'P':
                surfacetype = surfaces[i].type
                conic_coeff = str(float(surfaces[i].k))
                order = str(surfaces[i].J)
                temp = surfaces[i].coeff.flatten()
                coeff = ' '.join([str(x) for x in temp.tolist()])
                # JNS: XY多形式在保存时与读取时，在text中文件的格式不同，保存了整个系数数组，不只有个别
                coeff = conic_coeff + ' ' + order + ' ' + coeff
            if surfaces[i].isaperture:
                surfacetype = 'A'
            attribute.append(surfacetype)
            attribute.append(str(float(surfaces[i].thickness.detach())))
            roc = float(surfaces[i].c)
            if roc != 0:
                roc = 1 / roc
            attribute.append(str(roc))
            attribute.append(str(surfaces[i].semi_dia))
            attribute.append(str(self.materials[i + 1].name))
            attribute.append(coeff)
            attribute = ' '.join([str(x) for x in attribute])
            attribute = attribute.split()
            lensdata.append(attribute)
        # image surface
        str2 = 'I 0 0 0 AIR'
        str2 = str2.split()
        str2[3] = self.img.semi_dia
        lensdata.append(str2)

        return data_toExcel(lensdata, savepath)

    # ------------------------------------------------------------------------------------
    #  System parameters
    # ------------------------------------------------------------------------------------   

    # entrance pupil and aperture
    def cal_enp_xy(self, mode, f=lambda x: x):
        """
        先计算入瞳y方向孔径，再正向计算出光阑y方向孔径(也可能是直接读取孔径数据)
        再由光阑面型确定，光阑x方向孔径，再通过迭代，计算入瞳x方向孔径
        
        Args:
            mode: 系统孔径定义类型
            f: 光阑面型公式
        out:
            入瞳面x和y方向的长度
            光阑x和y方向孔金长度
        """
        enp_pos = self.find_enp()
        enp_y = self.cal_enp(mode, enp_pos)
        aperture_y = self.cal_aperture(enp_y, enp_pos)
        aperture_x = f(aperture_y)
        theta0 = torch.asin(torch.tensor(0.05))  # JNS:不同系统注意改变初始值
        theta, i = self.newton_method(theta0, aperture_x, direction='x', tolerance=1e-6, it_max=1000)
        distance = self.obj.thickness + enp_pos
        enp_x = torch.tan(theta) * distance * 2
        return enp_x, enp_y, enp_pos, aperture_x, aperture_y

    def find_enp(self, ):
        """
        计算入瞳位置，如果计算结果与zemax有差异，是因为折射率计算差异
        """
        if self.aperture_ind == 0:
            return 0
        # find entrance pupil when optical stop is not the first surface
        min_t = 1e-6  # [mm]
        ones = torch.ones([1, 2], device=self.device)
        zeros = torch.zeros([1, 2], device=self.device)

        o = torch.stack((zeros, zeros, zeros), dim=-1)
        stop_ind = self.aperture_ind
        k = torch.Tensor([1, 0.5]).to(self.device)

        diff = 1
        ang_init = torch.tensor(1).to(self.device)  # deg
        o = torch.stack((zeros, zeros, zeros), dim=-1)
        while diff > min_t:
            angle = torch.deg2rad(ang_init)
            angle = angle * k
            d = torch.stack((zeros, -torch.sin(angle) * ones, -torch.cos(angle) * ones), dim=-1)  # 反向传播
            ray = Ray(o, d, wavelength=self.wavelengths_center, device=self.device)
            ray_final, valid = self.trace(ray, stop_ind=stop_ind, is_fixed=True)
            d_enp = (0 - ray_final.o[..., 1]) / ray_final.d[..., 1] * ray_final.d[..., 2] + ray_final.o[..., 2]
            # d_enp: the distance between entrance pupil and the first surface. while "+" means at right "-" means at left
            diff = torch.abs(d_enp[0] - d_enp[1])
            ang_init = ang_init / 4
        return torch.mean(d_enp)

    def cal_enp(self, mode, enp_pos):
        """
        计算入瞳直径
        """
        if self.aperture_ind == 0:
            return self.surfaces[self.aperture_ind].semi_dia * 2
        else:
            if mode == 'enp':
                return self.enp_dia

            if mode == 'NA_obj':
                """
                直接根据数值孔径角，结合入瞳位置，计算入瞳直径
                """
                distance = self.obj.thickness + enp_pos
                return torch.tan(torch.asin(self.NA_obj)) * distance * 2

            if mode == 'aperture':
                """
                这里计算的入瞳直径是实际入瞳直径，而不是近轴的，但一般显示的都是近轴的
                一般不用这个，只是临时需要，也只适用视场为物高时
                过程：从物方轴上出发，迭代计算最大出射角使其交于光阑边缘处
                """
                target = self.aperture
                theta0 = torch.asin(torch.tensor(0.05))  # JNS:不同系统注意改变初始值
                theta, i = self.newton_method(theta0, target, direction='y', tolerance=1e-6, it_max=1000)
                distance = self.obj.thickness + enp_pos

                return (torch.tan(theta) * distance * 2).item()

    def find_exp(self):
        min_t = 1e-6  # [mm]
        ones = torch.ones([1, 2], device=self.device)
        zeros = torch.zeros([1, 2], device=self.device)
        start_ind = self.aperture_ind
        k = torch.Tensor([1, 0.5]).to(self.device)
        diff = 1
        ang_init = torch.tensor(1).to(self.device)  # deg
        o = torch.stack((zeros, zeros, zeros), dim=-1)
        while diff > min_t:
            angle = torch.deg2rad(ang_init)
            angle = angle * k
            d = torch.stack((zeros, torch.sin(angle) * ones, torch.cos(angle) * ones), dim=-1)  # 正向传播
            ray = Ray(o, d, wavelength=self.wavelengths_center, device=self.device)


            valid, ray_final = self._forward_tracing(ray, stop_ind=len(self.surfaces) - 1, start_ind=start_ind,
                                                     is_fixed=True)
            d_exp = (0 - ray_final.o[..., 1]) / ray_final.d[..., 1] * ray_final.d[..., 2] + ray_final.o[..., 2] - \
                    self.surfaces[-1].thickness


            # d_exp: the distance between exit pupil and the imaging surface. while "+" means at right "-" means at left

            diff = torch.abs(d_exp[..., 0] - d_exp[..., 1])
            ang_init = ang_init / 4
        return torch.mean(d_exp)

    def cal_aperture(self, enp_dia, enp_pos):
        """
        光阑面半径是通过在主波长上从物面中心到光阑面追迹边缘光线计算的
        """
        if (self.aperture_mode == 'aperture'):
            return self.aperture
        if self.view_type == 'height':
            zeros = torch.zeros([1, 1], device=self.device)
            ones = torch.ones([1, 1], device=self.device)
            p_obj = torch.stack((zeros, zeros, zeros), dim=-1)
            distance = self.obj.thickness + enp_pos
            d = torch.abs(normalize(torch.stack((zeros, enp_dia / 2 * ones, distance * ones), dim=-1)))
            t0 = self.obj.thickness / d[..., 2]  # 即追迹至虚拟平面
            o = p_obj + t0[..., None] * d
            o[..., 2] = 0
            ray = Ray(o, d, self.wavelengths_center, device=self.device)
            ray, valid = self.trace(ray, stop_ind=(self.aperture_ind - 1), is_fixed=True)
            t = (self.surfaces[self.aperture_ind - 1].thickness - ray.o[..., 2]) / ray.d[..., 2]
            return torch.abs(ray.o[..., 1] + t[..., None] * ray.d[..., 1]).item()
        if self.view_type == 'angle':
            zeros = torch.zeros([1, 1], device=self.device)
            ones = torch.ones([1, 1], device=self.device)
            d = torch.stack((zeros, zeros, ones), dim=-1)
            o = torch.stack((zeros, enp_dia / 2 * ones, zeros), dim=-1)
            ray = Ray(o, d, self.wavelengths_center, device=self.device)
            ray, valid = self.trace(ray, stop_ind=(self.aperture_ind - 1), is_fixed=True)
            t = (self.surfaces[self.aperture_ind - 1].thickness - ray.o[..., 2]) / ray.d[..., 2]
            return torch.abs(ray.o[..., 1] + t[..., None] * ray.d[..., 1]).item()

    def newton_method(self, theta0, target, direction='y', tolerance=1e-6, it_max=1000):
        # J NS:光阑是旋转对称的，但是面型不一定旋转对称
        # JNS:有没有更简单粗暴的迭代更新方法
        def numerical_derivative(fun, theta, h=1e-6):
            theta1 = theta - h
            theta2 = theta + h
            return (fun(theta2) - fun(theta1)) / (2 * h)

        def fun(theta):
            zeros = torch.zeros([1, 1], device=self.device)
            ones = torch.ones([1, 1], device=self.device)
            o = torch.stack((zeros, zeros, zeros), dim=-1)
            d = torch.stack((zeros, torch.sin(theta) * ones, torch.cos(theta) * ones), dim=-1)
            if direction == 'x':
                d[:, :, [0, 1]] = d[:, :, [1, 0]]
            t0 = self.obj.thickness / d[..., 2]  # 即追迹至虚拟平面
            o = o + t0[..., None] * d
            o[..., 2] = 0
            ray = Ray(o, d, self.wavelengths_center, device=self.device)
            ray, valid = self.trace(ray, stop_ind=(self.aperture_ind - 1), is_fixed=True)
            t = (self.surfaces[self.aperture_ind - 1].thickness - ray.o[..., 2]) / ray.d[..., 2]
            if direction == 'y':
                out = ray.o[..., 1] + t * ray.d[..., 1] - target
            else:
                out = ray.o[..., 0] + t * ray.d[..., 0] - target
            return out

        theta = theta0.to(self.device)
        theta.requires_grad = True

        for i in range(it_max):
            df = numerical_derivative(fun, theta, h=1e-6)
            if df == 0:
                raise ValueError('导数为零，牛顿迭代无法继续')
            out = fun(theta)
            theta_new = theta - out / df
            if abs(theta_new - theta) < tolerance:
                return theta_new.detach(), i
            theta = theta_new

        raise ValueError('牛顿法未能收敛')

    # 有效焦距EFL
    def cal_EFL(self, enp_dia):
        """
        计算有效焦距，采用傍轴光线(入瞳半径的2%)，平行入射，追迹至像方与轴上交点，确定近轴像距(确定近轴像距是否与实际一致)
        反向延长出射光线与入射光线的交点到像面的距离极为有效焦距
        """
        min_t = 1e-6  # [mm]
        ones = torch.ones([1, 2], device=self.device)
        zeros = torch.zeros([1, 2], device=self.device)

        wavelength = self.wavelengths_center
        k = torch.Tensor([1, 0.5]).to(self.device)
        diff = 1

        height = enp_dia / 2 * 0.02
        height = height * k
        d = torch.stack((zeros, zeros, ones), dim=-1)

        while diff > min_t:
            height = height / 2
            o = torch.stack((zeros, height * ones, zeros), dim=-1)
            ray = Ray(o, d, wavelength, device=self.device)
            ray_final, valid = self.trace(ray, is_fixed=True)
            d_enp = (0 - ray_final.o[..., 1]) / ray_final.d[..., 1] * ray_final.d[..., 2] + ray_final.o[..., 2]  # 像距
            # d_enp: the distance between entrance pupil and the first surface. while "+" means at right "-" means at left
            diff = torch.abs(d_enp[0] - d_enp[1])
        efl = (0 - height[..., 0]) / ray_final.d[..., 1][0] * ray_final.d[..., 2][0]
        return efl, torch.mean(d_enp)

    def n_image(self,lam):
        lam = lam * 1e6
        stop_ind = len(self.surfaces)
        i = stop_ind
        n = self.materials[i].ior(lam)
        return n

    def stop_semi_diameter(self):
        """光阑面半直径 [mm]，作为瞳面采样的缩放半径。

        取自实际光阑面而不是 ``self.aperture``：``self.aperture`` 只有在
        ``initial_check()`` 走过后才由 ``cal_aperture`` 填成物理值，而
        ``load_file`` 之后的常规 PSF 链路并不调用它，调用方写入的初值会一直留着。
        ``eye_image_glass.xlsx`` 的光阑半直径恰为 2.0000（与常见初值相同，故该模型
        逐位不变），``eye_image_glass_grad3.xlsx`` 为 1.5——用 2.0 会让边缘光线被
        光阑挡掉，追迹在光阑面直接失败。

        Returns:
            float: 光阑半直径 [mm]；光阑索引不可用时回退到 ``self.aperture``。
        """
        ap_ind = getattr(self, 'aperture_ind', None)
        if ap_ind is not None and 0 <= int(ap_ind) < len(self.surfaces):
            semi_dia = float(getattr(self.surfaces[int(ap_ind)], 'semi_dia', float('nan')))
            if np.isfinite(semi_dia) and semi_dia > 0:
                return semi_dia
        fallback = getattr(self, 'aperture', None)
        return float(fallback) if fallback is not None else 2.0

    def cal_WFNO(self, lam, wavelength=None):
        """Working F/# from four traced marginal rays.

        Args:
            lam: Wavelength in mm (the caller normally passes
                ``wavelength * 1e-6``). Used for the image-space index.
            wavelength: Wavelength in nm for ray tracing. Defaults to
                ``self.wavelengths_center``.

        Returns:
            float: Working F/# = mean over the four marginal rays of
            ``1 / (2 * n_image * sin(u_i))``.

        Notes:
            The marginal angle of each ray is taken along its own meridian:
            the ``(0, ±1)`` rays use the y direction cosine and the
            ``(±1, 0)`` rays use the x direction cosine. Using ``d_y`` for
            the x rays makes ``sin(u)`` collapse to ~5e-5 and inflates F/#
            by three orders of magnitude.

            The chief ray is built exactly as in ``fft_psf_i``: local origin
            ``(0, 0, defocus_shift)`` with direction ``(0, 0, -1)``, then
            reverse-traced to the first surface. The field angle is carried
            by the Excel CoordinateBreak tilts, not by this direction.
        """
        if wavelength is None:
            wavelength = self.wavelengths_center
        n = self.n_image(lam)
        n = n.item() if torch.is_tensor(n) else float(n)

        ray_rel = self._chief_ray_at_first_surface(wavelength)
        d = ray_rel.d
        ## 与 fft_psf_i 取同一个缩放半径：用 self.aperture 会在光阑半直径小于它时
        ## 让边缘光线打在光阑外，追迹在光阑面失败（GRAD3 光阑 1.5 vs 常见初值 2.0）。
        radius = self.stop_semi_diameter()

        ## 与 fft_psf_i 保持同一发射语义：有限物距时边缘光线也应从物点发散，
        ## 否则工作 F/# 与物距无关，而它本应随物距变化。
        object_thickness = float(self.obj.thickness)
        p_obj = None
        if object_thickness > 0:
            t_back = object_thickness / ray_rel.d[..., 2]
            p_obj = ray_rel.o - t_back[..., None] * ray_rel.d

        # (Px, Py) 与取哪个方向余弦配对：子午方向各取自身分量
        cases = ((0.0, 1.0, 1), (0.0, -1.0, 1), (1.0, 0.0, 0), (-1.0, 0.0, 0))
        total = 0.0
        for Px, Py, axis in cases:
            ones = torch.ones((1, 1), device=self.device)
            p_val = torch.stack((ones * ray_rel.o[..., 0].item(),
                                 ones * ray_rel.o[..., 1].item()), dim=-1)
            p_ref = torch.stack((ones * (Px * radius), ones * (Py * radius)), dim=-1)
            x, y = self.ray_aimming(p_val, d, p_ref, wavelength,
                                    tolerance=1e-6, it_max=1000, is_plot=False,
                                    p_obj=p_obj)
            o = torch.stack((x, y, torch.zeros_like(x, device=self.device)), dim=2)
            d_in = d if p_obj is None else self._diverging_direction(x, y, p_obj)
            ray = Ray(o, d_in, wavelength=wavelength, device=self.device)
            _, ray_out = self.trace_eyesensor(ray, ignore_invalid=False,
                                              is_fixed=True, flag=False)
            d_out = ray_out.d.reshape(-1, 3)[0]
            tan_u = abs(d_out[axis].item() / d_out[2].item())
            total += 1.0 / (2.0 * n * np.sin(np.arctan(tan_u)))
        return total / len(cases)

    def _chief_ray_at_first_surface(self, wavelength, defocus_shift=None):
        """Reverse-trace the local-axis chief ray to the first surface.

        Args:
            wavelength: Wavelength in nm.
            defocus_shift: Image-plane longitudinal offset in mm. Defaults to
                0.0, which puts the image plane where the Excel geometry puts
                it. ``fft_psf_i`` passes its own value explicitly.

        Returns:
            Ray: Object-space chief ray whose origin lies on the first
            surface and whose direction points into the system.

        Notes:
            Shared by ``cal_WFNO`` and the ``aimming`` branch of
            ``single_ray_trace`` so the marginal rays and the PSF pupil rays
            use one chief-ray definition. The field angle comes from the
            Excel CoordinateBreak tilts.
        """
        if defocus_shift is None:
            ## Task 5（2026-07-30）：不再按物距推经验离焦。旧实现用
            ## ``-(19.3**2) / object_distance``，其中 19.3 是调参得到的"等效焦距"，
            ## 与 Excel 实际后焦 16.820132 差约 31%。像面位置一律取 Excel 配置值。
            defocus_shift = 0.0
        o = torch.tensor((0.0, 0.0, float(defocus_shift))).to(self.device)
        d = torch.tensor((0.0, 0.0, -1.0)).to(self.device)
        ray_chief = Ray(o, d, wavelength=wavelength, device=self.device)
        _, ray_rel = self.trace_eyesensor(ray_chief, ignore_invalid=False,
                                          is_fixed=True, flag=False)
        ray_rel.d = -ray_rel.d
        return ray_rel

    def cal_WFNO_f(self, lam, wavelength):
        """已废弃：请改用 :meth:`cal_WFNO`。全项目无调用点，仅作历史留存。

        本函数存在两处已确认的缺陷，不要作为参考实现：

        1. ``o``/``d``/``aperture`` 全部硬编码为某次 10° 非平行工况的追迹结果，
           与当前 Excel 配置无关；``aperture = 1.954135753091987`` 并非正确瞳半径，
           正确值为 :meth:`stop_semi_diameter`（光阑面半直径）。
        2. 四条边缘光线一律取 ``d[1]/d[2]``。``(±1, 0)`` 两条子午光线应取
           ``d[0]/d[2]``，否则其 ``M ≈ 5.5e-5``，F/# 被抬高到 10^2~10^3 量级。

        :meth:`cal_WFNO` 已按 ``_chief_ray_at_first_surface`` 统一主光线定义，
        并按各自子午方向取方向余弦分量。
        """

        n = self.n_image(lam)
        lam = wavelength * 1e-6  # 波长[mm]

        cases = [
            (0, 1),  # Px = 0, Py = 1
            (0, -1),  # Px = 0, Py = -1
            (1, 0),  # Px = 1, Py = 0
            (-1, 0)  # Px = -1, Py = 0
        ]
        total_F = 0
        # 四次循环遍历
        for Px, Py in cases:
            Px = torch.tensor([Px]).view(1, 1).to(self.device)
            Py = torch.tensor([Py]).view(1, 1).to(self.device)
            aperture = 1.954135753091987E+000
            o = (1.4620000022E+00, 8.0347829004E-09, -9.7589441151E-02)  # 10_非平行
            d = (-0.0657063144, -0.0000000004, -0.9978390052)
            o = torch.tensor(o).to(self.device)
            d = torch.tensor(d).to(self.device)
            ray_chief1 = Ray(o, d, wavelength=wavelength, device=self.device)
            # 反向追迹到第一个面
            p, ray_rel = self.trace_eyesensor(ray_chief1, ignore_invalid=False, is_fixed=True, flag=False)
            ray_rel.d = -ray_rel.d
            ray_rel.o = ray_rel.o
            d = ray_rel.d

            x = torch.ones_like(Px) * ray_rel.o[..., 0].item()
            y = torch.ones_like(Py) * ray_rel.o[..., 1].item()
            p_ref = torch.stack((Px * aperture, Py * aperture), dim=-1)
            p_val = torch.stack((x, y), dim=-1)

            x, y = self.ray_aimming(p_val, d, p_ref, wavelength, tolerance=1e-6,
                                    it_max=1000, is_plot=False)
            o = torch.stack((x, y, torch.zeros_like(x)), dim=2)
            ray_in = Ray(o, d, wavelength=lam * 1e6, device=self.device)

            # 带相位的光线追迹
            p, ray = self.trace_eyesensor(ray_in, ignore_invalid=False, is_fixed=True, flag=False)
            d = ray.d
            result = [d[..., 0].cpu().item(), d[..., 1].cpu().item(), d[..., 2].cpu().item()]
            print(result)
            F1 = abs(1 / np.sin(np.arctan(d[..., 1].item() / d[..., 2].item())) / (2 * n))
            print(F1.item())

            total_F += (n ** 2) * (np.sin(np.arctan(d[..., 1].item() / d[..., 2].item())) ** 2)
        NA = (total_F / 4) ** 0.5
        F = 1 / (2 * NA)
        return F.item()

    # ------------------------------------------------------------------------------------
    # Ray Aimming
    # ------------------------------------------------------------------------------------

    # find_chief ray for every field of view
    def find_chief_ray(self, Hx, Hy, wavelength=None):
        """
        给定归一化的视场坐标，寻找主光线
        关键修复：确保主光线始终指向像面中心，不受旋转影响
        """
        if wavelength is None:
            wavelength = self.wavelengths_center
        if self.view_type == 'angle':
            # 确保输入是正确的张量形状
            if not isinstance(Hx, torch.Tensor):
                Hx = torch.tensor(Hx, device=self.device)
            if not isinstance(Hy, torch.Tensor):
                Hy = torch.tensor(Hy, device=self.device)
            
            # 确保张量有正确的形状
            if Hx.dim() == 0:
                Hx = Hx.unsqueeze(0).unsqueeze(0)
            if Hy.dim() == 0:
                Hy = Hy.unsqueeze(0).unsqueeze(0)
                
            angle_x = torch.deg2rad(self.FOV * Hx).to(self.device)  # [rad]
            angle_y = torch.deg2rad(self.FOV * Hy).to(self.device)  # [rad]
            ones = torch.ones_like(angle_x, device=self.device)
            zeros = torch.zeros_like(angle_x, device=self.device)
            d = torch.stack((torch.tan(angle_x), torch.tan(angle_y), ones), axis=-1)
            d = d / (torch.sqrt(torch.tan(angle_x) ** 2 + torch.tan(angle_y) ** 2 + 1))[..., None]

            # 关键修复：对于中心视场(0,0)，确保主光线严格沿光轴方向
            if torch.abs(Hx).item() < 1e-6 and torch.abs(Hy).item() < 1e-6:
                # 中心视场：主光线沿光轴方向，不受旋转影响
                d = torch.stack((zeros, zeros, ones), axis=-1)
                p_val = torch.stack((zeros, zeros, zeros), axis=-1).to(self.device)
            else:
                # 非中心视场：需要考虑旋转补偿
                # 但是要确保旋转后主光线仍然指向正确的像面位置
                p_val = torch.stack((zeros, -(d[...,1] * self.enp_pos), zeros),axis=-1).to(self.device)
            
            p_ref = torch.zeros_like(p_val)

            x_in, y_in = self.ray_aimming(p_val, d, p_ref, wavelength, is_fixed=True)
            p_enp = torch.stack((x_in, y_in, torch.zeros_like(x_in)), axis=2)
            ray = Ray(p_enp, d, wavelength, device=self.device)
            return ray
        elif self.view_type == 'height':
            raise ValueError('这部分代码未更新，需要重写')
            p_obj = torch.stack((Hx, Hy, -self.obj.thickness * torch.ones_like(Hx)), axis=2)
            if enp_pos is None:
                enp_pos = self.find_enp()
            p_enp = torch.stack((torch.zeros_like(Hx), torch.zeros_like(Hx), enp_pos * torch.ones_like(Hx)), axis=2)
            p_ref = torch.zeros_like(p_enp)
            x_in, y_in, rms = self.ray_aimming(p_obj, p_enp, p_ref, wavelength, is_fixed=True)
            # 用自由曲面的初始结构验证计算准确性
            p_enp = torch.stack((x_in, y_in, enp_pos * torch.ones_like(Hx)), axis=2)
            d = normalize(p_enp - p_obj)
            t0 = self.obj.thickness / d[..., 2]  # 追迹至虚拟平面
            o = p_obj + t0[..., None] * d
            ray = Ray(o, d, wavelength, device=self.device)
            return ray

    @staticmethod
    def _diverging_direction(x, y, p_obj):
        """从物点出发、经过虚拟面 (x, y, 0) 的发散光线方向。

        有限物距时每根瞳采样光线的方向都不同，必须由自身的发射点与物点决定，
        不能共用主光线方向（那等价于无穷远的平行束）。

        Args:
            x, y: 虚拟面 (z=0) 上的横向坐标，形状任意。
            p_obj: 物点坐标，最后一维为 3，可广播到 ``x`` 的形状。

        Returns:
            torch.Tensor: 归一化方向，形状为 ``x.shape + (3,)``。
        """
        p_sam = torch.stack((x, y, torch.zeros_like(x)), dim=-1)
        return normalize(p_sam - p_obj)

    def ray_aimming(self, p_val, d, p_ref, wavelength=None, views=None, tolerance=1e-6, it_max=1000, is_plot=False,
                    is_fixed=True, p_obj=None):
        """
        目前只适配视场为角度时
        p_val: 初始物方入射点坐标(注意是表面1的虚拟面的点坐标)
        d: 初始物方入射方向坐标。当 p_obj 给定时被忽略，方向由物点与发射点决定
        p_ref: 采样的光瞳坐标作为光线瞄准的参考坐标
        views: 用于确定入射方向。当视场为角度时，即角度;当视场为物高时，就是物点坐标。
        当视场是角度时views没什么特别的作用，当视场为物高比较重要
        tolerance: 最小容差
        it_max: 最大迭代次数
        is_plot: 绘制光瞳点列图，用于检查是否瞄准
        is_fixed:追迹时是否固定孔径
        p_obj: 可选物点坐标（最后一维为 3）。为 None 时沿用外部固定方向 d，行为逐位不变；
            给定时每次迭代按当前发射点重算 ``normalize(p_sam - p_obj)``，即从物点出发的
            发散束，用于有限物距。
        """
        if wavelength is None:
            wavelength = self.wavelengths_center
        zeros = torch.zeros_like(p_val[..., 0], device=self.device)
        ones = torch.ones_like(p_val[..., 0], device=self.device)
        x_in = p_val[..., 0]
        y_in = p_val[..., 1]
        x_ref = p_ref[..., 0]
        y_ref = p_ref[..., 1]
        if self.aperture_ind == 0:
            return x_ref, y_ref
        delta_x = torch.zeros_like(x_in)
        delta_y = torch.zeros_like(y_in)

        # 这里用fun来表示一个追迹的映射关系
        def fun(delta_x, delta_y, ):
            x = x_in + delta_x
            y = y_in + delta_y
            p_sam = torch.stack((x, y, zeros), dim=-1)
            ## 有限物距：方向随发射点变化，必须在迭代内部重算，否则牛顿迭代求解的是
            ## 一个方向被冻结的错误映射。
            d_iter = d if p_obj is None else self._diverging_direction(x, y, p_obj)
            ray = Ray(p_sam, d_iter, wavelength, device=self.device)
            ray, valid = self.trace(ray, stop_ind=(self.aperture_ind - 1), is_fixed=is_fixed)  # 先追迹至光阑前一表面，防止光阑孔径被改变
            t = (self.surfaces[self.aperture_ind - 1].thickness - ray.o[..., 2]) / ray.d[..., 2]
            if torch.isnan(ray.o).any():
                print(i)
                print(torch.isnan(x).any())
                print(torch.isnan(y).any())
                plt.figure()
                plt.scatter(x.cpu(), y.cpu())
                plt.axis('equal')

                raise ValueError('点坐标出现nan')
            if torch.isnan(ray.d).any():
                raise ValueError('方向坐标出现nan')
            p = ray.o + t[..., None] * ray.d

            p_out = p[..., :2] - p_ref[..., :2]
            return p_out

        def dfun(delta_x, delta_y, h=1e-6):
            # 修复：分别计算X和Y方向的梯度，而不是对角线方向
            # 这样可以确保所有象限的光线都能正确收敛
            f_x_plus = fun(delta_x + h, delta_y)
            f_x_minus = fun(delta_x - h, delta_y)
            df_dx = (f_x_plus - f_x_minus) / (2 * h)

            f_y_plus = fun(delta_x, delta_y + h)
            f_y_minus = fun(delta_x, delta_y - h)
            df_dy = (f_y_plus - f_y_minus) / (2 * h)

            # 返回对角元素：[∂fx/∂x, ∂fy/∂y]
            return torch.stack([df_dx[..., 0], df_dy[..., 1]], dim=-1)

        with torch.no_grad():
            for i in range(it_max):
                df = dfun(delta_x, delta_y, h=1e-6)
                p_out = fun(delta_x, delta_y)
                delta_x_new = delta_x - 0.5 * p_out[..., 0] / df[..., 0]
                delta_y_new = delta_y - 0.5 * p_out[..., 1] / df[..., 1]
                # JNS: 引入了阻尼因子lam=0.5
                if torch.sum(torch.abs(delta_x_new - delta_x)) < tolerance and torch.sum(
                        torch.abs(delta_y_new - delta_y)) < tolerance:
                    rms = torch.sum(p_out ** 2, dim=-1).flatten()
                    if is_plot:
                        plt.figure()
                        plt.plot(torch.arange(rms.shape[0]), rms.cpu())
                        plt.figure()
                        plt.scatter((p_out[..., 0] + p_ref[..., 0]).cpu(), (p_out[..., 1] + p_ref[..., 1]).cpu())
                        plt.axis('equal')
                    return x_in + delta_x_new, y_in + delta_y_new
                delta_x = delta_x_new
                delta_y = delta_y_new
        # 检查未收敛原因
        print(torch.isnan(delta_x).any())
        print(torch.isnan(delta_y).any())
        rms = torch.sum(p_out ** 2, axis=-1).flatten()
        plt.figure()
        plt.plot(torch.arange(rms.shape[0]), rms.cpu())
        # 查看表面1虚拟面的点图
        plt.figure()
        plt.scatter((x_in + delta_x).cpu(), (y_in + delta_y).cpu())
        plt.axis('equal')
        # 查看光阑上的点图
        plt.figure()
        plt.scatter((p_out[..., 0] + p_ref[..., 0]).cpu(), (p_out[..., 1] + p_ref[..., 1]).cpu())
        plt.axis('equal')
        raise ValueError('牛顿法未能收敛')

    def cal_p_ref(self, is_plot=False):
        """
        计算中心视场的参考坐标
        如今采用光瞳坐标就不需要再另外计算中心视场的参考坐标，不采用光瞳采样时到可以考虑，但是先搁置
        """
        enp_x, enp_y, enp_pos, aperture_x, aperture_y = self.cal_enp_xy(mode=self.aperture_mode, f=lambda x: x)
        M = 21
        # JNS:对于不同的情况，可以尝试用不同的采样方式，现在统一网格采样。
        x, y = torch.meshgrid(
            torch.linspace(-(enp_x / 2).item(), (enp_x / 2).item(), M, device=self.device),
            torch.linspace(-(enp_y / 2).item(), (enp_y / 2).item(), M, device=self.device),
            indexing='ij')
        p_enp = torch.stack((x, y, enp_pos * torch.ones_like(x, device=self.device)), axis=2)
        p_obj = torch.stack((torch.zeros_like(x), torch.zeros_like(x), -self.obj.thickness * torch.ones_like(x)),
                            axis=2)
        d = normalize(p_enp - p_obj)
        t0 = self.obj.thickness / d[..., 2]  # 追迹至虚拟平面
        o = p_obj + t0[..., None] * d
        o[..., 2] = 0
        ray = Ray(o, d, self.wavelengths_center, device=self.device)
        ray, valid = self.trace(ray, stop_ind=(self.aperture_ind - 1), is_fixed=True)
        t = (self.surfaces[self.aperture_ind - 1].thickness - ray.o[..., 2]) / ray.d[..., 2]
        p_ref = ray.o + t[..., None] * ray.d
        valid_in = self.in_optical_stop(p_ref, aperture_x, aperture_y)
        valid = valid_in & valid
        p_ref = p_ref[..., :2]
        p_ref = p_ref[valid].reshape(-1, 1, 2)
        p_number = valid.sum()
        p_enp = p_enp[valid].reshape(-1, 1, 3)
        # print('追迹有效光线数/总采样数：{}/{}'.format(p_number,M**2))
        if is_plot:
            plt.figure()
            plt.scatter(p_ref[..., 0].detach().cpu(), p_ref[..., 1].detach().cpu())
            plt.axis('square')
            plt.figure()
            plt.scatter(p_enp[..., 0].detach().cpu(), p_enp[..., 1].detach().cpu())
            plt.axis('square')
        return p_ref, p_enp, p_number

    def in_optical_stop(self, p, views):
        # JNS:取决于光阑的形状，光阑不同这里需要做出更改，默认为圆
        raise ValueError('尚未实现')

    def full_entrance_pupil(self, views=0.0, R=None):
        """
        在虚拟面矩形区域内采样大量光线，追迹至光阑面，剔除无效光线
        目前，仅限于视场为角度时
        """
        # maximum radius input
        if R is None:
            with torch.no_grad():
                sag = self.surfaces[0].get_sag(self.surfaces[0].semi_dia, 0.0)
                R = self.surfaces[0].semi_dia - torch.tan(views[1]) * sag  # [mm]
                R = R.item()

        APERTURE_SAMPLING = 101
        x, y = torch.meshgrid(
            torch.linspace(-R, R, APERTURE_SAMPLING, device=self.device),
            torch.linspace(-R, R, APERTURE_SAMPLING, device=self.device),
            indexing='ij'
        )  # indexing="ij"相当于转置

        # generate rays and find valid map
        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        o = torch.stack((x, y, zeros), axis=2)  # shape[n,n,3],其中o[...,0]表示所有采样光线的x坐标，shape[n,n],类推。z坐标都为0
        d = torch.stack((torch.tan(views[0]) * ones, torch.tan(views[1]) * ones, ones), axis=-1)
        d = d / torch.sqrt(torch.tan(views[0]) ** 2 + torch.tan(views[1]) ** 2 + 1)
        # d[...,0]表示x方向的矢量，其中y坐标都是0，所以光在XZ平面传播，
        ray = Ray(o, d, torch.Tensor([580.0]).to(self.device), device=self.device)  # 中心波长580nm，具有起点坐标和方向向量信息以及波长信息的光线
        valid_map = self.trace(ray)
        ray, valid_map = self.trace(ray, stop_ind=(self.aperture_ind), is_fixed=True)
        # find bounding box
        xs, ys = x[valid_map], y[valid_map]

        return valid_map, xs, ys, ray

    # ------------------------------------------------------------------------------------
    # sampling rays 
    # ------------------------------------------------------------------------------------
    """
    采样类型或着说采样面，主要由参数entrance_pupil决定：1、Fasle对应虚拟面采样，2、True对应光瞳面采样
    只有使用光瞳坐标时才考虑光线瞄准
    采样方式,由参数sampling决定分为常规网格('grid')，极坐标采样'radial'，斐波那契采样('Fibonacci')以及特殊的六边环，GQ(高斯求积)，RA(矩形阵列)
    光瞳采样只考虑GQ与RA采样
    """

    def sample_ray_angle(self, wavelength, Hx=0.0, Hy=0.0, M=0.22, R=None, entrance_pupil=False, sampling="RA"):
        """
        当视场类型为角度时，最常规简单的光线采样方式，可以用于绘制rms,优化镜头等功能的光线采样
        输入：
            wavelength: the wavelength of ray
            Hx: 归一化x视场坐标
            Hy: 归一化y视场坐标
            M: 各种采样方式的参数，形式不定。
            比如表示常规网格和极坐标采样的个数，或者表示RA算法的采样间隔，又或者是GQ算法的环数和幅数
            R: 采样孔径
            entrance_pupil: 确定采样面
            sampling: 采样方式
        """
        # 单位转化为弧度
        Hx = Hx.reshape(-1, 1)
        Hy = Hy.reshape(-1, 1)
        angle_x = torch.deg2rad(self.FOV * Hx)  # [rad]
        angle_y = torch.deg2rad(self.FOV * Hy)  # [rad]

        # maximum radius input
        if R is None:
            """
            当R没被给出时，在面1(物面后的第一个面)处构建虚拟面，求当前视场光线在该虚拟面处交点范围[-R,R]
            R就是某一视场在面0虚拟面上的最大光通径
            当R具体给出时就是在指定的[-R,R]内采样点
            """
            with torch.no_grad():
                sag = self.surfaces[0].get_sag(self.surfaces[0].semi_dia, 0.0)
                R = self.surfaces[0].semi_dia - torch.tan(angle_y) * sag  # [mm]
                R = R.item()

        if entrance_pupil:
            """
            初始光瞳坐标，根据采样的光瞳坐标以及视场方向，确定在虚拟面上的坐标,再考虑是否光线瞄准
            """
            ray = self.find_chief_ray(Hx, Hy)

            Px, Py, weight = self.sampling(M=M, method=sampling)

            x = torch.ones_like(Px) * ray.o[..., 0].item()
            y = torch.ones_like(Py) * ray.o[..., 1].item()
            #             up_bound = self.enp_dia / 4 + r
            #             down_bound = -self.enp_dia / 4 + r
            # 坐标变换[-1,1]至[down_bound,up_bound]
            #             x = Px*(up_bound-down_bound)/2+(up_bound+down_bound)/2
            #             y = Py*(up_bound-down_bound)/2+(up_bound+down_bound)/2
            if self.aimming:
                p_ref = torch.stack((Px * self.aperture, Py * self.aperture), axis=-1)
                p_val = torch.stack((x, y), axis=-1)

                ones = torch.ones_like(p_val[..., 0])
                d = torch.stack((torch.tan(angle_x) * ones, torch.tan(angle_y) * ones, ones), axis=-1)
                d = d / torch.sqrt(torch.tan(angle_x) ** 2 + torch.tan(angle_y) ** 2 + 1)
                x, y = self.ray_aimming(p_val, d, p_ref, wavelength, views=[angle_x, angle_y], tolerance=1e-6,
                                        it_max=1000, is_plot=False)

        else:
            Px, Py, weight = self.sampling(M=M, method=sampling)
            x = Px * R
            y = Py * R
        o = torch.stack((x, y, torch.zeros_like(x, device=self.device)), axis=2)
        d = torch.stack((
            torch.tan(angle_x) * torch.ones_like(x),
            torch.tan(angle_y) * torch.ones_like(x),
            torch.ones_like(x)), axis=-1)
        d = d / torch.sqrt(torch.tan(angle_x) ** 2 + torch.tan(angle_y) ** 2 + 1)
        return Ray(o, d, wavelength, device=self.device, weight=weight)

    def sample_ray_height(self, wavelength, views, M=5, R=None, sampling="grid", entrance_pupil=False):
        """
        sampling ray with object height view field
        entrance pupil: means sampling rays at entrance pupil or virtual plane 
        """
        # sampling at the entrance pupil
        raise ValueError('需要重新写')
        if R is None:
            with torch.no_grad():
                R = self.surfaces[0].semi_dia  # [mm]

        if entrance_pupil:
            p_ref, p_enp, p_number = self.cal_p_ref(is_plot=False)
            x_enp, y_enp, rms = self.ray_aimming(views, p_ref, p_enp, tolerance=1e-6, it_max=1000, is_plot=False)

            # print('采样光线数/中心视场光线数：{}/{}'.format(torch.numel(x_enp),p_number))
            d_enp = p_enp[..., 2]
            p_sam = torch.stack((x_enp, y_enp, d_enp), axis=-1)
            p_obj = torch.stack((
                views[0] * torch.ones_like(x_enp),
                views[1] * torch.ones_like(x_enp),
                -self.obj.thickness * torch.ones_like(x_enp)), axis=2)
        else:
            x, y = self.sampling(R, M=M, method=sampling)
            p_sam = torch.stack((x, y, torch.zeros_like(x, device=self.device)), axis=2)
            p_obj = torch.stack((
                views[0] * torch.ones_like(x),
                views[1] * torch.ones_like(x),
                -self.obj.thickness * torch.ones_like(x)), axis=2)

        d = normalize(p_sam - p_obj)
        t0 = self.obj.thickness / d[..., 2]  # 追迹至虚拟平面
        o = p_obj + t0[..., None] * d
        return Ray(o, d, wavelength, device=self.device)

    def sample_ray_real_height(self,wavelength, view , m, sampling ='grid'):


        o = (1.4620000004E+00, 0.0000000000E+00, -9.7589440915E-02)  # 0
        d = (-0.0657063142, 0.0000000000, -0.9978390052)

        # 将元组转换为张量
        o = torch.tensor(o).to(self.device)
        d = torch.tensor(d).to(self.device)
        ray_chief1 = Ray(o, d, wavelength=wavelength, device=self.device)
        ray_chief = Ray(o, -d, wavelength=wavelength, device=self.device)  # 主光线与像面的交点
        # 反向追迹到第一个面
        p, ray_rel = self.trace_eyesensor(ray_chief1, ignore_invalid=False, is_fixed=True, flag=False)
        ray_rel.d = -ray_rel.d
        ray_rel.o = ray_rel.o

        # 初始采样光线
        d = ray_rel.d
        Px, Py, weight = self.sampling(M=m, method='grid')
        aperture = 1.954135753091987E+000

        x = torch.ones_like(Px) * ray_rel.o[..., 0].item()
        y = torch.ones_like(Py) * ray_rel.o[..., 1].item()
        p_ref = torch.stack((Px * aperture, Py * aperture), dim=-1)
        p_val = torch.stack((x, y), dim=-1)

        x, y = self.ray_aimming(p_val, d, p_ref, wavelength, tolerance=1e-6,
                                it_max=1000, is_plot=False)
        o = torch.stack((x, y, torch.zeros_like(x)), dim=2)
        ray = Ray(o, d, wavelength=wavelength, weight=None,  device=self.device)

        return ray



    def sampling(self, M=0.22, method='RA'):
        """
        光线的采样方式，逐步增加，目前以写好GQ和RA但还没引入，六边环等采样方式还没写
        输出为归一化坐标
        M: 各种采样方式的参数，形式不定。
        """
        # JNS: 对于光瞳采样目前的写法，导致每次追迹一个视场都要重新计算参考光瞳坐标，写法待优化
        if method == 'grid':
            # 光线数为M**2
            device = self.device
            # Px, Py = torch.meshgrid(
            #     torch.linspace(-1, 1, M, device=self.device),
            #     torch.linspace(-1, 1, M, device=self.device),
            #     indexing='ij')
            x = torch.linspace(-1, 1, M, device=self.device)
            y = x
            X, Y = torch.meshgrid(x, y, indexing='xy')
            ind = (X ** 2 + Y ** 2) <= 1
            Pxy = torch.stack((X[ind], Y[ind], torch.ones_like(X[ind])), dim=-1)
            Px = Pxy[..., 0].reshape(-1, 1).to(device)  # 筛选x坐标并转换成列向量
            Py = Pxy[..., 1].reshape(-1, 1).to(device)
            weight = torch.ones_like(Px)
        elif method == 'radial':
            r = torch.linspace(0, 1, M, device=self.device)
            theta = torch.linspace(0, 2 * torch.pi, M + 1, device=self.device)[0:M]
            Px = r[None, ...] * torch.cos(theta[..., None])
            Py = r[None, ...] * torch.sin(theta[..., None])
            weight = torch.ones_like(Px)
        elif method == 'inverse':
            # M就是光线数量
            r = torch.sqrt(torch.rand(M, device=self.device))
            theta = 2 * torch.pi * torch.rand(M, device=self.device) - torch.pi  # 角度应该在0到2π之间
            Px = (r * torch.cos(theta)).view(-1, 1)
            Py = (r * torch.sin(theta)).view(-1, 1)
            weight = torch.ones_like(Px)
        elif method == 'Fibonacci':
            phi = torch.tensor((np.sqrt(5) - 1) / 2)
            Px = torch.zeros(M ** 2, device=self.device)
            Py = torch.zeros(M ** 2, device=self.device)
            for i in range(M ** 2):
                Px[i] = np.sqrt(i + 1) * torch.cos(2 * torch.pi * (i + 1) * phi)
                Py[i] = np.sqrt(i + 1) * torch.sin(2 * torch.pi * (i + 1) * phi)
            Px = Px / np.sqrt(M ** 2)
            Py = Py / np.sqrt(M ** 2)
            Px = torch.reshape(Px, (M, M))
            Py = torch.reshape(Py, (M, M))
            weight = torch.ones_like(Px)
        elif method == 'GQ':
            Px, Py, weight = self.GQ(rings=M[0], arms=M[1], is_symmetry=False)
        elif method == 'RA':
            Px, Py, weight = self.RA(DEL=M, is_symmetry=False)

        return Px, Py, weight

    # ------------------------------------------------------------------------------------
    # light source sampling 
    # ------------------------------------------------------------------------------------
    def source_sampling(self, EFL, n_view=21, n_ray=100, l0=None, w0=None, wavelength=None):
        """
        EFL: 理想焦距, 用于视场均匀采样确定视场能量权重
        l0: 理想接收器x方向的长度
        w0: 理想接收器y方向的宽度
        wavelength: 波长
        注释：l0与w0大小与比例主要取决于系统的焦距以及物方的长宽比例
        """
        if wavelength is None:
            wavelength = self.wavelengths_center
        ### 视场均匀采样:理想像点均匀采样，根据理想像高计算公式反算视场角
        height0 = EFL * torch.tan(self.angle)  # 理想像高
        if l0 is None:
            l0 = height0 * np.sqrt(2)
            w0 = l0
        ix, iy, _ = self.sampling(M=n_view, method='grid')
        hx = (ix * l0 / 2).view(-1, 1)
        hy = (iy * w0 / 2).view(-1, 1)

        k1 = EFL / torch.sqrt(hx ** 2 + hy ** 2 + EFL ** 2)  # 权重因子1
        # print('权重因子1的shape：', k1.shape)
        d = torch.stack((hx, hy, EFL * torch.ones_like(hx)), dim=-1)
        d = normalize(d)
        # print('d的shape：', d.shape)
        ### 光瞳采样
        # JNS：光阑位于最前方，直接采样，不瞄准
        px, py, _ = self.sampling(M=n_ray, method='inverse')
        p_pupil = torch.stack((px * self.aperture, py * self.aperture, torch.zeros_like(px)), dim=-1).view(-1, 1, 3)
        # print('p_pupil的shape：', p_pupil.shape)
        k2 = k1 ** 2

        repeated_p = p_pupil.repeat(d.shape[0], 1, 1)
        # print('repeated_p的shape：', repeated_p.shape)
        repeated_d = d.repeat(1, p_pupil.shape[0], 1).view(-1, 1, 3)
        # print('repeated_d的shape：', repeated_d.shape)
        k1 = k1.repeat(1, p_pupil.shape[0]).view(-1, 1)
        k2 = k2.repeat(1, p_pupil.shape[0]).view(-1, 1)

        # plt.figure()
        # plt.scatter(ix.cpu(), iy.cpu(), s=1)
        #
        # plt.figure()
        # plt.scatter(px.cpu(), py.cpu(), s=1)

        return Ray(repeated_p, repeated_d, wavelength=wavelength, device=self.device, weight=k1 * k2)

    # ------------------------------------------------------------------------------------
    # trace   attention: the rays must start at the virtual plane of the first surface 
    # ------------------------------------------------------------------------------------
    # def single_ray_trace(self, wavelength, Hx, Hy, Px, Py, stop_ind):
    #     """
    #     简单版，光阑位于最前方。视场为角度
    #     """
    #     Hx = torch.tensor([Hx]).view(1, 1).to(self.device)
    #     Hy = torch.tensor([Hy]).view(1, 1).to(self.device)
    #     Px = torch.tensor([Px]).view(1, 1).to(self.device)
    #     Py = torch.tensor([Py]).view(1, 1).to(self.device)
    #     Hx = Hx.reshape(-1, 1)
    #     Hy = Hy.reshape(-1, 1)
    #     angle_x = torch.deg2rad(self.FOV * Hx)  # [rad]
    #     angle_y = torch.deg2rad(self.FOV * Hy)  # [rad]
    #     sag = self.surfaces[0].get_sag(self.surfaces[0].semi_dia, 0.0)
    #     R = self.surfaces[0].semi_dia - torch.tan(angle_y) * sag  # [mm]
    #     print(R)
    #     R = R.item()
    #     x = Px * R
    #     y = Py * R
    #     o = torch.stack((x, y, torch.zeros_like(x, device=self.device)), dim=2)
    #     d = torch.stack((
    #         torch.tan(angle_x) * torch.ones_like(x),
    #         torch.tan(angle_y) * torch.ones_like(x),
    #         torch.ones_like(x)), dim=-1)
    #     d = d / torch.sqrt(torch.tan(angle_x) ** 2 + torch.tan(angle_y) ** 2 + 1)
    #     ray = Ray(o, d, wavelength, device=self.device)
    #     # p0, _ = self.trace2sensor(ray)
    #     # print(p0[..., 0].item())
    #     # print(p0[..., 1].item())
    #     # print(p0[..., 2].item())
    #
    #     ray, _ = self.trace(ray, stop_ind=stop_ind)
    #
    #     print(ray.o[..., 0].item())
    #     print(ray.o[..., 1].item())
    #     print(ray.o[..., 2].item())
    #     print(ray.d[..., 0].item())
    #     print(ray.d[..., 1].item())
    #     print(ray.d[..., 2].item())
    #
    #     return ray

    def single_ray_trace(self, Hx, Hy, Px, Py, wavelength=None, stop_ind=None):
        """
        单光线追迹，没有记录功能
        """
        if wavelength is None:
            wavelength = self.wavelengths_center
        Hx = torch.tensor([Hx]).view(1, 1).to(self.device)
        Hy = torch.tensor([Hy]).view(1, 1).to(self.device)
        Px = torch.tensor([Px]).view(1, 1).to(self.device)
        Py = torch.tensor([Py]).view(1, 1).to(self.device)
        Hx = Hx.reshape(-1, 1)
        Hy = Hy.reshape(-1, 1)
        angle_x = torch.deg2rad(self.FOV * Hx)  # [rad]
        angle_y = torch.deg2rad(self.FOV * Hy)  # [rad]
        ones = torch.ones_like(Px)
        d = torch.stack((
            torch.tan(angle_x) * ones,
            torch.tan(angle_y) * ones,
            ones), dim=-1)
        d = d / torch.sqrt(torch.tan(angle_x) ** 2 + torch.tan(angle_y) ** 2 + 1)
        if self.aimming:
            # 主光线沿局部光轴，视场角由 Excel CB 面的 tilt_x/tilt_y 承载，
            # 与 fft_psf_i 同源（见 _chief_ray_at_first_surface）。
            ray_rel = self._chief_ray_at_first_surface(wavelength)
            d = ray_rel.d
            x = ones * ray_rel.o[..., 0].item()
            y = ones * ray_rel.o[..., 1].item()
            aperture = float(self.aperture) if self.aperture is not None else 2.0

            p_ref = torch.stack((Px * aperture, Py * aperture), dim=-1)
            p_val = torch.stack((x, y), dim=-1)

            x, y = self.ray_aimming(p_val, d, p_ref, wavelength, tolerance=1e-6,
                                    it_max=1000, is_plot=False)
            o = torch.stack((x, y, torch.zeros_like(x, device=self.device)), dim=2)
        else:
            R = self.enp_dia / 2
            x = Px * R
            y = Py * R
            o = torch.stack((x, y, torch.zeros_like(x, device=self.device)), dim=2)
            sag = self.surfaces[0].get_sag(x, y)
            t = (sag - o[..., 2]) / d[..., 2]
            o = o - t[..., None] * d
            o[..., 2] = 0

        ray = Ray(o, d, wavelength=wavelength, device=self.device)
        if stop_ind == 'sensor':
            ray, _ = self.trace(ray)
            t = (self.surfaces[-1].thickness - ray.o[..., 2]) / ray.d[..., 2]
            p = ray.o + t[..., None] * ray.d
            p[..., 2] = 0
            return p, ray.d
        elif stop_ind == 'eyesensor':
            p, ray = self.trace_eyesensor(ray)

            return p, ray.d
        else:
            ray, _ = self.trace(ray, stop_ind=stop_ind)
            p = ray.o
            p = torch.reshape(p, (np.prod(p.shape[:-1]), 3))
            return p, ray.d

    def single_ray_noaimming(self,wavelength=None, stop_ind=None):
        """
        单光线追迹，没有光线瞄准功能
        """
        if wavelength is None:
            wavelength = self.wavelengths_center

        # 主光线与像面的交点（1.462，0）
        # o = (1.4620000007E+00, 2.9771259954E-09, -9.7589440956E-02)
        # d = (-0.0664619802, 0.0702248787, -0.9953146596)
        o = (-1.7333937059E+00,-3.5635514818E+00,0.0000000000E+00)
        d = (0.0885579114,0.2024271793,0.9752849499)

        # 将元组转换为张量
        o = torch.tensor(o).to(self.device)
        d = torch.tensor(d).to(self.device)
        ray = Ray(o, d, wavelength=wavelength, device=self.device)
        if stop_ind == 'sensor':
            ray, _ = self.trace(ray)
            t = (self.surfaces[-1].thickness - ray.o[..., 2]) / ray.d[..., 2]
            p = ray.o + t[..., None] * ray.d
            p[..., 2] = 0
            return p, ray.d
        elif stop_ind == 'eyesensor':
            p,ray = self.trace_eyesensor(ray)

            return p,ray.d
        else:
            ray, _ = self.trace(ray, stop_ind=stop_ind)
            p = ray.o
            # p = torch.reshape(p, (np.prod(p.shape[:-1]), 3))
            return p,ray.d


    def multi_ray_trace(self, wavelength, Hx, Hy,):
        """
        多根光线追迹
        """
        Hx = torch.tensor([Hx]).view(1, 1).to(self.device)
        Hy = torch.tensor([Hy]).view(1, 1).to(self.device)
        Hx = Hx.reshape(-1, 1)
        Hy = Hy.reshape(-1, 1)
        angle_x = torch.deg2rad(self.FOV * Hx)  # [rad]
        angle_y = torch.deg2rad(self.FOV * Hy)  # [rad]
        ray = self.find_chief_ray(Hx, Hy)

        Px, Py, weight = self.sampling(M=[6, 12], method='GQ')

        x = torch.ones_like(Px) * ray.o[..., 0].item()
        y = torch.ones_like(Py) * ray.o[..., 1].item()
        p_ref = torch.stack((Px * self.aperture, Py * self.aperture), axis=-1)
        p_val = torch.stack((x, y), axis=-1)

        ones = torch.ones_like(p_val[..., 0])
        d = torch.stack((torch.tan(angle_x) * ones, torch.tan(angle_y) * ones, ones), axis=-1)
        d = d / torch.sqrt(torch.tan(angle_x) ** 2 + torch.tan(angle_y) ** 2 + 1)
        x, y = self.ray_aimming(p_val, d, p_ref, wavelength, views=[angle_x, angle_y], tolerance=1e-6, it_max=1000,
                                is_plot=False)
        o = torch.stack((x, y, torch.zeros_like(x, device=self.device)), axis=2)
        ray = Ray(o, d, wavelength, device=self.device, weight=weight)

        p0, _ = self.trace2sensor(ray)

        # 计算psf(不考虑衍射）
        ps = p0[..., :2]
        ps_mean = torch.mean(ps, axis=0)
        print(ps_mean)
        M = 256
        pixel_size = 250e-3

        L = M * pixel_size
        image_plane_counts = torch.zeros([M, M])
        v = ((p0[..., 0] - ps_mean[..., 0]) / (L / 2) + 1) * (M - 1) / 2

        v = torch.floor(v)

        u = ((p0[..., 1] - ps_mean[..., 1]) / (L / 2) + 1) * (M - 1) / 2
        u = torch.floor(u)

        for i in range(torch.numel(u)):
            image_plane_counts[int(u[i]), int(v[i])] += 1
        image_plane_psf = image_plane_counts / image_plane_counts.sum()  # 归一化

        # 显示psf(不考虑衍射）
        plt.imshow(image_plane_psf, cmap='gray')
        plt.colorbar(label='Normalized Intensity')
        plt.title('PSF Image Plane')
        plt.show()

        print(p0[..., 0].tolist())
        print(p0[..., 1].tolist())
        print(p0[..., 2].tolist())
        return ray

    def sample_ray_noaimming(self,wavelength=None,num_samples=1000):
        if wavelength is None:
            wavelength = self.wavelengths_center

        o = (0.0000000000E+00, 1.4619999994E+00, -9.7589440786E-02)
        d = (0.0000000000, -0.0657063143, -0.9978390052)
        # 将元组转换为张量
        o = torch.tensor(o).to(self.device)
        d = torch.tensor(d).to(self.device)
        ray_chief = Ray(o, d, wavelength=wavelength, device=self.device)
        # 反向追迹到第一个面
        p,ray_rel = self.trace_eyesensor(ray_chief)
        ray_rel.d = -ray_rel.d
        ray_rel.o = ray_rel.o
        print(ray_rel.d.shape)

        # # 确保 ray_rel.d 的形状是 (3,)
        # if ray_rel.d.shape != (3,):
        #     ray_rel.d = ray_rel.d.reshape(3)
        #
        # # 选择一个与 z_axis = ray_rel.d 不共线的参考向量
        # reference_vector = torch.tensor([1.0, 0.0, 0.0]).to(self.device)
        # if torch.allclose(ray_rel.d, reference_vector) or torch.allclose(ray_rel.d, -reference_vector):
        #     reference_vector = torch.tensor([0.0, 1.0, 0.0]).to(self.device)
        #
        # # x 轴是 reference_vector 与 z_axis 的叉积
        # x_axis = torch.cross(reference_vector, ray_rel.d).to(self.device)
        # x_axis = x_axis / torch.norm(x_axis)
        #
        # # y 轴是 z_axis 与 x_axis 的叉积
        # y_axis = torch.cross(ray_rel.d, x_axis).to(self.device)
        # y_axis = y_axis / torch.norm(y_axis)
        #
        # # 在局部坐标系的 xy 平面上采样
        # aperture_radius = self.surfaces[self.aperture_ind].semi_dia
        # samples_local = (torch.rand(num_samples, 2) * 2 * aperture_radius - aperture_radius).to(self.device)
        #
        # # 将局部坐标映射到全局坐标系
        # samples_global = (
        #         ray_rel.o +
        #         samples_local[:, 0:1] * x_axis +
        #         samples_local[:, 1:2] * y_axis
        # )
        # # 扩展光线方向，使其形状与采样坐标一致
        # ray_directions = ray_rel.d.unsqueeze(0).expand(num_samples, -1)  # 形状从 (3,) 变为 (num_samples, 3)
        #
        # ray = Ray(samples_global, ray_directions, wavelength=wavelength, device=self.device)
        # # 剔除超出光阑范围的点
        # ray_aperture, valid_map = self.trace(ray, stop_ind=self.aperture_ind)
        # # print(valid_map.shape)
        # #
        # # # 筛选有效光线
        # # xs = ray_aperture.o[..., 0][valid_map]  # 使用 valid_indices 直接索引
        # # ys = ray_aperture.o[..., 1][valid_map]
        #
        # # 计算光线在光阑位置的横向距离
        # distances = torch.norm(ray_aperture.o[:, 0, :2], dim=1)  # 取第二维的索引 0，形状为 [num_samples]
        #
        # # 剔除距离大于光阑半径的光线
        # valid_indices = distances <= aperture_radius  # 形状为 [num_samples]
        #
        # # 筛选有效光线
        # xs = ray_aperture.o[valid_indices, 0, 0]  # 取第二维的索引 0，第三维的索引 0（x 坐标）
        # ys = ray_aperture.o[valid_indices, 0, 1]  # 取第二维的索引 0，第三维的索引 1（y 坐标）

        # 初始采样光线
        d = ray_rel.d
        Px, Py, weight = self.sampling(M=181 + 1, method='grid')
        aperture = 1.95413575309198
        print(self.aperture)
        x = torch.ones_like(Px) * ray_rel.o[..., 0].item()
        y = torch.ones_like(Py) * ray_rel.o[..., 1].item()
        p_ref = torch.stack((Px * aperture, Py * aperture), dim=-1)
        p_val = torch.stack((x, y), dim=-1)

        x, y = self.ray_aimming(p_val, d, p_ref, wavelength, tolerance=1e-6,
                                it_max=1000, is_plot=False)
        zeros = torch.zeros_like(p_val[..., 0], device=self.device)
        p_sam = torch.stack((x, y, zeros), dim=-1)
        ray = Ray(p_sam, d, wavelength, device=self.device)
        ray,_ = self.trace(ray,stop_ind=self.aperture_ind)
        # p,ray = self.trace_eyesensor(ray, is_fixed=True )  # 先追迹至光阑前一表面，防止光阑孔径被改变
        # t = (self.surfaces[self.aperture_ind - 1].thickness - ray.o[..., 2]) / ray.d[..., 2]
        # p = ray.o + t[..., None] * ray.d

        # 查看光阑上的点图
        plt.figure()
        plt.scatter(ray.o[...,0].cpu(), ray.o[...,1].cpu())
        plt.axis('equal')
        plt.show()

        return x, y

    def huygens_psf(self, n_p, n_i, wavelength, Hx, Hy, d_delta=0, ):
        """
        Args:
            n_p: 光瞳采样数
            n_i: 像面采样数
            wavelength:工作波长。单位是nm
            Hx,Hy: 归一化视场坐标
            d_delta: 像面采样间距。默认由光瞳采样数计算获得
        """
        # 参数设置
        Hx = torch.tensor([Hx]).view(1, 1).to(self.device)
        Hy = torch.tensor([Hy]).view(1, 1).to(self.device)
        Hx = Hx.reshape(-1, 1)
        Hy = Hy.reshape(-1, 1)
        ones = torch.ones([1, 1], device=self.device)
        angle_x = torch.deg2rad(self.FOV * Hx)  # [rad]
        angle_y = torch.deg2rad(self.FOV * Hy)  # [rad]
        lam = wavelength * 1e-6  # 波长[mm]
        if d_delta == 0:
            F_num = self.cal_WFNO(lam, wavelength)  # F/#
            print('check working F/#:{} '.format(F_num))
            d_delta = (F_num * lam / n_p ** 0.5).item()  # 默认网格间距[mm]
            print(d_delta)

        # 初始采样光线
        d = torch.stack((
            torch.tan(angle_x) * ones,
            torch.tan(angle_y) * ones,
            ones), dim=-1)  # shape(1,3)
        d = d / torch.sqrt(torch.tan(angle_x) ** 2 + torch.tan(angle_y) ** 2 + 1)
        Px, Py, weight = self.sampling(M=2 / n_p, method='RA')
        if self.aimming:
            ray = self.find_chief_ray(Hx, Hy)
            x = torch.ones_like(Px) * ray.o[..., 0].item()
            y = torch.ones_like(Py) * ray.o[..., 1].item()
            p_ref = torch.stack((Px * self.aperture, Py * self.aperture), dim=-1)
            p_val = torch.stack((x, y), dim=-1)

            x, y = self.ray_aimming(p_val, d, p_ref, wavelength, tolerance=1e-6,
                                    it_max=1000, is_plot=False)
        else:
            sag = self.surfaces[0].get_sag(self.surfaces[0].semi_dia, 0.0)
            R = self.surfaces[0].semi_dia - torch.tan(angle_y) * sag  # [mm]
            R = R.item()
            x = Px * R
            y = Py * R
        o = torch.stack((x, y, torch.zeros_like(x)), dim=2)
        distance = torch.sum(o * d, dim=-1)  # [mm]
        phase_init = (1 / lam * distance - torch.trunc(1 / lam * distance)) * 2 * torch.pi
        ray_in = Ray(o, d, wavelength=lam * 1e6, weight=None, phase=phase_init, device=self.device)

        # 带相位的光线追迹
        # ray_final, weight = self.trace(ray_in, stop_ind=None, is_fixed=True, flag=False, OPD_flag=True)
        # PD1 = (self.surfaces[-1].thickness - ray_final.o[..., 2]) / ray_final.d[..., 2]  # 最后透镜面到像面的距离
        # p = ray_final.o + PD1[..., None] * ray_final.d
        _,ray_final = self.trace_eyesensor(ray_in, ignore_invalid=False, is_fixed=True, flag=False)

        # p_mean = torch.mean(p, dim=0)  # 计算质心
        chief_ray = self.find_chief_ray(Hx, Hy)
        m, _ = self.trace_eyesensor(chief_ray)
        p_mean = m     # 计算主光线像点
        print(p_mean)

        # print('采样的光线数量', p.shape[0])
        # print('路径长度', PD1[0])

        # 平面像面坐标网格
        x = torch.linspace(-d_delta * n_i / 2, d_delta * n_i / 2, n_i + 1).to(self.device)
        y = torch.linspace(d_delta * n_i / 2, -d_delta * n_i / 2, n_i + 1).to(self.device)

        x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')

        print(self.img.k)
        # 得到归一化法向量
        Q = self.img.k + 1
        S_2 = self.distance_2(p_mean[..., 0], p_mean[..., 1])
        g_sp = self.img.c * S_2 / (1 + torch.sqrt(1 - Q * S_2 * self.img.c ** 2))
        dg_spdS_2 = self.img.c / (2 * torch.sqrt(1 - Q * S_2 * self.img.c ** 2))
        dS_2dx = 2 * p_mean[..., 0]
        dS_2dy = 2 * p_mean[..., 1]
        dg_spdx = dg_spdS_2 * dS_2dx
        dg_spdy = dg_spdS_2 * dS_2dy
        dgdx = -dg_spdx
        dgdy = -dg_spdy
        dhdz = torch.ones_like(p_mean[...,0])
        n = normalize(torch.stack((dgdx, dgdy, dhdz), dim=-1))

        # 创建一个表示z轴的单位向量，形状与 n 相同
        z_axis = torch.tensor([0, 0, 1], dtype=n.dtype, device=n.device).expand_as(n)

        # 计算法向量与z轴的点积
        dot_product = torch.sum(n * z_axis)

        # 计算夹角（以弧度为单位）
        angle_rad = torch.acos(dot_product)
        # 旋转z、y坐标
        z_rotated = -y_grid * torch.sin(angle_rad)
        y_grid = y_grid * torch.cos(angle_rad)

        x_grid = x_grid + p_mean[..., 0]
        y_grid = y_grid + p_mean[..., 1]
        z_rotated = z_rotated + p_mean[..., 2]


        # r = self.img.c
        # # 定义纬度φ和经度θ的采样点数
        # phi = torch.linspace(0, np.pi/2, n_i + 1).to(self.device)
        # theta = torch.linspace(0, 2 * np.pi, n_i + 1).to(self.device)
        #
        # # 生成网格点
        # Theta, Phi = torch.meshgrid(theta, phi, indexing='xy')
        #
        # # 使用球坐标系的转换公式
        # X = r * torch.sin(Phi) * torch.cos(Theta)
        # Y = r * torch.sin(Phi) * torch.sin(Theta)
        # Z = r * torch.cos(Phi)
        # x_grid = X + p_mean[..., 0]
        # y_grid = Y + p_mean[..., 1]
        # z_grid = Z + p_mean[..., 2]
        #
        # # 创建3D图形
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # # 绘制球面上的点
        # # ax.scatter(x_grid.cpu().numpy().ravel(), y_grid.cpu().numpy().ravel(), z_grid.cpu().numpy().ravel())
        # ax.plot_surface(x_grid.cpu().numpy(),  y_grid.cpu(),  z_grid.cpu(), color='c', edgecolor='k', linewidth=0.5)


        # 设置图形标签
        # ax.set_xlabel('X axis')
        # ax.set_ylabel('Y axis')
        # ax.set_zlabel('Z axis')

        # 显示图形
        # plt.show()

        # 球面波
        # 反向追迹至出瞳面
        # PD2 = self.exp_pos / ray_final.d[..., 2]  # 像点到出瞳的距离
        # PD = PD2 + PD1
        z = self.exp_pos
        t = self.exp_pos - ray_final.o[..., 2]
        PD = t / ray_final.d[...,2]
        p = ray_final.o + PD[..., None] * ray_final.d

        phase = ray_final.phase + 2 * torch.pi / lam * PD * self.n_image(lam)

        phase_matrix = torch.zeros((p.shape[0], n_i + 1, n_i + 1)).to(self.device)

        for i in range(p.shape[0]):
            phase_matrix[i] = phase[i] + 2 * torch.pi / lam * torch.sqrt(
                (x_grid - p[..., 0][i]) ** 2 + (y_grid - p[..., 1][i]) ** 2 + (z_rotated-z) ** 2) * self.n_image(lam)

        # # 平面波
        # phase = ray_final.phase + 2 * torch.pi / lam * PD1
        # grid = torch.stack((x_grid, y_grid, torch.zeros_like(x_grid)), dim=2)
        # phase_matrix = torch.zeros((p.shape[0], n_i+1, n_i+1)).to(self.device)
        # print(((grid-p[0]) * ray_final.d[0]).shape)
        # p[..., 2] = 0
        # for i in range(p.shape[0]):
        #     phase_matrix[i] = phase[i] + 2 * torch.pi / lam * (torch.sum((grid-p[i]) * ray_final.d[i], dim=2))

        # 相干叠加
        result = self.coherent_superposition(phase_matrix)

        # # 结束计时
        # end_time = time.time()
        # GPU_time = end_time - start_time
        # print(f"CPU time: {GPU_time:.4f} seconds")
        # 计算最大值
        max_energy = torch.max(result)

        # 归一化矩阵
        result = result / max_energy

        # x = torch.linspace(-d_delta * (n_i // 2), d_delta * (n_i // 2), n_i)  # 计算x和y轴的实际尺寸
        # y = torch.linspace(-d_delta * (n_i // 2), d_delta * (n_i // 2), n_i)

        plt.imshow(result.cpu(),  cmap='jet')

        plt.colorbar(label='Normalized Intensity')

        # # 设置坐标轴标签和标题
        # plt.xlabel('X (um)')
        # plt.ylabel('Y (um)')
        # plt.title('PSF Image')

        # 设置坐标轴只显示最大和最小值
        # 设置坐标轴只显示最大和最小值
        # plt.xticks([x[0], x[-1]], [f"{x[0]:.2f}", f"{x[-1]:.2f}"])
        # plt.yticks([y[0], y[-1]], [f"{y[0]:.2f}", f"{y[-1]:.2f}"])
        plt.show()
        return result, ray_final,d_delta

    def coherent_superposition(self, phase_matrix):
        """
        相干叠加，phase_matrix是多个次波源对应的相位矩阵，其shape为(n,n_i+1,n_i+1)
        """
        # phase_matrix = phase_matrix.cpu()
        # result = torch.zeros(phase_matrix.shape[1:])

        result = torch.sum(torch.exp(1j * phase_matrix), dim=0)
        result = torch.abs(result) ** 2
        # for i in range(phase_matrix.shape[0]):
        #     result += torch.sum(torch.cos(phase_matrix - phase_matrix[i]), dim=0)
        return result

    def distance_2(self, x, y):
        if torch.is_tensor(x) and torch.is_tensor(y):
            x = x
            y = y
        else:
            x, y = (torch.Tensor(np.array(v)) for v in [x, y])
        return x ** 2 + y ** 2

    # def huygens_psf(self, n_p, n_i, wavelength, Hx, Hy, d_delta=0, ):
    #     """
    #     Args:
    #         n_p: 光瞳采样数
    #         n_i: 像面采样数
    #         wavelength:工作波长。单位是nm
    #         Hx,Hy: 归一化视场坐标
    #         d_delta: 像面采样间距。默认由光瞳采样数计算获得
    #     """
    #     # 参数设置
    #     Hx = torch.tensor([Hx]).view(1, 1).to(self.device)
    #     Hy = torch.tensor([Hy]).view(1, 1).to(self.device)
    #     Hx = Hx.reshape(-1, 1)
    #     Hy = Hy.reshape(-1, 1)
    #     ones = torch.ones([1, 1], device=self.device)
    #     angle_x = torch.deg2rad(self.FOV * Hx)  # [rad]
    #     angle_y = torch.deg2rad(self.FOV * Hy)  # [rad]
    #     lam = wavelength * 1e-6  # 波长[mm]
    #     if d_delta == 0:
    #         F_num = self.cal_WFNO(lam, wavelength)  # F/#
    #         print('check working F/#:{} '.format(F_num))
    #         d_delta = (F_num * lam / n_p ** 0.5).item()  # 默认网格间距[mm]
    #         print(d_delta)
    #
    #     # 初始采样光线
    #     d = torch.stack((
    #         torch.tan(angle_x) * ones,
    #         torch.tan(angle_y) * ones,
    #         ones), dim=-1)  # shape(1,3)
    #     d = d / torch.sqrt(torch.tan(angle_x) ** 2 + torch.tan(angle_y) ** 2 + 1)
    #     Px, Py, weight = self.sampling(M=2 / n_p, method='RA')
    #     if self.aimming:
    #         ray = self.find_chief_ray(Hx, Hy)
    #         x = torch.ones_like(Px) * ray.o[..., 0].item()
    #         y = torch.ones_like(Py) * ray.o[..., 1].item()
    #         p_ref = torch.stack((Px * self.aperture, Py * self.aperture), dim=-1)
    #         p_val = torch.stack((x, y), dim=-1)
    #
    #         x, y = self.ray_aimming(p_val, d, p_ref, wavelength, tolerance=1e-6,
    #                                 it_max=1000, is_plot=False)
    #     else:
    #         sag = self.surfaces[0].get_sag(self.surfaces[0].semi_dia, 0.0)
    #         R = self.surfaces[0].semi_dia - torch.tan(angle_y) * sag  # [mm]
    #         R = R.item()
    #         x = Px * R
    #         y = Py * R
    #     o = torch.stack((x, y, torch.zeros_like(x)), dim=2)
    #     distance = torch.sum(o * d, dim=-1)  # [mm]
    #     phase_init = (1 / lam * distance - torch.trunc(1 / lam * distance)) * 2 * torch.pi
    #     ray_in = Ray(o, d, wavelength=lam * 1e6, weight=None, phase=phase_init, device=self.device)
    #
    #     # 带相位的光线追迹
    #     ray_final, weight = self.trace(ray_in, stop_ind=None, is_fixed=True, flag=False, OPD_flag=True)
    #     PD1 = (self.surfaces[-1].thickness - ray_final.o[..., 2]) / ray_final.d[..., 2]  # 最后透镜面到像面的距离
    #     p = ray_final.o + PD1[..., None] * ray_final.d
    #
    #     # p_mean = torch.mean(p, dim=0)  # 计算质心
    #     chief_ray = self.find_chief_ray(Hx, Hy)
    #     p_mean = self.trace2sensor(chief_ray)[0]  # 计算主光线像点
    #     # print('采样的光线数量', p.shape[0])
    #     # print('路径长度', PD1[0])
    #
    #     # 平面像面坐标网格
    #     x = torch.linspace(-d_delta * n_i / 2, d_delta * n_i / 2, n_i + 1).to(self.device)
    #     y = torch.linspace(d_delta * n_i / 2, -d_delta * n_i / 2, n_i + 1).to(self.device)
    #
    #     x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
    #     x_grid = x_grid + p_mean[..., 0]
    #     y_grid = y_grid + p_mean[..., 1]
    #     # r = self.img.c
    #     # # 定义纬度φ和经度θ的采样点数
    #     # phi = torch.linspace(0, np.pi/2, n_i + 1).to(self.device)
    #     # theta = torch.linspace(0, 2 * np.pi, n_i + 1).to(self.device)
    #     #
    #     # # 生成网格点
    #     # Theta, Phi = torch.meshgrid(theta, phi, indexing='xy')
    #     #
    #     # # 使用球坐标系的转换公式
    #     # X = r * torch.sin(Phi) * torch.cos(Theta)
    #     # Y = r * torch.sin(Phi) * torch.sin(Theta)
    #     # Z = r * torch.cos(Phi)
    #     # x_grid = X + p_mean[..., 0]
    #     # y_grid = Y + p_mean[..., 1]
    #     # z_grid = Z + p_mean[..., 2]
    #     #
    #     # # 创建3D图形
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(111, projection='3d')
    #     #
    #     # # 绘制球面上的点
    #     # # ax.scatter(x_grid.cpu().numpy().ravel(), y_grid.cpu().numpy().ravel(), z_grid.cpu().numpy().ravel())
    #     # ax.plot_surface(x_grid.cpu().numpy(),  y_grid.cpu(),  z_grid.cpu(), color='c', edgecolor='k', linewidth=0.5)
    #
    #     # 设置图形标签
    #     # ax.set_xlabel('X axis')
    #     # ax.set_ylabel('Y axis')
    #     # ax.set_zlabel('Z axis')
    #
    #     # 显示图形
    #     # plt.show()
    #
    #     # 球面波
    #     # 反向追迹至出瞳面
    #     PD2 = self.exp_pos / ray_final.d[..., 2]  # 像点到出瞳的距离
    #     PD = PD2 + PD1
    #     p = ray_final.o + PD[..., None] * ray_final.d
    #     z = self.exp_pos
    #     phase = ray_final.phase + 2 * torch.pi / lam * PD * self.n_image(lam)
    #     phase_matrix = torch.zeros((p.shape[0], n_i + 1, n_i + 1)).to(self.device)
    #
    #     for i in range(p.shape[0]):
    #         phase_matrix[i] = phase[i] + 2 * torch.pi / lam * torch.sqrt(
    #             (x_grid - p[..., 0][i]) ** 2 + (y_grid - p[..., 1][i]) ** 2 + z ** 2) * self.n_image(lam)
    #
    #     # # 平面波
    #     # phase = ray_final.phase + 2 * torch.pi / lam * PD1
    #     # grid = torch.stack((x_grid, y_grid, torch.zeros_like(x_grid)), dim=2)
    #     # phase_matrix = torch.zeros((p.shape[0], n_i+1, n_i+1)).to(self.device)
    #     # print(((grid-p[0]) * ray_final.d[0]).shape)
    #     # p[..., 2] = 0
    #     # for i in range(p.shape[0]):
    #     #     phase_matrix[i] = phase[i] + 2 * torch.pi / lam * (torch.sum((grid-p[i]) * ray_final.d[i], dim=2))
    #
    #     # 相干叠加
    #     result = self.coherent_superposition(phase_matrix)
    #
    #     # # 结束计时
    #     # end_time = time.time()
    #     # GPU_time = end_time - start_time
    #     # print(f"CPU time: {GPU_time:.4f} seconds")
    #     # 计算最大值
    #     max_energy = torch.max(result)
    #
    #     # 归一化矩阵
    #     result = result / max_energy
    #
    #     plt.imshow(result.cpu(), cmap='jet')
    #     plt.colorbar(label='Normalized Intensity')
    #     plt.title('PSF Image Plane')
    #     plt.show()
    #     return result, ray_final
    #
    # def coherent_superposition(self, phase_matrix):
    #     """
    #     相干叠加，phase_matrix是多个次波源对应的相位矩阵，其shape为(n,n_i+1,n_i+1)
    #     """
    #     # phase_matrix = phase_matrix.cpu()
    #     # result = torch.zeros(phase_matrix.shape[1:])
    #
    #     result = torch.sum(torch.exp(1j * phase_matrix), dim=0)
    #     result = torch.abs(result) ** 2
    #     # for i in range(phase_matrix.shape[0]):
    #     #     result += torch.sum(torch.cos(phase_matrix - phase_matrix[i]), dim=0)
    #     return result

    def fft_psf_a(self, n_p, n_i, wavelength, Hx, Hy, d_delta=0, ):

        # 参数设置
        Hx = torch.tensor([Hx]).view(1, 1).to(self.device)
        Hy = torch.tensor([Hy]).view(1, 1).to(self.device)
        Hx = Hx.reshape(-1, 1)
        Hy = Hy.reshape(-1, 1)
        ones = torch.ones([1, 1], device=self.device)
        angle_x = torch.deg2rad(self.FOV * Hx)  # [rad]
        angle_y = torch.deg2rad(self.FOV * Hy)  # [rad]
        lam = wavelength * 1e-6  # 波长[mm]
        if d_delta == 0:
            F_num = self.cal_WFNO(lam, wavelength)  # F/#
            print('check working F/#:{} '.format(F_num))

            if n_p == 32:
                d_delta = (F_num * lam * (n_p - 2) / 2 / n_p)
            else:
                d_delta = ((F_num * lam * (n_p - 2) / 2 / n_p) * ((32 / n_p) ** 0.5))

        if n_p == 32:
            n_p = n_p
        else:
            m = (np.log(n_p/32))/(np.log(2))
            n_p = 32 * ((2 ** 0.5) ** m)
            n_p = int(n_p)
        print(d_delta)

        # 初始采样光线
        d = torch.stack((
            torch.tan(angle_x) * ones,
            torch.tan(angle_y) * ones,
            ones), dim=-1)  # shape(1,3)
        d = d / torch.sqrt(torch.tan(angle_x) ** 2 + torch.tan(angle_y) ** 2 + 1)
        Px, Py, weight = self.sampling(M=n_p+1, method='grid')
        if self.aimming:
            ray = self.find_chief_ray(Hx, Hy)
            x = torch.ones_like(Px) * ray.o[..., 0].item()
            y = torch.ones_like(Py) * ray.o[..., 1].item()
            p_ref = torch.stack((Px * self.aperture, Py * self.aperture), dim=-1)
            p_val = torch.stack((x, y), dim=-1)

            x, y = self.ray_aimming(p_val, d, p_ref, wavelength, tolerance=1e-6,
                                    it_max=1000, is_plot=False)
        else:
            sag = self.surfaces[0].get_sag(self.surfaces[0].semi_dia, 0.0)
            R = self.surfaces[0].semi_dia - torch.tan(angle_y) * sag  # [mm]
            R = R.item()
            x = Px * R
            y = Py * R
        o = torch.stack((x, y, torch.zeros_like(x)), dim=2)
        distance = torch.sum(o * d, dim=-1)  # [mm]
        phase_init = (1 / lam * distance - torch.trunc(1 / lam * distance)) * 2 * torch.pi
        ray_in = Ray(o, d, wavelength=lam * 1e6, weight=None, phase=phase_init, device=self.device)

        # 带相位的光线追迹
        p, ray_final = self.trace_eyesensor(ray_in, ignore_invalid=False, is_fixed=True, flag=False)

        with torch.no_grad():
            chief_ray = self.find_chief_ray(Hx, Hy)
            m, c_ray = self.trace_eyesensor(chief_ray)
            p_mean = m  # 计算主光线像点
            d_mean = c_ray.d
            r = -(self.exp_pos-c_ray.o[...,2]) / d_mean[..., 2]  # 出瞳参考球面的曲率半径
        PD2 = -(r-c_ray.o[...,2]+ray_final.o[...,2]) / ray_final.d[..., 2]  # 像点到参考球面的虚拟面的距离
        p = p + ray_final.d * PD2[..., None] - p_mean
        p[..., 2] = 0
        xn, yn, zn = (p[..., i].clone() for i in range(3))
        dx, dy, dz = (ray_final.d[..., i].clone() for i in range(3))
        B = dz - 1 / r * (dx * xn + dy * yn)
        H = 1 / r * (xn ** 2 + yn ** 2)
        temp = B ** 2 - 1 / r * H
        delta1 = B - torch.sqrt(temp)
        delta2 = B + torch.sqrt(temp)
        delta = torch.where(torch.abs(delta1) < torch.abs(delta2), delta1, delta2)
        PD3 = delta * r  # 虚拟面到参考球面的距离
        # p = p + PD3[..., None] * ray_final.d + p_mean  # 参考球面上的点
        PD = PD3 + PD2

        phase = ray_final.phase + 2 * torch.pi / lam * PD * self.n_image(lam)
        phase1 = c_ray.phase - 2 * torch.pi / lam * r * self.n_image(lam)

        # phase = (phase - phase1)


        # # p_mean = torch.mean(p, dim=0)  # 计算质心
        # chief_ray = self.find_chief_ray(Hx, Hy)
        # ray_final_c, weight = self.trace(chief_ray, stop_ind=None, is_fixed=True, flag=False, OPD_flag=True)
        # PD1c = (self.surfaces[-1].thickness - ray_final_c.o[..., 2]) / ray_final_c.d[..., 2]
        # p_mean = self.trace2sensor(chief_ray)[0]  # 计算主光线像点
        # # print('采样的光线数量', p.shape[0])
        # # print('路径长度', PD1[0])
        #
        #
        # PD2 = self.exp_pos / ray_final.d[..., 2]  # 像点到出瞳的距离
        # PD = PD2 + PD1
        # p = ray_final.o + PD[..., None] * ray_final.d
        # PD2c = self.exp_pos / ray_final_c.d[..., 2]  # 像点到出瞳的距离
        # PD_c = PD1c + PD2c
        # phase = ray_final.phase + 2 * torch.pi / lam * PD1
        # phase1= 2 * torch.pi / lam * PD2c
        # phase=phase+phase1


        # 计算复数表示的光场
        # k = 2 * torch.pi / lam
        # complex_field = torch.exp(1j * phase)
        # z = self.exp_pos
        # print(phase.shape)
        # num_rays = n_p*n_p

        # a = torch.arange(2/(n_p+1)/ 2, 1, 2/(n_p+1))
        # y = torch.cat([-torch.flip(a, dims=[0]), a], dim=0)
        y = torch.linspace(-1,1,n_p+1,device=self.device)

        x = y
        x, y = torch.meshgrid(x, x, indexing='xy')
        x = x.flatten()
        y = y.flatten()
        R = torch.sqrt(x ** 2 + y ** 2)


        pupils = []

        #dtype=torch.complex128
        P = torch.zeros_like(x,dtype=torch.complex128).to(self.device)

        phase = phase.flatten()

        P[R <= 1] = torch.exp(1j * phase)


        # P = torch.reshape(P, (n_p, n_p)).numpy()
        P = P.cpu().reshape(n_p+1, n_p+1).numpy()

        pupils.append(P)
        pupils = np.array(pupils)
        plt.imshow((pupils.reshape(181,181)), cmap='jet')
        plt.colorbar()
        plt.title('phase')
        plt.show()


        return pupils,d_delta

    def zernike_polynomial_complex(self, n, m, rho, theta):
        """
        计算在给定极坐标网格上的【归一化】复数泽尼克多项式 Z_n^m。

        参数:
            n, m (int): Zernike 阶数和频率。
            rho, theta (np.ndarray): 归一化极坐标网格。

        返回:
            np.ndarray: 复数泽尼克多项式的值。
        """
        if (n - abs(m)) % 2 != 0:
            return np.zeros_like(rho, dtype=np.complex128)
        
        # 径向部分 R_n^|m|(rho)
        p = (n - abs(m)) // 2
        alpha = abs(m)
        jacobi_poly = eval_jacobi(p, alpha, 0, 1 - 2 * rho**2)
        R_nm = ((-1)**p) * (rho**abs(m)) * jacobi_poly
        
        # 归一化常数
        norm_factor = np.sqrt(2 * (n + 1) / (1 + (m == 0)))
        
        # 角向部分
        Z_nm = norm_factor * R_nm * np.exp(1j * m * theta)
        return Z_nm

    def project_on_zernike(self, pupil_func, n_max=4):
        """
        将光瞳函数投影到复数泽尼克基上以获得beta系数。

        参数:
            pupil_func (np.ndarray): 2D复数光瞳函数。
            n_max (int): 要计算的最高Zernike阶数。

        返回:
            dict: beta系数的字典，键为 (n, m)。
        """
        n_p = pupil_func.shape[0]
        y = np.linspace(-1, 1, n_p)
        x = np.linspace(-1, 1, n_p)
        X, Y = np.meshgrid(x, y)
        RHO = np.sqrt(X**2 + Y**2)
        THETA = np.arctan2(Y, X)

        pupil_func[RHO > 1] = 0
        dA = (X[0, 1] - X[0, 0]) * (Y[1, 0] - Y[0, 0]) # 面积元

        beta_coeffs = {}
        for n in range(n_max + 1):
            for m in range(-n, n + 1, 2):
                Z_nm_conj = np.conj(self.zernike_polynomial_complex(n, m, RHO, THETA))
                Z_nm_conj[RHO > 1] = 0
                
                # 计算内积 <P, Z>
                inner_product = np.sum(pupil_func * Z_nm_conj) * dA
                
                # 计算Zernike范数的平方 ||Z||^2
                norm_sq = np.sum(np.abs(Z_nm_conj)**2) * dA # 复用Z_nm_conj来计算范数
                # norm_sq = np.pi / (n + 1)
                # print(f"For (n={n}, m={m}), norm_sq = {norm_sq}")

                # 计算Beta系数
                if norm_sq > 1e-12: # 避免除以一个非常小的数
                    beta = inner_product / norm_sq
                else:
                    beta = 0.0
                
                beta = inner_product / norm_sq
                if np.abs(beta) > 1e-4: # 仅存储非零系数
                    beta_coeffs[(n, m)] = beta
        
        print('\n===== Calculated beta_coeffs Map =====')
        print('------------------------------------')
        print('| Key (n,m) | Beta Value (real + imag*i) |')
        print('------------------------------------')
        for key, val in beta_coeffs.items():
            print(f'| {str(key):<11} | {val.real:10.6f} {val.imag:+10.6f}i |')
        print('------------------------------------\n')

        return beta_coeffs
    
    def calculate_Vnm_janssen(self, n, m, r, f, max_l=25): #矢量化版本
        """
        使用Janssen的贝塞尔级数解计算 V_n^m(r, f)。
        这是 MATLAB/calculate_Vnm_janssen.m 的 Python 翻译版。
        此版本经过矢量化，可以接受一个NumPy数组作为半径r的输入。
        """
        r = np.asarray(r)  # 确保r是numpy数组
        v = 2 * np.pi * r
        abs_m = abs(m)

        if (n - abs_m) % 2 != 0 or n < abs_m:
            raise ValueError(f'无效的Zernike指数 (n={n}, m={m})')

        p = (n - abs_m) // 2
        q = (n + abs_m) // 2

        total_V = np.zeros_like(r, dtype=np.complex128)
        for l in range(1, max_l + 1):
            inner_sum = np.zeros_like(r, dtype=np.complex128)
            for j in range(p + 1):
                # 使用gammaln避免大数的阶乘计算
                log_nck1 = gammaln(abs_m + j + l) - gammaln(l) - gammaln(abs_m + j + 1)
                log_nck2 = gammaln(j + l) - gammaln(l) - gammaln(j + 1)
                log_nck3 = gammaln(l) - gammaln(p - j + 1) - gammaln(l - p + j + 1)
                log_nck4 = gammaln(q + l + j + 1) - gammaln(l + 1) - gammaln(q + j + 1)

                v_lj = (-1)**p * (abs_m + l + 2*j) * np.exp(log_nck1 + log_nck2 + log_nck3 - log_nck4)
                
                bessel_idx = abs_m + l + 2*j
                
                # 矢量化处理 bessel_term
                bessel_term = np.zeros_like(v, dtype=np.float64)
                mask = v > 1e-9
                bessel_term[mask] = jv(bessel_idx, v[mask]) / (l * v[mask]**l)
                
                inner_sum += v_lj * bessel_term

            term_l = (-2j * f)**(l - 1) * inner_sum
            total_V += term_l
            
            # 矢量化收敛检查
            if l > 5:
                # 仅在所有元素都收敛时才中断，这是一个简化的策略
                # 更复杂的策略可以只更新未收敛的元素
                if np.all(np.abs(term_l) < 1e-9 * np.abs(total_V)):
                    break
        
        epsilon_m = -1.0 # Python中通常使用1.0，与MATLAB的定义可能不同
        V = epsilon_m * np.exp(1j * f) * total_V
        return V

    # def calculate_Vnm_janssen(self, n, m, r, f, max_l=25): #标量版本
    #     """
    #     使用Janssen的贝塞尔级数解计算 V_n^m(r, f)。
    #     这是 MATLAB/calculate_Vnm_janssen.m 的 Python 翻译版。
    #     """
    #     v = 2 * np.pi * r
    #     abs_m = abs(m)

    #     if (n - abs_m) % 2 != 0 or n < abs_m:
    #         raise ValueError(f'无效的Zernike指数 (n={n}, m={m})')

    #     p = (n - abs_m) // 2
    #     q = (n + abs_m) // 2

    #     total_V = 0
    #     for l in range(1, max_l + 1):
    #         inner_sum = 0
    #         for j in range(p + 1):
    #             # 使用gammaln避免大数的阶乘计算
    #             log_nck1 = gammaln(abs_m + j + l) - gammaln(l) - gammaln(abs_m + j + 1)
    #             log_nck2 = gammaln(j + l) - gammaln(l) - gammaln(j + 1)
    #             log_nck3 = gammaln(l) - gammaln(p - j + 1) - gammaln(l - p + j) if p >= j else -np.inf
    #             log_nck4 = gammaln(q + l + j + 1) - gammaln(l + 1) - gammaln(q + j + 1)
                
    #             if np.isinf(log_nck3): continue

    #             v_lj = (-1)**p * (abs_m + l + 2*j) * np.exp(log_nck1 + log_nck2 + log_nck3 - log_nck4)
                
    #             bessel_idx = abs_m + l + 2*j
    #             bessel_term = jv(bessel_idx, v) / (l * v**l) if v > 1e-9 else 0
                
    #             inner_sum += v_lj * bessel_term

    #         term_l = (-2j * f)**(l - 1) * inner_sum
    #         total_V += term_l
            
    #         if l > 5 and abs(term_l) < 1e-9 * abs(total_V):
    #             break
        
    #     epsilon_m = -1.0 # Python中通常使用1.0，与MATLAB的定义可能不同
    #     V = epsilon_m * np.exp(1j * f) * total_V
    #     return V

    def calculate_enz_psf(self, beta_coeffs, n_i, d_delta, lam_mm, f_num, defocus=0, max_l=25):
        """
        使用beta系数计算ENZ PSF。
        """
        # 确保所有输入都是Python原生类型，而不是Tensor
        if isinstance(lam_mm, torch.Tensor):
            lam_mm = lam_mm.item()
        if isinstance(f_num, torch.Tensor):
            f_num = f_num.item()
        # 1. 创建图像平面网格 (物理单位: um)
        xy_max = d_delta * (n_i // 2)
        x_um = np.linspace(-xy_max, xy_max, n_i)
        y_um = np.linspace(-xy_max, xy_max, n_i)
        X_um, Y_um = np.meshgrid(x_um, y_um)

        # 2. 转换为无量纲坐标 (单位: lambda/NA)
        # NA ≈ 1 / (2 * F_num)
        # scaling_factor = lam_mm * 1e3 / NA = lam_mm * 1e3 * 2 * f_num
        scaling_factor = lam_mm * 2 * f_num * 1e3 # um per (lambda/NA) unit
        R_unitless = np.sqrt(X_um**2 + Y_um**2) / scaling_factor
        PHI_image = np.arctan2(Y_um, X_um)

        # 3. 计算复数场
        field = np.zeros_like(R_unitless, dtype=np.complex128)
        for (n, m), beta in beta_coeffs.items():
            # V_nm = np.zeros_like(R_unitless, dtype=np.complex128)
            # for idx in np.ndindex(R_unitless.shape):
            #     V_nm[idx] = self.calculate_Vnm_janssen(n, m, R_unitless[idx], defocus)

            # 直接调用矢量化函数，无需循环
            V_nm = self.calculate_Vnm_janssen(n, m, R_unitless, defocus, max_l)
            
            field += 2 * beta * (1j)**abs(m) * V_nm * np.exp(1j * m * PHI_image)
        
        psf = np.abs(field)**2
        psf = psf / np.max(psf) if np.max(psf) > 0 else psf
        return psf

    def fft_psf_i(
        self,
        n_p,
        n_i,
        wavelength,
        d_delta=0,
        Hx=0.0,
        Hy=0.0,
        legacy_pupil_phase=False,
        zernike_n_max=5,
    ):
        """
        Build the complex pupil for FFT PSF computation.

        Args:
            n_p: Requested pupil sampling count. Unitless grid count.
            n_i: Image-plane FFT sampling count. Unitless grid count.
            wavelength: Center wavelength in nm as a torch tensor or scalar.
            d_delta: Image-plane sample pitch in mm. If 0, computed from F/#.
            Hx, Hy: Normalized field coordinates. Physical field angle is
                Hx/Hy multiplied by self.FOV in degree.
            legacy_pupil_phase: If True, use the historical image-center
                phase reference. If False, use the current field's chief
                reference point to remove reference piston/tilt before FFT.
            zernike_n_max: Non-negative highest order for the independent
                real-Zernike wavefront OPD diagnostic. Default is 5.

        Returns:
            tuple: (pupils, d_delta). pupils is a list of complex NumPy arrays
            with shape [n_p, n_p]. Coordinates are normalized pupil coordinates.

        GPU/autograd:
            Ray tracing is performed on self.device. The returned NumPy pupil is
            detached for FFT post-processing and does not preserve autograd.
        """
        # 计算光学系统的光瞳函数
        # 该函数通过光线追迹计算波前像差，最终构建出光瞳函数。
        # 
        # 参数:
        #   Hx, Hy: 归一化视场坐标（视场角/FOV，范围约[-2, 2]对应[-20°, 20°]）
        #   这些参数用于计算视场依赖的像差补偿

        # 参数设置 o——主光线起点，d——主光线方向，wavelength——波长
        ## 像面纵向偏移恒为 0：像面就在 Excel 配置的位置（末面 thickness），不做任何
        ## 经验离焦或场曲补偿。有限物距下的离焦必须由真实几何在追迹中自然产生，
        ## 由波前的 defocus 项体现，而不是靠平移像面把它凑掉。
        ##
        ## Task 5（2026-07-30）删除的旧实现：
        ##   system_focal_length = 19.3            # 调参得到的"等效焦距"，Excel 实际后焦 16.820132
        ##   defocus_shift = -(19.3**2) / object_distance
        ##   field_curvature = 0.003 * (field_radius - 1.0)**2   # 仅对 1000mm 且视场半径>1 生效
        ## 两者都是为了让 1000mm 的 PSF 更接近参考值而拟合出来的，与项目规则冲突。
        ##
        ## 2026-07-31：有限物距的离焦已由下方的发散束发射真实产生（物点 -> 瞳采样点的
        ## 几何光程随瞳坐标变化）。1000mm 的 a(2,0) 由 -0.1607（无离焦）变为 +1.1389 波，
        ## 与 Zemax 的 PSF 二阶矩半径吻合到 1%。不要恢复上面的经验项。
        defocus_shift = 0.0

        # 主光线设置：沿局部光轴方向
        # 视场角通过Excel中CB表面的tilt_x/tilt_y来控制
        # CB表面旋转坐标系，但主光线在局部坐标系中仍沿z轴
        print(f"视场({Hx:.2f},{Hy:.2f}), 主光线在局部坐标系中沿z轴")
        ## 主光线代表理想参考光线。计算波前像差时它作为基准 (Reference)，所有采样光线的
        ## 光程都与它比较得到 OPD 和相位差；它定义了参考球面的中心。
        ## 构造与反向追迹统一在 _chief_ray_at_first_surface 内，cal_WFNO 与
        ## single_ray_trace 的 aimming 分支共用同一定义，避免多份硬编码。
        ray_rel = self._chief_ray_at_first_surface(wavelength, defocus_shift)
        ## 像方参考光线：像面中心指向 +z，用于出瞳参考球面半径 r 与参考点。
        ## 与 _chief_ray_at_first_surface 内反向追迹的起点同源，方向相反。
        ray_chief = Ray(torch.tensor((0.0, 0.0, float(defocus_shift))).to(self.device),
                        torch.tensor((0.0, 0.0, 1.0)).to(self.device),
                        wavelength=wavelength, device=self.device)
        print(f"Ray position at first surface x: {ray_rel.o[...,0]}")
        print(f"Ray position at first surface y: {ray_rel.o[...,1]}")
        print(f"Ray position at first surface z: {ray_rel.o[...,2]}")



        lam = wavelength * 1e-6  # 波长[mm]
        if d_delta == 0:
            F_num = self.cal_WFNO(lam, wavelength)  # F/#
            print('check working F/#:{} '.format(F_num))

            if n_p == 32:
                d_delta = (F_num * lam * (n_p - 2) / 2 / n_p)
            else:
                d_delta = ((F_num * lam * (n_p - 2) / 2 / n_p) * ((32 / n_p) ** 0.5))

        if n_p == 32:
            n_p = n_p
        else:
            m = (np.log(n_p/32))/(np.log(2))
            n_p = 32 * ((2 ** 0.5) ** m)
            n_p = int(n_p)
        print(f"Sampling interval: {d_delta}")

        # 初始采样光线
        d = ray_rel.d
        ## 有限物距（obj.thickness > 0）：从主光线反向定出物点，瞳采样光线各自从物点发散。
        ## obj.thickness == 0 是本代码库的无穷远标记（load_file 把 'Infinity' 映射为 0），
        ## 此时 p_obj 为 None，走原来的平行束路径，逐位不变。
        object_thickness = float(self.obj.thickness)
        p_obj = None
        if object_thickness > 0:
            ## ray_rel.o 在虚拟面 z=0 上，沿 -d 退到 z = -object_thickness 即物面。
            t_back = object_thickness / ray_rel.d[..., 2]
            p_obj = ray_rel.o - t_back[..., None] * ray_rel.d
            print(f"有限物距 {object_thickness} mm，物点 z={p_obj[..., 2].reshape(-1)[0].item():.6f} mm，发射发散束")
        Px, Py, weight = self.sampling(M=n_p, method='grid') #根据采样点数n_p在归一化单位圆上生成采样点坐标
        # Px = 1
        # Py = 0
        # Px = torch.tensor([Px]).view(1, 1).to(self.device)
        # Py = torch.tensor([Py]).view(1, 1).to(self.device)
        #aperture = 1.954135753091987E+000
        aperture = self.stop_semi_diameter()

        x = torch.ones_like(Px) * ray_rel.o[..., 0].item() #初始光线的X坐标
        y = torch.ones_like(Py) * ray_rel.o[..., 1].item() #初始光线的Y坐标
        p_ref = torch.stack((Px * aperture, Py * aperture), dim=-1) #瞳孔平面上的目标参考点
        p_val = torch.stack((x, y), dim=-1) #所有初始光线的初始位置

        x, y = self.ray_aimming(p_val, d, p_ref, wavelength, tolerance=1e-6,
                                it_max=1000, is_plot=False, p_obj=p_obj) #优化后的精确起始坐标，光线瞄准只调整光线的起始位置o
        o = torch.stack((x, y, torch.zeros_like(x)), dim=2) #最终的三维起始点坐标。

        if p_obj is None:
            ## 无穷远：平行束，初始光程取平面波在传播方向上的投影。
            d_in = d
            distance = torch.sum(o * d, dim=-1)  # [mm], continuous initial optical path
        else:
            ## 有限物距：每根光线方向由自身发射点与物点决定；初始光程是物点到发射点的
            ## 几何距离（物方为空气 n=1，几何距离即光程）。这一项随瞳坐标变化，正是
            ## 有限物距离焦的物理来源，平面波投影无法表达。
            d_in = self._diverging_direction(x, y, p_obj)
            distance = torch.linalg.vector_norm(o - p_obj, dim=-1)

        # Keep the traced OPD continuous for Zernike fitting.  A per-ray
        # integer-wave truncation leaves exp(1j*phase) unchanged, but tears
        # the continuous OPD into a sawtooth.  Only subtract a single piston
        # reference shared by every sampled pupil ray.
        ref_o = None
        ref_d = None
        ref_distance_source = "nearest_sampled_pupil_ray"
        if legacy_pupil_phase:
            center_index = torch.argmin(Px * Px + Py * Py)
            ref_distance = distance[center_index]
        else:
            zero = torch.zeros((1, 1), dtype=d.dtype, device=self.device)
            ref_start = torch.stack(
                (
                    torch.ones_like(zero) * ray_rel.o[..., 0].item(),
                    torch.ones_like(zero) * ray_rel.o[..., 1].item(),
                ),
                dim=-1,
            )
            ref_target = torch.zeros_like(ref_start)
            ref_x, ref_y = self.ray_aimming(
                ref_start,
                d,
                ref_target,
                wavelength,
                tolerance=1e-6,
                it_max=1000,
                is_plot=False,
                p_obj=p_obj,
            )
            ref_o = torch.stack((ref_x, ref_y, torch.zeros_like(ref_x)), dim=2)
            ## 参考光线必须有自己的方向：有限物距下 d_in 是逐条的 [n_rays,1,3]，
            ## 若把 [1,1,3] 的 ref_o 和它相乘会被广播成 n_rays 条光线。
            if p_obj is None:
                ref_d = d
                ref_distance = torch.sum(ref_o * d, dim=-1)
            else:
                ref_d = self._diverging_direction(ref_x, ref_y, p_obj)
                ref_distance = torch.linalg.vector_norm(ref_o - p_obj, dim=-1)
            ref_distance_source = "chief_reference_ray"

        phase_init = (distance - ref_distance) / lam * 2 * torch.pi
        ray_in = Ray(o, d_in, wavelength=lam * 1e6, weight=None, phase=phase_init, device=self.device) #初始入射光线，含初始相位，位置和方向

        # 带相位的光线追迹
        # ray,_= self.trace(ray_in,stop_ind= self.aperture_ind,is_fixed=True, flag=False, OPD_flag=False)
        # p = ray.o
        sensor_intersection, ray_final = self.trace_eyesensor(
            ray_in, ignore_invalid=False, is_fixed=True, flag=False
        ) #最后一个表面上的光线

        with torch.no_grad():
            c_ray = Ray(ray_chief.o, ray_chief.d, wavelength=wavelength, device=self.device)
            d_mean = c_ray.d
            exp_pos = self.find_exp()
            r = -(exp_pos-c_ray.o[...,2]) / d_mean[..., 2]  # 出瞳参考球面的曲率半径
            # r = (c_ray.o[...,1]) / d_mean[..., 1]

        legacy_reference_point = ray_chief.o.reshape(1, 1, 3)
        reference_point = legacy_reference_point
        reference_mode = "legacy_image_center"
        if not legacy_pupil_phase:
            ref_phase_init = torch.zeros_like(ref_distance)
            ref_ray = Ray(ref_o, ref_d, wavelength=lam * 1e6, weight=None, phase=ref_phase_init, device=self.device)
            reference_point, _ = self.trace_eyesensor(ref_ray, ignore_invalid=False, is_fixed=True, flag=False)
            reference_point = reference_point.reshape(1, 1, 3)
            reference_mode = "chief_reference_point"

        lateral_residual = torch.linalg.vector_norm(reference_point[..., :2])
        lateral_residual_mm = float(lateral_residual.detach().cpu().reshape(-1)[0].item())
        reference_point_z_mm = float(reference_point[..., 2].detach().cpu().reshape(-1)[0].item())
        if lateral_residual_mm > _PUPIL_REFERENCE_LATERAL_TOLERANCE_MM:
            raise RuntimeError(
                "Pupil reference point violated the coaxial local-frame lateral invariant: "
                f"residual={lateral_residual_mm:.12g} mm, "
                f"tolerance={_PUPIL_REFERENCE_LATERAL_TOLERANCE_MM:.12g} mm"
            )

        PD2 = -(r-c_ray.o[...,2]+ray_final.o[...,2]) / ray_final.d[..., 2]  # 像点到参考球面的虚拟面的距离
        projected_p = sensor_intersection + ray_final.d * PD2[..., None]
        dx, dy, dz = (ray_final.d[..., i].clone() for i in range(3))

        def phase_for_reference(ref_point):
            pupil_ref = projected_p - ref_point
            pupil_ref[..., 2] = 0
            xn, yn, zn = (pupil_ref[..., i].clone() for i in range(3))
            B = dz - 1 / r * (dx * xn + dy * yn)
            H = 1 / r * (xn ** 2 + yn ** 2)
            temp = B ** 2 - 1 / r * H
            delta1 = B - torch.sqrt(temp)
            delta2 = B + torch.sqrt(temp)
            delta = torch.where(torch.abs(delta1) < torch.abs(delta2), delta1, delta2)
            PD3 = delta * r  # 虚拟面到参考球面的距离
            PD = PD3 + PD2 #每根光线在真实像面落点到参考球面的距离
            return ray_final.phase + 2 * torch.pi / lam * PD * self.n_image(lam)

        legacy_phase = phase_for_reference(legacy_reference_point)
        phase = legacy_phase if legacy_pupil_phase else phase_for_reference(reference_point)
        # phase1 = c_ray.phase - 2 * torch.pi / lam * r * self.n_image(lam)

        # phase = (phase - phase1)


        # # p_mean = torch.mean(p, dim=0)  # 计算质心
        # chief_ray = self.find_chief_ray(Hx, Hy)
        # ray_final_c, weight = self.trace(chief_ray, stop_ind=None, is_fixed=True, flag=False, OPD_flag=True)
        # PD1c = (self.surfaces[-1].thickness - ray_final_c.o[..., 2]) / ray_final_c.d[..., 2]
        # p_mean = self.trace2sensor(chief_ray)[0]  # 计算主光线像点
        # # print('采样的光线数量', p.shape[0])
        # # print('路径长度', PD1[0])
        #
        #
        # PD2 = self.exp_pos / ray_final.d[..., 2]  # 像点到出瞳的距离
        # PD = PD2 + PD1
        # p = ray_final.o + PD[..., None] * ray_final.d
        # PD2c = self.exp_pos / ray_final_c.d[..., 2]  # 像点到出瞳的距离
        # PD_c = PD1c + PD2c
        # phase = ray_final.phase + 2 * torch.pi / lam * PD1
        # phase1= 2 * torch.pi / lam * PD2c
        # phase=phase+phase1


        # 计算复数表示的光场
        # k = 2 * torch.pi / lam
        # complex_field = torch.exp(1j * phase)
        # z = self.exp_pos
        # print(phase.shape)
        # num_rays = n_p*n_p

        # a = torch.arange(2/(n_p+1)/ 2, 1, 2/(n_p+1))
        # y = torch.cat([-torch.flip(a, dims=[0]), a], dim=0)
        y = torch.linspace(-1,1,n_p,device=self.device)

        x = y
        x, y = torch.meshgrid(x, x, indexing='xy')
        x = x.flatten()
        y = y.flatten()
        R = torch.sqrt(x ** 2 + y ** 2)


        pupils = []

        #dtype=torch.complex128
        P = torch.zeros_like(x,dtype=torch.complex128).to(self.device)

        phase = phase.flatten()
        legacy_phase = legacy_phase.flatten()

        pupil_mask = R <= 1
        if phase.numel() != int(pupil_mask.sum().detach().cpu().item()):
            raise ValueError(
                "Traced wavefront sample count does not match the FFT pupil mask: "
                f"phase={phase.numel()}, mask={int(pupil_mask.sum().detach().cpu().item())}"
            )
        phase_grid_flat = torch.zeros_like(x)
        phase_grid_flat[pupil_mask] = phase
        phase_grid = phase_grid_flat.reshape(n_p, n_p)
        wavelength_mm = float(lam.detach().cpu().item()) if torch.is_tensor(lam) else float(lam)
        wavefront_opd_mm = (phase_grid.detach().cpu().numpy() * wavelength_mm) / (2.0 * np.pi)
        zernike_coefficients, zernike_metrics = fit_wavefront_zernike(
            wavefront_opd_mm,
            x.reshape(n_p, n_p).detach().cpu().numpy(),
            y.reshape(n_p, n_p).detach().cpu().numpy(),
            pupil_mask.reshape(n_p, n_p).detach().cpu().numpy(),
            wavelength_mm,
            n_max=zernike_n_max,
        )
        raw_P = torch.zeros_like(x, dtype=torch.complex128).to(self.device)
        raw_P[pupil_mask] = torch.exp(1j * legacy_phase)
        P[pupil_mask] = torch.exp(1j * phase)

        P = P.reshape(n_p, n_p)
        raw_P = raw_P.reshape(n_p, n_p)
        x_grid = x.reshape(n_p, n_p)
        y_grid = y.reshape(n_p, n_p)
        pupil_mask = pupil_mask.reshape(n_p, n_p)

        pupil_piston = _complex_circular_mean_angle(P[pupil_mask])
        P = torch.where(pupil_mask, P * torch.exp(-1j * pupil_piston), torch.zeros_like(P))

        self.last_pupil_tilt_metrics = {
            "pupil_reference_mode": reference_mode,
            "legacy_pupil_phase": bool(legacy_pupil_phase),
            "pupil_reference_point_x_mm": float(reference_point[..., 0].detach().cpu().reshape(-1)[0].item()),
            "pupil_reference_point_y_mm": float(reference_point[..., 1].detach().cpu().reshape(-1)[0].item()),
            "pupil_reference_point_z_mm": reference_point_z_mm,
            "pupil_reference_point_lateral_residual_mm": lateral_residual_mm,
            "pupil_reference_invariant": "coaxial_local_frame_vertex_lateral",
            "pupil_reference_sphere_radius_mm": float(r.detach().cpu().reshape(-1)[0].item()),
            "pupil_reference_sphere_mode": "axial_vertex",
        }
        self.last_pupil_tilt_metrics.update(
            summarize_complex_pupil_phase(raw_P, x_grid, y_grid, pupil_mask, "raw_pupil")
        )
        self.last_pupil_tilt_metrics.update(
            summarize_complex_pupil_phase(P, x_grid, y_grid, pupil_mask, "reference_pupil")
        )
        self.last_wavefront_zernike_metrics = {
            **zernike_metrics,
            "zernike_reference_mode": reference_mode,
            "zernike_phase_source": "fft_reference_phase_before_complex_pupil_piston_correction",
            "zernike_phase_unwrap_policy": "reference_ray_constant_subtraction",
            "zernike_phase_reference_distance_mm": float(ref_distance.detach().cpu().reshape(-1)[0].item()),
            "zernike_phase_reference_distance_source": ref_distance_source,
            "pupil_reference_point_lateral_residual_mm": lateral_residual_mm,
            "pupil_reference_point_z_mm": reference_point_z_mm,
            "pupil_reference_invariant": "coaxial_local_frame_vertex_lateral",
            "pupil_reference_sphere_radius_mm": float(r.detach().cpu().reshape(-1)[0].item()),
            "pupil_reference_sphere_mode": "axial_vertex",
        }
        self.last_wavefront_zernike_coefficients = zernike_coefficients
        print(
            "Pupil reference phase: "
            f"mode={reference_mode}, legacy={legacy_pupil_phase}, "
            f"raw_tilt=({self.last_pupil_tilt_metrics['raw_pupil_tilt_x_rad_per_norm']:.6g}, "
            f"{self.last_pupil_tilt_metrics['raw_pupil_tilt_y_rad_per_norm']:.6g}) rad/norm, "
            f"reference_tilt=({self.last_pupil_tilt_metrics['reference_pupil_tilt_x_rad_per_norm']:.6g}, "
            f"{self.last_pupil_tilt_metrics['reference_pupil_tilt_y_rad_per_norm']:.6g}) rad/norm"
        )
        print(
            "Wavefront Zernike fit: "
            f"n_max={zernike_metrics['zernike_n_max']}, "
            f"modes={zernike_metrics['zernike_mode_count']}, "
            f"residual_rms={zernike_metrics['zernike_residual_rms_um']:.6g} um"
        )
        ''' 理论上需要使用参考球面上实际波前的相位差，这里省略了理想相位。
            因为参考球面上是等相位的，所以会引入一个相位偏移,
            经过FFT和模方后相位偏移会消失，不会影响最终的PSF和MTF计算。'''
        # P[R <= 1] = phase.to(torch.complex128)


        # P = torch.reshape(P, (n_p, n_p)).numpy()
        P = P.cpu().numpy()

        pupils.append(P)
        # pupils = np.array(pupils)
        # plt.imshow((np.real(pupils).reshape(182,182)), cmap='jet')
        # plt.colorbar()
        # plt.title('phase')
        # plt.show()


        return pupils,d_delta
        # return p


    def _compute_psf(self,pupils,n_i,d_delta,f_stop, display_size_um=None,defocus_f=0, methods='fft', show_plot=False):
        """
        输入光瞳函数,计算出系统的PSF和MTF,
        最后将结果通过图表可视化展示出来。
        增加了旋转偏移补偿，确保PSF图像中心稳定。

        Args:
            methods (str): 'fft' 或 'enz'，选择计算PSF的方法。

        Returns:
            np.ndarray: The computed PSF as a 2D numpy array.
        """
        if methods.lower() == 'fft':
            print("Calculating PSF using FFT methods.")
            # TODO: add ability to compute polychromatic PSF.
            # Interpolate for each wavelength, then incoherently sum.
            pupils = self._pad_pupils(pupils, n_i) #填充  
            norm_factor = self._get_normalization(pupils)

            psf = []
            for pupil in pupils:
                amp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
                psf.append(amp * np.conj(amp))
            psf_image = np.real(np.sum(psf, axis=0))
            if np.max(psf_image) > 0:
                psf_image = psf_image / np.max(psf_image)

        elif methods.lower() == 'enz': #注意要手动设置离焦参数和max_l
            print("Calculating PSF using Extended Nijboer-Zernike (ENZ) methods.")
            if len(pupils) > 1:
                print("Warning: ENZ methods currently supports monochromatic light. Using the first pupil function.")
            
            pupil = pupils[0] # 假设单色光
            # 将光瞳函数投影到Zernike基上，获取beta系数
            beta_coeffs = self.project_on_zernike(pupil, n_max=5) # n_max可以根据需要调整
            
            # 获取计算ENZ PSF所需的参数
            lam_mm = self.wavelengths_center * 1e-6
            f_num = self.cal_WFNO(lam_mm)
            print(f"F-number : {f_num}")
            
            psf_image = self.calculate_enz_psf(beta_coeffs, n_i, d_delta, lam_mm, f_num, defocus=defocus_f, max_l=max(25, 3 * defocus_f))
            print(f"ENZ PSF calculation complete (using defocus f = {defocus_f}).")
        else:
            raise ValueError("Methods must be 'fft' or 'enz'")

        # 关键修复：保持PSF形状和方向不变，仅调整位置居中
        # 使用整数像素偏移，完全保持原始形状和方向特征
        
        # 计算PSF最大值位置（更稳定的中心检测方法）
        max_pos = np.unravel_index(np.argmax(psf_image), psf_image.shape)
        max_y, max_x = max_pos
        
        # 同时计算质心位置作为备用方法
        y_indices, x_indices = np.mgrid[0:psf_image.shape[0], 0:psf_image.shape[1]]
        total_intensity = np.sum(psf_image)
        
        if total_intensity > 0:
            centroid_x = np.sum(x_indices * psf_image) / total_intensity
            centroid_y = np.sum(y_indices * psf_image) / total_intensity
            
            # 计算图像中心位置
            center_x, center_y = psf_image.shape[1] // 2, psf_image.shape[0] // 2
            
            # 优先使用最大值位置，如果质心和最大值位置差异太大则使用质心
            if abs(max_x - centroid_x) < 5 and abs(max_y - centroid_y) < 5:
                # 使用最大值位置（更稳定）
                current_x, current_y = max_x, max_y
                method = "最大值位置"
            else:
                # 使用质心位置（更精确）
                current_x, current_y = centroid_x, centroid_y
                method = "质心位置"
            
            # 计算需要的平移量（将当前位置移动到图像中心）
            shift_x = int(round(center_x - current_x))
            shift_y = int(round(center_y - current_y))

            ## 亚像素残差诊断。上面的 shift_x/shift_y 是给（已禁用的）np.roll 用的整数量，
            ## 早期版本直接把它们当成"残余偏移"打印，于是真实偏移被 round 吃掉：四例只会
            ## 输出 1.0 或 1.4(=|(±1,±1)|) 两个值，看起来像按视场分组，其实是量化桶边界。
            ## 这里给出未取整的残差，供判断偏移是否有物理来源；不参与任何计算。
            ##
            ## 三种估计并列：全帧质心会被大面积低值背景拉平（≈0），阈值核心质心只统计
            ## 20% 峰值以上的像元，抛物线插值用峰值邻域三点定亚像素极值。
            core_mask = psf_image >= 0.20 * float(np.max(psf_image))
            core_sum = float(np.sum(psf_image[core_mask]))
            if core_sum > 0:
                core_x = float(np.sum(x_indices[core_mask] * psf_image[core_mask]) / core_sum)
                core_y = float(np.sum(y_indices[core_mask] * psf_image[core_mask]) / core_sum)
            else:
                core_x, core_y = float(centroid_x), float(centroid_y)

            def _parabolic_peak(idx, axis_len, values):
                ## 三点抛物线顶点，边界上退化为整数峰位。
                if idx <= 0 or idx >= axis_len - 1:
                    return float(idx)
                y0, y1, y2 = (float(v) for v in values)
                denom = y0 - 2.0 * y1 + y2
                if denom == 0.0:
                    return float(idx)
                return float(idx) + 0.5 * (y0 - y2) / denom

            peak_x = _parabolic_peak(max_x, psf_image.shape[1],
                                     psf_image[max_y, max(max_x - 1, 0):max_x + 2])
            peak_y = _parabolic_peak(max_y, psf_image.shape[0],
                                     psf_image[max(max_y - 1, 0):max_y + 2, max_x])

            self.last_psf_centering_metrics = {
                "psf_center_index_x": int(center_x),
                "psf_center_index_y": int(center_y),
                "psf_peak_index_x": int(max_x),
                "psf_peak_index_y": int(max_y),
                "psf_peak_subpixel_x": peak_x,
                "psf_peak_subpixel_y": peak_y,
                "psf_centroid_full_x": float(centroid_x),
                "psf_centroid_full_y": float(centroid_y),
                "psf_centroid_core_x": core_x,
                "psf_centroid_core_y": core_y,
                "psf_core_threshold_fraction": 0.20,
                "psf_core_pixel_count": int(np.count_nonzero(core_mask)),
                "psf_offset_peak_subpixel_x": float(center_x) - peak_x,
                "psf_offset_peak_subpixel_y": float(center_y) - peak_y,
                "psf_offset_centroid_core_x": float(center_x) - core_x,
                "psf_offset_centroid_core_y": float(center_y) - core_y,
                "psf_offset_centroid_full_x": float(center_x) - float(centroid_x),
                "psf_offset_centroid_full_y": float(center_y) - float(centroid_y),
                "psf_offset_rounded_x": int(shift_x),
                "psf_offset_rounded_y": int(shift_y),
                "psf_roll_applied": False,
            }

            print(f"PSF {method}: ({current_x:.1f}, {current_y:.1f})")
            print(f"Image center: ({center_x}, {center_y})")
            print(f"Calculated offset: ({shift_x}, {shift_y})")
            print(
                "PSF residual offset [px] (unrounded, diagnostic): "
                f"peak=({float(center_x) - peak_x:+.4f}, {float(center_y) - peak_y:+.4f}), "
                f"core_centroid=({float(center_x) - core_x:+.4f}, {float(center_y) - core_y:+.4f}), "
                f"full_centroid=({float(center_x) - float(centroid_x):+.4f}, "
                f"{float(center_y) - float(centroid_y):+.4f})"
            )

            # 迭代居中算法：对于大偏移进行多次调整
            max_iterations = 0  # disabled: reference phase, not np.roll, must set PSF position
            threshold = 1  # 像素阈值
            
            for iteration in range(max_iterations):
                if abs(shift_x) >= threshold or abs(shift_y) >= threshold:
                    # 使用numpy.roll进行循环平移，完全保持形状和方向
                    psf_image = np.roll(psf_image, shift_y, axis=0)  # Y方向平移
                    psf_image = np.roll(psf_image, shift_x, axis=1)  # X方向平移
                    print(f"Iteration {iteration+1}: shift ({shift_x}, {shift_y}) pixels")
                    
                    # 重新计算居中后的位置
                    new_max_pos = np.unravel_index(np.argmax(psf_image), psf_image.shape)
                    new_centroid_x = np.sum(x_indices * psf_image) / total_intensity
                    new_centroid_y = np.sum(y_indices * psf_image) / total_intensity
                    
                    # 智能选择检测方法（重新评估）
                    if abs(new_max_pos[1] - new_centroid_x) < 5 and abs(new_max_pos[0] - new_centroid_y) < 5:
                        current_x, current_y = new_max_pos[1], new_max_pos[0]
                        method = "最大值位置"
                    else:
                        current_x, current_y = new_centroid_x, new_centroid_y
                        method = "质心位置"
                    
                    # 计算新的偏移量
                    shift_x = int(round(center_x - current_x))
                    shift_y = int(round(center_y - current_y))
                    
                    print(f"   New {method}: ({current_x:.1f}, {current_y:.1f}), new offset: ({shift_x}, {shift_y})")
                    
                    # 如果偏移量足够小，退出迭代
                    if abs(shift_x) < threshold and abs(shift_y) < threshold:
                        print(f"PSF perfectly centered after {iteration+1} iterations!")
                        break
                else:
                    if iteration == 0:
                        print(f"PSF already well centered (offset < {threshold} pixels), no adjustment needed")
                    break
            else:
                # 达到最大迭代次数
                final_offset = np.sqrt(shift_x**2 + shift_y**2)
                print(f"PSF image roll disabled; reference-phase offset remains {final_offset:.1f} pixels")

        # 确保 d_delta 是标量
        if torch.is_tensor(d_delta):
            d_delta = d_delta.item()
        
        print(f"Sampling interval: {d_delta}")
        # 计算x和y轴的实际尺寸。零点必须落在 fftshift 的 DC 索引 n_i//2 上，步长必须恰为
        # d_delta；旧的 linspace(-d*(n_i//2), d*(n_i//2), n_i) 把零点放在 n_i//2-0.5，
        # 且步长是 n_i*d/(n_i-1)（512 点时偏大 0.2%）。
        idx = torch.arange(n_i, dtype=torch.float64) - (n_i // 2)
        x = idx * d_delta
        y = idx * d_delta



        if show_plot:
            # 绘制PSF图像
            plt.figure()
            # plt.pcolormesh(X, Y, psf_image, shading='auto')
            plt.imshow(psf_image, extent=[x[0], x[-1], y[0], y[-1]],cmap='jet', origin='lower', aspect='equal')
            plt.colorbar(label='Normalized Intensity')

            # 设置坐标轴标签和标题
            plt.xlabel('X (um)')
            plt.ylabel('Y (um)')
            plt.title('PSF Image')

            # 如果指定了显示尺寸，则缩放视图
            if display_size_um is not None:
                half_size = display_size_um / 2
                plt.xlim(-half_size, half_size)
                plt.ylim(-half_size, half_size)
                # 更新刻度以反映新的范围
                plt.xticks([-half_size, half_size])
                plt.yticks([-half_size, half_size])
            else:
                # 设置坐标轴只显示最大和最小值
                plt.xticks([x[0], x[-1]], [f"{x[0]:.2f}", f"{x[-1]:.2f}"])
                plt.yticks([y[0], y[-1]], [f"{y[0]:.2f}", f"{y[-1]:.2f}"])

            # plt.imshow(psf_image, cmap='jet')
            # plt.colorbar(label='Normalized Intensity')
            # plt.title('FFT_PSF Image Plane')

        # # 计算MTF (能够被FFT法运行)
        # psf_tensor = torch.from_numpy(psf_image)  # 如果 psf 是 NumPy 数组
        # mtf = (torch.abs(torch.fft.fftshift(torch.fft.fft2(psf_tensor))))
        # mtf = mtf / torch.max(mtf).detach()
        # l0 = (n_i + 1) * d_delta * 1e-3
        # freq = (1 / l0) * torch.linspace(-n_i / 2, n_i / 2, n_i + 1)
        # # freq_pos = freq[freq >= 0]  # 只取正频率部分
        # ind = int(n_i / 2) + int(f_stop / (1 / l0)) #freq_cut处的索引
        # a = mtf[int(n_i / 2), ind]
        # b = mtf[int(n_i / 2), ind + 1]
        # mtf_s = (b - a) / (1 / l0) * (f_stop - freq[int(n_i / 2) + int(f_stop / (1 / l0))]) + a  # 弧矢 插值公式:y = y1 + (y2 - y1) / (x2 - x1) * (x - x1)
        # a = mtf[ind, int(n_i / 2)]
        # b = mtf[ind + 1, int(n_i / 2),]
        # mtf_t = (b - a) / (1 / l0) * (f_stop - freq[int(n_i / 2) + int(f_stop / (1 / l0))]) + a  # 子午


        # # 将 PyTorch 张量转换为 NumPy 数组（Matplotlib 需要）

        # # 生成正频率轴
        # freq_pos = (1 / l0) * torch.linspace(0, n_i / 2, n_i // 2 + 1)

        # # 提取正频率 MTF
        # mtf_s_pos = mtf[n_i // 2, n_i // 2:].numpy()  # 弧矢
        # mtf_t_pos = mtf[n_i // 2:, n_i // 2].numpy()  # 子午
        # # num_points = n_i // 2 + 1
        # # center_idx = n_i // 2
        # # mtf_s_pos = mtf[center_idx, center_idx : center_idx + num_points].cpu().numpy()
        # # mtf_t_pos = mtf[center_idx : center_idx + num_points, center_idx].cpu().numpy()
        # # 绘图
        # plt.figure(figsize=(10, 6))
        # plt.plot(freq_pos, mtf_s_pos, 'b--', label='Sagittal')
        # plt.plot(freq_pos, mtf_t_pos, 'r-', label='Tangential')
        # plt.xlabel('Spatial Frequency (cycles/mm)')
        # plt.ylabel('MTF')
        # plt.title('MTF (Positive Frequencies)')
        # plt.legend()
        # plt.grid(True)
        # plt.xlim(0, f_stop)  # 严格限制x轴范围到f_stop
        # plt.ylim(0, 1.0)  # y轴从0到1（因为MTF已归一化）

        if show_plot:
            plt.show()

        # 确保PSF尺寸正确（裁剪到n_i×n_i）
        # 这是为了匹配参考数据的尺寸（512×512）
        if psf_image.shape[0] != n_i or psf_image.shape[1] != n_i:
            # 计算裁剪范围（从中心裁剪）
            current_h, current_w = psf_image.shape
            start_h = (current_h - n_i) // 2
            start_w = (current_w - n_i) // 2
            psf_image = psf_image[start_h:start_h+n_i, start_w:start_w+n_i]
            print(f"PSF裁剪: {current_h}×{current_w} -> {n_i}×{n_i}")

        # Standardize PSF orientation for all entry points.
        psf_image = standardize_psf_orientation(psf_image)
        # Physical PSF for computation must be energy-normalized (sum=1).
        psf_image = sanitize_and_energy_normalize_psf(psf_image)
        return psf_image


    def _pad_pupils(self,pupils,grid_size):
        """
        Pad the pupils with zeros to match the grid size.

        Returns:
            list: A list of padded pupils.
        """

        pupils_padded = []
        for pupil in pupils:
            # 精确填充到 grid_size：旧实现用 `//2 + 1` 两侧对称填充，得到 grid_size+2，
            # 再由调用方居中裁剪回 grid_size，使 fftshift 的 DC 位置与坐标网格零点错开。
            pad = (grid_size - pupil.shape[0]) // 2
            pupil = np.pad(pupil,
                           ((pad, grid_size - pupil.shape[0] - pad),
                            (pad, grid_size - pupil.shape[1] - pad)),
                           mode='constant', constant_values=0)
            pupils_padded.append(pupil)
        return pupils_padded

    def _get_normalization(self,pupils):
        """
        Calculate the normalization factor for the Point Spread Function (PSF).

        Returns:
            float: The normalization factor for the PSF.
        """
        P_nom = pupils[0].copy()
        P_nom[P_nom != 0] = 1

        amp_norm = np.fft.fftshift(np.fft.fft2(P_nom))
        psf_norm = amp_norm * np.conj(amp_norm)
        return np.real(np.max(psf_norm) * len(pupils))

    def cal_mtf(self, psf, n_i, d_delta, f_stop, target=0.2, weight_s=1, weight_t=1, show_plot=True):
        """
        计算MTF评价函数。
        采用能量归一化PSF并进行DC归一化，确保MTF(0)=1。
        """
        _ = target
        _ = weight_s
        _ = weight_t

        if torch.is_tensor(d_delta):
            d_delta = float(d_delta.item())
        else:
            d_delta = float(d_delta)

        psf_norm = sanitize_and_energy_normalize_psf(psf)
        mtf_np = compute_dc_normalized_mtf(psf_norm)
        mtf = torch.from_numpy(mtf_np)

        n_mtf = int(mtf.shape[0])
        if n_i != n_mtf:
            n_i = n_mtf

        l0 = (n_i + 1) * d_delta
        freq_step = 1 / l0
        freq = freq_step * torch.linspace(-n_i / 2, n_i / 2, n_i + 1)

        center = n_i // 2
        ind = center + int(f_stop / freq_step)
        ind = max(center, min(ind, n_i - 2))

        a = mtf[center, ind]
        b = mtf[center, ind + 1]
        mtf_s = (b - a) / freq_step * (f_stop - freq[ind]) + a

        a = mtf[ind, center]
        b = mtf[ind + 1, center]
        mtf_t = (b - a) / freq_step * (f_stop - freq[ind]) + a

        if show_plot:
            freq_pos = freq_step * torch.linspace(0, n_i / 2, n_i // 2 + 1)
            mtf_s_pos = mtf[center, center:].numpy()
            mtf_t_pos = mtf[center:, center].numpy()
            plt.figure(figsize=(10, 6))
            plt.plot(freq_pos, mtf_s_pos, "b--", label="Sagittal")
            plt.plot(freq_pos, mtf_t_pos, "r-", label="Tangential")
            plt.xlabel("Spatial Frequency (cycles/mm)")
            plt.ylabel("MTF")
            plt.title("MTF (Positive Frequencies)")
            plt.legend()
            plt.grid(True)
            plt.xlim(0, f_stop)
            plt.ylim(0, 1.0)
            plt.show()

        return mtf_t, mtf_s


    def trace2sensor(self, ray, ignore_invalid=False, is_fixed=True, flag=False):
        """
        Trace rays towards intersecting onto the sensor plane.
        """
        # trace rays to last surface
        ray_final, valid = self.trace(ray, is_fixed=is_fixed, flag=flag)
        # intersecting sensor plane
        t = (self.surfaces[-1].thickness - ray_final.o[..., 2]) / ray_final.d[..., 2]
        p = ray_final.o + t[..., None] * ray_final.d
        weight = ray_final.weight
        p[..., 2] = 0

        if ignore_invalid:
            p = p[valid]
            weight = weight[valid]
            # print(valid.shape)
            # print(p.shape)
        else:
            if not valid.any():
                raise Exception('存在光线与像面无交点')
            if len(p.shape) < 2:
                return p
            p = torch.reshape(p, (np.prod(p.shape[:-1]), 3))  # reshape(-1,3)

        return p, weight.reshape(-1, 1)

    def trace_eyesensor(self, ray, ignore_invalid=False,is_fixed=True, flag=False):
        """
        双向追迹函数
        Trace rays towards intersecting onto the sensor plane.
        关键修复：确保图像始终居中，不受旋转影响
        """
        is_forward = (ray.d[..., 2] > 0).all()  # 检查z方向正负，全为正则为true
        if is_forward:
            ray_final, valid = self.trace(ray, stop_ind=None,is_fixed=is_fixed, flag=flag, OPD_flag=True)
            # intersecting sensor plane
            o = ray_final.o
            D = ray_final.d
            distance = self.surfaces[-1].thickness

            t0 = (distance - o[..., 2]) / D[..., 2]  # 即追迹至虚拟平面
            o = o + t0[..., None] * D
            o[..., 2] = 0  # 每次追迹到平面或虚拟平面，Z坐标清零

            # 移除强制居中代码，保持光学系统的真实性能
            # PSF居中应该在显示层面处理，而不是在光线追踪层面

            # pre-compute constants
            xn, yn, zn = (o[..., i].clone() for i in range(3))
            dx, dy, dz = (D[..., i].clone() for i in range(3))
            valid1 = torch.ones_like(zn).bool()


            c = self.img.c


            if c == 0:
                    t_delta = torch.zeros_like(zn)
            else:
                    B = dz - c * (dx * xn + dy * yn)
                    H = c * (xn ** 2 + yn ** 2)
                    temp = B ** 2 - c * H

                    # JNS: 当光线与面相切时，temp会很小但是符号为负，会导致计算错误，但是相切时也不会有折射，所以没问题

                    delta1 = B - torch.sqrt(temp)
                    delta2 = B + torch.sqrt(temp)
                    delta = torch.where(torch.abs(delta1) < torch.abs(delta2), delta1, delta2)
                    # JNS: 当delta1与delta2的绝对值相等时存在歧义，但是这种情况存在时一般相切

                    t_delta = delta / c
                    valid1 = (temp >= 0).bool()

            p = o + t_delta[..., None] * D
            weight = ray_final.weight

            ray.o=p
            ray.d=D
            wavelength = ray.wavelength
            PD = (p[...,2] + distance - ray_final.o[...,2]) / ray_final.d[...,2]
            OPD = self.n_image(wavelength*1e-6) * PD
            ray.phase = ray_final.phase + 2 * torch.pi / (wavelength*1e-6) * OPD
            if ignore_invalid:
                p = p[valid]
                weight = weight[valid]
                # print(valid.shape)
                # print(p.shape)
            else:
                if not valid.any():
                    raise Exception('存在光线与像面无交点')
                if len(p.shape) < 2:
                    return p

        else:
            ray = self.to_world.transform_ray(ray) #转换为全局坐标系下的Ray对象

            o = ray.o
            D = ray.d

            distance = -self.surfaces[-1].thickness

            t0 = (distance - o[..., 2]) / D[..., 2]  # 即追迹至虚拟平面
            o = o + t0[..., None] * D
            o[..., 2] = 0  # 每次追迹到虚拟平面，Z坐标清零

            # pre-compute constants
            xn, yn, zn = (o[..., i].clone() for i in range(3)) #位置分量
            dx, dy, dz = (D[..., i].clone() for i in range(3)) #方向分量
            valid1 = torch.ones_like(zn).bool()

            c = self.surfaces[len(self.surfaces) - 1].c

            if c == 0:
                t_delta = torch.zeros_like(zn)
            else:
                B = dz - c * (dx * xn + dy * yn)
                H = c * (xn ** 2 + yn ** 2)
                temp = B ** 2 - c * H

                # JNS: 当光线与面相切时，temp会很小但是符号为负，会导致计算错误，但是相切时也不会有折射，所以没问题

                delta1 = B - torch.sqrt(temp)
                delta2 = B + torch.sqrt(temp)
                delta = torch.where(torch.abs(delta1) < torch.abs(delta2), delta1, delta2)
                # JNS: 当delta1与delta2的绝对值相等时存在歧义，但是这种情况存在时一般相切

                t_delta = delta / c
                valid1 = (temp >= 0).bool()

            p = o - t_delta[..., None] * D
            weight = ray.weight

            ray.o = p
            ray.d = D

            ray= self.to_object.transform_ray(ray)

            valid, ray_out = self._backward_tracing(ray,stop_ind=len(self.surfaces) - 1 , is_fixed=is_fixed, flag=flag)
            # in world
            ray_final = self.to_world.transform_ray(ray_out)

            p = ray_final.o
            ray = ray_final


        return p,ray

    def trace_eyesensor_f(self, ray, ignore_invalid=False,is_fixed=True, flag=False):
        ray_final, valid = self.trace(ray, stop_ind=None, is_fixed=is_fixed, flag=flag, OPD_flag=True)
        # intersecting sensor plane
        o = ray_final.o
        D = ray_final.d
        distance = self.surfaces[-1].thickness

        t0 = (distance - o[..., 2]) / D[..., 2]  # 即追迹至虚拟平面
        o = o + t0[..., None] * D
        o[..., 2] = 0  # 每次追迹到平面或虚拟平面，Z坐标清零

        # pre-compute constants
        xn, yn, zn = (o[..., i].clone() for i in range(3))
        dx, dy, dz = (D[..., i].clone() for i in range(3))
        valid1 = torch.ones_like(zn).bool()

        c = self.img.c

        if c == 0:
            t_delta = torch.zeros_like(zn)
        else:
            B = dz - c * (dx * xn + dy * yn)
            H = c * (xn ** 2 + yn ** 2)
            temp = B ** 2 - c * H

            # JNS: 当光线与面相切时，temp会很小但是符号为负，会导致计算错误，但是相切时也不会有折射，所以没问题

            delta1 = B - torch.sqrt(temp)
            delta2 = B + torch.sqrt(temp)
            delta = torch.where(torch.abs(delta1) < torch.abs(delta2), delta1, delta2)
            # JNS: 当delta1与delta2的绝对值相等时存在歧义，但是这种情况存在时一般相切

            t_delta = delta / c
            valid1 = (temp >= 0).bool()

        p = o + t_delta[..., None] * D
        weight = ray_final.weight

        ray.o = p
        ray.d = D
        wavelength = ray.wavelength
        PD = (p[..., 2] + distance - ray_final.o[..., 2]) / ray_final.d[..., 2]
        OPD = self.n_image(wavelength * 1e-6) * PD
        ray.phase = ray_final.phase + 2 * torch.pi / (wavelength * 1e-6) * OPD
        if ignore_invalid:
            p = p[valid]
            weight = weight[valid]
            # print(valid.shape)
            # print(p.shape)
        else:
            if not valid.any():
                raise Exception('存在光线与像面无交点')
            if len(p.shape) < 2:
                return p

        return p,ray


    def trace(self, ray, stop_ind=None, is_fixed=True, flag=False, OPD_flag=False):
        """
        追迹至最后一表面，不包括像面
        """
        # update transformation when doing pose estimation
        if (
                self.origin.requires_grad
                or
                self.shift.requires_grad
                or
                self.theta_x.requires_grad
                or
                self.theta_y.requires_grad
                or
                self.theta_z.requires_grad
        ):
            self.update()

        # in local
        ray_in = self.to_object.transform_ray(ray)

        if stop_ind is None:
            # forward: last index to stop forward tracing; backward: first index to start backward tracing
            stop_ind = len(self.surfaces) - 1
        is_forward = (ray.d[..., 2] > 0).all()  # 检查z方向正负，全为正则为true

        # JNS: 只有当全为正Z方向时为Ture,用正向传播，存在是负Z方向的就会全部反向传播
        if is_forward:
            valid, ray_out = self._forward_tracing(ray_in, stop_ind, is_fixed=is_fixed, flag=flag, OPD_flag=OPD_flag)
        else:
            valid, ray_out = self._backward_tracing(ray_in, stop_ind, is_fixed=is_fixed, flag=flag)
        # in world
        ray_final = self.to_world.transform_ray(ray_out)
        return ray_final, valid
    
    def _forward_tracing(self, ray, stop_ind, start_ind=0, is_fixed=True, flag=False, OPD_flag=False):
        """
        forward tracing to the surface 
        """
        wavelength = ray.wavelength
        dim = ray.o[..., 2].shape
        step_size = 1e-3
        # 梯度介质 RK4 的默认 t 步长（mm）
        GRIN_STEP_DEFAULT = 5e-3
        valid = torch.ones(dim, device=self.device).bool()
        # 梯度区出射折射率，传给下一面作为入射侧折射率（None 表示用材料表）
        n_in_override = None
        for i in range(start_ind,stop_ind + 1):
            surface = self.surfaces[i]
            
            # 计算传播距离
            if i == start_ind:
                distance = 0
            else:
                distance = self.surfaces[i - 1].thickness
            
            # GRIN表面的特殊处理（直接检查表面类型）
            if isinstance(surface, Gradient_3):
                # 步骤1: 光线传播到GRIN表面并求交
                if surface.coeff is None:
                    valid_o, p = surface.ray_surface_intersection(distance, ray, valid, option='numerical',
                                                                           is_fixed=is_fixed)
                    n_surf = surface.normal(p[..., 0], p[..., 1])
                else:
                    valid_o, p = surface.ray_surface_intersection(distance, ray, valid, option='implicit',
                                                                           is_fixed=is_fixed)
                    n_surf = surface.normal(p[..., 0], p[..., 1])
                
                # 步骤2: 在GRIN表面折射进入
                # 交点的局部 z 就是该点 sag（顶点在 z=0），折射率必须取在交点上
                n_at_surface = self.surfaces[i].get_ior(p[..., 0], p[..., 1], p[..., 2])
                # 入射侧折射率：若上一段是梯度区，用其出射折射率，避免落回材料表的常数
                if n_in_override is None:
                    n_before = self.materials[i].ior(wavelength)
                    n_before = torch.as_tensor(n_before, dtype=p.dtype, device=p.device)
                    n_before = n_before.expand_as(n_at_surface)
                else:
                    n_before = n_in_override
                eta = n_before / n_at_surface  # n1/n2
                valid_d, d_refracted = self._refract(ray.d, n_surf, eta, flag=flag)

                # 入射段 OPD（与非 GRIN 分支同一约定：p 在本面帧，ray.o 在上一面帧）
                if OPD_flag:
                    PD_in = (p[..., 2] + distance - ray.o[..., 2]) / ray.d[..., 2]
                    ray.phase = ray.phase + 2 * torch.pi / (wavelength * 1e-6) * (n_before * PD_in)

                # 检查有效性
                valid = valid * valid_o * valid_d
                if torch.sum(valid) * 3 < torch.numel(valid):
                    print(f"GRIN surface {i}: valid rays = {valid.sum()}/{valid.numel()}")
                    raise Exception(f'GRIN surface {i}: too many invalid rays')
                
                # 步骤3: 设置Runge-Kutta初始条件
                # 位置在表面上，方向已折射，z_local=0作为起点
                # 初始动量 T = n(r,z=0) * d_refracted
                T_init = n_at_surface.unsqueeze(-1) * d_refracted
                
                # 步骤4: 在GRIN介质中传播，直到穿过下一面（终止于真实面形，含 sag）
                # Excel 中的 Δt 是 Zemax 自己积分格式下的步长上界。本实现用 t 参数化
                # （dz/dt≈n·d_z），直接照搬会过粗，故取上界与 GRIN 默认步长的较小者。
                # 默认 5e-3 mm：实测在 1e-2 mm 步长下落点与光程已收敛到 9 位小数，
                # 5e-3 留一倍余量，同时比 1e-3 快约 5 倍。
                dt_surf = self.surfaces[i].delta_t
                grin_step = min(dt_surf, GRIN_STEP_DEFAULT) if dt_surf > 0 else GRIN_STEP_DEFAULT
                if i + 1 < len(self.surfaces):
                    p_out, T_out, opl_grin, valid_g = self.surfaces[i].trace_to_next_surface(
                        p, T_init, grin_step, float(self.surfaces[i].thickness), self.surfaces[i + 1])
                    valid = valid * valid_g
                else:
                    p_out, T_out, opl_grin = self.surfaces[i].runge_kutta_a(
                        p, T_init, grin_step, float(self.surfaces[i].thickness), return_opl=True)

                # 步骤5: 将光学动量转换回方向向量 d = T/n
                # 出射点的局部 z 直接取积分终点（顶点在 z=0）
                n_exit = self.surfaces[i].get_ior(p_out[..., 0], p_out[..., 1], p_out[..., 2])
                d_out = T_out / n_exit.unsqueeze(-1)
                # 归一化方向向量
                d_out = d_out / torch.norm(d_out, dim=-1, keepdim=True)

                # 梯度区内光程 OPL = ∫n ds，直接累加到相位
                if OPD_flag:
                    ray.phase = ray.phase + 2 * torch.pi / (wavelength * 1e-6) * opl_grin

                # 更新光线：落点已在下一面上，但仍表达在本面帧中，
                # 下一轮求交为近似空操作（PD≈0），符合主循环的帧约定。
                ray.o = p_out
                ray.d = d_out
                # 把梯度区出射折射率传给下一面，作为其入射侧折射率
                n_in_override = n_exit

            else:
                # attention: the length of materials is different from the length of surfaces

            # ray intersecting surface
                if i == start_ind:
                    distance = 0
                else:
                    distance = self.surfaces[i - 1].thickness
                
                if surface.coeff is None:
                    valid_o, p = surface.ray_surface_intersection(distance, ray, valid, option='numerical',
                                                                           is_fixed=is_fixed)
                    n = surface.normal(p[..., 0], p[..., 1])
                else:
                    valid_o, p = surface.ray_surface_intersection(distance, ray, valid, option='implicit',
                                                                           is_fixed=is_fixed)
                    n = surface.normal(p[..., 0], p[..., 1])

                # 入射侧折射率：紧跟梯度区时用其出射折射率（材料表里的 grada 是常数，会失真）
                if n_in_override is None:
                    n_in_cur = self.materials[i].ior(wavelength)
                else:
                    n_in_cur = n_in_override
                # materials[k] is the homogeneous medium before surface k;
                # entering GRIN is handled only at the Gradient_3 surface.
                # A gradient material reaching this generic branch is a
                # malformed prescription and Material.ior() fails closed.
                eta = n_in_cur / self.materials[i + 1].ior(wavelength)
                # get surface normal and refract
                valid_d, d = self._refract(ray.d, n, eta, flag=flag)
                # check validity
                valid = valid * valid_o * valid_d

                if torch.sum(valid) * 3 < torch.numel(valid):
                    print(f"Forward tracing rays intersecting surface {surface.type}: {valid_o.sum()}")
                    print(f"Forward tracing rays normally refracted at surface {surface.type}: {valid_d.sum()}")


                    s = torch.sqrt(torch.sum(p[..., :2] ** 2, dim=-1))
                    print('Max ray radius:', s.max())
                    print('Min ray radius:', s.min())
                    print('Surface aperture length:', self.surfaces[i].semi_dia)
                    print('Forward tracing ray intersection position:', p)
                    print('Forward tracing ray intersection normal:', d)
                    raise Exception('forward trace: invalid ray， stop at surface {}'.format(str(i + 1)))
                if OPD_flag:
                    PD = (p[..., 2] + distance - ray.o[..., 2]) / ray.d[..., 2]
                    # print('路径长度', PD)
                    OPD = n_in_cur * PD
                    ray.phase = ray.phase + 2 * torch.pi / (wavelength * 1e-6) * OPD
                ray.o = p
                ray.d = d
                # 已消耗掉梯度区出射折射率
                n_in_override = None
                if self.surfaces[i].type == 'CB':
                    ray = self.transform_ray(ray, self.surfaces[i].tilt_x, self.surfaces[i].tilt_y,self.surfaces[i].tilt_z)

                else:
                    ray = ray

            if torch.isnan(ray.o).any() or torch.isnan(ray.d).any():
                print(f"NaN detected after tracing surface {i}!")
                raise Exception('前向光线追迹过程中出现NaN值！')

        ray.weight = ray.weight * valid

        return valid, ray
    
    def _backward_tracing(self, ray, stop_ind, is_fixed=True, flag=False):
        # JNS: 2024.1.16,修改了反向传播的代码，通过reverse把表面反转。实际追迹过程依然是正向的
        # JNS: 2024.1.16,最后输出y坐标和yz方向向量具有使用价值
        # JNS:2024.1.23 再次修改，逻辑为反向传播，方向为负，距离为负
        # JZY: 2024.9.14,修改了验证相交的输出代码

        #stop_ind :追迹的起始表面索引
        wavelength = ray.wavelength
        dim = ray.o[..., 2].shape

        valid = torch.ones(dim, device=ray.o.device).bool()

        for i in np.flip(range(stop_ind + 1)):  # 矩阵的翻转
            surface = self.surfaces[i]

            # ray intersecting surface
            if i == stop_ind:
                distance = 0
            else:
                distance = -self.surfaces[i].thickness

            if surface.coeff is None:
                valid_o, p = surface.ray_surface_intersection(distance, ray, valid, option='numerical',
                                                              is_fixed=is_fixed)
                n = surface.normal(p[..., 0], p[..., 1])
            else:
                valid_o, p = surface.ray_surface_intersection(distance, ray, valid, option='implicit',
                                                              is_fixed=is_fixed)
                n = surface.normal(p[..., 0], p[..., 1])

            # Gradient-3 indices are per-surface properties.  The material
            # table cannot represent consecutive GRIN surfaces with different
            # n0 values, so derive both sides of the interface from the
            # adjacent Gradient_3 objects where applicable (BIOT_vis contract).
            def _grin_axial(surf, at_exit):
                zero = torch.zeros((), dtype=surf.n0.dtype, device=surf.n0.device)
                z_local = torch.as_tensor(
                    float(surf.thickness) if at_exit else 0.0,
                    dtype=surf.n0.dtype,
                    device=surf.n0.device,
                )
                return surf.get_ior(zero, zero, z_local)

            cur_surface = self.surfaces[i]
            prev_surface = self.surfaces[i - 1] if i - 1 >= 0 else None
            if isinstance(cur_surface, Gradient_3):
                n_incident = _grin_axial(cur_surface, at_exit=False)
            else:
                n_incident = self.materials[i + 1].ior(wavelength)
            if isinstance(prev_surface, Gradient_3):
                n_exiting = _grin_axial(prev_surface, at_exit=True)
            else:
                n_exiting = self.materials[i].ior(wavelength)
            eta = n_incident / n_exiting

            # get surface normal and refract
            valid_d, d = self._refract(ray.d, -n, eta, flag=flag)  # backward: need to revert the normal

            # check validity
            valid = valid * valid_o * valid_d # valid_o: 有无成功相交, valid_d: 有无成功折射(没有发生全内反射)
            if not valid.any():
                print(f"Backward tracing rays intersecting surface {surface.type}: {valid_o.sum()}")
                print(f"Backward tracing rays normally refracted at surface {surface.type}: {valid_d.sum()}")

                s = torch.sqrt(torch.sum(p[..., :2] ** 2, dim=-1))
                print('Max ray radius:', s.max())
                print('Min ray radius:', s.min())
                print('Surface aperture length:', self.surfaces[i].semi_dia)
                print('Backward tracing ray intersection position:', p)
                print('Backward tracing ray intersection normal:', d)
                raise Exception('backward trace: invalid ray， stop at surface {}'.format(str(i + 1)))
            ray.o = p
            ray.d = d
            if self.surfaces[i].type == 'CB':
                ## 必须是正向旋转的严格逆（转置），不能靠取负角度：Rx/Ry/Rz 不可交换，
                ## 取负而不反转次序在 tilt_x 与 tilt_y 同时非零时不闭合。
                ray = self.transform_ray(ray, self.surfaces[i].tilt_x, self.surfaces[i].tilt_y,
                                         self.surfaces[i].tilt_z, inverse=True)
            else:
                ray = ray

        ray.weight = ray.weight * valid


        return valid, ray
    
    # def _backward_tracing(self, ray, stop_ind, is_fixed=True, flag=False):
    #     # JNS: 2024.1.16,修改了反向传播的代码，通过reverse把表面反转。实际追迹过程依然是正向的
    #     # JNS: 2024.1.16,最后输出y坐标和yz方向向量具有使用价值
    #     # JNS:2024.1.23 再次修改，逻辑为反向传播，方向为负，距离为负
    #     # JZY: 2024.8.1,修改了验证相交的输出代码
    #     # JZY: 2024.9.14,修改了梯度介质的处理逻辑 有很大问题

    #     #stop_ind :追迹的起始表面索引
    #     wavelength = ray.wavelength
    #     dim = ray.o[..., 2].shape
    #     step_size = 1e-3 # 保持与正向追迹一致
    #     valid = torch.ones(dim, device=ray.o.device).bool()

    #     for i in np.flip(range(stop_ind + 1)):  # 矩阵的翻转
    #         surface = self.surfaces[i]

    #         # 为反向追迹添加梯度介质处理逻辑
    #         if self.materials[i+1].name == 'grada':
    #             # 注意：反向时，是后一个材料/表面决定了当前介质
    #             distance = self.surfaces[i].thickness # 介质厚度总是正值
    #             # 反向追迹时，光学方向余弦 T 应该使用出射介质的折射率
    #             T = self.materials[i+1].ior(wavelength,0,0) * ray.d
    #             p, d = self.surfaces[i+1].runge_kutta_a(ray.o, T, step_size, distance)
    #             ray.o = p
    #             ray.d = d
    #             continue # 完成梯度追迹后，跳过本次循环的剩余部分，直接处理下一个表面

    #         # ray intersecting surface
    #         if i == stop_ind:
    #             distance = 0
    #         else:
    #             distance = -self.surfaces[i].thickness

    #         if surface.coeff is None:
    #             valid_o, p = surface.ray_surface_intersection(distance, ray, valid, option='numerical',
    #                                                           is_fixed=is_fixed)
    #             n = surface.normal(p[..., 0], p[..., 1])
    #         else:
    #             valid_o, p = surface.ray_surface_intersection(distance, ray, valid, option='implicit',
    #                                                           is_fixed=is_fixed)
    #             n = surface.normal(p[..., 0], p[..., 1])

    #         eta = self.materials[i + 1].ior(wavelength) / self.materials[i].ior(wavelength) # 入射介质折射率与出射介质折射率之比

    #         # get surface normal and refract
    #         valid_d, d = self._refract(ray.d, -n, eta, flag=flag)  # backward: need to revert the normal

    #         # check validity
    #         valid = valid * valid_o * valid_d # valid_o: 有无成功相交, valid_d: 有无成功折射(没有发生全内反射)
    #         if not valid.any():
    #             print(f"反向追迹与表面{surface.type}相交的光线数：{valid_o.sum()}")
    #             print(f"反向追迹与表面{surface.type}正常折射光线数：{valid_d.sum()}")


    #             s = torch.sqrt(torch.sum(p[..., :2] ** 2, dim=-1))
    #             print('最大光线半径:', s.max())
    #             print('最小光线半径:', s.min())
    #             print('终止面孔径长度', self.surfaces[i].semi_dia)
    #             print('反向追迹光线交点位置:', p)
    #             print('反向追迹光线交点法向量:', d)
    #             raise Exception('backward trace: invalid ray， stop at surface {}'.format(str(i + 1)))
    #         ray.o = p
    #         ray.d = d
    #         if self.surfaces[i].type == 'CB':
    #             ray = self.transform_ray(ray, -self.surfaces[i].tilt_x, self.surfaces[i].tilt_y,self.surfaces[i].tilt_z)
    #         else:
    #             ray = ray

    #     ray.weight = ray.weight * valid


    #     return valid, ray

    # def _forward_tracing(self, ray, stop_ind, start_ind=0, is_fixed=True, flag=False, OPD_flag=False):
    #     """
    #     forward tracing to the surface 
    #     """
    #     wavelength = ray.wavelength
    #     dim = ray.o[..., 2].shape
    #     step_size = 1e-3
    #     valid = torch.ones(dim, device=self.device).bool()
    #     for i in range(start_ind,stop_ind + 1):
    #         if self.materials[i].name == 'grada':
    #             if i == start_ind:
    #                 distance = 0
    #             else:
    #                 distance = self.surfaces[i - 1].thickness
    #             T = self.materials[i].ior(wavelength,0,0) * ray.d
    #             p, d = self.surfaces[i].runge_kutta_a(ray.o, T, step_size, distance)
    #             # 缺少相位计算
    #             ray.o = p
    #             ray.d = d

    #         else:
    #             # attention: the length of materials is different from the length of surfaces

    #         # ray intersecting surface
    #             if i == start_ind:
    #                 distance = 0
    #             else:
    #                 distance = self.surfaces[i - 1].thickness

    #                 if self.surfaces[i].coeff is None:
    #                     valid_o, p = self.surfaces[i].ray_surface_intersection(distance, ray, valid, option='numerical',
    #                                                                            is_fixed=is_fixed)
    #                 else:
    #                     valid_o, p = self.surfaces[i].ray_surface_intersection(distance, ray, valid, option='implicit',
    #                                                                            is_fixed=is_fixed)
    #                 if self.materials[i+1].name == 'grada' or self.materials[i+1].name == 'gradp':
    #                     eta = self.materials[i].ior(wavelength) / self.materials[i + 1].ior(wavelength,0,0)  # n1/n2
    #                 else:
    #                     eta = self.materials[i].ior(wavelength) / self.materials[i + 1].ior(wavelength)  # n1/n2
    #                 # get surface normal and refract

    #                 n = self.surfaces[i].normal(p[..., 0], p[..., 1])
    #                 valid_d, d = self._refract(ray.d, n, eta, flag=flag)
    #                 # check validity
    #                 valid = valid * valid_o * valid_d

    #                 if torch.sum(valid) * 3 < torch.numel(valid):
    #                     print("正向追迹与下一表面相交的光线数：{}".format(valid_o.sum()))
    #                     print("正向追迹与下一表面正常折射光线数：{}".format(valid_d.sum()))

    #                     s = torch.sqrt(torch.sum(p[..., :2] ** 2, dim=-1))
    #                     print('最大光线半径:', s.max())
    #                     print('最小光线半径:', s.min())
    #                     print('终止面孔径长度', self.surfaces[i].semi_dia)
    #                     print(p)
    #                     print(d)
    #                     raise Exception('forward trace: invalid ray， stop at surface {}'.format(str(i + 1)))
    #                 if OPD_flag:
    #                     PD = (p[..., 2] + distance - ray.o[..., 2]) / ray.d[..., 2]
    #                     # print('路径长度', PD)
    #                     OPD = self.materials[i].ior(wavelength) * PD
    #                     ray.phase = ray.phase + 2 * torch.pi / (wavelength * 1e-6) * OPD
    #                 ray.o = p
    #                 ray.d = d
    #                 if self.surfaces[i].type == 'CB':
    #                     ray = self.transform_ray(ray, self.surfaces[i].tilt_x, self.surfaces[i].tilt_y,self.surfaces[i].tilt_z)

    #                 else:
    #                     ray = ray

    #     ray.weight = ray.weight * valid

    #     return valid, ray


    def _refract(self, wi, n, eta, approx=False, flag=False):
        """
        Snell's law (surface normal n defined along the positive z axis)
        https://physics.stackexchange.com/a/436252/104805
        Args:
            wi: incident direction
            n: the normal vectors of the point 
            eta: the ratio of the refractive index, n1/n2
            flag: consider Fresnel loss or not
        returns:
            valid: valid map or ray weight map
            wt: outgoing direction 
        """
        if type(eta) is float:
            eta_ = eta
        else:
            if np.prod(eta.shape) > 1:
                eta_ = eta[..., None]
            else:
                eta_ = eta

        cosi = torch.sum(wi * n, dim=-1)  # 即入射角的余弦

        if approx:
            tmp = 1. - eta ** 2 * (1. - cosi)
            valid = tmp > 0.
            wt = tmp[..., None] * n + eta_ * (wi - cosi[..., None] * n)
        else:
            cost2 = 1. - (1. - cosi ** 2) * eta ** 2  # 出射角余弦的平方

            # 1. get valid map; 2. zero out invalid points; 3. add eps to avoid NaN grad at cost2==0.
            valid = cost2 > 0.
            cost2 = torch.clamp(cost2, min=1e-8)  # 将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量
            cost = torch.sqrt(cost2)

            # here we do not have to do normalization because if both wi and n are normalized,
            # then output is also normalized.
            wt = cost[..., None] * n + eta_ * (wi - cosi[..., None] * n)  # n2*wt = n1*wi +(n2*cost-n1*cosi)*n
        if flag:
            Rs = (eta * cosi - cost) ** 2 / (eta * cosi + cost) ** 2
            Rp = (eta * cost - cosi) ** 2 / (eta * cost + cosi) ** 2
            weight = (1 - 0.5 * (Rs + Rp))
            valid = valid * weight

        return valid, wt

    # ------------------------------------------------------------------------------------
    # optimization
    # ------------------------------------------------------------------------------------
    # 光瞳采样
    def GQ(self, rings=3, arms=6, is_symmetry=True, is_plot=False):
        """
        在高斯积分法中，所追迹的光线按径向方法排列，并加上一个最佳的权重因子以便用最少数量的光线来评估RMS。
        虽然这个方法很有效，但对一部分光线被表面孔径拦截的系统，它并不准确。
        在有表面孔径的系统中计算波前RMS要用矩形阵列的方法，且要有大量的光线以得到足够的精度。
        输入：
            rings: 环数，默认为3，最大12,目前只有rings=3或者6的情形
            arms： 幅角的数目，默认为6，最大为12，
            is_symmetry=True, 对称系统
            is_plot=False, 判断是否绘制光线网格
        根据高斯-勒让德求积公式的零点以及求积系数确定光瞳坐标与权重
        零点是[0,1]区间的n阶勒让德多项式的零点在开方后的结果，权重根据零点计算
        权重根据零点计算,可以根据[-1，1]的n阶勒让德多项式的零点计算
        旋转对称，减少追迹光线数,单视场追迹光线数=rings*arms/2
        不考虑旋转对称，单视场追迹光线数=rings*arms
        中间变量:
            Pxy: 第1列表示Px,第2列表示Py,第3列表示权重
        输出：
            Px,Py,weight 都reshape成2维矩阵
        """
        device = self.device
        if is_symmetry:
            N = int(arms / 2)
        else:
            N = arms
        k = 1 / 0.5 * 0.5235987755982938  # 乘以比例因子k后权重与软件相同
        if rings == 3:
            # 默认rings=3时的权重
            r = torch.tensor([0.33571069, 0.70710678, 0.94196515])
            weights = torch.tensor([0.13888889, 0.22222222, 0.13888889])  # 权重与zemax和codev不同，与论文相同不知道软件是怎么弄的
        elif rings == 6:
            r = torch.tensor([0.18375321, 0.41157661, 0.61700114, 0.78696226, 0.91137517, 0.98297241])
            weights = torch.tensor([0.04283112, 0.09019089, 0.11697848, 0.11697848, 0.09019039, 0.04283112])
        theta = torch.arange(1, N + 1)
        theta = (theta - 0.5) * 2 * torch.pi / arms - torch.pi / 2
        Pxy = []
        for i in range(rings):
            weight = weights[i] / N * torch.ones(N) * k
            Pxy.append(torch.stack((r[i] * torch.cos(theta), r[i] * torch.sin(theta), weight), dim=-1))
        Pxy = torch.cat(Pxy, dim=0)
        if is_plot:
            plt.figure()
            plt.scatter(Pxy[..., 0], Pxy[..., 1])
            plt.axis('equal')
        Px = Pxy[..., 0].reshape(-1, 1).to(device)
        Py = Pxy[..., 1].reshape(-1, 1).to(device)
        weight = Pxy[..., 2].reshape(-1, 1).to(device)
        return Px, Py, weight

    def RA(self, DEL=0.22, is_symmetry=True, is_plot=False):
        """
        矩形阵列，默认间隔为0.22，在半个入瞳内就会有34根光线，当需要更精确的控制面型时，比如非球面，采用更多的光线。
        输入：
            DEL: 光线间隔，默认为0.22
            is_symmetry=True, 对称系统
            is_plot=False, 判读是否绘制光线网格
        中间变量:
            Pxy: 第1列表示Px,第2列表示Py,第3列表示权重
        输出：
            Px,Py,weight 都reshape成2维矩阵
        """
        device = self.device
        a = torch.arange(DEL / 2, 1, DEL)
        y = torch.cat([-torch.flip(a, dims=[0]), a], dim=0)
        if is_symmetry:
            x = a
        else:
            x = y
        X, Y = torch.meshgrid(x, y, indexing='xy')
        ind = (X ** 2 + Y ** 2) <= 1
        Pxy = torch.stack((X[ind], Y[ind], torch.ones_like(X[ind])), dim=-1)
        if is_plot:
            plt.figure()
            plt.scatter(Pxy[..., 0], Pxy[..., 1])
            plt.axis('equal')
        Px = Pxy[..., 0].reshape(-1, 1).to(device)
        Py = Pxy[..., 1].reshape(-1, 1).to(device)
        n = Px.shape[0]
        weight = 2 * torch.ones_like(Px).to(device) / n
        return Px, Py, weight

    # ------------------------------------------------------------------------------------
    # Operands 操作数
    # ------------------------------------------------------------------------------------
    """
    所有操作数的说明都可参考ZEMAX中文使用手册或者https://www.optkt.cn/，
    注意：对比zemax，有些操作数名字相同，当与zemax有细微差别
    一般有两种输出模式
    当calculate=False时，仅输出操作数的评估值
    当calculate=True时，输出操作数的评估值，像差的加权平方，以及操作数的权重
    其中需要注意的是，要保证操作数本身的值结果是带梯度的，而另外两项不带梯度
    """

    def TRCXY(self, ps, weight, target=None, calculate=False):
        """
        TR表示像面径向垂轴像差，C表示质心
        计算垂轴像差TRCX,径向像差TRCY,参考为质心
        输入:
            ps: 单波长单视场下的像点坐标
            weight: 每根光线的权重
            calculate: 表示是否计算，加权平方，并求权重的和
        """
        weight = torch.tensor(weight).reshape(-1, 1)
        ps = ps[..., :2]
        if target is None:
            ps_mean = torch.mean(ps, axis=0)  # 计算质心
        else:
            ps_mean = target
        pr = (ps - ps_mean[None, ...])  # 相对坐标
        if calculate:
            weight = weight / 2
            return pr.reshape(-1, 1), torch.sum(torch.sum(pr.detach() ** 2, axis=-1) * weight / 2), torch.cat(
                [weight, weight], axis=0)
        else:
            return pr.reshape(-1, 1)

    def CENXY(self, Hx=0.0, Hy=0.0, wavelength=None, M=0.385, target=None, weight=1, calculate=False):
        """
        计算指定波长视场下像面的质心坐标
        输入:
            Hx,Hy: 归一化视场坐标
            wavelength: 波长。默认中心波长
            M: 矩形阵列采样，默认M=0.385，对应光瞳内有24根光线
            target: 目标值，包括x坐标与y坐标的目标值,一般是个列表[target_x,target_y]或者尺寸为(2,1)的tensor
            weight: 每根光线的权重
            calculate: 表示是否计算，加权平方，并求权重的和
        """
        weight = torch.tensor(weight).reshape(-1, 1)
        if wavelength is None:
            wavelength = self.wavelengths_center
        # 计算质心，可以只采样少量的光线，RA采样，M=0.385，对应光瞳内有24根光线
        # JNS: 默认视场为角度，当视场为非角度时需要另外写
        ray = self.sample_ray_angle(wavelength=wavelength, Hx=Hx, Hy=Hy, M=M, R=None, entrance_pupil=True,
                                    sampling="RA")
        ps, weight = self.trace2sensor(ray, ignore_invalid=True, is_fixed=True)
        ps = ps[..., :2]
        ps_mean = torch.mean(ps, axis=0).reshape(-1, 1)  # 计算质心
        if calculate:
            target = torch.tensor(target, axis=0).reshape(-1, 1)
            loss = weight * (ps_mean - target) ** 2
            return ps_mean - target, torch.cat([weight, weight], axis=0)
        else:
            return ps_mean

    def EFFL(self, target, weight=1, calculate=False):
        # 需要重新计算，直接引用属性self.EFL可能会没有梯度
        # self.enp_dia可以直接引用
        weight = torch.tensor(weight).reshape(-1, 1).to(self.device)
        EFL = (self.cal_EFL(self.enp_dia))[0].reshape(-1, 1)
        if calculate:
            loss = weight * (EFL.detach() - target) ** 2
            return EFL - target, loss, weight
        else:
            return EFL

    def MNCA(self, surf1, target=0.01, weight=1, calculate=False):
        """
        空气的最小中央厚度。该边界操作数限制从Surf1到Surf2中每个玻璃类型为空气的面的中央厚度要大于指定的目标值.
        建议该操作数一次只约束一个面，可以重复使用；一次约束多个面，影响的只是权重以及贡献
        输入:
            surf1: 面的索引，注意代码中，第一个面的索引是0，不同于zemax
        注意保证所计算的面的厚度被设为变量，这样存在梯度且为1，不然没有梯度
        与zemax不同，显示的评估值就是实际厚度，不会根据与target的大小关系而改变
        但是会改变target的值，因此要额外输出target
        """
        # 没实现一次约束多个面，感觉没必要
        i = int(surf1)
        th = (self.surfaces[i].thickness).detach()
        if self.materials[i + 1].name == 'air':
            target = th if th > target else target
        else:
            target = th
        if calculate:
            loss = weight * (th - target) ** 2
            return th - target, loss, weight
        else:
            return th

    def MXCA(self, surf1, target=1e3, weight=1, calculate=False):
        """
        空气的最大中央厚度。该边界操作数限制Surf1到Surf2中每个玻璃类型为空气（即没有玻璃）的面的中央厚度要小于指定的目标值。
        """
        i = int(surf1)
        th = (self.surfaces[i].thickness).detach()
        if self.materials[i + 1].name == 'air':
            target = th if th < target else target
        else:
            target = th
        if calculate:
            loss = weight * (th - target) ** 2
            return th - target, loss, weight
        else:
            return th

    def MNEA(self, surf1, target=0.01, weight=1, calculate=False):
        """
        空气中的最小边缘厚度。该边界操作数限制从Surf1到Surf2中每个玻璃类型为空气（即没有玻璃）的面的边缘厚度要大于指定的目标值。
        """
        i = int(surf1)
        th = (self.surfaces[i].thickness).detach()
        y1 = torch.tensor([self.surfaces[i].semi_dia]).reshape(1, 1).to(self.device)
        x = torch.zeros_like(y1)
        if i < len(self.surfaces) - 1:
            y2 = torch.tensor([self.surfaces[i + 1].semi_dia]).reshape(1, 1).to(self.device)
            y = y2 if y2 < y1 else y1
            sag2 = self.surfaces[i + 1].get_sag(x, y)
        else:
            sag2 = 0.
        sag1 = self.surfaces[i].get_sag(x, y1)
        th_edge = th + sag2 - sag1
        if self.materials[i + 1].name == 'air':
            target = th_edge if th_edge > target else target
        else:
            target = th_edge
        if calculate:
            loss = weight * (th_edge - target) ** 2
            return th_edge - target, loss, weight
        else:
            return th_edge

    def MNCG(self, surf1, target, weight=1, calculate=False):
        """
        玻璃的最小中央厚度。该边界操作数限制从Surf1到Surf2中每个玻璃类型为非空气的面的中央厚度要大于指定的目标值

        """
        i = int(surf1)
        th = (self.surfaces[i].thickness).detach()
        if self.materials[i + 1].name == 'air':
            target = th
        else:
            target = th if th > target else target
        if calculate:
            loss = weight * (th - target) ** 2
            return th - target, loss, weight
        else:
            return th

    def MXCG(self, surf1, target, weight=1, calculate=False):
        """
        玻璃的最大中央厚度。该边界操作数限制从Surf1到Surf2中每个玻璃类型为非空气的面的中央厚度要小于指定的目标值

        """
        i = int(surf1)
        th = (self.surfaces[i].thickness).detach()
        if self.materials[i + 1].name == 'air':
            target = th
        else:
            target = th if th < target else target
        if calculate:
            loss = weight * (th - target) ** 2
            return th - target, loss, weight
        else:
            return th

    def MNEG(self, surf1, target=0.01, weight=1, calculate=False):
        """
        玻璃中的最小边缘厚度。该边界操作数限制从Surf1到Surf2中每个玻璃类型为非空气的面的中央厚度要大于指定的目标值
        """
        i = int(surf1)
        th = (self.surfaces[i].thickness).detach()
        y1 = torch.tensor([self.surfaces[i].semi_dia]).reshape(1, 1).to(self.device)
        x = torch.zeros_like(y1)
        if i < len(self.surfaces) - 1:
            y2 = torch.tensor([self.surfaces[i + 1].semi_dia]).reshape(1, 1).to(self.device)
            y = y2 if y2 < y1 else y1
            sag2 = self.surfaces[i + 1].get_sag(x, y)
        else:
            sag2 = 0.
        sag1 = self.surfaces[i].get_sag(x, y1)
        th_edge = th + sag2 - sag1
        if self.materials[i + 1].name == 'air':
            target = th_edge
        else:
            target = th_edge if th_edge > target else target
        if calculate:
            loss = weight * (th_edge - target) ** 2
            return th_edge - target, loss, weight
        else:
            return th_edge

    def TRC_default(self, Hxy, M, sampling='RA'):
        """
        操作数计算所有波长所有视场的垂轴像差TRCX和径向像差TRCY
        Hxy: 归一化视场坐标
        M : 光瞳采样参数
        sampling: 光瞳采样方式‘RA’或‘GQ’
        
        """

        def render_single(Hxy, M, wavelength):
            """
            单波长，所有视场的rms。返回点列spot，loss和rms
            """
            p_real = []  # 单波长实际像点坐标
            p_relative = []  # 单波长实际相对像点坐标(实际像点坐标相对质心或相对主光线)
            loss_single = 0  # 单波长的像差加权平方和
            weight_single = 0  # 单波长的像差权重之和
            for i in range(Hxy.shape[0]):
                ray = self.sample_ray_angle(wavelength, Hx=Hxy[i][0], Hy=Hxy[i][1], M=0.22, R=None, entrance_pupil=True,
                                            sampling="RA")
                ps, weight = self.trace2sensor(ray, ignore_invalid=True, is_fixed=True)
                p_real.append(ps[..., :2])
                pr, loss_view, weight_view = self.TRCXY(ps[..., :2], weight, calculate=True)
                loss_single += loss_view
                weight_single += weight_view
                p_relative.append(pr)
            return p_real, p_relative, loss_single, weight_single

        # all tracing points
        p_all = []
        pss_all = []
        loss_sum = 0
        weight_sum = 0
        # 计算所有波长的像差
        for wavelength in self.wavelengths:
            out = render_single(Hxy, M, wavelength)
            p_all.append(out[0])
            pss_all.append(out[1])
            loss_sum = loss_sum + out[2]
            weight_sum = weight_sum + out[3]
        return p_all, pss_all, loss_sum, weight_sum

        def thick_bound(self, gl=None, air=None):
            """
            MNCA,MXCA,MNEA,MNCG,MXCG,MNEG
            """
            n = len(self.surfaces)
            bound = []
            for i in range(n):
                bound.append(self.MNCA(surf1=i, target=air[0], weight=1, calculate=True)[:2])
                bound.append(self.MXCA(surf1=i, target=air[1], weight=1, calculate=True)[:2])
                bound.append(self.MNEA(surf1=i, target=air[2], weight=1, calculate=True)[:2])
                bound.append(self.MNCG(surf1=i, target=gl[0], weight=1, calculate=True)[:2])
                bound.append(self.MXCG(surf1=i, target=gl[1], weight=1, calculate=True)[:2])
                bound.append(self.MNEG(surf1=i, target=gl[2], weight=1, calculate=True)[:2])
            temp = torch.vstack([torch.hstack(bound[i]) for i in range(len(bound))])
            th_op = temp[..., 0].reshape(-1, 1)
            loss = torch.sum(temp[..., 1], axis=0)
            weight_sum = 3 * n
            return th_op, loss, weight_sum

    # ------------------------------------------------------------------------------------
    # visualizations
    # ------------------------------------------------------------------------------------
    def plot_setup2D(self, ax=None, fig=None, show=True, color='k', with_sensor=True):
        """
        2D Viewer.
        """
        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            show = False

    # ------------------------------------------------------------------------------------
    # analysis    
    # ------------------------------------------------------------------------------------
    def render_single(self, Hxy, wavelength, analyse=True):
        """
        单波长，所有视场的rms。返回点列spot，loss和rms
        """
        p_real = []  # 单波长实际像点坐标

        for i in range(Hxy.shape[0]):
            ray = self.sample_ray_angle(wavelength, Hx=Hxy[i][0], Hy=Hxy[i][1], M=25, R=None, entrance_pupil=False,
                                        sampling="Fibonacci")
            ps, weight = self.trace2sensor(ray, ignore_invalid=False, is_fixed=True)
            p_real.append(ps[..., :2])
            if analyse:
                print("-------- Field x: {} Field y: {} --------".format(Hxy[i][0], Hxy[i][1]))
                self.spot_diagram(ps[..., :2])

        return p_real

    def rms(self, ps, units=1, option='centroid', refer=None, squared=False):
        ps = ps[..., :2] * units  # shape(n,2)
        if option == 'centroid':
            ps_mean = torch.mean(ps, axis=0)  # shape(2,)

        if option == 'target':
            if refer is None:
                ps_mean = torch.mean(ps, axis=0)
            else:
                ps_mean = refer.to(self.device)

        ps = ps - ps_mean[None, ...]  # we now use normalized ps
        if squared:
            # MSE
            return torch.mean(torch.sum(ps ** 2, axis=-1)), ps / units, ps_mean
        else:
            # RMS
            return torch.sqrt(torch.mean(torch.sum(ps ** 2, axis=-1))), ps / units, ps_mean

    def spot_diagram(self, ps, show=True, xlims=None, ylims=None, color='b.', savepath=None):
        """
        Plot spot diagram.
        """
        units = 1
        spot_rms = float(self.rms(ps, units)[0])  # rms with centroid as reference point
        ps = ps.cpu().detach().numpy()[..., :2]
        ps_mean = np.mean(ps, axis=0)  # centroid
        ps = ps - ps_mean[None, ...]  # we now use normalized ps
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(ps[..., 0], ps[..., 1], color)
        ax.set_aspect(1)
        plt.gca().set_aspect('equal', adjustable='box')
        if (xlims is not None) or (ylims is not None):
            plt.xlim(*xlims)
            plt.ylim(*ylims)
            min_ = xlims[0] if xlims[0] < ylims[0] else ylims[0]
            max_ = xlims[1] if xlims[1] > ylims[1] else ylims[1]

            plt.xticks(np.linspace(min_, max_, 11))
            plt.yticks(np.linspace(min_, max_, 11))

        # ax.set_aspect(1./ax.get_data_ratio())
        units_str = '[mm]'
        plt.xlabel('x ' + units_str)
        plt.ylabel('y ' + units_str)
        plt.title('RMS radius: {}  units: [um]  reference: centroid'.format(spot_rms * 1000))
        # plt.grid(True)

        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

        return spot_rms

    def dist(self, ps, refer=None):
        # JNS:这里的畸变反映的是光线质心与目标点的畸变，与常规的不同
        ps = ps[..., :2]
        ps_mean = torch.mean(ps, axis=0)  # centroid
        refer = refer.to(self.device)
        eps = 1e-19
        distortion = torch.sqrt(torch.sum((refer - ps_mean) ** 2) / (torch.sum(refer ** 2) + eps))
        return distortion

    def dist_grid(self, ps, show=True, xlims=5, ylims=5, num=5, color='b.', savepath=None):
        # 在横纵坐标上根据刻度添加网格线
        plt.figure()
        plt.grid(axis='x', linestyle='-.', linewidth=1, color='black', alpha=0.2)
        plt.grid(axis='y', linestyle='-.', linewidth=1, color='black', alpha=0.2)

        plt.scatter(ps[..., 0], ps[..., 1], s=1)
        plt.axis("square")

        # 修改横轴坐标刻度
        plt.xticks(np.linspace(xlims, 0, num, endpoint=True))
        plt.yticks(np.linspace(ylims, 0, num, endpoint=True))
        plt.title('Distortion grid')

        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

        return ps

    def spot_grid(self):
        raise NotImplementedError()

    def rays2grids(self, x, y, w, x0=None, y0=None, l0=None, w0=None, m=None, n=None):
        """ 可微地进行光线能量的统计
        Args:
            x:  点列的x坐标
            y:  点列的y坐标
            w:  该点的权重
            x0:  接收器中心坐标
            y0:  接收器中心坐标
            l0:  接收器x方向的长度
            w0:  接收器y方向的宽度
            m:  接收器x方向的格点数
            n:  接收器y方向的格点数
        Returns:
            x_grid:  照度的坐标网格
            y_grid:  照度的坐标网格
            I:  统计的能量值
        """
        if x0 is None:
            x0 = 0
        if y0 is None:
            y0 = 0

        # the pixel interval in the x,y direction
        step_x = l0 / n
        step_y = w0 / m

        # coordinate grids
        # JNS：所以这是接收器格点的中心坐标
        [x_grid, y_grid] = torch.meshgrid(
            torch.linspace(-l0 / 2 + step_x / 2, l0 / 2 - step_x / 2, n, device=self.device),
            torch.linspace(-w0 / 2 + step_y / 2, w0 / 2 - step_y / 2, m, device=self.device),
            indexing='xy')

        # Taking the pixel center as the reference frame
        l1 = l0 - step_x
        w1 = w0 - step_y
        x = x - x0
        y = y - y0  # 点列从全局坐标转接收器坐标
        x_grid = x_grid + x0  # 格点坐标从接收器坐标转全局
        y_grid = y_grid + y0

        # the index where (x,y) is in the image
        index_x = torch.floor((x + l1 / 2) / step_x).long()  # JNS: 为了实现可微，需要确定点列对其周围4个像素格点中心的权重，所以必须向下取整
        index_y = torch.floor(
            (y + w1 / 2) / step_y).long()  # JNS: 为此也舍弃了格点中心围成的网格之外的点列。(实际这里只是舍弃了左侧和下侧，但是后面for循环又把上侧和右侧给舍弃了)
        flag = (index_x * step_x <= l1) & (index_x * step_x >= 0) & (index_y * step_y <= w1) & (index_y * step_y >= 0)
        index_x = index_x[flag]
        index_y = index_y[flag]
        x = x[flag]
        y = y[flag]
        w = w[flag]

        # distance from points to the line of pixel center 点列中每个点到其对应格点周围4边界的距离
        h1 = (index_x + 1) * step_x - x - l1 / 2  # 右边界
        h2 = (index_y + 1) * step_y - y - w1 / 2  # 上边界
        h3 = x - index_x * step_x + l1 / 2  # 左边界
        h4 = y - index_y * step_y + w1 / 2  # 下边界

        s1 = h1 * h2 / step_x / step_y * w
        s2 = h2 * h3 / step_x / step_y * w
        s3 = h1 * h4 / step_x / step_y * w
        s4 = h3 * h4 / step_x / step_y * w

        s = torch.zeros([m, n], device=self.device)
        for i in range(n - 1):  # index of x x方向为列
            for j in range(m - 1):  # index of y y方向为行
                flag = (index_x == i) & (index_y == j)
                s[j, i] = s[j, i] + torch.sum(s1[flag])
                s[j, i + 1] = s[j, i + 1] + torch.sum(s2[flag])
                s[j + 1, i] = s[j + 1, i] + torch.sum(s3[flag])
                s[j + 1, i + 1] = s[j + 1, i + 1] + torch.sum(s4[flag])
        s = s / step_x / step_y  # 能量/面积

        del index_x, index_y

        return x_grid, y_grid, s
        # JNS: 由于舍弃了点列，实际最外一圈的像素是不准确的，所以舍弃最外围像素
        # JNs: 但是如果l0与w0足够大，使得x,y点列都在格点中心坐标构成的网格内，此时就无所谓了
        # return x_grid[1:-1,1:-1], y_grid[1:-1,1:-1], s[1:-1,1:-1]

    # ------------------------------------------------------------------------------------
    # visualizations
    # ------------------------------------------------------------------------------------


class Surface(PrettyPrinter):
    """
    This is the base class for optical surfaces.
    The surface is parameterized as an implicit function f(x,y,z) = 0.
    For simplicity, we assume the surface function f(x,y,z) can be decomposed as:
    
    f(x,y,z) = h(z) - g(x,y),

    where g(x,y) and h(z) are explicit functions to be defined in sub-classes.

    Args:
        semi_dia: Radius of the aperture (default to be circular, unless specified as square).
        position: Distance of z-direction in global coordinate
        thickness: the center distance between the surface and the next surface
        is_square: is the aperture square
        device: Torch device
    """

    def __init__(self, semi_dia, position, thickness, is_square=False,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        if torch.is_tensor(position):
            self.position = position
        else:
            self.position = torch.Tensor(np.asarray(float(position))).to(device)
        self.device = device
        self.is_square = is_square
        self.semi_dia = float(semi_dia)
        self.fixed = True  # wheteher semi-diameter is fixed or not
        self.thickness = torch.tensor(thickness)

        # There are the parameters controlling the accuracy of ray tracing.
        self.NEWTONS_MAXITER = 100
        self.NEWTONS_TOLERANCE_TIGHT = 50e-6  # in [mm], i.e. 50 [nm] here (up to <10 [nm])
        self.NEWTONS_TOLERANCE_LOOSE = 300e-6  # in [mm], i.e. 300 [nm] here (up to <10 [nm])
        self.APERTURE_SAMPLING = 257

    def normal(self, x, y):
        """
        Returns the 3D normal vector of the surface at 2D coordinate (x,y), in local coordinate.
        """
        dfdx, dfdy, dfdz = self.get_surface(x, y)[1:]
        # JNS: 我的法向量z方向是正的，和源码不同，注意细节差异
        return normalize(torch.stack((dfdx, dfdy, dfdz), dim=-1))

    def sdf_approx(self, p, is_fixed=True):
        """
        (Approximated) Signed Distance Function (SDF) of a 2D point p to the surface's aperture boundary.
        If:
        - Returns > 0: p is within the surface's aperture.
        - Returns = 0: p is at the surface's aperture boundary.
        - Returns < 0: p is outside of the surface's aperture.

        Args:
            p: Local 2D point.
        
        Returns:
            A SDF mask.
        """
        if self.is_square:
            return torch.max(self.semi_dia - torch.abs(p), dim=-1)[0]
        else:  # is round
            r_2 = torch.sum(p ** 2, dim=-1)
            if is_fixed == None:
                return torch.ones_like(r_2)
            if is_fixed == False:
                r_max = torch.sqrt(torch.max(r_2)).item()
                if r_max >= self.semi_dia:
                    self.semi_dia = r_max
            return self.semi_dia ** 2 - r_2

    def is_valid(self, p, is_fixed=True):
        """
        If a 2D point p is valid, i.e. if p is within the surface's aperture.
        """
        tolerance = -1e-6
        return (self.sdf_approx(p, is_fixed=is_fixed) >= -1e-6).bool()

    def ray_surface_intersection(self, distance, ray, active=None, option='implicit', is_fixed=True):
        """
        Computes ray-surface intersection.
        Given ray(s) and an activity mask, 
        the function computes the intersection point(s),
        and determines if the intersection is valid, 
        and update the active mask accordingly. 
        
        Args:
            distance: The on-axis distance of the surface and ray
            ray: Rays.
            active: The initial active mask.

        Returns:
            valid_o: The updated active mask (if the current ray is physically active in tracing).
            local: The computed intersection point(s).    
        """
        solution_found, local = self.newtons_method(ray.maxt, distance, ray.o, ray.d, option)
        valid_o = solution_found * self.is_valid(local[..., 0:2], is_fixed=is_fixed)
        if active is not None:
            valid_o = active * valid_o

        return valid_o, local

    def Da(self, m):
        x = m[..., 0]
        y = m[..., 1]
        z = m[..., 2]

        # 定义折射率分布函数
        n00 = 1.368
        n01 = 0             # 0.049057
        n02 = 0             # -0.015427
        n10 = -0.001978
        n = n00 + n01 * z + n02 * z ** 2 + n10 * (x ** 2 + y ** 2)
        dn2dx = (4 * n00 * n10 * x + 4 * n01 * n10 * x * z + 4 * n02 * n10 * x * z ** 2 + 4 * (n10 ** 2) * (
                x ** 2 + y ** 2) * x) / 2
        dn2dy = (4 * n00 * n10 * y + 4 * n01 * n10 * y * z + 4 * n02 * n10 * y * z ** 2 + 4 * (n10 ** 2) * (
                x ** 2 + y ** 2) * y) / 2
        dn2dz = (2 * n00 * n01 + 4 * n00 * n02 * z + 2 * n01 * n01 * z + 6 * n01 * n02 * z ** 2 + 2 * n01 * n10 * (
                x ** 2 + y ** 2) + 4 * n02 * n02 * z ** 3 + 4 * n02 * n10 * (x ** 2 + y ** 2) * z) / 2
        gradients = torch.stack([dn2dx, dn2dy, dn2dz], dim=-1)
        return gradients



    def runge_kutta_a(self, initial_o, initial_T, step_size, distance,
                      z_vertex=None, return_opl=False):
        """
        在GRIN介质中使用Runge-Kutta方法追迹光线（Sharma t-参数化）

        统一采用 t 参数化的光线方程（t 满足 ds = n dt）：
        - dr/dt = T   （T = n*d 是光学动量）
        - dT/dt = grad(n^2)/2

        不要与弧长参数化（dr/ds = T/n, dT/ds = grad(n)）混用：混用会把
        梯度弯折放大约 n 倍。

        光程 OPL = ∫ n ds = ∫ n^2 dt，与位置/动量同步用同一套 RK 权重累加。

        Args:
            initial_o: 起点位置（全局坐标，z 为当前面局部帧下的 sag）
            initial_T: 起点光学动量 T = n*d
            step_size: t 方向步长（mm，按 ds≈n·dt 折算）
            distance: 要传播的几何 z 深度（mm），用于估算步数
            z_vertex: 表面顶点在当前局部帧中的 z（默认 0，即 newtons_method 约定）
            return_opl: True 时额外返回累计光程 OPL

        Returns:
            p: 终点位置
            T_out: 终点光学动量
            （return_opl=True 时追加 opl）
        """
        # 表面顶点在当前局部帧中的 z。newtons_method 之后 o[...,2] 是 sag，
        # 顶点固定在 z=0，因此 z_local = z（不能再减去每根光线自己的 sag，
        # 否则 Nz1*z 项对边缘光线会被错误地平移）。
        if z_vertex is None:
            z_local_offset = torch.zeros_like(initial_o[..., 2])
        else:
            z_local_offset = z_vertex if torch.is_tensor(z_vertex) else \
                torch.full_like(initial_o[..., 2], float(z_vertex))

        # t 参数化下 dz/dt = T_z = n·d_z，而 n 沿光路变化，所以不能用起点的 T_z
        # 线性折算出一个固定的 t 区间：在 Nz1>0 的介质里 n 随 z 增大，
        # t_span = distance/T_z(起点) 会积过头。实测纯 z 梯度解析情形下
        # （n0=1.368, Nz1=0.049057, 深度 1.59）终点 z 会冲到 1.6362，
        # 光程偏大 0.0669 mm，且不随步长收敛——这是区间设定错，不是离散误差。
        # 因此改为按几何目标 z 推进到跨过后再割线细化，与
        # trace_to_next_surface 用同一套终止判据。
        z_target = z_local_offset + float(distance)

        Tz = initial_T[..., 2].abs().clamp_min(1e-12)
        t_span = float(distance) / float(Tz.mean().item())
        base_steps = max(1, int(abs(t_span) / max(step_size, 1e-9)) + 1)
        h = t_span / base_steps
        # 留余量：n 增大时实际需要的步数比线性估计少，n 减小时更多。
        max_steps = int(base_steps * 1.6) + 4

        r = initial_o.clone()
        T = initial_T.clone()
        opl = torch.zeros_like(initial_o[..., 2])

        reached = (r[..., 2] - z_target) >= 0
        for _ in range(max_steps):
            if bool(reached.all()):
                break
            r_try, T_try, d_opl = self._rk4_step(r, T, h, z_local_offset)
            advance = (~reached).unsqueeze(-1)
            r = torch.where(advance, r_try, r)
            T = torch.where(advance, T_try, T)
            opl = opl + torch.where(reached, torch.zeros_like(d_opl), d_opl)
            reached = reached | ((r[..., 2] - z_target) >= 0)

        # 割线细化到 z == z_target；dF/dt = dz/dt = T_z。
        for _ in range(12):
            F = r[..., 2] - z_target
            if bool((F.abs() < 1e-12).all()):
                break
            dF_dt = T[..., 2]
            dF_dt = torch.where(dF_dt.abs() < 1e-12,
                                torch.full_like(dF_dt, 1e-12), dF_dt)
            dt = -F / dF_dt
            r, T, d_opl = self._rk4_step(r, T, dt, z_local_offset)
            opl = opl + d_opl

        # 返回终点位置和光学动量
        # 注意：光线方向 d = T/n，调用者需要转换
        if return_opl:
            return r, T, opl
        return r, T

    def _rk4_step(self, r, T, h, z_local_offset):
        """
        单步 4 阶 Runge-Kutta（t 参数化）。

        dr/dt = T,  dT/dt = grad(n^2)/2,  dOPL/dt = n^2

        Args:
            r: 位置（局部帧，z 从 0 起算于表面顶点）
            T: 光学动量 n*d
            h: t 步长，可以是标量或按光线的张量（形状同 r[...,0]）
            z_local_offset: 顶点 z 偏移，通常为 0

        Returns:
            (r_new, T_new, d_opl)
        """
        if torch.is_tensor(h):
            hh = h.unsqueeze(-1)
        else:
            hh = h

        def deriv(rr, TT):
            x, y, z = rr[..., 0], rr[..., 1], rr[..., 2]
            z_loc = z - z_local_offset
            n = self.get_ior(x, y, z_loc)
            # Da 返回 grad(n^2)，此处按 z_local_offset 把顶点对齐到局部 z=0
            grad_n2 = self.Da(rr, z_local_offset)
            return TT, 0.5 * grad_n2, n ** 2

        k1_r, k1_T, k1_L = deriv(r, T)
        k2_r, k2_T, k2_L = deriv(r + 0.5 * hh * k1_r, T + 0.5 * hh * k1_T)
        k3_r, k3_T, k3_L = deriv(r + 0.5 * hh * k2_r, T + 0.5 * hh * k2_T)
        k4_r, k4_T, k4_L = deriv(r + hh * k3_r, T + hh * k3_T)

        r_new = r + hh * (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
        T_new = T + hh * (k1_T + 2 * k2_T + 2 * k3_T + k4_T) / 6
        h_s = h if torch.is_tensor(h) else torch.full_like(k1_L, float(h))
        d_opl = h_s * (k1_L + 2 * k2_L + 2 * k3_L + k4_L) / 6
        return r_new, T_new, d_opl

    def trace_to_next_surface(self, p_in, T_in, step_size, z_offset, next_surface,
                              max_extra_frac=1.6):
        """
        在梯度介质中一直积分，直到穿过“下一面”，再把落点精修到该面上。

        与固定参数长度积分的区别：Zemax 的梯度区终止于下一面的实际面形
        （含 sag），而不是走完等于 thickness 的参数长度。

        Args:
            p_in: 入射点（当前面局部帧，z 为当前面 sag）
            T_in: 入射光学动量 n*d
            step_size: t 步长
            z_offset: 下一面顶点在当前局部帧中的 z（= 当前面 thickness）
            next_surface: 下一面对象，需提供 surface(x, y) 给出 sag
            max_extra_frac: 允许的最大额外行程系数（防止不收敛时死循环）

        Returns:
            (p_out, T_out, opl, valid)
        """
        z_local_offset = torch.zeros_like(p_in[..., 2])

        def signed_gap(rr):
            # F<0 表示还在梯度区内，F>=0 表示已穿过下一面
            sag_next = next_surface.get_sag(rr[..., 0], rr[..., 1])
            return rr[..., 2] - (z_offset + sag_next)

        Tz = T_in[..., 2].abs().clamp_min(1e-12)
        t_span = float(z_offset) / float(Tz.mean().item())
        base_steps = max(1, int(abs(t_span) / max(step_size, 1e-9)) + 1)
        h = t_span / base_steps
        max_steps = int(base_steps * max_extra_frac) + 4

        r = p_in.clone()
        T = T_in.clone()
        opl = torch.zeros_like(p_in[..., 2])
        crossed = signed_gap(r) >= 0

        for _ in range(max_steps):
            if bool(crossed.all()):
                break
            r_try, T_try, d_opl = self._rk4_step(r, T, h, z_local_offset)
            advance = (~crossed).unsqueeze(-1)
            r = torch.where(advance, r_try, r)
            T = torch.where(advance, T_try, T)
            opl = opl + torch.where(crossed, torch.zeros_like(d_opl), d_opl)
            crossed = crossed | (signed_gap(r) >= 0)

        # 精修：把已越过的光线沿反向拉回到面上。用 secant 迭代求 t 修正量。
        F = signed_gap(r)
        for _ in range(12):
            n_here = self.get_ior(r[..., 0], r[..., 1], r[..., 2] - z_local_offset)
            # dF/dt ≈ T_z - dsag/d(x,y)·T_xy，主导项用 T_z 足够收敛
            dF_dt = T[..., 2].clone()
            dF_dt = torch.where(dF_dt.abs() < 1e-12, torch.full_like(dF_dt, 1e-12), dF_dt)
            dt = -F / dF_dt
            dt = torch.where(F.abs() < 1e-12, torch.zeros_like(dt), dt)
            if float(dt.abs().max().item()) < 1e-13:
                break
            r, T, d_opl = self._rk4_step(r, T, dt, z_local_offset)
            opl = opl + d_opl
            F = signed_gap(r)
            _ = n_here  # 保留局部折射率读取，便于调试

        valid = torch.isfinite(F) & (F.abs() < 1e-6)
        return r, T, opl, valid

    def newtons_method(self, maxt, distance, o, D, option='implicit'):
        """
        Newton's method to find the root of the ray-surface intersection point.

        Two modes are supported here:

        1. 'explicit": This implements the loop using autodiff, and gradients will be
        accurate for o, D, and self.parameters. Slow and memory-consuming.

        2. 'implicit": This implements the loop using implicit-layer theory, find the
        solution without autodiff, then hook up the gradient. Less memory-consuming.

        3. 'numerical': This means the intersections have exact numerical solutions

        Args:
            maxt: The maximum travel distance of a single ray.
            o: The origins of the rays.
            D: The directional vector of the rays.
            option: The computing modes.

        Returns:
            valid: The updated active mask (if the current ray is physically active in tracing).
            p: The computed intersection point(s).
        """

        # initial guess of t

        t0 = (distance - o[..., 2]) / D[..., 2]  # 即追迹至虚拟平面
        o = o + t0[..., None] * D
        o[..., 2] = 0  # 每次追迹到平面或虚拟平面，Z坐标清零
        # pre-compute constants
        xn, yn, zn = (o[..., i].clone() for i in range(3))
        dx, dy, dz = (D[..., i].clone() for i in range(3))
        valid = torch.ones_like(zn).bool()
        if self.type == 'P':
            return valid, o
        if option == 'explicit':
            t_delta, valid = self.newtons_method_impl(maxt, dx, dy, dz, xn, yn, zn)
        elif option == 'implicit':
            with torch.no_grad():
                t_delta, valid = self.newtons_method_impl(maxt, dx, dy, dz, xn, yn, zn)
                zm, dfdx, dfdy, dfdz = self.get_surface(xn + t_delta * dx, yn + t_delta * dy)
            t_delta = t_delta + (zm - t_delta * dz) / (dfdx * dx + dfdy * dy + dfdz * dz)

        elif option == 'numerical':
            if self.c == 0:
                t_delta = torch.zeros_like(zn)
            else:
                B = dz - self.c * (dx * xn + dy * yn)
                H = self.c * (xn ** 2 + yn ** 2)
                temp = B ** 2 - self.c * H

                # JNS: 当光线与面相切时，temp会很小但是符号为负，会导致计算错误，但是相切时也不会有折射，所以没问题

                delta1 = B - torch.sqrt(temp)
                delta2 = B + torch.sqrt(temp)
                delta = torch.where(torch.abs(delta1) < torch.abs(delta2), delta1, delta2)
                # JNS: 当delta1与delta2的绝对值相等时存在歧义，但是这种情况存在时一般相切

                t_delta = delta / self.c
                valid = (temp >= 0).bool()

        else:
            raise Exception('option={} is not available!'.format(option))

        p = o + t_delta[..., None] * D

        return valid, p

    def newtons_method_impl(self, maxt, dx, dy, dz, xn, yn, zn):
        """
        The actual implementation of Newton's method.

        Args:
            dx,dy,dz,xn,yn,zn: Variables to a quadratic problem.
        
        Returns:
            t: The travel distance of the ray.
            t_delta: The incremental change of t at each iteration.
            valid: The updated active mask (if the current ray is physically active in tracing).
        """

        t_delta = torch.zeros_like(zn)
        # iterate until the intersection error is small
        residual = maxt * torch.ones_like(zn)
        it = 0

        if self.c == 0:
            t_delta = torch.zeros_like(zn)
        else:
            B = dz - self.c * (dx * xn + dy * yn)
            H = self.c * (xn ** 2 + yn ** 2)
            temp = B ** 2 - self.c * H

            # JNS: 当光线与面相切时，temp会很小但是符号为负，会导致计算错误，但是相切时也不会有折射，所以没问题

            delta1 = B - torch.sqrt(temp)
            delta2 = B + torch.sqrt(temp)
            delta = torch.where(torch.abs(delta1) < torch.abs(delta2), delta1, delta2)
            # JNS: 当delta1与delta2的绝对值相等时存在歧义，但是这种情况存在时一般相切

            t_delta = delta / self.c
            # valid = (temp >= 0).bool()

        # 迭代过程
        while (torch.abs(residual) > self.NEWTONS_TOLERANCE_TIGHT).any() and (it < self.NEWTONS_MAXITER):
            it += 1
            _xn = xn + t_delta * dx
            _yn = yn + t_delta * dy

            _zn = zn + t_delta * dz
            # ind = (self.k+1)*(_xn**2+_yn**2)>=(1/self.c)**2

            zm, dfdx, dfdy, dfdz = self.get_surface(_xn, _yn)

            residual = zm - _zn
            t_delta = t_delta + residual / (dfdx * dx + dfdy * dy + dfdz * dz)

        valid = (torch.abs(residual) < self.NEWTONS_TOLERANCE_LOOSE) & (t_delta <= maxt)
        return t_delta, valid


class Asphere(Surface):
    """
    surface function f(x,y,z)=0    
    f(x,y,z) = h(z) - g(x,y),

    where g(x,y) and h(z) are explicit functions(for asphere):
    g(x,y) = c * r**2 / (1 + sqrt( 1 - (1+k) * r**2*c**2 )) + ai[0] * r**4 + ai[1] * r**6 + \cdots.
    h(z) = z
    dfdx = -dgdx;
    dfdy = -dgdy;
    dfdz=dhdz;
    g(x,y) can be regarded as the sum of a spherical component and a polynomial component
    Args (old attributes):
        semi_dia:半孔径
        position:该面在系统中的位置，考虑物距，但当无穷远时不考虑物距，在代码中透镜的距离是指该表面到前表面的距离与设计软件不同
        thickness:该面到后一个面的距离
    Args (new attributes):
        c: Surface curvature,可以为负
        k:Conic coefficient
        coeff_list: Asphere parameters, could be a vector. When None, the surface is spherical.[4,6,8,...]
    """

    def __init__(self, semi_dia, position, thickness, c=0, k=0,dec_x=0,dec_y=0,tilt_x=0,tilt_y=0,tilt_z=0, coeff_list=None, is_square=False,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        Surface.__init__(self, semi_dia, position, thickness, is_square, device)
        self.type = 'S'
        self.isaperture = False
        self.c = torch.tensor(c)  # JNS：c不等于0，c等于0时，应该视为平面，另外计算了
        self.k = torch.tensor(k)
        self.dec_x = torch.tensor(dec_x)
        self.dec_y = torch.tensor(dec_y)
        self.tilt_x = torch.tensor(tilt_x)
        self.tilt_y = torch.tensor(tilt_y)
        self.tilt_z = torch.tensor(tilt_z)
        self.coeff = None
        self.m = None
        if coeff_list is not None:
            self.coeff = torch.Tensor(np.array(coeff_list)).to(device)
            self.m = len(coeff_list)

    # Asphere surface equation = sphere + polynomial 

    # the shape of x,y is (u,v) and the output 
    def distance_2(self, x, y): #径向距离平方
        if torch.is_tensor(x) and torch.is_tensor(y):
            x = x
            y = y
        else:
            x, y = (torch.Tensor(np.array(v)) for v in [x, y])
        return x ** 2 + y ** 2

    def get_sphere(self, x, y):
        Q = self.k + 1
        S_2 = self.distance_2(x, y)
        g_sp = self.c * S_2 / (1 + torch.sqrt(1 - (Q) * S_2 * self.c ** 2))
        dg_spdS_2 = self.c / (2 * torch.sqrt(1 - Q * S_2 * self.c ** 2))
        dS_2dx = 2 * x
        dS_2dy = 2 * y
        dg_spdx = dg_spdS_2 * dS_2dx
        dg_spdy = dg_spdS_2 * dS_2dy
        return g_sp, dg_spdx, dg_spdy

    def get_polynomial(self, x, y):
        S_2 = self.distance_2(x, y)
        S = torch.sqrt(S_2)
        g_poly = 0
        dg_polydS_2 = 0
        for i in range(self.m):
            g_poly = g_poly + self.coeff[i] * S_2 ** (i + 2)
            dg_polydS_2 = dg_polydS_2 + self.coeff[i] * (i + 2) * S_2 ** (i + 1)
        dS_2dx = 2 * x
        dS_2dy = 2 * y
        dg_polydx = dg_polydS_2 * dS_2dx
        dg_polydy = dg_polydS_2 * dS_2dy
        return g_poly, dg_polydx, dg_polydy

    def get_surface(self, x, y):
        g_sp, dg_spdx, dg_spdy = self.get_sphere(x, y)
        g_poly = 0
        dg_polydx = 0
        dg_polydy = 0
        if self.m:
            g_poly, dg_polydx, dg_polydy = self.get_polynomial(x, y)
        g = g_sp + g_poly
        # z=g
        dgdx = dg_spdx + dg_polydx
        dgdy = dg_spdy + dg_polydy
        dhdz = torch.ones_like(x)
        return g, -dgdx, -dgdy, dhdz

    def get_sag(self, x, y):
        # only for sag
        Q = self.k + 1
        S_2 = self.distance_2(x, y)
        g_sp = self.c * S_2 / (1 + torch.sqrt(1 - (Q) * S_2 * self.c ** 2))
        g_poly = 0
        if self.m:
            for i in range(self.m):
                g_poly = g_poly + self.coeff[i] * S_2 ** (i + 2)
        g = g_sp + g_poly
        # z=g
        return g

    def sag_max(self, x, y):
        # JNS:因为旋转对称只要算一个就行
        Q = self.k + 1
        S_2 = 1 / Q / self.c ** 2
        g_sp = self.c * S_2
        g_poly = 0
        if self.m:
            for i in range(self.m):
                g_poly = g_poly + self.coeff[i] * S_2 ** (i + 2)
        g = g_sp + g_poly

        g = g * torch.ones_like(y)

        return g

    def get_derivatives(self, x, y):
        # only for derivatives
        Q = self.k + 1
        S_2 = self.distance_2(x, y)
        dg_spdS_2 = self.c / (2 * torch.sqrt(1 - Q * S_2 * self.c ** 2))
        dS_2dx = 2 * x
        dS_2dy = 2 * y
        dg_spdx = dg_spdS_2 * dS_2dx
        dg_spdy = dg_spdS_2 * dS_2dy

        dg_polydS_2 = 0
        if self.m:
            for i in range(self.m):
                dg_polydS_2 = dg_polydS_2 + self.coeff[i] * (i + 2) * S_2 ** (i + 1)

        dS_2dx = 2 * x
        dS_2dy = 2 * y
        dg_polydx = dg_polydS_2 * dS_2dx
        dg_polydy = dg_polydS_2 * dS_2dy

        dgdx = dg_spdx + dg_polydx
        dgdy = dg_spdy + dg_polydy
        dhdz = torch.ones_like(x)
        return -dgdx, -dgdy, dhdz

    def reverse(self):
        self.c = -self.c
        if self.coeff is not None:
            self.coeff = -self.coeff

class CoordinateBreak(Surface):
    """
    surface function f(x,y,z)=0
    f(x,y,z) = h(z) - g(x,y),

    where g(x,y) and h(z) are explicit functions(for asphere):
    g(x,y) = c * r**2 / (1 + sqrt( 1 - (1+k) * r**2*c**2 )) + ai[0] * r**4 + ai[1] * r**6 + \cdots.
    h(z) = z
    dfdx = -dgdx;
    dfdy = -dgdy;
    dfdz=dhdz;
    g(x,y) can be regarded as the sum of a spherical component and a polynomial component
    Args (old attributes):
        semi_dia:半孔径
        position:该面在系统中的位置，考虑物距，但当无穷远时不考虑物距，在代码中透镜的距离是指该表面到前表面的距离与设计软件不同
        thickness:该面到后一个面的距离
    Args (new attributes):
        c: Surface curvature,可以为负
        k:Conic coefficient
        coeff_list: Asphere parameters, could be a vector. When None, the surface is spherical.[4,6,8,...]
    """

    def __init__(self, semi_dia, position, thickness, c=0, k=0,dec_x=0,dec_y=0,tilt_x=0,tilt_y=0,tilt_z=0, coeff_list=None, is_square=False,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        Surface.__init__(self, semi_dia, position, thickness, is_square, device)
        self.type = 'CB'
        self.isaperture = False
        self.c = torch.tensor(c)  # JNS：c不等于0，c等于0时，应该视为平面，另外计算了
        self.k = torch.tensor(k)
        self.dec_x = torch.tensor(dec_x)
        self.dec_y = torch.tensor(dec_y)
        self.tilt_x = torch.tensor(tilt_x)
        self.tilt_y = torch.tensor(tilt_y)
        self.tilt_z = torch.tensor(tilt_z)
        self.coeff = None
        self.m = None
        if coeff_list is not None:
            self.coeff = torch.Tensor(np.array(coeff_list)).to(device)
            self.m = len(coeff_list)


    # the shape of x,y is (u,v) and the output
    def distance_2(self, x, y):
        if torch.is_tensor(x) and torch.is_tensor(y):
            x = x
            y = y
        else:
            x, y = (torch.Tensor(np.array(v)) for v in [x, y])
        return x ** 2 + y ** 2

    def get_sphere(self, x, y):
        Q = self.k + 1
        S_2 = self.distance_2(x, y)
        g_sp = self.c * S_2 / (1 + torch.sqrt(1 - (Q) * S_2 * self.c ** 2))
        dg_spdS_2 = self.c / (2 * torch.sqrt(1 - Q * S_2 * self.c ** 2))
        dS_2dx = 2 * x
        dS_2dy = 2 * y
        dg_spdx = dg_spdS_2 * dS_2dx
        dg_spdy = dg_spdS_2 * dS_2dy
        return g_sp, dg_spdx, dg_spdy

    def get_polynomial(self, x, y):
        S_2 = self.distance_2(x, y)
        S = torch.sqrt(S_2)
        g_poly = 0
        dg_polydS_2 = 0
        for i in range(self.m):
            g_poly = g_poly + self.coeff[i] * S_2 ** (i + 2)
            dg_polydS_2 = dg_polydS_2 + self.coeff[i] * (i + 2) * S_2 ** (i + 1)
        dS_2dx = 2 * x
        dS_2dy = 2 * y
        dg_polydx = dg_polydS_2 * dS_2dx
        dg_polydy = dg_polydS_2 * dS_2dy
        return g_poly, dg_polydx, dg_polydy

    def get_surface(self, x, y):
        g_sp, dg_spdx, dg_spdy = self.get_sphere(x, y)
        g_poly = 0
        dg_polydx = 0
        dg_polydy = 0
        if self.m:
            g_poly, dg_polydx, dg_polydy = self.get_polynomial(x, y)
        g = g_sp + g_poly
        # z=g
        dgdx = dg_spdx + dg_polydx
        dgdy = dg_spdy + dg_polydy
        dhdz = torch.ones_like(x)
        return g, -dgdx, -dgdy, dhdz

    def get_sag(self, x, y):
        # only for sag
        Q = self.k + 1
        S_2 = self.distance_2(x, y)
        g_sp = self.c * S_2 / (1 + torch.sqrt(1 - (Q) * S_2 * self.c ** 2))
        g_poly = 0
        if self.m:
            for i in range(self.m):
                g_poly = g_poly + self.coeff[i] * S_2 ** (i + 2)
        g = g_sp + g_poly
        # z=g
        return g

    def sag_max(self, x, y):
        # JNS:因为旋转对称只要算一个就行
        Q = self.k + 1
        S_2 = 1 / Q / self.c ** 2
        g_sp = self.c * S_2
        g_poly = 0
        if self.m:
            for i in range(self.m):
                g_poly = g_poly + self.coeff[i] * S_2 ** (i + 2)
        g = g_sp + g_poly

        g = g * torch.ones_like(y)

        return g

    def get_derivatives(self, x, y):
        # only for derivatives
        Q = self.k + 1
        S_2 = self.distance_2(x, y)
        dg_spdS_2 = self.c / (2 * torch.sqrt(1 - Q * S_2 * self.c ** 2))
        dS_2dx = 2 * x
        dS_2dy = 2 * y
        dg_spdx = dg_spdS_2 * dS_2dx
        dg_spdy = dg_spdS_2 * dS_2dy

        dg_polydS_2 = 0
        if self.m:
            for i in range(self.m):
                dg_polydS_2 = dg_polydS_2 + self.coeff[i] * (i + 2) * S_2 ** (i + 1)

        dS_2dx = 2 * x
        dS_2dy = 2 * y
        dg_polydx = dg_polydS_2 * dS_2dx
        dg_polydy = dg_polydS_2 * dS_2dy

        dgdx = dg_spdx + dg_polydx
        dgdy = dg_spdy + dg_polydy
        dhdz = torch.ones_like(x)
        return -dgdx, -dgdy, dhdz

    def reverse(self):
        self.c = -self.c
        if self.coeff is not None:
            self.coeff = -self.coeff

class XYPolynomial(Surface):
    """
    This is the XY polynomial surface class, for freeform surfaces.
    
    The surface is parameterized as an implicit function f(x,y,z) = 0.
    For simplicity, we assume the surface function f(x,y,z) can be decomposed as:
    
    f(x,y,z) = h(z) - g(x,y) ,

    where g(x,y) and h(z) are explicit functions:
    
    g(x,y) = c * r**2 / (1 + sqrt( 1 - (1+k) * r**2*c**2 )) + \sum{i,j} a_ij x^i y^j.(1<i+j<J,J is the Polynomial order)
    h(z) = z.
    dfdx = -dgdx;
    dfdy = -dgdy;
    dfdz=dhdz;
    Args (new attributes):

        c: Surface curvature,可以为负
        k:Conic coefficient
        coeff_list: XYpolynomial parameters,it's a list
        J: Polynomial order.
        symmetry: When None, the surface have no symmetry; When "YZ" ,symmetry relative to the Y-Z plane; When "XZ",Symmetry relative to the X-Z plane
        center_symmetry:Symmetrical in the center, but not rotationally symmetrical
    """

    def __init__(self, semi_dia, position, thickness, c=0, k=0, coeff_list=None, J=0, norm_radiu=1, symmetry=None,
                 is_square=False, center_symmetry=True,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        Surface.__init__(self, semi_dia, position, thickness, is_square)
        self.type = 'X'
        self.isaperture = False
        self.c = torch.tensor(c)
        self.k = torch.tensor(k)
        self.symmetry = symmetry
        self.center_symmetry = center_symmetry
        self.J = J
        self.norm_radiu = norm_radiu
        self.coeff = torch.Tensor(np.array(coeff_list)).to(device)
        self.eps = torch.finfo(float).eps

    def poly(self, x, y):
        norm_x = x / self.norm_radiu
        norm_y = y / self.norm_radiu
        y_n = []
        x_n = []
        if self.center_symmetry:
            # JNS：偶数对称
            n = self.J // 2 + 1
            y_n = y_n + [2 * torch.arange(n - i) for i in range(n)]
            x_n = x_n + [2 * i * torch.ones(n - i) for i in range(n)]
            y_order = torch.cat(y_n)[1:].to(self.device)
            x_order = torch.cat(x_n)[1:].to(self.device)
        else:
            if self.symmetry is None:
                raise Exception("尚未使用过，有待验证")
                # 全系数
                y_n = y_n + [torch.arange(J + 1 - i) for i in range(J + 1)]
                x_n = x_n + [i * torch.ones(J + 1 - i) for i in range(J + 1)]
                y_order = torch.cat(y_n)[2:].to(self.device)  # 不取X0Y0,X0Y1
                x_order = torch.cat(x_n)[2:].to(self.device)
            else:
                if self.symmetry == "YZ":
                    raise Exception("尚未使用过，有待验证")
                    # YZ对称
                    n = J // 2 + 1
                    y_n = y_n + [torch.arange(J + 1 - 2 * i) for i in range(n)]
                    x_n = x_n + [2 * i * torch.ones(J + 1 - 2 * i) for i in range(n)]
                    y_order = torch.cat(y_n)[2:].to(self.device)  # 不取X0Y0,X0Y1
                    x_order = torch.cat(x_n)[2:].to(self.device)
                if self.symmetry == "XZ":
                    raise Exception("尚未使用过，有待验证")
                    # XZ对称
                    n = J // 2 + 1
                    y_n = y_n + [2 * torch.arange((J - i) // 2 + 1) for i in range(J + 1)]
                    x_n = x_n + [i * torch.ones((J - i) // 2 + 1) for i in range(J + 1)]
                    y_order = torch.cat(y_n)[1:].to(self.device)
                    x_order = torch.cat(x_n)[1:].to(self.device)
        # 对于不同模式，搞定order就行
        # print(x.shape)
        xs = norm_x.unsqueeze(-1).expand(-1, -1, len(x_order))
        ys = norm_y.unsqueeze(-1).expand(-1, -1, len(y_order))
        x_p = torch.pow(xs + self.eps, x_order)
        y_p = torch.pow(ys + self.eps, y_order)
        poly = x_p * y_p

        dx_order = x_order - 1
        dx_order[x_order == 0] = 0
        dx_p = torch.pow(xs + self.eps, dx_order)
        dpolydx = x_order * dx_p * y_p / self.norm_radiu
        dy_order = y_order - 1
        dy_order[y_order == 0] = 0
        dy_p = torch.pow(ys + self.eps, dy_order)
        dpolydy = y_order * dy_p * x_p / self.norm_radiu

        return poly, dpolydx, dpolydy

    def distance_2(self, x, y):
        if torch.is_tensor(x) and torch.is_tensor(y):
            x = x
            y = y
        else:
            x, y = (torch.Tensor(np.array(v)) for v in [x, y])
        return x ** 2 + y ** 2

    def get_sphere(self, x, y):
        Q = self.k + 1
        S_2 = self.distance_2(x, y)
        g_sp = self.c * S_2 / (1 + torch.sqrt(1 - (Q) * S_2 * self.c ** 2))
        dg_spdS_2 = self.c / (2 * torch.sqrt(1 - Q * S_2 * self.c ** 2))
        dS_2dx = 2 * x;
        dS_2dy = 2 * y
        dg_spdx = dg_spdS_2 * dS_2dx
        dg_spdy = dg_spdS_2 * dS_2dy
        return g_sp, dg_spdx, dg_spdy

    def get_polynomial(self, x, y):
        poly, dpolydx, dpolydy = self.poly(x, y)
        g_poly = torch.sum(poly * self.coeff, axis=-1)
        dg_polydx = torch.sum(dpolydx * self.coeff, axis=-1)
        dg_polydy = torch.sum(dpolydy * self.coeff, axis=-1)
        return g_poly, dg_polydx, dg_polydy

    def get_surface(self, x, y):
        g_sp, dg_spdx, dg_spdy = self.get_sphere(x, y)
        g_poly, dg_polydx, dg_polydy = self.get_polynomial(x, y)
        g = g_sp + g_poly
        # z=g
        dgdx = dg_spdx + dg_polydx
        dgdy = dg_spdy + dg_polydy
        dhdz = torch.ones_like(x)
        return g, -dgdx, -dgdy, dhdz

    def get_sag(self, x, y):
        Q = self.k + 1
        S_2 = self.distance_2(x, y)
        g_sp = self.c * S_2 / (1 + torch.sqrt(1 - (Q) * S_2 * self.c ** 2))
        g_poly = self.get_polynomial(x, y)[0]
        g = g_sp + g_poly
        # z=g
        return g

    def sag_max(self, x, y):
        norm = torch.sqrt(x ** 2 + y ** 2)
        Q = self.k + 1
        x = x / norm / torch.sqrt(Q)
        y = y / norm / torch.sqrt(Q)
        g_sp = self.c * (x ** 2 + y ** 2)
        g_poly = self.get_polynomial(x / abs(self.c), y / abs(self.c))[0]
        g = g_sp + g_poly

        return g

    def get_derivatives(self, x, y):
        # only for derivatives
        Q = self.k + 1
        S_2 = x ** 2 + y ** 2
        S = torch.sqrt(x ** 2 + y ** 2)
        dg_spdS_2 = self.c / (2 * torch.sqrt(1 - Q * S_2 * self.c ** 2))
        dS_2dx = 2 * x;
        dS_2dy = 2 * y
        dg_spdx = dg_spdS_2 * dS_2dx
        dg_spdy = dg_spdS_2 * dS_2dy
        dg_polydx, dg_polydy = self.get_polynomial(x, y)[1:]

        dgdx = dg_spdx + dg_polydx
        dgdy = dg_spdy + dg_polydy
        dhdz = torch.ones_like(x)
        return -dgdx, -dgdy, dhdz

    def reverse(self):
        self.c = -self.c
        self.coeff = -self.coeff


class Phase_Plate(Surface):
    """
    相位板实际的厚度与曲率都为0，但矢高表达式则用自由曲面的矢高表达式表征，此时曲率并不为0，虚拟曲率c。
    相位板仅仅是省去求交点的过程，因为认为实际厚度与曲率都为0，但是法向量并不是垂直，而是根据虚拟矢高求导计算
    在定义镜头参数时，相位板后会跟着一个平面，主要用于切换介质材料(变会空气)以及确定下一表面位置(下一表面到该相位板的距离)
    定义相位板时，实际没有机械孔径的概念，足够大即可
    """

    # 目前仅用偶次项的XY多项式表征矢高
    def __init__(self, semi_dia, position, thickness, c=0, k=0, coeff_list=None, J=0, norm_radiu=1, symmetry=None,
                 is_square=False, center_symmetry=True,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        Surface.__init__(self, semi_dia, position, thickness, is_square)
        self.type = 'P'
        self.isaperture = False
        self.c = torch.tensor(c)
        self.k = torch.tensor(k)
        self.symmetry = symmetry
        self.center_symmetry = center_symmetry
        self.J = J
        self.norm_radiu = norm_radiu
        self.coeff = torch.Tensor(np.array(coeff_list)).to(device)
        self.eps = torch.finfo(float).eps

    def poly(self, x, y):
        norm_x = x / self.norm_radiu
        norm_y = y / self.norm_radiu
        y_n = []
        x_n = []
        if self.center_symmetry:
            # JNS：偶数对称
            n = self.J // 2 + 1
            y_n = y_n + [2 * torch.arange(n - i) for i in range(n)]
            x_n = x_n + [2 * i * torch.ones(n - i) for i in range(n)]
            y_order = torch.cat(y_n)[1:].to(self.device)
            x_order = torch.cat(x_n)[1:].to(self.device)
        else:
            if self.symmetry is None:
                raise Exception("尚未使用过，有待验证")
                # 全系数
                y_n = y_n + [torch.arange(J + 1 - i) for i in range(J + 1)]
                x_n = x_n + [i * torch.ones(J + 1 - i) for i in range(J + 1)]
                y_order = torch.cat(y_n)[2:].to(self.device)  # 不取X0Y0,X0Y1
                x_order = torch.cat(x_n)[2:].to(self.device)
            else:
                if self.symmetry == "YZ":
                    raise Exception("尚未使用过，有待验证")
                    # YZ对称
                    n = J // 2 + 1
                    y_n = y_n + [torch.arange(J + 1 - 2 * i) for i in range(n)]
                    x_n = x_n + [2 * i * torch.ones(J + 1 - 2 * i) for i in range(n)]
                    y_order = torch.cat(y_n)[2:].to(self.device)  # 不取X0Y0,X0Y1
                    x_order = torch.cat(x_n)[2:].to(self.device)
                if self.symmetry == "XZ":
                    raise Exception("尚未使用过，有待验证")
                    # XZ对称
                    n = J // 2 + 1
                    y_n = y_n + [2 * torch.arange((J - i) // 2 + 1) for i in range(J + 1)]
                    x_n = x_n + [i * torch.ones((J - i) // 2 + 1) for i in range(J + 1)]
                    y_order = torch.cat(y_n)[1:].to(self.device)
                    x_order = torch.cat(x_n)[1:].to(self.device)
        # 对于不同模式，搞定order就行
        # print(x.shape)
        xs = norm_x.unsqueeze(-1).expand(-1, -1, len(x_order))
        ys = norm_y.unsqueeze(-1).expand(-1, -1, len(y_order))
        x_p = torch.pow(xs + self.eps, x_order)
        y_p = torch.pow(ys + self.eps, y_order)
        poly = x_p * y_p

        dx_order = x_order - 1
        dx_order[x_order == 0] = 0
        dx_p = torch.pow(xs + self.eps, dx_order)
        dpolydx = x_order * dx_p * y_p / self.norm_radiu
        dy_order = y_order - 1
        dy_order[y_order == 0] = 0
        dy_p = torch.pow(ys + self.eps, dy_order)
        dpolydy = y_order * dy_p * x_p / self.norm_radiu

        return poly, dpolydx, dpolydy

    def distance_2(self, x, y):
        if torch.is_tensor(x) and torch.is_tensor(y):
            x = x
            y = y
        else:
            x, y = (torch.Tensor(np.array(v)) for v in [x, y])
        return x ** 2 + y ** 2

    def get_sphere(self, x, y):
        Q = self.k + 1
        S_2 = self.distance_2(x, y)
        g_sp = self.c * S_2 / (1 + torch.sqrt(1 - (Q) * S_2 * self.c ** 2))
        dg_spdS_2 = self.c / (2 * torch.sqrt(1 - Q * S_2 * self.c ** 2))
        dS_2dx = 2 * x
        dS_2dy = 2 * y
        dg_spdx = dg_spdS_2 * dS_2dx
        dg_spdy = dg_spdS_2 * dS_2dy
        return g_sp, dg_spdx, dg_spdy

    def get_polynomial(self, x, y):
        poly, dpolydx, dpolydy = self.poly(x, y)
        g_poly = torch.sum(poly * self.coeff, axis=-1)
        dg_polydx = torch.sum(dpolydx * self.coeff, axis=-1)
        dg_polydy = torch.sum(dpolydy * self.coeff, axis=-1)
        return g_poly, dg_polydx, dg_polydy

    def get_surface(self, x, y):
        g_sp, dg_spdx, dg_spdy = self.get_sphere(x, y)
        g_poly, dg_polydx, dg_polydy = self.get_polynomial(x, y)
        g = g_sp + g_poly
        # z=g
        dgdx = dg_spdx + dg_polydx
        dgdy = dg_spdy + dg_polydy
        dhdz = torch.ones_like(x)
        return g, -dgdx, -dgdy, dhdz

    def get_sag(self, x, y):
        Q = self.k + 1
        S_2 = self.distance_2(x, y)
        g_sp = self.c * S_2 / (1 + torch.sqrt(1 - (Q) * S_2 * self.c ** 2))
        g_poly = self.get_polynomial(x, y)[0]
        g = g_sp + g_poly
        # z=g
        return g

    def sag_max(self, x, y):
        norm = torch.sqrt(x ** 2 + y ** 2)
        Q = self.k + 1
        x = x / norm / torch.sqrt(Q)
        y = y / norm / torch.sqrt(Q)
        g_sp = self.c * (x ** 2 + y ** 2)
        g_poly = self.get_polynomial(x / abs(self.c), y / abs(self.c))[0]
        g = g_sp + g_poly

        return g

    def get_derivatives(self, x, y):
        # only for derivatives
        Q = self.k + 1
        S_2 = x ** 2 + y ** 2
        S = torch.sqrt(x ** 2 + y ** 2)
        dg_spdS_2 = self.c / (2 * torch.sqrt(1 - Q * S_2 * self.c ** 2))
        dS_2dx = 2 * x
        dS_2dy = 2 * y
        dg_spdx = dg_spdS_2 * dS_2dx
        dg_spdy = dg_spdS_2 * dS_2dy
        dg_polydx, dg_polydy = self.get_polynomial(x, y)[1:]

        dgdx = dg_spdx + dg_polydx
        dgdy = dg_spdy + dg_polydy
        dhdz = torch.ones_like(x)
        return -dgdx, -dgdy, dhdz

class GridSag(Surface):
    """
    由离散网格矢高数据定义的表面，通过 B 样条进行拟合。
    该类使用 scipy.interpolate.RectBivariateSpline 将网格数据拟合为 B 样条曲面，
    并利用父类的牛顿-拉普森法进行光线-表面求交。

    Args:
        semi_dia (float): 曲面的半口径 (mm)，用于归一化坐标。
        position (float): 曲面在 Z 轴上的位置。
        thickness (float): 曲面到下一曲面的中心厚度。
        sag_file_path (str): 包含矢高数据的 XLSX 文件的路径。
        grid_shape (tuple): 矢高网格的形状，例如 (81, 81)。
        kx (int): x 方向的 B 样条次数。
        ky (int): y 方向的 B 样条次数。
        smooth_s (float): 平滑因子。s=0 表示严格插值。
        is_square (bool): 曲面孔径是否为方形。
        device: PyTorch 设备。
    """
    def __init__(self, semi_dia, position, thickness, sag_file_path, grid_shape=(81, 81), 
                 kx=3, ky=3, smooth_s=0.0, is_square=False,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        # 1. 初始化父类和基本属性
        Surface.__init__(self, semi_dia, position, thickness, is_square, device)
        self.type = 'GridSag'
        self.isaperture = False
        self.grid_shape = grid_shape
        self.sag_file_path = str(sag_file_path)
        self.coeff = None
        self.c = torch.tensor(0)

        # 2. 加载矢高数据
        sag_data_np = pd.read_excel(sag_file_path, header=None, engine='openpyxl').values
        if sag_data_np.shape != self.grid_shape:
            sag_data_np = sag_data_np.reshape(self.grid_shape)

        # 垂直翻转矢高数据以匹配 meshgrid 网格坐标系   
        sag_data_np = np.flip(sag_data_np, axis=0) # 沿着第一个轴（行）进行翻转

        # 3. 创建坐标网格
        H, W = self.grid_shape
        x_np = np.linspace(-self.semi_dia, self.semi_dia, W)
        y_np = np.linspace(-self.semi_dia, self.semi_dia, H)
        self.xx, self.yy = np.meshgrid(x_np, y_np)

        # 4. 使用 RectBivariateSpline 进行 B 样条拟合
        self.spline = RectBivariateSpline(y_np, x_np, sag_data_np, kx=kx, ky=ky, s=smooth_s)
        self.coeff = torch.tensor(self.spline.get_coeffs(), device=self.device, dtype=torch.float64)

        # 5. 验证拟合误差
        reconstructed_sag = self.spline(y_np, x_np)
        rms_error = np.sqrt(np.mean((sag_data_np - reconstructed_sag)**2))
        print(f"B-spline fitting completed (kx={kx}, ky={ky}, s={smooth_s})")
        print(f"Fitting RMS error: {rms_error * 1e6:.4f} nm")
        if rms_error * 1e6 > 500:
            print(f"Warning: Fitting RMS error (={rms_error * 1e6:.1f} nm) is large, try increasing smoothing parameter s")
                                                                            
        # 默认不将重构数据写盘，避免运行过程污染工作区。
        # 如需导出调试，可设置环境变量 BIOT_SAVE_RECONSTRUCTED_SAG=1。
        if os.environ.get("BIOT_SAVE_RECONSTRUCTED_SAG", "0") == "1":
            df_reconstructed = pd.DataFrame(np.flip(reconstructed_sag, axis=0))
            save_path = "reconstructed_sag.xlsx"
            df_reconstructed.to_excel(save_path, index=False, header=False)
            print(f"B-spline reconstructed sag data saved to: {save_path}")

        # 存储绘图和 get_surface 所需的数据
        self.sag_data_np = sag_data_np
        self.reconstructed_sag = reconstructed_sag

        # self.plot_surface()
        

    def get_surface(self, x, y):
        """
        计算由 B-spline 定义的表面矢高及其偏导数。
        这是与父类牛顿法求解器接口的关键方法。

        Args:
            x (torch.Tensor): x 坐标
            y (torch.Tensor): y 坐标

        Returns:
            tuple: (g, -dgdx, -dgdy, dhdz)
                   g: 矢高 z = g(x, y)
                   -dgdx: 法向量的 x 分量
                   -dgdy: 法向量的 y 分量
                   dhdz: 法向量的 z 分量 (恒为 1)
        """
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        # 函数值与一阶偏导
        g_np    = self.spline.ev(y_np, x_np)                 # g(x,y)
        dgdx_np = self.spline.ev(y_np, x_np, dx=0, dy=1)     # ∂g/∂x
        dgdy_np = self.spline.ev(y_np, x_np, dx=1, dy=0)    # ∂g/∂y

        # 回到 torch（保持 dtype/device）
        g    = torch.tensor(g_np,    device=self.device, dtype=x.dtype)
        dgdx = torch.tensor(dgdx_np, device=self.device, dtype=x.dtype)
        dgdy = torch.tensor(dgdy_np, device=self.device, dtype=x.dtype)

        dhdz = torch.ones_like(x, device=self.device)
        return g, -dgdx, -dgdy, dhdz
    

    def plot_surface(self, plot_original=True, plot_reconstructed=True, plot_difference=False, save_path=None):
        """
        可视化原始矢高数据、B-spline 拟合重建的表面以及两者之间的差异。
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        legend_handles = []

        if plot_original:
            ax.plot_surface(self.xx, self.yy, self.sag_data_np, cmap='viridis', alpha=0.7)
            legend_handles.append(Patch(color=plt.cm.viridis(0.5), label='Original Sag'))
        
        if plot_reconstructed:
            ax.plot_surface(self.xx, self.yy, self.reconstructed_sag, cmap='plasma', alpha=0.7)
            legend_handles.append(Patch(color=plt.cm.plasma(0.5), label='Reconstructed Sag'))

        if plot_difference:
            difference = self.sag_data_np - self.reconstructed_sag
            ax.plot_surface(self.xx, self.yy, difference * 1e6, cmap='coolwarm')
            legend_handles.append(Patch(color=plt.cm.coolwarm(0.5), label='Difference (nm)'))
            ax.set_zlabel('Sag / diff (nm)')
        else:
            ax.set_zlabel('Sag (mm)')

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Sag Surface Visualization (B-spline)')
        if legend_handles:
            ax.legend(handles=legend_handles)
        ax.view_init(elev=30, azim=-60)

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

class Gradient_3(Surface):
    """
    "渐变3" (Gradient 3) 表面，它定义了一个非球面，并附加了梯度折射率材料属性。
    该梯度折射率模型遵循 Zemax 的 "Gradient 3" 定义。

    折射率公式:
    n(r, z) = n0 + Nr2*r^2 + Nr4*r^4 + Nr6*r^6 + Nz1*z + Nz2*z^2 + Nz3*z^3
    其中 r^2 = x^2 + y^2, z 是从表面顶点开始计算的轴向距离。

    表面形状 (sag) 与 Asphere 类相同:
    g(x,y) = c * r**2 / (1 + sqrt( 1 - (1+k) * r**2*c**2 )) + ai[0] * r**4 + ...

    Args:
        n0: 基础折射率
        Nr2: 径向二次梯度系数
        Nr4: 径向四次梯度系数
        Nr6: 径向六次梯度系数
        Nz1: 轴向一次梯度系数
        Nz2: 轴向二次梯度系数
        Nz3: 轴向三次梯度系数
        delta_t: Zemax 的梯度积分步长 Δt (mm)，<=0 时回退到默认步长
        其他参数与 Asphere 类相同。
    """

    def __init__(self, semi_dia, position, thickness, c=0, k=0, n0=1.0, Nr2=0.0, Nr4=0.0, Nr6=0.0,
                 Nz1=0.0, Nz2=0.0, Nz3=0.0, delta_t=0.0, coeff_list=None, is_square=False,
                 material_name=None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        # 调用父类 Surface 的构造函数
        Surface.__init__(self, semi_dia, position, thickness, is_square=is_square, device=device)
        
        # 设置表面类型为 'GRIN3'
        self.type = 'GRIN3'
        self.isaperture = False
        # 初始化非球面形状参数
        self.c = torch.tensor(c)
        self.k = torch.tensor(k)
        self.coeff = None
        self.m = None
        if coeff_list is not None:
            self.coeff = torch.Tensor(np.array(coeff_list)).to(device)
            self.m = len(coeff_list)

        # 初始化梯度折射率参数
        self.n0 = torch.tensor(n0, device=device)
        self.Nr2 = torch.tensor(Nr2, device=device)
        self.Nr4 = torch.tensor(Nr4, device=device)
        self.Nr6 = torch.tensor(Nr6, device=device)
        self.Nz1 = torch.tensor(Nz1, device=device)
        self.Nz2 = torch.tensor(Nz2, device=device)
        self.Nz3 = torch.tensor(Nz3, device=device)
        # Zemax "Gradient 3" 的 Δt 积分步长（mm）。<=0 表示未指定，由调用方给默认值。
        self.delta_t = float(delta_t)
        self.material_name = None if material_name is None else str(material_name).strip().lower()

    def axial_ior(self, wavelength=None):
        """Return the per-surface on-axis entrance index at local z=0."""
        zero = torch.zeros((), dtype=self.n0.dtype, device=self.n0.device)
        return self.get_ior(zero, zero, zero)

    def get_ior(self, x, y, z, wavelength=None):
        """
        计算在局部坐标 (x, y, z) 处的折射率。
        注意：这里的 z 是相对于表面顶点的局部 z 坐标。
        """
        r2 = x**2 + y**2
        n = (self.n0
             + self.Nr2 * r2 + self.Nr4 * r2**2 + self.Nr6 * r2**3
             + self.Nz1 * z + self.Nz2 * z**2 + self.Nz3 * z**3)
        return n

    def Da(self, m, z_surface=None):
        """
        计算折射率 n^2 对 x, y, z 的偏导数。
        返回一个包含 [dn^2/dx, dn^2/dy, dn^2/dz] 的张量。
        这对于使用 Runge-Kutta 等方法进行梯度介质中的光线追迹是必需的。
        此方法覆盖了 Surface 基类中的 Da 方法。
        
        Args:
            m: 位置向量（全局坐标）
            z_surface: 表面顶点的全局z坐标（用于计算局部z）
        """
        x, y, z_global = m[..., 0], m[..., 1], m[..., 2]
        
        # 转换为局部z坐标（从表面顶点开始）
        if z_surface is not None:
            z_local = z_global - z_surface
        else:
            z_local = z_global  # 兼容旧代码
        
        r2 = x**2 + y**2
        n = self.get_ior(x, y, z_local)
        
        # 计算 n 对 x, y, z 的偏导数（注意：对z_local求导）
        dn_dr2 = self.Nr2 + 2 * self.Nr4 * r2 + 3 * self.Nr6 * r2**2
        dn_dx = 2 * x * dn_dr2
        dn_dy = 2 * y * dn_dr2
        dn_dz = self.Nz1 + 2 * self.Nz2 * z_local + 3 * self.Nz3 * z_local**2
        
        # 使用链式法则计算 dn^2/dx, dn^2/dy, dn^2/dz
        dn2_dx = 2 * n * dn_dx
        dn2_dy = 2 * n * dn_dy
        dn2_dz = 2 * n * dn_dz
        
        return torch.stack([dn2_dx, dn2_dy, dn2_dz], dim=-1)

    # --- 以下是从 Asphere 类复制的表面形状定义方法 ---

    def distance_2(self, x, y): #径向距离平方
        if torch.is_tensor(x) and torch.is_tensor(y):
            x = x
            y = y
        else:
            x, y = (torch.Tensor(np.array(v)) for v in [x, y])
        return x ** 2 + y ** 2

    def get_sphere(self, x, y):
        Q = self.k + 1
        S_2 = self.distance_2(x, y)
        g_sp = self.c * S_2 / (1 + torch.sqrt(1 - (Q) * S_2 * self.c ** 2))
        dg_spdS_2 = self.c / (2 * torch.sqrt(1 - Q * S_2 * self.c ** 2))
        dS_2dx = 2 * x
        dS_2dy = 2 * y
        dg_spdx = dg_spdS_2 * dS_2dx
        dg_spdy = dg_spdS_2 * dS_2dy
        return g_sp, dg_spdx, dg_spdy

    def get_polynomial(self, x, y):
        S_2 = self.distance_2(x, y)
        g_poly = 0
        dg_polydS_2 = 0
        if self.m:
            for i in range(self.m):
                g_poly = g_poly + self.coeff[i] * S_2 ** (i + 2)
                dg_polydS_2 = dg_polydS_2 + self.coeff[i] * (i + 2) * S_2 ** (i + 1)
        dS_2dx = 2 * x
        dS_2dy = 2 * y
        dg_polydx = dg_polydS_2 * dS_2dx
        dg_polydy = dg_polydS_2 * dS_2dy
        return g_poly, dg_polydx, dg_polydy

    def get_surface(self, x, y):
        g_sp, dg_spdx, dg_spdy = self.get_sphere(x, y)
        g_poly, dg_polydx, dg_polydy = self.get_polynomial(x, y)
        g = g_sp + g_poly
        dgdx = dg_spdx + dg_polydx
        dgdy = dg_spdy + dg_polydy
        dhdz = torch.ones_like(x)
        return g, -dgdx, -dgdy, dhdz

    def get_sag(self, x, y):
        Q = self.k + 1
        S_2 = self.distance_2(x, y)
        g_sp = self.c * S_2 / (1 + torch.sqrt(1 - (Q) * S_2 * self.c ** 2))
        g_poly = 0
        if self.m:
            for i in range(self.m):
                g_poly = g_poly + self.coeff[i] * S_2 ** (i + 2)
        g = g_sp + g_poly
        return g

    def get_derivatives(self, x, y):
        Q = self.k + 1
        S_2 = self.distance_2(x, y)
        dg_spdS_2 = self.c / (2 * torch.sqrt(1 - Q * S_2 * self.c ** 2))
        dS_2dx = 2 * x
        dS_2dy = 2 * y
        dg_spdx = dg_spdS_2 * dS_2dx
        dg_spdy = dg_spdS_2 * dS_2dy

        dg_polydS_2 = 0
        if self.m:
            for i in range(self.m):
                dg_polydS_2 = dg_polydS_2 + self.coeff[i] * (i + 2) * S_2 ** (i + 1)

        dg_polydx = dg_polydS_2 * dS_2dx
        dg_polydy = dg_polydS_2 * dS_2dy

        dgdx = dg_spdx + dg_polydx
        dgdy = dg_spdy + dg_polydy
        dhdz = torch.ones_like(x)
        return -dgdx, -dgdy, dhdz

    def reverse(self):
        self.c = -self.c
        if self.coeff is not None:
            self.coeff = -self.coeff
