from __future__ import annotations

from dataclasses import dataclass

import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TorchFFTPSFResult:
    """Torch FFT PSF diagnostic bundle.

    Shapes:
        complex_pupil: ``[P, P]`` complex pupil field.
        psf: ``[H, W]`` energy-normalized image-plane intensity.
        valid_pupil: ``[P, P]`` boolean aperture and valid-ray mask.

    Units:
        phase_rad is in radians. PSF pixel spacing is supplied by the caller
        and is not used by the FFT itself.
    """

    complex_pupil: torch.Tensor
    psf: torch.Tensor
    valid_pupil: torch.Tensor


def standardize_fft_psf_orientation(psf: torch.Tensor) -> torch.Tensor:
    """Reflect PSF rows about the FFT DC index, preserving differentiability.

    The stable BIOT mapping is ``r -> (n-r) mod n``. For an even-sized array
    this is a vertical flip followed by a one-row roll; bare ``torch.flip`` is
    one pixel wrong.
    """
    if psf.ndim < 2:
        raise ValueError("PSF must have at least two dimensions")
    return torch.roll(torch.flip(psf, dims=(-2,)), shifts=1, dims=(-2,))


def circular_pupil_mask(
    sample_count: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return BIOT-style circular mask on a square normalized pupil grid."""
    if int(sample_count) <= 0:
        raise ValueError("sample_count must be positive")
    coord = torch.linspace(-1.0, 1.0, int(sample_count), device=device, dtype=dtype)
    xx, yy = torch.meshgrid(coord, coord, indexing="xy")
    return xx.pow(2) + yy.pow(2) <= 1.0


def complex_pupil_from_phase(
    phase_rad: torch.Tensor,
    valid: torch.Tensor | None,
    *,
    sample_count: int,
    remove_piston: bool = True,
    remove_tilt: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a square complex pupil from flattened circular-pupil phases.

    The flattened ray order matches ``pupil_disk_grid()``: normalized square
    grid samples are masked by ``R <= 1`` and then flattened. Values outside
    the aperture, or marked invalid, are zero. The output remains
    differentiable with respect to ``phase_rad``.
    """
    if phase_rad.ndim != 1:
        raise ValueError("phase_rad must be a 1D tensor matching pupil rays")
    aperture = circular_pupil_mask(sample_count, device=phase_rad.device, dtype=phase_rad.dtype)
    ray_count = int(aperture.sum().detach().cpu().item())
    if phase_rad.numel() != ray_count:
        raise ValueError(f"phase length {phase_rad.numel()} does not match circular pupil ray count {ray_count}")

    valid_flat = torch.ones_like(phase_rad, dtype=torch.bool) if valid is None else valid.to(device=phase_rad.device, dtype=torch.bool)
    if valid_flat.shape != phase_rad.shape:
        raise ValueError("valid must match phase_rad shape")

    phase = _remove_phase_plane(
        phase_rad,
        valid_flat,
        aperture,
        sample_count=int(sample_count),
        remove_piston=remove_piston,
        remove_tilt=remove_tilt,
    )

    flat_field = torch.zeros((int(sample_count) * int(sample_count),), device=phase_rad.device, dtype=torch.complex128)
    phasors = torch.exp(1j * phase.to(torch.complex128)) * valid_flat.to(torch.complex128)
    flat_mask = aperture.reshape(-1)
    flat_field = flat_field.masked_scatter(flat_mask, phasors)
    pupil = flat_field.reshape(int(sample_count), int(sample_count))
    valid_pupil_flat = torch.zeros_like(flat_mask, dtype=torch.bool)
    valid_pupil_flat = valid_pupil_flat.masked_scatter(flat_mask, valid_flat)
    return pupil, valid_pupil_flat.reshape_as(aperture)


def _aperture_xy_coordinates(
    aperture: torch.Tensor,
    sample_count: int,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    coord = torch.linspace(-1.0, 1.0, int(sample_count), device=aperture.device, dtype=dtype)
    xx, yy = torch.meshgrid(coord, coord, indexing="xy")
    flat_mask = aperture.reshape(-1)
    return xx.reshape(-1)[flat_mask], yy.reshape(-1)[flat_mask]


def _remove_phase_plane(
    phase_rad: torch.Tensor,
    valid_flat: torch.Tensor,
    aperture: torch.Tensor,
    *,
    sample_count: int,
    remove_piston: bool,
    remove_tilt: bool,
) -> torch.Tensor:
    if not remove_tilt:
        if remove_piston and torch.any(valid_flat):
            return phase_rad - phase_rad[valid_flat].mean()
        return phase_rad

    x, y = _aperture_xy_coordinates(aperture, int(sample_count), dtype=phase_rad.dtype)
    slope_x, slope_y = _estimate_wrapped_phase_slopes(
        phase_rad,
        valid_flat,
        aperture,
        sample_count=int(sample_count),
    )
    phase = phase_rad - (slope_x * x + slope_y * y)
    if remove_piston and torch.any(valid_flat):
        phase = phase - phase[valid_flat].mean()
    return phase


def _estimate_wrapped_phase_slopes(
    phase_rad: torch.Tensor,
    valid_flat: torch.Tensor,
    aperture: torch.Tensor,
    *,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate linear phase slopes from complex adjacent differences.

    The input phase can span many 2*pi cycles. Fitting a plane directly to the
    wrapped absolute phase is unstable, so this estimates the local phase
    increment from neighboring complex phasor ratios and converts that
    increment to radians per normalized pupil coordinate.
    """
    n = int(sample_count)
    flat_phase = torch.zeros((n * n,), device=phase_rad.device, dtype=phase_rad.dtype)
    flat_valid = torch.zeros((n * n,), device=phase_rad.device, dtype=torch.bool)
    flat_mask = aperture.reshape(-1)
    flat_phase = flat_phase.masked_scatter(flat_mask, phase_rad)
    flat_valid = flat_valid.masked_scatter(flat_mask, valid_flat)
    phase_grid = flat_phase.reshape(n, n)
    valid_grid = flat_valid.reshape(n, n)
    phasor = torch.exp(1j * phase_grid.to(torch.complex128))
    step = torch.as_tensor(2.0 / float(max(1, n - 1)), device=phase_rad.device, dtype=phase_rad.dtype)

    x_pairs = valid_grid[:, 1:] & valid_grid[:, :-1]
    y_pairs = valid_grid[1:, :] & valid_grid[:-1, :]
    slope_x = _mean_wrapped_increment(phasor[:, 1:] * torch.conj(phasor[:, :-1]), x_pairs, step, phase_rad)
    slope_y = _mean_wrapped_increment(phasor[1:, :] * torch.conj(phasor[:-1, :]), y_pairs, step, phase_rad)
    return slope_x, slope_y


def _mean_wrapped_increment(
    unit_steps: torch.Tensor,
    mask: torch.Tensor,
    step: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not torch.any(mask):
        return torch.zeros((), device=reference.device, dtype=reference.dtype)
    mean_step = unit_steps[mask].mean()
    return torch.angle(mean_step).to(dtype=reference.dtype) / step


def torch_fft_psf_from_complex_pupil(
    complex_pupil: torch.Tensor,
    *,
    psf_size_px: int,
    standardize_orientation: bool = True,
    eps: float = 1.0e-30,
) -> torch.Tensor:
    """Compute an energy-normalized PSF from a complex pupil using torch FFT.

    Padding intentionally follows BIOT ``Lensdata._pad_pupils()`` closely:
    a symmetric pad of ``(psf_size_px - pupil_size) // 2 + 1`` is applied, and
    the FFT result is center-cropped back to ``psf_size_px`` if needed.
    """
    if complex_pupil.ndim != 2:
        raise ValueError("complex_pupil must be a 2D tensor")
    if complex_pupil.shape[0] != complex_pupil.shape[1]:
        raise ValueError("complex_pupil must be square")
    if int(psf_size_px) <= 0:
        raise ValueError("psf_size_px must be positive")

    pupil_size = int(complex_pupil.shape[0])
    pad = (int(psf_size_px) - pupil_size) // 2 + 1
    if pad < 0:
        raise ValueError("psf_size_px must be at least the pupil size")
    padded = F.pad(complex_pupil, (pad, pad, pad, pad))
    amplitude = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(padded)))
    psf = amplitude.real.pow(2) + amplitude.imag.pow(2)
    if psf.shape[-1] != int(psf_size_px):
        current_h, current_w = psf.shape
        start_h = (current_h - int(psf_size_px)) // 2
        start_w = (current_w - int(psf_size_px)) // 2
        psf = psf[start_h : start_h + int(psf_size_px), start_w : start_w + int(psf_size_px)]
    if standardize_orientation:
        psf = standardize_fft_psf_orientation(psf)
    energy = psf.sum()
    if not torch.isfinite(energy) or energy <= float(eps):
        raise ValueError("FFT PSF energy is zero or non-finite")
    return psf / energy


def torch_fft_psf_from_phase(
    phase_rad: torch.Tensor,
    valid: torch.Tensor | None,
    *,
    sample_count: int,
    psf_size_px: int,
    remove_piston: bool = True,
    remove_tilt: bool = False,
) -> TorchFFTPSFResult:
    """Build a complex pupil from phase and compute a torch FFT PSF."""
    pupil, valid_pupil = complex_pupil_from_phase(
        phase_rad,
        valid,
        sample_count=sample_count,
        remove_piston=remove_piston,
        remove_tilt=remove_tilt,
    )
    psf = torch_fft_psf_from_complex_pupil(pupil, psf_size_px=psf_size_px)
    return TorchFFTPSFResult(complex_pupil=pupil, psf=psf, valid_pupil=valid_pupil)
def effective_biot_pupil_sample_count(requested_np: int) -> int:
    """Match BIOT ``fft_psf_i`` pupil-side sampling for a requested ``np``."""
    requested = int(requested_np)
    if requested <= 0:
        raise ValueError("requested pupil sampling must be positive")
    exponent = math.log(requested / 32.0, 2.0)
    return int(32.0 * ((2.0**0.5) ** exponent))

