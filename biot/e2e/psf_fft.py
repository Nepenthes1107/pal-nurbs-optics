from __future__ import annotations

from dataclasses import dataclass

import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TorchFFTPSFResult:
    """Torch FFT PSF diagnostic bundle.

    Shapes:
        complex_pupil: ``[..., P, P]`` complex pupil field.
        psf: ``[..., H, W]`` energy-normalized image-plane intensity.
        valid_pupil: ``[..., P, P]`` boolean aperture and valid-ray mask.

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
    if phase_rad.ndim < 1:
        raise ValueError("phase_rad must have shape [..., pupil_rays]")
    aperture = circular_pupil_mask(sample_count, device=phase_rad.device, dtype=phase_rad.dtype)
    ray_count = int(aperture.sum().detach().cpu().item())
    if int(phase_rad.shape[-1]) != ray_count:
        raise ValueError(
            f"phase length {int(phase_rad.shape[-1])} does not match circular pupil ray count {ray_count}"
        )

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

    batch_shape = tuple(int(value) for value in phase_rad.shape[:-1])
    flat_shape = (*batch_shape, int(sample_count) * int(sample_count))
    flat_field = torch.zeros(flat_shape, device=phase_rad.device, dtype=torch.complex128)
    phasors = torch.exp(1j * phase.to(torch.complex128)) * valid_flat.to(torch.complex128)
    flat_mask = aperture.reshape(-1).expand(flat_shape)
    flat_field = flat_field.masked_scatter(flat_mask, phasors.reshape(-1))
    pupil = flat_field.reshape(*batch_shape, int(sample_count), int(sample_count))
    valid_pupil_flat = torch.zeros(flat_shape, device=phase_rad.device, dtype=torch.bool)
    valid_pupil_flat = valid_pupil_flat.masked_scatter(flat_mask, valid_flat.reshape(-1))
    valid_pupil = valid_pupil_flat.reshape(
        *batch_shape, int(sample_count), int(sample_count)
    )
    return pupil, valid_pupil


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
    if not bool(valid_flat.any(dim=-1).all()):
        raise ValueError("each pupil case must contain at least one valid ray")
    if not remove_tilt:
        if remove_piston:
            count = valid_flat.sum(dim=-1).to(dtype=phase_rad.dtype)
            mean = torch.where(valid_flat, phase_rad, torch.zeros_like(phase_rad)).sum(dim=-1) / count
            return phase_rad - mean.unsqueeze(-1)
        return phase_rad

    x, y = _aperture_xy_coordinates(aperture, int(sample_count), dtype=phase_rad.dtype)
    slope_x, slope_y = _estimate_wrapped_phase_slopes(
        phase_rad,
        valid_flat,
        aperture,
        sample_count=int(sample_count),
    )
    phase = phase_rad - (slope_x.unsqueeze(-1) * x + slope_y.unsqueeze(-1) * y)
    if remove_piston:
        count = valid_flat.sum(dim=-1).to(dtype=phase.dtype)
        mean = torch.where(valid_flat, phase, torch.zeros_like(phase)).sum(dim=-1) / count
        phase = phase - mean.unsqueeze(-1)
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
    batch_shape = tuple(int(value) for value in phase_rad.shape[:-1])
    flat_shape = (*batch_shape, n * n)
    flat_phase = torch.zeros(flat_shape, device=phase_rad.device, dtype=phase_rad.dtype)
    flat_valid = torch.zeros(flat_shape, device=phase_rad.device, dtype=torch.bool)
    flat_mask = aperture.reshape(-1).expand(flat_shape)
    flat_phase = flat_phase.masked_scatter(flat_mask, phase_rad.reshape(-1))
    flat_valid = flat_valid.masked_scatter(flat_mask, valid_flat.reshape(-1))
    phase_grid = flat_phase.reshape(*batch_shape, n, n)
    valid_grid = flat_valid.reshape(*batch_shape, n, n)
    phasor = torch.exp(1j * phase_grid.to(torch.complex128))
    step = torch.as_tensor(2.0 / float(max(1, n - 1)), device=phase_rad.device, dtype=phase_rad.dtype)

    x_pairs = valid_grid[..., :, 1:] & valid_grid[..., :, :-1]
    y_pairs = valid_grid[..., 1:, :] & valid_grid[..., :-1, :]
    slope_x = _mean_wrapped_increment(
        phasor[..., :, 1:] * torch.conj(phasor[..., :, :-1]),
        x_pairs,
        step,
        phase_rad,
    )
    slope_y = _mean_wrapped_increment(
        phasor[..., 1:, :] * torch.conj(phasor[..., :-1, :]),
        y_pairs,
        step,
        phase_rad,
    )
    return slope_x, slope_y


def _mean_wrapped_increment(
    unit_steps: torch.Tensor,
    mask: torch.Tensor,
    step: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    reduce_dims = (-2, -1)
    count = mask.sum(dim=reduce_dims)
    summed = torch.where(mask, unit_steps, torch.zeros_like(unit_steps)).sum(dim=reduce_dims)
    safe_count = count.clamp_min(1).to(dtype=summed.real.dtype)
    mean_step = summed / safe_count
    slope = torch.angle(mean_step).to(dtype=reference.dtype) / step
    return torch.where(count > 0, slope, torch.zeros_like(slope))


def torch_fft_psf_from_complex_pupil(
    complex_pupil: torch.Tensor,
    *,
    psf_size_px: int,
    standardize_orientation: bool = True,
    eps: float = 1.0e-30,
) -> torch.Tensor:
    """Compute an energy-normalized PSF from a complex pupil using torch FFT.

    Padding exactly follows BIOT ``Lensdata._pad_pupils()``: the pupil is
    centered in an array whose final shape is exactly ``psf_size_px``.  For an
    odd padding remainder, the extra sample is placed on the high-index side.
    """
    if complex_pupil.ndim < 2:
        raise ValueError("complex_pupil must have shape [..., P, P]")
    if complex_pupil.shape[-2] != complex_pupil.shape[-1]:
        raise ValueError("complex_pupil must be square")
    if int(psf_size_px) <= 0:
        raise ValueError("psf_size_px must be positive")

    pupil_size = int(complex_pupil.shape[-1])
    total_pad = int(psf_size_px) - pupil_size
    if total_pad < 0:
        raise ValueError("psf_size_px must be at least the pupil size")
    pad_low = total_pad // 2
    pad_high = total_pad - pad_low
    padded = F.pad(
        complex_pupil,
        (pad_low, pad_high, pad_low, pad_high),
    )
    amplitude = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(padded, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )
    psf = amplitude.real.pow(2) + amplitude.imag.pow(2)
    if psf.shape[-1] != int(psf_size_px):
        current_h, current_w = psf.shape[-2:]
        start_h = (current_h - int(psf_size_px)) // 2
        start_w = (current_w - int(psf_size_px)) // 2
        psf = psf[
            ...,
            start_h : start_h + int(psf_size_px),
            start_w : start_w + int(psf_size_px),
        ]
    if standardize_orientation:
        psf = standardize_fft_psf_orientation(psf)
    energy = psf.sum(dim=(-2, -1), keepdim=True)
    if not bool(torch.isfinite(energy).all()) or bool((energy <= float(eps)).any()):
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
