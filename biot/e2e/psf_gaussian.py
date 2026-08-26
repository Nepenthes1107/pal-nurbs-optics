from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PSFGrid:
    """Square image-plane PSF grid definition.

    Units:
        size_px: number of pixels along each side.
        pixel_pitch_mm: physical pixel pitch on image plane, in mm/pixel.

    Coordinate convention:
        Pixel coordinates are centered at zero. For an odd grid, one pixel
        center lies exactly at ``x=y=0``. For an even grid, zero lies between
        the central four pixels. Output coordinate arrays have shape ``[H, W]``.
    """

    size_px: int
    pixel_pitch_mm: float

    def __post_init__(self) -> None:
        if int(self.size_px) <= 0:
            raise ValueError("size_px must be positive")
        if float(self.pixel_pitch_mm) <= 0:
            raise ValueError("pixel_pitch_mm must be positive")

    def coordinates(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.arange(int(self.size_px), device=device, dtype=dtype)
        center = (float(self.size_px) - 1.0) / 2.0
        coord = (idx - center) * float(self.pixel_pitch_mm)
        y_mm, x_mm = torch.meshgrid(coord, coord, indexing="ij")
        return x_mm, y_mm


def gaussianized_ray_landing_psf(
    spots_mm: torch.Tensor,
    weights: torch.Tensor | None,
    *,
    psf_size_px: int,
    pixel_pitch_mm: float,
    sigma_mm: float | None = None,
    sigma_px: float | None = None,
    valid: torch.Tensor | None = None,
    eps: float = 1.0e-30,
    ray_chunk_size: int | None = None,
) -> torch.Tensor:
    """Convert ray landing spots into an energy-normalized differentiable PSF.

    Formula:
        ``PSF[m,n] = sum_i w_i exp(-||pixel[m,n] - spot_i||^2 / (2 sigma^2))``

    Args:
        spots_mm: Ray landing coordinates in mm, shape ``[N, 2]`` or
            ``[B, N, 2]``.
        weights: Ray energy weights, shape ``[N]`` or ``[B, N]``. If ``None``,
            uniform weights are used.
        psf_size_px: Output square PSF side length in pixels.
        pixel_pitch_mm: Image-plane pixel pitch in mm/pixel.
        sigma_mm: Gaussian sigma in mm.
        sigma_px: Gaussian sigma in pixels. Used only when ``sigma_mm`` is not
            provided.
        valid: Optional valid ray mask, same shape as weights. Invalid rays
            contribute zero energy.
        ray_chunk_size: Optional number of rays accumulated per chunk. This
            preserves the exact Gaussian sum while avoiding allocation of the
            full ``[B, N, H, W]`` kernel tensor for large validation runs.

    Returns:
        Energy-normalized PSF with shape ``[H, W]`` for unbatched input or
        ``[B, H, W]`` for batched input. The normalization is total-energy
        normalization and is the physical tensor to use for losses/MTF.
    """
    if spots_mm.shape[-1] != 2:
        raise ValueError("spots_mm must have shape [N, 2] or [B, N, 2]")
    if sigma_mm is None:
        if sigma_px is None:
            sigma_mm = float(pixel_pitch_mm)
        else:
            sigma_mm = float(sigma_px) * float(pixel_pitch_mm)
    if float(sigma_mm) <= 0:
        raise ValueError("sigma must be positive")

    batched = spots_mm.ndim == 3
    if spots_mm.ndim == 2:
        spots = spots_mm.unsqueeze(0)
    elif spots_mm.ndim == 3:
        spots = spots_mm
    else:
        raise ValueError("spots_mm must be rank 2 or 3")

    if weights is None:
        ray_weights = torch.ones(spots.shape[:-1], device=spots.device, dtype=spots.dtype)
    else:
        ray_weights = weights.to(device=spots.device, dtype=spots.dtype)
        if ray_weights.ndim == 1:
            ray_weights = ray_weights.unsqueeze(0)
    if ray_weights.shape != spots.shape[:-1]:
        raise ValueError("weights must match spots ray dimensions")
    if valid is not None:
        valid_mask = valid.to(device=spots.device, dtype=torch.bool)
        if valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)
        if valid_mask.shape != ray_weights.shape:
            raise ValueError("valid mask must match weights shape")
        ray_weights = ray_weights * valid_mask.to(spots.dtype)

    grid = PSFGrid(psf_size_px, pixel_pitch_mm)
    x_mm, y_mm = grid.coordinates(device=spots.device, dtype=spots.dtype)
    if ray_chunk_size is None:
        dx = x_mm.reshape(1, 1, int(psf_size_px), int(psf_size_px)) - spots[..., 0].reshape(*spots.shape[:-1], 1, 1)
        dy = y_mm.reshape(1, 1, int(psf_size_px), int(psf_size_px)) - spots[..., 1].reshape(*spots.shape[:-1], 1, 1)
        exponent = -(dx.pow(2) + dy.pow(2)) / (2.0 * float(sigma_mm) ** 2)
        kernels = torch.exp(exponent)
        psf = (kernels * ray_weights[..., None, None]).sum(dim=1)
    else:
        chunk = int(ray_chunk_size)
        if chunk <= 0:
            raise ValueError("ray_chunk_size must be positive")
        psf = torch.zeros(
            (spots.shape[0], int(psf_size_px), int(psf_size_px)),
            device=spots.device,
            dtype=spots.dtype,
        )
        x_grid = x_mm.reshape(1, 1, int(psf_size_px), int(psf_size_px))
        y_grid = y_mm.reshape(1, 1, int(psf_size_px), int(psf_size_px))
        for start in range(0, spots.shape[1], chunk):
            end = min(start + chunk, spots.shape[1])
            spot_chunk = spots[:, start:end, :]
            weight_chunk = ray_weights[:, start:end]
            dx = x_grid - spot_chunk[..., 0].reshape(spot_chunk.shape[0], spot_chunk.shape[1], 1, 1)
            dy = y_grid - spot_chunk[..., 1].reshape(spot_chunk.shape[0], spot_chunk.shape[1], 1, 1)
            exponent = -(dx.pow(2) + dy.pow(2)) / (2.0 * float(sigma_mm) ** 2)
            psf = psf + (torch.exp(exponent) * weight_chunk[..., None, None]).sum(dim=1)
    energy = psf.sum(dim=(-2, -1), keepdim=True)
    if not torch.all(energy > float(eps)):
        raise ValueError("PSF energy is zero; check spots, weights, and valid mask")
    psf = psf / energy
    return psf if batched else psf.squeeze(0)


def psf_centroid_mm(psf: torch.Tensor, pixel_pitch_mm: float) -> torch.Tensor:
    """Compute PSF centroid in image-plane mm coordinates.

    Args:
        psf: Energy-normalized PSF with shape ``[H, W]`` or ``[B, H, W]``.
        pixel_pitch_mm: Physical pixel pitch in mm/pixel.

    Returns:
        Centroid tensor ``[2]`` or ``[B, 2]`` ordered as ``(x_mm, y_mm)``.
    """
    if psf.ndim == 2:
        batched = False
        psf_eval = psf.unsqueeze(0)
    elif psf.ndim == 3:
        batched = True
        psf_eval = psf
    else:
        raise ValueError("psf must have shape [H, W] or [B, H, W]")
    h, w = psf_eval.shape[-2:]
    if h != w:
        raise ValueError("psf must be square")
    grid = PSFGrid(h, pixel_pitch_mm)
    x_mm, y_mm = grid.coordinates(device=psf_eval.device, dtype=psf_eval.dtype)
    energy = psf_eval.sum(dim=(-2, -1)).clamp_min(torch.finfo(psf_eval.dtype).eps)
    cx = (psf_eval * x_mm).sum(dim=(-2, -1)) / energy
    cy = (psf_eval * y_mm).sum(dim=(-2, -1)) / energy
    centroid = torch.stack((cx, cy), dim=-1)
    return centroid if batched else centroid.squeeze(0)
