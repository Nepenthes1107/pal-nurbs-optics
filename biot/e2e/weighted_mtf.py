"""Canonical Ahumada weighted-MTF metric shared by training and evaluation."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from optics import compute_dc_normalized_mtf


COMMON_FREQ_LPMM = np.linspace(0.0, 100.0, 1000, dtype=np.float64)
CSF_MM_PER_DEG = 0.291
CSF_F0 = 4.1726
CSF_F1 = 1.3625
CSF_A = 0.8493
CSF_P = 0.7786
CSF_GAIN = 373.08


def _ahumada_weight() -> np.ndarray:
    cycles_per_degree = COMMON_FREQ_LPMM * CSF_MM_PER_DEG
    sech = lambda value: 1.0 / np.cosh(value)
    weight = np.maximum(
        CSF_GAIN
        * (
            sech((cycles_per_degree / CSF_F0) ** CSF_P)
            - CSF_A * sech(cycles_per_degree / CSF_F1)
        ),
        0.0,
    )
    normalization = float(np.trapz(weight, COMMON_FREQ_LPMM))
    if not math.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("Ahumada CSF normalization is invalid")
    return weight / normalization


AHUMADA_WEIGHT_NORMALIZED = _ahumada_weight()


def _native_frequency(size: int, pitch_mm: float) -> np.ndarray:
    if isinstance(size, bool) or int(size) < 2:
        raise ValueError("weighted-MTF size must be at least two")
    if not math.isfinite(float(pitch_mm)) or float(pitch_mm) <= 0.0:
        raise ValueError("weighted-MTF pixel pitch must be finite and positive")
    center = int(size) // 2
    frequency = (
        1.0 / ((int(size) + 1) * float(pitch_mm))
    ) * np.arange(center + 1, dtype=np.float64)
    return frequency


@lru_cache(maxsize=512)
def _projection_weights(size: int, pitch_mm: float) -> np.ndarray:
    """Return the fixed linear map from a native MTF slice to its CSF score.

    Cubic interpolation and trapezoidal integration are linear in the sampled
    MTF values.  The matrix is therefore computed once from non-trainable FFT
    sampling metadata; PSF-dependent values remain in Torch/autograd.
    """
    frequency = _native_frequency(int(size), float(pitch_mm))
    frequency = frequency[: min(frequency.size, int(size) - int(size) // 2)]
    if frequency[-1] < float(COMMON_FREQ_LPMM[-1]):
        raise ValueError(
            f"native MTF support ends at {frequency[-1]:g} cycles/mm, below 100"
        )
    basis = np.eye(frequency.size, dtype=np.float64)
    interpolated = CubicSpline(
        frequency, basis, axis=0, extrapolate=False
    )(COMMON_FREQ_LPMM)
    if not np.isfinite(interpolated).all():
        raise ValueError("weighted-MTF interpolation projection is non-finite")
    projection = np.trapz(
        AHUMADA_WEIGHT_NORMALIZED[:, None] * interpolated,
        COMMON_FREQ_LPMM,
        axis=0,
    )
    projection = np.asarray(projection, dtype=np.float64)
    if not np.isfinite(projection).all():
        raise ValueError("weighted-MTF projection is non-finite")
    projection.setflags(write=False)
    return projection


def weighted_mtf_numpy(
    psf: np.ndarray, pitch_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return evaluator-compatible sagittal, tangential, and mean scores."""
    array = np.asarray(psf, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("weighted-MTF PSF must be a square 2-D array")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("weighted-MTF PSF must be finite and non-negative")
    energy = float(array.sum())
    if not math.isfinite(energy) or energy <= 0.0:
        raise ValueError("weighted-MTF PSF energy must be finite and positive")
    if abs(energy - 1.0) > 1.0e-10:
        raise ValueError(f"weighted-MTF PSF energy is not normalized: {energy}")
    mtf = np.asarray(compute_dc_normalized_mtf(array), dtype=np.float64)
    size, center = int(mtf.shape[0]), int(mtf.shape[0]) // 2
    frequency = _native_frequency(size, float(pitch_mm))
    sagittal = np.clip(
        np.asarray(mtf[center, center : center + center + 1], dtype=np.float64),
        0.0,
        1.0,
    )
    tangential = np.clip(
        np.asarray(mtf[center : center + center + 1, center], dtype=np.float64),
        0.0,
        1.0,
    )
    count = min(frequency.size, sagittal.size, tangential.size)
    frequency = frequency[:count]
    if frequency[-1] < float(COMMON_FREQ_LPMM[-1]):
        raise ValueError(
            f"native MTF support ends at {frequency[-1]:g} cycles/mm, below 100"
        )
    sagittal_common = CubicSpline(
        frequency, sagittal[:count], extrapolate=False
    )(COMMON_FREQ_LPMM)
    tangential_common = CubicSpline(
        frequency, tangential[:count], extrapolate=False
    )(COMMON_FREQ_LPMM)
    if not np.isfinite(sagittal_common).all() or not np.isfinite(tangential_common).all():
        raise ValueError("MTF interpolation produced non-finite values")
    sagittal_score = float(
        np.trapz(
            AHUMADA_WEIGHT_NORMALIZED * sagittal_common, COMMON_FREQ_LPMM
        )
    )
    tangential_score = float(
        np.trapz(
            AHUMADA_WEIGHT_NORMALIZED * tangential_common, COMMON_FREQ_LPMM
        )
    )
    scores = np.asarray(
        [
            sagittal_score,
            tangential_score,
            0.5 * (sagittal_score + tangential_score),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(scores).all():
        raise ValueError("weighted-MTF score is non-finite")
    return scores, sagittal_common, tangential_common


def weighted_mtf_mean_torch_batch(
    psf: torch.Tensor, *, pixel_pitch_mm: torch.Tensor,
) -> torch.Tensor:
    """Return differentiable evaluator-equivalent mean scores for ``[B,H,W]`` PSFs.

    ``pixel_pitch_mm`` is fixed FFT sampling metadata and must not require a
    gradient.  Only its cached interpolation projection is built outside the
    graph; all PSF-dependent operations remain Torch tensors.
    """
    if psf.ndim != 3 or int(psf.shape[-2]) != int(psf.shape[-1]):
        raise ValueError("weighted-MTF PSF batch must have square shape [B,H,W]")
    batch_size, size = int(psf.shape[0]), int(psf.shape[-1])
    if tuple(pixel_pitch_mm.shape) != (batch_size,):
        raise ValueError("pixel_pitch_mm must contain one value per PSF")
    if bool(pixel_pitch_mm.requires_grad):
        raise ValueError("pixel_pitch_mm is fixed sampling metadata and cannot require grad")
    if not bool(torch.isfinite(psf).all()) or bool((psf < 0.0).any()):
        raise ValueError("weighted-MTF PSF batch must be finite and non-negative")
    if not bool(torch.isfinite(pixel_pitch_mm).all()) or bool((pixel_pitch_mm <= 0.0).any()):
        raise ValueError("pixel_pitch_mm must be finite and positive")
    energy = psf.sum(dim=(-2, -1))
    if bool((energy - 1.0).abs().max() > 1.0e-10):
        raise ValueError("weighted-MTF PSF energy is not normalized")
    shifted = torch.fft.ifftshift(psf, dim=(-2, -1))
    otf = torch.fft.fftshift(
        torch.fft.fft2(shifted, dim=(-2, -1)), dim=(-2, -1)
    )
    magnitude = torch.abs(otf)
    center = size // 2
    dc = magnitude[:, center, center]
    if not bool(torch.isfinite(dc).all()) or bool((dc <= 0.0).any()):
        raise ValueError("weighted-MTF OTF DC must be finite and positive")
    mtf = magnitude / dc[:, None, None]
    sagittal = mtf[:, center, center : center + center + 1].clamp(0.0, 1.0)
    tangential = mtf[:, center : center + center + 1, center].clamp(0.0, 1.0)
    means: list[torch.Tensor] = []
    for index in range(batch_size):
        pitch = float(pixel_pitch_mm[index].detach().cpu())
        projection = torch.as_tensor(
            np.array(_projection_weights(size, pitch), copy=True),
            device=psf.device,
            dtype=psf.dtype,
        )
        count = int(projection.numel())
        sag_score = (sagittal[index, :count] * projection).sum()
        tan_score = (tangential[index, :count] * projection).sum()
        means.append(0.5 * (sag_score + tan_score))
    scores = torch.stack(means)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("weighted-MTF score is non-finite")
    return scores
