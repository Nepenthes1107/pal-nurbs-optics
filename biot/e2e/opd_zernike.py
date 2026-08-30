from __future__ import annotations

import math

import torch


LOW_ORDER_TERMS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, -1),
    (1, 1),
    (2, -2),
    (2, 0),
    (2, 2),
)
Z4_DEFOCUS_INDEX = LOW_ORDER_TERMS.index((2, 0))


def low_order_osa_ansi_basis(
    x_norm: torch.Tensor,
    y_norm: torch.Tensor,
) -> torch.Tensor:
    """Return the BIOT-compatible real RMS-normalized OSA/ANSI n<=2 basis.

    The last axis follows :data:`LOW_ORDER_TERMS`.  Coordinates are
    dimensionless normalized-pupil coordinates and must lie in the unit disk.
    """
    x, y = torch.broadcast_tensors(x_norm, y_norm)
    if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise ValueError("normalized pupil coordinates must be finite")
    rho2 = x.square() + y.square()
    if bool((rho2 > 1.0 + 1.0e-12).any()):
        raise ValueError("normalized pupil coordinates must lie inside the unit disk")
    one = torch.ones_like(x)
    return torch.stack(
        (
            one,
            2.0 * y,
            2.0 * x,
            math.sqrt(6.0) * (2.0 * x * y),
            math.sqrt(3.0) * (2.0 * rho2 - 1.0),
            math.sqrt(6.0) * (x.square() - y.square()),
        ),
        dim=-1,
    )


def normalized_pupil_disk_coordinates(
    sample_count: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened unit-disk coordinates in the PAL pupil-ray order."""
    count = int(sample_count)
    if count <= 0:
        raise ValueError("sample_count must be positive")
    coord = torch.linspace(-1.0, 1.0, count, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(coord, coord, indexing="xy")
    aperture = xx.square() + yy.square() <= 1.0
    return xx[aperture], yy[aperture]


def fit_low_order_opd_zernike_torch(
    reference_opl_mm: torch.Tensor,
    valid: torch.Tensor,
    *,
    sample_count: int,
) -> torch.Tensor:
    """Fit continuous reference-sphere OPL to n<=2 OSA/ANSI modes.

    Args:
        reference_opl_mm: ``[..., N]`` continuous OPD/relative OPL in mm.
        valid: Boolean tensor with the same shape.
        sample_count: Square pupil-grid size used by ``pupil_disk_grid``.

    Returns:
        Tensor ``[..., 6]`` in mm, ordered as :data:`LOW_ORDER_TERMS`.

    The reduced QR solve is differentiable with respect to
    ``reference_opl_mm``.  Invalid, underdetermined, or rank-deficient pupils
    fail explicitly; no regularization or alternate metric is substituted.
    """
    if reference_opl_mm.ndim < 1:
        raise ValueError("reference_opl_mm must have shape [..., pupil_rays]")
    if not reference_opl_mm.is_floating_point():
        raise TypeError("reference_opl_mm must be a floating tensor")
    valid_mask = valid.to(device=reference_opl_mm.device, dtype=torch.bool)
    if valid_mask.shape != reference_opl_mm.shape:
        raise ValueError("valid must match reference_opl_mm shape")

    x, y = normalized_pupil_disk_coordinates(
        sample_count,
        device=reference_opl_mm.device,
        dtype=reference_opl_mm.dtype,
    )
    ray_count = int(x.numel())
    if int(reference_opl_mm.shape[-1]) != ray_count:
        raise ValueError(
            f"OPL length {int(reference_opl_mm.shape[-1])} does not match "
            f"unit-disk ray count {ray_count}"
        )
    basis = low_order_osa_ansi_basis(x, y)
    flat_opl = reference_opl_mm.reshape(-1, ray_count)
    flat_valid = valid_mask.reshape(-1, ray_count)
    coefficients: list[torch.Tensor] = []
    mode_count = len(LOW_ORDER_TERMS)
    for case_index in range(int(flat_opl.shape[0])):
        case_valid = flat_valid[case_index]
        valid_count = int(case_valid.sum().detach().cpu())
        if valid_count < mode_count:
            raise ValueError(
                f"case {case_index} has {valid_count} valid pupil samples for "
                f"{mode_count} Zernike modes"
            )
        values = flat_opl[case_index, case_valid]
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"case {case_index} contains non-finite valid-ray OPL")
        design = basis[case_valid]
        q, upper = torch.linalg.qr(design, mode="reduced")
        diagonal = torch.abs(torch.diagonal(upper))
        scale = torch.max(torch.abs(design)).clamp_min(1.0)
        tolerance = (
            torch.finfo(design.dtype).eps
            * float(max(valid_count, mode_count))
            * scale
        )
        if bool((~torch.isfinite(diagonal)).any() or (diagonal <= tolerance).any()):
            raise ValueError(f"case {case_index} Zernike design matrix is rank deficient")
        rhs = q.transpose(-2, -1) @ values
        coefficient = torch.linalg.solve_triangular(
            upper, rhs.unsqueeze(-1), upper=True
        ).squeeze(-1)
        coefficients.append(coefficient)
    stacked = torch.stack(coefficients, dim=0)
    return stacked.reshape(*reference_opl_mm.shape[:-1], mode_count)


def z4_defocus_loss(
    reference_opl_mm: torch.Tensor,
    valid: torch.Tensor,
    *,
    sample_count: int,
) -> torch.Tensor:
    """Return squared Z4 defocus OPD coefficient in ``mm^2``."""
    coefficients_mm = fit_low_order_opd_zernike_torch(
        reference_opl_mm,
        valid,
        sample_count=sample_count,
    )
    return coefficients_mm[..., Z4_DEFOCUS_INDEX].square()
