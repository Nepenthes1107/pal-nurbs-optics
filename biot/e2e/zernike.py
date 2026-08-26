from __future__ import annotations

import torch


DEFAULT_MODES = ("defocus", "astigmatism_0", "astigmatism_45", "coma_x", "coma_y")


def zernike_basis(
    x_mm: torch.Tensor,
    y_mm: torch.Tensor,
    radius_mm: float,
    modes: tuple[str, ...] = DEFAULT_MODES,
) -> torch.Tensor:
    """Evaluate low-order Zernike-like basis terms on normalized pupil coords.

    Piston and tilt are intentionally not included in ``DEFAULT_MODES`` so the
    first e2e surface parameterization does not silently absorb global offset or
    pointing errors.

    Args:
        x_mm, y_mm: Broadcast-compatible coordinates in mm, shape ``[...]``.
        radius_mm: Normalization radius in mm.
        modes: Mode names.

    Returns:
        Basis tensor with shape ``[..., len(modes)]``. Values outside the
        radius are set to zero.
    """
    if radius_mm <= 0:
        raise ValueError("radius_mm must be positive")
    x_b, y_b = torch.broadcast_tensors(x_mm, y_mm)
    x_n = x_b / float(radius_mm)
    y_n = y_b / float(radius_mm)
    rho2 = x_n.pow(2) + y_n.pow(2)
    mask = (rho2 <= 1.0).to(dtype=x_b.dtype, device=x_b.device)

    values = []
    for mode in modes:
        if mode == "defocus":
            value = 2.0 * rho2 - 1.0
        elif mode == "astigmatism_0":
            value = x_n.pow(2) - y_n.pow(2)
        elif mode == "astigmatism_45":
            value = 2.0 * x_n * y_n
        elif mode == "coma_x":
            value = (3.0 * rho2 - 2.0) * x_n
        elif mode == "coma_y":
            value = (3.0 * rho2 - 2.0) * y_n
        elif mode == "spherical":
            value = 6.0 * rho2.pow(2) - 6.0 * rho2 + 1.0
        else:
            raise ValueError(f"Unsupported Zernike mode: {mode!r}")
        values.append(value * mask)
    if not values:
        return x_b.new_zeros((*x_b.shape, 0))
    return torch.stack(values, dim=-1)


def zernike_perturbation(
    x_mm: torch.Tensor,
    y_mm: torch.Tensor,
    coefficients_mm: torch.Tensor,
    radius_mm: float,
    modes: tuple[str, ...] = DEFAULT_MODES,
) -> torch.Tensor:
    """Evaluate a low-order Zernike sag perturbation in mm."""
    if coefficients_mm.ndim != 1:
        raise ValueError("coefficients_mm must have shape [num_modes]")
    if coefficients_mm.shape[0] != len(modes):
        raise ValueError("coefficient count must match modes")
    basis = zernike_basis(x_mm, y_mm, radius_mm, modes)
    coeffs = coefficients_mm.to(device=basis.device, dtype=basis.dtype)
    return torch.einsum("...k,k->...", basis, coeffs)
