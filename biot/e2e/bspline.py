from __future__ import annotations

from typing import Sequence

import torch


def open_uniform_knots(
    num_control: int,
    degree: int,
    value_range: tuple[float, float],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Create an open uniform knot vector for a 1D B-spline basis.

    Args:
        num_control: Number of control points / basis functions.
        degree: Polynomial degree.
        value_range: Coordinate range in mm as ``(min, max)``.

    Returns:
        Tensor with shape ``[num_control + degree + 1]``.
    """
    if num_control <= degree:
        raise ValueError("num_control must be greater than degree")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    start, end = value_range
    if not start < end:
        raise ValueError("value_range must satisfy min < max")

    kwargs = {"device": device, "dtype": dtype or torch.float64}
    internal_count = num_control - degree - 1
    start_knots = torch.full((degree + 1,), float(start), **kwargs)
    end_knots = torch.full((degree + 1,), float(end), **kwargs)
    if internal_count <= 0:
        return torch.cat([start_knots, end_knots])
    internal = torch.linspace(float(start), float(end), internal_count + 2, **kwargs)[1:-1]
    return torch.cat([start_knots, internal, end_knots])


def bspline_basis_1d(x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate 1D B-spline basis functions with Cox-de Boor recursion.

    Args:
        x: Coordinates in mm with arbitrary shape ``[...]``.
        knots: Knot vector with shape ``[num_basis + degree + 1]``.
        degree: Polynomial degree.

    Returns:
        Basis tensor with shape ``[..., num_basis]``.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if knots.ndim != 1:
        raise ValueError("knots must be a 1D tensor")
    if knots.shape[0] <= degree + 1:
        raise ValueError("knot vector is too short for degree")

    x_eval = x.to(device=knots.device, dtype=knots.dtype)
    x_expanded = x_eval.unsqueeze(-1)

    left_edges = knots[:-1]
    right_edges = knots[1:]
    basis = ((x_expanded >= left_edges) & (x_expanded < right_edges)).to(knots.dtype)

    last_interval = (x_expanded == knots[-1]).to(knots.dtype)
    if basis.shape[-1] > 0:
        basis = basis.clone()
        basis[..., -1] = torch.maximum(basis[..., -1], last_interval.squeeze(-1))

    for current_degree in range(1, degree + 1):
        term_count = knots.shape[0] - current_degree - 1
        left_denom = knots[current_degree : current_degree + term_count] - knots[:term_count]
        right_denom = knots[current_degree + 1 : current_degree + 1 + term_count] - knots[1 : 1 + term_count]
        left_safe = torch.where(left_denom != 0, left_denom, torch.ones_like(left_denom))
        right_safe = torch.where(right_denom != 0, right_denom, torch.ones_like(right_denom))
        left = (
            (x_expanded - knots[:term_count]) / left_safe * basis[..., :term_count]
        ) * (left_denom != 0).to(knots.dtype)
        right = (
            (knots[current_degree + 1 : current_degree + 1 + term_count] - x_expanded)
            / right_safe
            * basis[..., 1 : term_count + 1]
        ) * (right_denom != 0).to(knots.dtype)
        basis = left + right

    endpoint_mask = x_eval == knots[-1]
    endpoint_basis = torch.zeros_like(basis)
    endpoint_basis[..., -1] = endpoint_mask.to(knots.dtype)
    basis = torch.where(endpoint_mask.unsqueeze(-1), endpoint_basis, basis)

    return basis


def bspline_basis_derivative_1d(x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate the exact first derivative of every 1-D B-spline basis."""
    num_basis = int(knots.shape[0] - degree - 1)
    if degree < 0 or num_basis <= 0:
        raise ValueError("invalid knot vector or degree")
    if degree == 0:
        return torch.zeros((*x.shape, num_basis), device=knots.device, dtype=knots.dtype)
    lower = bspline_basis_1d(x, knots, degree - 1)
    # With an original clamped degree-p knot vector, evaluating its degree
    # p-1 basis creates one extra, zero-width terminal basis.  At the upper
    # endpoint the derivative needs the left-limit lower basis (index -2),
    # not that degenerate terminal basis (index -1).
    endpoint = x == knots[-1]
    if lower.shape[-1] >= 2:
        endpoint_lower = torch.zeros_like(lower)
        endpoint_lower[..., -2] = endpoint.to(knots.dtype)
        lower = torch.where(endpoint.unsqueeze(-1), endpoint_lower, lower)
    left_denom = knots[degree : degree + num_basis] - knots[:num_basis]
    right_denom = knots[degree + 1 : degree + 1 + num_basis] - knots[1 : 1 + num_basis]
    left_safe = torch.where(left_denom != 0, left_denom, torch.ones_like(left_denom))
    right_safe = torch.where(right_denom != 0, right_denom, torch.ones_like(right_denom))
    left = float(degree) * lower[..., :num_basis] / left_safe
    right = float(degree) * lower[..., 1 : num_basis + 1] / right_safe
    left = left * (left_denom != 0).to(knots.dtype)
    right = right * (right_denom != 0).to(knots.dtype)
    return left - right


def bspline_surface_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    control: torch.Tensor,
    x_knots: torch.Tensor,
    y_knots: torch.Tensor,
    *,
    degree_x: int = 3,
    degree_y: int = 3,
) -> torch.Tensor:
    """Evaluate a tensor-product B-spline surface.

    Args:
        x, y: Broadcast-compatible coordinate tensors in mm, shape ``[...]``.
        control: Control grid in mm, shape ``[nx, ny]``.
        x_knots, y_knots: Knot vectors for x and y.

    Returns:
        Sag values in mm with shape ``[...]``.
    """
    if control.ndim != 2:
        raise ValueError("control must have shape [nx, ny]")
    x_b, y_b = torch.broadcast_tensors(x, y)
    control_eval = control.to(device=x_knots.device, dtype=x_knots.dtype)
    if control_eval.shape != (x_knots.shape[0] - degree_x - 1, y_knots.shape[0] - degree_y - 1):
        raise ValueError("control shape does not match knot vectors and degrees")

    bx = bspline_basis_1d(x_b, x_knots, degree_x)
    by = bspline_basis_1d(y_b, y_knots, degree_y)
    return torch.einsum("...i,ij,...j->...", bx, control_eval, by)


def bspline_surface_2d_with_derivatives(
    x: torch.Tensor,
    y: torch.Tensor,
    control: torch.Tensor,
    x_knots: torch.Tensor,
    y_knots: torch.Tensor,
    *,
    degree_x: int = 3,
    degree_y: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return tensor-product sag and exact coordinate derivatives."""
    if control.ndim != 2:
        raise ValueError("control must have shape [nx, ny]")
    x_b, y_b = torch.broadcast_tensors(x, y)
    control_eval = control.to(device=x_knots.device, dtype=x_knots.dtype)
    expected = (x_knots.shape[0] - degree_x - 1, y_knots.shape[0] - degree_y - 1)
    if tuple(control_eval.shape) != expected:
        raise ValueError("control shape does not match knot vectors and degrees")
    bx = bspline_basis_1d(x_b, x_knots, degree_x)
    by = bspline_basis_1d(y_b, y_knots, degree_y)
    dbx = bspline_basis_derivative_1d(x_b, x_knots, degree_x)
    dby = bspline_basis_derivative_1d(y_b, y_knots, degree_y)
    sag = torch.einsum("...i,ij,...j->...", bx, control_eval, by)
    dz_dx = torch.einsum("...i,ij,...j->...", dbx, control_eval, by)
    dz_dy = torch.einsum("...i,ij,...j->...", bx, control_eval, dby)
    return sag, dz_dx, dz_dy


def control_laplacian_smoothness(control: torch.Tensor) -> torch.Tensor:
    """Second-difference smoothness loss for a 2D control grid.

    Args:
        control: Control grid in mm, shape ``[nx, ny]``.

    Returns:
        Scalar non-negative tensor in mm^2 units.
    """
    if control.ndim != 2:
        raise ValueError("control must have shape [nx, ny]")
    loss = control.new_zeros(())
    if control.shape[0] >= 3:
        d2x = control[2:, :] - 2.0 * control[1:-1, :] + control[:-2, :]
        loss = loss + d2x.pow(2).mean()
    if control.shape[1] >= 3:
        d2y = control[:, 2:] - 2.0 * control[:, 1:-1] + control[:, :-2]
        loss = loss + d2y.pow(2).mean()
    return loss


def make_control_grid(
    shape: Sequence[int],
    value: float = 0.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Create a 2D control grid tensor for e2e surface parameters."""
    if len(shape) != 2:
        raise ValueError("shape must contain two dimensions")
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("control grid dimensions must be positive")
    return torch.full((int(shape[0]), int(shape[1])), float(value), device=device, dtype=dtype or torch.float64)
