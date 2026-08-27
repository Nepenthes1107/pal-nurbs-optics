"""Fixed-weight tensor-product B-spline controls for PAL optimization.

The multidistance PAL method uses one sealed cubic 7x7 control lattice.
Only the control-point heights are trainable. Coordinates supplied to
``delta_raw`` follow the workbook/GridSag row convention; trace coordinates
mirror physical y exactly at the optical-system boundary.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .bspline import bspline_basis_1d
from .surfaces import SurfaceDomain


DEGREE = 3
DOMAIN_MM = (-40.0, 40.0)
CONTROL_COUNT = 7
_INTERNAL_KNOTS = (-20.0, 0.0, 20.0)


def fixed_knots(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the fixed cubic 7-control knot vector on ``[-40, 40] mm``."""
    values = (
        (DOMAIN_MM[0],) * (DEGREE + 1)
        + _INTERNAL_KNOTS
        + (DOMAIN_MM[1],) * (DEGREE + 1)
    )
    return torch.tensor(values, dtype=dtype, device=torch.device(device))


def bspline_basis_derivative(
    x: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
    order: int,
) -> torch.Tensor:
    """Evaluate an exact B-spline derivative of order zero, one, or two."""
    p = int(degree)
    k = int(order)
    if k < 0 or k > 2:
        raise ValueError("only derivative orders 0, 1, and 2 are supported")
    n = int(knots.numel() - p - 1)
    if n <= 0 or p < 0:
        raise ValueError("invalid knot vector or degree")
    if k == 0:
        return bspline_basis_1d(x, knots, p)
    if p == 0:
        return torch.zeros((*x.shape, n), dtype=knots.dtype, device=knots.device)
    lower = bspline_basis_derivative(x, knots, p - 1, k - 1)
    left_den = knots[p : p + n] - knots[:n]
    right_den = knots[p + 1 : p + 1 + n] - knots[1 : 1 + n]
    left_safe = torch.where(left_den != 0, left_den, torch.ones_like(left_den))
    right_safe = torch.where(right_den != 0, right_den, torch.ones_like(right_den))
    left = float(p) * lower[..., :n] / left_safe
    right = float(p) * lower[..., 1 : n + 1] / right_safe
    left = left * (left_den != 0).to(knots.dtype)
    right = right * (right_den != 0).to(knots.dtype)
    return left - right


def _contract(bx: torch.Tensor, values: torch.Tensor, by: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...i,ij,...j->...", bx, values, by)


def rational_surface_with_derivatives(
    x_mm: torch.Tensor,
    y_mm: torch.Tensor,
    control_mm: torch.Tensor,
    weights: torch.Tensor,
    x_knots: torch.Tensor,
    y_knots: torch.Tensor,
    *,
    degree: int = DEGREE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate a standard rational NURBS surface through second order.

    Returns ``sag, dx, dy, dxx, dxy, dyy`` with length units ``mm, 1, 1,
    mm^-1, mm^-1, mm^-1`` respectively.
    """
    if control_mm.ndim != 2 or tuple(weights.shape) != tuple(control_mm.shape):
        raise ValueError("control_mm and weights must be matching 2-D tensors")
    expected = (x_knots.numel() - degree - 1, y_knots.numel() - degree - 1)
    if tuple(control_mm.shape) != expected:
        raise ValueError("control shape does not match knots")
    if not bool(torch.all(weights > 0)):
        raise ValueError("all NURBS weights must be positive")
    x, y = torch.broadcast_tensors(
        x_mm.to(device=control_mm.device, dtype=control_mm.dtype),
        y_mm.to(device=control_mm.device, dtype=control_mm.dtype),
    )
    bx = [bspline_basis_derivative(x, x_knots, degree, k) for k in range(3)]
    by = [bspline_basis_derivative(y, y_knots, degree, k) for k in range(3)]
    w = weights.to(device=control_mm.device, dtype=control_mm.dtype)
    a = w * control_mm
    A = _contract(bx[0], a, by[0])
    Ax = _contract(bx[1], a, by[0])
    Ay = _contract(bx[0], a, by[1])
    Axx = _contract(bx[2], a, by[0])
    Axy = _contract(bx[1], a, by[1])
    Ayy = _contract(bx[0], a, by[2])
    W = _contract(bx[0], w, by[0])
    Wx = _contract(bx[1], w, by[0])
    Wy = _contract(bx[0], w, by[1])
    Wxx = _contract(bx[2], w, by[0])
    Wxy = _contract(bx[1], w, by[1])
    Wyy = _contract(bx[0], w, by[2])
    eps = torch.finfo(W.dtype).eps
    if bool(torch.any(W <= eps)):
        raise ValueError("NURBS denominator is not strictly positive")
    s = A / W
    sx = Ax / W - A * Wx / W.square()
    sy = Ay / W - A * Wy / W.square()
    sxx = Axx / W - A * Wxx / W.square() - 2.0 * Ax * Wx / W.square() + 2.0 * A * Wx.square() / W.pow(3)
    sxy = Axy / W - Ax * Wy / W.square() - Ay * Wx / W.square() - A * Wxy / W.square() + 2.0 * A * Wx * Wy / W.pow(3)
    syy = Ayy / W - A * Wyy / W.square() - 2.0 * Ay * Wy / W.square() + 2.0 * A * Wy.square() / W.pow(3)
    return s, sx, sy, sxx, sxy, syy


class FixedWeightNURBSPerturbation(torch.nn.Module):
    """Cubic fixed-one-weight NURBS sag with a hard-zero outer control ring."""

    def __init__(
        self,
        *,
        max_abs_control_mm: float = 0.12,
        domain: SurfaceDomain | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        inner_q: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if max_abs_control_mm <= 0:
            raise ValueError("max_abs_control_mm must be positive")
        n = CONTROL_COUNT
        self.control_shape = (n, n)
        self.degree = DEGREE
        self.domain = domain or SurfaceDomain(DOMAIN_MM, DOMAIN_MM)
        if self.domain.x_range_mm != DOMAIN_MM or self.domain.y_range_mm != DOMAIN_MM:
            raise ValueError("the fixed 7x7 B-spline lattice requires [-40,40] mm in x and y")
        self.max_sag_mm = float(max_abs_control_mm)
        initial = torch.zeros((n - 2, n - 2), dtype=dtype, device=torch.device(device))
        if inner_q is not None:
            if tuple(inner_q.shape) != tuple(initial.shape):
                raise ValueError("inner_q shape does not match the fixed 7x7 lattice")
            initial.copy_(inner_q.to(device=initial.device, dtype=initial.dtype))
        if bool(torch.any(initial.abs() > 1.0)):
            raise ValueError("normalized controls must lie in [-1,1]")
        self.inner_q = torch.nn.Parameter(initial)
        self.register_buffer("x_knots", fixed_knots(dtype=dtype, device=device))
        self.register_buffer("y_knots", fixed_knots(dtype=dtype, device=device))
        self.register_buffer("weights", torch.ones((n, n), dtype=dtype, device=torch.device(device)))

    @property
    def trainable_dof(self) -> int:
        return int(self.inner_q.numel())

    def physical_control_mm(self) -> torch.Tensor:
        return F.pad(self.max_sag_mm * self.inner_q, (1, 1, 1, 1), value=0.0)

    def all_derivatives_raw(self, x_mm: torch.Tensor, y_mm: torch.Tensor):
        return rational_surface_with_derivatives(
            x_mm, y_mm, self.physical_control_mm(), self.weights,
            self.x_knots, self.y_knots, degree=self.degree,
        )

    def delta_raw(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> torch.Tensor:
        return self.all_derivatives_raw(x_mm, y_mm)[0]

    def delta_trace(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> torch.Tensor:
        return self.delta_raw(x_mm, -y_mm)

    def delta_trace_and_derivatives(self, x_mm: torch.Tensor, y_mm: torch.Tensor):
        sag, dx, dy_raw, *_ = self.all_derivatives_raw(x_mm, -y_mm)
        return sag, dx, -dy_raw
