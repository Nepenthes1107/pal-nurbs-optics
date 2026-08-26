from __future__ import annotations

"""Fixed-weight tensor-product NURBS controls for regional PAL optimization.

Only the control-point heights are trainable.  Coordinates supplied to
``delta_raw`` follow the workbook/GridSag row convention; trace coordinates
mirror physical y exactly as the established bounded 7x7 path does.
"""

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from .bspline import bspline_basis_1d
from .surfaces import SurfaceDomain


DEGREE = 3
DOMAIN_MM = (-40.0, 40.0)
STAGE_SIZES = (7, 11, 19)


def stage_knots(
    control_count: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the sealed cubic knot ladder on ``[-40, 40] mm``."""
    internal_by_size = {
        7: (-20.0, 0.0, 20.0),
        11: (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0),
        19: tuple(float(v) for v in range(-35, 40, 5)),
    }
    size = int(control_count)
    if size not in internal_by_size:
        raise ValueError(f"unsupported coarse-to-fine control count: {size}")
    values = (DOMAIN_MM[0],) * 4 + internal_by_size[size] + (DOMAIN_MM[1],) * 4
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


def knot_refinement_matrix(
    old_knots: torch.Tensor,
    new_knots: torch.Tensor,
    *,
    degree: int = DEGREE,
) -> torch.Tensor:
    """Return the exact Boehm insertion matrix from an old to a nested basis."""
    if old_knots.ndim != 1 or new_knots.ndim != 1:
        raise ValueError("knot vectors must be one-dimensional")
    old_n = int(old_knots.numel() - degree - 1)
    new_n = int(new_knots.numel() - degree - 1)
    if new_n < old_n:
        raise ValueError("refined knot vector cannot have fewer controls")
    matrix = torch.eye(old_n, dtype=old_knots.dtype, device=old_knots.device)
    current = old_knots.clone()
    remaining = list(new_knots.detach().cpu().tolist())
    for value in old_knots.detach().cpu().tolist():
        remaining.remove(value)
    for value in remaining:
        u = torch.as_tensor(value, dtype=current.dtype, device=current.device)
        n = int(current.numel() - degree - 1)
        span = int(torch.searchsorted(current, u, right=True).item()) - 1
        span = min(max(span, degree), n - 1)
        insertion = torch.zeros((n + 1, n), dtype=current.dtype, device=current.device)
        for i in range(n + 1):
            if i <= span - degree:
                insertion[i, i] = 1.0
            elif i >= span + 1:
                insertion[i, i - 1] = 1.0
            else:
                den = current[i + degree] - current[i]
                alpha = (u - current[i]) / den
                insertion[i, i] = alpha
                insertion[i, i - 1] = 1.0 - alpha
        matrix = insertion @ matrix
        current = torch.cat((current[: span + 1], u.reshape(1), current[span + 1 :]))
    if current.shape != new_knots.shape or not bool(torch.allclose(current, new_knots, atol=0.0, rtol=0.0)):
        raise ValueError("new knot vector is not an exact insertion refinement")
    return matrix


def refine_homogeneous_control(
    control_mm: torch.Tensor,
    weights: torch.Tensor,
    old_knots: torch.Tensor,
    new_knots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine a tensor-product rational surface in homogeneous coordinates."""
    transform = knot_refinement_matrix(old_knots, new_knots)
    hw = weights * control_mm
    refined_w = transform @ weights @ transform.T
    refined_hw = transform @ hw @ transform.T
    if bool(torch.any(refined_w <= 0)):
        raise RuntimeError("refinement produced a non-positive rational weight")
    return refined_hw / refined_w, refined_w


@dataclass(frozen=True)
class RefinementAudit:
    max_abs_sag_mm: float
    max_abs_first_derivative: float
    max_abs_second_derivative_per_mm: float


class FixedWeightNURBSPerturbation(torch.nn.Module):
    """Cubic fixed-one-weight NURBS sag with a hard-zero outer control ring."""

    def __init__(
        self,
        control_count: int = 7,
        *,
        max_abs_control_mm: float = 0.12,
        domain: SurfaceDomain | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        inner_q: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        n = int(control_count)
        if n not in STAGE_SIZES:
            raise ValueError(f"control_count must be one of {STAGE_SIZES}")
        if max_abs_control_mm <= 0:
            raise ValueError("max_abs_control_mm must be positive")
        self.control_shape = (n, n)
        self.degree = DEGREE
        self.domain = domain or SurfaceDomain(DOMAIN_MM, DOMAIN_MM)
        if self.domain.x_range_mm != DOMAIN_MM or self.domain.y_range_mm != DOMAIN_MM:
            raise ValueError("the sealed NURBS ladder requires [-40,40] mm in x and y")
        self.max_sag_mm = float(max_abs_control_mm)
        initial = torch.zeros((n - 2, n - 2), dtype=dtype, device=torch.device(device))
        if inner_q is not None:
            if tuple(inner_q.shape) != tuple(initial.shape):
                raise ValueError("inner_q shape does not match the requested stage")
            initial.copy_(inner_q.to(device=initial.device, dtype=initial.dtype))
        if bool(torch.any(initial.abs() > 1.0)):
            raise ValueError("normalized controls must lie in [-1,1]")
        self.inner_q = torch.nn.Parameter(initial)
        self.register_buffer("x_knots", stage_knots(n, dtype=dtype, device=device))
        self.register_buffer("y_knots", stage_knots(n, dtype=dtype, device=device))
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

    def refined(self, control_count: int) -> "FixedWeightNURBSPerturbation":
        new_n = int(control_count)
        if new_n not in STAGE_SIZES or new_n <= self.control_shape[0]:
            raise ValueError("refinement target must be a later sealed stage")
        control, weights = refine_homogeneous_control(
            self.physical_control_mm(), self.weights, self.x_knots, stage_knots(new_n, dtype=self.x_knots.dtype, device=self.x_knots.device)
        )
        if not bool(torch.allclose(weights, torch.ones_like(weights), atol=1e-13, rtol=0.0)):
            raise RuntimeError("fixed-one weights changed during exact refinement")
        boundary = torch.cat((control[0], control[-1], control[1:-1, 0], control[1:-1, -1]))
        if not bool(torch.allclose(boundary, torch.zeros_like(boundary), atol=1e-13, rtol=0.0)):
            raise RuntimeError("exact refinement violated the hard-zero outer ring")
        return FixedWeightNURBSPerturbation(
            new_n, max_abs_control_mm=self.max_sag_mm, dtype=control.dtype,
            device=control.device, inner_q=control[1:-1, 1:-1] / self.max_sag_mm,
        )


def audit_exact_refinement(
    coarse: FixedWeightNURBSPerturbation,
    fine: FixedWeightNURBSPerturbation,
    *,
    samples: int = 257,
) -> RefinementAudit:
    coordinates = torch.linspace(DOMAIN_MM[0], DOMAIN_MM[1], int(samples), dtype=coarse.x_knots.dtype, device=coarse.x_knots.device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    a = coarse.all_derivatives_raw(xx, yy)
    b = fine.all_derivatives_raw(xx, yy)
    return RefinementAudit(
        max_abs_sag_mm=float((a[0] - b[0]).abs().amax().detach().cpu()),
        max_abs_first_derivative=float(torch.stack(((a[1] - b[1]).abs().amax(), (a[2] - b[2]).abs().amax())).amax().detach().cpu()),
        max_abs_second_derivative_per_mm=float(torch.stack(tuple((a[i] - b[i]).abs().amax() for i in (3, 4, 5))).amax().detach().cpu()),
    )
