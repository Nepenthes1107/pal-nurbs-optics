from __future__ import annotations

import pytest
import torch

from biot.e2e.regional_nurbs import (
    FixedWeightNURBSPerturbation,
    audit_exact_refinement,
    bspline_basis_derivative,
)


def test_fixed_weight_nurbs_partition_boundary_and_convex_hull() -> None:
    surface = FixedWeightNURBSPerturbation(7)
    with torch.no_grad():
        surface.inner_q.copy_(torch.linspace(-0.8, 0.8, 25).reshape(5, 5))
    x = torch.linspace(-40.0, 40.0, 101, dtype=torch.float64)
    y = torch.linspace(-40.0, 40.0, 101, dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    values = surface.delta_raw(xx, yy)
    control = surface.physical_control_mm()
    assert torch.equal(surface.weights, torch.ones_like(surface.weights))
    assert float(values.min()) >= float(control.min()) - 1e-13
    assert float(values.max()) <= float(control.max()) + 1e-13
    assert float(values[0].abs().max()) <= 1e-14
    assert float(values[-1].abs().max()) <= 1e-14
    assert float(values[:, 0].abs().max()) <= 1e-14
    assert float(values[:, -1].abs().max()) <= 1e-14
    bx = bspline_basis_derivative(x, surface.x_knots, 3, 0)
    assert float((bx.sum(dim=-1) - 1.0).abs().max()) <= 1e-12


def test_nurbs_coordinate_and_control_gradients_are_finite() -> None:
    surface = FixedWeightNURBSPerturbation(7)
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        surface.inner_q.copy_(0.5 * (2.0 * torch.rand((5, 5), generator=generator) - 1.0))
    x = torch.tensor([-39.5, -20.0, -3.2, 19.9, 39.5], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([-31.0, -10.0, 2.5, 20.0, 33.0], dtype=torch.float64, requires_grad=True)
    sag, dx, dy, *_ = surface.all_derivatives_raw(x, y)
    adx, ady = torch.autograd.grad(sag.sum(), (x, y), retain_graph=True)
    assert torch.allclose(dx, adx, atol=2e-12, rtol=2e-11)
    assert torch.allclose(dy, ady, atol=2e-12, rtol=2e-11)
    sag.square().sum().backward()
    assert surface.inner_q.grad is not None
    assert torch.isfinite(surface.inner_q.grad).all()


@pytest.mark.parametrize("target", [11, 19])
def test_homogeneous_boehm_refinement_is_exact(target: int) -> None:
    surface = FixedWeightNURBSPerturbation(7)
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        surface.inner_q.copy_(0.3 * (2.0 * torch.rand((5, 5), generator=generator) - 1.0))
    fine = surface.refined(11)
    if target == 19:
        fine = fine.refined(19)
    audit = audit_exact_refinement(surface, fine, samples=129)
    assert audit.max_abs_sag_mm <= 1e-12
    assert audit.max_abs_first_derivative <= 1e-12
    assert audit.max_abs_second_derivative_per_mm <= 1e-11
