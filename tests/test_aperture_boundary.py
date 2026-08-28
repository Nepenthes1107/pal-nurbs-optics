"""孔径边界与求交 loose residual 容差的一致性回归。"""

import torch

from optics import Surface
from biot.e2e.system import LocalAsphereSurface


def test_biot_round_aperture_uses_newton_residual_scale() -> None:
    surface = Surface(2.0, 0.0, 0.0, device=torch.device("cpu"))
    near_edge = torch.tensor([[2.0002, 0.0]], dtype=torch.float64)
    outside_tolerance = torch.tensor([[2.0005, 0.0]], dtype=torch.float64)
    assert bool(surface.is_valid(near_edge))
    assert not bool(surface.is_valid(outside_tolerance))


def test_e2e_round_aperture_matches_biot_boundary_contract() -> None:
    surface = LocalAsphereSurface(
        semi_diameter_mm=2.0,
        curvature_inv_mm=0.0,
        conic=0.0,
        coeff=None,
        n_after=1.0,
    )
    points = torch.tensor([[2.0002, 0.0, 0.0], [2.0005, 0.0, 0.0]], dtype=torch.float64)
    mask = surface.aperture_mask(points)
    assert mask.tolist() == [True, False]


def test_square_aperture_uses_linear_solver_tolerance() -> None:
    surface = Surface(2.0, 0.0, 0.0, is_square=True, device=torch.device("cpu"))
    near_edge = torch.tensor([[2.0002, 0.0]], dtype=torch.float64)
    outside_tolerance = torch.tensor([[2.0005, 0.0]], dtype=torch.float64)
    assert bool(surface.is_valid(near_edge))
    assert not bool(surface.is_valid(outside_tolerance))
