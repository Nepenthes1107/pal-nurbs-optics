from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .bspline import (
    bspline_surface_2d,
    control_laplacian_smoothness,
    make_control_grid,
    open_uniform_knots,
)
from .zernike import DEFAULT_MODES, zernike_perturbation


@dataclass(frozen=True)
class SurfaceDomain:
    """Rectangular surface domain in millimeters."""

    x_range_mm: tuple[float, float] = (-20.0, 20.0)
    y_range_mm: tuple[float, float] = (-20.0, 20.0)


class DifferentiablePALSurface(torch.nn.Module):
    """Low-dimensional differentiable PAL back-surface perturbation model.

    Sag definition:
        ``z(x, y; theta) = z0(x, y) + B(x, y; theta_bspline)
        + Z(x, y; theta_zernike)``

    Shapes:
        x_mm/y_mm: broadcast-compatible coordinate tensors ``[...]`` in mm.
        sag output: ``[...]`` in mm.
        normal output: ``[..., 3]`` unit vectors.
        theta_bspline: ``[nx, ny]`` control-point perturbations in mm.
        theta_zernike: ``[num_modes]`` coefficients in mm.
    """

    def __init__(
        self,
        base_control_mm: torch.Tensor,
        *,
        domain: SurfaceDomain | None = None,
        degree: int = 3,
        perturbation_control_shape: Sequence[int] | None = None,
        zernike_modes: tuple[str, ...] = DEFAULT_MODES,
        zernike_radius_mm: float | None = None,
    ) -> None:
        super().__init__()
        if base_control_mm.ndim != 2:
            raise ValueError("base_control_mm must have shape [nx, ny]")
        if degree < 1:
            raise ValueError("degree must be at least 1")
        self.domain = domain or SurfaceDomain()
        self.degree = int(degree)
        self.zernike_modes = tuple(zernike_modes)
        x_radius = max(abs(self.domain.x_range_mm[0]), abs(self.domain.x_range_mm[1]))
        y_radius = max(abs(self.domain.y_range_mm[0]), abs(self.domain.y_range_mm[1]))
        self.zernike_radius_mm = float(zernike_radius_mm if zernike_radius_mm is not None else min(x_radius, y_radius))

        base = base_control_mm.clone()
        self.register_buffer("base_control_mm", base)
        self.register_buffer(
            "base_x_knots",
            open_uniform_knots(base.shape[0], self.degree, self.domain.x_range_mm, device=base.device, dtype=base.dtype),
        )
        self.register_buffer(
            "base_y_knots",
            open_uniform_knots(base.shape[1], self.degree, self.domain.y_range_mm, device=base.device, dtype=base.dtype),
        )

        perturb_shape = tuple(int(v) for v in (perturbation_control_shape or base.shape))
        if len(perturb_shape) != 2:
            raise ValueError("perturbation_control_shape must have two dimensions")
        if perturb_shape[0] <= self.degree or perturb_shape[1] <= self.degree:
            raise ValueError("perturbation control dimensions must be greater than degree")

        theta = make_control_grid(perturb_shape, device=base.device, dtype=base.dtype)
        self.theta_bspline = torch.nn.Parameter(theta)
        self.register_buffer(
            "perturb_x_knots",
            open_uniform_knots(perturb_shape[0], self.degree, self.domain.x_range_mm, device=base.device, dtype=base.dtype),
        )
        self.register_buffer(
            "perturb_y_knots",
            open_uniform_knots(perturb_shape[1], self.degree, self.domain.y_range_mm, device=base.device, dtype=base.dtype),
        )
        self.theta_zernike = torch.nn.Parameter(base.new_zeros((len(self.zernike_modes),)))

    @classmethod
    def from_quadratic_base(
        cls,
        *,
        control_shape: Sequence[int] = (5, 5),
        domain: SurfaceDomain | None = None,
        curvature_x_inv_mm: float = 0.0,
        curvature_y_inv_mm: float = 0.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> "DifferentiablePALSurface":
        """Create a small analytic-style base sag control grid for tests.

        The base control values are in mm and sampled from
        ``0.5 * (cx * x^2 + cy * y^2)``. This is a compact Phase 1 stand-in for
        a future BIOT/GridSag numeric import path.
        """
        surface_domain = domain or SurfaceDomain()
        if len(control_shape) != 2:
            raise ValueError("control_shape must contain two dimensions")
        xs = torch.linspace(
            surface_domain.x_range_mm[0],
            surface_domain.x_range_mm[1],
            int(control_shape[0]),
            dtype=dtype,
            device=device,
        )
        ys = torch.linspace(
            surface_domain.y_range_mm[0],
            surface_domain.y_range_mm[1],
            int(control_shape[1]),
            dtype=dtype,
            device=device,
        )
        xx, yy = torch.meshgrid(xs, ys, indexing="ij")
        base = 0.5 * (float(curvature_x_inv_mm) * xx.pow(2) + float(curvature_y_inv_mm) * yy.pow(2))
        return cls(base, domain=surface_domain)

    def base_sag(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> torch.Tensor:
        """Evaluate base sag ``z0`` in mm with output shape ``[...]``."""
        x_eval, y_eval = self._coordinates(x_mm, y_mm)
        return bspline_surface_2d(
            x_eval,
            y_eval,
            self.base_control_mm,
            self.base_x_knots,
            self.base_y_knots,
            degree_x=self.degree,
            degree_y=self.degree,
        )

    def sag(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> torch.Tensor:
        """Evaluate total sag in mm with output shape ``[...]``."""
        x_eval, y_eval = self._coordinates(x_mm, y_mm)
        base = self.base_sag(x_eval, y_eval)
        bspline_delta = bspline_surface_2d(
            x_eval,
            y_eval,
            self.theta_bspline,
            self.perturb_x_knots,
            self.perturb_y_knots,
            degree_x=self.degree,
            degree_y=self.degree,
        )
        zernike_delta = zernike_perturbation(
            x_eval,
            y_eval,
            self.theta_zernike,
            self.zernike_radius_mm,
            self.zernike_modes,
        )
        return base + bspline_delta + zernike_delta

    def normal(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> torch.Tensor:
        """Return normalized surface normals with shape ``[..., 3]``."""
        x_eval, y_eval = self._coordinates(x_mm, y_mm)
        if not x_eval.requires_grad:
            x_eval = x_eval.clone().requires_grad_(True)
        if not y_eval.requires_grad:
            y_eval = y_eval.clone().requires_grad_(True)

        sag = self.sag(x_eval, y_eval)
        dz_dx, dz_dy = torch.autograd.grad(
            sag.sum(),
            (x_eval, y_eval),
            create_graph=True,
            retain_graph=True,
        )
        normal = torch.stack((-dz_dx, -dz_dy, torch.ones_like(sag)), dim=-1)
        norm = torch.sqrt(normal.pow(2).sum(dim=-1, keepdim=True).clamp_min(torch.finfo(normal.dtype).eps))
        return normal / norm

    def smoothness_loss(self) -> torch.Tensor:
        """Return non-negative control smoothness regularization."""
        loss = control_laplacian_smoothness(self.theta_bspline)
        if self.theta_zernike.numel() > 0:
            loss = loss + self.theta_zernike.pow(2).mean()
        return loss

    def _coordinates(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = self.base_control_mm.dtype
        device = self.base_control_mm.device
        x_eval = x_mm.to(device=device, dtype=dtype)
        y_eval = y_mm.to(device=device, dtype=dtype)
        return torch.broadcast_tensors(x_eval, y_eval)
