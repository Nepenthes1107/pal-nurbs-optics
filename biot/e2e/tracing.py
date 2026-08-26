from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .rays import RayBundle, normalize_vector
from .surfaces import DifferentiablePALSurface


def _as_scalar_tensor(value: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _safe_denominator(value: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    eps_tensor = torch.as_tensor(eps, device=value.device, dtype=value.dtype)
    sign = torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))
    return torch.where(value.abs() > eps_tensor, value, sign * eps_tensor)


def intersect_z_plane(
    ray: RayBundle,
    z_mm: float | torch.Tensor,
    *,
    min_t_mm: float = 1.0e-9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intersect rays with a constant-z image or reference plane.

    Returns:
        ``(points_mm, valid, distance_mm)`` with shapes ``[..., 3]``, ``[...]``,
        and ``[...]``. Distances are geometric path lengths in mm because ray
        directions are normalized.
    """
    z = _as_scalar_tensor(z_mm, ray.origins_mm)
    dz = ray.directions[..., 2]
    denom = _safe_denominator(dz)
    t = (z - ray.origins_mm[..., 2]) / denom
    points = ray.origins_mm + t[..., None] * ray.directions
    valid = (dz.abs() > 1.0e-12) & torch.isfinite(t) & (t >= float(min_t_mm))
    return points, valid, t.clamp_min(0.0)


def snell_refract(
    incident: torch.Tensor,
    normal: torch.Tensor,
    n_before: float | torch.Tensor,
    n_after: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply vector Snell refraction at a smooth interface.

    The surface normal must point toward the transmitted side for forward
    tracing. ``n_before`` and ``n_after`` are refractive indices of the incident
    and transmitted media. Returns ``(direction, valid)``; invalid marks total
    internal reflection or non-finite results.
    """
    wi = normalize_vector(incident)
    n = normalize_vector(normal)
    n1 = _as_scalar_tensor(n_before, wi)
    n2 = _as_scalar_tensor(n_after, wi)
    eta = n1 / n2
    eta_b = eta if eta.ndim > 0 else eta.reshape(())
    cos_i = (wi * n).sum(dim=-1)
    sin_t2 = eta_b.pow(2) * (1.0 - cos_i.pow(2))
    cos_t2 = 1.0 - sin_t2
    valid = cos_t2 >= 0.0
    cos_t = torch.sqrt(cos_t2.clamp_min(torch.finfo(wi.dtype).eps))
    wt = cos_t[..., None] * n + eta_b * (wi - cos_i[..., None] * n)
    wt = normalize_vector(wt)
    valid = valid & torch.all(torch.isfinite(wt), dim=-1)
    return wt, valid


class SagSurface(torch.nn.Module):
    """Base class for explicit sag surfaces ``z = vertex_z + sag(x, y)``."""

    def __init__(
        self,
        *,
        vertex_z_mm: float,
        n_after: float,
        aperture_radius_mm: float | None = None,
    ) -> None:
        super().__init__()
        self.vertex_z_mm = float(vertex_z_mm)
        self.n_after = float(n_after)
        self.aperture_radius_mm = None if aperture_radius_mm is None else float(aperture_radius_mm)

    def relative_sag(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def surface_z(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> torch.Tensor:
        return self.relative_sag(x_mm, y_mm) + float(self.vertex_z_mm)

    def aperture_mask(self, points_mm: torch.Tensor) -> torch.Tensor:
        if self.aperture_radius_mm is None:
            return torch.ones(points_mm.shape[:-1], device=points_mm.device, dtype=torch.bool)
        r2 = points_mm[..., 0].pow(2) + points_mm[..., 1].pow(2)
        return r2 <= float(self.aperture_radius_mm) ** 2 + 1.0e-12

    def domain_mask(self, points_mm: torch.Tensor) -> torch.Tensor:
        return torch.ones(points_mm.shape[:-1], device=points_mm.device, dtype=torch.bool)

    def sag_grad(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_eval = x_mm if x_mm.requires_grad else x_mm.clone().requires_grad_(True)
        y_eval = y_mm if y_mm.requires_grad else y_mm.clone().requires_grad_(True)
        sag = self.relative_sag(x_eval, y_eval)
        dz_dx, dz_dy = torch.autograd.grad(
            sag.sum(),
            (x_eval, y_eval),
            create_graph=True,
            retain_graph=True,
        )
        return sag, dz_dx, dz_dy

    def normal_at(self, points_mm: torch.Tensor) -> torch.Tensor:
        _, dz_dx, dz_dy = self.sag_grad(points_mm[..., 0], points_mm[..., 1])
        normal = torch.stack((-dz_dx, -dz_dy, torch.ones_like(dz_dx)), dim=-1)
        return normalize_vector(normal)

    def intersect(
        self,
        ray: RayBundle,
        *,
        newton_iterations: int = 12,
        tolerance_mm: float = 1.0e-8,
        min_t_mm: float = 1.0e-9,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Intersect rays with this sag surface using explicit Newton updates."""
        dz = ray.directions[..., 2]
        t = (float(self.vertex_z_mm) - ray.origins_mm[..., 2]) / _safe_denominator(dz)
        for _ in range(int(newton_iterations)):
            points = ray.origins_mm + t[..., None] * ray.directions
            sag, dz_dx, dz_dy = self.sag_grad(points[..., 0], points[..., 1])
            residual = points[..., 2] - (float(self.vertex_z_mm) + sag)
            derivative = ray.directions[..., 2] - dz_dx * ray.directions[..., 0] - dz_dy * ray.directions[..., 1]
            t = t - residual / _safe_denominator(derivative)

        points = ray.origins_mm + t[..., None] * ray.directions
        residual = points[..., 2] - self.surface_z(points[..., 0], points[..., 1])
        valid = (
            torch.isfinite(t)
            & torch.isfinite(residual)
            & (t >= float(min_t_mm))
            & (residual.abs() <= float(tolerance_mm))
            & self.aperture_mask(points)
            & self.domain_mask(points)
        )
        return points, valid, t.clamp_min(0.0)


class PlaneSurface(SagSurface):
    """Plane optical interface at constant ``z = vertex_z_mm``."""

    def relative_sag(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> torch.Tensor:
        return x_mm * 0.0 + y_mm * 0.0


class SphericalSurface(SagSurface):
    """Spherical optical interface represented by vertex sag.

    ``radius_mm`` is signed. Positive radius gives positive sag for positive
    radial coordinates under the same convention as BIOT's spherical asphere.
    """

    def __init__(
        self,
        *,
        vertex_z_mm: float,
        radius_mm: float,
        n_after: float,
        aperture_radius_mm: float | None = None,
    ) -> None:
        if float(radius_mm) == 0.0:
            raise ValueError("radius_mm must be non-zero")
        super().__init__(vertex_z_mm=vertex_z_mm, n_after=n_after, aperture_radius_mm=aperture_radius_mm)
        self.radius_mm = float(radius_mm)

    def relative_sag(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> torch.Tensor:
        c = 1.0 / float(self.radius_mm)
        r2 = x_mm.pow(2) + y_mm.pow(2)
        radicand = (1.0 - (c * c) * r2).clamp_min(torch.finfo(x_mm.dtype).eps)
        return c * r2 / (1.0 + torch.sqrt(radicand))

    def domain_mask(self, points_mm: torch.Tensor) -> torch.Tensor:
        r2 = points_mm[..., 0].pow(2) + points_mm[..., 1].pow(2)
        return r2 <= float(self.radius_mm) ** 2 + 1.0e-12


class PALSurface(SagSurface):
    """Adapter exposing Phase 1 DifferentiablePALSurface as an optical interface."""

    def __init__(
        self,
        pal_surface: DifferentiablePALSurface,
        *,
        vertex_z_mm: float,
        n_after: float,
        aperture_radius_mm: float | None = None,
    ) -> None:
        super().__init__(vertex_z_mm=vertex_z_mm, n_after=n_after, aperture_radius_mm=aperture_radius_mm)
        self.pal_surface = pal_surface

    def relative_sag(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> torch.Tensor:
        return self.pal_surface.sag(x_mm, y_mm)

    def normal_at(self, points_mm: torch.Tensor) -> torch.Tensor:
        return self.pal_surface.normal(points_mm[..., 0], points_mm[..., 1])

    def domain_mask(self, points_mm: torch.Tensor) -> torch.Tensor:
        domain = self.pal_surface.domain
        x = points_mm[..., 0]
        y = points_mm[..., 1]
        return (
            (x >= domain.x_range_mm[0])
            & (x <= domain.x_range_mm[1])
            & (y >= domain.y_range_mm[0])
            & (y <= domain.y_range_mm[1])
        )


@dataclass(frozen=True)
class TraceResult:
    """Result of e2e geometric tracing.

    Shapes:
        spots_mm: ``[..., 2]`` image-plane landing coordinates in mm.
        valid: ``[...]`` mask; false includes failed intersection, aperture
            miss, total internal reflection, or image-plane miss.
        geometric_path_length_mm / optical_path_length_mm: ``[...]`` path
            lengths accumulated from valid segments.
    """

    spots_mm: torch.Tensor
    valid: torch.Tensor
    geometric_path_length_mm: torch.Tensor
    optical_path_length_mm: torch.Tensor
    final_ray: RayBundle


def trace_refractive_surfaces(
    ray: RayBundle,
    surfaces: Sequence[SagSurface],
    *,
    n_initial: float = 1.0,
) -> tuple[RayBundle, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Trace a ray bundle through sequential refractive sag surfaces."""
    current_ray = ray.normalized()
    active = torch.ones(current_ray.origins_mm.shape[:-1], device=ray.device, dtype=torch.bool)
    geometric_path = torch.zeros_like(current_ray.weights)
    optical_path = torch.zeros_like(current_ray.weights)
    n_current = float(n_initial)

    for surface in surfaces:
        points, hit_valid, distance = surface.intersect(current_ray)
        normals = surface.normal_at(points)
        refracted, refract_valid = snell_refract(current_ray.directions, normals, n_current, surface.n_after)
        step_valid = hit_valid & refract_valid
        active = active & step_valid
        geometric_path = geometric_path + torch.where(active, distance, torch.zeros_like(distance))
        optical_path = optical_path + torch.where(active, distance * n_current, torch.zeros_like(distance))
        weights = current_ray.weights * active.to(current_ray.dtype)
        current_ray = current_ray.with_state(points, refracted, weights=weights)
        n_current = surface.n_after

    return current_ray, active, geometric_path, optical_path, n_current


def trace_to_image_plane(
    ray: RayBundle,
    surfaces: Sequence[SagSurface],
    *,
    image_z_mm: float,
    n_initial: float = 1.0,
) -> TraceResult:
    """Trace through refractive surfaces and intersect the final image plane."""
    final_ray, active, geometric_path, optical_path, n_current = trace_refractive_surfaces(
        ray,
        surfaces,
        n_initial=n_initial,
    )
    image_points, image_valid, image_distance = intersect_z_plane(final_ray, image_z_mm)
    valid = active & image_valid
    geometric_path = geometric_path + torch.where(valid, image_distance, torch.zeros_like(image_distance))
    optical_path = optical_path + torch.where(valid, image_distance * n_current, torch.zeros_like(image_distance))
    weights = final_ray.weights * valid.to(final_ray.dtype)
    image_ray = final_ray.with_state(image_points, final_ray.directions, weights=weights)
    return TraceResult(
        spots_mm=image_points[..., :2],
        valid=valid,
        geometric_path_length_mm=geometric_path,
        optical_path_length_mm=optical_path,
        final_ray=image_ray,
    )
