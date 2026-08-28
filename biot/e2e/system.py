from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.checkpoint import checkpoint

from biot.services.single_field_service import modify_excel_config
from optics import Gradient_3, GridSag, Lensdata

from .bspline import bspline_surface_2d, bspline_surface_2d_with_derivatives, open_uniform_knots
from .rays import RayBundle, normalize_vector, pupil_disk_grid
from .validation import fit_bspline_control_grid, sample_biot_surface
from .surfaces import SurfaceDomain


GRIN_ACTIVATION_CHECKPOINT = True
GRIN_CASE_BATCH_RAY_CHUNK_SIZE = 2048


def _safe_denominator(value: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    eps_tensor = torch.as_tensor(eps, device=value.device, dtype=value.dtype)
    sign = torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))
    return torch.where(value.abs() > eps_tensor, value, sign * eps_tensor)


def _as_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def rotation_matrix_xyz(
    tilt_x_deg: float | torch.Tensor,
    tilt_y_deg: float | torch.Tensor,
    tilt_z_deg: float | torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """BIOT/BIOT_vis CoordinateBreak passive rotation: Rz @ Ry @ Rx."""
    tx = torch.deg2rad(torch.as_tensor(tilt_x_deg, device=device, dtype=dtype))
    ty = torch.deg2rad(torch.as_tensor(tilt_y_deg, device=device, dtype=dtype))
    tz = torch.deg2rad(torch.as_tensor(tilt_z_deg, device=device, dtype=dtype))
    tx, ty, tz = torch.broadcast_tensors(tx, ty, tz)
    one = torch.ones_like(tx)
    zero = torch.zeros_like(tx)
    cx, sx = torch.cos(tx), torch.sin(tx)
    cy, sy = torch.cos(ty), torch.sin(ty)
    cz, sz = torch.cos(tz), torch.sin(tz)
    rx = torch.stack(
        (
            torch.stack((one, zero, zero), dim=-1),
            torch.stack((zero, cx, -sx), dim=-1),
            torch.stack((zero, sx, cx), dim=-1),
        ),
        dim=-2,
    )
    ry = torch.stack(
        (
            torch.stack((cy, zero, sy), dim=-1),
            torch.stack((zero, one, zero), dim=-1),
            torch.stack((-sy, zero, cy), dim=-1),
        ),
        dim=-2,
    )
    rz = torch.stack(
        (
            torch.stack((cz, -sz, zero), dim=-1),
            torch.stack((sz, cz, zero), dim=-1),
            torch.stack((zero, zero, one), dim=-1),
        ),
        dim=-2,
    )
    return rz @ ry @ rx


@dataclass(frozen=True)
class FitSpec:
    control_shape: tuple[int, int] = (21, 21)
    sample_shape: tuple[int, int] = (81, 81)
    degree: int = 3


@dataclass(frozen=True)
class E2ETraceResult:
    spots_mm: torch.Tensor
    valid: torch.Tensor
    weights: torch.Tensor
    final_ray: RayBundle


@dataclass(frozen=True)
class E2EPhaseTraceResult:
    """Fitted-system trace result with optical path and FFT pupil phase.

    Shapes:
        spots_mm: ``[..., N, 2]`` image-surface landing coordinates in mm.
        valid: ``[..., N]`` final valid-ray mask.
        optical_path_mm: ``[..., N]`` accumulated path to the image surface in mm.
        reference_opl_mm: ``[..., N]`` continuous OPL at the selected phase
            reference, including the launch term.  Its complex phasor equals
            ``phase_rad`` on valid rays without phase unwrapping.
        phase_rad: ``[..., N]`` phase in radians, using ``wavelength_nm``.

    The optical path includes the BIOT/BIOT_vis launch term relative to the
    separately aimed centre-pupil reference ray and every refractive segment.
    Piston removal is intentionally left to the complex-pupil builder.
    """

    spots_mm: torch.Tensor
    valid: torch.Tensor
    weights: torch.Tensor
    final_ray: RayBundle
    optical_path_mm: torch.Tensor
    reference_opl_mm: torch.Tensor
    phase_rad: torch.Tensor


IMPLICIT_INTERSECTION_GRADIENT = False
"""Whether ``LocalSurface.intersect`` keeps its Newton search path in the graph.

Default ``False`` reproduces the original code path byte for byte, so every
existing caller -- Phase 6, Phase 7, the stable BIOT tracing paths and every
sealed baseline -- is unaffected.  Only a caller that opts in explicitly, via
:func:`implicit_intersection_gradient`, takes the reduced-memory path.

Why the option exists: the Newton loop calls ``sag_and_derivatives`` once per
iteration plus once after convergence, thirteen B-spline evaluations per surface
crossing.  At the Phase 15 sampling density (821904 float64 pupil rays, a 41x41
degree-3 fit) one evaluation holds about 4.16 GiB of Cox-de Boor basis
intermediates, and a single case's measured backward peak is 3.53 GiB against a
measured 5.0 GiB usable ceiling on an 8 GiB card.  That margin is what the
Phase 15 R21 training stage ran out of.

Why the reduced path is the same quantity, not an approximation: the iteration's
physical output is its fixed point, where ``F(t, theta) = sag(p(t), theta) -
p_z(t) = 0``.  The implicit function theorem gives ``dt/dtheta = -(dF/dtheta) /
(dF/dt)`` at the root, and one in-graph Newton step taken *from the detached
root* is exactly that expression, because the step is ``-F / (dF/dt)`` and only
``F`` depends on theta.  The search history is a numerical path, not a physical
quantity, so it carries no gradient of its own.

Measured on case F001_D500 against the full-graph path: max abs gradient
difference 7.1e-13, max relative 2.1e-11, L2 relative 1.6e-12 -- float64
rounding -- with the peak falling from 3.531 GiB to 0.383 GiB.

Note for anyone tempted by the simpler variant: merely detaching the iterate
history *without* the in-graph correction step leaves the score and the valid
fraction bit-identical while getting the gradient wrong by 185% relative.  A
forward-value check cannot detect that error; only a gradient comparison can.
"""


@contextlib.contextmanager
def implicit_intersection_gradient(enabled: bool = True):
    """Opt into the implicit-function intersection gradient for this block."""
    global IMPLICIT_INTERSECTION_GRADIENT
    previous = IMPLICIT_INTERSECTION_GRADIENT
    IMPLICIT_INTERSECTION_GRADIENT = bool(enabled)
    try:
        yield
    finally:
        IMPLICIT_INTERSECTION_GRADIENT = previous


class LocalSurface(torch.nn.Module):
    """BIOT-like local sag surface traced by previous-surface thickness."""

    def __init__(self, *, semi_diameter_mm: float, n_after: float, is_aperture: bool = False) -> None:
        super().__init__()
        self.semi_diameter_mm = float(semi_diameter_mm)
        self.n_after = float(n_after)
        self.is_aperture = bool(is_aperture)

    def sag_and_derivatives(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def aperture_mask(self, points_mm: torch.Tensor) -> torch.Tensor:
        if self.semi_diameter_mm == float("inf"):
            return torch.ones(points_mm.shape[:-1], device=points_mm.device, dtype=torch.bool)
        r2 = points_mm[..., 0].pow(2) + points_mm[..., 1].pow(2)
        # Match ``optics.Surface.is_valid`` / BIOT_vis exactly.  Its circular
        # aperture SDF is ``semi_dia**2 - r2`` and accepts values down to
        # -1e-6 mm^2 so aimed marginal rays are classified identically.
        return r2 <= self.semi_diameter_mm**2 + 1.0e-6

    def intersect(
        self,
        ray: RayBundle,
        distance_mm: float | torch.Tensor,
        *,
        newton_iterations: int = 12,
        tolerance_mm: float = 1.0e-7,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dz = ray.directions[..., 2]
        distance = torch.as_tensor(distance_mm, device=ray.device, dtype=ray.dtype)
        t0 = (distance - ray.origins_mm[..., 2]) / _safe_denominator(dz)
        local_origin = ray.origins_mm + t0[..., None] * ray.directions
        local_origin = local_origin.clone()
        local_origin[..., 2] = 0.0
        if IMPLICIT_INTERSECTION_GRADIENT:
            return self._intersect_implicit(ray, local_origin, dz, t0, newton_iterations, tolerance_mm)
        t_delta = torch.zeros_like(dz)
        for _ in range(int(newton_iterations)):
            points = local_origin + t_delta[..., None] * ray.directions
            sag, dz_dx, dz_dy = self.sag_and_derivatives(points[..., 0], points[..., 1])
            residual = sag - points[..., 2]
            derivative = ray.directions[..., 2] - dz_dx * ray.directions[..., 0] - dz_dy * ray.directions[..., 1]
            t_delta = t_delta + residual / _safe_denominator(derivative)
        points = local_origin + t_delta[..., None] * ray.directions
        sag, _, _ = self.sag_and_derivatives(points[..., 0], points[..., 1])
        residual = sag - points[..., 2]
        valid = torch.isfinite(t_delta) & torch.isfinite(residual) & (residual.abs() <= float(tolerance_mm))
        valid = valid & self.aperture_mask(points)
        return points, valid, t0 + t_delta

    def _intersect_implicit(
        self,
        ray: RayBundle,
        local_origin: torch.Tensor,
        dz: torch.Tensor,
        t0: torch.Tensor,
        newton_iterations: int,
        tolerance_mm: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same intersection, with the Newton search path outside the graph.

        See :data:`IMPLICIT_INTERSECTION_GRADIENT` for why this is the same
        quantity as the full-graph loop and for the measured agreement.

        The loop locates the root under ``no_grad``; the parameter derivative is
        then reintroduced by one in-graph Newton step from the detached root,
        which equals ``-(dF/dtheta) / (dF/dt)`` because the step is
        ``-F / (dF/dt)`` and only ``F`` depends on the surface parameters.
        ``dF/dt`` is detached deliberately: it is the derivative *at* the root,
        a coefficient of the correction, not a path back into the search.
        """
        with torch.no_grad():
            t_root = torch.zeros_like(dz)
            for _ in range(int(newton_iterations)):
                points = local_origin + t_root[..., None] * ray.directions
                sag, dz_dx, dz_dy = self.sag_and_derivatives(points[..., 0], points[..., 1])
                residual = sag - points[..., 2]
                derivative = ray.directions[..., 2] - dz_dx * ray.directions[..., 0] - dz_dy * ray.directions[..., 1]
                t_root = t_root + residual / _safe_denominator(derivative)
        t_root = t_root.detach()
        points_root = local_origin + t_root[..., None] * ray.directions
        sag, dz_dx, dz_dy = self.sag_and_derivatives(points_root[..., 0], points_root[..., 1])
        residual = sag - points_root[..., 2]
        derivative = ray.directions[..., 2] - dz_dx.detach() * ray.directions[..., 0] - dz_dy.detach() * ray.directions[..., 1]
        t_delta = t_root + residual / _safe_denominator(derivative)
        points = local_origin + t_delta[..., None] * ray.directions
        sag_final, _, _ = self.sag_and_derivatives(points[..., 0], points[..., 1])
        residual_final = sag_final - points[..., 2]
        valid = torch.isfinite(t_delta) & torch.isfinite(residual_final) & (residual_final.abs() <= float(tolerance_mm))
        valid = valid & self.aperture_mask(points)
        return points, valid, t0 + t_delta

    def normal_at(self, points_mm: torch.Tensor) -> torch.Tensor:
        _, dz_dx, dz_dy = self.sag_and_derivatives(points_mm[..., 0], points_mm[..., 1])
        return normalize_vector(torch.stack((-dz_dx, -dz_dy, torch.ones_like(dz_dx)), dim=-1))

    def after_interaction(self, ray: RayBundle) -> RayBundle:
        return ray


class LocalAsphereSurface(LocalSurface):
    """Fixed S/A asphere with analytic sag derivatives."""

    def __init__(
        self,
        *,
        semi_diameter_mm: float,
        curvature_inv_mm: float,
        conic: float,
        coeff: Sequence[float] | None,
        n_after: float,
        is_aperture: bool = False,
    ) -> None:
        super().__init__(semi_diameter_mm=semi_diameter_mm, n_after=n_after, is_aperture=is_aperture)
        self.curvature_inv_mm = float(curvature_inv_mm)
        self.conic = float(conic)
        self.coeff = tuple(float(v) for v in (coeff or ()))

    def sag_and_derivatives(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = torch.as_tensor(self.curvature_inv_mm, device=x_mm.device, dtype=x_mm.dtype)
        q = torch.as_tensor(self.conic + 1.0, device=x_mm.device, dtype=x_mm.dtype)
        r2 = x_mm.pow(2) + y_mm.pow(2)
        if self.curvature_inv_mm == 0.0:
            sag = torch.zeros_like(x_mm)
            dz_dr2 = torch.zeros_like(x_mm)
        else:
            root = torch.sqrt((1.0 - q * r2 * c.pow(2)).clamp_min(torch.finfo(x_mm.dtype).eps))
            sag = c * r2 / (1.0 + root)
            dz_dr2 = c / (2.0 * root)
        for i, coeff in enumerate(self.coeff):
            power = i + 2
            a = torch.as_tensor(coeff, device=x_mm.device, dtype=x_mm.dtype)
            sag = sag + a * r2.pow(power)
            dz_dr2 = dz_dr2 + a * float(power) * r2.pow(power - 1)
        dz_dx = dz_dr2 * 2.0 * x_mm
        dz_dy = dz_dr2 * 2.0 * y_mm
        return sag, dz_dx, dz_dy

    def intersect(self, ray: RayBundle, distance_mm: float | torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.curvature_inv_mm != 0.0 or self.coeff:
            return super().intersect(ray, distance_mm, **kwargs)
        dz = ray.directions[..., 2]
        distance = torch.as_tensor(distance_mm, device=ray.device, dtype=ray.dtype)
        t = (distance - ray.origins_mm[..., 2]) / _safe_denominator(dz)
        points = ray.origins_mm + t[..., None] * ray.directions
        points = points.clone()
        points[..., 2] = 0.0
        valid = torch.isfinite(t) & self.aperture_mask(points)
        return points, valid, t


class _LocalSagAdapter:
    """Expose an E2E local surface through BIOT's ``get_sag`` protocol."""

    def __init__(self, surface: LocalSurface) -> None:
        self.surface = surface

    def get_sag(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> torch.Tensor:
        return self.surface.sag_and_derivatives(x_mm, y_mm)[0]


class LocalGradient3Surface(LocalSurface):
    """E2E wrapper around BIOT/BIOT_vis's authoritative Gradient 3 solver.

    The refractive-index polynomial, ``grad(n^2)``, Sharma t-parameterized RK4,
    OPL quadrature, integration step cap, and termination on the next real sag
    surface are delegated directly to :class:`optics.Gradient_3`.  E2E only
    supplies its local-surface intersection protocol and the differentiable
    next-surface sag adapter, so there is no second GRIN physics implementation.
    """

    GRIN_STEP_DEFAULT_MM = 5.0e-3

    def __init__(self, source: Gradient_3) -> None:
        super().__init__(
            semi_diameter_mm=float(source.semi_dia),
            # GRIN exit index is position-dependent and is handled explicitly
            # by the trace loop.  n_after is metadata only for this subclass.
            n_after=_as_float(source.n0),
            is_aperture=bool(getattr(source, "isaperture", False)),
        )
        self.source = source
        self.thickness_mm = _as_float(source.thickness)
        delta_t = float(getattr(source, "delta_t", 0.0))
        self.integration_step_mm = (
            min(delta_t, self.GRIN_STEP_DEFAULT_MM)
            if delta_t > 0.0
            else self.GRIN_STEP_DEFAULT_MM
        )

    def sag_and_derivatives(
        self, x_mm: torch.Tensor, y_mm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sag, negative_dz_dx, negative_dz_dy, _ = self.source.get_surface(x_mm, y_mm)
        return sag, -negative_dz_dx, -negative_dz_dy

    def index_at(self, points_mm: torch.Tensor) -> torch.Tensor:
        return self.source.get_ior(
            points_mm[..., 0], points_mm[..., 1], points_mm[..., 2]
        )

    def trace_to_next_surface(
        self,
        points_mm: torch.Tensor,
        optical_momentum: torch.Tensor,
        next_surface: LocalSurface,
        *,
        case_axis: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        adapter = _LocalSagAdapter(next_surface)

        def trace_grin(
            points: torch.Tensor, momentum: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return self.source.trace_to_next_surface(
                points,
                momentum,
                self.integration_step_mm,
                self.thickness_mm,
                adapter,
                case_axis=case_axis,
            )

        if (
            GRIN_ACTIVATION_CHECKPOINT
            and torch.is_grad_enabled()
            and (points_mm.requires_grad or optical_momentum.requires_grad)
        ):
            if case_axis == 0:
                if points_mm.ndim != 3 or optical_momentum.shape != points_mm.shape:
                    raise ValueError("case-batch GRIN tensors must have shape [B,N,3]")
                # The authoritative scalar solver derives one fixed RK4 schedule
                # from all pupil rays in each case.  Derive that same schedule
                # before ray chunking, then reuse it for every chunk so memory
                # reduction cannot alter a case's numerical trajectory.
                tz = optical_momentum[..., 2].abs().clamp_min(1.0e-12)
                t_span = (float(self.thickness_mm) / tz.mean(dim=1)).detach()
                base_steps = torch.floor(
                    t_span.abs() / max(float(self.integration_step_mm), 1.0e-9)
                ).to(dtype=torch.int64) + 1
                base_steps = base_steps.clamp_min(1)
                case_step_h = (
                    t_span / base_steps.to(dtype=t_span.dtype)
                ).unsqueeze(-1)
                case_max_steps = torch.floor(
                    base_steps.to(dtype=t_span.dtype) * 1.6
                ).to(dtype=torch.int64) + 4

                def trace_grin_chunk(
                    points: torch.Tensor, momentum: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                    return self.source.trace_to_next_surface(
                        points,
                        momentum,
                        self.integration_step_mm,
                        self.thickness_mm,
                        adapter,
                        case_axis=0,
                        case_step_h=case_step_h,
                        case_max_steps=case_max_steps,
                    )

                chunk_outputs: list[
                    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                ] = []
                ray_count = int(points_mm.shape[1])
                for ray_start in range(0, ray_count, GRIN_CASE_BATCH_RAY_CHUNK_SIZE):
                    ray_end = min(
                        ray_start + GRIN_CASE_BATCH_RAY_CHUNK_SIZE, ray_count
                    )
                    chunk_outputs.append(
                        checkpoint(
                            trace_grin_chunk,
                            points_mm[:, ray_start:ray_end],
                            optical_momentum[:, ray_start:ray_end],
                        )
                    )
                return tuple(
                    torch.cat([output[index] for output in chunk_outputs], dim=1)
                    for index in range(4)
                )
            # Forward physics is unchanged. Backward deterministically recomputes
            # the fixed-step RK4 segment instead of retaining every step tensor.
            return checkpoint(trace_grin, points_mm, optical_momentum)
        return trace_grin(points_mm, optical_momentum)


class LocalBSplineSurface(LocalSurface):
    """Fitted tensor-product B-spline sag surface in mm."""

    def __init__(
        self,
        *,
        control_mm: torch.Tensor,
        domain: SurfaceDomain,
        degree: int,
        semi_diameter_mm: float,
        n_after: float,
        trainable: bool = False,
        perturbation: torch.nn.Module | None = None,
    ) -> None:
        super().__init__(semi_diameter_mm=semi_diameter_mm, n_after=n_after)
        self.domain = domain
        self.degree = int(degree)
        control = control_mm.clone()
        if trainable:
            self.control_mm = torch.nn.Parameter(control)
        else:
            self.register_buffer("control_mm", control)
        self.perturbation = perturbation
        self.register_buffer("x_knots", open_uniform_knots(control.shape[0], degree, domain.x_range_mm, device=control.device, dtype=control.dtype))
        self.register_buffer("y_knots", open_uniform_knots(control.shape[1], degree, domain.y_range_mm, device=control.device, dtype=control.dtype))

    def sag(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> torch.Tensor:
        base = bspline_surface_2d(
            x_mm,
            y_mm,
            self.control_mm,
            self.x_knots,
            self.y_knots,
            degree_x=self.degree,
            degree_y=self.degree,
        )
        if self.perturbation is None:
            return base
        delta_trace = getattr(self.perturbation, "delta_trace", None)
        if delta_trace is None:
            raise TypeError("back-surface perturbation must provide delta_trace(x_mm, y_mm)")
        return base + delta_trace(x_mm, y_mm)

    def sag_and_derivatives(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sag, dz_dx, dz_dy = bspline_surface_2d_with_derivatives(
            x_mm,
            y_mm,
            self.control_mm,
            self.x_knots,
            self.y_knots,
            degree_x=self.degree,
            degree_y=self.degree,
        )
        if self.perturbation is not None:
            derivative_fn = getattr(self.perturbation, "delta_trace_and_derivatives", None)
            if derivative_fn is None:
                raise TypeError("back-surface perturbation must provide delta_trace_and_derivatives")
            delta, delta_dx, delta_dy = derivative_fn(x_mm, y_mm)
            sag = sag + delta
            dz_dx = dz_dx + delta_dx
            dz_dy = dz_dy + delta_dy
        return sag, dz_dx, dz_dy


class LocalGridSagSurface(LocalSurface):
    """Exact Torch form of BIOT's Original ``GridSag`` B-spline plus residual.

    This copies the knots and coefficients of the already-loaded
    ``scipy.interpolate.RectBivariateSpline``; it does not refit or smooth the
    PAL baseline.  The x/y coefficient axes are transposed once from SciPy's
    ``[y, x]`` storage into this module's ``[x, y]`` tensor convention.
    """

    def __init__(
        self,
        source: GridSag,
        *,
        n_after: float,
        dtype: torch.dtype,
        device: torch.device | str,
        perturbation: torch.nn.Module | None = None,
    ) -> None:
        super().__init__(
            semi_diameter_mm=float(source.semi_dia),
            n_after=float(n_after),
            is_aperture=bool(getattr(source, "isaperture", False)),
        )
        y_knots_np, x_knots_np = source.spline.get_knots()
        degree_y, degree_x = (int(value) for value in source.spline.degrees)
        basis_y = len(y_knots_np) - degree_y - 1
        basis_x = len(x_knots_np) - degree_x - 1
        coefficients_yx = torch.as_tensor(
            source.spline.get_coeffs(), device=device, dtype=dtype
        ).reshape(basis_y, basis_x)
        self.register_buffer("control_mm", coefficients_yx.transpose(0, 1).contiguous())
        self.register_buffer(
            "x_knots", torch.as_tensor(x_knots_np, device=device, dtype=dtype).clone()
        )
        self.register_buffer(
            "y_knots", torch.as_tensor(y_knots_np, device=device, dtype=dtype).clone()
        )
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.perturbation = perturbation

    def sag_and_derivatives(
        self, x_mm: torch.Tensor, y_mm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sag, dz_dx, dz_dy = bspline_surface_2d_with_derivatives(
            x_mm,
            y_mm,
            self.control_mm,
            self.x_knots,
            self.y_knots,
            degree_x=self.degree_x,
            degree_y=self.degree_y,
        )
        if self.perturbation is not None:
            derivative_fn = getattr(
                self.perturbation, "delta_trace_and_derivatives", None
            )
            if derivative_fn is None:
                raise TypeError(
                    "back-surface perturbation must provide delta_trace_and_derivatives"
                )
            delta, delta_dx, delta_dy = derivative_fn(x_mm, y_mm)
            sag = sag + delta
            dz_dx = dz_dx + delta_dx
            dz_dy = dz_dy + delta_dy
        return sag, dz_dx, dz_dy


class LocalCoordinateBreakSurface(LocalAsphereSurface):
    """Plane CoordinateBreak that applies BIOT's post-surface rotation."""

    def __init__(self, *, semi_diameter_mm: float, n_after: float, tilt_x_deg: float, tilt_y_deg: float, tilt_z_deg: float) -> None:
        super().__init__(
            semi_diameter_mm=semi_diameter_mm,
            curvature_inv_mm=0.0,
            conic=0.0,
            coeff=None,
            n_after=n_after,
        )
        self.tilt_x_deg = float(tilt_x_deg)
        self.tilt_y_deg = float(tilt_y_deg)
        self.tilt_z_deg = float(tilt_z_deg)

    def after_interaction(self, ray: RayBundle) -> RayBundle:
        rot = rotation_matrix_xyz(
            self.tilt_x_deg,
            self.tilt_y_deg,
            self.tilt_z_deg,
            device=ray.device,
            dtype=ray.dtype,
        )
        origins = torch.matmul(rot, ray.origins_mm.unsqueeze(-1)).squeeze(-1)
        directions = torch.matmul(rot, ray.directions.unsqueeze(-1)).squeeze(-1)
        return ray.with_state(origins, directions, weights=ray.weights)


@dataclass
class FittedE2ESystem:
    lens: Lensdata | None
    surfaces: list[LocalSurface]
    image_surface: LocalAsphereSurface
    front_surface: LocalSurface
    back_surface: LocalSurface
    fit_spec: FitSpec
    wavelength_nm: float
    surface_distances_mm: tuple[float, ...]
    image_distance_value_mm: float
    initial_ior: float
    object_distance_mm: float
    exit_pupil_position_mm: float
    stop_semi_diameter_mm: float | None = None
    reference_ray: RayBundle | None = None
    physical_fft_pixel_pitch_mm: float | None = None

    @property
    def image_distance_mm(self) -> float:
        return float(self.image_distance_value_mm)

    def release_biot_lens(self) -> None:
        """Drop the heavyweight BIOT object after aiming metadata has been captured."""
        self.lens = None


def material_ior(material, wavelength_nm: float) -> float:
    value = material.ior(torch.as_tensor(float(wavelength_nm), dtype=torch.float64))
    return _as_float(value)


def _convert_asphere_surface(source, *, n_after: float) -> LocalAsphereSurface:
    if source.type != "S":
        raise TypeError(f"expected BIOT S surface, got {source.type}")
    coeff = (
        None
        if source.coeff is None
        else [float(value.detach().cpu().item()) for value in source.coeff.reshape(-1)]
    )
    return LocalAsphereSurface(
        semi_diameter_mm=float(source.semi_dia),
        curvature_inv_mm=_as_float(source.c),
        conic=_as_float(source.k),
        coeff=coeff,
        n_after=float(n_after),
        is_aperture=bool(getattr(source, "isaperture", False)),
    )


def biot_fft_defocus_shift_mm(system: FittedE2ESystem) -> float:
    """BIOT/BIOT_vis use the workbook image plane; finite distance comes from ray divergence."""
    _ = system
    return 0.0


def load_lens_for_field(
    excel_path: str | Path,
    *,
    object_distance: float | str,
    field_x_deg: float,
    field_y_deg: float,
    wavelength_nm: float = 555.0,
    device: torch.device | str = "cpu",
) -> tuple[Lensdata, Path | None]:
    """Load a BIOT Lensdata after applying the same field mapping as multi_rays."""
    text = str(object_distance).strip().lower()
    obj_dist = "Infinity" if text in {"inf", "infinity"} else float(object_distance)
    # Keep the temporary workbook beside its source workbook. GridSag paths in
    # the workbook are intentionally relative to that directory; placing the
    # temporary file in the process CWD breaks those references for exported
    # optimized systems.
    source_path = Path(excel_path)
    temp_path = source_path.parent / (
        f"temp_e2e_validation_{text}_field{field_x_deg}_{field_y_deg}.xlsx"
    )
    modify_excel_config(str(excel_path), temp_path, obj_dist, float(field_x_deg), float(field_y_deg))
    lens = Lensdata(device=torch.device(device))
    lens.aperture = 2.0
    lens.view_type = "angle"
    lens.FOV = 10
    lens.wavelengths = torch.tensor([float(wavelength_nm)], device=torch.device(device))
    lens.wavelengths_center = torch.tensor(
        [float(wavelength_nm)], device=torch.device(device)
    )
    lens.aimming = True
    lens.load_file(temp_path, extension=".xlsx")
    aperture_index = getattr(lens, "aperture_ind", None)
    if aperture_index is None or not 0 <= int(aperture_index) < len(lens.surfaces):
        raise RuntimeError(f"E2E prescription has no valid physical aperture surface: {excel_path}")
    aperture_surface = lens.surfaces[int(aperture_index)]
    physical_stop = float(getattr(aperture_surface, "semi_dia", math.nan))
    if (
        not bool(getattr(aperture_surface, "isaperture", False))
        or not math.isfinite(physical_stop)
        or physical_stop <= 0.0
    ):
        raise RuntimeError(
            f"E2E physical stop must be a finite positive aperture surface: {excel_path}"
        )
    lens.aperture = physical_stop
    return lens, temp_path


def build_fitted_e2e_system(
    excel_path: str | Path,
    *,
    object_distance: float | str = 1000.0,
    field_x_deg: float = 0.0,
    field_y_deg: float = 0.0,
    wavelength_nm: float = 555.0,
    fit_spec: FitSpec = FitSpec(),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    train_back_surface: bool = False,
    back_perturbation: torch.nn.Module | None = None,
    prefit_controls: dict[int, torch.Tensor] | None = None,
) -> tuple[FittedE2ESystem, Path | None]:
    lens, temp_path = load_lens_for_field(
        excel_path,
        object_distance=object_distance,
        field_x_deg=field_x_deg,
        field_y_deg=field_y_deg,
        wavelength_nm=wavelength_nm,
        device=device,
    )
    controls = {} if prefit_controls is None else prefit_controls
    front_source = lens.surfaces[1]
    front = _convert_asphere_surface(
        front_source,
        n_after=material_ior(lens.materials[2], wavelength_nm),
    )
    back_source = lens.surfaces[2]
    if train_back_surface:
        if back_perturbation is not None:
            raise ValueError("trainable fitted back surface and NURBS residual are mutually exclusive")
        back: LocalSurface = _fit_surface_from_lens(
            Path(temp_path or excel_path),
            2,
            lens,
            fit_spec,
            wavelength_nm,
            dtype,
            device,
            trainable=True,
            prefit_control_mm=controls.get(2),
        )
    else:
        if not isinstance(back_source, GridSag):
            raise TypeError(
                "Original GridSag + NURBS residual contract requires surface 2 to be optics.GridSag"
            )
        back = LocalGridSagSurface(
            back_source,
            n_after=material_ior(lens.materials[3], wavelength_nm),
            dtype=dtype,
            device=device,
            perturbation=back_perturbation,
        )
    converted: list[LocalSurface] = []
    for index, surface in enumerate(lens.surfaces):
        if index == 1:
            converted.append(front)
        elif index == 2:
            converted.append(back)
        elif isinstance(surface, Gradient_3) or surface.type in {"GRIN3", "GRAD3"}:
            if not isinstance(surface, Gradient_3):
                raise TypeError(
                    f"surface {index} declares {surface.type} but is not optics.Gradient_3"
                )
            converted.append(LocalGradient3Surface(surface))
        elif surface.type == "CB":
            n_after = material_ior(lens.materials[index + 1], wavelength_nm)
            converted.append(
                LocalCoordinateBreakSurface(
                    semi_diameter_mm=float("inf") if surface.semi_dia == float("inf") else float(surface.semi_dia),
                    n_after=n_after,
                    tilt_x_deg=_as_float(surface.tilt_x),
                    tilt_y_deg=_as_float(surface.tilt_y),
                    tilt_z_deg=_as_float(surface.tilt_z),
                )
            )
        elif surface.type == "S":
            n_after = material_ior(lens.materials[index + 1], wavelength_nm)
            converted.append(_convert_asphere_surface(surface, n_after=n_after))
        else:
            raise NotImplementedError(f"Unsupported e2e validation surface type: {surface.type}")
    img = lens.img
    image_surface = LocalAsphereSurface(
        semi_diameter_mm=float(img.semi_dia),
        curvature_inv_mm=_as_float(img.c),
        conic=_as_float(img.k),
        coeff=None if img.coeff is None else [float(v.detach().cpu().item()) for v in img.coeff.reshape(-1)],
        n_after=material_ior(lens.materials[-1], wavelength_nm),
    )
    return (
        FittedE2ESystem(
            lens=lens,
            surfaces=converted,
            image_surface=image_surface,
            front_surface=front,
            back_surface=back,
            fit_spec=fit_spec,
            wavelength_nm=float(wavelength_nm),
            surface_distances_mm=tuple(
                0.0 if index == 0 else _as_float(lens.surfaces[index - 1].thickness)
                for index in range(len(lens.surfaces))
            ),
            image_distance_value_mm=_as_float(lens.surfaces[-1].thickness),
            initial_ior=material_ior(lens.materials[0], wavelength_nm),
            object_distance_mm=_as_float(lens.obj.thickness) if lens.obj is not None else 0.0,
            exit_pupil_position_mm=_as_float(lens.find_exp()),
            stop_semi_diameter_mm=float(lens.stop_semi_diameter()),
        ),
        temp_path,
    )


def _fit_surface_from_lens(
    excel_path: Path,
    surface_index: int,
    lens: Lensdata,
    fit_spec: FitSpec,
    wavelength_nm: float,
    dtype: torch.dtype,
    device: torch.device | str,
    *,
    trainable: bool,
    perturbation: torch.nn.Module | None = None,
    prefit_control_mm: torch.Tensor | None = None,
) -> LocalBSplineSurface:
    biot_surface = lens.surfaces[surface_index]
    semi = float(biot_surface.semi_dia)
    domain = SurfaceDomain(x_range_mm=(-semi, semi), y_range_mm=(-semi, semi))
    if prefit_control_mm is None:
        x, y, sag, mask, _ = sample_biot_surface(
            excel_path,
            surface_index=surface_index,
            sample_shape=fit_spec.sample_shape,
            device=device,
        )
        control, _ = fit_bspline_control_grid(
            x.to(dtype=dtype),
            y.to(dtype=dtype),
            sag.to(dtype=dtype),
            mask,
            control_shape=fit_spec.control_shape,
            domain=domain,
            degree=fit_spec.degree,
        )
    else:
        expected = tuple(int(value) for value in fit_spec.control_shape)
        if tuple(prefit_control_mm.shape) != expected:
            raise ValueError(f"prefit control shape {tuple(prefit_control_mm.shape)} != {expected}")
        control = prefit_control_mm.to(device=device, dtype=dtype)
    return LocalBSplineSurface(
        control_mm=control.to(device=device, dtype=dtype),
        domain=domain,
        degree=fit_spec.degree,
        semi_diameter_mm=semi,
        n_after=material_ior(lens.materials[surface_index + 1], wavelength_nm),
        trainable=trainable,
        perturbation=perturbation,
    )


def make_aimed_pupil_rays(
    system: FittedE2ESystem,
    *,
    sample_count: int,
    pupil_radius_mm: float | None,
    field_x_deg: float,
    field_y_deg: float,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> RayBundle:
    """Use the BIOT/BIOT_vis pupil launch contract without changing ray count."""
    if system.lens is None:
        raise RuntimeError("BIOT lens was released before pupil-ray aiming")
    stop_radius = system.stop_semi_diameter_mm
    if stop_radius is None:
        raise RuntimeError("fitted system is missing the physical stop semi-diameter")
    radius = float(stop_radius if pupil_radius_mm is None else pupil_radius_mm)
    if not math.isclose(radius, float(stop_radius), rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            f"pupil_radius_mm={radius:g} does not match the physical stop radius "
            f"{float(stop_radius):g} mm"
        )
    points, weights = pupil_disk_grid(sample_count, radius, device=device, dtype=dtype)
    origins, directions, launch_opl, wavelength = _biot_aimed_launch(
        system, points, dtype=dtype, device=device
    )
    _ = (field_x_deg, field_y_deg)
    return RayBundle(
        origins_mm=origins,
        directions=directions,
        weights=weights,
        wavelength_nm=wavelength,
        launch_opl_mm=launch_opl,
    ).normalized()


def _biot_aimed_launch(
    system: FittedE2ESystem,
    pupil_targets_mm: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror ``Lensdata.fft_psf_i`` chief ray, finite object and aiming semantics."""
    if system.lens is None:
        raise RuntimeError("BIOT lens was released before pupil-ray aiming")
    if pupil_targets_mm.ndim != 2 or pupil_targets_mm.shape[-1] != 2:
        raise ValueError("pupil_targets_mm must have shape [N, 2]")
    lens = system.lens
    wavelength = torch.as_tensor(float(system.wavelength_nm), device=device, dtype=dtype)
    with torch.no_grad():
        ray_rel = lens._chief_ray_at_first_surface(wavelength, defocus_shift=0.0)
        direction = ray_rel.d.reshape(1, 3).to(device=device, dtype=dtype)
        chief_origin = ray_rel.o.reshape(-1, 3)[0].to(device=device, dtype=dtype)
        object_thickness = float(lens.obj.thickness)
        object_point = None
        if object_thickness > 0.0:
            axial_direction = direction[..., 2]
            if (
                not bool(torch.isfinite(axial_direction).all())
                or bool((axial_direction == 0.0).any())
            ):
                raise RuntimeError("BIOT finite-object chief ray has invalid axial direction")
            travel_back = object_thickness / axial_direction
            object_point = chief_origin.reshape(1, 3) - travel_back[..., None] * direction
        target = pupil_targets_mm.to(device=device, dtype=dtype).reshape(-1, 1, 2)
        p_val = chief_origin[:2].reshape(1, 1, 2).expand_as(target)
        x_aim, y_aim = lens.ray_aimming(
            p_val,
            direction.reshape(1, 1, 3),
            target,
            wavelength,
            tolerance=1.0e-6,
            it_max=1000,
            is_plot=False,
            is_fixed=True,
            p_obj=None if object_point is None else object_point.reshape(1, 1, 3),
        )
        origins = torch.stack(
            (x_aim.reshape(-1), y_aim.reshape(-1), torch.zeros_like(x_aim.reshape(-1))),
            dim=-1,
        )
        if object_point is None:
            directions = direction.expand_as(origins)
            launch_opl = torch.sum(origins * directions, dim=-1)
        else:
            directions = normalize_vector(origins - object_point.reshape(1, 3))
            launch_opl = torch.linalg.norm(origins - object_point.reshape(1, 3), dim=-1)
    if not (
        bool(torch.isfinite(origins).all())
        and bool(torch.isfinite(directions).all())
        and bool(torch.isfinite(launch_opl).all())
    ):
        raise RuntimeError("non-finite BIOT pupil launch geometry")
    return origins, directions, launch_opl, wavelength


def trace_system_to_image(system: FittedE2ESystem, rays: RayBundle) -> E2ETraceResult:
    trace = trace_system_to_image_with_phase(system, rays)
    return E2ETraceResult(
        spots_mm=trace.spots_mm,
        valid=trace.valid,
        weights=trace.weights,
        final_ray=trace.final_ray,
    )


def _equal_tensor_or_value(left, right) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            return False
        return bool(torch.equal(left, right))
    return left == right


def _assert_surface_batch_compatible(
    reference: LocalSurface,
    candidate: LocalSurface,
    *,
    surface_index: int,
) -> None:
    if type(candidate) is not type(reference):
        raise ValueError(f"case-batch surface type mismatch at index {surface_index}")
    for name in ("semi_diameter_mm", "n_after", "is_aperture"):
        if getattr(candidate, name) != getattr(reference, name):
            raise ValueError(f"case-batch surface {name} mismatch at index {surface_index}")
    if isinstance(reference, LocalCoordinateBreakSurface):
        # Field-dependent tilts are intentionally handled by batched rotations.
        return
    for name in (
        "curvature_inv_mm",
        "conic",
        "coeff",
        "degree",
        "degree_x",
        "degree_y",
        "domain",
        "thickness_mm",
        "integration_step_mm",
    ):
        if hasattr(reference, name) and not _equal_tensor_or_value(
            getattr(reference, name), getattr(candidate, name)
        ):
            raise ValueError(f"case-batch surface {name} mismatch at index {surface_index}")
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    if set(reference_state) != set(candidate_state):
        raise ValueError(f"case-batch surface state keys mismatch at index {surface_index}")
    for name, value in reference_state.items():
        if not torch.equal(value, candidate_state[name]):
            raise ValueError(
                f"case-batch surface tensor {name} mismatch at index {surface_index}"
            )
    if isinstance(reference, LocalGradient3Surface):
        source_names = (
            "c", "k", "coeff", "n0", "Nr2", "Nr4", "Nr6",
            "Nz1", "Nz2", "Nz3", "delta_t", "material_name",
        )
        for name in source_names:
            if not _equal_tensor_or_value(
                getattr(reference.source, name), getattr(candidate.source, name)
            ):
                raise ValueError(
                    f"case-batch Gradient_3 {name} mismatch at index {surface_index}"
                )


def _assert_system_batch_compatible(systems: Sequence[FittedE2ESystem]) -> None:
    if not systems:
        raise ValueError("cannot batch an empty optical-system sequence")
    reference = systems[0]
    for case_index, system in enumerate(systems[1:], start=1):
        if len(system.surfaces) != len(reference.surfaces):
            raise ValueError(f"case-batch surface count mismatch for case {case_index}")
        if len(system.surface_distances_mm) != len(reference.surface_distances_mm):
            raise ValueError(f"case-batch distance count mismatch for case {case_index}")
        if not math.isclose(
            float(system.wavelength_nm), float(reference.wavelength_nm), rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError(f"case-batch wavelength mismatch for case {case_index}")
        for surface_index, (left, right) in enumerate(
            zip(reference.surfaces, system.surfaces)
        ):
            _assert_surface_batch_compatible(left, right, surface_index=surface_index)
        _assert_surface_batch_compatible(
            reference.image_surface,
            system.image_surface,
            surface_index=len(reference.surfaces),
        )


def _stack_ray_bundles(rays: Sequence[RayBundle]) -> RayBundle:
    if not rays:
        raise ValueError("cannot stack an empty ray sequence")
    reference_shape = tuple(rays[0].origins_mm.shape)
    if any(tuple(ray.origins_mm.shape) != reference_shape for ray in rays):
        raise ValueError("case-batch rays must have matching shapes")
    launch_states = [ray.launch_opl_mm is not None for ray in rays]
    if any(state != launch_states[0] for state in launch_states):
        raise ValueError("case-batch launch OPL presence must match")
    launch = None
    if launch_states[0]:
        launch = torch.stack([ray.launch_opl_mm for ray in rays], dim=0)
    return RayBundle(
        origins_mm=torch.stack([ray.origins_mm for ray in rays], dim=0),
        directions=torch.stack([ray.directions for ray in rays], dim=0),
        weights=torch.stack([ray.weights for ray in rays], dim=0),
        wavelength_nm=torch.stack([ray.wavelength_nm.reshape(()) for ray in rays], dim=0),
        launch_opl_mm=launch,
    )


@dataclass(frozen=True)
class _BatchImageSurfaceTrace:
    spots_mm: torch.Tensor
    valid: torch.Tensor
    image_ray: RayBundle
    optical_path_mm: torch.Tensor
    continuous_opl_mm: torch.Tensor
    phase_to_image_rad: torch.Tensor
    image_ior: torch.Tensor


def _apply_case_batch_coordinate_break(
    surfaces: Sequence[LocalCoordinateBreakSurface], ray: RayBundle
) -> RayBundle:
    rotations = rotation_matrix_xyz(
        torch.as_tensor(
            [surface.tilt_x_deg for surface in surfaces],
            device=ray.device,
            dtype=ray.dtype,
        ),
        torch.as_tensor(
            [surface.tilt_y_deg for surface in surfaces],
            device=ray.device,
            dtype=ray.dtype,
        ),
        torch.as_tensor(
            [surface.tilt_z_deg for surface in surfaces],
            device=ray.device,
            dtype=ray.dtype,
        ),
        device=ray.device,
        dtype=ray.dtype,
    )
    origins = torch.einsum("bij,bnj->bni", rotations, ray.origins_mm)
    directions = torch.einsum("bij,bnj->bni", rotations, ray.directions)
    return ray.with_state(origins, directions, weights=ray.weights)


def _trace_case_batch_to_image_surface(
    systems: Sequence[FittedE2ESystem],
    rays: RayBundle,
    *,
    launch_reference_mm: torch.Tensor,
) -> _BatchImageSurfaceTrace:
    batch_size = len(systems)
    if rays.origins_mm.ndim != 3 or int(rays.origins_mm.shape[0]) != batch_size:
        raise ValueError("batched rays must have shape [B,N,3]")
    current = rays.normalized()
    active = torch.ones_like(current.weights, dtype=torch.bool)
    n_current = torch.as_tensor(
        [float(system.initial_ior) for system in systems],
        device=current.device,
        dtype=current.dtype,
    ).reshape(batch_size, 1)
    wavelength_mm = torch.as_tensor(
        [float(system.wavelength_nm) * 1.0e-6 for system in systems],
        device=current.device,
        dtype=current.dtype,
    ).reshape(batch_size, 1)
    initial_distance = (
        torch.sum(current.origins_mm * current.directions, dim=-1)
        if current.launch_opl_mm is None
        else current.launch_opl_mm
    )
    initial_relative_opl = initial_distance - launch_reference_mm
    phase_rad = initial_relative_opl * (2.0 * torch.pi / wavelength_mm)
    optical_path = initial_relative_opl.clone()
    continuous_opl = initial_relative_opl

    for surface_index in range(len(systems[0].surfaces)):
        case_surfaces = [system.surfaces[surface_index] for system in systems]
        surface = case_surfaces[0]
        distances = torch.as_tensor(
            [float(system.surface_distances_mm[surface_index]) for system in systems],
            device=current.device,
            dtype=current.dtype,
        ).reshape(batch_size, 1)
        points, hit_valid, segment_distance = surface.intersect(current, distances)
        normals = surface.normal_at(points)
        if isinstance(surface, LocalGradient3Surface):
            n_entry = surface.index_at(points)
            refracted, refract_valid = _snell(
                current.directions, normals, n_current, n_entry
            )
        else:
            refracted, refract_valid = _snell(
                current.directions, normals, n_current, surface.n_after
            )
        step_valid = active & hit_valid & refract_valid
        segment_opl = segment_distance * n_current
        phase_rad = phase_rad + torch.where(
            step_valid,
            2.0 * torch.pi * segment_opl / wavelength_mm,
            torch.zeros_like(segment_opl),
        )
        optical_path = optical_path + torch.where(
            step_valid, segment_opl, torch.zeros_like(segment_distance)
        )
        continuous_opl = continuous_opl + torch.where(
            step_valid, segment_opl, torch.zeros_like(segment_distance)
        )
        if isinstance(surface, LocalGradient3Surface):
            if surface_index + 1 >= len(systems[0].surfaces):
                raise RuntimeError("Gradient 3 medium has no terminating surface")
            next_surfaces = [system.surfaces[surface_index + 1] for system in systems]
            optical_momentum = n_entry[..., None] * refracted
            grin_points, grin_momentum, grin_opl, grin_valid = surface.trace_to_next_surface(
                points,
                optical_momentum,
                next_surfaces[0],
                case_axis=0,
            )
            n_exit = surface.index_at(grin_points)
            grin_direction = normalize_vector(
                grin_momentum / _safe_denominator(n_exit)[..., None]
            )
            grin_finite = (
                torch.all(torch.isfinite(grin_points), dim=-1)
                & torch.all(torch.isfinite(grin_direction), dim=-1)
                & torch.isfinite(grin_opl)
                & torch.isfinite(n_exit)
                & (n_exit > 0.0)
            )
            active = step_valid & grin_valid & grin_finite
            phase_rad = phase_rad + torch.where(
                active,
                2.0 * torch.pi * grin_opl / wavelength_mm,
                torch.zeros_like(grin_opl),
            )
            optical_path = optical_path + torch.where(
                active, grin_opl, torch.zeros_like(grin_opl)
            )
            continuous_opl = continuous_opl + torch.where(
                active, grin_opl, torch.zeros_like(grin_opl)
            )
            current = current.with_state(
                grin_points,
                grin_direction,
                weights=current.weights * active.to(current.dtype),
            )
            n_current = n_exit
            continue
        active = step_valid
        current = current.with_state(
            points,
            refracted,
            weights=current.weights * active.to(current.dtype),
        )
        if isinstance(surface, LocalCoordinateBreakSurface):
            current = _apply_case_batch_coordinate_break(case_surfaces, current)
        else:
            current = surface.after_interaction(current)
        n_current = torch.as_tensor(
            surface.n_after, device=current.device, dtype=current.dtype
        )

    image_surface = systems[0].image_surface
    image_distances = torch.as_tensor(
        [float(system.image_distance_mm) for system in systems],
        device=current.device,
        dtype=current.dtype,
    ).reshape(batch_size, 1)
    image_points, image_valid, image_distance = image_surface.intersect(
        current, image_distances
    )
    valid = active & image_valid
    image_opl = image_distance * n_current
    phase_to_image = phase_rad + torch.where(
        valid,
        2.0 * torch.pi * image_opl / wavelength_mm,
        torch.zeros_like(image_opl),
    )
    optical_path_to_image = optical_path + torch.where(
        valid, image_opl, torch.zeros_like(image_distance)
    )
    continuous_opl_to_image = continuous_opl + torch.where(
        valid, image_opl, torch.zeros_like(image_distance)
    )
    image_ray = current.with_state(
        image_points,
        current.directions,
        weights=current.weights * valid.to(current.dtype),
    )
    return _BatchImageSurfaceTrace(
        spots_mm=image_points[..., :2],
        valid=valid,
        image_ray=image_ray,
        optical_path_mm=optical_path_to_image,
        continuous_opl_mm=continuous_opl_to_image,
        phase_to_image_rad=phase_to_image,
        image_ior=torch.as_tensor(
            n_current, device=current.device, dtype=current.dtype
        ),
    )


def _case_batch_reference_sphere_path_mm(
    systems: Sequence[FittedE2ESystem],
    image_ray: RayBundle,
    sensor_intersection_mm: torch.Tensor,
    reference_points_mm: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size = len(systems)
    chief_z = torch.as_tensor(
        [biot_fft_defocus_shift_mm(system) for system in systems],
        device=image_ray.device,
        dtype=image_ray.dtype,
    ).reshape(batch_size, 1)
    radius = -(
        torch.as_tensor(
            [float(system.exit_pupil_position_mm) for system in systems],
            device=image_ray.device,
            dtype=image_ray.dtype,
        ).reshape(batch_size, 1)
        - chief_z
    )
    if not bool(torch.isfinite(radius).all()) or bool((radius == 0.0).any()):
        raise RuntimeError("case-batch reference-sphere radii must be finite and non-zero")
    valid = valid_mask.to(device=image_ray.device, dtype=torch.bool)
    if valid.shape != image_ray.directions[..., 2].shape:
        raise ValueError("case-batch reference-sphere valid mask shape mismatch")
    if not bool(valid.any(dim=1).all()):
        raise RuntimeError("each case-batch reference sphere requires valid rays")
    dx, dy, dz = (image_ray.directions[..., index] for index in range(3))
    if not bool(torch.isfinite(torch.where(valid, dz, torch.zeros_like(dz))).all()):
        raise RuntimeError("valid case-batch reference-sphere directions are non-finite")
    if bool(torch.where(valid, dz == 0.0, torch.zeros_like(valid)).any()):
        raise RuntimeError("valid case-batch reference-sphere axial directions are zero")
    denominator = torch.where(valid, dz, torch.ones_like(dz))
    projected_distance = -(
        radius - chief_z + image_ray.origins_mm[..., 2]
    ) / denominator
    projected = sensor_intersection_mm + image_ray.directions * projected_distance[..., None]
    pupil_ref = projected - reference_points_mm
    pupil_ref = pupil_ref.clone()
    pupil_ref[..., 2] = 0.0
    xn, yn = pupil_ref[..., 0], pupil_ref[..., 1]
    b = dz - (dx * xn + dy * yn) / radius
    h = (xn.square() + yn.square()) / radius
    discriminant = b.square() - h / radius
    valid_discriminant = torch.where(valid, discriminant, torch.ones_like(discriminant))
    if not bool(torch.isfinite(valid_discriminant).all()) or bool((valid_discriminant < 0.0).any()):
        raise RuntimeError("valid case-batch rays do not intersect the reference sphere")
    sqrt_disc = torch.sqrt(valid_discriminant)
    delta1 = b - sqrt_disc
    delta2 = b + sqrt_disc
    delta = torch.where(delta1.abs() < delta2.abs(), delta1, delta2)
    reference_path = projected_distance + delta * radius
    if not bool(torch.isfinite(torch.where(valid, reference_path, torch.zeros_like(reference_path))).all()):
        raise RuntimeError("valid case-batch reference paths are non-finite")
    return torch.where(valid, reference_path, torch.zeros_like(reference_path))


def trace_system_batch_to_image_with_phase(
    systems: Sequence[FittedE2ESystem],
    rays_by_case: Sequence[RayBundle],
    *,
    phase_reference: str = "image_surface",
) -> E2EPhaseTraceResult:
    """Trace a true tensorized case batch with shapes ``[B,N,...]``."""
    if phase_reference not in {"image_surface", "biot_reference_sphere"}:
        raise ValueError("phase_reference must be 'image_surface' or 'biot_reference_sphere'")
    systems = tuple(systems)
    rays_by_case = tuple(rays_by_case)
    if len(systems) != len(rays_by_case):
        raise ValueError("case-batch systems and rays must have matching lengths")
    _assert_system_batch_compatible(systems)
    reference_rays: list[RayBundle] = []
    for case_index, system in enumerate(systems):
        if system.reference_ray is None:
            raise RuntimeError(f"case {case_index} is missing its pre-aimed reference ray")
        reference_rays.append(system.reference_ray)
    stacked_reference = _stack_ray_bundles(reference_rays)
    if stacked_reference.launch_opl_mm is None:
        raise RuntimeError("case-batch reference rays are missing launch OPL")
    launch_reference = stacked_reference.launch_opl_mm.reshape(len(systems), -1)[:, :1]
    reference_trace = _trace_case_batch_to_image_surface(
        systems,
        stacked_reference,
        launch_reference_mm=launch_reference,
    )

    stacked_rays = _stack_ray_bundles(rays_by_case)
    main_trace = _trace_case_batch_to_image_surface(
        systems,
        stacked_rays,
        launch_reference_mm=launch_reference,
    )
    phase_rad = main_trace.phase_to_image_rad
    reference_opl = main_trace.continuous_opl_mm
    if phase_reference == "biot_reference_sphere":
        reference_points = reference_trace.image_ray.origins_mm
        reference_path = _case_batch_reference_sphere_path_mm(
            systems,
            main_trace.image_ray,
            main_trace.image_ray.origins_mm,
            reference_points,
            valid_mask=main_trace.valid,
        )
        # Match the scalar BIOT path: the reference-sphere segment propagates
        # in the medium immediately before the image plane.  The image
        # surface's ``n_after`` describes the material after that plane and is
        # not the optical-path multiplier for this segment.
        n_image = main_trace.image_ior
        wavelength_mm = torch.as_tensor(
            [float(system.wavelength_nm) * 1.0e-6 for system in systems],
            device=phase_rad.device,
            dtype=phase_rad.dtype,
        ).reshape(len(systems), 1)
        phase_rad = phase_rad + 2.0 * torch.pi * reference_path * n_image / wavelength_mm
        reference_opl = reference_opl + reference_path * n_image
    return E2EPhaseTraceResult(
        spots_mm=main_trace.spots_mm,
        valid=main_trace.valid,
        weights=stacked_rays.weights,
        final_ray=main_trace.image_ray,
        optical_path_mm=main_trace.optical_path_mm,
        reference_opl_mm=reference_opl,
        phase_rad=phase_rad,
    )


def trace_system_to_image_with_phase(
    system: FittedE2ESystem,
    rays: RayBundle,
    *,
    phase_reference: str = "image_surface",
) -> E2EPhaseTraceResult:
    if phase_reference not in {"image_surface", "biot_reference_sphere"}:
        raise ValueError("phase_reference must be 'image_surface' or 'biot_reference_sphere'")
    current = rays.normalized()
    active = torch.ones_like(current.weights, dtype=torch.bool)
    n_current = torch.as_tensor(
        float(system.initial_ior), device=current.device, dtype=current.dtype
    )
    wavelength_mm = torch.as_tensor(float(system.wavelength_nm) * 1.0e-6, device=current.device, dtype=current.dtype)
    initial_distance = (
        torch.sum(current.origins_mm * current.directions, dim=-1)
        if current.launch_opl_mm is None
        else current.launch_opl_mm
    )
    launch_reference = torch.zeros((), device=current.device, dtype=current.dtype)
    if current.launch_opl_mm is not None:
        if system.reference_ray is None:
            if system.lens is None:
                raise RuntimeError("finite-distance launch is missing its aimed reference ray")
            system.reference_ray = make_aimed_reference_ray(
                system, dtype=current.dtype, device=current.device
            )
        if system.reference_ray.launch_opl_mm is None:
            raise RuntimeError("aimed reference ray is missing launch_opl_mm")
        launch_reference = system.reference_ray.launch_opl_mm.to(
            device=current.device, dtype=current.dtype
        ).reshape(-1)[0]
    initial_relative_opl = initial_distance - launch_reference
    phase_rad = initial_relative_opl * (2.0 * torch.pi / wavelength_mm)
    optical_path = initial_relative_opl.clone()
    continuous_opl = initial_relative_opl
    for index, surface in enumerate(system.surfaces):
        distance = float(system.surface_distances_mm[index])
        points, hit_valid, segment_distance = surface.intersect(current, distance)
        normals = surface.normal_at(points)
        if isinstance(surface, LocalGradient3Surface):
            n_entry = surface.index_at(points)
            refracted, refract_valid = _snell(
                current.directions, normals, n_current, n_entry
            )
        else:
            refracted, refract_valid = _snell(
                current.directions, normals, n_current, surface.n_after
            )
        step_valid = active & hit_valid & refract_valid
        segment_opl = segment_distance * n_current
        phase_rad = phase_rad + torch.where(
            step_valid,
            2.0 * torch.pi * segment_opl / wavelength_mm,
            torch.zeros_like(segment_opl),
        )
        optical_path = optical_path + torch.where(
            step_valid,
            segment_opl,
            torch.zeros_like(segment_distance),
        )
        continuous_opl = continuous_opl + torch.where(
            step_valid,
            segment_opl,
            torch.zeros_like(segment_distance),
        )
        if isinstance(surface, LocalGradient3Surface):
            if index + 1 >= len(system.surfaces):
                raise RuntimeError("Gradient 3 medium has no terminating surface")
            optical_momentum = n_entry[..., None] * refracted
            grin_points, grin_momentum, grin_opl, grin_valid = (
                surface.trace_to_next_surface(
                    points, optical_momentum, system.surfaces[index + 1]
                )
            )
            n_exit = surface.index_at(grin_points)
            grin_direction = normalize_vector(
                grin_momentum / _safe_denominator(n_exit)[..., None]
            )
            grin_finite = (
                torch.all(torch.isfinite(grin_points), dim=-1)
                & torch.all(torch.isfinite(grin_direction), dim=-1)
                & torch.isfinite(grin_opl)
                & torch.isfinite(n_exit)
                & (n_exit > 0.0)
            )
            active = step_valid & grin_valid & grin_finite
            phase_rad = phase_rad + torch.where(
                active,
                2.0 * torch.pi * grin_opl / wavelength_mm,
                torch.zeros_like(grin_opl),
            )
            optical_path = optical_path + torch.where(
                active, grin_opl, torch.zeros_like(grin_opl)
            )
            continuous_opl = continuous_opl + torch.where(
                active, grin_opl, torch.zeros_like(grin_opl)
            )
            weights = current.weights * active.to(current.dtype)
            current = current.with_state(grin_points, grin_direction, weights=weights)
            n_current = n_exit
            continue
        active = step_valid
        weights = current.weights * active.to(current.dtype)
        current = current.with_state(points, refracted, weights=weights)
        current = surface.after_interaction(current)
        n_current = torch.as_tensor(
            surface.n_after, device=current.device, dtype=current.dtype
        )
    ray_before_image = current
    optical_path_before_image = optical_path
    image_points, image_valid, image_distance = system.image_surface.intersect(current, system.image_distance_mm)
    valid = active & image_valid
    image_opl = image_distance * n_current
    phase_to_image_rad = phase_rad + torch.where(
        valid,
        2.0 * torch.pi * image_opl / wavelength_mm,
        torch.zeros_like(image_opl),
    )
    optical_path_to_image = optical_path + torch.where(
        valid,
        image_opl,
        torch.zeros_like(image_distance),
    )
    continuous_opl_to_image = continuous_opl + torch.where(
        valid,
        image_opl,
        torch.zeros_like(image_distance),
    )
    image_ray = current.with_state(image_points, current.directions, weights=current.weights * valid.to(current.dtype))
    if phase_reference == "biot_reference_sphere":
        reference_path = _biot_reference_sphere_path_mm(
            system,
            image_ray,
            image_points,
            valid_mask=valid,
        )
        phase_rad = phase_to_image_rad + 2.0 * torch.pi * reference_path * n_current / wavelength_mm
        phase_opl = optical_path_to_image
        reference_opl = continuous_opl_to_image + reference_path * n_current
    else:
        phase_rad = phase_to_image_rad
        phase_opl = optical_path_to_image
        reference_opl = continuous_opl_to_image
    return E2EPhaseTraceResult(
        spots_mm=image_points[..., :2],
        valid=valid,
        weights=rays.weights,
        final_ray=image_ray,
        optical_path_mm=phase_opl,
        reference_opl_mm=reference_opl,
        phase_rad=phase_rad,
    )


def _biot_reference_sphere_phase(
    system: FittedE2ESystem,
    image_ray: RayBundle,
    sensor_intersection_mm: torch.Tensor,
    phase_to_image_rad: torch.Tensor,
    *,
    n_image: float,
    wavelength_mm: torch.Tensor,
) -> torch.Tensor:
    """Apply BIOT ``fft_psf_i`` reference-sphere path correction in torch.

    This mirrors the scalar formula around ``phase_for_reference()`` in
    ``optics.py``. The reference point is the current field's chief reference
    image point traced from the aimed center-pupil ray.
    """
    reference_path = _biot_reference_sphere_path_mm(
        system,
        image_ray,
        sensor_intersection_mm,
        valid_mask=image_ray.weights > 0.0,
    )
    return phase_to_image_rad + 2.0 * torch.pi * reference_path * float(n_image) / wavelength_mm


def _biot_reference_sphere_path_mm(
    system: FittedE2ESystem,
    image_ray: RayBundle,
    sensor_intersection_mm: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the continuous geometric path from the image plane to BIOT's sphere."""
    exp_pos = float(system.exit_pupil_position_mm)
    chief_z = biot_fft_defocus_shift_mm(system)
    reference_radius = -(float(exp_pos) - float(chief_z))
    if not math.isfinite(reference_radius) or reference_radius == 0.0:
        raise RuntimeError(
            f"BIOT reference-sphere radius must be finite and non-zero, got {reference_radius} mm"
        )
    r = torch.as_tensor(reference_radius, device=image_ray.device, dtype=image_ray.dtype)
    reference_point = _biot_chief_reference_image_point(
        system,
        device=image_ray.device,
        dtype=image_ray.dtype,
    )

    dx = image_ray.directions[..., 0]
    dy = image_ray.directions[..., 1]
    dz = image_ray.directions[..., 2]
    valid = valid_mask.to(device=image_ray.device, dtype=torch.bool)
    if valid.shape != dz.shape:
        raise ValueError("reference-sphere valid mask shape does not match rays")
    if not bool(valid.any()):
        raise RuntimeError("reference-sphere path has no valid rays")
    if not bool(torch.isfinite(dz[valid]).all()) or bool((dz[valid] == 0.0).any()):
        raise RuntimeError("valid reference-sphere rays have invalid axial direction")
    chief_z_t = torch.as_tensor(float(chief_z), device=image_ray.device, dtype=image_ray.dtype)
    denominator = torch.where(valid, dz, torch.ones_like(dz))
    projected_distance = -(r - chief_z_t + image_ray.origins_mm[..., 2]) / denominator
    projected_p = sensor_intersection_mm + image_ray.directions * projected_distance[..., None]
    pupil_ref = projected_p - reference_point
    pupil_ref = pupil_ref.clone()
    pupil_ref[..., 2] = 0.0
    xn = pupil_ref[..., 0]
    yn = pupil_ref[..., 1]
    b = dz - (dx * xn + dy * yn) / r
    h = (xn.pow(2) + yn.pow(2)) / r
    discriminant = b.pow(2) - h / r
    if not bool(torch.isfinite(discriminant[valid]).all()) or bool((discriminant[valid] < 0.0).any()):
        raise RuntimeError("valid rays do not intersect the BIOT reference sphere")
    sqrt_disc = torch.sqrt(torch.where(valid, discriminant, torch.ones_like(discriminant)))
    delta1 = b - sqrt_disc
    delta2 = b + sqrt_disc
    delta = torch.where(delta1.abs() < delta2.abs(), delta1, delta2)
    reference_path = projected_distance + delta * r
    if not bool(torch.isfinite(reference_path[valid]).all()):
        raise RuntimeError("valid BIOT reference-sphere paths are non-finite")
    return torch.where(valid, reference_path, torch.zeros_like(reference_path))


def _biot_chief_reference_image_point(
    system: FittedE2ESystem,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Trace BIOT's non-legacy center-pupil reference ray to the image plane."""
    if system.reference_ray is not None:
        reference_ray = RayBundle(
            origins_mm=system.reference_ray.origins_mm.to(device=device, dtype=dtype),
            directions=system.reference_ray.directions.to(device=device, dtype=dtype),
            weights=system.reference_ray.weights.to(device=device, dtype=dtype),
            wavelength_nm=system.reference_ray.wavelength_nm.to(device=device, dtype=dtype),
            launch_opl_mm=None if system.reference_ray.launch_opl_mm is None else system.reference_ray.launch_opl_mm.to(device=device, dtype=dtype),
        )
        reference_trace = trace_system_to_image(system, reference_ray)
        return reference_trace.final_ray.origins_mm.reshape(-1, 3)[0]
    if system.lens is None:
        raise RuntimeError("Missing pre-aimed reference ray after BIOT lens release")
    reference_ray = make_aimed_reference_ray(system, dtype=dtype, device=device)
    reference_trace = trace_system_to_image(system, reference_ray)
    return reference_trace.final_ray.origins_mm.reshape(-1, 3)[0]


def make_aimed_reference_ray(
    system: FittedE2ESystem,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> RayBundle:
    """Capture the fixed BIOT chief-reference input ray before releasing Lensdata."""
    if system.lens is None:
        raise RuntimeError("BIOT lens was released before reference-ray aiming")
    targets = torch.zeros((1, 2), device=device, dtype=dtype)
    origin, direction, launch_opl, wavelength = _biot_aimed_launch(
        system, targets, dtype=dtype, device=device
    )
    return RayBundle(
        origins_mm=origin,
        directions=direction,
        weights=torch.ones((1,), device=device, dtype=dtype),
        wavelength_nm=wavelength,
        launch_opl_mm=launch_opl,
    ).normalized()


def _snell(
    incident: torch.Tensor,
    normal: torch.Tensor,
    n_before: float | torch.Tensor,
    n_after: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    wi = normalize_vector(incident)
    n = normalize_vector(normal)
    n1 = torch.as_tensor(n_before, device=wi.device, dtype=wi.dtype)
    n2 = torch.as_tensor(n_after, device=wi.device, dtype=wi.dtype)
    eta = n1 / n2
    eta_v = eta[..., None] if eta.ndim > 0 else eta
    cos_i = (wi * n).sum(dim=-1)
    cos_t2 = 1.0 - eta.pow(2) * (1.0 - cos_i.pow(2))
    valid = (
        (n1 > 0.0)
        & (n2 > 0.0)
        & torch.isfinite(n1)
        & torch.isfinite(n2)
        & (cos_t2 > 0.0)
    )
    # BIOT/BIOT_vis ``Lensdata._refract`` uses a fixed 1e-8 floor after the
    # physical TIR validity test.  Keep the same near-critical numerical path.
    cos_t = torch.sqrt(cos_t2.clamp_min(1.0e-8))
    wt = cos_t[..., None] * n + eta_v * (wi - cos_i[..., None] * n)
    wt = normalize_vector(wt)
    return wt, valid & torch.all(torch.isfinite(wt), dim=-1)
