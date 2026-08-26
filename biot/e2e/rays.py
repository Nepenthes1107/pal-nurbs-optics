from __future__ import annotations

from dataclasses import dataclass

import torch


def normalize_vector(vector: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    """Normalize vectors along the last dimension.

    Args:
        vector: Tensor with shape ``[..., 3]``.
        eps: Minimum norm used only to avoid division by zero.

    Returns:
        Unit vector tensor with the same shape as ``vector``.
    """
    if vector.shape[-1] != 3:
        raise ValueError("vector must have shape [..., 3]")
    floor = eps if eps is not None else torch.finfo(vector.dtype).eps
    norm = torch.linalg.norm(vector, dim=-1, keepdim=True).clamp_min(floor)
    return vector / norm


@dataclass(frozen=True)
class RayBundle:
    """Geometric ray bundle for e2e tracing.

    Shapes:
        origins_mm: ``[..., 3]`` ray origins in mm.
        directions: ``[..., 3]`` unit direction vectors.
        weights: ``[...]`` relative ray energy weights. The default pupil
            sampling helpers normalize weights to sum to 1.
        wavelength_nm: scalar wavelength in nm, stored for metadata and later
            OPD/dispersion extensions.
        launch_opl_mm: optional ``[...]`` optical path accumulated before the
            virtual first-surface plane. For finite object distance this is
            the object-point-to-launch distance; for infinity it is
            ``dot(origin, direction)``.
    """

    origins_mm: torch.Tensor
    directions: torch.Tensor
    weights: torch.Tensor
    wavelength_nm: torch.Tensor
    launch_opl_mm: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.origins_mm.shape != self.directions.shape:
            raise ValueError("origins_mm and directions must have identical shape")
        if self.origins_mm.shape[-1] != 3:
            raise ValueError("origins_mm and directions must have shape [..., 3]")
        if self.weights.shape != self.origins_mm.shape[:-1]:
            raise ValueError("weights must have shape matching ray batch dimensions")
        if self.launch_opl_mm is not None and self.launch_opl_mm.shape != self.weights.shape:
            raise ValueError("launch_opl_mm must have shape matching ray batch dimensions")

    @property
    def device(self) -> torch.device:
        return self.origins_mm.device

    @property
    def dtype(self) -> torch.dtype:
        return self.origins_mm.dtype

    def normalized(self) -> "RayBundle":
        return RayBundle(
            origins_mm=self.origins_mm,
            directions=normalize_vector(self.directions),
            weights=self.weights,
            wavelength_nm=self.wavelength_nm,
            launch_opl_mm=self.launch_opl_mm,
        )

    def with_state(
        self,
        origins_mm: torch.Tensor,
        directions: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> "RayBundle":
        return RayBundle(
            origins_mm=origins_mm,
            directions=normalize_vector(directions),
            weights=self.weights if weights is None else weights,
            wavelength_nm=self.wavelength_nm,
            launch_opl_mm=self.launch_opl_mm,
        )


def field_direction(
    field_x_deg: float | torch.Tensor,
    field_y_deg: float | torch.Tensor,
    *,
    direction_z: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Create a unit direction from x/y field angles.

    The convention matches BIOT's angle-field sampling: components are
    proportional to ``tan(field_x)``, ``tan(field_y)``, and ``direction_z``.
    Angles are specified in degree and converted to radians before torch
    trigonometry.
    """
    fx = torch.as_tensor(field_x_deg, device=device, dtype=dtype)
    fy = torch.as_tensor(field_y_deg, device=device, dtype=dtype)
    tx = torch.tan(torch.deg2rad(fx))
    ty = torch.tan(torch.deg2rad(fy))
    dz = torch.full_like(tx + ty, float(direction_z))
    return normalize_vector(torch.stack((tx, ty, dz), dim=-1))


def pupil_disk_grid(
    sample_count: int,
    pupil_radius_mm: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a circular pupil by masking a square grid.

    Args:
        sample_count: Number of points along one square-grid axis.
        pupil_radius_mm: Pupil radius in mm.

    Returns:
        ``(points_mm, weights)`` where ``points_mm`` has shape ``[N, 2]`` and
        weights has shape ``[N]`` with sum 1.
    """
    if int(sample_count) <= 0:
        raise ValueError("sample_count must be positive")
    if float(pupil_radius_mm) <= 0:
        raise ValueError("pupil_radius_mm must be positive")
    coords = torch.linspace(
        -float(pupil_radius_mm),
        float(pupil_radius_mm),
        int(sample_count),
        device=device,
        dtype=dtype,
    )
    xx, yy = torch.meshgrid(coords, coords, indexing="xy")
    mask = xx.pow(2) + yy.pow(2) <= float(pupil_radius_mm) ** 2
    points = torch.stack((xx[mask], yy[mask]), dim=-1)
    if points.numel() == 0:
        raise ValueError("pupil sampling produced no valid points")
    weights = torch.full(
        (points.shape[0],),
        1.0 / float(points.shape[0]),
        device=device,
        dtype=dtype,
    )
    return points, weights


def make_pupil_rays(
    *,
    sample_count: int,
    pupil_radius_mm: float,
    field_x_deg: float | torch.Tensor = 0.0,
    field_y_deg: float | torch.Tensor = 0.0,
    pupil_z_mm: float = 0.0,
    direction_z: float = 1.0,
    wavelength_nm: float = 555.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> RayBundle:
    """Create field-directed rays launched from a circular pupil plane.

    Origins are in mm and lie on ``z = pupil_z_mm``. Directions follow field
    angles in degree and are normalized. Output shape is ``[N, 3]``.
    """
    points_xy, weights = pupil_disk_grid(
        sample_count,
        pupil_radius_mm,
        device=device,
        dtype=dtype,
    )
    z = torch.full((points_xy.shape[0], 1), float(pupil_z_mm), device=device, dtype=dtype)
    origins = torch.cat((points_xy, z), dim=-1)
    direction = field_direction(
        field_x_deg,
        field_y_deg,
        direction_z=direction_z,
        device=device,
        dtype=dtype,
    )
    directions = direction.reshape(1, 3).expand_as(origins)
    wavelength = torch.as_tensor(float(wavelength_nm), device=device, dtype=dtype)
    return RayBundle(origins, directions, weights, wavelength)
