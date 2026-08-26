from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from .enums import Device, DistortionType
from .serialization import (
    SCHEMA_VERSION,
    dataclass_to_dict,
    parse_float,
    require_fields,
    require_schema,
    utc_now_iso,
)


@dataclass
class SystemConfig:
    """Runtime system configuration for BIOT computations.

    Units:
    - object_distance_mm, pupil_radius_mm: mm. `inf` is represented by float("inf").
    - wavelength_nm: nm.
    - lens_rotation_deg: degree.
    - No GPU tensors are stored here; JSON round-trip is CPU-only and not autograd-aware.
    """

    excel_path: Path
    object_distance_mm: float
    excel_sha256: str = ""
    wavelength_nm: float = 555.0
    pupil_radius_mm: float = 2.0
    lens_rotation_deg: float = 0.0
    np_pupil: int = 256
    ni_image: int = 512
    zernike_n_max: int = 5
    device: Device = Device.AUTO
    lens_front_index: int | None = None
    lens_back_index: int | None = None
    legacy_pupil_phase: bool = False
    write_temp_excel: bool = True
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.device, Device):
            self.device = Device(self.device)
        self.excel_path = Path(self.excel_path)
        if isinstance(self.zernike_n_max, bool) or int(self.zernike_n_max) != self.zernike_n_max or int(self.zernike_n_max) < 0:
            raise ValueError("zernike_n_max must be a non-negative integer")
        self.zernike_n_max = int(self.zernike_n_max)

    def to_dict(self) -> dict:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SystemConfig":
        require_schema(data)
        require_fields(data, ["excel_path", "object_distance_mm"])
        return cls(
            excel_path=Path(data["excel_path"]),
            object_distance_mm=parse_float(data["object_distance_mm"]),
            excel_sha256=str(data.get("excel_sha256", "")),
            wavelength_nm=float(data.get("wavelength_nm", 555.0)),
            pupil_radius_mm=float(data.get("pupil_radius_mm", 2.0)),
            lens_rotation_deg=float(data.get("lens_rotation_deg", 0.0)),
            np_pupil=int(data.get("np_pupil", 256)),
            ni_image=int(data.get("ni_image", 512)),
            zernike_n_max=int(data.get("zernike_n_max", 5)),
            device=Device(data.get("device", Device.AUTO.value)),
            lens_front_index=data.get("lens_front_index"),
            lens_back_index=data.get("lens_back_index"),
            legacy_pupil_phase=bool(data.get("legacy_pupil_phase", False)),
            write_temp_excel=bool(data.get("write_temp_excel", True)),
            created_at=str(data.get("created_at", utc_now_iso())),
        )


@dataclass
class SingleFieldRequest:
    """Single-field PSF/MTF computation request.

    Units:
    - field_x_deg, field_y_deg: degree.
    - cutoff_cyc_per_mm: cycles/mm.
    - system contains length units in mm and wavelength in nm.
    - Service may use CPU or GPU depending on system.device; request itself stores no tensors.
    """

    system: SystemConfig
    field_x_deg: float
    field_y_deg: float
    cutoff_cyc_per_mm: float = 100.0
    with_mtf: bool = False
    with_chart_convolution: bool = False
    chart_path: Path | None = None
    output_dir: Path | None = None
    tag: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        if self.chart_path is not None:
            self.chart_path = Path(self.chart_path)

    def to_dict(self) -> dict:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SingleFieldRequest":
        require_schema(data)
        require_fields(data, ["system", "field_x_deg", "field_y_deg"])
        return cls(
            system=SystemConfig.from_dict(data["system"]),
            field_x_deg=float(data["field_x_deg"]),
            field_y_deg=float(data["field_y_deg"]),
            cutoff_cyc_per_mm=float(data.get("cutoff_cyc_per_mm", 100.0)),
            with_mtf=bool(data.get("with_mtf", False)),
            with_chart_convolution=bool(data.get("with_chart_convolution", False)),
            chart_path=Path(data["chart_path"]) if data.get("chart_path") else None,
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            tag=data.get("tag"),
            request_id=str(data.get("request_id", uuid4())),
            created_at=str(data.get("created_at", utc_now_iso())),
        )


@dataclass
class RayVisualizationRequest:
    """Request for detached real-ray optical-path visualization.

    Units: object distance and pupil radius are mm, field angles are degree,
    wavelength is nm, and surface_samples/counts are unitless. The service uses
    ``system.device`` and never modifies ``system.excel_path`` in place.
    """

    system: SystemConfig
    field_x_deg: float = 0.0
    field_y_deg: float = 0.0
    object_distance_mm: float | None = None
    wavelength_nm: float | None = None
    sampling: str = "fan"
    fan_axis: str = "y"
    fan_count: int = 9
    ring_count: int = 3
    azimuth_count: int = 12
    pupil_radius_mm: float | None = None
    include_chief_ray: bool = True
    surface_samples: int = 81
    output_dir: Path | None = None
    tag: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.sampling not in {"chief", "fan", "rings"}:
            raise ValueError(f"Unsupported ray visualization sampling mode: {self.sampling!r}")
        if self.fan_axis not in {"x", "y"}:
            raise ValueError(f"Unsupported fan axis: {self.fan_axis!r}")
        if int(self.fan_count) <= 0 or int(self.ring_count) <= 0 or int(self.azimuth_count) <= 0:
            raise ValueError("Ray visualization sampling counts must be positive.")
        if int(self.surface_samples) < 3:
            raise ValueError("surface_samples must be at least 3.")
        if self.pupil_radius_mm is not None and float(self.pupil_radius_mm) <= 0.0:
            raise ValueError("pupil_radius_mm must be positive when provided.")
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)

    @property
    def resolved_object_distance_mm(self) -> float:
        return float(self.system.object_distance_mm if self.object_distance_mm is None else self.object_distance_mm)

    @property
    def resolved_wavelength_nm(self) -> float:
        return float(self.system.wavelength_nm if self.wavelength_nm is None else self.wavelength_nm)

    def to_dict(self) -> dict:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RayVisualizationRequest":
        require_schema(data)
        require_fields(data, ["system"])
        return cls(
            system=SystemConfig.from_dict(data["system"]),
            field_x_deg=float(data.get("field_x_deg", 0.0)),
            field_y_deg=float(data.get("field_y_deg", 0.0)),
            object_distance_mm=parse_float(data["object_distance_mm"]) if data.get("object_distance_mm") is not None else None,
            wavelength_nm=float(data["wavelength_nm"]) if data.get("wavelength_nm") is not None else None,
            sampling=str(data.get("sampling", "fan")),
            fan_axis=str(data.get("fan_axis", "y")),
            fan_count=int(data.get("fan_count", 9)),
            ring_count=int(data.get("ring_count", 3)),
            azimuth_count=int(data.get("azimuth_count", 12)),
            pupil_radius_mm=float(data["pupil_radius_mm"]) if data.get("pupil_radius_mm") is not None else None,
            include_chief_ray=bool(data.get("include_chief_ray", True)),
            surface_samples=int(data.get("surface_samples", 81)),
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            tag=data.get("tag"),
            request_id=str(data.get("request_id", uuid4())),
            created_at=str(data.get("created_at", utc_now_iso())),
        )


@dataclass
class SweepRequest:
    """Field-grid PSF/MTF sweep request.

    Units:
    - field_*_deg: degree.
    - cutoff_cyc_per_mm: cycles/mm.
    - system length units are mm and wavelength is nm.
    - Service may use CPU or GPU through `system.device`; request stores no tensors
      and does not preserve autograd.
    """

    system: SystemConfig
    field_x_min_deg: float = -10.0
    field_x_max_deg: float = 10.0
    field_x_step_deg: float = 5.0
    field_y_min_deg: float = -10.0
    field_y_max_deg: float = 10.0
    field_y_step_deg: float = 5.0
    cutoff_cyc_per_mm: float = 100.0
    with_mtf: bool = False
    with_chart_stitch: bool = False
    with_mtf_grid: bool = False
    chart_path: Path | None = None
    output_dir: Path | None = None
    use_cache: bool = True
    tag: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        if self.chart_path is not None:
            self.chart_path = Path(self.chart_path)

    def to_dict(self) -> dict:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SweepRequest":
        require_schema(data)
        require_fields(data, ["system"])
        return cls(
            system=SystemConfig.from_dict(data["system"]),
            field_x_min_deg=float(data.get("field_x_min_deg", -10.0)),
            field_x_max_deg=float(data.get("field_x_max_deg", 10.0)),
            field_x_step_deg=float(data.get("field_x_step_deg", 5.0)),
            field_y_min_deg=float(data.get("field_y_min_deg", -10.0)),
            field_y_max_deg=float(data.get("field_y_max_deg", 10.0)),
            field_y_step_deg=float(data.get("field_y_step_deg", 5.0)),
            cutoff_cyc_per_mm=float(data.get("cutoff_cyc_per_mm", 100.0)),
            with_mtf=bool(data.get("with_mtf", False)),
            with_chart_stitch=bool(data.get("with_chart_stitch", False)),
            with_mtf_grid=bool(data.get("with_mtf_grid", False)),
            chart_path=Path(data["chart_path"]) if data.get("chart_path") else None,
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            use_cache=bool(data.get("use_cache", True)),
            tag=data.get("tag"),
            request_id=str(data.get("request_id", uuid4())),
            created_at=str(data.get("created_at", utc_now_iso())),
        )


@dataclass
class PowerAstigmatismRequest:
    """Power and astigmatism curve request.

    Units:
    - fov_deg, lens_fov_deg: degree.
    - aperture_mm, differential_aperture_mm: mm.
    - wavelength_nm: nm.
    - target_focal_power_d: diopter.
    The service may use CPU or GPU through `system.device`; request
    serialization stores no tensors and does not preserve autograd.
    """

    system: SystemConfig
    fov_deg: float = 25.0
    field_num: int = 51
    axis: str = "y"
    lens_fov_deg: float = 25.0
    aperture_mm: float = 2.0
    wavelength_nm: float = 555.0
    differential_aperture_mm: float = 0.01
    target_focal_power_d: float = 0.0
    averfang_crib_diameter_mm: float = 80.0
    output_dir: Path | None = None
    tag: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)

    def to_dict(self) -> dict:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PowerAstigmatismRequest":
        require_schema(data)
        require_fields(data, ["system"])
        return cls(
            system=SystemConfig.from_dict(data["system"]),
            fov_deg=float(data.get("fov_deg", 25.0)),
            field_num=int(data.get("field_num", 51)),
            axis=str(data.get("axis", "y")),
            lens_fov_deg=float(data.get("lens_fov_deg", 25.0)),
            aperture_mm=float(data.get("aperture_mm", 2.0)),
            wavelength_nm=float(data.get("wavelength_nm", 555.0)),
            differential_aperture_mm=float(data.get("differential_aperture_mm", 0.01)),
            target_focal_power_d=float(data.get("target_focal_power_d", 0.0)),
            averfang_crib_diameter_mm=float(data.get("averfang_crib_diameter_mm", 80.0)),
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            tag=data.get("tag"),
            request_id=str(data.get("request_id", uuid4())),
            created_at=str(data.get("created_at", utc_now_iso())),
        )


@dataclass
class DistortionCurveRequest:
    """One-dimensional distortion curve request.

    Units:
    - fov_deg, lens_fov_deg: degree.
    - aperture_mm, near_object_distance_mm, pupil_distance_mm: mm.
    - wavelength_nm: nm.
    Output distortion is unitless and percent values are reported by the core
    metric table. The request stores no tensors and does not preserve autograd.
    """

    system: SystemConfig
    fov_deg: float = 25.0
    field_num: int = 51
    axis: str = "y"
    distortion_type: DistortionType = DistortionType.ROTATING_EYE_FAR
    lens_fov_deg: float = 25.0
    aperture_mm: float = 2.0
    wavelength_nm: float = 555.0
    near_object_distance_mm: float = 250.0
    pupil_distance_mm: float = 250.0
    output_dir: Path | None = None
    tag: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.distortion_type, DistortionType):
            self.distortion_type = DistortionType(self.distortion_type)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)

    def to_dict(self) -> dict:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DistortionCurveRequest":
        require_schema(data)
        require_fields(data, ["system"])
        return cls(
            system=SystemConfig.from_dict(data["system"]),
            fov_deg=float(data.get("fov_deg", 25.0)),
            field_num=int(data.get("field_num", 51)),
            axis=str(data.get("axis", "y")),
            distortion_type=DistortionType(data.get("distortion_type", DistortionType.ROTATING_EYE_FAR.value)),
            lens_fov_deg=float(data.get("lens_fov_deg", 25.0)),
            aperture_mm=float(data.get("aperture_mm", 2.0)),
            wavelength_nm=float(data.get("wavelength_nm", 555.0)),
            near_object_distance_mm=float(data.get("near_object_distance_mm", 250.0)),
            pupil_distance_mm=float(data.get("pupil_distance_mm", 250.0)),
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            tag=data.get("tag"),
            request_id=str(data.get("request_id", uuid4())),
            created_at=str(data.get("created_at", utc_now_iso())),
        )


@dataclass
class DistortionGridRequest:
    """Two-dimensional distortion grid request.

    Units:
    - fov_x_deg, fov_y_deg, lens_fov_deg: degree.
    - aperture_mm, near_object_distance_mm, pupil_distance_mm: mm.
    - wavelength_nm: nm.
    Grid output coordinates are either degree or mm as recorded in metadata.
    The request stores no tensors and does not preserve autograd.
    """

    system: SystemConfig
    fov_x_deg: float = 25.0
    fov_y_deg: float = 25.0
    field_num: int = 21
    display_grid_num: int = 21
    distortion_type: DistortionType = DistortionType.ROTATING_EYE_FAR
    lens_fov_deg: float = 25.0
    aperture_mm: float = 2.0
    wavelength_nm: float = 555.0
    near_object_distance_mm: float = 250.0
    pupil_distance_mm: float = 250.0
    fix_original_grid_axis_bug: bool = False
    output_dir: Path | None = None
    tag: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.distortion_type, DistortionType):
            self.distortion_type = DistortionType(self.distortion_type)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)

    def to_dict(self) -> dict:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DistortionGridRequest":
        require_schema(data)
        require_fields(data, ["system"])
        return cls(
            system=SystemConfig.from_dict(data["system"]),
            fov_x_deg=float(data.get("fov_x_deg", 25.0)),
            fov_y_deg=float(data.get("fov_y_deg", 25.0)),
            field_num=int(data.get("field_num", 21)),
            display_grid_num=int(data.get("display_grid_num", 21)),
            distortion_type=DistortionType(data.get("distortion_type", DistortionType.ROTATING_EYE_FAR.value)),
            lens_fov_deg=float(data.get("lens_fov_deg", 25.0)),
            aperture_mm=float(data.get("aperture_mm", 2.0)),
            wavelength_nm=float(data.get("wavelength_nm", 555.0)),
            near_object_distance_mm=float(data.get("near_object_distance_mm", 250.0)),
            pupil_distance_mm=float(data.get("pupil_distance_mm", 250.0)),
            fix_original_grid_axis_bug=bool(data.get("fix_original_grid_axis_bug", False)),
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            tag=data.get("tag"),
            request_id=str(data.get("request_id", uuid4())),
            created_at=str(data.get("created_at", utc_now_iso())),
        )
