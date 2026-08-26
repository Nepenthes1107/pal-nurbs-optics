from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .enums import ResultStatus
from .serialization import SCHEMA_VERSION, dataclass_to_dict, require_fields, require_schema, utc_now_iso


@dataclass
class SingleFieldResult:
    """Single-field PSF/MTF result.

    Units:
    - d_delta_mm: image-plane sampling interval in mm/pixel.
    - psf is a 2D energy-normalized intensity array [H, W].
    - mtf arrays, when present, are DC-normalized and CPU numpy arrays.
    - Arrays are kept in memory for GUI rendering and omitted from JSON serialization.
    """

    request_id: str
    request_snapshot: dict
    status: ResultStatus
    output_dir: Path | None = None
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None
    duration_seconds: float = 0.0
    artifacts: dict[str, Path] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    log_excerpt: str = ""
    error: str | None = None
    psf: np.ndarray | None = field(default=None, repr=False, compare=False)
    mtf_curve: np.ndarray | None = field(default=None, repr=False, compare=False)
    chart_image: np.ndarray | None = field(default=None, repr=False, compare=False)
    d_delta_mm: float | None = None
    mtf_metrics: dict | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        data = dataclass_to_dict(self)
        data.pop("psf", None)
        data.pop("mtf_curve", None)
        data.pop("chart_image", None)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SingleFieldResult":
        require_schema(data)
        require_fields(data, ["request_id", "request_snapshot", "status"])
        artifacts = {name: Path(path) for name, path in data.get("artifacts", {}).items()}
        return cls(
            request_id=str(data["request_id"]),
            request_snapshot=dict(data["request_snapshot"]),
            status=ResultStatus(data["status"]),
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            started_at=str(data.get("started_at", utc_now_iso())),
            finished_at=data.get("finished_at"),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            artifacts=artifacts,
            metrics=dict(data.get("metrics", {})),
            log_excerpt=str(data.get("log_excerpt", "")),
            error=data.get("error"),
            d_delta_mm=float(data["d_delta_mm"]) if data.get("d_delta_mm") is not None else None,
            mtf_metrics=data.get("mtf_metrics"),
        )


@dataclass
class SweepResult:
    """Field-grid PSF/MTF sweep result.

    Units:
    - field_grid entries are [field_x_deg, field_y_deg] in degree.
    - stitched_psf is a display-preview mosaic assembled from energy-normalized
      PSF arrays; it is not used for MTF or physics metrics.
    - Per-point PSF/MTF arrays remain in point artifact files. The in-memory
      stitched preview is omitted from JSON serialization.
    """

    request_id: str
    request_snapshot: dict
    status: ResultStatus
    output_dir: Path | None = None
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None
    duration_seconds: float = 0.0
    artifacts: dict[str, Path] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    point_metrics: list[dict] = field(default_factory=list)
    field_grid: list[tuple[float, float]] = field(default_factory=list)
    log_excerpt: str = ""
    error: str | None = None
    stitched_psf: np.ndarray | None = field(default=None, repr=False, compare=False)
    stitched_chart: np.ndarray | None = field(default=None, repr=False, compare=False)
    mtf_grid: np.ndarray | None = field(default=None, repr=False, compare=False)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        data = dataclass_to_dict(self)
        data.pop("stitched_psf", None)
        data.pop("stitched_chart", None)
        data.pop("mtf_grid", None)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SweepResult":
        require_schema(data)
        require_fields(data, ["request_id", "request_snapshot", "status"])
        artifacts = {name: Path(path) for name, path in data.get("artifacts", {}).items()}
        return cls(
            request_id=str(data["request_id"]),
            request_snapshot=dict(data["request_snapshot"]),
            status=ResultStatus(data["status"]),
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            started_at=str(data.get("started_at", utc_now_iso())),
            finished_at=data.get("finished_at"),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            artifacts=artifacts,
            metrics=dict(data.get("metrics", {})),
            point_metrics=list(data.get("point_metrics", [])),
            field_grid=[tuple(item) for item in data.get("field_grid", [])],
            log_excerpt=str(data.get("log_excerpt", "")),
            error=data.get("error"),
        )


@dataclass
class PowerAstigmatismResult:
    """Power/astigmatism curve result.

    Units:
    - table column `theta_deg`: degree.
    - power and astigmatism columns: diopter.
    - Arrays are CPU NumPy data for GUI rendering and omitted from JSON.
    """

    request_id: str
    request_snapshot: dict
    status: ResultStatus
    output_dir: Path | None = None
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None
    duration_seconds: float = 0.0
    artifacts: dict[str, Path] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    table_columns: list[str] = field(default_factory=list)
    table_rows: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    log_excerpt: str = ""
    error: str | None = None
    table_data: np.ndarray | None = field(default=None, repr=False, compare=False)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        data = dataclass_to_dict(self)
        data.pop("table_data", None)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PowerAstigmatismResult":
        require_schema(data)
        require_fields(data, ["request_id", "request_snapshot", "status"])
        artifacts = {name: Path(path) for name, path in data.get("artifacts", {}).items()}
        return cls(
            request_id=str(data["request_id"]),
            request_snapshot=dict(data["request_snapshot"]),
            status=ResultStatus(data["status"]),
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            started_at=str(data.get("started_at", utc_now_iso())),
            finished_at=data.get("finished_at"),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            artifacts=artifacts,
            metrics=dict(data.get("metrics", {})),
            table_columns=list(data.get("table_columns", [])),
            table_rows=list(data.get("table_rows", [])),
            metadata=dict(data.get("metadata", {})),
            log_excerpt=str(data.get("log_excerpt", "")),
            error=data.get("error"),
        )


@dataclass
class DistortionCurveResult:
    """One-dimensional distortion curve result.

    Units:
    - table column `theta_deg`: degree.
    - height columns: mm.
    - distortion is unitless; `distortion_percent` is percent.
    - Arrays are CPU NumPy data for GUI rendering and omitted from JSON.
    """

    request_id: str
    request_snapshot: dict
    status: ResultStatus
    output_dir: Path | None = None
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None
    duration_seconds: float = 0.0
    artifacts: dict[str, Path] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    table_columns: list[str] = field(default_factory=list)
    table_rows: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    log_excerpt: str = ""
    error: str | None = None
    table_data: np.ndarray | None = field(default=None, repr=False, compare=False)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        data = dataclass_to_dict(self)
        data.pop("table_data", None)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "DistortionCurveResult":
        require_schema(data)
        require_fields(data, ["request_id", "request_snapshot", "status"])
        artifacts = {name: Path(path) for name, path in data.get("artifacts", {}).items()}
        return cls(
            request_id=str(data["request_id"]),
            request_snapshot=dict(data["request_snapshot"]),
            status=ResultStatus(data["status"]),
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            started_at=str(data.get("started_at", utc_now_iso())),
            finished_at=data.get("finished_at"),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            artifacts=artifacts,
            metrics=dict(data.get("metrics", {})),
            table_columns=list(data.get("table_columns", [])),
            table_rows=list(data.get("table_rows", [])),
            metadata=dict(data.get("metadata", {})),
            log_excerpt=str(data.get("log_excerpt", "")),
            error=data.get("error"),
        )


@dataclass
class DistortionGridResult:
    """Two-dimensional distortion grid result.

    Units:
    - sample table angles are degree.
    - grid coordinate unit is recorded in metadata as `grid_coordinate_unit`.
    - Arrays are CPU NumPy data for GUI rendering and omitted from JSON.
    """

    request_id: str
    request_snapshot: dict
    status: ResultStatus
    output_dir: Path | None = None
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None
    duration_seconds: float = 0.0
    artifacts: dict[str, Path] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    table_columns: list[str] = field(default_factory=list)
    table_rows: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    log_excerpt: str = ""
    error: str | None = None
    table_data: np.ndarray | None = field(default=None, repr=False, compare=False)
    regular_grid: np.ndarray | None = field(default=None, repr=False, compare=False)
    distorted_grid: np.ndarray | None = field(default=None, repr=False, compare=False)
    magnification_grid: np.ndarray | None = field(default=None, repr=False, compare=False)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        data = dataclass_to_dict(self)
        data.pop("table_data", None)
        data.pop("regular_grid", None)
        data.pop("distorted_grid", None)
        data.pop("magnification_grid", None)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "DistortionGridResult":
        require_schema(data)
        require_fields(data, ["request_id", "request_snapshot", "status"])
        artifacts = {name: Path(path) for name, path in data.get("artifacts", {}).items()}
        return cls(
            request_id=str(data["request_id"]),
            request_snapshot=dict(data["request_snapshot"]),
            status=ResultStatus(data["status"]),
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            started_at=str(data.get("started_at", utc_now_iso())),
            finished_at=data.get("finished_at"),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            artifacts=artifacts,
            metrics=dict(data.get("metrics", {})),
            table_columns=list(data.get("table_columns", [])),
            table_rows=list(data.get("table_rows", [])),
            metadata=dict(data.get("metadata", {})),
            log_excerpt=str(data.get("log_excerpt", "")),
            error=data.get("error"),
        )
