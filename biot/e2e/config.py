from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional
import random

import torch


SCHEMA_VERSION = "0.1"
SUPPORTED_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class E2EConfig:
    """Configuration for the differentiable PAL lens-eye experiment chain.

    Units:
        excel_path: required optical prescription path, usually relative to repo root.
        chart_path: optional scene/chart path; callers that render a chart must
            provide one explicitly.
        wavelength_nm: nanometers.
        field_x_deg/field_y_deg: degrees.
        pupil_radius_mm: millimeters.
        psf_size_px: square PSF side length in pixels.
        psf_pixel_pitch_mm: millimeters per PSF pixel.

    Tensor policy:
        device/dtype are parsed by as_torch_device_dtype() and should be used
        by e2e tensor creation so later phases keep dtype and device explicit.
    """

    excel_path: Path = Path("eye_image_glass.xlsx")
    chart_path: Optional[Path] = None
    wavelength_nm: float = 555.0
    device: str = "cpu"
    dtype: str = "float64"
    field_x_deg: float = 0.0
    field_y_deg: float = 0.0
    pupil_radius_mm: float = 2.0
    psf_size_px: int = 64
    psf_pixel_pitch_mm: float = 0.002
    random_seed: int = 20260605
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype {self.dtype!r}; expected one of {sorted(SUPPORTED_DTYPES)}")
        if self.psf_size_px <= 0:
            raise ValueError("psf_size_px must be positive")
        if self.psf_pixel_pitch_mm <= 0:
            raise ValueError("psf_pixel_pitch_mm must be positive")
        if self.pupil_radius_mm <= 0:
            raise ValueError("pupil_radius_mm must be positive")
        if self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config snapshot."""
        return {field.name: _serialize_value(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "E2EConfig":
        """Recreate E2EConfig from a serialized snapshot."""
        expected = data.get("schema_version", SCHEMA_VERSION)
        if expected != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version {expected!r}; expected {SCHEMA_VERSION!r}")

        values: dict[str, Any] = {}
        field_names = {field.name for field in fields(cls)}
        for name, value in data.items():
            if name not in field_names:
                continue
            if name in {"excel_path", "chart_path"}:
                values[name] = None if value is None else Path(value)
            else:
                values[name] = value
        return cls(**values)

    def existing_input_paths(self) -> dict[str, Path]:
        """Return required input paths after checking they exist.

        The keys are stable metadata names. This helper is non-differentiable
        setup code and does not participate in autograd.
        """
        paths = {"excel_path": self.excel_path}
        if self.chart_path is not None:
            paths["chart_path"] = self.chart_path
        missing = [f"{name}={path}" for name, path in paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing e2e input path(s): " + ", ".join(missing))
        return paths


def as_torch_device_dtype(device: str = "cpu", dtype: str = "float64") -> tuple[torch.device, torch.dtype]:
    """Parse e2e device/dtype strings into torch objects."""
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {sorted(SUPPORTED_DTYPES)}")
    return torch.device(device), SUPPORTED_DTYPES[dtype]


def set_random_seed(seed: int, deterministic: bool = True) -> int:
    """Set Python and torch RNG seeds for reproducible e2e CPU smoke tests."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed
