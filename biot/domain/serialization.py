from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "1.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return {"array_shape": list(value.shape), "array_dtype": str(value.dtype)}
    if is_dataclass(value):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
    return value


def parse_float(value: Any) -> float:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"inf", "infinity"}:
            return float("inf")
        if text in {"-inf", "-infinity"}:
            return float("-inf")
    return float(value)


def dataclass_to_dict(instance: Any) -> dict[str, Any]:
    data = {field.name: serialize_value(getattr(instance, field.name)) for field in fields(instance)}
    data["schema_version"] = getattr(instance, "schema_version", SCHEMA_VERSION)
    return data


def require_schema(data: dict[str, Any], expected: str = SCHEMA_VERSION) -> None:
    actual = data.get("schema_version", expected)
    if actual != expected:
        raise ValueError(f"Unsupported schema_version: {actual!r}; expected {expected!r}")


def require_fields(data: dict[str, Any], field_names: list[str]) -> None:
    missing = [name for name in field_names if name not in data]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
