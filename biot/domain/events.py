from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event

from .serialization import SCHEMA_VERSION, dataclass_to_dict, require_fields, require_schema, utc_now_iso


@dataclass
class ProgressEvent:
    """Progress event for long-running services.

    Units:
    - current/total are unitless work item counts.
    - created_at is UTC ISO8601.
    """

    phase: str
    current: int
    total: int
    message: str = ""
    request_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProgressEvent":
        require_schema(data)
        require_fields(data, ["phase", "current", "total"])
        return cls(
            phase=str(data["phase"]),
            current=int(data["current"]),
            total=int(data["total"]),
            message=str(data.get("message", "")),
            request_id=data.get("request_id"),
            created_at=str(data.get("created_at", utc_now_iso())),
        )


@dataclass
class LogEvent:
    """Structured log event emitted by services."""

    level: str
    message: str
    request_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LogEvent":
        require_schema(data)
        require_fields(data, ["level", "message"])
        return cls(
            level=str(data["level"]),
            message=str(data["message"]),
            request_id=data.get("request_id"),
            created_at=str(data.get("created_at", utc_now_iso())),
        )


class CancelToken:
    """Thread-safe cancellation token used by services."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

