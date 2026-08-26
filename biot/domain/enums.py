from __future__ import annotations

from enum import Enum


class Device(str, Enum):
    """Torch device selection strategy."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class ResultStatus(str, Enum):
    """Lifecycle status for a compute result."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DistortionType(str, Enum):
    """Lens distortion definition used by lens geometry metrics."""

    ROTATING_EYE_FAR = "rotating_eye_far"
    FIXED_EYE_FAR = "fixed_eye_far"
    ROTATING_EYE_NEAR = "rotating_eye_near"
    FIXED_EYE_NEAR = "fixed_eye_near"
    HANDHELD_NEAR = "handheld_near"
