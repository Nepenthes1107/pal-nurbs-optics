"""PAL-NURBS V3 优化链：真实追迹 + 路由物理指标 + Adam 小步更新。"""
from __future__ import annotations

import copy
import contextlib
import csv
import gc
import hashlib
import io
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from averfang import load_sag_xlsx
from lens_metrics_core import build_legacy_adapter, load_lens, resolve_device
from optics import GridSag

from .pal_case_layout import (
    FUNCTIONAL_GROUPS,
    GROUP_TO_ZONE,
    PERIPHERAL_BAND_COUNTS,
    PERIPHERAL_GROUPS,
    TOTAL_TRAINING_CASES,
    TRAINING_GROUP_COUNTS,
    _sha256_file,
    classify_partition_point,
    generate_dense_candidate_fields,
    qualified_source_key,
    select_training_cases,
    trace_candidate_fields,
    write_preoptimization_artifacts,
)
from .opd_zernike import fit_low_order_opd_zernike_torch
from .psf_database import compute_physical_fft_pixel_pitch_mm
from .psf_fft import effective_biot_pupil_sample_count, torch_fft_psf_from_phase
from .regional_nurbs import FixedWeightNURBSPerturbation, audit_exact_refinement
from .system import (
    FitSpec,
    FittedE2ESystem,
    LocalCoordinateBreakSurface,
    build_fitted_e2e_system,
    make_aimed_pupil_rays,
    make_aimed_reference_ray,
    trace_system_batch_to_image_with_phase,
    trace_system_to_image_with_phase,
    _snell,
)


METHOD_NAME = "pal_109case_stratified_corridor_csf_z4_smooth_v3zones_no_clearance_filter"

DEFAULT_GROUP_WEIGHTS = {
    "far": 0.22,
    "far_robustness": 0.02,
    "corridor_upper": 0.07,
    "corridor_middle": 0.10,
    "corridor_lower": 0.11,
    "near": 0.18,
    "near_robustness": 0.02,
    "near_edge_astig": 0.04,
    "peripheral_left": 0.12,
    "peripheral_right": 0.12,
}


@dataclass(frozen=True)
class MinimalConfig:
    # PAL-NURBS 后续实验统一使用已验证在 D500、(-40°, -40°) 可完成追迹的 grad3 系统。
    # 历史 r1/r5 配置会在各自 run/config.json 中保留原始输入，不被此默认值改写。
    excel: str = "eye_image_glass_grad3.xlsx"
    support_json: str = "inputs/pal/psf_supports.json"
    zones_json: str = "inputs/pal/zones.json"
    output: str = "results/optimization/run_001"
    device: str = "cuda"
    wavelength_nm: float = 555.0
    requested_np: int = 256
    fft_size_px: int = 512
    case_batch_size: int = 8
    # PAL 训练固定采用 BIOT 已验证的非 legacy 参考球合同。
    legacy_pupil_phase: bool = False
    phase_reference: str = "biot_reference_sphere"
    remove_tilt: bool = False
    kernel_size_px: int = 130
    pupil_radius_mm: float | None = None
    learning_rate: float = 2.0e-3
    minimum_learning_rate: float = 1.0e-6
    max_steps_7: int = 10
    max_steps_11: int = 10
    max_steps_19: int = 30
    early_stopping_patience: int = 7
    relative_improvement_threshold: float = 1.0e-3
    max_extra_terminal_stage_steps: int = 30
    max_backtracks: int = 8
    step_sag_limit_mm: float = 2.0e-3
    far_tolerance_D: float = 0.15
    add_tolerance_D: float = 0.25
    lower_edge_power_tolerance_D: float = 0.50
    lower_edge_astig_tolerance_D: float = 0.80
    minimum_valid_fraction_ratio: float = 0.5
    seed: int = 42
    far_object_distance_mm: float = float("inf")
    intermediate_object_distance_mm: float = 1000.0
    near_object_distance_mm: float = 500.0
    candidate_field_x_min_deg: float = -45.0
    candidate_field_x_max_deg: float = 45.0
    candidate_field_y_min_deg: float = -60.0
    candidate_field_y_max_deg: float = 55.0
    candidate_field_step_deg: float = 1.0
    # Deprecated square-grid aliases.  They remain explicit identity fields
    # for old callers but are not used unless supplied.
    candidate_field_min_deg: float | None = None
    candidate_field_max_deg: float | None = None
    # Retained for backward-compatible config loading; candidate eligibility
    # no longer uses zone/aperture clearance safety filters.
    zone_boundary_safety_mm: float = 1.5
    corridor_zone_boundary_safety_mm: float = 1.0
    aperture_edge_safety_mm: float = 1.5
    group_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_GROUP_WEIGHTS)
    )
    near_edge_astig_A_weight: float = 0.10
    smooth_lambda: float = 0.05
    candidate_trace_import: str | None = None
    forward_qualification_import: str | None = None
    final_phase_qualification_import: str | None = None
    baseline_state_import: str | None = None
    parent_run: str | None = None
    start_stage: int | None = None

    def __post_init__(self) -> None:
        if bool(self.legacy_pupil_phase):
            raise ValueError("legacy_pupil_phase=True is not supported by the PAL contract")
        if self.phase_reference != "biot_reference_sphere":
            raise ValueError("PAL requires phase_reference='biot_reference_sphere'")
        if bool(self.remove_tilt):
            raise ValueError("PAL requires remove_tilt=False with the BIOT reference sphere")
        if int(self.requested_np) <= 0:
            raise ValueError("requested_np must be positive")
        weights = {str(name): float(value) for name, value in self.group_weights.items()}
        if set(weights) != set(TRAINING_GROUP_COUNTS):
            raise ValueError("group_weights must define exactly the ten training groups")
        if any(not math.isfinite(value) or value <= 0.0 for value in weights.values()):
            raise ValueError("group_weights must be finite and positive")
        if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("group_weights must sum to 1")
        if not 0.0 <= float(self.near_edge_astig_A_weight) <= 1.0:
            raise ValueError("near_edge_astig_A_weight must be in [0,1]")
        if not math.isfinite(float(self.smooth_lambda)) or float(self.smooth_lambda) < 0.0:
            raise ValueError("smooth_lambda must be finite and non-negative")
        if (self.candidate_field_min_deg is None) != (self.candidate_field_max_deg is None):
            raise ValueError("deprecated candidate field min/max aliases must be supplied together")
        if int(self.fft_size_px) <= 0:
            raise ValueError("fft_size_px must be positive")
        if int(self.case_batch_size) <= 0:
            raise ValueError("case_batch_size must be a positive integer")
        for name in (
            "max_steps_7",
            "max_steps_11",
            "max_steps_19",
            "max_extra_terminal_stage_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (self.parent_run is None) != (self.start_stage is None):
            raise ValueError("parent_run and start_stage must be supplied together")
        if self.start_stage is not None:
            if isinstance(self.start_stage, bool) or int(self.start_stage) not in STAGE_LADDER:
                raise ValueError("start_stage must be one of 7, 11, or 19")
            explicit_imports = {
                "candidate_trace_import": self.candidate_trace_import,
                "forward_qualification_import": self.forward_qualification_import,
                "final_phase_qualification_import": self.final_phase_qualification_import,
                "baseline_state_import": self.baseline_state_import,
            }
            mixed = [name for name, value in explicit_imports.items() if value is not None]
            if mixed:
                raise ValueError(
                    "parent_run cannot be combined with explicit evidence imports: "
                    + ", ".join(mixed)
                )
            budgets = {
                7: int(self.max_steps_7),
                11: int(self.max_steps_11),
                19: int(self.max_steps_19),
            }
            earlier = [control for control in STAGE_LADDER if control < int(self.start_stage)]
            if any(budgets[control] != 0 for control in earlier):
                raise ValueError("training budgets before start_stage must be zero")
            if budgets[int(self.start_stage)] <= 0:
                raise ValueError("start_stage training budget must be positive")
        if (
            isinstance(self.early_stopping_patience, bool)
            or not isinstance(self.early_stopping_patience, int)
            or self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be a positive integer")
        if (
            not math.isfinite(float(self.relative_improvement_threshold))
            or float(self.relative_improvement_threshold) <= 0.0
        ):
            raise ValueError("relative_improvement_threshold must be finite and positive")


RUN_IDENTITY_SCHEMA_VERSION = 9
PARENT_RUN_IDENTITY_SCHEMA_VERSIONS = (8, 9)
CASE_LAYOUT_STATE_SCHEMA_VERSION = 10
BASELINE_STATE_SCHEMA_VERSION = 6
BASELINE_PROGRESS_SCHEMA_VERSION = 6
STAGE_RESUME_SCHEMA_VERSION = 3
RUN_STATE_SCHEMA_VERSION = 1
STAGE_LADDER = (7, 11, 19)
FORWARD_POOL_MULTIPLIER = 4
FORWARD_POOL_GROUP_COUNTS = {
    **{
        group: TRAINING_GROUP_COUNTS[group] * FORWARD_POOL_MULTIPLIER
        for group in FUNCTIONAL_GROUPS
    },
    # Keep the already audited 52-case pool per side.  This is the smallest
    # pool that contains the complete 16/16/20 band strata while supporting
    # the final 5/5/6 selection without increasing qualification compute.
    "peripheral_left": 52,
    "peripheral_right": 52,
}
FORWARD_POOL_PERIPHERAL_BAND_COUNTS = {
    "upper": 16, "middle": 16, "lower": 20,
}


class MinimumTrainingBudgetError(RuntimeError):
    """Raised when the learning-rate floor prevents a mandatory stage budget."""


def _training_stage_specs(
    config: MinimalConfig,
) -> tuple[tuple[int, int, int, bool], ...]:
    """Return ``(control, minimum, maximum, is_terminal)`` for the NURBS ladder."""
    minimum_by_control = {
        7: int(config.max_steps_7),
        11: int(config.max_steps_11),
        19: int(config.max_steps_19),
    }
    active = [control for control in STAGE_LADDER if minimum_by_control[control] > 0]
    if not active:
        raise ValueError("--steps must contain at least one positive training budget")
    terminal_control_count = active[-1]
    return tuple(
        (
            control_count,
            minimum_by_control[control_count],
            minimum_by_control[control_count]
            + (
                int(config.max_extra_terminal_stage_steps)
                if control_count == terminal_control_count
                else 0
            ),
            control_count == terminal_control_count,
        )
        for control_count in STAGE_LADDER
    )


def _stage_boundary_stop_reason(
    *,
    control_count: int,
    is_terminal_stage: bool,
    completed_step: int,
    minimum_steps: int,
    maximum_steps: int,
    learning_rate: float,
    minimum_learning_rate: float,
    no_improvement_attempts: int,
    early_stopping_patience: int,
) -> str | None:
    """Return the deterministic stop decision at an atomically saved attempt boundary."""
    if completed_step < minimum_steps:
        if learning_rate < minimum_learning_rate:
            return "minimum_not_reached"
        return None
    if not is_terminal_stage:
        if completed_step != minimum_steps or maximum_steps != minimum_steps:
            raise ValueError("fixed stages must finish exactly at their minimum budget")
        return "minimum_completed"
    if no_improvement_attempts >= early_stopping_patience:
        return "early_stopping"
    if learning_rate < minimum_learning_rate:
        return "learning_rate_floor"
    if completed_step >= maximum_steps:
        return "max_extra_reached"
    return None


def _early_stopping_observation(
    *,
    best_before: float,
    candidate: float,
    accepted: bool,
    threshold: float,
    no_improvement_attempts: int,
) -> tuple[bool, float, bool, int]:
    """Classify one 19x19 attempt against the update-before-attempt best."""
    if not math.isfinite(best_before) or best_before <= 0.0:
        raise RuntimeError("19x19 best objective must be finite and positive")
    best_refreshed = bool(accepted and math.isfinite(candidate) and candidate < best_before)
    relative_improvement = (
        (best_before - candidate) / abs(best_before) if best_refreshed else 0.0
    )
    significant = bool(best_refreshed and relative_improvement > threshold)
    next_counter = 0 if significant else no_improvement_attempts + 1
    return best_refreshed, relative_improvement, significant, next_counter


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def _replace_atomic(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    temporary = _temporary_sibling(destination)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_atomic(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_save_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    temporary = _temporary_sibling(destination)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        # PyTorch 2.0.1 on Windows cannot reliably pass non-ASCII paths to its
        # C++ zip writer.  A Python binary handle preserves the exact path and
        # still uses the same torch serialization format.
        with temporary.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_atomic(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _import_complete_pool_progress(
    *, source_path: str | Path | None, destination_path: Path,
    pool_identity_sha256: str, progress_name: str,
) -> None:
    """Import completed, pool-identity-bound qualification evidence atomically."""
    if source_path is None or destination_path.exists():
        return
    source = Path(source_path)
    payload = _read_json(source)
    if payload.get("schema_version") != 1:
        raise ValueError(f"{progress_name} import has an unsupported schema")
    if payload.get("status") != "complete":
        raise ValueError(f"{progress_name} import must be complete")
    if payload.get("pool_identity_sha256") != pool_identity_sha256:
        raise ValueError(f"{progress_name} import pool identity mismatch")
    temporary = _temporary_sibling(destination_path)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(source, temporary)
        _replace_atomic(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)


def _implementation_closure_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    relative = [
        "averfang.py",
        "basics.py",
        "lens_metrics_core.py",
        "multi_rays.py",
        "optics.py",
        "run_pal_nurbs.py",
        "biot/domain/__init__.py",
        "biot/domain/enums.py",
        "biot/domain/events.py",
        "biot/domain/requests.py",
        "biot/domain/results.py",
        "biot/domain/serialization.py",
        "biot/infra/field_mapping.py",
        "biot/services/single_field_service.py",
        "biot/e2e/bspline.py",
        "biot/e2e/pal_case_layout.py",
        "biot/e2e/pal_case_layout_plotter.py",
        "biot/e2e/pal_nurbs.py",
        "biot/e2e/psf_database.py",
        "biot/e2e/psf_fft.py",
        "biot/e2e/rays.py",
        "biot/e2e/regional_nurbs.py",
        "biot/e2e/surfaces.py",
        "biot/e2e/system.py",
        "biot/e2e/validation.py",
    ]
    paths = [root / name for name in relative]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("implementation closure is incomplete: " + ", ".join(missing))
    return paths


def _identity_input_paths(
    config: MinimalConfig, *, parent_context: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    paths = {
        "excel": Path(config.excel),
        "support_json": Path(config.support_json),
        "zones_json": Path(config.zones_json),
    }
    if config.candidate_trace_import is not None:
        paths["candidate_trace_import"] = Path(config.candidate_trace_import)
    if config.forward_qualification_import is not None:
        paths["forward_qualification_import"] = Path(config.forward_qualification_import)
    if config.final_phase_qualification_import is not None:
        paths["final_phase_qualification_import"] = Path(
            config.final_phase_qualification_import
        )
    if config.baseline_state_import is not None:
        paths["baseline_state_import"] = Path(config.baseline_state_import)
    if parent_context is not None:
        for name, path in dict(parent_context["artifact_paths"]).items():
            paths[f"parent_{name}"] = Path(path)
    lens = load_lens(
        Path(config.excel), device=resolve_device("cpu"), wavelength_nm=config.wavelength_nm,
    )
    for index, surface in enumerate(lens.surfaces):
        sag_path = getattr(surface, "sag_file_path", None)
        if sag_path:
            paths[f"surface_{index}_sag"] = Path(sag_path)
    missing = [f"{name}={path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("run identity input is missing: " + ", ".join(missing))
    return paths


def _build_run_identity(config: MinimalConfig) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    parent_context = _validate_parent_run_source(config, device="cpu")
    input_paths = (
        _identity_input_paths(config)
        if parent_context is None else
        _identity_input_paths(config, parent_context=parent_context)
    )
    inputs = {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in sorted(input_paths.items())
    }
    implementation: dict[str, str] = {}
    for path in _implementation_closure_paths():
        resolved = path.resolve()
        try:
            name = resolved.relative_to(root).as_posix()
        except ValueError:
            name = resolved.as_posix()
        implementation[name] = _sha256_file(resolved)
    body: dict[str, Any] = {
        "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "method": METHOD_NAME,
        "config": asdict(config),
        "config_sha256": _canonical_json_sha256(asdict(config)),
        "inputs": inputs,
        "implementation_closure": implementation,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_cuda": None if torch.version.cuda is None else str(torch.version.cuda),
            "numpy": str(np.__version__),
            "platform": sys.platform,
        },
    }
    if parent_context is not None:
        body["parent_lineage"] = dict(parent_context["identity_metadata"])
    return {**body, "identity_sha256": _canonical_json_sha256(body)}


def _validate_identity_payload(payload: Mapping[str, Any]) -> str:
    saved = dict(payload)
    claimed = str(saved.pop("identity_sha256", ""))
    actual = _canonical_json_sha256(saved)
    if not claimed or claimed != actual:
        raise ValueError("run_identity.json is malformed or has been modified")
    if int(saved.get("schema_version", -1)) != RUN_IDENTITY_SCHEMA_VERSION:
        raise ValueError("unsupported run identity schema")
    if saved.get("method") != METHOD_NAME:
        raise ValueError("unsupported PAL method identity")
    return claimed


def _open_run_directory(config: MinimalConfig, *, resume: bool) -> tuple[Path, dict[str, Any]]:
    output = Path(config.output)
    if resume:
        if not output.is_dir():
            raise FileNotFoundError(f"resume requires an existing run directory: {output}")
    else:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing run: {output}")

    current = _build_run_identity(config)
    if not resume:
        output.mkdir(parents=True)
    identity_path = output / "run_identity.json"
    if resume:
        if not identity_path.is_file():
            raise ValueError(f"resume requires run_identity.json: {identity_path}")
        saved = _read_json(identity_path)
        saved_hash = _validate_identity_payload(saved)
        if saved_hash != current["identity_sha256"] or saved != current:
            raise ValueError(
                "resume identity mismatch: config, input hashes, implementation closure, "
                "or runtime changed"
            )
    else:
        _write_json_atomic(identity_path, current)
        _write_json_atomic(output / "config.json", asdict(config))
    return output, current


@dataclass(frozen=True)
class PALPowerConfig:
    semi_diameter_mm: float
    refractive_index: float
    front_radius_mm: float
    center_thickness_mm: float
    crib_diameter_mm: float = 80.0


@dataclass(frozen=True)
class FieldResult:
    kernel: torch.Tensor
    valid_fraction: torch.Tensor
    pixel_pitch_mm: float
    edge_fraction: torch.Tensor
    valid_mask: torch.Tensor | None = None
    raw_psf: torch.Tensor | None = None
    raw_pixel_pitch_mm: float | None = None
    zernike_coefficients_mm: torch.Tensor | None = None
    z4_defocus_mm2: torch.Tensor | None = None


@dataclass(frozen=True)
class BatchFieldResult:
    """批量前向结果；首维是 case，后两维是 raw 物理 FFT PSF。"""

    kernels: torch.Tensor
    valid_fraction: torch.Tensor
    pixel_pitch_mm: torch.Tensor
    edge_fraction: torch.Tensor
    valid_mask: torch.Tensor | None = None
    zernike_coefficients_mm: torch.Tensor | None = None
    z4_defocus_mm2: torch.Tensor | None = None


@dataclass(frozen=True)
class RawPSFBatchResult:
    """评价用原生 FFT PSF 批量结果；首维严格对应输入 case 顺序。"""

    psf: torch.Tensor
    valid_fraction: torch.Tensor
    pixel_pitch_mm: torch.Tensor


def _elapsed_seconds(output: Path) -> float:
    path = output / "run_state.json"
    if not path.is_file():
        return 0.0
    payload = _read_json(path)
    value = float(payload.get("elapsed_seconds", 0.0))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("run_state.json contains an invalid elapsed_seconds")
    return value


def _write_run_state(
    output: Path,
    *,
    identity_sha256: str,
    status: str,
    phase: str,
    elapsed_seconds: float,
    **details: Any,
) -> None:
    _write_json_atomic(
        output / "run_state.json",
        {
            "schema_version": RUN_STATE_SCHEMA_VERSION,
            "identity_sha256": identity_sha256,
            "status": status,
            "phase": phase,
            "elapsed_seconds": float(elapsed_seconds),
            "updated_unix_time": time.time(),
            **details,
        },
    )


def _load_identity_bound_torch(
    path: str | Path,
    *,
    identity_sha256: str,
    schema_version: int,
    map_location: torch.device | str,
) -> dict[str, Any]:
    source = Path(path)
    with source.open("rb") as handle:
        payload = torch.load(handle, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"state payload must be a mapping: {path}")
    if int(payload.get("schema_version", -1)) != int(schema_version):
        raise ValueError(f"state schema mismatch: {path}")
    if str(payload.get("identity_sha256", "")) != str(identity_sha256):
        raise ValueError(f"state identity mismatch: {path}")
    return payload


def _capture_rng_state() -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(payload: Mapping[str, Any]) -> None:
    cpu = payload.get("torch_cpu")
    if not torch.is_tensor(cpu):
        raise ValueError("resume state is missing the CPU RNG state")
    torch.set_rng_state(cpu.detach().cpu())
    cuda = payload.get("torch_cuda", [])
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("resume state contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda])


def _training_case_keys(
    model: MinimalOpticalModel, cases: Sequence[Mapping[str, Any]],
) -> set[tuple[float, float, float]]:
    return {
        model._key(
            float(case["distance_mm"]),
            float(case["field_x_deg"]),
            float(case["field_y_deg"]),
        )
        for case in cases
    }


def _retain_training_cache(
    model: MinimalOpticalModel,
    cases: Sequence[Mapping[str, Any]],
    *,
    extra_cases: Sequence[Mapping[str, Any]] = (),
) -> dict[str, int]:
    required_cases = [
        case
        for case in (*cases, *extra_cases)
        if str(case.get("training_group", "")) not in PERIPHERAL_GROUPS
    ]
    keep = _training_case_keys(model, required_cases)
    materialized = 0
    system_and_rays = getattr(model, "_system_and_rays", None)
    if callable(system_and_rays):
        for case in required_cases:
            key = model._key(
                float(case["distance_mm"]),
                float(case["field_x_deg"]),
                float(case["field_y_deg"]),
            )
            if key in model._cache:
                continue
            with torch.no_grad():
                system_and_rays(*key)
            materialized += 1
            _release_inactive_case_cuda_cache(model.device)
    before = len(model._cache)
    for key in list(model._cache):
        if key in keep:
            continue
        system, _ = model._cache.pop(key)
        system.release_biot_lens()
    gc.collect()
    if model.device.type == "cuda":
        torch.cuda.empty_cache()
    templates = getattr(model, "_templates", None)
    released_templates = 0
    if isinstance(templates, dict):
        released_templates = len(templates)
        for template, _ in templates.values():
            template.release_biot_lens()
        templates.clear()
        model._cache_frozen = True
        gc.collect()
        _release_inactive_case_cuda_cache(model.device)
    audit = {"before": before, "retained": len(model._cache), "removed": before - len(model._cache)}
    if callable(system_and_rays):
        audit.update(
            {
                "materialized": materialized,
                "released_templates": released_templates,
            }
        )
    return audit


def _normalize_psf(psf: torch.Tensor) -> torch.Tensor:
    if psf.ndim != 2 or not bool(torch.isfinite(psf).all()) or bool((psf < 0).any()):
        raise ValueError("physical PSF must be a finite non-negative 2-D tensor")
    energy = psf.sum()
    if not bool(torch.isfinite(energy)) or not bool(energy > 0):
        raise ValueError("physical PSF must have positive finite energy")
    return psf / energy


def _normalize_psf_batch(psf: torch.Tensor) -> torch.Tensor:
    if psf.ndim < 3 or not bool(torch.isfinite(psf).all()) or bool((psf < 0).any()):
        raise ValueError("physical PSF batch must be finite, non-negative and have shape [B,H,W]")
    energy = psf.sum(dim=(-2, -1), keepdim=True)
    if not bool(torch.isfinite(energy).all()) or bool((energy <= 0).any()):
        raise ValueError("each physical PSF in the batch must have positive finite energy")
    return psf / energy


def _release_inactive_case_cuda_cache(device: torch.device | str) -> None:
    """Return completed per-case graph blocks to CUDA/WDDM after tensor deletion.

    The production GRIN3 backward uses only a few MiB after each case has been
    backpropagated, but PyTorch 1.11 on Windows can retain more than 16 GiB of
    inactive allocator blocks.  Under WDDM those blocks consume host commit and
    can terminate the process without a Python exception.  ``empty_cache`` does
    not change live tensors, accumulated gradients, RNG, sampling, or physics;
    it only releases allocator blocks that are already inactive.
    """

    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.empty_cache()


def crop_resize_fft_psf(
    raw_psf: torch.Tensor, *, pixel_pitch_mm: float, size_reference_mm: float, output_size_px: int
) -> torch.Tensor:
    raw = _normalize_psf(raw_psf)
    crop_size = float(size_reference_mm) / float(pixel_pitch_mm)
    half = int(round(crop_size / 2.0))
    center = (int(raw.shape[0]) + 1) / 2.0 - 1.0
    start = int(round(center - half))
    stop = start + int(round(crop_size)) + 1
    if start < 0 or stop > int(raw.shape[0]):
        raise ValueError("declared PSF support does not fit the raw FFT window")
    crop = _normalize_psf(raw[start:stop, start:stop])
    resized = F.interpolate(
        crop[None, None], size=(int(output_size_px), int(output_size_px)),
        mode="bicubic", align_corners=False, antialias=False,
    )[0, 0].clamp_min(0.0)
    return _normalize_psf(resized)


def psf_second_moment_mm2(psf: torch.Tensor, *, pixel_pitch_mm: float) -> torch.Tensor:
    normalized = _normalize_psf(psf)
    height, width = normalized.shape
    y = (torch.arange(height, device=psf.device, dtype=psf.dtype) - 0.5 * (height - 1)) * pixel_pitch_mm
    x = (torch.arange(width, device=psf.device, dtype=psf.dtype) - 0.5 * (width - 1)) * pixel_pitch_mm
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    cx = (normalized * xx).sum()
    cy = (normalized * yy).sum()
    return (normalized * ((xx - cx).square() + (yy - cy).square())).sum()


def psf_second_moment_mm2_batch(
    psf: torch.Tensor, *, pixel_pitch_mm: torch.Tensor | float,
) -> torch.Tensor:
    """Return centroid-relative M2 for every kernel in a real tensor batch."""
    normalized = _normalize_psf_batch(psf)
    height, width = normalized.shape[-2:]
    dtype, device = normalized.dtype, normalized.device
    y = (torch.arange(height, device=device, dtype=dtype) - 0.5 * (height - 1))
    x = (torch.arange(width, device=device, dtype=dtype) - 0.5 * (width - 1))
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    pitch = torch.as_tensor(pixel_pitch_mm, device=device, dtype=dtype).reshape(-1)
    if pitch.numel() == 1:
        pitch = pitch.expand(normalized.shape[0])
    if pitch.shape[0] != normalized.shape[0]:
        raise ValueError("pixel_pitch_mm batch length must match PSF batch")
    pitch = pitch.reshape(-1, 1, 1)
    xx_mm, yy_mm = xx * pitch, yy * pitch
    cx = (normalized * xx_mm).sum(dim=(-2, -1), keepdim=True)
    cy = (normalized * yy_mm).sum(dim=(-2, -1), keepdim=True)
    return (normalized * ((xx_mm - cx).square() + (yy_mm - cy).square())).sum(dim=(-2, -1))


def csf_weighted_mtf_loss_batch(
    psf_kernel: torch.Tensor,
    *,
    pixel_pitch_mm: torch.Tensor,
    maximum_frequency_lpmm: float = 30.0,
    frequency_sample_count: int = 60,
    angular_sample_count: int = 72,
) -> torch.Tensor:
    """Return differentiable Mannos-Sakrison CSF-weighted MTF losses."""
    if psf_kernel.ndim != 3:
        raise ValueError("batched CSF-MTF requires PSF shape [B,H,W]")
    if pixel_pitch_mm.shape != (int(psf_kernel.shape[0]),):
        raise ValueError("pixel_pitch_mm must have one value per PSF case")
    if not bool(torch.isfinite(psf_kernel).all()) or bool((psf_kernel < 0.0).any()):
        raise ValueError("CSF-MTF PSF must be finite and non-negative")
    if not bool(torch.isfinite(pixel_pitch_mm).all()) or bool((pixel_pitch_mm <= 0.0).any()):
        raise ValueError("CSF-MTF pixel pitch must be finite and positive")
    h, w = int(psf_kernel.shape[-2]), int(psf_kernel.shape[-1])
    otf = torch.fft.fftshift(
        torch.fft.fft2(psf_kernel, dim=(-2, -1)), dim=(-2, -1)
    )
    magnitude = torch.abs(otf)
    dc = magnitude[:, h // 2, w // 2]
    if not bool(torch.isfinite(dc).all()) or bool((dc <= 0.0).any()):
        raise ValueError("CSF-MTF OTF DC must be finite and positive")
    mtf = magnitude / dc[:, None, None]
    targets = torch.linspace(
        0.0,
        float(maximum_frequency_lpmm),
        int(frequency_sample_count),
        device=psf_kernel.device,
        dtype=psf_kernel.dtype,
    )
    if int(targets.numel()) < 2:
        raise ValueError("frequency_sample_count must be at least two")
    if int(angular_sample_count) < 8:
        raise ValueError("angular_sample_count must be at least eight")
    angles = torch.arange(
        int(angular_sample_count), device=psf_kernel.device, dtype=psf_kernel.dtype
    ) * (2.0 * math.pi / int(angular_sample_count))
    radial_profiles: list[torch.Tensor] = []
    for case_index in range(int(psf_kernel.shape[0])):
        pitch = float(pixel_pitch_mm[case_index].detach().cpu())
        nyquist = 0.5 / pitch
        if float(maximum_frequency_lpmm) > nyquist + 1.0e-12:
            raise ValueError(
                "CSF-MTF maximum frequency exceeds the PSF sampling Nyquist limit"
            )
        # Sample the 2-D MTF on concentric circles with bilinear interpolation.
        # This avoids empty hard radial bins while keeping the complete loss in
        # Torch/autograd. fftshift DC lies at (w//2, h//2), including even sizes.
        frequency_x = targets[:, None] * torch.cos(angles)[None, :]
        frequency_y = targets[:, None] * torch.sin(angles)[None, :]
        index_x = frequency_x * (w * pitch) + float(w // 2)
        index_y = frequency_y * (h * pitch) + float(h // 2)
        grid_x = 2.0 * index_x / float(w - 1) - 1.0
        grid_y = 2.0 * index_y / float(h - 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        samples = F.grid_sample(
            mtf[case_index][None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0]
        radial_profiles.append(samples.mean(dim=-1))
    radial = torch.stack(radial_profiles, dim=0)
    frequency_cpd = targets * 0.291
    csf = (
        2.6
        * (0.0192 + 0.114 * frequency_cpd)
        * torch.exp(-torch.pow(0.114 * frequency_cpd, 1.1))
    )
    csf = csf / csf.sum()
    loss = 1.0 - (radial * csf.unsqueeze(0)).sum(dim=-1)
    if not bool(torch.isfinite(loss).all()) or bool((loss < -1.0e-12).any()):
        raise ValueError("CSF-weighted MTF loss is invalid")
    return loss.clamp_min(0.0)


def laplacian_regularizer(
    module: FixedWeightNURBSPerturbation,
) -> torch.Tensor:
    """Return the normalized-control second-difference penalty."""
    q = module.inner_q
    if q.ndim != 2 or min(int(q.shape[0]), int(q.shape[1])) < 3:
        raise ValueError("NURBS inner control grid is too small for a Laplacian penalty")
    lap_y = q[2:, :] - 2.0 * q[1:-1, :] + q[:-2, :]
    lap_x = q[:, 2:] - 2.0 * q[:, 1:-1] + q[:, :-2]
    value = lap_y.square().mean() + lap_x.square().mean()
    if not bool(torch.isfinite(value)):
        raise ValueError("NURBS Laplacian regularizer is non-finite")
    return value


def _edge_fraction_batch(psf: torch.Tensor, edge_px: int = 5) -> torch.Tensor:
    normalized = _normalize_psf_batch(psf)
    edge = min(int(edge_px), int(normalized.shape[-2]) // 2, int(normalized.shape[-1]) // 2)
    mask = torch.zeros_like(normalized, dtype=torch.bool)
    mask[..., :edge, :] = True
    mask[..., -edge:, :] = True
    mask[..., :, :edge] = True
    mask[..., :, -edge:] = True
    return torch.where(mask, normalized, torch.zeros_like(normalized)).sum(dim=(-2, -1))


def _edge_fraction(psf: torch.Tensor, edge_px: int = 5) -> torch.Tensor:
    psf = _normalize_psf(psf)
    edge = min(int(edge_px), int(psf.shape[0]) // 2)
    mask = torch.zeros_like(psf, dtype=torch.bool)
    mask[:edge] = mask[-edge:] = True
    mask[:, :edge] = mask[:, -edge:] = True
    return torch.where(mask, psf, torch.zeros_like(psf)).sum()


class MinimalOpticalModel:
    """复用已验证 system.py 追迹、reference-sphere OPD 与 FFT PSF。"""

    def __init__(self, config: MinimalConfig, perturbation: torch.nn.Module) -> None:
        self.config = config
        self.perturbation = perturbation
        self.device = torch.device(config.device)
        self.sample_count = effective_biot_pupil_sample_count(config.requested_np)
        self.fit_spec = FitSpec(control_shape=(41, 41), sample_shape=(81, 81), degree=3)
        support_payload = json.loads(Path(config.support_json).read_text(encoding="utf-8-sig"))
        self.size_reference_mm = {
            float(str(name)[1:] if str(name).startswith("D") else str(name)): float(value)
            for name, value in support_payload["size_reference_mm"].items()
        }
        if not self.size_reference_mm or any(not math.isfinite(v) or v <= 0 for v in self.size_reference_mm.values()):
            raise ValueError("support_json contains invalid physical PSF supports")
        self._cache: dict[tuple[float, float, float], tuple[FittedE2ESystem, object]] = {}
        self._templates: dict[float, tuple[FittedE2ESystem, object]] = {}
        self._cache_frozen = False
        self._pal_sag: torch.Tensor | None = None
        self._pal_power_config: PALPowerConfig | None = None
        self._pal_zones: Mapping[str, torch.Tensor] | None = None

    def close(self) -> None:
        """Release all systems and heavyweight BIOT lenses owned by this model."""
        for system, _ in self._cache.values():
            system.release_biot_lens()
        for template, _ in self._templates.values():
            template.release_biot_lens()
        self._cache.clear()
        self._templates.clear()
        self._pal_sag = None
        self._pal_power_config = None
        self._pal_zones = None

    def set_prescription_context(
        self,
        sag: torch.Tensor,
        power_config: PALPowerConfig,
        zones: Mapping[str, torch.Tensor],
    ) -> None:
        """Attach the differentiable PAL M/A context used by peripheral loss."""
        self._pal_sag = sag
        self._pal_power_config = power_config
        self._pal_zones = zones

    def astig_A_by_zone(self) -> dict[str, torch.Tensor]:
        if self._pal_sag is None or self._pal_power_config is None or self._pal_zones is None:
            raise RuntimeError("PAL prescription context is required for astigmatism loss")
        coord = torch.linspace(
            -self._pal_power_config.semi_diameter_mm,
            self._pal_power_config.semi_diameter_mm,
            int(self._pal_sag.shape[0]),
            device=self._pal_sag.device,
            dtype=self._pal_sag.dtype,
        )
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        sag = self._pal_sag + self.perturbation.delta_raw(xx, yy)
        maps = torch_averfang_maps(sag, self._pal_power_config)
        result: dict[str, torch.Tensor] = {}
        zone_masks = {
            "astig_left": "peripheral_astig_left",
            "astig_right": "peripheral_astig_right",
            "near": "near",
        }
        for zone, mask_name in zone_masks.items():
            mask = self._pal_zones[mask_name] & maps["valid"]
            if not bool(mask.any()):
                raise ValueError(f"M/A mask has no valid samples for {zone}")
            value = maps["A_D"][mask].mean()
            if not bool(torch.isfinite(value)) or bool(value <= 0.0):
                raise ValueError(f"M/A astigmatism A is invalid for {zone}")
            result[zone] = value
        return result

    @staticmethod
    def _key(distance: float, x: float, y: float) -> tuple[float, float, float]:
        return round(float(distance), 6), round(float(x), 6), round(float(y), 6)

    @staticmethod
    def _set_field(lens: object, x: float, y: float) -> None:
        for surface in lens.surfaces:
            if getattr(surface, "type", "") == "CB":
                for name, value in (("tilt_x", -float(y)), ("tilt_y", -float(x))):
                    old = getattr(surface, name)
                    setattr(surface, name, torch.as_tensor(value, device=old.device, dtype=old.dtype))

    @staticmethod
    def _clone(template: FittedE2ESystem, lens: object, x: float, y: float) -> FittedE2ESystem:
        surfaces = [
            LocalCoordinateBreakSurface(
                semi_diameter_mm=s.semi_diameter_mm, n_after=s.n_after,
                tilt_x_deg=-float(y), tilt_y_deg=-float(x), tilt_z_deg=s.tilt_z_deg,
            ) if isinstance(s, LocalCoordinateBreakSurface) else s
            for s in template.surfaces
        ]
        return FittedE2ESystem(
            lens=lens, surfaces=surfaces, image_surface=template.image_surface,
            front_surface=template.front_surface, back_surface=template.back_surface,
            fit_spec=template.fit_spec, wavelength_nm=template.wavelength_nm,
            surface_distances_mm=template.surface_distances_mm,
            image_distance_value_mm=template.image_distance_value_mm,
            initial_ior=template.initial_ior, object_distance_mm=template.object_distance_mm,
            exit_pupil_position_mm=template.exit_pupil_position_mm,
            stop_semi_diameter_mm=template.stop_semi_diameter_mm,
        )

    def _new_system(self, distance: float, x: float, y: float) -> FittedE2ESystem:
        if distance not in self._templates:
            template, temporary = build_fitted_e2e_system(
                self.config.excel, object_distance=distance, field_x_deg=x, field_y_deg=y,
                wavelength_nm=self.config.wavelength_nm, fit_spec=self.fit_spec,
                device=self.device, dtype=torch.float64, train_back_surface=False,
                back_perturbation=self.perturbation,
            )
            try:
                if template.lens is None:
                    raise RuntimeError("fitted template has no Lensdata for aiming")
                self._templates[distance] = (template, template.lens)
            finally:
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)
        template, lens = self._templates[distance]
        self._set_field(lens, x, y)
        return self._clone(template, lens, x, y)

    def _system_and_rays(self, distance: float, x: float, y: float):
        key = self._key(distance, x, y)
        if key not in self._cache:
            if self._cache_frozen:
                raise RuntimeError(
                    "training cache is frozen and lacks the requested case: "
                    f"distance={distance:g}, field=({x:g}, {y:g})"
                )
            system = self._new_system(distance, x, y)
            system.reference_ray = make_aimed_reference_ray(
                system, dtype=torch.float64, device=self.device
            )
            rays = make_aimed_pupil_rays(
                system, sample_count=self.sample_count, pupil_radius_mm=self.config.pupil_radius_mm,
                field_x_deg=x, field_y_deg=y, dtype=torch.float64, device=self.device,
            )
            system.physical_fft_pixel_pitch_mm = compute_physical_fft_pixel_pitch_mm(
                system.lens, pupil_sample_count=self.sample_count,
                psf_size_px=self.config.fft_size_px, wavelength_nm=self.config.wavelength_nm,
            )
            system.release_biot_lens()
            self._cache[key] = system, rays
        return self._cache[key]

    def reference_rear_intersection(
        self, distance_mm: float, field_x_deg: float, field_y_deg: float
    ) -> tuple[float, float]:
        """Trace the aimed centre-pupil ray to the original PAL rear surface.

        Coordinates are physical local-surface ``(x, y)`` in mm.  This is the
        geometric bridge between an object-field case and the lens partition;
        no pupil bundle, PSF, interpolation, or nearest-case fallback is used.
        """

        system = self._new_system(float(distance_mm), float(field_x_deg), float(field_y_deg))
        ray = make_aimed_reference_ray(system, dtype=torch.float64, device=self.device)
        current = ray.normalized()
        active = torch.ones_like(current.weights, dtype=torch.bool)
        n_current = float(system.initial_ior)
        rear_point: torch.Tensor | None = None
        with torch.no_grad():
            for index, surface in enumerate(system.surfaces):
                distance = float(system.surface_distances_mm[index])
                points, hit_valid, _ = surface.intersect(current, distance)
                normals = surface.normal_at(points)
                refracted, refract_valid = _snell(
                    current.directions, normals, n_current, surface.n_after
                )
                active = active & hit_valid & refract_valid
                if surface is system.back_surface:
                    rear_point = points.reshape(-1, 3)[0]
                    break
                current = current.with_state(
                    points,
                    refracted,
                    weights=current.weights * active.to(current.dtype),
                )
                current = surface.after_interaction(current)
                n_current = surface.n_after
        system.release_biot_lens()
        if rear_point is None:
            raise RuntimeError("fitted system does not contain its PAL rear surface")
        if not bool(active.all()) or not bool(torch.isfinite(rear_point).all()):
            raise RuntimeError(
                "invalid centre-pupil reference ray at field "
                f"({float(field_x_deg):g}, {float(field_y_deg):g}) deg, "
                f"distance={float(distance_mm):g} mm"
            )
        return float(rear_point[0].cpu()), float(rear_point[1].cpu())

    def validate_training_case_forward(
        self, case: Mapping[str, Any]
    ) -> dict[str, float | int]:
        """Qualify a sampled field on the exact pre-FFT training forward path.

        A centre-pupil rear-surface hit is necessary for lens-plane sampling but
        is not sufficient for a usable PSF case: BIOT pupil aiming (including
        ``cal_WFNO``) and the complete refractive trace must also succeed.  This
        method deliberately stops before FFT construction, so it changes neither
        forward physics nor the training metric.
        """

        distance = float(case["distance_mm"])
        x = float(case["field_x_deg"])
        y = float(case["field_y_deg"])
        with torch.no_grad():
            system, rays = self._system_and_rays(distance, x, y)
            if not bool(torch.isfinite(rays.origins_mm).all()):
                raise RuntimeError(f"non-finite aimed pupil origins for {case['case_id']}")
            if not bool(torch.isfinite(rays.directions).all()):
                raise RuntimeError(f"non-finite aimed pupil directions for {case['case_id']}")
            trace = trace_system_to_image_with_phase(
                system, rays, phase_reference=self.config.phase_reference
            )
            valid_count = int(trace.valid.sum().detach().cpu())
            ray_count = int(trace.valid.numel())
            if valid_count <= 0:
                raise RuntimeError(f"no valid forward rays for {case['case_id']}")
            if not bool(torch.isfinite(trace.phase_rad[trace.valid]).all()):
                raise RuntimeError(f"non-finite valid-ray phase for {case['case_id']}")
            pixel_pitch = float(system.physical_fft_pixel_pitch_mm)
            if not math.isfinite(pixel_pitch) or pixel_pitch <= 0.0:
                raise RuntimeError(f"invalid physical FFT pixel pitch for {case['case_id']}")
        return {
            "ray_count": ray_count,
            "valid_ray_count": valid_count,
            "valid_fraction": valid_count / ray_count,
            "physical_fft_pixel_pitch_mm": pixel_pitch,
        }

    def validate_training_case_wfno(
        self, case: Mapping[str, Any]
    ) -> dict[str, float]:
        """Qualify the BIOT_vis field-dependent chief/marginal-ray WFNO path."""

        distance = float(case["distance_mm"])
        x = float(case["field_x_deg"])
        y = float(case["field_y_deg"])
        system = self._new_system(distance, x, y)
        try:
            with torch.no_grad():
                pixel_pitch = compute_physical_fft_pixel_pitch_mm(
                    system.lens,
                    pupil_sample_count=self.sample_count,
                    psf_size_px=self.config.fft_size_px,
                    wavelength_nm=self.config.wavelength_nm,
                )
                pixel_pitch = float(pixel_pitch)
            if not math.isfinite(pixel_pitch) or pixel_pitch <= 0.0:
                raise RuntimeError(f"invalid physical FFT pixel pitch for {case['case_id']}")
        finally:
            system.release_biot_lens()
        return {"physical_fft_pixel_pitch_mm": pixel_pitch}

    def field(self, case: Mapping[str, Any]) -> FieldResult:
        distance = float(case["distance_mm"])
        x, y = float(case["field_x_deg"]), float(case["field_y_deg"])
        system, rays = self._system_and_rays(distance, x, y)
        trace = trace_system_to_image_with_phase(
            system, rays, phase_reference=self.config.phase_reference
        )
        if not bool(trace.valid.any()):
            raise RuntimeError(f"no valid rays for {case['case_id']}")
        fft = torch_fft_psf_from_phase(
            trace.phase_rad, trace.valid, sample_count=self.sample_count,
            psf_size_px=self.config.fft_size_px,
            remove_piston=True,
            remove_tilt=self.config.remove_tilt,
        )
        if system.physical_fft_pixel_pitch_mm is None:
            raise RuntimeError("missing physical FFT pixel pitch")
        psf = _normalize_psf(fft.psf)
        zernike_coefficients = fit_low_order_opd_zernike_torch(
            trace.reference_opl_mm,
            trace.valid,
            sample_count=self.sample_count,
        )
        pitch = float(system.physical_fft_pixel_pitch_mm)
        return FieldResult(
            # ``kernel`` is retained as the historical field name, but now
            # carries the authoritative raw physical FFT PSF used by training.
            kernel=psf,
            valid_fraction=trace.valid.to(torch.float64).mean(),
            pixel_pitch_mm=pitch,
            edge_fraction=_edge_fraction(psf),
            valid_mask=trace.valid,
            raw_psf=psf,
            raw_pixel_pitch_mm=pitch,
            zernike_coefficients_mm=zernike_coefficients,
            z4_defocus_mm2=zernike_coefficients[..., 4].square(),
        )

    def field_batch(self, cases: Sequence[Mapping[str, Any]]) -> BatchFieldResult:
        """Run one true tensor batch through trace, FFT and PSF construction."""
        if not cases:
            raise ValueError("cannot evaluate an empty case batch")
        systems: list[FittedE2ESystem] = []
        rays: list[object] = []
        for case in cases:
            distance = float(case["distance_mm"])
            x, y = float(case["field_x_deg"]), float(case["field_y_deg"])
            system, pupil_rays = self._system_and_rays(distance, x, y)
            systems.append(system)
            rays.append(pupil_rays)
        trace = trace_system_batch_to_image_with_phase(
            systems, rays, phase_reference=self.config.phase_reference
        )
        if not bool(trace.valid.any(dim=1).all()):
            invalid = [str(case["case_id"]) for index, case in enumerate(cases) if not bool(trace.valid[index].any())]
            raise RuntimeError("no valid rays for case batch: " + ", ".join(invalid))
        fft = torch_fft_psf_from_phase(
            trace.phase_rad, trace.valid, sample_count=self.sample_count,
            psf_size_px=self.config.fft_size_px,
            remove_piston=True,
            remove_tilt=self.config.remove_tilt,
        )
        physical_pitch = torch.as_tensor(
            [float(system.physical_fft_pixel_pitch_mm) for system in systems],
            device=fft.psf.device, dtype=fft.psf.dtype,
        )
        kernels = _normalize_psf_batch(fft.psf)
        zernike_coefficients = fit_low_order_opd_zernike_torch(
            trace.reference_opl_mm,
            trace.valid,
            sample_count=self.sample_count,
        )
        return BatchFieldResult(
            kernels=kernels,
            valid_fraction=trace.valid.to(kernels.dtype).mean(dim=1),
            pixel_pitch_mm=physical_pitch,
            edge_fraction=_edge_fraction_batch(kernels),
            valid_mask=trace.valid,
            zernike_coefficients_mm=zernike_coefficients,
            z4_defocus_mm2=zernike_coefficients[..., 4].square(),
        )

    def raw_psf_batch(self, cases: Sequence[Mapping[str, Any]]) -> RawPSFBatchResult:
        """为离线评价执行真实 case 小批量追迹与原生 FFT PSF 计算。"""
        if not cases:
            raise ValueError("cannot evaluate an empty raw PSF case batch")
        systems: list[FittedE2ESystem] = []
        rays: list[object] = []
        pitches: list[float] = []
        for case in cases:
            distance = float(case["distance_mm"])
            x, y = float(case["field_x_deg"]), float(case["field_y_deg"])
            system, pupil_rays = self._system_and_rays(distance, x, y)
            pitch = system.physical_fft_pixel_pitch_mm
            if pitch is None or not math.isfinite(float(pitch)) or float(pitch) <= 0.0:
                raise RuntimeError(
                    f"invalid physical raw FFT pixel pitch for {case['case_id']}"
                )
            systems.append(system)
            rays.append(pupil_rays)
            pitches.append(float(pitch))
        trace = trace_system_batch_to_image_with_phase(
            systems, rays, phase_reference=self.config.phase_reference
        )
        if not bool(trace.valid.any(dim=1).all()):
            invalid = [
                str(case["case_id"])
                for index, case in enumerate(cases)
                if not bool(trace.valid[index].any())
            ]
            raise RuntimeError("no valid rays for raw PSF case batch: " + ", ".join(invalid))
        fft = torch_fft_psf_from_phase(
            trace.phase_rad,
            trace.valid,
            sample_count=self.sample_count,
            psf_size_px=self.config.fft_size_px,
            remove_piston=True,
            remove_tilt=self.config.remove_tilt,
        )
        return RawPSFBatchResult(
            psf=fft.psf,
            valid_fraction=trace.valid.to(dtype=fft.psf.dtype).mean(dim=1),
            pixel_pitch_mm=torch.as_tensor(
                pitches, device=fft.psf.device, dtype=fft.psf.dtype
            ),
        )


def _finite_difference(values: torch.Tensor, pitch: float) -> tuple[torch.Tensor, torch.Tensor]:
    dy, dx = torch.empty_like(values), torch.empty_like(values)
    dy[1:-1], dy[0], dy[-1] = (values[2:] - values[:-2]) / (2 * pitch), (values[1] - values[0]) / pitch, (values[-1] - values[-2]) / pitch
    dx[:, 1:-1], dx[:, 0], dx[:, -1] = (values[:, 2:] - values[:, :-2]) / (2 * pitch), (values[:, 1] - values[:, 0]) / pitch, (values[:, -1] - values[:, -2]) / pitch
    return dy, dx


def torch_averfang_maps(sag: torch.Tensor, config: PALPowerConfig) -> dict[str, torch.Tensor]:
    pitch = 2 * config.semi_diameter_mm / (int(sag.shape[0]) - 1)
    zy, zx = _finite_difference(sag, pitch)
    zyy, zyx = _finite_difference(zy, pitch)
    zxy, zxx = _finite_difference(zx, pitch)
    zxy = 0.5 * (zxy + zyx)
    e, f, g = 1 + zx.square(), zx * zy, 1 + zy.square()
    scale = torch.sqrt(1 + zx.square() + zy.square())
    l, m, n = zxx / scale, zxy / scale, zyy / scale
    eps = torch.finfo(sag.dtype).eps
    denom = (e * g - f.square()).clamp_min(eps)
    gaussian = (l * n - m.square()) / denom.square().clamp_min(eps)
    mean = (e * n + g * l - 2 * f * m) / (2 * denom.pow(1.5).clamp_min(eps))
    disc = (mean.square() - gaussian).clamp_min(0)
    pmax, pmin = mean + torch.sqrt(disc), mean - torch.sqrt(disc)
    pmax = torch.where(pmax.abs() > eps, pmax, torch.full_like(pmax, eps))
    pmin = torch.where(pmin.abs() > eps, pmin, torch.full_like(pmin, eps))
    rear_radius = -0.5 * (1 / pmax + 1 / pmin)
    curvature_diff = pmax - pmin
    count = int(round(config.crib_diameter_mm)) + 1
    start = (int(sag.shape[0]) - count) // 2
    stop = start + count
    rr, cdiff, z = rear_radius[start:stop, start:stop], curvature_diff[start:stop, start:stop], sag[start:stop, start:stop]
    coord = torch.linspace(-config.semi_diameter_mm, config.semi_diameter_mm, int(sag.shape[0]), device=sag.device, dtype=sag.dtype)[start:stop]
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    rq = torch.as_tensor(config.front_radius_mm, device=sag.device, dtype=sag.dtype)
    front = -torch.sqrt((rq.square() - xx.square() - yy.square()).clamp_min(0)) + rq
    thickness = z - front + config.center_thickness_mm
    nl = torch.as_tensor(config.refractive_index, device=sag.device, dtype=sag.dtype)
    denominator = -nl * rr * rq + (nl - 1) * thickness * rr
    denominator = torch.where(denominator.abs() > eps, denominator, torch.full_like(denominator, eps))
    power = (nl - 1) * 1000 * (nl * (-rr - rq) + (nl - 1) * thickness) / denominator
    astig = (-(curvature_diff[start:stop, start:stop]) * (nl - 1) * 1000).abs()
    valid = (xx.square() + yy.square() <= (config.crib_diameter_mm / 2) ** 2 + 1e-9) & torch.isfinite(power) & torch.isfinite(astig)
    return {
        "power_D": power,
        "A_D": astig,
        "astigmatism_D": astig,
        "valid": valid,
    }


def load_pal(config: MinimalConfig, device: torch.device) -> tuple[torch.Tensor, PALPowerConfig, dict[str, torch.Tensor]]:
    lens = load_lens(Path(config.excel), device=resolve_device("cpu"), wavelength_nm=config.wavelength_nm)
    adapter = build_legacy_adapter(lens, wavelength_nm=config.wavelength_nm)
    back = lens.surfaces[2]
    if not isinstance(back, GridSag):
        raise ValueError("PAL rear surface must be GridSag")
    sag = torch.as_tensor(load_sag_xlsx(Path(back.sag_file_path), grid_shape=back.grid_shape), dtype=torch.float64, device=device)
    power_config = PALPowerConfig(float(back.semi_dia), float(adapter.n1), float(1 / adapter.c0), float(adapter.h_glass_mm))
    payload = json.loads(Path(config.zones_json).read_text(encoding="utf-8-sig"))
    zones = {name: torch.as_tensor(value, dtype=torch.bool, device=device) for name, value in payload["masks"].items()}
    return sag, power_config, zones


def prescription_metrics(
    sag: torch.Tensor,
    power_config: PALPowerConfig,
    zones: Mapping[str, torch.Tensor],
    *,
    baseline_sag: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    maps = torch_averfang_maps(sag, power_config)
    valid = maps["valid"]
    far = zones["far_reference"] & valid
    near = zones["near_reference"] & valid
    monitored = zones["monitored"] & valid
    if not bool(far.any() and near.any() and monitored.any()):
        raise ValueError("P_far/ADD masks have no valid power samples")
    pfar = maps["power_D"][far].mean()
    add = maps["power_D"][near].mean() - pfar
    guard = zones.get("lower_edge_guard")
    if guard is None:
        raise ValueError("zones.json must define lower_edge_guard")
    guard_mask = guard & valid
    if not bool(guard_mask.any()):
        raise ValueError("lower_edge_guard has no valid power samples")
    if baseline_sag is None:
        guard_power_change = torch.zeros((), device=sag.device, dtype=sag.dtype)
        guard_astig_change = torch.zeros((), device=sag.device, dtype=sag.dtype)
    else:
        baseline_maps = torch_averfang_maps(baseline_sag, power_config)
        common_guard = guard_mask & baseline_maps["valid"]
        if not bool(common_guard.any()):
            raise ValueError("lower_edge_guard has no common baseline/candidate samples")
        guard_power_change = (
            maps["power_D"][common_guard] - baseline_maps["power_D"][common_guard]
        ).abs().max()
        guard_astig_change = (
            maps["astigmatism_D"][common_guard]
            - baseline_maps["astigmatism_D"][common_guard]
        ).abs().max()
    return {
        "P_far_D": pfar,
        "ADD_D": add,
        "astig_mean_D": maps["astigmatism_D"][monitored].mean(),
        "lower_edge_max_abs_power_change_D": guard_power_change,
        "lower_edge_max_abs_astig_change_D": guard_astig_change,
    }


def build_joint_training_cases(
    traced_candidates: Sequence[Mapping[str, Any]], config: MinimalConfig,
    *, corridor_y_min_mm: float, corridor_y_max_mm: float,
    zones_payload: Mapping[str, Any],
    group_counts: Mapping[str, int] = TRAINING_GROUP_COUNTS,
    peripheral_band_counts: Mapping[str, int] = PERIPHERAL_BAND_COUNTS,
) -> list[dict[str, Any]]:
    """Build the fixed 109-case contract from traced dense-field candidates."""
    maps = dict(zones_payload.get("maps", {}))
    if "power_D" not in maps:
        raise ValueError("zones.json must store the Original PAL power_D map")
    power_map = np.asarray(maps["power_D"], dtype=np.float64)
    far_reference = np.asarray(
        dict(zones_payload["masks"])["far_reference"], dtype=bool
    )
    if power_map.shape != far_reference.shape or not bool(far_reference.any()):
        raise ValueError("zones Original PAL power map/far reference is malformed")
    pfar = float(np.mean(power_map[far_reference]))
    return select_training_cases(
        traced_candidates,
        far_object_distance_mm=config.far_object_distance_mm,
        intermediate_object_distance_mm=config.intermediate_object_distance_mm,
        near_object_distance_mm=config.near_object_distance_mm,
        corridor_y_min_mm=corridor_y_min_mm,
        corridor_y_max_mm=corridor_y_max_mm,
        power_map=power_map,
        pfar=pfar,
        zones_payload=zones_payload,
        group_counts=group_counts,
        peripheral_band_counts=peripheral_band_counts,
    )


def _trace_preoptimization_case_geometry(
    model: MinimalOpticalModel,
    cases: Sequence[Mapping[str, Any]],
    *,
    zones_json: str | Path,
    reference_distance_mm: float = float("inf"),
) -> list[dict[str, Any]]:
    zones_payload = json.loads(Path(zones_json).read_text(encoding="utf-8-sig"))
    result: list[dict[str, Any]] = []
    for case in cases:
        fx = float(case["field_x_deg"])
        fy = float(case["field_y_deg"])
        if str(case["training_group"]) in PERIPHERAL_GROUPS:
            reference_x = float(case["reference_lens_x_mm"])
            reference_y = float(case["reference_lens_physical_y_mm"])
            case_x, case_y = reference_x, reference_y
        else:
            reference_x, reference_y = model.reference_rear_intersection(
                reference_distance_mm, fx, fy
            )
            case_x, case_y = model.reference_rear_intersection(
                float(case["distance_mm"]), fx, fy
            )
        result.append(
            {
                **dict(case),
                "partition_reference_distance_mm": float(reference_distance_mm),
                "reference_lens_x_mm": reference_x,
                "reference_lens_physical_y_mm": reference_y,
                "reference_partition_zone": classify_partition_point(
                    zones_payload, x_mm=reference_x, physical_y_mm=reference_y
                ),
                "case_lens_x_mm": case_x,
                "case_lens_physical_y_mm": case_y,
                "case_position_partition_zone": classify_partition_point(
                    zones_payload, x_mm=case_x, physical_y_mm=case_y
                ),
            }
        )
    return result


def _write_run_config(output: Path, config: MinimalConfig) -> None:
    _write_json_atomic(output / "config.json", asdict(config))


def _qualified_pool_case_for_saved_attempt(
    attempt: Mapping[str, Any],
    qualified_pool: Sequence[dict[str, Any]],
    qualified_pool_by_key: Mapping[
        tuple[str, str, float, float, float], dict[str, Any]
    ],
) -> dict[str, Any] | None:
    """Resolve current progress by stable source identity, with legacy-only fallbacks."""
    if attempt.get("training_group") is not None:
        try:
            return qualified_pool_by_key.get(qualified_source_key(attempt))
        except KeyError:
            return None
    exact_case_id = str(attempt.get("case_id", ""))
    for row in qualified_pool:
        if str(row.get("case_id")) == exact_case_id:
            return row
    if attempt.get("candidate_id") is not None:
        prefix_matches = [
            row for row in qualified_pool
            if str(row.get("candidate_id")) == str(attempt["candidate_id"])
            and exact_case_id.startswith(str(row.get("training_group")) + "_")
        ]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
    return None


def _prepare_case_layout(
    config: MinimalConfig, output: Path, model: MinimalOpticalModel
) -> list[dict[str, Any]]:
    zones_payload = json.loads(Path(config.zones_json).read_text(encoding="utf-8-sig"))
    corridor_stats = dict(dict(zones_payload.get("statistics", {})).get("corridor", {}))
    corridor_range = corridor_stats.get("physical_y_range_mm")
    if not isinstance(corridor_range, list) or len(corridor_range) != 2:
        raise ValueError("zones.json must declare corridor physical_y_range_mm")
    candidate_fields = generate_dense_candidate_fields(
        field_x_min_deg=(
            None if config.candidate_field_min_deg is not None
            else config.candidate_field_x_min_deg
        ),
        field_x_max_deg=(
            None if config.candidate_field_max_deg is not None
            else config.candidate_field_x_max_deg
        ),
        field_y_min_deg=(
            None if config.candidate_field_min_deg is not None
            else config.candidate_field_y_min_deg
        ),
        field_y_max_deg=(
            None if config.candidate_field_max_deg is not None
            else config.candidate_field_y_max_deg
        ),
        field_step_deg=config.candidate_field_step_deg,
        field_min_deg=config.candidate_field_min_deg,
        field_max_deg=config.candidate_field_max_deg,
    )
    candidate_progress_path = output / "candidate_trace_progress.json"
    candidate_trace_import = (
        Path(config.parent_run) / "candidate_trace_progress.json"
        if config.parent_run is not None else
        (None if config.candidate_trace_import is None else Path(config.candidate_trace_import))
    )
    if candidate_trace_import is not None and not candidate_progress_path.exists():
        source = candidate_trace_import
        imported = _read_json(source)
        if imported.get("status") != "complete":
            raise ValueError("candidate_trace_import must be a complete trace-progress artifact")
        temporary = _temporary_sibling(candidate_progress_path)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(source, temporary)
            _replace_atomic(temporary, candidate_progress_path)
        finally:
            temporary.unlink(missing_ok=True)
    traced_candidates = trace_candidate_fields(
        candidate_fields,
        trace_reference=lambda fx, fy: model.reference_rear_intersection(
            config.far_object_distance_mm, fx, fy
        ),
        zones_payload=zones_payload,
        zone_boundary_safety_mm={
            "default": config.zone_boundary_safety_mm,
            "corridor": config.corridor_zone_boundary_safety_mm,
        },
        aperture_edge_safety_mm=config.aperture_edge_safety_mm,
        progress_path=candidate_progress_path,
        trace_identity={
            "excel_sha256": _sha256_file(config.excel),
            "reference_distance_mm": config.far_object_distance_mm,
            "wavelength_nm": config.wavelength_nm,
            "forward_contract": "Original S + exact Original GridSag + BIOT_vis GRIN3",
        },
    )
    qualification_pool = build_joint_training_cases(
        traced_candidates, config,
        corridor_y_min_mm=float(corridor_range[0]),
        corridor_y_max_mm=float(corridor_range[1]),
        zones_payload=zones_payload,
        group_counts=FORWARD_POOL_GROUP_COUNTS,
        peripheral_band_counts=FORWARD_POOL_PERIPHERAL_BAND_COUNTS,
    )
    pool_identity = _canonical_json_sha256([
        {
            "case_id": str(case["case_id"]),
            "candidate_id": str(case["candidate_id"]),
            "distance_mm": float(case["distance_mm"]),
            "field_x_deg": float(case["field_x_deg"]),
            "field_y_deg": float(case["field_y_deg"]),
        }
        for case in qualification_pool
    ])
    qualification_progress_path = output / "forward_qualification_progress.json"
    _import_complete_pool_progress(
        source_path=(
            Path(config.parent_run) / "forward_qualification_progress.json"
            if config.parent_run is not None else config.forward_qualification_import
        ),
        destination_path=qualification_progress_path,
        pool_identity_sha256=pool_identity,
        progress_name="forward qualification progress",
    )
    pool_attempts: list[dict[str, Any]] = []
    if qualification_progress_path.is_file():
        saved = _read_json(qualification_progress_path)
        if saved.get("schema_version") != 1 or saved.get("pool_identity_sha256") != pool_identity:
            raise ValueError("forward qualification progress identity mismatch")
        rows = saved.get("pool_attempts")
        if not isinstance(rows, list):
            raise ValueError("forward qualification progress rows are malformed")
        pool_attempts = [dict(row) for row in rows]
    attempted = {str(row["pool_case_id"]): row for row in pool_attempts}
    traced_by_id = {str(row["candidate_id"]): row for row in traced_candidates}

    def save_qualification_progress(status: str) -> None:
        _write_json_atomic(
            qualification_progress_path,
            {
                "schema_version": 1,
                "status": status,
                "pool_identity_sha256": pool_identity,
                "pool_size": len(qualification_pool),
                "pool_attempts": pool_attempts,
            },
        )

    for pool_index, case in enumerate(qualification_pool, 1):
        candidate_id = str(case["candidate_id"])
        pool_case_id = str(case["case_id"])
        group = str(case["training_group"])
        prior = attempted.get(pool_case_id)
        if prior is None:
            if group in PERIPHERAL_GROUPS:
                prior = {
                    "candidate_id": candidate_id,
                    "pool_case_id": str(case["case_id"]),
                    "training_group": group,
                    "distance_mm": float(case["distance_mm"]),
                    "field_x_deg": float(case["field_x_deg"]),
                    "field_y_deg": float(case["field_y_deg"]),
                    "status": "surface_only",
                    "qualification_mode": "no_ray_trace_surface_astigmatism",
                }
            else:
                diagnostic_stream = io.StringIO()
                try:
                    with contextlib.redirect_stdout(diagnostic_stream), contextlib.redirect_stderr(diagnostic_stream):
                        aiming_audit = model.validate_training_case_wfno(case)
                    prior = {
                        "candidate_id": candidate_id,
                        "pool_case_id": str(case["case_id"]),
                        "training_group": group,
                        "distance_mm": float(case["distance_mm"]),
                        "field_x_deg": float(case["field_x_deg"]),
                        "field_y_deg": float(case["field_y_deg"]),
                        "status": "ok",
                        **dict(aiming_audit),
                    }
                except Exception as exc:
                    diagnostic = diagnostic_stream.getvalue()
                    prior = {
                        "candidate_id": candidate_id,
                        "pool_case_id": str(case["case_id"]),
                        "training_group": group,
                        "distance_mm": float(case["distance_mm"]),
                        "field_x_deg": float(case["field_x_deg"]),
                        "field_y_deg": float(case["field_y_deg"]),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "diagnostic_sha256": hashlib.sha256(diagnostic.encode("utf-8")).hexdigest(),
                        "diagnostic_tail": diagnostic[-4000:],
                    }
            pool_attempts.append(prior)
            attempted[pool_case_id] = prior
            save_qualification_progress("running")
            if pool_index % 20 == 0 or pool_index == len(qualification_pool):
                print(
                    f"forward WFNO pool: {pool_index}/{len(qualification_pool)}, "
                    f"failures={sum(row['status'] == 'failed' for row in pool_attempts)}",
                    flush=True,
                )
        source = traced_by_id[candidate_id]
        source["forward_wfno_status"] = str(prior["status"])
        if prior["status"] in {"ok", "surface_only"}:
            case["forward_wfno_validation"] = (
                {"physical_fft_pixel_pitch_mm": prior["physical_fft_pixel_pitch_mm"]}
                if prior["status"] == "ok"
                else {"qualification_mode": prior["qualification_mode"]}
            )
        else:
            case["eligible"] = False
            source["forward_wfno_error_type"] = str(prior["error_type"])
            source["forward_wfno_error"] = str(prior["error"])
    save_qualification_progress("complete")

    qualified_pool = [dict(case) for case in qualification_pool if bool(case.get("eligible"))]
    forward_attempts: list[dict[str, Any]] = []
    phase_progress_path = output / "final_phase_qualification_progress.json"
    _import_complete_pool_progress(
        source_path=(
            Path(config.parent_run) / "final_phase_qualification_progress.json"
            if config.parent_run is not None else config.final_phase_qualification_import
        ),
        destination_path=phase_progress_path,
        pool_identity_sha256=pool_identity,
        progress_name="final phase qualification progress",
    )
    qualified_pool_by_key: dict[tuple[str, str, float, float, float], dict[str, Any]] = {}
    for pool_row in qualified_pool:
        key = qualified_source_key(pool_row)
        if key in qualified_pool_by_key:
            raise ValueError(
                "qualified pool contains duplicate stable case key: "
                f"group={key[0]}, candidate={key[1]}, distance={key[2]:g}, "
                f"field=({key[3]:g},{key[4]:g})"
            )
        qualified_pool_by_key[key] = pool_row

    def pool_case_for_final_case(case: Mapping[str, Any]) -> dict[str, Any]:
        key = qualified_source_key(case)
        pool_case = qualified_pool_by_key.get(key)
        if pool_case is None:
            raise ValueError(
                "final training case has no matching qualified-pool record; "
                f"stable_key={key!r}, final_case_id={case.get('case_id')!r}. "
                "The final case must retain the same training_group, candidate_id, "
                "distance and field coordinates as its qualified source."
            )
        return pool_case

    if phase_progress_path.is_file():
        saved = _read_json(phase_progress_path)
        if saved.get("schema_version") != 1 or saved.get("pool_identity_sha256") != pool_identity:
            raise ValueError("final phase qualification progress identity mismatch")
        rows = saved.get("forward_attempts")
        if not isinstance(rows, list):
            raise ValueError("final phase qualification progress rows are malformed")
        forward_attempts = [dict(row) for row in rows]
        for attempt in forward_attempts:
            exact_case_id = str(attempt.get("case_id", ""))
            pool_case = _qualified_pool_case_for_saved_attempt(
                attempt, qualified_pool, qualified_pool_by_key
            )
            if pool_case is None:
                raise ValueError(
                    "final phase qualification progress references a foreign or "
                    "renumbered candidate; include training_group/candidate_id and "
                    f"stable coordinates in the progress record (case_id={exact_case_id!r})"
                )
            if attempt["status"] in {"ok", "surface_only"}:
                pool_case["forward_training_status"] = str(attempt["status"])
                pool_case["forward_training_validation"] = (
                    {
                        name: attempt[name]
                        for name in (
                            "ray_count", "valid_ray_count", "valid_fraction",
                            "physical_fft_pixel_pitch_mm",
                        )
                    }
                    if attempt["status"] == "ok"
                    else {"qualification_mode": attempt["qualification_mode"]}
                )
            elif attempt["status"] == "failed":
                pool_case["eligible"] = False
            else:
                raise ValueError("final phase qualification progress has an invalid status")

    def save_phase_progress(status: str) -> None:
        _write_json_atomic(
            phase_progress_path,
            {
                "schema_version": 1,
                "status": status,
                "pool_identity_sha256": pool_identity,
                "forward_attempts": forward_attempts,
            },
        )

    validation_round = max(
        (int(row.get("validation_round", 0)) for row in forward_attempts),
        default=0,
    )
    while True:
        validation_round += 1
        training_cases = build_joint_training_cases(
            qualified_pool, config,
            corridor_y_min_mm=float(corridor_range[0]),
            corridor_y_max_mm=float(corridor_range[1]),
            zones_payload=zones_payload,
        )
        training_cases = _trace_preoptimization_case_geometry(
            model, training_cases, zones_json=config.zones_json,
            reference_distance_mm=config.far_object_distance_mm,
        )
        failed_ids: set[str] = set()
        for case in training_cases:
            candidate_id = str(case["candidate_id"])
            case_id = str(case["case_id"])
            pool_case = pool_case_for_final_case(case)
            if pool_case.get("forward_training_status") in {"ok", "surface_only"}:
                case["forward_training_validation"] = dict(
                    pool_case["forward_training_validation"]
                )
                continue
            if str(case["training_group"]) in PERIPHERAL_GROUPS:
                audit = {"qualification_mode": "no_ray_trace_surface_astigmatism"}
                pool_case["forward_training_status"] = "surface_only"
                pool_case["forward_training_validation"] = dict(audit)
                case["forward_training_validation"] = dict(audit)
                forward_attempts.append({
                    "validation_round": validation_round,
                    "candidate_id": candidate_id,
                    "case_id": case_id,
                    "training_group": str(case["training_group"]),
                    "distance_mm": float(case["distance_mm"]),
                    "field_x_deg": float(case["field_x_deg"]),
                    "field_y_deg": float(case["field_y_deg"]),
                    "status": "surface_only",
                    **audit,
                })
                save_phase_progress("running")
                continue
            diagnostic_stream = io.StringIO()
            try:
                with contextlib.redirect_stdout(diagnostic_stream), contextlib.redirect_stderr(diagnostic_stream):
                    audit = model.validate_training_case_forward(case)
                pool_case["forward_training_status"] = "ok"
                pool_case["forward_training_validation"] = dict(audit)
                case["forward_training_validation"] = dict(audit)
                forward_attempts.append({
                    "validation_round": validation_round,
                    "candidate_id": candidate_id,
                    "case_id": str(case["case_id"]),
                    "training_group": str(case["training_group"]),
                    "distance_mm": float(case["distance_mm"]),
                    "field_x_deg": float(case["field_x_deg"]),
                    "field_y_deg": float(case["field_y_deg"]),
                    "status": "ok",
                    **dict(audit),
                })
                save_phase_progress("running")
            except Exception as exc:
                diagnostic = diagnostic_stream.getvalue()
                pool_case["eligible"] = False
                failed_ids.add(case_id)
                forward_attempts.append({
                    "validation_round": validation_round,
                    "candidate_id": candidate_id,
                    "case_id": str(case["case_id"]),
                    "training_group": str(case["training_group"]),
                    "distance_mm": float(case["distance_mm"]),
                    "field_x_deg": float(case["field_x_deg"]),
                    "field_y_deg": float(case["field_y_deg"]),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "diagnostic_sha256": hashlib.sha256(diagnostic.encode("utf-8")).hexdigest(),
                    "diagnostic_tail": diagnostic[-4000:],
                })
                save_phase_progress("running")
        if not failed_ids:
            break
        print(
            f"final training-case phase qualification round {validation_round}: "
            f"rejected {len(failed_ids)} pool candidates; rerunning regional FPS",
            flush=True,
        )
    save_phase_progress("complete")
    cache_audit = _retain_training_cache(
        model,
        training_cases,
        extra_cases=_startup_cases(config),
    )
    _write_json_atomic(
        output / "preoptimization" / "forward_qualification_audit.json",
        {
            "schema_version": 1,
            "contract": (
                f"fixed {sum(FORWARD_POOL_GROUP_COUNTS.values())}-case regional FPS pool + "
                "exact BIOT_vis field-dependent WFNO for traced functional cases + "
                "surface-only peripheral qualification + final group-preserving "
                "coverage-constrained selection + "
                "complete pre-FFT phase trace for 79 functional cases"
            ),
            "pool_group_counts": FORWARD_POOL_GROUP_COUNTS,
            "pool_peripheral_band_counts": FORWARD_POOL_PERIPHERAL_BAND_COUNTS,
            "pool_attempt_count": len(pool_attempts),
            "pool_failure_count": sum(row["status"] == "failed" for row in pool_attempts),
            "pool_attempts": pool_attempts,
            "validation_round_count": validation_round,
            "attempt_count": len(forward_attempts),
            "failure_count": sum(row["status"] == "failed" for row in forward_attempts),
            "attempts": forward_attempts,
            "final_cache_audit": cache_audit,
        },
    )
    write_preoptimization_artifacts(
        output_dir=output / "preoptimization",
        excel_path=config.excel,
        zones_json=config.zones_json,
        # The selected-case anchor metadata and coverage bounds are defined on
        # the forward-qualified FPS pool, not on the original dense geometric
        # grid.  Auditing against the dense grid mixes two different sampling
        # domains after failed pool candidates have been excluded.
        candidates=qualified_pool,
        cases=training_cases,
        dense_candidates=traced_candidates,
        reference_distance_mm=config.far_object_distance_mm,
        sampling_contract={
            "method": (
                "dense field -> Original PAL rear trace -> classified partition -> "
                "fixed region-wise lens-plane FPS pool -> exact aiming/WFNO qualification for functional cases -> "
                "final group-preserving coverage-constrained selection -> 79-case complete "
                "pre-FFT phase qualification; peripheral is surface-only"
            ),
            "field_grid_deg": {
                "x_min": config.candidate_field_x_min_deg,
                "x_max": config.candidate_field_x_max_deg,
                "y_min": config.candidate_field_y_min_deg,
                "y_max": config.candidate_field_y_max_deg,
                "step": config.candidate_field_step_deg,
                "deprecated_square_min": config.candidate_field_min_deg,
                "deprecated_square_max": config.candidate_field_max_deg,
            },
            "candidate_eligibility": (
                "trace_status=ok and reference_partition_zone is classified; "
                "zone/aperture clearance filters disabled"
            ),
            "object_distance_mm": {
                "far": config.far_object_distance_mm,
                "intermediate": config.intermediate_object_distance_mm,
                "near": config.near_object_distance_mm,
            },
            "peripheral_band_distance_mm": {
                "upper": config.far_object_distance_mm,
                "middle": config.far_object_distance_mm,
                "lower": config.far_object_distance_mm,
            },
            "group_counts": TRAINING_GROUP_COUNTS,
            "corridor": "upper/middle/lower ADD strata; five lens-plane FPS points per stratum",
            "peripheral": "15 exact field-mirror pairs per side; upper/middle/lower = 4/5/6",
            "peripheral_distance": "surface-only metric; all cases retain the far reference distance",
            "forward_qualification": (
                f"fixed {sum(FORWARD_POOL_GROUP_COUNTS.values())}-case spatial pool; exact BIOT_vis "
                "field-dependent WFNO for functional cases; complete pre-FFT phase trace on final "
                "79 functional cases; 30 peripheral cases use no ray trace"
            ),
            "forward_pool_group_counts": FORWARD_POOL_GROUP_COUNTS,
            "forward_pool_peripheral_band_counts": FORWARD_POOL_PERIPHERAL_BAND_COUNTS,
        },
    )
    return training_cases


def _validate_case_layout_state(
    training_cases: Sequence[Mapping[str, Any]],
) -> None:
    if not training_cases:
        raise ValueError("case layout state contains an empty training case set")
    counts = {
        name: sum(str(case.get("training_group")) == name for case in training_cases)
        for name in TRAINING_GROUP_COUNTS
    }
    if counts != TRAINING_GROUP_COUNTS:
        raise ValueError(f"case layout state violates the training group contract: {counts}")
    training_ids = [str(case["case_id"]) for case in training_cases]
    if len(training_ids) != len(set(training_ids)):
        raise ValueError("training case IDs are not unique")


def _prepare_or_load_case_layout(
    config: MinimalConfig,
    output: Path,
    model: MinimalOpticalModel,
    *,
    identity_sha256: str,
) -> list[dict[str, Any]]:
    state_path = output / "case_layout_state.json"
    if state_path.is_file():
        payload = _read_json(state_path)
        if int(payload.get("schema_version", -1)) != CASE_LAYOUT_STATE_SCHEMA_VERSION:
            raise ValueError("case layout state schema mismatch")
        if str(payload.get("identity_sha256", "")) != identity_sha256:
            raise ValueError("case layout state identity mismatch")
        training_cases = payload.get("training_cases")
        if not isinstance(training_cases, list):
            raise ValueError("case layout state is malformed")
        training = [dict(case) for case in training_cases]
        _validate_case_layout_state(training)
        expected_hash = _canonical_json_sha256({"training_cases": training})
        if str(payload.get("case_payload_sha256", "")) != expected_hash:
            raise ValueError("case layout state payload hash mismatch")
        return training

    training = _prepare_case_layout(config, output, model)
    _validate_case_layout_state(training)
    case_payload = {"training_cases": training}
    _write_json_atomic(
        state_path,
        {
            "schema_version": CASE_LAYOUT_STATE_SCHEMA_VERSION,
            "identity_sha256": identity_sha256,
            "case_payload_sha256": _canonical_json_sha256(case_payload),
            **case_payload,
        },
    )
    return training


def prepare_only(config: MinimalConfig, *, resume: bool = False) -> Path:
    """Generate the partition/case contract without starting PSF optimization."""

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output, identity = _open_run_directory(config, resume=resume)
    summary_path = output / "summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if str(summary.get("identity_sha256", "")) != str(identity["identity_sha256"]):
            raise ValueError("completed summary identity mismatch")
        return output / "preoptimization"
    _write_run_config(output, config)
    prior_elapsed = _elapsed_seconds(output)
    session_start = time.time()
    _write_run_state(
        output,
        identity_sha256=identity["identity_sha256"],
        status="running",
        phase="case_layout",
        elapsed_seconds=prior_elapsed,
    )
    module = FixedWeightNURBSPerturbation(7, device=device, dtype=torch.float64)
    model = MinimalOpticalModel(config, module)
    _prepare_or_load_case_layout(
        config, output, model, identity_sha256=identity["identity_sha256"],
    )
    _write_run_state(
        output,
        identity_sha256=identity["identity_sha256"],
        status="prepared",
        phase="case_layout_complete",
        elapsed_seconds=prior_elapsed + time.time() - session_start,
    )
    return output / "preoptimization"


def _summarize_training_baseline(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, dict[str, Any]]:
    required = set(FUNCTIONAL_GROUPS + PERIPHERAL_GROUPS)
    actual = {str(row.get("training_group")) for row in rows}
    if actual != required:
        raise ValueError(f"baseline rows do not contain exactly the ten groups: {sorted(actual)}")
    maximum_edge = max(float(row["edge_fraction"]) for row in rows)
    health: dict[str, Any] = {
        "minimum_valid_fraction_ratio": 1.0,
        "maximum_edge_fraction": maximum_edge,
        "objective_name": (
            "J=sum(group_weight*mean(normalized routed metric)); "
            "far=CSF-MTF loss, corridor/near=Z4 OPD mm^2, peripheral=surface A_D"
        ),
    }
    for name in FUNCTIONAL_GROUPS + PERIPHERAL_GROUPS:
        health[f"J_{name}"] = 1.0
    health.update(
        {
            "J_mid": 1.0,
            "J_functional": 1.0,
            "J_peripheral": 1.0,
            "J_total": 1.0,
        }
    )
    return 1.0, health


def _validate_baseline_progress_prefix(
    rows: Sequence[Mapping[str, Any]], expected_ids: Sequence[str], *, label: str,
) -> None:
    if len(rows) > len(expected_ids):
        raise ValueError(f"baseline {label} progress exceeds its case list")
    actual_ids = [str(row.get("case_id")) for row in rows]
    if actual_ids != list(expected_ids[: len(rows)]):
        raise ValueError(f"baseline {label} progress case order/IDs do not match")


def _case_batches(
    cases: Sequence[Mapping[str, Any]], batch_size: int,
) -> Sequence[Sequence[Mapping[str, Any]]]:
    if int(batch_size) <= 0:
        raise ValueError("case batch size must be positive")
    return [cases[start : start + int(batch_size)] for start in range(0, len(cases), int(batch_size))]


def _field_batch(model: Any, cases: Sequence[Mapping[str, Any]]) -> BatchFieldResult:
    """Get a batch result; the scalar adapter is only for lightweight unit doubles."""
    method = getattr(model, "field_batch", None)
    if callable(method):
        result = method(cases)
        if not isinstance(result, BatchFieldResult):
            raise TypeError("field_batch must return BatchFieldResult")
        return result
    # Production MinimalOpticalModel always implements field_batch.  This
    # explicit adapter keeps old scalar test doubles useful without making the
    # production path silently fall back to serial tracing.
    scalar = [model.field(case) for case in cases]
    kernels = torch.stack([item.kernel for item in scalar])
    pitch = torch.as_tensor([item.pixel_pitch_mm for item in scalar], device=kernels.device, dtype=kernels.dtype)
    return BatchFieldResult(
        kernels=kernels,
        valid_fraction=torch.stack([item.valid_fraction.to(kernels.dtype) for item in scalar]),
        pixel_pitch_mm=pitch,
        edge_fraction=torch.stack([item.edge_fraction.to(kernels.dtype) for item in scalar]),
        valid_mask=None,
        zernike_coefficients_mm=(
            None
            if any(item.zernike_coefficients_mm is None for item in scalar)
            else torch.stack([item.zernike_coefficients_mm for item in scalar])
        ),
        z4_defocus_mm2=(
            None
            if any(item.z4_defocus_mm2 is None for item in scalar)
            else torch.stack([item.z4_defocus_mm2 for item in scalar])
        ),
    )


def _batch_rows(
    cases: Sequence[Mapping[str, Any]], result: BatchFieldResult,
    moments: torch.Tensor, loss_metrics: torch.Tensor,
    loss_metric_names: Sequence[str], baseline_valid: Mapping[str, float] | None,
    scores: torch.Tensor,
    *,
    z4_defocus_mm2: torch.Tensor,
    csf_mtf_loss: torch.Tensor,
    astig_A_D: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        vf = float(result.valid_fraction[index].detach().cpu())
        ratio = 1.0 if baseline_valid is None else vf / float(baseline_valid[str(case["case_id"])])
        rows.append({
            **dict(case),
            "m2_mm2": float(moments[index].detach().cpu()),
            "astig_A_D": float(astig_A_D[index].detach().cpu()),
            "z4_defocus_mm2": float(z4_defocus_mm2[index].detach().cpu()),
            "csf_mtf_loss": float(csf_mtf_loss[index].detach().cpu()),
            "loss_metric": float(loss_metrics[index].detach().cpu()),
            "loss_metric_name": loss_metric_names[index],
            "score": float(scores[index].detach().cpu()),
            "valid_fraction": vf,
            "valid_fraction_ratio": ratio,
            "edge_fraction": float(result.edge_fraction[index].detach().cpu()),
        })
    return rows


def _loss_metrics_for_batch(
    model: MinimalOpticalModel,
    cases: Sequence[Mapping[str, Any]],
    result: BatchFieldResult,
    moments: torch.Tensor,
) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    if result.z4_defocus_mm2 is None:
        raise RuntimeError("traced batch is missing continuous-OPD Z4 coefficients")
    z4_values = result.z4_defocus_mm2.reshape(-1)
    if z4_values.shape != moments.shape:
        raise ValueError("Z4 batch shape does not match traced cases")
    csf_values = csf_weighted_mtf_loss_batch(
        result.kernels, pixel_pitch_mm=result.pixel_pitch_mm
    )
    needs_astig = any(str(case["training_group"]) == "near_edge_astig" for case in cases)
    astig_by_zone = model.astig_A_by_zone() if needs_astig else {}
    metrics: list[torch.Tensor] = []
    names: list[str] = []
    astig_values: list[torch.Tensor] = []
    for index, case in enumerate(cases):
        group = str(case["training_group"])
        if group in {"far", "far_robustness"}:
            metrics.append(csf_values[index])
            names.append("csf_mtf_loss")
            astig_values.append(torch.zeros_like(metrics[-1]))
        elif group == "near_edge_astig":
            metrics.append(z4_values[index])
            names.append("z4_defocus_mm2_plus_astig_A_D")
            astig_values.append(astig_by_zone["near"])
        else:
            metrics.append(z4_values[index])
            names.append("z4_defocus_mm2")
            astig_values.append(torch.zeros_like(metrics[-1]))
    return (
        torch.stack(metrics),
        names,
        torch.stack(astig_values),
        z4_values,
        csf_values,
    )


def _evaluate_original_training_baseline_with_resume(
    model: MinimalOpticalModel,
    training_cases: Sequence[Mapping[str, Any]],
    *,
    progress_path: str | Path,
    identity_sha256: str,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    path = Path(progress_path)
    training_ids = [str(case["case_id"]) for case in training_cases]
    if path.is_file():
        payload = _load_identity_bound_torch(
            path,
            identity_sha256=identity_sha256,
            schema_version=BASELINE_PROGRESS_SCHEMA_VERSION,
            map_location=model.device,
        )
        if payload.get("training_case_ids") != training_ids:
            raise ValueError("baseline progress training case order/IDs changed")
        training_rows = [dict(row) for row in payload.get("training_rows", [])]
        status = str(payload.get("status", ""))
        if status not in {"training", "complete"}:
            raise ValueError("baseline progress status is invalid")
        _validate_baseline_progress_prefix(training_rows, training_ids, label="training")
        if int(payload.get("next_training_index", -1)) != len(training_rows):
            raise ValueError("baseline training next index is inconsistent")
        if status == "complete" and len(training_rows) != len(training_cases):
            raise ValueError("completed baseline progress is incomplete")
        if int(payload.get("control_count", -1)) != int(model.perturbation.control_shape[0]):
            raise ValueError("baseline progress control count changed")
        saved_model_state = payload.get("model_state")
        if not isinstance(saved_model_state, dict):
            raise ValueError("baseline progress model state is malformed")
        expected_model_state = model.perturbation.state_dict()
        if set(saved_model_state) != set(expected_model_state):
            raise ValueError("baseline progress model state keys changed")
        for name, expected in expected_model_state.items():
            saved = saved_model_state[name]
            if not torch.is_tensor(saved) or not torch.equal(
                saved.to(device=expected.device, dtype=expected.dtype), expected
            ):
                raise ValueError(
                    f"baseline progress is not bound to the Original PAL zero residual: {name}"
                )
        model.perturbation.load_state_dict(saved_model_state)
        _restore_rng_state(payload["rng_state"])
    else:
        training_rows = []

    batch_size = int(getattr(getattr(model, "config", None), "case_batch_size", 1))
    traced_cases = [
        case for case in training_cases
        if str(case["training_group"]) not in PERIPHERAL_GROUPS
    ]
    peripheral_cases = [
        case for case in training_cases
        if str(case["training_group"]) in PERIPHERAL_GROUPS
    ]
    if (
        len(training_rows) < len(traced_cases)
        and len(training_rows) % batch_size != 0
    ):
        raise ValueError("baseline progress ends inside a case batch and cannot be resumed")

    def save_progress(status: str) -> None:
        _torch_save_atomic(
            path,
            {
                "schema_version": BASELINE_PROGRESS_SCHEMA_VERSION,
                "identity_sha256": identity_sha256,
                "status": status,
                "control_count": int(model.perturbation.control_shape[0]),
                "model_state": copy.deepcopy(model.perturbation.state_dict()),
                "training_case_ids": training_ids,
                "next_training_index": len(training_rows),
                "training_rows": training_rows,
                "rng_state": _capture_rng_state(),
            },
        )

    if not path.is_file():
        save_progress("training")
    remaining = traced_cases[len(training_rows) :] if len(training_rows) < len(traced_cases) else []
    batches = _case_batches(remaining, batch_size)
    total_batches = (len(traced_cases) + batch_size - 1) // batch_size
    first_batch = len(training_rows) // batch_size
    for batch_index, batch in enumerate(batches, start=first_batch + 1):
        with torch.no_grad():
            result = _field_batch(model, batch)
            if not bool(torch.isfinite(result.kernels).all()) or bool((result.kernels < 0).any()):
                raise ValueError("baseline batch contains an invalid physical PSF")
            energy = result.kernels.sum(dim=(-2, -1))
            if bool((energy - 1.0).abs().max() > 1e-10):
                raise ValueError("baseline batch contains a non-unit PSF energy")
            moments = psf_second_moment_mm2_batch(
                result.kernels, pixel_pitch_mm=result.pixel_pitch_mm,
            )
            (
                loss_metrics,
                loss_metric_names,
                astig_values,
                z4_values,
                csf_values,
            ) = _loss_metrics_for_batch(model, batch, result, moments)
            scores = torch.ones_like(loss_metrics)
        batch_rows = _batch_rows(
            batch, result, moments, loss_metrics, loss_metric_names, None, scores,
            z4_defocus_mm2=z4_values,
            csf_mtf_loss=csf_values,
            astig_A_D=astig_values,
        )
        for row in batch_rows:
            row["group_loss"] = 1.0
        training_rows.extend(batch_rows)
        save_progress("training" if len(training_rows) < len(training_cases) else "complete")
        print(
            f"[pal-nurbs] baseline training batch {batch_index}/{total_batches} "
            f"cases {len(training_rows) - len(batch) + 1}-{len(training_rows)}/{len(training_cases)}",
            flush=True,
        )
        result_device = result.kernels.device
        del result, energy, moments, loss_metrics, scores
        _release_inactive_case_cuda_cache(result_device)
    if len(training_rows) >= len(traced_cases) and len(training_rows) < len(training_cases):
        completed_peripheral = len(training_rows) - len(traced_cases)
        with torch.no_grad():
            astig_by_zone = model.astig_A_by_zone()
        for case in peripheral_cases[completed_peripheral:]:
            group = str(case["training_group"])
            value = astig_by_zone[GROUP_TO_ZONE[group]]
            training_rows.append({
                **dict(case),
                "m2_mm2": 0.0,
                "astig_A_D": float(value.detach().cpu()),
                "z4_defocus_mm2": 0.0,
                "csf_mtf_loss": 0.0,
                "loss_metric": float(value.detach().cpu()),
                "loss_metric_name": "astig_A_D",
                "score": 1.0,
                "valid_fraction": 1.0,
                "valid_fraction_ratio": 1.0,
                "edge_fraction": 0.0,
                "group_loss": 1.0,
            })
        save_progress("complete")
    save_progress("complete")
    baseline_value, baseline_health = _summarize_training_baseline(training_rows)
    return baseline_value, training_rows, baseline_health


def _evaluate(
    model: MinimalOpticalModel, cases: Sequence[Mapping[str, Any]], baseline: Mapping[str, Mapping[str, float]] | None,
    *, with_grad: bool, baseline_valid: Mapping[str, float] | None = None,
    group_weights: Mapping[str, float] = DEFAULT_GROUP_WEIGHTS,
    near_edge_astig_A_weight: float = 0.10,
    progress_stage: str | None = None,
    progress_step: str | None = None,
    progress_learning_rate: float | None = None,
    progress_update: str = "PENDING",
    print_progress: bool = True,
) -> tuple[float, list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    group_values: dict[str, list[float]] = {}
    if not cases:
        raise ValueError("cannot evaluate an empty case set")
    if any("training_group" not in case for case in cases):
        raise ValueError("training evaluation requires a training_group on every case")
    group_counts: dict[str, int] = {}
    for case in cases:
        group = str(case["training_group"])
        group_counts[group] = group_counts.get(group, 0) + 1
    required = set(FUNCTIONAL_GROUPS + PERIPHERAL_GROUPS)
    actual = set(group_counts)
    if actual != required or any(group_counts[name] <= 0 for name in required):
        raise ValueError(
            "grouped training schema must contain exactly the ten groups "
            f"{sorted(required)}, got {sorted(actual)}"
        )
    weights = {str(name): float(value) for name, value in group_weights.items()}
    if set(weights) != required or not math.isclose(
        sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("group_weights must define the ten groups and sum to 1")
    minimum_ratio, maximum_edge = math.inf, 0.0
    batch_size = int(getattr(getattr(model, "config", None), "case_batch_size", 1))
    traced_cases = [
        case for case in cases
        if str(case["training_group"]) not in PERIPHERAL_GROUPS
    ]
    peripheral_cases = [
        case for case in cases
        if str(case["training_group"]) in PERIPHERAL_GROUPS
    ]
    batches = _case_batches(traced_cases, batch_size)
    for batch_index, batch in enumerate(batches, start=1):
        with torch.set_grad_enabled(with_grad):
            result = _field_batch(model, batch)
            energy = result.kernels.sum(dim=(-2, -1))
            if not bool(torch.isfinite(result.kernels).all()) or bool((result.kernels < 0).any()):
                raise ValueError("evaluation batch contains an invalid physical PSF")
            if bool((energy - 1.0).abs().max() > 1e-10):
                raise ValueError("evaluation batch contains a non-unit PSF energy")
            moments = psf_second_moment_mm2_batch(result.kernels, pixel_pitch_mm=result.pixel_pitch_mm)
            (
                loss_metrics,
                loss_metric_names,
                astig_values,
                z4_values,
                csf_values,
            ) = _loss_metrics_for_batch(model, batch, result, moments)
            if baseline is None:
                scores = torch.ones_like(loss_metrics)
            else:
                score_values: list[torch.Tensor] = []
                for index, case in enumerate(batch):
                    group = str(case["training_group"])
                    denominator = float(baseline[str(case["case_id"])]["loss_metric"])
                    if not math.isfinite(denominator) or denominator <= 0.0:
                        raise ValueError(
                            f"invalid Original PAL loss denominator for {case['case_id']}: {denominator}"
                        )
                    primary = loss_metrics[index] / denominator
                    if group == "near_edge_astig":
                        astig_denominator = float(
                            baseline[str(case["case_id"])]["astig_A_D"]
                        )
                        if not math.isfinite(astig_denominator) or astig_denominator <= 0.0:
                            raise ValueError(
                                f"invalid Original PAL astig denominator for {case['case_id']}"
                            )
                        blend = float(near_edge_astig_A_weight)
                        primary = (
                            (1.0 - blend) * primary
                            + blend * astig_values[index] / astig_denominator
                        )
                    score_values.append(primary)
                scores = torch.stack(score_values)
            coefficients = torch.as_tensor(
                [
                    weights[str(case["training_group"])]
                    / group_counts[str(case["training_group"])]
                    for case in batch
                ], device=scores.device, dtype=scores.dtype,
            )
            batch_loss = (scores * coefficients).sum()
            if with_grad:
                if not bool(scores.requires_grad):
                    raise RuntimeError("training scores are detached from NURBS parameters")
                if not batch_loss.requires_grad:
                    raise RuntimeError("batch training loss is detached from NURBS parameters")
                batch_loss.backward()
        for index, case in enumerate(batch):
            group = str(case["training_group"])
            group_values.setdefault(group, []).append(float(scores[index].detach().cpu()))
            vf = float(result.valid_fraction[index].detach().cpu())
            ratio = 1.0 if baseline_valid is None else vf / float(baseline_valid[str(case["case_id"])])
            minimum_ratio = min(minimum_ratio, ratio)
            maximum_edge = max(maximum_edge, float(result.edge_fraction[index].detach().cpu()))
        rows.extend(
            _batch_rows(
                batch,
                result,
                moments,
                loss_metrics,
                loss_metric_names,
                baseline_valid,
                scores,
                z4_defocus_mm2=z4_values,
                csf_mtf_loss=csf_values,
                astig_A_D=astig_values,
            )
        )
        completed = min(batch_index * batch_size, len(traced_cases))
        if print_progress and (batch_index % 8 == 0 or batch_index == len(batches)):
            batch_loss_value = float(batch_loss.detach().cpu())
            if progress_stage is None:
                print(
                    f"[pal-eval] batch={batch_index}/{len(batches)} "
                    f"cases={completed - len(batch) + 1}-{completed}/{len(traced_cases)} "
                    f"loss={batch_loss_value:.6g}", flush=True,
                )
            else:
                step_text = "-" if progress_step is None else progress_step
                lr_text = "-" if progress_learning_rate is None else f"{progress_learning_rate:.6g}"
                print(
                    f"[pal-train] stage={progress_stage} step={step_text} "
                    f"batch={batch_index}/{len(batches)} "
                    f"cases={completed - len(batch) + 1}-{completed}/{len(traced_cases)} "
                    f"loss={batch_loss_value:.6g} update={progress_update} lr={lr_text}",
                    flush=True,
                )
        result_device = result.kernels.device
        del result, energy, moments, loss_metrics, scores, coefficients, batch_loss
        _release_inactive_case_cuda_cache(result_device)
    with torch.set_grad_enabled(with_grad):
        astig_by_zone = model.astig_A_by_zone()
        peripheral_loss: torch.Tensor | None = None
        for case in peripheral_cases:
            group = str(case["training_group"])
            raw = astig_by_zone[GROUP_TO_ZONE[group]]
            if baseline is None:
                score = torch.ones_like(raw)
            else:
                denominator = float(baseline[str(case["case_id"])]["loss_metric"])
                if not math.isfinite(denominator) or denominator <= 0.0:
                    raise ValueError(
                        f"invalid Original PAL loss denominator for {case['case_id']}: {denominator}"
                    )
                score = raw / denominator
            weighted = score * (weights[group] / group_counts[group])
            peripheral_loss = weighted if peripheral_loss is None else peripheral_loss + weighted
            value = float(score.detach().cpu())
            group_values.setdefault(group, []).append(value)
            rows.append({
                **dict(case),
                "m2_mm2": 0.0,
                "astig_A_D": float(raw.detach().cpu()),
                "z4_defocus_mm2": 0.0,
                "csf_mtf_loss": 0.0,
                "loss_metric": float(raw.detach().cpu()),
                "loss_metric_name": "astig_A_D",
                "score": value,
                "valid_fraction": 1.0,
                "valid_fraction_ratio": 1.0,
                "edge_fraction": 0.0,
            })
        if with_grad:
            if peripheral_loss is None or not peripheral_loss.requires_grad:
                raise RuntimeError(
                    "surface-only peripheral loss is detached from NURBS parameters"
                )
            peripheral_loss.backward()
    group_losses = {
        name: sum(group_values[name]) / len(group_values[name])
        for name in FUNCTIONAL_GROUPS + PERIPHERAL_GROUPS
    }
    for row in rows:
        row["group_loss"] = float(group_losses[row["training_group"]])
    group_summary = {f"J_{name}": float(value) for name, value in group_losses.items()}
    group_summary["J_mid"] = sum(
        group_losses[name]
        for name in ("corridor_upper", "corridor_middle", "corridor_lower")
    ) / 3.0
    functional_weight = sum(weights[name] for name in FUNCTIONAL_GROUPS)
    peripheral_weight = sum(weights[name] for name in PERIPHERAL_GROUPS)
    functional = sum(
        weights[name] * group_losses[name] for name in FUNCTIONAL_GROUPS
    ) / functional_weight
    peripheral = sum(
        weights[name] * group_losses[name] for name in PERIPHERAL_GROUPS
    ) / peripheral_weight
    objective_value = sum(
        weights[name] * group_losses[name]
        for name in FUNCTIONAL_GROUPS + PERIPHERAL_GROUPS
    )
    group_summary.update({
        "J_functional": float(functional),
        "J_peripheral": float(peripheral),
        "J_total": float(objective_value),
    })
    objective_name = (
        "J=sum(group_weight*mean(normalized routed metric)); "
        "far=CSF-MTF loss, corridor/near=Z4 OPD mm^2, peripheral=surface A_D"
    )
    return float(objective_value), rows, {
        "minimum_valid_fraction_ratio": minimum_ratio,
        "maximum_edge_fraction": maximum_edge,
        "objective_name": objective_name,
        **group_summary,
    }


def _save_checkpoint(path: Path, module: FixedWeightNURBSPerturbation, **metadata: Any) -> None:
    _torch_save_atomic(
        path,
        {"control_count": module.control_shape[0], "state_dict": module.state_dict(), **metadata},
    )


def _joint_metric_fields(value: float, health: Mapping[str, float]) -> dict[str, float]:
    return {
        "J": float(value),
        "J_far": float(health["J_far"]),
        "J_mid": float(health["J_mid"]),
        "J_near": float(health["J_near"]),
        "J_peripheral": float(health["J_peripheral"]),
        "J_functional": float(health["J_functional"]),
    }


def _accumulate_startup_case_gradients(
    model: MinimalOpticalModel,
    module: FixedWeightNURBSPerturbation,
    config: MinimalConfig,
    cases: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    """Backpropagate the startup batch once while accumulating its total gradient."""

    if not cases:
        raise ValueError("startup gradient check requires at least one case")
    if not callable(getattr(model, "field_batch", None)):
        # This branch exists only for explicit scalar test doubles.  It still
        # requires the routed Z4 metric and never substitutes M2.
        for case in cases:
            result = model.field(case)
            if result.z4_defocus_mm2 is None:
                raise RuntimeError("startup scalar field is missing continuous-OPD Z4")
            case_loss = result.z4_defocus_mm2
            case_loss.backward()
            result_device = result.kernel.device
            del case_loss, result
            _release_inactive_case_cuda_cache(result_device)
        grad = module.inner_q.grad
        if grad is None or not bool(torch.isfinite(grad).all()) or int((grad.abs() > 0).sum()) < 2:
            raise RuntimeError("startup gradient check failed: fewer than two finite non-zero zp gradients")
        return grad.detach().clone()
    result = _field_batch(model, cases)
    if result.z4_defocus_mm2 is None:
        raise RuntimeError("startup batch is missing continuous-OPD Z4")
    result.z4_defocus_mm2.sum().backward()
    result_device = result.kernels.device
    del result
    _release_inactive_case_cuda_cache(result_device)
    grad = module.inner_q.grad
    if grad is None or not bool(torch.isfinite(grad).all()) or int((grad.abs() > 0).sum()) < 2:
        raise RuntimeError("startup gradient check failed: fewer than two finite non-zero zp gradients")
    return grad.detach().clone()


def _startup_cases(config: MinimalConfig) -> list[dict[str, Any]]:
    return [
        {
            "case_id": "startup_center",
            "distance_mm": float(config.intermediate_object_distance_mm),
            "field_x_deg": 0.0,
            "field_y_deg": 0.0,
        },
        {
            "case_id": "startup_edge",
            "distance_mm": float(config.intermediate_object_distance_mm),
            "field_x_deg": 40.0,
            "field_y_deg": 40.0,
        },
    ]


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    temporary = _temporary_sibling(path)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_atomic(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_training_log(path: Path, message: str) -> None:
    """Append one durable, human-readable PAL training progress record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{timestamp} {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _make_stage_resume_payload(
    *,
    identity_sha256: str,
    status: str,
    control_count: int,
    minimum_steps: int,
    maximum_steps: int,
    terminal_control_count: int,
    early_stopping_patience: int,
    relative_improvement_threshold: float,
    max_extra_terminal_stage_steps: int,
    no_improvement_attempts: int,
    stop_reason: str | None,
    module: FixedWeightNURBSPerturbation,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    completed_step: int,
    history: Sequence[Mapping[str, Any]],
    stage_initial: float,
    stage_initial_groups: Mapping[str, float],
    best: float,
    best_state: Mapping[str, Any],
    best_health: Mapping[str, Any],
    stage_summary: Mapping[str, Any] | None = None,
    optimizer_model_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in {"active", "completed"}:
        raise ValueError(f"invalid stage resume status: {status}")
    if int(completed_step) != len(history):
        raise ValueError("stage completed_step must equal history length")
    if status == "active" and stop_reason is not None:
        raise ValueError("active stage resume state cannot have a stop reason")
    if status == "completed" and not stop_reason:
        raise ValueError("completed stage resume state requires a stop reason")
    return {
        "schema_version": STAGE_RESUME_SCHEMA_VERSION,
        "identity_sha256": identity_sha256,
        "status": status,
        "control_count": int(control_count),
        "minimum_steps": int(minimum_steps),
        "maximum_steps": int(maximum_steps),
        "terminal_control_count": int(terminal_control_count),
        "early_stopping_patience": int(early_stopping_patience),
        "relative_improvement_threshold": float(relative_improvement_threshold),
        "max_extra_terminal_stage_steps": int(max_extra_terminal_stage_steps),
        "no_improvement_attempts": int(no_improvement_attempts),
        "stop_reason": stop_reason,
        "model_state": copy.deepcopy(module.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "optimizer_model_state": (
            None if optimizer_model_state is None else copy.deepcopy(dict(optimizer_model_state))
        ),
        "learning_rate": float(learning_rate),
        "completed_step": int(completed_step),
        "history": [dict(row) for row in history],
        "stage_initial": float(stage_initial),
        "stage_initial_groups": dict(stage_initial_groups),
        "best": float(best),
        "best_state": copy.deepcopy(dict(best_state)),
        "best_health": dict(best_health),
        "stage_summary": None if stage_summary is None else dict(stage_summary),
        "rng_state": _capture_rng_state(),
    }


def _load_stage_resume_state(
    path: str | Path,
    *,
    identity_sha256: str,
    control_count: int,
    minimum_steps: int,
    maximum_steps: int,
    terminal_control_count: int,
    early_stopping_patience: int,
    relative_improvement_threshold: float,
    max_extra_terminal_stage_steps: int,
    device: torch.device | str,
) -> dict[str, Any]:
    payload = _load_identity_bound_torch(
        path,
        identity_sha256=identity_sha256,
        schema_version=STAGE_RESUME_SCHEMA_VERSION,
        map_location=device,
    )
    if int(payload.get("control_count", -1)) != int(control_count):
        raise ValueError(f"stage control-count mismatch: {path}")
    expected_contract = {
        "minimum_steps": int(minimum_steps),
        "maximum_steps": int(maximum_steps),
        "terminal_control_count": int(terminal_control_count),
        "early_stopping_patience": int(early_stopping_patience),
        "max_extra_terminal_stage_steps": int(max_extra_terminal_stage_steps),
    }
    for name, expected in expected_contract.items():
        if int(payload.get(name, -1)) != expected:
            raise ValueError(f"stage {name.replace('_', '-')} mismatch: {path}")
    saved_threshold = float(payload.get("relative_improvement_threshold", math.nan))
    if saved_threshold != float(relative_improvement_threshold):
        raise ValueError(f"stage relative-improvement-threshold mismatch: {path}")
    status = str(payload.get("status", ""))
    if status not in {"active", "completed"}:
        raise ValueError(f"stage resume status is invalid: {path}")
    history = payload.get("history")
    if not isinstance(history, list):
        raise ValueError(f"stage history is malformed: {path}")
    completed_step = int(payload.get("completed_step", -1))
    if completed_step != len(history):
        raise ValueError(f"stage step/history mismatch: {path}")
    if completed_step < 0 or completed_step > int(maximum_steps):
        raise ValueError(f"stage completed step is outside its budget: {path}")
    learning_rate = float(payload.get("learning_rate", math.nan))
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError(f"stage learning rate is invalid: {path}")
    expected_steps = list(range(1, completed_step + 1))
    actual_steps = [int(row.get("step", -1)) for row in history]
    if actual_steps != expected_steps:
        raise ValueError(f"stage history steps are not contiguous: {path}")
    no_improvement_attempts = int(payload.get("no_improvement_attempts", -1))
    if no_improvement_attempts < 0:
        raise ValueError(f"stage patience counter is invalid: {path}")
    is_terminal_stage = int(control_count) == int(terminal_control_count)
    if is_terminal_stage:
        expected_no_improvement = 0
        for row in history:
            if bool(row.get("significant_improvement", False)):
                expected_no_improvement = 0
            else:
                expected_no_improvement += 1
            if int(row.get("no_improvement_attempts", -1)) != expected_no_improvement:
                raise ValueError(f"stage patience history is inconsistent: {path}")
        if no_improvement_attempts != expected_no_improvement:
            raise ValueError(f"stage patience counter/history mismatch: {path}")
    elif no_improvement_attempts != 0:
        raise ValueError(f"fixed stage cannot carry an early-stopping counter: {path}")
    stop_reason = payload.get("stop_reason")
    if status == "active" and stop_reason is not None:
        raise ValueError(f"active stage has a stop reason: {path}")
    if status == "completed":
        allowed_reasons = (
            {"minimum_completed"}
            if not is_terminal_stage
            else {"early_stopping", "learning_rate_floor", "max_extra_reached"}
        )
        if stop_reason not in allowed_reasons:
            raise ValueError(f"completed stage stop reason is invalid: {path}")
    if status == "completed" and not isinstance(payload.get("stage_summary"), dict):
        raise ValueError(f"completed stage lacks its summary: {path}")
    if status == "completed" and not isinstance(payload.get("optimizer_model_state"), dict):
        raise ValueError(f"completed stage lacks the model paired with its Adam state: {path}")
    for name in ("model_state", "optimizer_state", "best_state", "best_health", "rng_state"):
        if not isinstance(payload.get(name), dict):
            raise ValueError(f"stage resume state lacks {name}: {path}")
    if status == "completed":
        stage_summary = payload["stage_summary"]
        if (
            int(stage_summary.get("actual_steps", -1)) != completed_step
            or int(stage_summary.get("minimum_steps", -1)) != int(minimum_steps)
            or int(stage_summary.get("maximum_steps", -1)) != int(maximum_steps)
            or int(stage_summary.get("extra_steps", -1))
            != max(0, completed_step - int(minimum_steps))
            or stage_summary.get("stop_reason") != stop_reason
            or int(stage_summary.get("no_improvement_attempts", -1))
            != no_improvement_attempts
        ):
            raise ValueError(f"completed stage summary counters are inconsistent: {path}")
    return payload


_PARENT_CONFIG_FORK_FIELDS = frozenset({
    "output",
    "max_steps_7",
    "max_steps_11",
    "max_steps_19",
    "max_extra_terminal_stage_steps",
    "candidate_trace_import",
    "forward_qualification_import",
    "final_phase_qualification_import",
    "baseline_state_import",
    "parent_run",
    "start_stage",
})


def _load_torch_mapping(
    path: str | Path, *, map_location: torch.device | str,
) -> dict[str, Any]:
    source = Path(path)
    with source.open("rb") as handle:
        payload = torch.load(handle, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"state payload must be a mapping: {source}")
    return payload


def _validate_parent_identity_payload(payload: Mapping[str, Any]) -> str:
    saved = dict(payload)
    claimed = str(saved.pop("identity_sha256", ""))
    if not claimed or _canonical_json_sha256(saved) != claimed:
        raise ValueError("parent run_identity.json is malformed or has been modified")
    if int(saved.get("schema_version", -1)) not in PARENT_RUN_IDENTITY_SCHEMA_VERSIONS:
        raise ValueError("parent run identity schema is not supported for read-only import")
    if saved.get("method") != METHOD_NAME:
        raise ValueError("parent run uses a different PAL method identity")
    return claimed


def _assert_state_dict_equal(
    left: Mapping[str, Any], right: Mapping[str, Any], *, context: str,
) -> None:
    if set(left) != set(right):
        raise ValueError(f"{context} state_dict keys do not match")
    for name, left_value in left.items():
        right_value = right[name]
        if not torch.is_tensor(left_value) or not torch.is_tensor(right_value):
            raise ValueError(f"{context} state_dict entry is not a tensor: {name}")
        if not torch.equal(left_value.detach().cpu(), right_value.detach().cpu()):
            raise ValueError(f"{context} state_dict mismatch: {name}")


def _validate_parent_run_source(
    config: MinimalConfig, *, device: torch.device | str,
) -> dict[str, Any] | None:
    """Validate a completed parent run without mutating it and seal its lineage inputs."""
    if config.parent_run is None:
        return None
    if config.start_stage is None:
        raise ValueError("parent_run requires start_stage")
    parent = Path(config.parent_run).resolve()
    if not parent.is_dir():
        raise FileNotFoundError(f"parent run directory does not exist: {parent}")
    if parent == Path(config.output).resolve():
        raise ValueError("child output must differ from parent run directory")

    base_paths = {
        "run_identity": parent / "run_identity.json",
        "run_state": parent / "run_state.json",
        "summary": parent / "summary.json",
        "config": parent / "config.json",
        "case_layout_state": parent / "case_layout_state.json",
        "candidate_trace_progress": parent / "candidate_trace_progress.json",
        "forward_qualification_progress": parent / "forward_qualification_progress.json",
        "final_phase_qualification_progress": parent / "final_phase_qualification_progress.json",
        "baseline_state": parent / "baseline_state.pt",
    }
    missing = [f"{name}={path}" for name, path in base_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("parent run evidence is incomplete: " + ", ".join(missing))

    parent_identity_payload = _read_json(base_paths["run_identity"])
    parent_identity_sha256 = _validate_parent_identity_payload(parent_identity_payload)
    parent_state = _read_json(base_paths["run_state"])
    parent_summary = _read_json(base_paths["summary"])
    parent_config = _read_json(base_paths["config"])
    if parent_state.get("status") != "complete" or parent_state.get("phase") != "complete":
        raise ValueError("parent run must have complete run_state status and phase")
    if str(parent_state.get("identity_sha256", "")) != parent_identity_sha256:
        raise ValueError("parent run_state identity mismatch")
    if str(parent_summary.get("identity_sha256", "")) != parent_identity_sha256:
        raise ValueError("parent summary identity mismatch")

    child_config = asdict(config)
    compared_fields = sorted(set(child_config) - _PARENT_CONFIG_FORK_FIELDS)
    drift = [
        name for name in compared_fields
        if name not in parent_config or parent_config[name] != child_config[name]
    ]
    if drift:
        raise ValueError(
            "parent/child configuration mismatch outside the permitted fork fields: "
            + ", ".join(drift)
        )

    parent_budgets = {
        7: int(parent_config.get("max_steps_7", -1)),
        11: int(parent_config.get("max_steps_11", -1)),
        19: int(parent_config.get("max_steps_19", -1)),
    }
    active_parent_stages = [
        control for control in STAGE_LADDER if parent_budgets[control] > 0
    ]
    if not active_parent_stages:
        raise ValueError("parent run config has no positive training stage")
    parent_terminal = active_parent_stages[-1]
    if int(parent_summary.get("terminal_control_count", -1)) != parent_terminal:
        raise ValueError("parent terminal stage is inconsistent between config and summary")
    start_stage = int(config.start_stage)
    if start_stage < parent_terminal:
        raise ValueError(
            f"child start_stage {start_stage} cannot precede parent terminal stage "
            f"{parent_terminal}"
        )

    stage_rows = parent_summary.get("stages")
    if not isinstance(stage_rows, list):
        raise ValueError("parent summary stages are malformed")
    rows_by_control: dict[int, dict[str, Any]] = {}
    for row in stage_rows:
        if not isinstance(row, dict):
            raise ValueError("parent summary contains a malformed stage record")
        control = int(row.get("control_count", -1))
        if control not in STAGE_LADDER or control in rows_by_control:
            raise ValueError("parent summary contains invalid or duplicate stage records")
        rows_by_control[control] = dict(row)
    if set(rows_by_control) != set(STAGE_LADDER):
        raise ValueError("parent summary must contain exactly the complete 7/11/19 ladder")
    for control, row in rows_by_control.items():
        if bool(row.get("is_terminal_stage")) != (control == parent_terminal):
            raise ValueError("parent summary terminal stage flags are inconsistent")
        if int(row.get("minimum_steps", -1)) != parent_budgets[control]:
            raise ValueError(
                f"parent {control}x{control} summary minimum budget mismatch"
            )
        actual = int(row.get("actual_steps", -1))
        if actual < parent_budgets[control]:
            raise ValueError(
                f"parent {control}x{control} did not complete its minimum budget"
            )
        if control != parent_terminal and actual != parent_budgets[control]:
            raise ValueError(
                f"parent non-terminal {control}x{control} exceeded its fixed budget"
            )
    if int(parent_summary.get("actual_training_steps", -1)) != sum(
        int(row["actual_steps"]) for row in rows_by_control.values()
    ):
        raise ValueError("parent summary total actual_training_steps mismatch")
    if int(parent_summary.get("final_control_count", -1)) != STAGE_LADDER[-1]:
        raise ValueError("parent summary final control count must be 19")
    for control in STAGE_LADDER:
        if parent_terminal < control <= start_stage:
            if int(rows_by_control[control].get("actual_steps", -1)) != 0:
                raise ValueError(
                    "parent stages after terminal and through child start must be zero-budget "
                    f"exact refinements: {control}x{control}"
                )

    selected_final_path = parent / f"stage_{start_stage}x{start_stage}" / "final.pt"
    selected_resume_path = parent / f"stage_{start_stage}x{start_stage}" / "resume.pt"
    terminal_final_path = (
        parent / f"stage_{parent_terminal}x{parent_terminal}" / "final.pt"
    )
    lineage_stage_paths: dict[str, Path] = {}
    for control in STAGE_LADDER:
        if parent_terminal <= control <= start_stage:
            lineage_stage_paths[f"stage_{control}_final"] = (
                parent / f"stage_{control}x{control}" / "final.pt"
            )
            lineage_stage_paths[f"stage_{control}_resume"] = (
                parent / f"stage_{control}x{control}" / "resume.pt"
            )
    for name, path in (
        ("selected_stage_final", selected_final_path),
        ("selected_stage_resume", selected_resume_path),
        ("terminal_stage_final", terminal_final_path),
        *lineage_stage_paths.items(),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"parent {name} is missing: {path}")

    parent_minimum = parent_budgets[start_stage]
    parent_maximum = parent_minimum + (
        int(parent_config.get("max_extra_terminal_stage_steps", 0))
        if start_stage == parent_terminal else 0
    )
    selected_resume = _load_stage_resume_state(
        selected_resume_path,
        identity_sha256=parent_identity_sha256,
        control_count=start_stage,
        minimum_steps=parent_minimum,
        maximum_steps=parent_maximum,
        terminal_control_count=parent_terminal,
        early_stopping_patience=int(parent_config["early_stopping_patience"]),
        relative_improvement_threshold=float(
            parent_config["relative_improvement_threshold"]
        ),
        max_extra_terminal_stage_steps=int(
            parent_config["max_extra_terminal_stage_steps"]
        ),
        device=device,
    )
    if selected_resume.get("status") != "completed":
        raise ValueError("parent selected stage resume state must be complete")
    selected_checkpoint = _load_torch_mapping(
        selected_final_path, map_location=device,
    )
    if int(selected_checkpoint.get("control_count", -1)) != start_stage:
        raise ValueError("parent selected checkpoint control_count mismatch")
    if str(selected_checkpoint.get("identity_sha256", "")) != parent_identity_sha256:
        raise ValueError("parent selected checkpoint identity mismatch")
    selected_state = selected_checkpoint.get("state_dict")
    if not isinstance(selected_state, dict):
        raise ValueError("parent selected checkpoint state_dict is malformed")
    _assert_state_dict_equal(
        selected_state, selected_resume["best_state"],
        context="parent final/best",
    )
    _assert_state_dict_equal(
        selected_state, selected_resume["model_state"],
        context="parent final/resume model",
    )
    if int(selected_checkpoint.get("step", -1)) != int(
        rows_by_control[start_stage].get("actual_steps", -2)
    ):
        raise ValueError("parent selected checkpoint step disagrees with summary")

    terminal_checkpoint = _load_torch_mapping(
        terminal_final_path, map_location=device,
    )
    if int(terminal_checkpoint.get("control_count", -1)) != parent_terminal:
        raise ValueError("parent terminal checkpoint control_count mismatch")
    if str(terminal_checkpoint.get("identity_sha256", "")) != parent_identity_sha256:
        raise ValueError("parent terminal checkpoint identity mismatch")
    terminal_state = terminal_checkpoint.get("state_dict")
    if not isinstance(terminal_state, dict):
        raise ValueError("parent terminal checkpoint state_dict is malformed")
    previous_module: FixedWeightNURBSPerturbation | None = None
    for control in STAGE_LADDER:
        if not parent_terminal <= control <= start_stage:
            continue
        stage_minimum = parent_budgets[control]
        stage_maximum = stage_minimum + (
            int(parent_config.get("max_extra_terminal_stage_steps", 0))
            if control == parent_terminal else 0
        )
        stage_resume = _load_stage_resume_state(
            lineage_stage_paths[f"stage_{control}_resume"],
            identity_sha256=parent_identity_sha256,
            control_count=control,
            minimum_steps=stage_minimum,
            maximum_steps=stage_maximum,
            terminal_control_count=parent_terminal,
            early_stopping_patience=int(parent_config["early_stopping_patience"]),
            relative_improvement_threshold=float(
                parent_config["relative_improvement_threshold"]
            ),
            max_extra_terminal_stage_steps=int(
                parent_config["max_extra_terminal_stage_steps"]
            ),
            device=device,
        )
        if stage_resume.get("status") != "completed":
            raise ValueError(f"parent {control}x{control} resume state must be complete")
        stage_checkpoint = _load_torch_mapping(
            lineage_stage_paths[f"stage_{control}_final"], map_location=device,
        )
        if int(stage_checkpoint.get("control_count", -1)) != control:
            raise ValueError(f"parent {control}x{control} checkpoint control_count mismatch")
        if str(stage_checkpoint.get("identity_sha256", "")) != parent_identity_sha256:
            raise ValueError(f"parent {control}x{control} checkpoint identity mismatch")
        stage_state = stage_checkpoint.get("state_dict")
        if not isinstance(stage_state, dict):
            raise ValueError(f"parent {control}x{control} checkpoint state_dict is malformed")
        _assert_state_dict_equal(
            stage_state, stage_resume["best_state"],
            context=f"parent {control}x{control} final/best",
        )
        _assert_state_dict_equal(
            stage_state, stage_resume["model_state"],
            context=f"parent {control}x{control} final/resume model",
        )
        if int(stage_checkpoint.get("step", -1)) != int(
            rows_by_control[control].get("actual_steps", -2)
        ):
            raise ValueError(
                f"parent {control}x{control} checkpoint step disagrees with summary"
            )
        current_module = FixedWeightNURBSPerturbation(
            control, device=device, dtype=torch.float64,
        )
        current_module.load_state_dict(stage_state)
        if previous_module is not None:
            expected = previous_module.refined(control)
            refinement_audit = audit_exact_refinement(
                expected, current_module, samples=129,
            )
            if max(
                refinement_audit.max_abs_sag_mm,
                refinement_audit.max_abs_first_derivative,
                refinement_audit.max_abs_second_derivative_per_mm,
            ) > 1.0e-10:
                raise ValueError(
                    f"parent {control}x{control} is not an exact zero-budget refinement"
                )
        previous_module = current_module

    case_layout_state = _read_json(base_paths["case_layout_state"])
    if str(case_layout_state.get("identity_sha256", "")) != parent_identity_sha256:
        raise ValueError("parent case layout state identity mismatch")
    candidate_progress = _read_json(base_paths["candidate_trace_progress"])
    if candidate_progress.get("status") != "complete":
        raise ValueError("parent candidate trace progress must be complete")
    for name in ("forward_qualification_progress", "final_phase_qualification_progress"):
        progress = _read_json(base_paths[name])
        if progress.get("status") != "complete":
            raise ValueError(f"parent {name} must be complete")
    baseline_state = _load_torch_mapping(base_paths["baseline_state"], map_location=device)
    if int(baseline_state.get("schema_version", -1)) != BASELINE_STATE_SCHEMA_VERSION:
        raise ValueError("parent baseline state schema mismatch")
    if str(baseline_state.get("identity_sha256", "")) != parent_identity_sha256:
        raise ValueError("parent baseline state identity mismatch")

    artifact_paths = {
        **base_paths,
        **lineage_stage_paths,
        "selected_stage_final": selected_final_path,
        "selected_stage_resume": selected_resume_path,
        "terminal_stage_final": terminal_final_path,
    }
    evidence_hashes = {
        name: _sha256_file(path) for name, path in sorted(artifact_paths.items())
    }
    parent_steps_by_stage = {
        str(control): int(rows_by_control.get(control, {}).get("actual_steps", 0))
        for control in STAGE_LADDER
    }
    return {
        "root": parent,
        "identity_sha256": parent_identity_sha256,
        "summary": parent_summary,
        "config": parent_config,
        "terminal_control_count": parent_terminal,
        "start_stage": start_stage,
        "selected_checkpoint": selected_checkpoint,
        "selected_resume": selected_resume,
        "parent_steps_by_stage": parent_steps_by_stage,
        "artifact_paths": artifact_paths,
        "identity_metadata": {
            "source_run_identity_sha256": parent_identity_sha256,
            "source_identity_schema_version": int(
                parent_identity_payload["schema_version"]
            ),
            "source_terminal_control_count": parent_terminal,
            "child_start_stage": start_stage,
            "selected_checkpoint_path": selected_final_path.relative_to(parent).as_posix(),
            "selected_checkpoint_sha256": evidence_hashes["selected_stage_final"],
            "selected_resume_sha256": evidence_hashes["selected_stage_resume"],
            "optimizer_policy": "parent_best_fresh_adam",
            "parent_actual_training_steps_by_stage": parent_steps_by_stage,
            "parent_stage_history": [dict(row) for row in stage_rows],
            "parent_summary_sha256": evidence_hashes["summary"],
            "evidence_sha256": evidence_hashes,
        },
    }


def _activate_parent_best(
    model: MinimalOpticalModel,
    parent_context: Mapping[str, Any],
    *,
    device: torch.device,
) -> FixedWeightNURBSPerturbation:
    start_stage = int(parent_context["start_stage"])
    selected_checkpoint = parent_context["selected_checkpoint"]
    selected_state = selected_checkpoint.get("state_dict")
    if not isinstance(selected_state, dict):
        raise ValueError("validated parent checkpoint lost its state_dict")
    module = FixedWeightNURBSPerturbation(
        start_stage, device=device, dtype=torch.float64,
    )
    module.load_state_dict(selected_state)
    model.perturbation = module
    for template, _ in model._templates.values():
        template.back_surface.perturbation = module
    for cached_system, _ in model._cache.values():
        cached_system.back_surface.perturbation = module
    _restore_rng_state(parent_context["selected_resume"]["rng_state"])
    return module


def _restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    state: Mapping[str, Any],
    module: FixedWeightNURBSPerturbation,
) -> None:
    optimizer.load_state_dict(dict(state))
    device = module.inner_q.device
    parameters = optimizer.param_groups[0]["params"] if len(optimizer.param_groups) == 1 else []
    if len(parameters) != 1 or parameters[0] is not module.inner_q:
        raise ValueError("restored Adam parameter group is not bound to the active NURBS parameter")
    for parameter, parameter_state in optimizer.state.items():
        if parameter is not module.inner_q:
            raise ValueError("restored Adam state is bound to a different parameter")
        for name, value in list(parameter_state.items()):
            if torch.is_tensor(value):
                parameter_state[name] = value.to(device=device)
                if parameter_state[name].device != device:
                    raise RuntimeError(f"Adam state tensor {name} is on the wrong device")


def _run_bound(
    config: MinimalConfig,
    *,
    output: Path,
    identity: Mapping[str, Any],
    device: torch.device,
    prior_elapsed: float,
    session_start: float,
) -> Path:
    identity_sha256 = str(identity["identity_sha256"])
    parent_context = _validate_parent_run_source(config, device=device)

    def write_state(phase: str, **details: Any) -> None:
        _write_run_state(
            output,
            identity_sha256=identity_sha256,
            status="running",
            phase=phase,
            elapsed_seconds=prior_elapsed + time.time() - session_start,
            **details,
        )

    training_log_path = output / "training.log"

    def log_progress(message: str) -> None:
        line = f"[pal-train] {message}"
        _append_training_log(training_log_path, line)
        print(line, flush=True)

    summary_path = output / "summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if str(summary.get("identity_sha256", "")) != identity_sha256:
            raise ValueError("completed summary identity mismatch")
        _write_run_state(
            output,
            identity_sha256=identity_sha256,
            status="complete",
            phase="complete",
            elapsed_seconds=float(summary["runtime_seconds"]),
        )
        return output

    write_state("load_original_pal")
    base_sag, power_config, zones = load_pal(config, device)
    module = FixedWeightNURBSPerturbation(7, device=device, dtype=torch.float64)
    model = MinimalOpticalModel(config, module)
    model.set_prescription_context(base_sag, power_config, zones)
    write_state("case_layout")
    training_cases = _prepare_or_load_case_layout(
        config, output, model, identity_sha256=identity_sha256,
    )
    objective_options = {
        "group_weights": config.group_weights,
        "near_edge_astig_A_weight": config.near_edge_astig_A_weight,
    }

    baseline_state_path = output / "baseline_state.pt"
    baseline_import = (
        Path(config.parent_run) / "baseline_state.pt"
        if config.parent_run is not None else
        (None if config.baseline_state_import is None else Path(config.baseline_state_import))
    )
    if baseline_state_path.is_file() or baseline_import is not None:
        imported_baseline = not baseline_state_path.is_file()
        write_state("import_baseline" if imported_baseline else "restore_baseline")
        if imported_baseline:
            baseline_state = _load_torch_mapping(baseline_import, map_location=device)
            if int(baseline_state.get("schema_version", -1)) != BASELINE_STATE_SCHEMA_VERSION:
                raise ValueError("baseline state import schema mismatch")
            source_identity = str(baseline_state.get("identity_sha256", ""))
            if not source_identity:
                raise ValueError("baseline state import lacks its source identity")
        else:
            baseline_state = _load_identity_bound_torch(
                baseline_state_path,
                identity_sha256=identity_sha256,
                schema_version=BASELINE_STATE_SCHEMA_VERSION,
                map_location=device,
            )
        if int(baseline_state.get("control_count", -1)) != 7:
            raise ValueError("baseline state must use the 7x7 zero-residual module")
        expected_training_ids = [str(case["case_id"]) for case in training_cases]
        if baseline_state.get("training_case_ids") != expected_training_ids:
            raise ValueError("baseline state training case IDs do not match the case layout")
        saved_baseline_model = baseline_state.get("model_state")
        expected_baseline_model = module.state_dict()
        if not isinstance(saved_baseline_model, dict) or set(saved_baseline_model) != set(expected_baseline_model):
            raise ValueError("baseline state model payload is malformed")
        for name, expected in expected_baseline_model.items():
            saved = saved_baseline_model[name]
            if not torch.is_tensor(saved) or not torch.equal(
                saved.to(device=expected.device, dtype=expected.dtype), expected
            ):
                raise ValueError(f"baseline state is not Original PAL zero residual: {name}")
        module.load_state_dict(saved_baseline_model)
        baseline_value = float(baseline_state["baseline_value"])
        baseline_rows = [dict(row) for row in baseline_state["baseline_rows"]]
        baseline_health = dict(baseline_state["baseline_health"])
        baseline_power = {
            "P_far_D": float(baseline_state["baseline_power"]["P_far_D"]),
            "ADD_D": float(baseline_state["baseline_power"]["ADD_D"]),
        }
        if not all(math.isfinite(value) for value in baseline_power.values()):
            raise ValueError("baseline state contains non-finite prescription metrics")
        _restore_rng_state(baseline_state["rng_state"])
        if imported_baseline:
            imported_payload = dict(baseline_state)
            imported_payload.update(
                {
                    "identity_sha256": identity_sha256,
                    "import_source_identity_sha256": source_identity,
                    "import_source_file_sha256": _sha256_file(baseline_import),
                }
            )
            _torch_save_atomic(baseline_state_path, imported_payload)
    else:
        write_state("startup_gradient_check")
        baseline_power_tensors = prescription_metrics(base_sag, power_config, zones)
        baseline_power = {
            "P_far_D": float(baseline_power_tensors["P_far_D"].detach().cpu()),
            "ADD_D": float(baseline_power_tensors["ADD_D"].detach().cpu()),
        }
        # 保留原有最小启动检查：中心/边缘真实 PSF 与非零有限梯度。
        startup_cases = _startup_cases(config)
        _accumulate_startup_case_gradients(model, module, config, startup_cases)
        module.zero_grad(set_to_none=True)

        write_state("baseline_training_cases")
        (
            baseline_value,
            baseline_rows,
            baseline_health,
        ) = _evaluate_original_training_baseline_with_resume(
            model,
            training_cases,
            progress_path=output / "baseline_progress.pt",
            identity_sha256=identity_sha256,
        )
        baseline_state = {
            "schema_version": BASELINE_STATE_SCHEMA_VERSION,
            "identity_sha256": identity_sha256,
            "control_count": 7,
            "model_state": copy.deepcopy(module.state_dict()),
            "training_case_ids": [str(case["case_id"]) for case in training_cases],
            "baseline_value": float(baseline_value),
            "baseline_rows": baseline_rows,
            "baseline_health": baseline_health,
            "baseline_power": baseline_power,
            "rng_state": _capture_rng_state(),
        }
        _torch_save_atomic(baseline_state_path, baseline_state)

    baseline = {str(row["case_id"]): row for row in baseline_rows}
    baseline_valid = {str(row["case_id"]): float(row["valid_fraction"]) for row in baseline_rows}
    _torch_save_atomic(
        output / "baseline.pt",
        {
            "identity_sha256": identity_sha256,
            **_joint_metric_fields(baseline_value, baseline_health),
            "objective_name": baseline_health["objective_name"],
            "cases": baseline_rows,
            "P_far_D": float(baseline_power["P_far_D"]),
            "ADD_D": float(baseline_power["ADD_D"]),
        },
    )
    cache_audit = _retain_training_cache(model, training_cases)
    write_state("baseline_complete", cache_audit=cache_audit)
    log_progress(
        f"stage=baseline step=complete "
        f"batch={(len(training_cases) + config.case_batch_size - 1) // config.case_batch_size}/"
        f"{(len(training_cases) + config.case_batch_size - 1) // config.case_batch_size} "
        f"loss={baseline_value:.6g} update=INITIAL lr=-"
    )

    stage_summaries: list[dict[str, Any]] = []
    stage_specs = _training_stage_specs(config)
    if parent_context is not None:
        start_stage = int(parent_context["start_stage"])
        stage_specs = tuple(
            spec for spec in stage_specs if int(spec[0]) >= start_stage
        )
        selected_checkpoint = dict(parent_context["selected_checkpoint"])
        module = _activate_parent_best(model, parent_context, device=device)
        write_state(
            "parent_stage_loaded",
            parent_identity_sha256=parent_context["identity_sha256"],
            parent_terminal_control_count=parent_context["terminal_control_count"],
            start_stage=start_stage,
            selected_checkpoint_sha256=identity["parent_lineage"][
                "selected_checkpoint_sha256"
            ],
        )
        log_progress(
            f"stage={start_stage}x{start_stage} step=parent-best batch=complete "
            f"loss={float(selected_checkpoint['J']):.6g} update=PARENT_IMPORT lr=RESET"
        )
    terminal_control_count = next(
        control_count
        for control_count, _, _, is_terminal_stage in stage_specs
        if is_terminal_stage
    )
    for stage_index, (
        control_count,
        minimum_steps,
        maximum_steps,
        is_terminal_stage,
    ) in enumerate(stage_specs):
        resume_path = output / f"stage_{control_count}x{control_count}" / "resume.pt"
        later = [
            output / f"stage_{later_count}x{later_count}" / "resume.pt"
            for later_count, _, _, _ in stage_specs[stage_index + 1 :]
        ]
        if not resume_path.is_file():
            if any(path.is_file() for path in later):
                raise ValueError(f"stage resume sequence is incomplete before {control_count}x{control_count}")

        if control_count != module.control_shape[0]:
            coarse = module
            module = module.refined(control_count)
            audit = audit_exact_refinement(coarse, module, samples=129)
            if max(
                audit.max_abs_sag_mm,
                audit.max_abs_first_derivative,
                audit.max_abs_second_derivative_per_mm,
            ) > 1e-10:
                raise RuntimeError("NURBS refinement changed the physical surface")
            model.perturbation = module
            for template, _ in model._templates.values():
                template.back_surface.perturbation = module
            for cached_system, _ in model._cache.values():
                cached_system.back_surface.perturbation = module

        stage_dir = output / f"stage_{control_count}x{control_count}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            stage_dir / "config.json",
            {
                **asdict(config),
                "identity_sha256": identity_sha256,
                "control_count": control_count,
                "terminal_control_count": terminal_control_count,
                "is_terminal_stage": is_terminal_stage,
                "minimum_steps": minimum_steps,
                "maximum_steps": maximum_steps,
            },
        )
        optimizer = torch.optim.Adam([module.inner_q], lr=config.learning_rate)

        if resume_path.is_file():
            stage_state = _load_stage_resume_state(
                resume_path,
                identity_sha256=identity_sha256,
                control_count=control_count,
                minimum_steps=minimum_steps,
                maximum_steps=maximum_steps,
                terminal_control_count=terminal_control_count,
                early_stopping_patience=config.early_stopping_patience,
                relative_improvement_threshold=config.relative_improvement_threshold,
                max_extra_terminal_stage_steps=config.max_extra_terminal_stage_steps,
                device=device,
            )
            module.load_state_dict(stage_state["model_state"])
            _restore_optimizer_state(optimizer, stage_state["optimizer_state"], module)
            lr = float(stage_state["learning_rate"])
            history = [dict(row) for row in stage_state["history"]]
            if history:
                _write_history(stage_dir / "history.csv", history)
            completed_step = int(stage_state["completed_step"])
            stage_initial = float(stage_state["stage_initial"])
            stage_initial_groups = dict(stage_state["stage_initial_groups"])
            best = float(stage_state["best"])
            best_state = copy.deepcopy(stage_state["best_state"])
            best_health = dict(stage_state["best_health"])
            no_improvement_attempts = int(stage_state["no_improvement_attempts"])
            _restore_rng_state(stage_state["rng_state"])
            if stage_state["status"] != "completed" and any(path.is_file() for path in later):
                raise ValueError(
                    f"later-stage state exists while {control_count}x{control_count} is still active"
                )
            if stage_state["status"] == "completed":
                stage_summary = dict(stage_state["stage_summary"])
                stage_summaries.append(stage_summary)
                write_state(
                    "stage_complete",
                    control_count=control_count,
                    completed_step=completed_step,
                )
                log_progress(
                    f"stage={control_count}x{control_count} "
                    f"step={completed_step}/{maximum_steps} batch=complete "
                    f"loss={best:.6g} update=RESUME lr={lr:.6g}"
                    + (
                        f" minimum={minimum_steps} rel="
                        f"{float(history[-1]['relative_best_improvement']) if history else 0.0:.3e} "
                        f"patience={no_improvement_attempts}/"
                        f"{config.early_stopping_patience}"
                        if is_terminal_stage else ""
                    )
                )
                continue
        else:
            write_state("stage_initialize", control_count=control_count, completed_step=0)
            if control_count == 7 and parent_context is None:
                # The 7x7 module is verified above to be the exact zero-residual
                # Original PAL state. Every case denominator is that same physical
                # baseline, so all group objectives and J are exactly 1 without an
                # additional 109-case evaluation pass.
                current = float(baseline_value)
                health = dict(baseline_health)
                if abs(current - 1.0) > 1.0e-15 or any(
                    abs(float(health[f"J_{name}"]) - 1.0) > 1.0e-15
                    for name in FUNCTIONAL_GROUPS + PERIPHERAL_GROUPS
                ):
                    raise ValueError("Original PAL baseline is not the exact 7x7 initial objective")
            else:
                current, _, health = _evaluate(
                    model,
                    training_cases,
                    baseline,
                    with_grad=False,
                    baseline_valid=baseline_valid,
                    **objective_options,
                    progress_stage=f"{control_count}x{control_count}",
                    progress_step=f"0/{maximum_steps}",
                    progress_learning_rate=config.learning_rate,
                    progress_update="INITIAL",
                )
            stage_initial = current
            stage_initial_groups = _joint_metric_fields(current, health)
            best, best_state, best_health = (
                current,
                copy.deepcopy(module.state_dict()),
                dict(health),
            )
            history = []
            completed_step = 0
            no_improvement_attempts = 0
            lr = config.learning_rate
            _save_checkpoint(
                stage_dir / "initial.pt",
                module,
                identity_sha256=identity_sha256,
                **_joint_metric_fields(current, health),
                step=0,
            )
            _torch_save_atomic(
                resume_path,
                _make_stage_resume_payload(
                    identity_sha256=identity_sha256,
                    status="active",
                    control_count=control_count,
                    minimum_steps=minimum_steps,
                    maximum_steps=maximum_steps,
                    terminal_control_count=terminal_control_count,
                    early_stopping_patience=config.early_stopping_patience,
                    relative_improvement_threshold=config.relative_improvement_threshold,
                    max_extra_terminal_stage_steps=config.max_extra_terminal_stage_steps,
                    no_improvement_attempts=no_improvement_attempts,
                    stop_reason=None,
                    module=module,
                    optimizer=optimizer,
                    learning_rate=lr,
                    completed_step=0,
                    history=history,
                    stage_initial=stage_initial,
                    stage_initial_groups=stage_initial_groups,
                    best=best,
                    best_state=best_state,
                    best_health=best_health,
                ),
            )
            minimum_log = f" minimum={minimum_steps}" if is_terminal_stage else ""
            log_progress(
                f"stage={control_count}x{control_count} step=0/{maximum_steps}"
                f"{minimum_log} batch=complete loss={stage_initial:.6g} "
                f"update=INITIAL lr={lr:.6g}"
                + (
                    f" rel=0 patience={no_improvement_attempts}/"
                    f"{config.early_stopping_patience}"
                    if is_terminal_stage else ""
                )
            )

        stop_reason = _stage_boundary_stop_reason(
            control_count=control_count,
            is_terminal_stage=is_terminal_stage,
            completed_step=completed_step,
            minimum_steps=minimum_steps,
            maximum_steps=maximum_steps,
            learning_rate=lr,
            minimum_learning_rate=config.minimum_learning_rate,
            no_improvement_attempts=no_improvement_attempts,
            early_stopping_patience=config.early_stopping_patience,
        )
        if stop_reason == "minimum_not_reached":
            log_progress(
                f"stage={control_count}x{control_count} step={completed_step}/{maximum_steps} "
                f"batch=complete loss={best:.6g} update=MINIMUM_NOT_REACHED lr={lr:.6g}"
            )
            write_state(
                "minimum_not_reached",
                control_count=control_count,
                completed_step=completed_step,
                minimum_steps=minimum_steps,
                learning_rate=lr,
            )
            raise MinimumTrainingBudgetError(
                f"{control_count}x{control_count} learning rate fell below the floor "
                f"at attempt {completed_step}/{minimum_steps}"
            )

        for step in range(completed_step + 1, int(maximum_steps) + 1):
            if stop_reason is not None:
                break
            if control_count == 19 and (not math.isfinite(best) or best <= 0.0):
                raise RuntimeError("19x19 best objective must be finite and positive")
            minimum_log = f" minimum={minimum_steps}" if is_terminal_stage else ""
            patience_log = (
                f" patience={no_improvement_attempts}/{config.early_stopping_patience}"
                if is_terminal_stage else ""
            )
            if stop_reason is None:
                log_progress(
                    f"stage={control_count}x{control_count} step={step}/{maximum_steps}"
                    f"{minimum_log} "
                    f"batch=0/{(len(training_cases) + config.case_batch_size - 1) // config.case_batch_size} "
                    f"loss=- update=PENDING lr={lr:.6g}{patience_log}"
                )
                module.zero_grad(set_to_none=True)
                optimizer.param_groups[0]["lr"] = lr
                current, _, health = _evaluate(
                    model,
                    training_cases,
                    baseline,
                    with_grad=True,
                    baseline_valid=baseline_valid,
                    **objective_options,
                    progress_stage=f"{control_count}x{control_count}",
                    progress_step=f"{step}/{maximum_steps}",
                    progress_learning_rate=lr,
                )
                smooth_loss_value = 0.0
                if control_count == 19 and config.smooth_lambda > 0.0:
                    smooth_loss = laplacian_regularizer(module)
                    (float(config.smooth_lambda) * smooth_loss).backward()
                    smooth_loss_value = float(smooth_loss.detach().cpu())
                if module.inner_q.grad is None or not bool(torch.isfinite(module.inner_q.grad).all()):
                    raise RuntimeError("non-finite Adam gradient")
                parameter_state = module.inner_q.detach().clone()
                optimizer_state = copy.deepcopy(optimizer.state_dict())
                step_coord = torch.linspace(-40.0, 40.0, 161, device=device, dtype=torch.float64)
                step_yy, step_xx = torch.meshgrid(step_coord, step_coord, indexing="ij")
                old_surface = module.delta_raw(step_xx, step_yy).detach().clone()
                accepted = False
                reason = "backtracks_exhausted"
                candidate = current
                for backtrack in range(config.max_backtracks + 1):
                    with torch.no_grad():
                        module.inner_q.copy_(parameter_state)
                    optimizer.load_state_dict(optimizer_state)
                    trial_lr = lr * (0.5 ** backtrack)
                    optimizer.param_groups[0]["lr"] = trial_lr
                    optimizer.step()
                    with torch.no_grad():
                        module.inner_q.clamp_(-1.0, 1.0)
                    step_sag = float(
                        (module.delta_raw(step_xx, step_yy) - old_surface)
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                    )
                    if step_sag > config.step_sag_limit_mm:
                        reason = "step_sag"
                        continue
                    coord = torch.linspace(
                        -power_config.semi_diameter_mm,
                        power_config.semi_diameter_mm,
                        int(base_sag.shape[0]),
                        device=device,
                        dtype=torch.float64,
                    )
                    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
                    power = prescription_metrics(
                        base_sag + module.delta_raw(xx, yy),
                        power_config,
                        zones,
                        baseline_sag=base_sag,
                    )
                    far_change = abs(
                        float(power["P_far_D"].detach().cpu())
                        - float(baseline_power["P_far_D"])
                    )
                    add_change = abs(
                        float(power["ADD_D"].detach().cpu())
                        - float(baseline_power["ADD_D"])
                    )
                    guard_power_change = float(
                        power["lower_edge_max_abs_power_change_D"].detach().cpu()
                    )
                    guard_astig_change = float(
                        power["lower_edge_max_abs_astig_change_D"].detach().cpu()
                    )
                    if (
                        far_change > config.far_tolerance_D
                        or add_change > config.add_tolerance_D
                        or guard_power_change > config.lower_edge_power_tolerance_D
                        or guard_astig_change > config.lower_edge_astig_tolerance_D
                    ):
                        reason = "prescription"
                        continue
                    candidate, candidate_rows, candidate_health = _evaluate(
                        model,
                        training_cases,
                        baseline,
                        with_grad=False,
                        baseline_valid=baseline_valid,
                        **objective_options,
                        progress_stage=f"{control_count}x{control_count}",
                        progress_step=f"{step}/{maximum_steps}",
                        progress_learning_rate=trial_lr,
                        progress_update="TRIAL",
                        print_progress=False,
                    )
                    candidate_smooth_loss = (
                        float(laplacian_regularizer(module).detach().cpu())
                        if control_count == 19 and config.smooth_lambda > 0.0
                        else 0.0
                    )
                    current_total_loss = (
                        current + float(config.smooth_lambda) * smooth_loss_value
                    )
                    candidate_total_loss = (
                        candidate
                        + float(config.smooth_lambda) * candidate_smooth_loss
                    )
                    if (
                        candidate_health["minimum_valid_fraction_ratio"]
                        < config.minimum_valid_fraction_ratio
                    ):
                        reason = "health"
                        continue
                    if (
                        not math.isfinite(candidate_total_loss)
                        or candidate_total_loss > current_total_loss
                    ):
                        reason = "objective"
                        continue
                    accepted, reason, lr = (
                        True,
                        "accepted",
                        min(config.learning_rate, trial_lr * 1.1),
                    )
                    health = candidate_health
                    break
                if not accepted:
                    with torch.no_grad():
                        module.inner_q.copy_(parameter_state)
                    optimizer.load_state_dict(optimizer_state)
                    lr *= 0.5

                best_before = best
                if is_terminal_stage:
                    (
                        best_refreshed,
                        relative_best_improvement,
                        significant_improvement,
                        no_improvement_attempts,
                    ) = _early_stopping_observation(
                        best_before=best_before,
                        candidate=candidate,
                        accepted=accepted,
                        threshold=config.relative_improvement_threshold,
                        no_improvement_attempts=no_improvement_attempts,
                    )
                else:
                    best_refreshed = bool(accepted and candidate < best_before)
                    relative_best_improvement = (
                        (best_before - candidate) / abs(best_before)
                        if best_refreshed else 0.0
                    )
                    significant_improvement = False
                if best_refreshed:
                    best, best_state, best_health = (
                        candidate,
                        copy.deepcopy(module.state_dict()),
                        dict(candidate_health),
                    )
                    _save_checkpoint(
                        stage_dir / "best.pt",
                        module,
                        identity_sha256=identity_sha256,
                        **_joint_metric_fields(best, best_health),
                        step=step,
                    )
                history.append(
                    {
                        "step": step,
                        "accepted": accepted,
                        "reason": reason,
                        **_joint_metric_fields(candidate if accepted else current, health),
                        "best_J": best,
                        "best_J_far": best_health["J_far"],
                        "best_J_mid": best_health["J_mid"],
                        "best_J_near": best_health["J_near"],
                        "best_J_peripheral": best_health["J_peripheral"],
                        "best_J_functional": best_health["J_functional"],
                        "learning_rate": lr,
                        "minimum_valid_fraction_ratio": health["minimum_valid_fraction_ratio"],
                        "maximum_edge_fraction": health["maximum_edge_fraction"],
                        "smooth_laplacian": smooth_loss_value,
                        "smooth_weighted_loss": float(config.smooth_lambda) * smooth_loss_value,
                        "minimum_steps": minimum_steps,
                        "maximum_steps": maximum_steps,
                        "actual_steps": step,
                        "extra_steps": max(0, step - minimum_steps),
                        "is_extra_step": bool(is_terminal_stage and step > minimum_steps),
                        "best_refreshed": best_refreshed,
                        "relative_best_improvement": relative_best_improvement,
                        "significant_improvement": significant_improvement,
                        "no_improvement_attempts": no_improvement_attempts,
                        "stop_reason": "",
                    }
                )
                completed_step = step
                _torch_save_atomic(
                    resume_path,
                    _make_stage_resume_payload(
                        identity_sha256=identity_sha256,
                        status="active",
                        control_count=control_count,
                        minimum_steps=minimum_steps,
                        maximum_steps=maximum_steps,
                        terminal_control_count=terminal_control_count,
                        early_stopping_patience=config.early_stopping_patience,
                        relative_improvement_threshold=config.relative_improvement_threshold,
                        max_extra_terminal_stage_steps=(
                            config.max_extra_terminal_stage_steps
                        ),
                        no_improvement_attempts=no_improvement_attempts,
                        stop_reason=None,
                        module=module,
                        optimizer=optimizer,
                        learning_rate=lr,
                        completed_step=completed_step,
                        history=history,
                        stage_initial=stage_initial,
                        stage_initial_groups=stage_initial_groups,
                        best=best,
                        best_state=best_state,
                        best_health=best_health,
                    ),
                )
                _write_history(stage_dir / "history.csv", history)
                rel_log = (
                    f" rel={relative_best_improvement:.3e} "
                    f"patience={no_improvement_attempts}/{config.early_stopping_patience}"
                    if is_terminal_stage else ""
                )
                log_progress(
                    f"stage={control_count}x{control_count} step={step}/{maximum_steps}"
                    f"{minimum_log} "
                    f"batch={(len(training_cases) + config.case_batch_size - 1) // config.case_batch_size}/"
                    f"{(len(training_cases) + config.case_batch_size - 1) // config.case_batch_size} "
                    f"loss={candidate if accepted else current:.6g} "
                    f"update={'ACCEPT' if accepted else 'REJECT:' + reason} lr={lr:.6g}"
                    f"{rel_log}"
                )
                write_state(
                    "stage_training",
                    control_count=control_count,
                    completed_step=completed_step,
                    minimum_steps=minimum_steps,
                    maximum_steps=maximum_steps,
                    learning_rate=lr,
                    no_improvement_attempts=no_improvement_attempts,
                )
                stop_reason = _stage_boundary_stop_reason(
                    control_count=control_count,
                    is_terminal_stage=is_terminal_stage,
                    completed_step=completed_step,
                    minimum_steps=minimum_steps,
                    maximum_steps=maximum_steps,
                    learning_rate=lr,
                    minimum_learning_rate=config.minimum_learning_rate,
                    no_improvement_attempts=no_improvement_attempts,
                    early_stopping_patience=config.early_stopping_patience,
                )
                if stop_reason is not None:
                    history[-1]["stop_reason"] = stop_reason
                    _write_history(stage_dir / "history.csv", history)
                if stop_reason == "minimum_not_reached":
                    log_progress(
                        f"stage={control_count}x{control_count} step={step}/{maximum_steps} "
                        f"batch=complete loss={best:.6g} update=MINIMUM_NOT_REACHED lr={lr:.6g}"
                    )
                    write_state(
                        "minimum_not_reached",
                        control_count=control_count,
                        completed_step=completed_step,
                        minimum_steps=minimum_steps,
                        learning_rate=lr,
                    )
                    raise MinimumTrainingBudgetError(
                        f"{control_count}x{control_count} learning rate fell below the floor "
                        f"at attempt {completed_step}/{minimum_steps}"
                    )
                if stop_reason is not None:
                    break

        if stop_reason is None:
            raise RuntimeError(f"{control_count}x{control_count} exhausted without a stop reason")
        if history and history[-1].get("stop_reason") != stop_reason:
            history[-1]["stop_reason"] = stop_reason
            _write_history(stage_dir / "history.csv", history)
        if is_terminal_stage:
            terminal_update = {
                "early_stopping": "EARLY_STOPPING",
                "learning_rate_floor": "LEARNING_RATE_FLOOR",
                "max_extra_reached": "MAX_EXTRA_REACHED",
            }[stop_reason]
            terminal_relative = (
                float(history[-1]["relative_best_improvement"]) if history else 0.0
            )
            log_progress(
                f"stage={control_count}x{control_count} "
                f"step={completed_step}/{maximum_steps} minimum={minimum_steps} "
                f"batch=complete loss={best:.6g} update={terminal_update} lr={lr:.6g} "
                f"rel={terminal_relative:.3e} "
                f"patience={no_improvement_attempts}/{config.early_stopping_patience}"
            )

        if history:
            _write_history(stage_dir / "history.csv", history)
        optimizer_model_state = copy.deepcopy(module.state_dict())
        module.load_state_dict(best_state)
        _save_checkpoint(
            stage_dir / "final.pt",
            module,
            identity_sha256=identity_sha256,
            **_joint_metric_fields(best, best_health),
            step=len(history),
        )
        if not (stage_dir / "best.pt").exists():
            _save_checkpoint(
                stage_dir / "best.pt",
                module,
                identity_sha256=identity_sha256,
                **_joint_metric_fields(best, best_health),
                step=0,
            )
        improvement = (stage_initial - best) / stage_initial
        stage_summary = {
            "control_count": control_count,
            "is_terminal_stage": is_terminal_stage,
            "initial_J": stage_initial,
            "initial_groups": stage_initial_groups,
            "best_J": best,
            "best_groups": _joint_metric_fields(best, best_health),
            "relative_stage_improvement": improvement,
            "steps": len(history),
            "minimum_steps": minimum_steps,
            "maximum_steps": maximum_steps,
            "actual_steps": len(history),
            "extra_steps": max(0, len(history) - minimum_steps),
            "early_stopping_patience": config.early_stopping_patience,
            "relative_improvement_threshold": config.relative_improvement_threshold,
            "no_improvement_attempts": no_improvement_attempts,
            "stop_reason": stop_reason,
        }
        _torch_save_atomic(
            resume_path,
            _make_stage_resume_payload(
                identity_sha256=identity_sha256,
                status="completed",
                control_count=control_count,
                minimum_steps=minimum_steps,
                maximum_steps=maximum_steps,
                terminal_control_count=terminal_control_count,
                early_stopping_patience=config.early_stopping_patience,
                relative_improvement_threshold=config.relative_improvement_threshold,
                max_extra_terminal_stage_steps=config.max_extra_terminal_stage_steps,
                no_improvement_attempts=no_improvement_attempts,
                stop_reason=stop_reason,
                module=module,
                optimizer=optimizer,
                learning_rate=lr,
                completed_step=len(history),
                history=history,
                stage_initial=stage_initial,
                stage_initial_groups=stage_initial_groups,
                best=best,
                best_state=best_state,
                best_health=best_health,
                stage_summary=stage_summary,
                optimizer_model_state=optimizer_model_state,
            ),
        )
        stage_summaries.append(stage_summary)
        write_state(
            "stage_complete",
            control_count=control_count,
            completed_step=len(history),
        )
        log_progress(
            f"stage={control_count}x{control_count} step={len(history)}/{maximum_steps} "
            f"batch=complete loss={best:.6g} update=STAGE_COMPLETE lr={lr:.6g} "
            f"reason={stop_reason}"
            + (
                f" minimum={minimum_steps} rel="
                f"{float(history[-1]['relative_best_improvement']) if history else 0.0:.3e} "
                f"patience={no_improvement_attempts}/{config.early_stopping_patience}"
                if is_terminal_stage else ""
            )
        )

    write_state("final_training_evaluation")
    final_value, _, final_health = _evaluate(
        model,
        training_cases,
        baseline,
        with_grad=False,
        baseline_valid=baseline_valid,
        **objective_options,
    )
    coord = torch.linspace(
        -power_config.semi_diameter_mm,
        power_config.semi_diameter_mm,
        int(base_sag.shape[0]),
        device=device,
        dtype=torch.float64,
    )
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    delta = module.delta_raw(xx, yy)
    final_power = prescription_metrics(
        base_sag + delta, power_config, zones, baseline_sag=base_sag
    )
    final_p_far_D = float(final_power["P_far_D"])
    final_add_D = float(final_power["ADD_D"])
    max_abs_sag_delta_mm = float(delta.abs().max())
    metric_names = ("far", "mid", "near", "peripheral", "functional")
    improvement_by_group = {
        name: 100.0 * (1.0 - float(final_health[f"J_{name}"]))
        for name in metric_names
    }
    runtime_seconds = prior_elapsed + time.time() - session_start
    terminal_stage_summary = next(
        stage for stage in stage_summaries if bool(stage["is_terminal_stage"])
    )
    summary = {
        "identity_sha256": identity_sha256,
        "baseline_J": baseline_value,
        "objective_name": final_health["objective_name"],
        "group_weights": dict(config.group_weights),
        "near_edge_astig_A_weight": config.near_edge_astig_A_weight,
        "smooth_lambda": config.smooth_lambda,
        "training_case_count": len(training_cases),
        "training_groups": {
            name: sum(row["training_group"] == name for row in training_cases)
            for name in TRAINING_GROUP_COUNTS
        },
        "stages": stage_summaries,
        "minimum_training_steps": (
            config.max_steps_7 + config.max_steps_11 + config.max_steps_19
        ),
        "actual_training_steps": sum(int(stage["actual_steps"]) for stage in stage_summaries),
        "terminal_control_count": terminal_control_count,
        "extra_terminal_stage_steps": int(terminal_stage_summary["extra_steps"]),
        "training_stop_reason": str(terminal_stage_summary["stop_reason"]),
        "early_stopping_patience": config.early_stopping_patience,
        "relative_improvement_threshold": config.relative_improvement_threshold,
        "max_extra_terminal_stage_steps": config.max_extra_terminal_stage_steps,
        "final_J": final_value,
        "final_metrics": _joint_metric_fields(final_value, final_health),
        "improvement_percent": 100 * (1 - final_value / baseline_value),
        "improvement_percent_by_group": improvement_by_group,
        "final_control_count": module.control_shape[0],
        "P_far_D": final_p_far_D,
        "ADD_D": final_add_D,
        "max_abs_sag_delta_mm": max_abs_sag_delta_mm,
        "P_far_change_D": final_p_far_D - float(baseline_power["P_far_D"]),
        "ADD_change_D": final_add_D - float(baseline_power["ADD_D"]),
        "lower_edge_max_abs_power_change_D": float(
            final_power["lower_edge_max_abs_power_change_D"]
        ),
        "lower_edge_max_abs_astig_change_D": float(
            final_power["lower_edge_max_abs_astig_change_D"]
        ),
        "runtime_seconds": runtime_seconds,
        "training_log": "training.log",
        "trace_psf_exception": False,
        "health": final_health,
    }
    if parent_context is not None:
        child_steps_by_stage = {
            str(control): next(
                (
                    int(stage["actual_steps"])
                    for stage in stage_summaries
                    if int(stage["control_count"]) == control
                ),
                0,
            )
            for control in STAGE_LADDER
        }
        parent_steps_by_stage = dict(parent_context["parent_steps_by_stage"])
        lineage_steps_by_stage = {
            str(control): (
                int(parent_steps_by_stage[str(control)])
                + int(child_steps_by_stage[str(control)])
            )
            for control in STAGE_LADDER
        }
        stage_lineage = [
            {
                "control_count": control,
                "parent_actual_steps": int(parent_steps_by_stage[str(control)]),
                "child_actual_steps": int(child_steps_by_stage[str(control)]),
                "lineage_actual_steps": int(lineage_steps_by_stage[str(control)]),
            }
            for control in STAGE_LADDER
        ]
        summary.update({
            "parent_lineage": dict(identity["parent_lineage"]),
            "child_start_stage": int(parent_context["start_stage"]),
            "optimizer_policy": "parent_best_fresh_adam",
            "parent_stage_history": [
                dict(stage) for stage in parent_context["summary"]["stages"]
            ],
            "child_stage_history": [dict(stage) for stage in stage_summaries],
            "parent_actual_training_steps_by_stage": parent_steps_by_stage,
            "child_actual_training_steps_by_stage": child_steps_by_stage,
            "lineage_actual_training_steps_by_stage": lineage_steps_by_stage,
            "stage_lineage": stage_lineage,
            "source_parent_run_unchanged": True,
        })
    _write_json_atomic(summary_path, summary)
    _write_run_state(
        output,
        identity_sha256=identity_sha256,
        status="complete",
        phase="complete",
        elapsed_seconds=runtime_seconds,
    )
    return output


def run(config: MinimalConfig, *, resume: bool = False) -> Path:
    _training_stage_specs(config)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output, identity = _open_run_directory(config, resume=resume)
    _write_run_config(output, config)
    prior_elapsed = _elapsed_seconds(output)
    session_start = time.time()
    try:
        return _run_bound(
            config,
            output=output,
            identity=identity,
            device=device,
            prior_elapsed=prior_elapsed,
            session_start=session_start,
        )
    except BaseException as exc:
        phase = "interrupted"
        state_path = output / "run_state.json"
        if state_path.is_file():
            try:
                phase = str(_read_json(state_path).get("phase", phase))
            except Exception:
                phase = "interrupted"
        failed = isinstance(exc, MinimumTrainingBudgetError)
        event = "FAILED" if failed else "INTERRUPTED"
        _append_training_log(
            output / "training.log",
            f"[pal-train] {event} stage_phase={phase} error={type(exc).__name__}: {exc}",
        )
        _write_run_state(
            output,
            identity_sha256=str(identity["identity_sha256"]),
            status="failed" if failed else "interrupted",
            phase=phase,
            elapsed_seconds=prior_elapsed + time.time() - session_start,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise
