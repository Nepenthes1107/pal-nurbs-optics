"""PAL-NURBS 最简优化链：真实追迹 + 分区联合 M2 + Adam 小步更新。"""
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
from dataclasses import asdict, dataclass
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
    PERIPHERAL_BAND_COUNTS,
    PERIPHERAL_GROUPS,
    TOTAL_TRAINING_CASES,
    TRAINING_GROUP_COUNTS,
    _sha256_file,
    classify_partition_point,
    generate_dense_candidate_fields,
    select_training_cases,
    trace_candidate_fields,
    write_preoptimization_artifacts,
)
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
    trace_system_to_image_with_phase,
    _snell,
)


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
    requested_np: int = 1024
    fft_size_px: int = 512
    kernel_size_px: int = 130
    pupil_radius_mm: float | None = None
    learning_rate: float = 2.0e-3
    minimum_learning_rate: float = 1.0e-6
    max_steps_7: int = 10
    max_steps_11: int = 10
    max_steps_19: int = 10
    max_backtracks: int = 8
    step_sag_limit_mm: float = 2.0e-3
    far_tolerance_D: float = 0.2
    add_tolerance_D: float = 0.3
    minimum_valid_fraction_ratio: float = 0.5
    minimum_stage_relative_improvement: float = 1.0e-3
    seed: int = 42
    far_object_distance_mm: float = 100000.0
    intermediate_object_distance_mm: float = 2000.0
    near_object_distance_mm: float = 500.0
    candidate_field_min_deg: float = -55.0
    candidate_field_max_deg: float = 55.0
    candidate_field_step_deg: float = 1.0
    zone_boundary_safety_mm: float = 1.5
    corridor_zone_boundary_safety_mm: float = 1.0
    aperture_edge_safety_mm: float = 1.5
    functional_objective_weight: float = 0.85
    peripheral_objective_weight: float = 0.15
    candidate_trace_import: str | None = None
    forward_qualification_import: str | None = None
    final_phase_qualification_import: str | None = None
    baseline_state_import: str | None = None


RUN_IDENTITY_SCHEMA_VERSION = 2
CASE_LAYOUT_STATE_SCHEMA_VERSION = 4
BASELINE_STATE_SCHEMA_VERSION = 2
BASELINE_PROGRESS_SCHEMA_VERSION = 2
STAGE_RESUME_SCHEMA_VERSION = 1
RUN_STATE_SCHEMA_VERSION = 1
STAGE_LADDER = (7, 11, 19)
FORWARD_POOL_MULTIPLIER = 4
FORWARD_POOL_GROUP_COUNTS = {
    "far": TRAINING_GROUP_COUNTS["far"] * FORWARD_POOL_MULTIPLIER,
    "intermediate": TRAINING_GROUP_COUNTS["intermediate"],
    "near": TRAINING_GROUP_COUNTS["near"] * FORWARD_POOL_MULTIPLIER,
    # Keep the already audited 52-case pool per side.  This is the smallest
    # pool that contains the complete 16/16/20 band strata while supporting
    # the final 5/5/6 selection without increasing qualification compute.
    "peripheral_left": 52,
    "peripheral_right": 52,
}
FORWARD_POOL_PERIPHERAL_BAND_COUNTS = {
    "upper": 16, "middle": 16, "lower": 20,
}


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


def _identity_input_paths(config: MinimalConfig) -> dict[str, Path]:
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
    inputs = {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in sorted(_identity_input_paths(config).items())
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
    return {**body, "identity_sha256": _canonical_json_sha256(body)}


def _validate_identity_payload(payload: Mapping[str, Any]) -> str:
    saved = dict(payload)
    claimed = str(saved.pop("identity_sha256", ""))
    actual = _canonical_json_sha256(saved)
    if not claimed or claimed != actual:
        raise ValueError("run_identity.json is malformed or has been modified")
    if int(saved.get("schema_version", -1)) != RUN_IDENTITY_SCHEMA_VERSION:
        raise ValueError("unsupported run identity schema")
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
    payload = torch.load(Path(path), map_location=map_location)
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
    required_cases = [*cases, *extra_cases]
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
                system, rays, phase_reference="biot_reference_sphere"
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
        trace = trace_system_to_image_with_phase(system, rays, phase_reference="biot_reference_sphere")
        if not bool(trace.valid.any()):
            raise RuntimeError(f"no valid rays for {case['case_id']}")
        fft = torch_fft_psf_from_phase(
            trace.phase_rad, trace.valid, sample_count=self.sample_count,
            psf_size_px=self.config.fft_size_px, remove_piston=True, remove_tilt=True,
        )
        if system.physical_fft_pixel_pitch_mm is None:
            raise RuntimeError("missing physical FFT pixel pitch")
        kernel = crop_resize_fft_psf(
            fft.psf, pixel_pitch_mm=float(system.physical_fft_pixel_pitch_mm),
            size_reference_mm=self.size_reference_mm[distance], output_size_px=self.config.kernel_size_px,
        )
        return FieldResult(
            kernel=kernel, valid_fraction=trace.valid.to(torch.float64).mean(),
            pixel_pitch_mm=self.size_reference_mm[distance] / self.config.kernel_size_px,
            edge_fraction=_edge_fraction(kernel), valid_mask=trace.valid,
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
    return {"power_D": power, "astigmatism_D": astig, "valid": valid}


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


def prescription_metrics(sag: torch.Tensor, power_config: PALPowerConfig, zones: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    maps = torch_averfang_maps(sag, power_config)
    valid = maps["valid"]
    far = zones["far_reference"] & valid
    near = zones["near_reference"] & valid
    monitored = zones["monitored"] & valid
    if not bool(far.any() and near.any() and monitored.any()):
        raise ValueError("P_far/ADD masks have no valid power samples")
    pfar = maps["power_D"][far].mean()
    add = maps["power_D"][near].mean() - pfar
    return {"P_far_D": pfar, "ADD_D": add, "astig_mean_D": maps["astigmatism_D"][monitored].mean()}


def build_joint_training_cases(
    traced_candidates: Sequence[Mapping[str, Any]], config: MinimalConfig,
    *, corridor_y_min_mm: float, corridor_y_max_mm: float,
    group_counts: Mapping[str, int] = TRAINING_GROUP_COUNTS,
    peripheral_band_counts: Mapping[str, int] = PERIPHERAL_BAND_COUNTS,
) -> list[dict[str, Any]]:
    """Build the fixed 80-case contract from traced dense-field candidates."""
    return select_training_cases(
        traced_candidates,
        far_object_distance_mm=config.far_object_distance_mm,
        intermediate_object_distance_mm=config.intermediate_object_distance_mm,
        near_object_distance_mm=config.near_object_distance_mm,
        corridor_y_min_mm=corridor_y_min_mm,
        corridor_y_max_mm=corridor_y_max_mm,
        group_counts=group_counts,
        peripheral_band_counts=peripheral_band_counts,
    )


def _trace_preoptimization_case_geometry(
    model: MinimalOpticalModel,
    cases: Sequence[Mapping[str, Any]],
    *,
    zones_json: str | Path,
    reference_distance_mm: float = 100000.0,
) -> list[dict[str, Any]]:
    zones_payload = json.loads(Path(zones_json).read_text(encoding="utf-8-sig"))
    result: list[dict[str, Any]] = []
    for case in cases:
        fx = float(case["field_x_deg"])
        fy = float(case["field_y_deg"])
        reference_x, reference_y = model.reference_rear_intersection(
            reference_distance_mm, fx, fy
        )
        case_x, case_y = model.reference_rear_intersection(float(case["distance_mm"]), fx, fy)
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


def _prepare_case_layout(
    config: MinimalConfig, output: Path, model: MinimalOpticalModel
) -> list[dict[str, Any]]:
    zones_payload = json.loads(Path(config.zones_json).read_text(encoding="utf-8-sig"))
    corridor_stats = dict(dict(zones_payload.get("statistics", {})).get("corridor", {}))
    corridor_range = corridor_stats.get("physical_y_range_mm")
    if not isinstance(corridor_range, list) or len(corridor_range) != 2:
        raise ValueError("zones.json must declare corridor physical_y_range_mm")
    candidate_fields = generate_dense_candidate_fields(
        field_min_deg=config.candidate_field_min_deg,
        field_max_deg=config.candidate_field_max_deg,
        field_step_deg=config.candidate_field_step_deg,
    )
    candidate_progress_path = output / "candidate_trace_progress.json"
    if config.candidate_trace_import is not None and not candidate_progress_path.exists():
        source = Path(config.candidate_trace_import)
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
            "excel_path": str(Path(config.excel).resolve()),
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
        group_counts=FORWARD_POOL_GROUP_COUNTS,
        peripheral_band_counts=FORWARD_POOL_PERIPHERAL_BAND_COUNTS,
    )
    pool_identity = _canonical_json_sha256([
        {
            "candidate_id": str(case["candidate_id"]),
            "distance_mm": float(case["distance_mm"]),
            "field_x_deg": float(case["field_x_deg"]),
            "field_y_deg": float(case["field_y_deg"]),
        }
        for case in qualification_pool
    ])
    qualification_progress_path = output / "forward_qualification_progress.json"
    _import_complete_pool_progress(
        source_path=config.forward_qualification_import,
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
    attempted = {str(row["candidate_id"]): row for row in pool_attempts}
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
        prior = attempted.get(candidate_id)
        if prior is None:
            diagnostic_stream = io.StringIO()
            try:
                with contextlib.redirect_stdout(diagnostic_stream), contextlib.redirect_stderr(diagnostic_stream):
                    aiming_audit = model.validate_training_case_wfno(case)
                prior = {
                    "candidate_id": candidate_id,
                    "pool_case_id": str(case["case_id"]),
                    "training_group": str(case["training_group"]),
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
                    "training_group": str(case["training_group"]),
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
            attempted[candidate_id] = prior
            save_qualification_progress("running")
            if pool_index % 20 == 0 or pool_index == len(qualification_pool):
                print(
                    f"forward WFNO pool: {pool_index}/{len(qualification_pool)}, "
                    f"failures={sum(row['status'] == 'failed' for row in pool_attempts)}",
                    flush=True,
                )
        source = traced_by_id[candidate_id]
        source["forward_wfno_status"] = str(prior["status"])
        if prior["status"] == "ok":
            case["forward_wfno_validation"] = {
                name: prior[name]
                for name in ("physical_fft_pixel_pitch_mm",)
            }
        else:
            case["eligible"] = False
            source["forward_wfno_error_type"] = str(prior["error_type"])
            source["forward_wfno_error"] = str(prior["error"])
    save_qualification_progress("complete")

    qualified_pool = [dict(case) for case in qualification_pool if bool(case.get("eligible"))]
    forward_attempts: list[dict[str, Any]] = []
    phase_progress_path = output / "final_phase_qualification_progress.json"
    _import_complete_pool_progress(
        source_path=config.final_phase_qualification_import,
        destination_path=phase_progress_path,
        pool_identity_sha256=pool_identity,
        progress_name="final phase qualification progress",
    )
    if phase_progress_path.is_file():
        saved = _read_json(phase_progress_path)
        if saved.get("schema_version") != 1 or saved.get("pool_identity_sha256") != pool_identity:
            raise ValueError("final phase qualification progress identity mismatch")
        rows = saved.get("forward_attempts")
        if not isinstance(rows, list):
            raise ValueError("final phase qualification progress rows are malformed")
        forward_attempts = [dict(row) for row in rows]
        pool_by_id = {str(row["candidate_id"]): row for row in qualified_pool}
        for attempt in forward_attempts:
            pool_case = pool_by_id.get(str(attempt["candidate_id"]))
            if pool_case is None:
                raise ValueError("final phase qualification progress references a foreign candidate")
            if attempt["status"] == "ok":
                pool_case["forward_training_status"] = "ok"
                pool_case["forward_training_validation"] = {
                    name: attempt[name]
                    for name in (
                        "ray_count", "valid_ray_count", "valid_fraction",
                        "physical_fft_pixel_pitch_mm",
                    )
                }
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
        )
        training_cases = _trace_preoptimization_case_geometry(
            model, training_cases, zones_json=config.zones_json,
            reference_distance_mm=config.far_object_distance_mm,
        )
        failed_ids: set[str] = set()
        for case in training_cases:
            candidate_id = str(case["candidate_id"])
            pool_case = next(
                row for row in qualified_pool
                if str(row.get("candidate_id")) == candidate_id
            )
            if pool_case.get("forward_training_status") == "ok":
                case["forward_training_validation"] = dict(
                    pool_case["forward_training_validation"]
                )
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
                    "status": "ok",
                    **dict(audit),
                })
                save_phase_progress("running")
            except Exception as exc:
                diagnostic = diagnostic_stream.getvalue()
                pool_case["eligible"] = False
                failed_ids.add(candidate_id)
                forward_attempts.append({
                    "validation_round": validation_round,
                    "candidate_id": candidate_id,
                    "case_id": str(case["case_id"]),
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
        extra_cases=_startup_cases(),
    )
    _write_json_atomic(
        output / "preoptimization" / "forward_qualification_audit.json",
        {
            "schema_version": 1,
            "contract": (
                "fixed 260-case regional FPS pool + exact BIOT_vis field-dependent WFNO + "
                "final regional FPS + complete pre-FFT phase trace"
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
        reference_distance_mm=config.far_object_distance_mm,
        sampling_contract={
            "method": (
                "dense field -> Original PAL rear trace -> mask/clearance -> "
                "fixed 260-case region-wise lens-plane FPS pool -> exact aiming/WFNO qualification -> "
                "final region-wise FPS -> complete pre-FFT phase qualification"
            ),
            "field_grid_deg": {
                "min": config.candidate_field_min_deg,
                "max": config.candidate_field_max_deg,
                "step": config.candidate_field_step_deg,
            },
            "zone_boundary_safety_mm": {
                "default": config.zone_boundary_safety_mm,
                "corridor": config.corridor_zone_boundary_safety_mm,
            },
            "aperture_edge_safety_mm": config.aperture_edge_safety_mm,
            "object_distance_mm": {
                "far": config.far_object_distance_mm,
                "intermediate": config.intermediate_object_distance_mm,
                "near": config.near_object_distance_mm,
            },
            "peripheral_band_distance_mm": {
                "upper": config.far_object_distance_mm,
                "middle": config.intermediate_object_distance_mm,
                "lower": config.near_object_distance_mm,
            },
            "group_counts": TRAINING_GROUP_COUNTS,
            "corridor": "4 vertical layers x 3 FPS points; centre and both side boundaries",
            "peripheral": "16 exact field-mirror pairs per side; upper/middle/lower = 5/5/6; upper preserves the outer-x anchor",
            "peripheral_distance": "upper=far, middle=intermediate, lower=near; one distance per case",
            "forward_qualification": (
                "fixed 260-case spatial pool; exact BIOT_vis field-dependent WFNO on the pool; "
                f"complete pre-FFT phase trace on final {TOTAL_TRAINING_CASES} cases"
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


def _training_baseline_case_row(
    model: MinimalOpticalModel, case: Mapping[str, Any],
) -> dict[str, Any]:
    with torch.no_grad():
        result = model.field(case)
        energy = result.kernel.sum()
        if abs(float(energy.detach().cpu()) - 1.0) > 1e-10:
            raise ValueError(f"non-unit PSF energy for {case['case_id']}")
        moment = psf_second_moment_mm2(
            result.kernel, pixel_pitch_mm=result.pixel_pitch_mm,
        )
    row = {
        **dict(case),
        "m2_mm2": float(moment.detach().cpu()),
        "score": 1.0,
        "valid_fraction": float(result.valid_fraction.detach().cpu()),
        "valid_fraction_ratio": 1.0,
        "edge_fraction": float(result.edge_fraction.detach().cpu()),
    }
    row["group_loss"] = 1.0
    result_device = result.kernel.device
    del result, energy, moment
    _release_inactive_case_cuda_cache(result_device)
    return row


def _summarize_training_baseline(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, dict[str, Any]]:
    required = set(FUNCTIONAL_GROUPS + PERIPHERAL_GROUPS)
    actual = {str(row.get("training_group")) for row in rows}
    if actual != required:
        raise ValueError(f"baseline rows do not contain exactly the five groups: {sorted(actual)}")
    maximum_edge = max(float(row["edge_fraction"]) for row in rows)
    health: dict[str, Any] = {
        "minimum_valid_fraction_ratio": 1.0,
        "maximum_edge_fraction": maximum_edge,
        "objective_name": "J=0.85*J_functional+0.15*J_peripheral",
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
    for index, case in enumerate(
        training_cases[len(training_rows) :], start=len(training_rows) + 1,
    ):
        training_rows.append(_training_baseline_case_row(model, case))
        save_progress("training" if index < len(training_cases) else "complete")
        print(f"[pal-nurbs] baseline training {index}/{len(training_cases)}", flush=True)
    save_progress("complete")
    baseline_value, baseline_health = _summarize_training_baseline(training_rows)
    return baseline_value, training_rows, baseline_health


def _evaluate(
    model: MinimalOpticalModel, cases: Sequence[Mapping[str, Any]], baseline: Mapping[str, Mapping[str, float]] | None,
    *, with_grad: bool, baseline_valid: Mapping[str, float] | None = None,
    functional_weight: float = 0.85, peripheral_weight: float = 0.15,
) -> tuple[float, list[dict[str, Any]], dict[str, float]]:
    rows, group_values = [], {}
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
            "grouped training schema must contain exactly the five groups "
            f"{sorted(required)}, got {sorted(actual)}"
        )
    minimum_ratio, maximum_edge = math.inf, 0.0
    for index, case in enumerate(cases, start=1):
        with torch.set_grad_enabled(with_grad):
            result = model.field(case)
            energy = result.kernel.sum()
            if abs(float(energy.detach().cpu()) - 1.0) > 1e-10:
                raise ValueError(f"non-unit PSF energy for {case['case_id']}")
            moment = psf_second_moment_mm2(result.kernel, pixel_pitch_mm=result.pixel_pitch_mm)
            if baseline is None:
                score = torch.ones_like(moment)
            else:
                denominator = float(baseline[str(case["case_id"])]["m2_mm2"])
                if not math.isfinite(denominator) or denominator <= 0.0:
                    raise ValueError(f"invalid Original PAL M2 denominator for {case['case_id']}: {denominator}")
                score = moment / denominator
            group = str(case["training_group"])
            group_values.setdefault(group, []).append(float(score.detach().cpu()))
            if with_grad:
                if not score.requires_grad:
                    raise RuntimeError(
                        f"training score is detached from NURBS parameters for {case['case_id']}"
                    )
                if group in FUNCTIONAL_GROUPS:
                    coefficient = functional_weight / (3.0 * group_counts[group])
                elif group in PERIPHERAL_GROUPS:
                    peripheral_count = sum(group_counts[name] for name in PERIPHERAL_GROUPS)
                    coefficient = peripheral_weight / peripheral_count
                else:
                    raise ValueError(f"unexpected training group: {group}")
                score.backward(torch.as_tensor(coefficient, device=score.device, dtype=score.dtype))
        vf = float(result.valid_fraction.detach().cpu())
        ratio = 1.0 if baseline_valid is None else vf / float(baseline_valid[str(case["case_id"])])
        edge = float(result.edge_fraction.detach().cpu())
        minimum_ratio, maximum_edge = min(minimum_ratio, ratio), max(maximum_edge, edge)
        rows.append({**dict(case), "m2_mm2": float(moment.detach().cpu()), "score": float(score.detach().cpu()), "valid_fraction": vf, "valid_fraction_ratio": ratio, "edge_fraction": edge})
        result_device = result.kernel.device
        del result, energy, moment, score
        _release_inactive_case_cuda_cache(result_device)
        if index % 16 == 0 or index == len(cases):
            print(f"[pal-nurbs] evaluated {index}/{len(cases)} cases (grad={with_grad})", flush=True)
    if abs(functional_weight - 0.85) > 1e-12 or abs(peripheral_weight - 0.15) > 1e-12:
        raise ValueError("Phase 16 dense/FPS objective weights are fixed at 0.85/0.15")
    group_losses = {
        name: sum(group_values[name]) / len(group_values[name])
        for name in FUNCTIONAL_GROUPS + PERIPHERAL_GROUPS
    }
    for row in rows:
        row["group_loss"] = float(group_losses[row["training_group"]])
    group_summary = {f"J_{name}": float(value) for name, value in group_losses.items()}
    group_summary["J_mid"] = group_summary["J_intermediate"]
    functional = sum(group_losses[name] for name in FUNCTIONAL_GROUPS) / 3.0
    peripheral_count = sum(len(group_values[name]) for name in PERIPHERAL_GROUPS)
    peripheral = sum(sum(group_values[name]) for name in PERIPHERAL_GROUPS) / peripheral_count
    objective_value = functional_weight * functional + peripheral_weight * peripheral
    group_summary.update({
        "J_functional": float(functional),
        "J_peripheral": float(peripheral),
        "J_total": float(objective_value),
    })
    objective_name = "J=0.85*J_functional+0.15*J_peripheral"
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
    """Backpropagate each startup M2 immediately while accumulating one total gradient."""

    pixel_pitch_mm = model.size_reference_mm[2000.0] / config.kernel_size_px
    for case in cases:
        result = model.field(case)
        case_loss = psf_second_moment_mm2(
            result.kernel, pixel_pitch_mm=pixel_pitch_mm,
        )
        case_loss.backward()
        result_device = result.kernel.device
        del case_loss, result
        _release_inactive_case_cuda_cache(result_device)
    grad = module.inner_q.grad
    if grad is None or not bool(torch.isfinite(grad).all()) or int((grad.abs() > 0).sum()) < 2:
        raise RuntimeError("startup gradient check failed: fewer than two finite non-zero zp gradients")
    return grad.detach().clone()


def _startup_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "startup_center",
            "distance_mm": 2000.0,
            "field_x_deg": 0.0,
            "field_y_deg": 0.0,
        },
        {
            "case_id": "startup_edge",
            "distance_mm": 2000.0,
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


def _make_stage_resume_payload(
    *,
    identity_sha256: str,
    status: str,
    control_count: int,
    max_steps: int,
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
    return {
        "schema_version": STAGE_RESUME_SCHEMA_VERSION,
        "identity_sha256": identity_sha256,
        "status": status,
        "control_count": int(control_count),
        "max_steps": int(max_steps),
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
    max_steps: int,
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
    if int(payload.get("max_steps", -1)) != int(max_steps):
        raise ValueError(f"stage max-steps mismatch: {path}")
    status = str(payload.get("status", ""))
    if status not in {"active", "completed"}:
        raise ValueError(f"stage resume status is invalid: {path}")
    history = payload.get("history")
    if not isinstance(history, list):
        raise ValueError(f"stage history is malformed: {path}")
    completed_step = int(payload.get("completed_step", -1))
    if completed_step != len(history):
        raise ValueError(f"stage step/history mismatch: {path}")
    if completed_step < 0 or completed_step > int(max_steps):
        raise ValueError(f"stage completed step is outside its budget: {path}")
    learning_rate = float(payload.get("learning_rate", math.nan))
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError(f"stage learning rate is invalid: {path}")
    expected_steps = list(range(1, completed_step + 1))
    actual_steps = [int(row.get("step", -1)) for row in history]
    if actual_steps != expected_steps:
        raise ValueError(f"stage history steps are not contiguous: {path}")
    if status == "completed" and not isinstance(payload.get("stage_summary"), dict):
        raise ValueError(f"completed stage lacks its summary: {path}")
    if status == "completed" and not isinstance(payload.get("optimizer_model_state"), dict):
        raise ValueError(f"completed stage lacks the model paired with its Adam state: {path}")
    for name in ("model_state", "optimizer_state", "best_state", "best_health", "rng_state"):
        if not isinstance(payload.get(name), dict):
            raise ValueError(f"stage resume state lacks {name}: {path}")
    return payload


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

    def write_state(phase: str, **details: Any) -> None:
        _write_run_state(
            output,
            identity_sha256=identity_sha256,
            status="running",
            phase=phase,
            elapsed_seconds=prior_elapsed + time.time() - session_start,
            **details,
        )

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
    write_state("case_layout")
    training_cases = _prepare_or_load_case_layout(
        config, output, model, identity_sha256=identity_sha256,
    )
    objective_options = {
        "functional_weight": config.functional_objective_weight,
        "peripheral_weight": config.peripheral_objective_weight,
    }

    baseline_state_path = output / "baseline_state.pt"
    baseline_import = (
        None if config.baseline_state_import is None else Path(config.baseline_state_import)
    )
    if baseline_state_path.is_file() or baseline_import is not None:
        imported_baseline = not baseline_state_path.is_file()
        write_state("import_baseline" if imported_baseline else "restore_baseline")
        if imported_baseline:
            baseline_state = torch.load(baseline_import, map_location=device)
            if not isinstance(baseline_state, dict):
                raise ValueError("baseline state import must contain a mapping")
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
        startup_cases = _startup_cases()
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

    stage_summaries: list[dict[str, Any]] = []
    stage_specs = (
        (7, config.max_steps_7),
        (11, config.max_steps_11),
        (19, config.max_steps_19),
    )
    for stage_index, (control_count, max_steps) in enumerate(stage_specs):
        resume_path = output / f"stage_{control_count}x{control_count}" / "resume.pt"
        later = [
            output / f"stage_{later_count}x{later_count}" / "resume.pt"
            for later_count, _ in stage_specs[stage_index + 1 :]
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
                "max_steps": max_steps,
            },
        )
        optimizer = torch.optim.Adam([module.inner_q], lr=config.learning_rate)

        if resume_path.is_file():
            stage_state = _load_stage_resume_state(
                resume_path,
                identity_sha256=identity_sha256,
                control_count=control_count,
                max_steps=max_steps,
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
                if (
                    control_count == 11
                    and float(stage_summary["relative_stage_improvement"])
                    < config.minimum_stage_relative_improvement
                ):
                    if (output / "stage_19x19" / "resume.pt").is_file():
                        raise ValueError("19x19 state exists although the 11x11 stop rule fired")
                    break
                continue
        else:
            write_state("stage_initialize", control_count=control_count, completed_step=0)
            if control_count == 7:
                # The 7x7 module is verified above to be the exact zero-residual
                # Original PAL state. Every case denominator is that same physical
                # baseline, so all group objectives and J are exactly 1 without an
                # additional 80-case PSF pass.
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
                    max_steps=max_steps,
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

        if lr >= config.minimum_learning_rate:
            for step in range(completed_step + 1, int(max_steps) + 1):
                module.zero_grad(set_to_none=True)
                optimizer.param_groups[0]["lr"] = lr
                current, _, health = _evaluate(
                    model,
                    training_cases,
                    baseline,
                    with_grad=True,
                    baseline_valid=baseline_valid,
                    **objective_options,
                )
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
                        base_sag + module.delta_raw(xx, yy), power_config, zones
                    )
                    far_change = abs(
                        float(power["P_far_D"].detach().cpu())
                        - float(baseline_power["P_far_D"])
                    )
                    add_change = abs(
                        float(power["ADD_D"].detach().cpu())
                        - float(baseline_power["ADD_D"])
                    )
                    if far_change > config.far_tolerance_D or add_change > config.add_tolerance_D:
                        reason = "prescription"
                        continue
                    candidate, candidate_rows, candidate_health = _evaluate(
                        model,
                        training_cases,
                        baseline,
                        with_grad=False,
                        baseline_valid=baseline_valid,
                        **objective_options,
                    )
                    if (
                        candidate_health["minimum_valid_fraction_ratio"]
                        < config.minimum_valid_fraction_ratio
                    ):
                        reason = "health"
                        continue
                    if not math.isfinite(candidate) or candidate > current:
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
                if accepted and candidate < best:
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
                    }
                )
                completed_step = step
                _torch_save_atomic(
                    resume_path,
                    _make_stage_resume_payload(
                        identity_sha256=identity_sha256,
                        status="active",
                        control_count=control_count,
                        max_steps=max_steps,
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
                write_state(
                    "stage_training",
                    control_count=control_count,
                    completed_step=completed_step,
                    learning_rate=lr,
                )
                if lr < config.minimum_learning_rate:
                    break

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
            "initial_J": stage_initial,
            "initial_groups": stage_initial_groups,
            "best_J": best,
            "best_groups": _joint_metric_fields(best, best_health),
            "relative_stage_improvement": improvement,
            "steps": len(history),
        }
        _torch_save_atomic(
            resume_path,
            _make_stage_resume_payload(
                identity_sha256=identity_sha256,
                status="completed",
                control_count=control_count,
                max_steps=max_steps,
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
        if control_count == 11 and improvement < config.minimum_stage_relative_improvement:
            break

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
    final_power = prescription_metrics(base_sag + delta, power_config, zones)
    final_p_far_D = float(final_power["P_far_D"])
    final_add_D = float(final_power["ADD_D"])
    max_abs_sag_delta_mm = float(delta.abs().max())
    metric_names = ("far", "mid", "near", "peripheral", "functional")
    improvement_by_group = {
        name: 100.0 * (1.0 - float(final_health[f"J_{name}"]))
        for name in metric_names
    }
    runtime_seconds = prior_elapsed + time.time() - session_start
    summary = {
        "identity_sha256": identity_sha256,
        "baseline_J": baseline_value,
        "objective_name": "J=0.85*J_functional+0.15*J_peripheral",
        "training_case_count": len(training_cases),
        "training_groups": {
            name: sum(row["training_group"] == name for row in training_cases)
            for name in TRAINING_GROUP_COUNTS
        },
        "stages": stage_summaries,
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
        "runtime_seconds": runtime_seconds,
        "trace_psf_exception": False,
        "health": final_health,
    }
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
        _write_run_state(
            output,
            identity_sha256=str(identity["identity_sha256"]),
            status="interrupted",
            phase=phase,
            elapsed_seconds=prior_elapsed + time.time() - session_start,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise
