"""固定三物距、分区/物距归一化目标的 PAL-B-spline 优化链。

训练主链只有以下物理步骤：

``7x7 PAL 后表面参数`` → ``D500/D1000/Dinf 共用 FOV 网格`` →
``真实可微光线追迹`` → ``去 pupil tilt 的物理 FFT PSF`` →
``分区指标(M2 或 M/A 的 A)的 baseline normalization`` → ``Adam 更新``。

旧的密集候选点、FPS 选择、WFNO 资格轮次、覆盖审计、PSF crop/resize、
80-case 分组目标和 7x7→11x11→19x19 阶梯都不属于本方法。
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from averfang import load_sag_xlsx
from lens_metrics_core import build_legacy_adapter, load_lens, resolve_device
from optics import GridSag

from .pal_case_layout import (
    DISTANCE_SPECS,
    PARTITION_ZONES,
    PartitionMap,
    build_multidistance_layout,
    generate_fov_grid,
    load_weight_spec,
)
from .psf_fft import effective_biot_pupil_sample_count, torch_fft_psf_from_phase
from .regional_nurbs import CONTROL_COUNT, FixedWeightNURBSPerturbation
from .system import (
    FitSpec,
    FittedE2ESystem,
    LocalCoordinateBreakSurface,
    _snell,
    build_fitted_e2e_system,
    implicit_intersection_gradient,
    make_aimed_pupil_rays,
    make_aimed_reference_ray,
    trace_system_batch_to_image_with_phase,
    trace_system_to_image_with_phase,
)


METHOD_NAME = "pal_multidistance_raw_psf_nonlegacy_m2_astig_a_bspline7"
RUN_IDENTITY_SCHEMA_VERSION = 5
CASE_LAYOUT_SCHEMA_VERSION = 1
EVALUATION_PROGRESS_SCHEMA_VERSION = 2
TRAINING_RESUME_SCHEMA_VERSION = 2
RUN_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MinimalConfig:
    """Configuration for the fixed multidistance PAL experiment.

    Units:
        wavelength_nm: nm; FOV bounds: degree; pupil/sag/PSF pitch: mm;
        ``fft_size_px`` and ``requested_np``: pixels/samples.
    """

    excel: str = "eye_image_glass_grad3.xlsx"
    zones_json: str = "inputs/pal/zones.json"
    weights_json: str = "inputs/pal/multidistance_weights.json"
    output: str = "results/optimization/run_001"
    device: str = "cuda"
    wavelength_nm: float = 555.0
    requested_np: int = 256
    fft_size_px: int = 512
    pupil_radius_mm: float | None = None
    fov_min_deg: float = -55.0
    fov_max_deg: float = 55.0
    fov_count: int = 11
    legacy_pupil_phase: bool = False
    phase_reference: str = "biot_reference_sphere"
    remove_tilt: bool = False
    case_batch_size: int = 8
    max_accepted_steps: int = 50
    early_stopping_patience: int = 7
    relative_improvement_threshold: float = 1.0e-4
    learning_rate: float = 2.0e-3
    minimum_learning_rate: float = 1.0e-6
    max_abs_control_mm: float = 0.12
    step_sag_limit_mm: float = 2.0e-3
    far_tolerance_D: float = 0.2
    add_tolerance_D: float = 0.3
    minimum_valid_fraction: float = 0.5
    maximum_edge_fraction: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        if bool(self.legacy_pupil_phase):
            raise ValueError("legacy_pupil_phase=True is not supported by the PAL contract")
        if self.phase_reference != "biot_reference_sphere":
            raise ValueError("PAL requires phase_reference='biot_reference_sphere'")
        if bool(self.remove_tilt):
            raise ValueError("PAL requires remove_tilt=False with the BIOT reference sphere")
        if int(self.fov_count) != 11:
            raise ValueError("the multidistance PAL method requires fov_count=11")
        expected_step = (55.0 - (-55.0)) / 10.0
        if float(self.fov_min_deg) != -55.0 or float(self.fov_max_deg) != 55.0:
            raise ValueError("the multidistance PAL method requires FOV bounds -55..55 degrees")
        if not math.isclose(expected_step, 11.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("the multidistance PAL method requires an 11 degree FOV step")
        if int(self.case_batch_size) <= 0:
            raise ValueError("case_batch_size must be positive")
        if not math.isfinite(float(self.fov_min_deg)) or not math.isfinite(float(self.fov_max_deg)):
            raise ValueError("FOV bounds must be finite")
        if float(self.fov_max_deg) <= float(self.fov_min_deg):
            raise ValueError("fov_max_deg must be greater than fov_min_deg")
        if int(self.requested_np) <= 1 or int(self.fft_size_px) <= 0:
            raise ValueError("requested_np must be >1 and fft_size_px must be positive")
        if self.pupil_radius_mm is not None and (
            not math.isfinite(float(self.pupil_radius_mm)) or float(self.pupil_radius_mm) <= 0.0
        ):
            raise ValueError("pupil_radius_mm must be finite and positive when supplied")
        for name in (
            "learning_rate",
            "minimum_learning_rate",
            "max_abs_control_mm",
            "step_sag_limit_mm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if int(self.max_accepted_steps) < 0:
            raise ValueError("max_accepted_steps must be non-negative")
        if int(self.early_stopping_patience) < 1:
            raise ValueError("early_stopping_patience must be positive")
        if not math.isfinite(float(self.relative_improvement_threshold)) or float(self.relative_improvement_threshold) <= 0.0:
            raise ValueError("relative_improvement_threshold must be finite and positive")
        if not 0.0 < float(self.minimum_valid_fraction) <= 1.0:
            raise ValueError("minimum_valid_fraction must be in (0,1]")
        if not 0.0 <= float(self.maximum_edge_fraction) < 1.0:
            raise ValueError("maximum_edge_fraction must be in [0,1)")


@dataclass(frozen=True)
class PALPowerConfig:
    semi_diameter_mm: float
    refractive_index: float
    front_radius_mm: float
    center_thickness_mm: float
    crib_diameter_mm: float = 80.0


@dataclass(frozen=True)
class FieldResult:
    """Physical raw FFT PSF data for one case or a leading case batch."""

    psf: torch.Tensor
    valid_fraction: torch.Tensor
    pixel_pitch_mm: float | torch.Tensor
    edge_fraction: torch.Tensor
    valid_mask: torch.Tensor | None = None


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def _replace_atomic(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    temporary = _temporary_sibling(destination)
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_atomic(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_save_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    temporary = _temporary_sibling(destination)
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
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


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    temporary = _temporary_sibling(path)
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_atomic(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_training_log(path: Path, message: str) -> None:
    """Append one durable human-readable progress record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{timestamp} {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


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
        "biot/e2e/__init__.py",
        "biot/e2e/pal_case_layout.py",
        "biot/e2e/pal_nurbs.py",
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
        "zones_json": Path(config.zones_json),
        "weights_json": Path(config.weights_json),
    }
    with _silence_setup_output():
        lens = load_lens(
            Path(config.excel),
            device=resolve_device("cpu"),
            wavelength_nm=config.wavelength_nm,
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
        raise ValueError("run identity belongs to a different PAL method")
    return claimed


def _open_run_directory(config: MinimalConfig, *, resume: bool) -> tuple[Path, dict[str, Any]]:
    output = Path(config.output)
    if resume:
        if not output.is_dir():
            raise FileNotFoundError(f"resume requires an existing run directory: {output}")
    elif output.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    current = _build_run_identity(config)
    if not resume:
        output.mkdir(parents=True)
        _write_json_atomic(output / "run_identity.json", current)
        _write_json_atomic(output / "config.json", asdict(config))
        return output, current
    identity_path = output / "run_identity.json"
    if not identity_path.is_file():
        raise ValueError(f"resume requires run_identity.json: {identity_path}")
    saved = _read_json(identity_path)
    saved_hash = _validate_identity_payload(saved)
    if saved_hash != current["identity_sha256"] or saved != current:
        raise ValueError(
            "resume identity mismatch: config, input hashes, implementation closure, or runtime changed"
        )
    return output, current


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
    value = float(_read_json(path).get("elapsed_seconds", 0.0))
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
            "method": METHOD_NAME,
            "status": status,
            "phase": phase,
            "elapsed_seconds": float(elapsed_seconds),
            "updated_unix_time": time.time(),
            **details,
        },
    )


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(payload: Mapping[str, Any]) -> None:
    python_state = payload.get("python")
    if python_state is not None:
        random.setstate(python_state)
    cpu = payload.get("torch_cpu")
    if not torch.is_tensor(cpu):
        raise ValueError("resume state is missing the CPU RNG state")
    torch.set_rng_state(cpu.detach().cpu())
    cuda = payload.get("torch_cuda", [])
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("resume state contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda])


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


def _silence_setup_output():
    import contextlib
    import io

    return contextlib.redirect_stdout(io.StringIO())


def _normalize_psf(psf: torch.Tensor) -> torch.Tensor:
    if psf.ndim < 2:
        raise ValueError("physical PSF must have shape [..., H, W]")
    if not bool(torch.isfinite(psf).all()) or bool((psf < 0.0).any()):
        raise ValueError("physical PSF must be finite and non-negative")
    energy = psf.sum(dim=(-2, -1), keepdim=True)
    if not bool(torch.isfinite(energy).all()) or not bool((energy > 0.0).all()):
        raise ValueError("physical PSF must have positive finite energy")
    return psf / energy


def psf_second_moment_mm2(
    psf: torch.Tensor, *, pixel_pitch_mm: float | torch.Tensor
) -> torch.Tensor:
    """Return centered intensity second moments in mm² for ``[...,H,W]``.

    The input is the raw physical FFT PSF.  No crop, interpolation, filtering
    or display transform is applied before this calculation.
    """
    pitch = torch.as_tensor(pixel_pitch_mm, device=psf.device, dtype=psf.dtype)
    if not bool(torch.isfinite(pitch).all()) or bool((pitch <= 0.0).any()):
        raise ValueError("PSF pixel pitch must be finite and positive")
    normalized = _normalize_psf(psf)
    height, width = normalized.shape[-2:]
    y = torch.arange(height, device=psf.device, dtype=psf.dtype) - 0.5 * (height - 1)
    x = torch.arange(width, device=psf.device, dtype=psf.dtype) - 0.5 * (width - 1)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx_mm = xx * pitch[..., None, None]
    yy_mm = yy * pitch[..., None, None]
    cx = (normalized * xx_mm).sum(dim=(-2, -1))
    cy = (normalized * yy_mm).sum(dim=(-2, -1))
    moment = (
        normalized
        * (
            (xx_mm - cx[..., None, None]).square()
            + (yy_mm - cy[..., None, None]).square()
        )
    ).sum(dim=(-2, -1))
    if not bool(torch.isfinite(moment).all()) or bool((moment < 0.0).any()):
        raise ValueError("PSF second moment is non-finite or negative")
    return moment


def _edge_fraction(psf: torch.Tensor, edge_px: int = 5) -> torch.Tensor:
    normalized = _normalize_psf(psf)
    edge = min(max(int(edge_px), 1), int(normalized.shape[-2]) // 2)
    mask = torch.zeros(normalized.shape[-2:], device=normalized.device, dtype=torch.bool)
    mask[:edge] = True
    mask[-edge:] = True
    mask[:, :edge] = True
    mask[:, -edge:] = True
    return torch.where(mask, normalized, torch.zeros_like(normalized)).sum(dim=(-2, -1))


def _release_inactive_case_cuda_cache(device: torch.device | str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def _physical_fft_pixel_pitch_mm(
    lens: object,
    *,
    pupil_sample_count: int,
    psf_size_px: int,
    wavelength_nm: float,
) -> float:
    """Compute raw FFT image-plane pitch from BIOT's field-dependent WFNO."""
    lam_mm = torch.as_tensor(
        float(wavelength_nm) * 1.0e-6,
        device=getattr(lens, "device", torch.device("cpu")),
        dtype=torch.float64,
    )
    f_number = lens.cal_WFNO(lam_mm)
    if torch.is_tensor(f_number):
        f_number = float(f_number.detach().cpu().reshape(-1)[0].item())
    f_number = float(f_number)
    if not math.isfinite(f_number) or f_number <= 0.0:
        raise ValueError(f"BIOT WFNO is invalid: {f_number}")
    n_pupil = int(pupil_sample_count)
    n_fft = int(psf_size_px)
    if n_pupil <= 1 or n_fft <= 0:
        raise ValueError("invalid pupil/FFT sizes")
    pitch = float(wavelength_nm) * 1.0e-6 * f_number * float(n_pupil - 1) / float(n_fft)
    if not math.isfinite(pitch) or pitch <= 0.0:
        raise ValueError(f"physical FFT pixel pitch is invalid: {pitch}")
    return pitch


class MinimalOpticalModel:
    """Reusable real differentiable trace model for the fixed case layout."""

    def __init__(self, config: MinimalConfig, perturbation: torch.nn.Module) -> None:
        self.config = config
        self.perturbation = perturbation
        self.device = torch.device(config.device)
        self.dtype = torch.float64
        self.sample_count = effective_biot_pupil_sample_count(config.requested_np)
        self.fit_spec = FitSpec(control_shape=(41, 41), sample_shape=(81, 81), degree=3)
        self._cache: dict[tuple[float, float, float], tuple[FittedE2ESystem, object]] = {}
        self._templates: dict[float, tuple[FittedE2ESystem, object]] = {}
        self._pal_sag: torch.Tensor | None = None
        self._pal_power_config: PALPowerConfig | None = None
        self._pal_zones: Mapping[str, torch.Tensor] | None = None

    def set_prescription_context(
        self,
        sag: torch.Tensor,
        power_config: PALPowerConfig,
        zones: Mapping[str, torch.Tensor],
    ) -> None:
        """Attach the differentiable PAL M/A context used by astigmatism loss."""
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
        for zone in ("astig_left", "astig_right"):
            mask_name = "peripheral_" + zone
            mask = self._pal_zones[mask_name] & maps["valid"]
            if not bool(mask.any()):
                raise ValueError(f"M/A mask has no valid samples for {zone}")
            value = maps["A_D"][mask].mean()
            if not bool(torch.isfinite(value)) or bool(value <= 0.0):
                raise ValueError(f"M/A astigmatism A is invalid for {zone}")
            result[zone] = value
        return result

    @staticmethod
    def _key(distance_mm: float, field_x_deg: float, field_y_deg: float) -> tuple[float, float, float]:
        return round(float(distance_mm), 6), round(float(field_x_deg), 6), round(float(field_y_deg), 6)

    @staticmethod
    def _set_field(lens: object, field_x_deg: float, field_y_deg: float) -> None:
        for surface in lens.surfaces:
            if getattr(surface, "type", "") == "CB":
                for name, value in (("tilt_x", -float(field_y_deg)), ("tilt_y", -float(field_x_deg))):
                    old = getattr(surface, name)
                    setattr(
                        surface,
                        name,
                        torch.as_tensor(value, device=old.device, dtype=old.dtype),
                    )

    @staticmethod
    def _clone(template: FittedE2ESystem, lens: object, field_x_deg: float, field_y_deg: float) -> FittedE2ESystem:
        surfaces = [
            LocalCoordinateBreakSurface(
                semi_diameter_mm=surface.semi_diameter_mm,
                n_after=surface.n_after,
                tilt_x_deg=-float(field_y_deg),
                tilt_y_deg=-float(field_x_deg),
                tilt_z_deg=surface.tilt_z_deg,
            )
            if isinstance(surface, LocalCoordinateBreakSurface)
            else surface
            for surface in template.surfaces
        ]
        return FittedE2ESystem(
            lens=lens,
            surfaces=surfaces,
            image_surface=template.image_surface,
            front_surface=template.front_surface,
            back_surface=template.back_surface,
            fit_spec=template.fit_spec,
            wavelength_nm=template.wavelength_nm,
            surface_distances_mm=template.surface_distances_mm,
            image_distance_value_mm=template.image_distance_value_mm,
            initial_ior=template.initial_ior,
            object_distance_mm=template.object_distance_mm,
            exit_pupil_position_mm=template.exit_pupil_position_mm,
            stop_semi_diameter_mm=template.stop_semi_diameter_mm,
        )

    def _new_system(self, distance_mm: float, field_x_deg: float, field_y_deg: float) -> FittedE2ESystem:
        distance = float(distance_mm)
        if distance not in self._templates:
            template, temporary = build_fitted_e2e_system(
                self.config.excel,
                object_distance=distance,
                field_x_deg=field_x_deg,
                field_y_deg=field_y_deg,
                wavelength_nm=self.config.wavelength_nm,
                fit_spec=self.fit_spec,
                device=self.device,
                dtype=self.dtype,
                train_back_surface=False,
                back_perturbation=self.perturbation,
            )
            if template.lens is None:
                raise RuntimeError("fitted template has no BIOT Lensdata for aiming")
            self._templates[distance] = (template, template.lens)
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
        template, lens = self._templates[distance]
        self._set_field(lens, field_x_deg, field_y_deg)
        return self._clone(template, lens, field_x_deg, field_y_deg)

    def _system_and_rays(self, distance_mm: float, field_x_deg: float, field_y_deg: float):
        key = self._key(distance_mm, field_x_deg, field_y_deg)
        if key not in self._cache:
            system = self._new_system(*key)
            system.reference_ray = make_aimed_reference_ray(
                system, dtype=self.dtype, device=self.device
            )
            rays = make_aimed_pupil_rays(
                system,
                sample_count=self.sample_count,
                pupil_radius_mm=self.config.pupil_radius_mm,
                field_x_deg=field_x_deg,
                field_y_deg=field_y_deg,
                dtype=self.dtype,
                device=self.device,
            )
            if system.lens is None:
                raise RuntimeError("BIOT lens was released before FFT pitch calculation")
            system.physical_fft_pixel_pitch_mm = _physical_fft_pixel_pitch_mm(
                system.lens,
                pupil_sample_count=self.sample_count,
                psf_size_px=self.config.fft_size_px,
                wavelength_nm=self.config.wavelength_nm,
            )
            system.release_biot_lens()
            self._cache[key] = system, rays
        return self._cache[key]

    def reference_rear_intersection(
        self, distance_mm: float, field_x_deg: float, field_y_deg: float
    ) -> tuple[float, float]:
        """Trace one chief ray to the original PAL rear surface for layout."""
        system = self._new_system(float(distance_mm), float(field_x_deg), float(field_y_deg))
        try:
            ray = make_aimed_reference_ray(system, dtype=self.dtype, device=self.device)
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
            if rear_point is None:
                raise RuntimeError("fitted system does not contain its PAL rear surface")
            if not bool(active.all()) or not bool(torch.isfinite(rear_point).all()):
                raise RuntimeError(
                    "invalid chief-ray rear intersection at "
                    f"field=({field_x_deg:g},{field_y_deg:g}) deg, distance={distance_mm:g} mm"
                )
            return float(rear_point[0].detach().cpu()), float(rear_point[1].detach().cpu())
        finally:
            system.release_biot_lens()

    def field(self, case: Mapping[str, Any]) -> FieldResult:
        label = str(case["distance_label"])
        spec = next((item for item in DISTANCE_SPECS if item.label == label), None)
        if spec is None:
            raise ValueError(f"unknown distance label: {label}")
        field_x = float(case["field_x_deg"])
        field_y = float(case["field_y_deg"])
        system, rays = self._system_and_rays(spec.object_distance_mm, field_x, field_y)
        trace = trace_system_to_image_with_phase(system, rays, phase_reference=self.config.phase_reference)
        if not bool(trace.valid.any()):
            raise RuntimeError(f"no valid rays for {case['case_id']}")
        fft = torch_fft_psf_from_phase(
            trace.phase_rad,
            trace.valid,
            sample_count=self.sample_count,
            psf_size_px=self.config.fft_size_px,
            remove_piston=True,
            # The non-legacy BIOT reference sphere already defines the
            # de-tilted physical pupil.  A second fitted linear-phase removal
            # shifts the authoritative FFT PSF and must not be applied here.
            remove_tilt=self.config.remove_tilt,
        )
        pitch = system.physical_fft_pixel_pitch_mm
        if pitch is None:
            raise RuntimeError("missing physical raw FFT pixel pitch")
        return FieldResult(
            psf=fft.psf,
            valid_fraction=trace.valid.to(dtype=self.dtype).mean(),
            pixel_pitch_mm=float(pitch),
            edge_fraction=_edge_fraction(fft.psf),
            valid_mask=trace.valid,
        )

    def field_batch(self, cases: Sequence[Mapping[str, Any]]) -> FieldResult:
        """Trace and FFT a true case batch with leading dimension ``B``."""
        if not cases:
            raise ValueError("cannot evaluate an empty field batch")
        systems: list[FittedE2ESystem] = []
        rays_by_case: list[Any] = []
        pitches: list[float] = []
        for case in cases:
            label = str(case["distance_label"])
            spec = next((item for item in DISTANCE_SPECS if item.label == label), None)
            if spec is None:
                raise ValueError(f"unknown distance label: {label}")
            system, rays = self._system_and_rays(
                spec.object_distance_mm,
                float(case["field_x_deg"]),
                float(case["field_y_deg"]),
            )
            if system.physical_fft_pixel_pitch_mm is None:
                raise RuntimeError(f"missing physical raw FFT pixel pitch for {case['case_id']}")
            systems.append(system)
            rays_by_case.append(rays)
            pitches.append(float(system.physical_fft_pixel_pitch_mm))
        # A true case batch multiplies the B-spline Newton-search graph by B.
        # Use the existing exact implicit-function derivative for intersections:
        # the converged forward root is unchanged and one in-graph correction
        # carries its derivative, while the non-physical search history is not
        # retained eight times.  This is a fixed batch contract, not an OOM
        # fallback or an adaptive batch-size change.
        with implicit_intersection_gradient(True):
            trace = trace_system_batch_to_image_with_phase(
                systems,
                rays_by_case,
                phase_reference=self.config.phase_reference,
            )
        if not bool(trace.valid.any(dim=1).all()):
            failed = [
                str(case["case_id"])
                for case, valid in zip(cases, trace.valid.any(dim=1))
                if not bool(valid)
            ]
            raise RuntimeError("no valid rays for case batch: " + ", ".join(failed))
        fft = torch_fft_psf_from_phase(
            trace.phase_rad,
            trace.valid,
            sample_count=self.sample_count,
            psf_size_px=self.config.fft_size_px,
            remove_piston=True,
            remove_tilt=self.config.remove_tilt,
        )
        pitch = torch.as_tensor(pitches, device=self.device, dtype=self.dtype)
        return FieldResult(
            psf=fft.psf,
            valid_fraction=trace.valid.to(dtype=self.dtype).mean(dim=1),
            pixel_pitch_mm=pitch,
            edge_fraction=_edge_fraction(fft.psf),
            valid_mask=trace.valid,
        )

    def raw_psf_batch(self, cases: Sequence[Mapping[str, Any]]) -> RawPSFBatchResult:
        """发布评价所需的原生 PSF 批量，不重写已验证的批量追迹。"""
        result = self.field_batch(cases)
        if not isinstance(result.pixel_pitch_mm, torch.Tensor):
            raise TypeError("field_batch pixel_pitch_mm must be a tensor for raw PSF evaluation")
        return RawPSFBatchResult(
            psf=result.psf,
            valid_fraction=result.valid_fraction,
            pixel_pitch_mm=result.pixel_pitch_mm,
        )

    def close(self) -> None:
        for system, _ in self._cache.values():
            system.release_biot_lens()
        for system, _ in self._templates.values():
            system.release_biot_lens()
        self._cache.clear()
        self._templates.clear()

def _finite_difference(values: torch.Tensor, pitch: float) -> tuple[torch.Tensor, torch.Tensor]:
    dy = torch.empty_like(values)
    dx = torch.empty_like(values)
    dy[1:-1] = (values[2:] - values[:-2]) / (2.0 * pitch)
    dy[0] = (values[1] - values[0]) / pitch
    dy[-1] = (values[-1] - values[-2]) / pitch
    dx[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2.0 * pitch)
    dx[:, 0] = (values[:, 1] - values[:, 0]) / pitch
    dx[:, -1] = (values[:, -1] - values[:, -2]) / pitch
    return dy, dx


def torch_averfang_maps(sag: torch.Tensor, config: PALPowerConfig) -> dict[str, torch.Tensor]:
    pitch = 2.0 * config.semi_diameter_mm / (int(sag.shape[0]) - 1)
    zy, zx = _finite_difference(sag, pitch)
    zyy, zyx = _finite_difference(zy, pitch)
    zxy, zxx = _finite_difference(zx, pitch)
    zxy = 0.5 * (zxy + zyx)
    e, f, g = 1.0 + zx.square(), zx * zy, 1.0 + zy.square()
    scale = torch.sqrt(1.0 + zx.square() + zy.square())
    l, m, n = zxx / scale, zxy / scale, zyy / scale
    eps = torch.finfo(sag.dtype).eps
    denom = (e * g - f.square()).clamp_min(eps)
    gaussian = (l * n - m.square()) / denom.square().clamp_min(eps)
    mean = (e * n + g * l - 2.0 * f * m) / (2.0 * denom.pow(1.5).clamp_min(eps))
    disc = (mean.square() - gaussian).clamp_min(0.0)
    pmax, pmin = mean + torch.sqrt(disc), mean - torch.sqrt(disc)
    pmax = torch.where(pmax.abs() > eps, pmax, torch.full_like(pmax, eps))
    pmin = torch.where(pmin.abs() > eps, pmin, torch.full_like(pmin, eps))
    rear_radius = -0.5 * (1.0 / pmax + 1.0 / pmin)
    curvature_diff = pmax - pmin
    count = int(round(config.crib_diameter_mm)) + 1
    start = (int(sag.shape[0]) - count) // 2
    stop = start + count
    rr = rear_radius[start:stop, start:stop]
    cdiff = curvature_diff[start:stop, start:stop]
    z = sag[start:stop, start:stop]
    coord = torch.linspace(
        -config.semi_diameter_mm,
        config.semi_diameter_mm,
        int(sag.shape[0]),
        device=sag.device,
        dtype=sag.dtype,
    )[start:stop]
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    rq = torch.as_tensor(config.front_radius_mm, device=sag.device, dtype=sag.dtype)
    front = -torch.sqrt((rq.square() - xx.square() - yy.square()).clamp_min(0.0)) + rq
    thickness = z - front + config.center_thickness_mm
    nl = torch.as_tensor(config.refractive_index, device=sag.device, dtype=sag.dtype)
    denominator = -nl * rr * rq + (nl - 1.0) * thickness * rr
    denominator = torch.where(denominator.abs() > eps, denominator, torch.full_like(denominator, eps))
    power = (nl - 1.0) * 1000.0 * (
        nl * (-rr - rq) + (nl - 1.0) * thickness
    ) / denominator
    astig = (-(cdiff) * (nl - 1.0) * 1000.0).abs()
    valid = (
        (xx.square() + yy.square() <= (config.crib_diameter_mm / 2.0) ** 2 + 1.0e-9)
        & torch.isfinite(power)
        & torch.isfinite(astig)
    )
    return {
        "power_D": power,
        # A is the M/A astigmatism component in diopters. Keep the legacy
        # spelling for prescription reports, but make the loss input explicit.
        "A_D": astig,
        "astigmatism_D": astig,
        "valid": valid,
    }


def load_pal(
    config: MinimalConfig, device: torch.device
) -> tuple[torch.Tensor, PALPowerConfig, dict[str, torch.Tensor]]:
    lens = load_lens(
        Path(config.excel), device=resolve_device("cpu"), wavelength_nm=config.wavelength_nm
    )
    adapter = build_legacy_adapter(lens, wavelength_nm=config.wavelength_nm)
    back = lens.surfaces[2]
    if not isinstance(back, GridSag):
        raise ValueError("PAL rear surface must be GridSag")
    sag = torch.as_tensor(
        load_sag_xlsx(Path(back.sag_file_path), grid_shape=back.grid_shape),
        dtype=torch.float64,
        device=device,
    )
    power_config = PALPowerConfig(
        float(back.semi_dia),
        float(adapter.n1),
        float(1.0 / adapter.c0),
        float(adapter.h_glass_mm),
    )
    payload = _read_json(config.zones_json)
    zones = {
        name: torch.as_tensor(value, dtype=torch.bool, device=device)
        for name, value in payload["masks"].items()
    }
    return sag, power_config, zones


def prescription_metrics(
    sag: torch.Tensor,
    power_config: PALPowerConfig,
    zones: Mapping[str, torch.Tensor],
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
    return {
        "P_far_D": pfar,
        "ADD_D": add,
        "astig_mean_D": maps["A_D"][monitored].mean(),
    }


def _layout_identity(config: MinimalConfig) -> dict[str, Any]:
    body = {
        "schema_version": CASE_LAYOUT_SCHEMA_VERSION,
        "method": METHOD_NAME,
        "field_grid": {
            "count": int(config.fov_count),
            "min_deg": float(config.fov_min_deg),
            "max_deg": float(config.fov_max_deg),
        },
        "distance_specs": [
            {
                "label": spec.label,
                "object_distance_mm": spec.serialized_distance,
                "focus_zone": spec.focus_zone,
            }
            for spec in DISTANCE_SPECS
        ],
        "zones_sha256": _sha256_file(config.zones_json),
        "weights_sha256": _sha256_file(config.weights_json),
        "partition_rule": "stored mask else nearest partition cell inside monitored aperture",
    }
    return {**body, "layout_identity_sha256": _canonical_json_sha256(body)}


def _validate_layout_cases(cases: Sequence[Mapping[str, Any]], config: MinimalConfig) -> None:
    block_size = int(config.fov_count) * int(config.fov_count)
    expected = len(DISTANCE_SPECS) * block_size
    if len(cases) != expected:
        raise ValueError(f"case layout has {len(cases)} cases, expected {expected}")
    ids = [str(case.get("case_id")) for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case layout contains duplicate case IDs")
    expected_indices = list(range(expected))
    actual_indices = [int(case.get("case_index", -1)) for case in cases]
    if actual_indices != expected_indices:
        raise ValueError("case layout case indices are not contiguous")
    weights = [float(case.get("objective_weight", math.nan)) for case in cases]
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("case layout contains invalid objective weights")
    if abs(sum(weights) - 1.0) > 1.0e-12:
        raise ValueError("case layout objective weights are not normalized")
    field_grid = generate_fov_grid(
        field_min_deg=config.fov_min_deg,
        field_max_deg=config.fov_max_deg,
        count=config.fov_count,
    )
    expected_grid = [
        (
            int(field["grid_row"]),
            int(field["grid_column"]),
            float(field["field_x_deg"]),
            float(field["field_y_deg"]),
        )
        for field in field_grid
    ]
    labels = [str(spec.label) for spec in DISTANCE_SPECS]
    for index, spec in enumerate(DISTANCE_SPECS):
        block = cases[index * block_size : (index + 1) * block_size]
        if any(str(case.get("distance_label")) != spec.label for case in block):
            raise ValueError(f"case layout distance block {spec.label} is malformed")
        if any(str(case.get("zone")) not in PARTITION_ZONES for case in block):
            raise ValueError(f"case layout distance block {spec.label} has an unknown zone")
        expected_distance = spec.serialized_distance
        for field_index, case in enumerate(block):
            expected_case_id = (
                f"{spec.label}_r{expected_grid[field_index][0]:02d}_c{expected_grid[field_index][1]:02d}"
            )
            if str(case.get("case_id")) != expected_case_id:
                raise ValueError(f"case layout case ID changed for {expected_case_id}")
            if str(case.get("focus_zone")) != spec.focus_zone:
                raise ValueError(f"case layout focus zone changed for {expected_case_id}")
            actual_distance = case.get("object_distance_mm")
            if isinstance(expected_distance, str):
                distance_matches = actual_distance == expected_distance
            else:
                try:
                    distance_matches = math.isfinite(float(actual_distance)) and math.isclose(
                        float(actual_distance), float(expected_distance), rel_tol=0.0, abs_tol=0.0
                    )
                except (TypeError, ValueError):
                    distance_matches = False
            if not distance_matches:
                raise ValueError(f"case layout object distance changed for {expected_case_id}")
            actual_grid = (
                int(case.get("grid_row", -1)),
                int(case.get("grid_column", -1)),
                float(case.get("field_x_deg", math.nan)),
                float(case.get("field_y_deg", math.nan)),
            )
            if actual_grid != expected_grid[field_index]:
                raise ValueError(f"case layout FOV grid changed for {expected_case_id}")
    reference_grid = expected_grid
    for offset in (block_size, 2 * block_size):
        grid = [
            (
                int(case["grid_row"]),
                int(case["grid_column"]),
                float(case["field_x_deg"]),
                float(case["field_y_deg"]),
            )
            for case in cases[offset : offset + block_size]
        ]
        if grid != reference_grid:
            raise ValueError("the three object distances do not share the same FOV grid")
    if labels != [str(case["distance_label"]) for case in cases[::block_size]]:
        raise ValueError("case layout distance ordering changed")


def _write_case_layout_csv(path: Path, cases: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "case_index",
        "case_id",
        "distance_label",
        "object_distance_mm",
        "grid_row",
        "grid_column",
        "field_x_deg",
        "field_y_deg",
        "partition_x_mm",
        "partition_physical_y_mm",
        "zone",
        "partition_mode",
        "nearest_partition_distance_mm",
        "zone_distance_mass",
        "objective_weight",
    ]
    temporary = _temporary_sibling(path)
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(cases)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_atomic(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_or_load_case_layout(
    config: MinimalConfig,
    output: Path,
    model: MinimalOpticalModel,
    *,
    identity_sha256: str,
) -> list[dict[str, Any]]:
    layout_path = output / "preoptimization" / "case_layout.json"
    layout_identity = _layout_identity(config)
    if layout_path.is_file():
        payload = _read_json(layout_path)
        if str(payload.get("identity_sha256")) != identity_sha256:
            raise ValueError("case layout identity mismatch")
        if str(payload.get("layout_identity_sha256")) != layout_identity["layout_identity_sha256"]:
            raise ValueError("case layout configuration mismatch")
        cases = payload.get("cases")
        if not isinstance(cases, list):
            raise ValueError("case layout payload is malformed")
        if str(payload.get("case_payload_sha256")) != _canonical_json_sha256(cases):
            raise ValueError("case layout payload hash mismatch")
        _validate_layout_cases(cases, config)
        return [dict(case) for case in cases]

    partition_map = PartitionMap.from_json(config.zones_json)
    weight_spec = load_weight_spec(config.weights_json)
    progress_path = output / "case_layout_progress.json"
    prefix: list[dict[str, Any]] = []
    if progress_path.is_file():
        progress = _read_json(progress_path)
        if str(progress.get("identity_sha256")) != identity_sha256:
            raise ValueError("case layout progress identity mismatch")
        if str(progress.get("layout_identity_sha256")) != layout_identity["layout_identity_sha256"]:
            raise ValueError("case layout progress configuration mismatch")
        rows = progress.get("cases")
        if not isinstance(rows, list):
            raise ValueError("case layout progress cases are malformed")
        if int(progress.get("next_case_index", -1)) != len(rows):
            raise ValueError("case layout progress index is inconsistent")
        prefix = [dict(row) for row in rows]

    def save_progress(rows: Sequence[Mapping[str, Any]]) -> None:
        _write_json_atomic(
            progress_path,
            {
                "schema_version": CASE_LAYOUT_SCHEMA_VERSION,
                "identity_sha256": identity_sha256,
                "layout_identity_sha256": layout_identity["layout_identity_sha256"],
                "next_case_index": len(rows),
                "cases": [dict(row) for row in rows],
            },
        )

    cases = build_multidistance_layout(
        field_min_deg=config.fov_min_deg,
        field_max_deg=config.fov_max_deg,
        field_count=config.fov_count,
        partition_map=partition_map,
        weight_spec=weight_spec,
        trace_reference=model.reference_rear_intersection,
        prefix_cases=prefix,
        progress_callback=save_progress,
    )
    _validate_layout_cases(cases, config)
    payload = {
        **layout_identity,
        "identity_sha256": identity_sha256,
        "case_payload_sha256": _canonical_json_sha256(cases),
        "case_count": len(cases),
        "weight_spec": weight_spec,
        "cases": cases,
    }
    _write_json_atomic(layout_path, payload)
    _write_case_layout_csv(output / "preoptimization" / "case_layout.csv", cases)
    progress_path.unlink(missing_ok=True)
    return [dict(case) for case in cases]


def prepare_only(config: MinimalConfig, *, resume: bool = False) -> Path:
    """Prepare only the fixed case/weight layout; no PSF evaluation runs."""
    _seed_everything(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output, identity = _open_run_directory(config, resume=resume)
    summary_path = output / "summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if str(summary.get("identity_sha256")) != str(identity["identity_sha256"]):
            raise ValueError("completed summary identity mismatch")
        return output / "preoptimization"
    prior_elapsed = _elapsed_seconds(output)
    session_start = time.time()
    _write_run_state(
        output,
        identity_sha256=identity["identity_sha256"],
        status="running",
        phase="case_layout",
        elapsed_seconds=prior_elapsed,
    )
    module = FixedWeightNURBSPerturbation(device=device, dtype=torch.float64)
    model = MinimalOpticalModel(config, module)
    try:
        _prepare_or_load_case_layout(
            config, output, model, identity_sha256=identity["identity_sha256"]
        )
    finally:
        model.close()
    _write_run_state(
        output,
        identity_sha256=identity["identity_sha256"],
        status="prepared",
        phase="case_layout_complete",
        elapsed_seconds=prior_elapsed + time.time() - session_start,
    )
    return output / "preoptimization"


def _metric_key(case: Mapping[str, Any]) -> str:
    return f"{str(case['zone'])}/{str(case['distance_label'])}"


def _baseline_metric_table(
    rows: Sequence[Mapping[str, Any]], *, require_complete: bool = True
) -> dict[str, float]:
    values_by_key: dict[str, list[float]] = {}
    for row in rows:
        key = _metric_key(row)
        value = float(row["loss_metric"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"baseline metric must be finite and positive for {key}")
        values_by_key.setdefault(key, []).append(value)
    table = {
        key: sum(values) / len(values)
        for key, values in values_by_key.items()
    }
    expected = {
        f"{zone}/{spec.label}"
        for zone in PARTITION_ZONES
        for spec in DISTANCE_SPECS
    }
    if require_complete and set(table) != expected:
        raise ValueError("baseline metrics do not cover all zone/distance combinations")
    return table


def _normalize_evaluation_rows(
    rows: Sequence[Mapping[str, Any]], baseline_metrics: Mapping[str, float]
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        key = _metric_key(row)
        if key not in baseline_metrics:
            raise ValueError(f"missing baseline metric for {key}")
        denominator = float(baseline_metrics[key])
        value = float(row["loss_metric"])
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(f"baseline metric is invalid for {key}")
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"loss metric is invalid for {key}")
        row_copy = dict(row)
        row_copy["baseline_metric"] = denominator
        row_copy["normalized_metric"] = value / denominator
        row_copy["weighted_loss"] = float(row_copy["objective_weight"]) * float(row_copy["normalized_metric"])
        normalized.append(row_copy)
    return normalized


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[float, dict[str, Any]]:
    if not rows:
        raise ValueError("cannot summarize an empty PSF evaluation")
    weights = [float(row["objective_weight"]) for row in rows]
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("evaluation rows contain invalid objective weights")
    if abs(sum(weights) - 1.0) > 1.0e-12:
        raise ValueError("evaluation row weights are not normalized")
    for row in rows:
        for name in ("loss_metric", "normalized_metric", "weighted_loss"):
            value = float(row[name])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"evaluation row {name} is invalid")
    loss = sum(float(row["weighted_loss"]) for row in rows)
    min_valid = min(float(row["valid_fraction"]) for row in rows)
    max_edge = max(float(row["edge_fraction"]) for row in rows)
    max_energy_error = max(abs(float(row["energy"]) - 1.0) for row in rows)
    by_distance: dict[str, dict[str, float | int]] = {}
    by_zone: dict[str, dict[str, float | int]] = {}
    for group_name, target in (("distance_label", by_distance), ("zone", by_zone)):
        groups = sorted({str(row[group_name]) for row in rows})
        for group in groups:
            members = [row for row in rows if str(row[group_name]) == group]
            target[group] = {
                "case_count": len(members),
                "mean_m2_mm2": sum(float(row["m2_mm2"]) for row in members) / len(members),
                "mean_astig_A_D": sum(float(row["astig_A_D"]) for row in members) / len(members),
                "mean_normalized_metric": sum(float(row["normalized_metric"]) for row in members) / len(members),
                "weighted_loss": sum(float(row["weighted_loss"]) for row in members),
                "mean_valid_fraction": sum(float(row["valid_fraction"]) for row in members) / len(members),
            }
    health = {
        "case_count": len(rows),
        "minimum_valid_fraction": min_valid,
        "maximum_edge_fraction": max_edge,
        "maximum_energy_error": max_energy_error,
        "by_distance": by_distance,
        "by_zone": by_zone,
    }
    return float(loss), health


def _save_evaluation_progress(
    path: Path,
    *,
    identity_sha256: str,
    case_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    _torch_save_atomic(
        path,
        {
            "schema_version": EVALUATION_PROGRESS_SCHEMA_VERSION,
            "identity_sha256": identity_sha256,
            "case_ids": list(case_ids),
            "next_case_index": len(rows),
            "rows": [dict(row) for row in rows],
        },
    )


def _evaluate(
    model: MinimalOpticalModel,
    cases: Sequence[Mapping[str, Any]],
    *,
    with_grad: bool,
    baseline_metrics: Mapping[str, float] | None = None,
    progress_path: Path | None = None,
    identity_sha256: str | None = None,
    print_progress: bool = True,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    """Evaluate true case batches and backpropagate once per positive-loss batch."""
    if not cases:
        raise ValueError("cannot evaluate an empty case set")
    if with_grad and progress_path is not None:
        raise ValueError("gradient evaluations cannot resume from a partial case sweep")
    case_ids = [str(case["case_id"]) for case in cases]
    rows: list[dict[str, Any]] = []
    start_index = 0
    if progress_path is not None and progress_path.is_file():
        if identity_sha256 is None:
            raise ValueError("evaluation progress requires an identity hash")
        payload = torch.load(progress_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("evaluation progress is malformed")
        if int(payload.get("schema_version", -1)) != EVALUATION_PROGRESS_SCHEMA_VERSION:
            raise ValueError("evaluation progress schema mismatch")
        if str(payload.get("identity_sha256")) != identity_sha256:
            raise ValueError("evaluation progress identity mismatch")
        if payload.get("case_ids") != case_ids:
            raise ValueError("evaluation progress case order changed")
        rows = [dict(row) for row in payload.get("rows", [])]
        start_index = int(payload.get("next_case_index", -1))
        if start_index != len(rows) or start_index > len(cases):
            raise ValueError("evaluation progress index is inconsistent")
    batch_size = int(model.config.case_batch_size)
    if batch_size <= 0:
        raise ValueError("model case_batch_size must be positive")
    if start_index < len(cases) and start_index % batch_size != 0:
        raise ValueError("evaluation progress does not end at a complete case batch")
    total_batches = (len(cases) + batch_size - 1) // batch_size
    for batch_start in range(start_index, len(cases), batch_size):
        batch_cases = cases[batch_start : batch_start + batch_size]
        batch_end = batch_start + len(batch_cases)
        weights = [float(case["objective_weight"]) for case in batch_cases]
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("case batch has invalid objective weights")
        with torch.set_grad_enabled(with_grad):
            result = model.field_batch(batch_cases)
            if result.psf.ndim != 3 or int(result.psf.shape[0]) != len(batch_cases):
                raise ValueError("field_batch must return PSF shape [B,H,W]")
            energy = result.psf.sum(dim=(-2, -1))
            if not bool(torch.isfinite(energy).all()) or bool(
                ((energy - 1.0).abs() > 1.0e-10).any()
            ):
                raise ValueError("physical PSF batch energy is not one")
            moments = psf_second_moment_mm2(
                result.psf, pixel_pitch_mm=result.pixel_pitch_mm
            )
            astig_by_zone: dict[str, torch.Tensor] = {}
            if any(
                str(case["zone"]) in ("astig_left", "astig_right")
                for case in batch_cases
            ):
                astig_by_zone = model.astig_A_by_zone()
            metric_tensors: list[torch.Tensor] = []
            metric_names: list[str] = []
            baseline_values: list[float] = []
            for case_index, case in enumerate(batch_cases):
                zone = str(case["zone"])
                if zone in ("astig_left", "astig_right"):
                    metric_tensors.append(astig_by_zone[zone])
                    metric_names.append("astig_A_D")
                else:
                    metric_tensors.append(moments[case_index])
                    metric_names.append("m2_mm2")
                if baseline_metrics is None:
                    baseline_values.append(1.0)
                else:
                    key = _metric_key(case)
                    if key not in baseline_metrics:
                        raise ValueError(f"missing baseline metric for {key}")
                    denominator = float(baseline_metrics[key])
                    if not math.isfinite(denominator) or denominator <= 0.0:
                        raise ValueError(f"baseline metric is invalid for {key}")
                    baseline_values.append(denominator)
            metrics = torch.stack(metric_tensors)
            baseline_tensor = torch.as_tensor(
                baseline_values, device=metrics.device, dtype=metrics.dtype
            )
            normalized_metrics = metrics / baseline_tensor
            weight_tensor = torch.as_tensor(
                weights, device=metrics.device, dtype=metrics.dtype
            )
            weighted = normalized_metrics * weight_tensor
            positive = weight_tensor > 0.0
            batch_loss = weighted[positive].sum()
            if with_grad and bool(positive.any()):
                if not batch_loss.requires_grad:
                    raise RuntimeError("positive case-batch loss is detached from PAL parameters")
                batch_loss.backward()
        valid_fractions = result.valid_fraction.detach().cpu().tolist()
        edge_fractions = result.edge_fraction.detach().cpu().tolist()
        moment_values = moments.detach().cpu().tolist()
        metric_values = metrics.detach().cpu().tolist()
        normalized_values = normalized_metrics.detach().cpu().tolist()
        weighted_values = weighted.detach().cpu().tolist()
        energy_values = energy.detach().cpu().tolist()
        pitch_values = torch.as_tensor(result.pixel_pitch_mm).detach().cpu().reshape(-1).tolist()
        valid_counts = (
            result.valid_mask.sum(dim=1).detach().cpu().tolist()
            if result.valid_mask is not None
            else [None] * len(batch_cases)
        )
        ray_counts = (
            [int(result.valid_mask.shape[1])] * len(batch_cases)
            if result.valid_mask is not None
            else [None] * len(batch_cases)
        )
        for case_index, case in enumerate(batch_cases):
            rows.append(
                {
                    **dict(case),
                    "m2_mm2": float(moment_values[case_index]),
                    "astig_A_D": float(metric_values[case_index])
                    if metric_names[case_index] == "astig_A_D"
                    else 0.0,
                    "loss_metric": float(metric_values[case_index]),
                    "loss_metric_name": metric_names[case_index],
                    "baseline_metric": float(baseline_values[case_index]),
                    "normalized_metric": float(normalized_values[case_index]),
                    "weighted_loss": float(weighted_values[case_index]),
                    "weighted_m2_mm2": float(moment_values[case_index] * weights[case_index]),
                    "energy": float(energy_values[case_index]),
                    "valid_fraction": float(valid_fractions[case_index]),
                    "valid_ray_count": None
                    if valid_counts[case_index] is None
                    else int(valid_counts[case_index]),
                    "ray_count": ray_counts[case_index],
                    "edge_fraction": float(edge_fractions[case_index]),
                    "pixel_pitch_mm": float(pitch_values[case_index]),
                }
            )
        result_device = result.psf.device
        batch_loss_value = float(batch_loss.detach().cpu())
        del (
            result,
            energy,
            moments,
            metrics,
            normalized_metrics,
            weighted,
            batch_loss,
            astig_by_zone,
        )
        _release_inactive_case_cuda_cache(result_device)
        if progress_path is not None:
            _save_evaluation_progress(
                progress_path,
                identity_sha256=str(identity_sha256),
                case_ids=case_ids,
                rows=rows,
            )
        batch_number = batch_start // batch_size + 1
        if print_progress and (
            batch_number == total_batches or batch_number % 8 == 0
        ):
            print(
                "[pal-eval] "
                f"batch {batch_number}/{total_batches} "
                f"cases {batch_start + 1}-{batch_end}/{len(cases)} "
                f"grad={with_grad} loss={batch_loss_value:.6g}",
                flush=True,
            )
    loss, health = _summarize_rows(rows)
    return loss, rows, health


def _power_and_sag(
    base_sag: torch.Tensor,
    module: FixedWeightNURBSPerturbation,
    power_config: PALPowerConfig,
    zones: Mapping[str, torch.Tensor],
) -> tuple[dict[str, float], torch.Tensor, float]:
    with torch.no_grad():
        coord = torch.linspace(
            -power_config.semi_diameter_mm,
            power_config.semi_diameter_mm,
            int(base_sag.shape[0]),
            device=base_sag.device,
            dtype=base_sag.dtype,
        )
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        delta = module.delta_raw(xx, yy)
        metrics = prescription_metrics(base_sag + delta, power_config, zones)
        power = {name: float(value.detach().cpu()) for name, value in metrics.items()}
        max_abs_delta = float(delta.detach().abs().max().cpu())
    return power, delta, max_abs_delta


def _feasibility(
    health: Mapping[str, Any],
    power: Mapping[str, float],
    baseline_power: Mapping[str, float],
    config: MinimalConfig,
    *,
    step_sag_mm: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if float(health["minimum_valid_fraction"]) < float(config.minimum_valid_fraction):
        reasons.append("valid_fraction")
    if float(health["maximum_edge_fraction"]) > float(config.maximum_edge_fraction):
        reasons.append("edge_fraction")
    if abs(float(power["P_far_D"]) - float(baseline_power["P_far_D"])) > float(config.far_tolerance_D):
        reasons.append("P_far")
    if abs(float(power["ADD_D"]) - float(baseline_power["ADD_D"])) > float(config.add_tolerance_D):
        reasons.append("ADD")
    if float(step_sag_mm) > float(config.step_sag_limit_mm):
        reasons.append("step_sag")
    return not reasons, reasons


def _save_checkpoint(
    path: Path,
    module: FixedWeightNURBSPerturbation,
    *,
    identity_sha256: str,
    case_layout_sha256: str,
    step: int,
    normalized_loss: float,
    health: Mapping[str, Any],
    power: Mapping[str, float],
    feasible: bool,
) -> None:
    _torch_save_atomic(
        path,
        {
            "method": METHOD_NAME,
            "identity_sha256": identity_sha256,
            "case_layout_sha256": case_layout_sha256,
            "control_count": CONTROL_COUNT,
            "step": int(step),
            "normalized_loss": float(normalized_loss),
            "health": dict(health),
            "power": dict(power),
            "feasible": bool(feasible),
            "state_dict": copy.deepcopy(module.state_dict()),
        },
    )


def _make_training_resume_payload(
    *,
    identity_sha256: str,
    case_layout_sha256: str,
    module: FixedWeightNURBSPerturbation,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    completed_attempts: int,
    max_accepted_steps: int,
    accepted_steps: int,
    no_improvement_accepted_steps: int,
    history: Sequence[Mapping[str, Any]],
    best_normalized_loss: float,
    best_step: int,
    best_state: Mapping[str, Any],
    best_health: Mapping[str, Any],
    best_power: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_RESUME_SCHEMA_VERSION,
        "method": METHOD_NAME,
        "identity_sha256": identity_sha256,
        "case_layout_sha256": case_layout_sha256,
        "control_count": CONTROL_COUNT,
        "max_accepted_steps": int(max_accepted_steps),
        "completed_attempts": int(completed_attempts),
        "accepted_steps": int(accepted_steps),
        "no_improvement_accepted_steps": int(no_improvement_accepted_steps),
        "model_state": copy.deepcopy(module.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "learning_rate": float(learning_rate),
        "history": [dict(row) for row in history],
        "best_normalized_loss": float(best_normalized_loss),
        "best_step": int(best_step),
        "best_state": copy.deepcopy(dict(best_state)),
        "best_health": dict(best_health),
        "best_power": dict(best_power),
        "rng_state": _capture_rng_state(),
    }


def _validate_training_resume(
    payload: Mapping[str, Any],
    *,
    identity_sha256: str,
    case_layout_sha256: str,
    max_accepted_steps: int,
) -> None:
    if int(payload.get("schema_version", -1)) != TRAINING_RESUME_SCHEMA_VERSION:
        raise ValueError("training resume schema mismatch")
    if payload.get("method") != METHOD_NAME:
        raise ValueError("training resume method mismatch")
    if str(payload.get("identity_sha256")) != identity_sha256:
        raise ValueError("training resume identity mismatch")
    if str(payload.get("case_layout_sha256")) != case_layout_sha256:
        raise ValueError("training resume case layout mismatch")
    if int(payload.get("control_count", -1)) != CONTROL_COUNT:
        raise ValueError("training resume control count mismatch")
    if int(payload.get("max_accepted_steps", -1)) != int(max_accepted_steps):
        raise ValueError("training resume max_accepted_steps mismatch")
    completed = int(payload.get("completed_attempts", -1))
    accepted = int(payload.get("accepted_steps", -1))
    no_improvement = int(payload.get("no_improvement_accepted_steps", -1))
    history = payload.get("history")
    if not isinstance(history, list) or completed != len(history) or completed < 0:
        raise ValueError("training resume attempt/history mismatch")
    if accepted < 0 or accepted > max_accepted_steps or no_improvement < 0:
        raise ValueError("training resume accepted-step state is invalid")
    if [int(row.get("attempt", -1)) for row in history] != list(range(1, completed + 1)):
        raise ValueError("training resume history is not contiguous")
    for name in ("model_state", "optimizer_state", "best_state", "best_health", "best_power", "rng_state"):
        if not isinstance(payload.get(name), dict):
            raise ValueError(f"training resume lacks {name}")


def _restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    state: Mapping[str, Any],
    module: FixedWeightNURBSPerturbation,
) -> None:
    optimizer.load_state_dict(dict(state))
    device = module.inner_q.device
    for parameter, parameter_state in optimizer.state.items():
        if parameter is not module.inner_q:
            raise ValueError("restored optimizer is bound to a different PAL parameter")
        for name, value in list(parameter_state.items()):
            if torch.is_tensor(value):
                parameter_state[name] = value.to(device=device)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def _load_baseline_state(
    path: Path,
    *,
    identity_sha256: str,
    case_layout_sha256: str,
    module: FixedWeightNURBSPerturbation,
    device: torch.device,
) -> dict[str, Any]:
    payload = _load_identity_bound_torch(
        path,
        identity_sha256=identity_sha256,
        schema_version=EVALUATION_PROGRESS_SCHEMA_VERSION,
        map_location=device,
    )
    if payload.get("method") != METHOD_NAME:
        raise ValueError("baseline state method mismatch")
    if str(payload.get("case_layout_sha256")) != case_layout_sha256:
        raise ValueError("baseline state case layout mismatch")
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("baseline state model payload is malformed")
    expected = module.state_dict()
    if set(state) != set(expected):
        raise ValueError("baseline state parameter keys changed")
    for name, expected_value in expected.items():
        value = state[name]
        if not torch.is_tensor(value):
            raise ValueError("baseline state contains a non-tensor parameter")
        if not torch.equal(value.to(device=expected_value.device, dtype=expected_value.dtype), expected_value):
            raise ValueError("baseline state is not the zero-residual 7x7 PAL")
    if int(payload.get("case_count", -1)) <= 0:
        raise ValueError("baseline state case count is invalid")
    metrics = payload.get("baseline_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("baseline state lacks baseline metrics")
    return payload


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

    training_log_path = output / "training.log"

    def log_progress(message: str) -> None:
        line = f"[pal-train] {message}"
        _append_training_log(training_log_path, line)
        print(line, flush=True)

    summary_path = output / "summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if str(summary.get("identity_sha256")) != identity_sha256:
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
    module = FixedWeightNURBSPerturbation(
        max_abs_control_mm=config.max_abs_control_mm,
        device=device,
        dtype=torch.float64,
    )
    model = MinimalOpticalModel(config, module)
    model.set_prescription_context(base_sag, power_config, zones)
    try:
        write_state("case_layout")
        cases = _prepare_or_load_case_layout(
            config, output, model, identity_sha256=identity_sha256
        )
        case_layout_sha256 = _canonical_json_sha256(cases)
        case_ids = [str(case["case_id"]) for case in cases]

        baseline_path = output / "baseline.pt"
        if baseline_path.is_file():
            baseline = _load_baseline_state(
                baseline_path,
                identity_sha256=identity_sha256,
                case_layout_sha256=case_layout_sha256,
                module=module,
                device=device,
            )
            baseline_loss = float(baseline["normalized_loss"])
            baseline_metrics = {str(key): float(value) for key, value in baseline["baseline_metrics"].items()}
            baseline_rows = [dict(row) for row in baseline["rows"]]
            baseline_health = dict(baseline["health"])
            baseline_power = {name: float(value) for name, value in baseline["power"].items()}
            if baseline.get("case_ids") != case_ids:
                raise ValueError("baseline case order changed")
        else:
            write_state("baseline_psf_sweep")
            _, raw_baseline_rows, _ = _evaluate(
                model,
                cases,
                with_grad=False,
                progress_path=output / "baseline_progress.pt",
                identity_sha256=identity_sha256,
            )
            baseline_metrics = _baseline_metric_table(raw_baseline_rows)
            baseline_rows = _normalize_evaluation_rows(raw_baseline_rows, baseline_metrics)
            baseline_loss, baseline_health = _summarize_rows(baseline_rows)
            baseline_power, _, baseline_sag = _power_and_sag(
                base_sag, module, power_config, zones
            )
            baseline_feasible, baseline_reasons = _feasibility(
                baseline_health,
                baseline_power,
                baseline_power,
                config,
                step_sag_mm=0.0,
            )
            if not baseline_feasible:
                raise RuntimeError(
                    "Original PAL baseline is not feasible for the fixed layout: "
                    + ", ".join(baseline_reasons)
                )
            _torch_save_atomic(
                baseline_path,
                {
                    "schema_version": EVALUATION_PROGRESS_SCHEMA_VERSION,
                    "method": METHOD_NAME,
                    "identity_sha256": identity_sha256,
                    "case_layout_sha256": case_layout_sha256,
                    "case_ids": case_ids,
                    "case_count": len(cases),
                    "model_state": copy.deepcopy(module.state_dict()),
                    "normalized_loss": baseline_loss,
                    "baseline_metrics": baseline_metrics,
                    "health": baseline_health,
                    "power": baseline_power,
                    "rows": baseline_rows,
                    "max_abs_sag_delta_mm": baseline_sag,
                },
            )
            (output / "baseline_progress.pt").unlink(missing_ok=True)
        expected_metric_keys = {
            f"{zone}/{spec.label}"
            for zone in PARTITION_ZONES
            for spec in DISTANCE_SPECS
        }
        if set(baseline_metrics) != expected_metric_keys:
            raise ValueError("baseline metrics do not cover all zone/distance combinations")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in baseline_metrics.values()
        ):
            raise ValueError("baseline metrics must be finite and positive")
        baseline_rows = _normalize_evaluation_rows(baseline_rows, baseline_metrics)
        baseline_loss, baseline_health = _summarize_rows(baseline_rows)
        _torch_save_atomic(
            output / "baseline_metrics.pt",
            {
                "method": METHOD_NAME,
                "identity_sha256": identity_sha256,
                "case_layout_sha256": case_layout_sha256,
                "normalized_loss": baseline_loss,
                "baseline_metrics": baseline_metrics,
                "health": baseline_health,
                "power": baseline_power,
                "rows": baseline_rows,
            },
        )
        write_state("baseline_complete", normalized_loss=baseline_loss)
        log_progress(
            f"baseline complete cases={len(cases)} loss={baseline_loss:.6g} "
            f"batch_size={config.case_batch_size}"
        )

        resume_path = output / "resume.pt"
        optimizer = torch.optim.Adam([module.inner_q], lr=config.learning_rate)
        history: list[dict[str, Any]] = []
        completed_attempts = 0
        accepted_steps = 0
        no_improvement_accepted_steps = 0
        learning_rate = float(config.learning_rate)
        best_loss = float(baseline_loss)
        best_step = 0
        best_state = copy.deepcopy(module.state_dict())
        best_health = dict(baseline_health)
        best_power = dict(baseline_power)
        if resume_path.is_file():
            payload = _load_identity_bound_torch(
                resume_path,
                identity_sha256=identity_sha256,
                schema_version=TRAINING_RESUME_SCHEMA_VERSION,
                map_location=device,
            )
            _validate_training_resume(
                payload,
                identity_sha256=identity_sha256,
                case_layout_sha256=case_layout_sha256,
                max_accepted_steps=config.max_accepted_steps,
            )
            module.load_state_dict(payload["model_state"])
            _restore_optimizer_state(optimizer, payload["optimizer_state"], module)
            learning_rate = float(payload["learning_rate"])
            completed_attempts = int(payload["completed_attempts"])
            accepted_steps = int(payload["accepted_steps"])
            no_improvement_accepted_steps = int(payload["no_improvement_accepted_steps"])
            history = [dict(row) for row in payload["history"]]
            best_loss = float(payload["best_normalized_loss"])
            best_step = int(payload["best_step"])
            best_state = copy.deepcopy(payload["best_state"])
            best_health = dict(payload["best_health"])
            best_power = {name: float(value) for name, value in payload["best_power"].items()}
            _restore_rng_state(payload["rng_state"])
            if history:
                _write_history(output / "history.csv", history)
            log_progress(
                f"resume attempt={completed_attempts} accepted={accepted_steps}/"
                f"{config.max_accepted_steps} best={best_loss:.6g} "
                f"patience={no_improvement_accepted_steps}/{config.early_stopping_patience} "
                f"lr={learning_rate:.6g}"
            )
        else:
            _save_checkpoint(
                output / "initial.pt",
                module,
                identity_sha256=identity_sha256,
                case_layout_sha256=case_layout_sha256,
                step=0,
                normalized_loss=baseline_loss,
                health=baseline_health,
                power=baseline_power,
                feasible=True,
            )
            _save_checkpoint(
                output / "best_feasible.pt",
                module,
                identity_sha256=identity_sha256,
                case_layout_sha256=case_layout_sha256,
                step=0,
                normalized_loss=baseline_loss,
                health=baseline_health,
                power=baseline_power,
                feasible=True,
            )
            _torch_save_atomic(
                resume_path,
                _make_training_resume_payload(
                    identity_sha256=identity_sha256,
                    case_layout_sha256=case_layout_sha256,
                    module=module,
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    completed_attempts=0,
                    max_accepted_steps=config.max_accepted_steps,
                    accepted_steps=0,
                    no_improvement_accepted_steps=0,
                    history=history,
                    best_normalized_loss=best_loss,
                    best_step=best_step,
                    best_state=best_state,
                    best_health=best_health,
                    best_power=best_power,
                ),
            )
            log_progress(
                f"training start accepted=0/{config.max_accepted_steps} "
                f"patience=0/{config.early_stopping_patience} lr={learning_rate:.6g}"
            )

        attempt = completed_attempts
        while accepted_steps < int(config.max_accepted_steps):
            attempt += 1
            write_state(
                "training_sweep",
                attempt=attempt,
                accepted_steps=accepted_steps,
                max_accepted_steps=config.max_accepted_steps,
            )
            log_progress(
                f"attempt={attempt} evaluating accepted={accepted_steps}/"
                f"{config.max_accepted_steps} cases={len(cases)}"
            )
            module.zero_grad(set_to_none=True)
            loss, rows, health = _evaluate(
                model,
                cases,
                with_grad=True,
                baseline_metrics=baseline_metrics,
                print_progress=False,
            )
            power, old_delta, current_sag = _power_and_sag(
                base_sag, module, power_config, zones
            )
            feasible, reasons = _feasibility(
                health,
                power,
                baseline_power,
                config,
                step_sag_mm=0.0,
            )
            relative_improvement = (best_loss - float(loss)) / abs(best_loss)
            if feasible and loss < best_loss:
                best_loss = float(loss)
                best_step = int(accepted_steps)
                best_state = copy.deepcopy(module.state_dict())
                best_health = dict(health)
                best_power = dict(power)
                _save_checkpoint(
                    output / "best_feasible.pt",
                    module,
                    identity_sha256=identity_sha256,
                    case_layout_sha256=case_layout_sha256,
                    step=accepted_steps,
                    normalized_loss=best_loss,
                    health=best_health,
                    power=best_power,
                    feasible=True,
                )
            gradient = module.inner_q.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("PAL gradient is missing or non-finite")
            old_parameter = module.inner_q.detach().clone()
            old_optimizer_state = copy.deepcopy(optimizer.state_dict())
            optimizer.param_groups[0]["lr"] = learning_rate
            optimizer.step()
            update_applied = True
            update_reason = "applied"
            with torch.no_grad():
                control_bound_ok = bool(torch.all(module.inner_q.abs() <= 1.0))
                new_delta = module.delta_raw(
                    torch.linspace(
                        -power_config.semi_diameter_mm,
                        power_config.semi_diameter_mm,
                        int(base_sag.shape[1]),
                        device=base_sag.device,
                        dtype=base_sag.dtype,
                    ).reshape(1, -1).expand(int(base_sag.shape[0]), -1),
                    torch.linspace(
                        -power_config.semi_diameter_mm,
                        power_config.semi_diameter_mm,
                        int(base_sag.shape[0]),
                        device=base_sag.device,
                        dtype=base_sag.dtype,
                    ).reshape(-1, 1).expand(-1, int(base_sag.shape[1])),
                )
                step_sag = float((new_delta - old_delta).abs().max().cpu())
            if not control_bound_ok:
                update_applied = False
                update_reason = "control_bound"
            elif not math.isfinite(step_sag) or step_sag > float(config.step_sag_limit_mm):
                update_applied = False
                update_reason = "step_sag"
            else:
                candidate_power, _, _ = _power_and_sag(
                    base_sag, module, power_config, zones
                )
                if abs(candidate_power["P_far_D"] - baseline_power["P_far_D"]) > config.far_tolerance_D:
                    update_applied = False
                    update_reason = "P_far"
                elif abs(candidate_power["ADD_D"] - baseline_power["ADD_D"]) > config.add_tolerance_D:
                    update_applied = False
                    update_reason = "ADD"
            if not update_applied:
                with torch.no_grad():
                    module.inner_q.copy_(old_parameter)
                _restore_optimizer_state(optimizer, old_optimizer_state, module)
                learning_rate *= 0.5
            else:
                accepted_steps += 1
                if feasible and relative_improvement > float(config.relative_improvement_threshold):
                    no_improvement_accepted_steps = 0
                else:
                    no_improvement_accepted_steps += 1
            history.append(
                {
                    "attempt": attempt,
                    "accepted_steps": accepted_steps,
                    "evaluated_normalized_loss": float(loss),
                    "evaluated_feasible": bool(feasible),
                    "evaluated_infeasible_reasons": ",".join(reasons),
                    "evaluated_minimum_valid_fraction": float(health["minimum_valid_fraction"]),
                    "evaluated_maximum_edge_fraction": float(health["maximum_edge_fraction"]),
                    "evaluated_max_abs_sag_delta_mm": float(current_sag),
                    "update_applied": bool(update_applied),
                    "update_reason": update_reason,
                    "update_step_sag_mm": float(step_sag) if math.isfinite(step_sag) else math.nan,
                    "learning_rate": float(learning_rate),
                    "best_normalized_loss": float(best_loss),
                    "best_feasible_step": int(best_step),
                    "no_improvement_accepted_steps": no_improvement_accepted_steps,
                    "relative_improvement_threshold": float(config.relative_improvement_threshold),
                }
            )
            completed_attempts = attempt
            _write_history(output / "history.csv", history)
            _torch_save_atomic(
                resume_path,
                _make_training_resume_payload(
                    identity_sha256=identity_sha256,
                    case_layout_sha256=case_layout_sha256,
                    module=module,
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    completed_attempts=completed_attempts,
                    max_accepted_steps=config.max_accepted_steps,
                    accepted_steps=accepted_steps,
                    no_improvement_accepted_steps=no_improvement_accepted_steps,
                    history=history,
                    best_normalized_loss=best_loss,
                    best_step=best_step,
                    best_state=best_state,
                    best_health=best_health,
                    best_power=best_power,
                ),
            )
            update_label = "ACCEPT" if update_applied else f"REJECT:{update_reason}"
            feasible_label = "yes" if feasible else "no"
            log_progress(
                f"attempt={attempt} accepted={accepted_steps}/"
                f"{config.max_accepted_steps} update={update_label} "
                f"loss={float(loss):.6g} best={best_loss:.6g} feasible={feasible_label} "
                f"rel={relative_improvement:.3e} "
                f"lr={learning_rate:.6g} patience={no_improvement_accepted_steps}/"
                f"{config.early_stopping_patience}"
            )
            if no_improvement_accepted_steps >= int(config.early_stopping_patience):
                write_state(
                    "early_stopping",
                    accepted_steps=accepted_steps,
                    patience=config.early_stopping_patience,
                    relative_improvement_threshold=config.relative_improvement_threshold,
                )
                log_progress(
                    f"early stopping accepted={accepted_steps}/{config.max_accepted_steps} "
                    f"patience={no_improvement_accepted_steps}/{config.early_stopping_patience}"
                )
                break
            if learning_rate < float(config.minimum_learning_rate) and not update_applied:
                log_progress(
                    f"stop learning_rate={learning_rate:.6g} below minimum="
                    f"{config.minimum_learning_rate:.6g} after rejected update"
                )
                break

        if not (output / "best_feasible.pt").is_file():
            module.load_state_dict(best_state)
            _save_checkpoint(
                output / "best_feasible.pt",
                module,
                identity_sha256=identity_sha256,
                case_layout_sha256=case_layout_sha256,
                step=best_step,
                normalized_loss=best_loss,
                health=best_health,
                power=best_power,
                feasible=True,
            )
        module.load_state_dict(best_state)
        write_state("final_evaluation", best_feasible_step=best_step)
        log_progress(
            f"final evaluation best_step={best_step} best_loss={best_loss:.6g} "
            f"accepted={accepted_steps}/{config.max_accepted_steps} attempts={completed_attempts}"
        )
        final_loss, final_rows, final_health = _evaluate(
            model,
            cases,
            with_grad=False,
            baseline_metrics=baseline_metrics,
            print_progress=False,
        )
        final_power, final_delta, final_sag = _power_and_sag(
            base_sag, module, power_config, zones
        )
        _save_checkpoint(
            output / "final.pt",
            module,
            identity_sha256=identity_sha256,
            case_layout_sha256=case_layout_sha256,
            step=best_step,
            normalized_loss=final_loss,
            health=final_health,
            power=final_power,
            feasible=True,
        )
        runtime_seconds = prior_elapsed + time.time() - session_start
        summary = {
            "method": METHOD_NAME,
            "identity_sha256": identity_sha256,
            "case_layout_sha256": case_layout_sha256,
            "case_count": len(cases),
            "distance_specs": [
                {
                    "label": spec.label,
                    "object_distance_mm": spec.serialized_distance,
                    "focus_zone": spec.focus_zone,
                }
                for spec in DISTANCE_SPECS
            ],
            "fov_grid": {
                "count": config.fov_count,
                "min_deg": config.fov_min_deg,
                "max_deg": config.fov_max_deg,
            },
            "case_batch_size": config.case_batch_size,
            "case_batch_count": (len(cases) + config.case_batch_size - 1)
            // config.case_batch_size,
            "objective": "sum(zone_distance_weight * metric / baseline_metric), metric=M2 for far/corridor/near and M/A A for astig_left/right",
            "baseline_metrics": baseline_metrics,
            "initial_normalized_loss": baseline_loss,
            "final_normalized_loss": final_loss,
            "improvement_percent": 100.0 * (1.0 - final_loss / baseline_loss),
            "best_feasible_step": best_step,
            "completed_attempts": completed_attempts,
            "accepted_steps": accepted_steps,
            "max_accepted_steps": config.max_accepted_steps,
            "early_stopping_patience": config.early_stopping_patience,
            "relative_improvement_threshold": config.relative_improvement_threshold,
            "control_count": CONTROL_COUNT,
            "max_abs_sag_delta_mm": final_sag,
            "P_far_D": final_power["P_far_D"],
            "ADD_D": final_power["ADD_D"],
            "P_far_change_D": final_power["P_far_D"] - baseline_power["P_far_D"],
            "ADD_change_D": final_power["ADD_D"] - baseline_power["ADD_D"],
            "baseline_health": baseline_health,
            "final_health": final_health,
            "runtime_seconds": runtime_seconds,
            "training_log": "training.log",
        }
        _write_json_atomic(summary_path, summary)
        _write_run_state(
            output,
            identity_sha256=identity_sha256,
            status="complete",
            phase="complete",
            elapsed_seconds=runtime_seconds,
            best_feasible_step=best_step,
            final_normalized_loss=final_loss,
        )
        return output
    finally:
        model.close()


def run(config: MinimalConfig, *, resume: bool = False) -> Path:
    _seed_everything(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output, identity = _open_run_directory(config, resume=resume)
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
        _append_training_log(
            output / "training.log",
            f"[pal-train] INTERRUPTED phase={phase} error={type(exc).__name__}: {exc}",
        )
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
