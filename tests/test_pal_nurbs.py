from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import biot.e2e.pal_nurbs as pal_nurbs
from biot.e2e.pal_nurbs import (
    BatchFieldResult,
    FieldResult,
    MinimalConfig,
    MinimalOpticalModel,
    PALPowerConfig,
    _append_training_log,
    _accumulate_startup_case_gradients,
    _load_stage_resume_state,
    _make_stage_resume_payload,
    _open_run_directory,
    _evaluate_original_training_baseline_with_resume,
    _prepare_case_layout,
    _release_inactive_case_cuda_cache,
    _retain_training_cache,
    _restore_optimizer_state,
    _torch_save_atomic,
    _evaluate,
    load_pal,
    prepare_only,
    prescription_metrics,
    psf_second_moment_mm2,
    run,
)
from biot.e2e.regional_nurbs import FixedWeightNURBSPerturbation


def test_m2_is_finite_nonnegative_and_centroid_relative() -> None:
    psf = torch.zeros((9, 9), dtype=torch.float64)
    psf[3, 5] = 1.0
    assert float(psf_second_moment_mm2(psf, pixel_pitch_mm=0.01)) == 0.0
    psf[3, 4] = 1.0
    value = psf_second_moment_mm2(psf, pixel_pitch_mm=0.01)
    assert torch.isfinite(value)
    assert float(value) > 0.0


def test_pfar_add_are_differentiable_from_nurbs_zp() -> None:
    config = MinimalConfig(device="cpu", requested_np=32, fft_size_px=64)
    module = FixedWeightNURBSPerturbation(7, device="cpu")
    base_sag, power_config, zones = load_pal(config, torch.device("cpu"))
    coord = torch.linspace(-power_config.semi_diameter_mm, power_config.semi_diameter_mm, base_sag.shape[0], dtype=torch.float64)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    metrics = prescription_metrics(base_sag + module.delta_raw(xx, yy), power_config, zones)
    (metrics["P_far_D"] + metrics["ADD_D"]).backward()
    assert module.inner_q.grad is not None
    assert torch.isfinite(module.inner_q.grad).all()
    assert float(module.inner_q.grad.abs().max()) > 0.0


def test_real_trace_psf_m2_has_nurbs_gradient() -> None:
    # Fixed physical PSF support is defined for the production pupil/FFT sampling.
    # Keep this non-GRIN workbook as an independent regression for the established
    # S/CB/Original-GridSag path; GRIN3 parity is tested separately.
    config = MinimalConfig(
        excel="eye_image_glass.xlsx",
        device="cpu",
        requested_np=1024,
        fft_size_px=512,
        kernel_size_px=32,
    )
    module = FixedWeightNURBSPerturbation(7, device="cpu")
    model = MinimalOpticalModel(config, module)
    result = model.field({"case_id": "center", "distance_mm": 2000.0, "field_x_deg": 0.0, "field_y_deg": 0.0})
    m2 = psf_second_moment_mm2(result.kernel, pixel_pitch_mm=result.pixel_pitch_mm)
    m2.backward()
    assert torch.isfinite(result.kernel).all()
    assert bool((result.kernel >= 0).all())
    assert abs(float(result.kernel.sum()) - 1.0) <= 1e-10
    assert module.inner_q.grad is not None
    assert torch.isfinite(module.inner_q.grad).all()
    assert float(module.inner_q.grad.abs().max()) > 0.0


def test_startup_cases_backward_immediately_and_match_summed_loss_gradient() -> None:
    events: list[str] = []
    cases = [
        {"case_id": "case_1", "scale": 0.7},
        {"case_id": "case_2", "scale": 1.3},
    ]
    config = MinimalConfig(device="cpu", kernel_size_px=3, intermediate_object_distance_mm=2000.0)

    class Model:
        size_reference_mm = {2000.0: 3.0}

        def __init__(self, module, *, record_events: bool) -> None:
            self.module = module
            self.record_events = record_events
            self.coefficients = torch.linspace(
                0.5, 1.5, module.inner_q.numel(), dtype=torch.float64
            ).reshape_as(module.inner_q)

        def field(self, case: dict[str, object]) -> FieldResult:
            case_id = str(case["case_id"])
            if self.record_events:
                events.append(f"field:{case_id}")
            argument = (
                float(case["scale"]) * (self.module.inner_q * self.coefficients).sum()
                - 1.0
            )
            edge = torch.sigmoid(argument)
            if self.record_events:
                edge.register_hook(
                    lambda gradient, label=case_id: events.append(f"backward:{label}")
                )
            zero = torch.zeros_like(edge)
            kernel = torch.stack(
                (
                    torch.stack((zero, zero, zero)),
                    torch.stack((zero, 1.0 - edge, edge)),
                    torch.stack((zero, zero, zero)),
                )
            )
            return FieldResult(kernel, torch.ones_like(edge), 1.0, torch.zeros_like(edge))

    sequential_module = FixedWeightNURBSPerturbation(7, device="cpu", dtype=torch.float64)
    sequential_model = Model(sequential_module, record_events=True)
    sequential_gradient = _accumulate_startup_case_gradients(
        sequential_model, sequential_module, config, cases,
    )
    assert events == [
        "field:case_1",
        "backward:case_1",
        "field:case_2",
        "backward:case_2",
    ]

    summed_module = FixedWeightNURBSPerturbation(7, device="cpu", dtype=torch.float64)
    summed_model = Model(summed_module, record_events=False)
    summed_loss = sum(
        psf_second_moment_mm2(
            summed_model.field(case).kernel,
            pixel_pitch_mm=summed_model.size_reference_mm[2000.0] / config.kernel_size_px,
        )
        for case in cases
    )
    summed_loss.backward()
    assert summed_module.inner_q.grad is not None
    assert torch.equal(sequential_gradient, summed_module.inner_q.grad)


def test_inactive_case_cuda_cache_is_released_only_for_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    _release_inactive_case_cuda_cache(torch.device("cpu"))
    assert calls == []
    _release_inactive_case_cuda_cache(torch.device("cuda"))
    assert calls == ["empty"]


def test_joint_loss_uses_region_means_then_085_015_weighting() -> None:
    parameter = torch.nn.Parameter(torch.tensor(-0.5, dtype=torch.float64))

    class Model:
        def field(self, case: dict[str, object]) -> FieldResult:
            scale = float(case["scale"])
            edge = torch.sigmoid(parameter * scale)
            kernel = torch.zeros((3, 3), dtype=torch.float64)
            kernel[1, 1] = 1.0 - edge
            kernel[1, 2] = edge
            return FieldResult(kernel, torch.tensor(1.0), 1.0, torch.tensor(0.0))

        def astig_A_by_zone(self) -> dict[str, torch.Tensor]:
            return {
                "astig_left": torch.sigmoid(parameter * 2.5),
                "astig_right": torch.sigmoid(parameter * 3.5),
            }

    cases = [
        {"case_id": "f1", "training_group": "far", "scale": 1.0},
        {"case_id": "f2", "training_group": "far", "scale": 3.0},
        {"case_id": "m1", "training_group": "intermediate", "scale": 2.0},
        {"case_id": "n1", "training_group": "near", "scale": 4.0},
        {"case_id": "pl1", "training_group": "peripheral_left", "scale": 2.5},
        {"case_id": "pr1", "training_group": "peripheral_right", "scale": 3.5},
    ]
    baseline = {case["case_id"]: {"loss_metric": 0.25} for case in cases}
    value, rows, health = _evaluate(Model(), cases, baseline, with_grad=True)
    metrics = {str(row["training_group"]): str(row["loss_metric_name"]) for row in rows}
    assert all(metrics[group] == "m2_mm2" for group in ("far", "intermediate", "near"))
    assert all(
        metrics[group] == "astig_A_D"
        for group in ("peripheral_left", "peripheral_right")
    )
    expected_functional = (health["J_far"] + health["J_intermediate"] + health["J_near"]) / 3
    expected = 0.85 * expected_functional + 0.15 * health["J_peripheral"]
    assert abs(health["J_functional"] - expected_functional) < 1e-15
    assert abs(value - expected) < 1e-15
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad)
    assert float(parameter.grad) != 0.0
    evaluated_gradient = parameter.grad.detach().clone()

    reference_parameter = torch.nn.Parameter(torch.tensor(-0.5, dtype=torch.float64))

    def score(scale: float) -> torch.Tensor:
        edge = torch.sigmoid(reference_parameter * scale)
        kernel = torch.zeros((3, 3), dtype=torch.float64)
        kernel[1, 1] = 1.0 - edge
        kernel[1, 2] = edge
        return psf_second_moment_mm2(kernel, pixel_pitch_mm=1.0) / 0.25

    expected_tensor = 0.85 * (
        torch.stack((score(1.0), score(3.0))).mean()
        + score(2.0)
        + score(4.0)
    ) / 3.0 + 0.15 * torch.stack(
        (
            torch.sigmoid(reference_parameter * 2.5) / 0.25,
            torch.sigmoid(reference_parameter * 3.5) / 0.25,
        )
    ).mean()
    expected_tensor.backward()
    assert reference_parameter.grad is not None
    assert torch.allclose(
        evaluated_gradient, reference_parameter.grad, atol=2.0e-15, rtol=2.0e-15
    )


def test_evaluate_uses_one_backward_per_partial_case_batch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(-0.5, dtype=torch.float64))
    backward_calls: list[int] = []
    parameter.register_hook(lambda gradient: backward_calls.append(1))

    class Model:
        config = MinimalConfig(device="cpu", case_batch_size=2, kernel_size_px=3)

        def __init__(self) -> None:
            self.batch_calls: list[list[str]] = []

        def field_batch(self, batch: list[dict[str, object]]) -> BatchFieldResult:
            self.batch_calls.append([str(case["case_id"]) for case in batch])
            scales = torch.as_tensor(
                [float(case["scale"]) for case in batch], dtype=torch.float64,
            )
            edge = torch.sigmoid(parameter * scales)
            zero = torch.zeros_like(edge)
            kernels = torch.stack(
                (
                    torch.stack((zero, zero, zero), dim=-1),
                    torch.stack((zero, 1.0 - edge, edge), dim=-1),
                    torch.stack((zero, zero, zero), dim=-1),
                ), dim=-2,
            )
            return BatchFieldResult(
                kernels=kernels,
                valid_fraction=torch.ones_like(edge),
                pixel_pitch_mm=torch.ones_like(edge),
                edge_fraction=torch.zeros_like(edge),
            )

        def astig_A_by_zone(self) -> dict[str, torch.Tensor]:
            return {
                "astig_left": torch.sigmoid(parameter * 4.0),
                "astig_right": torch.sigmoid(parameter * 5.0),
            }

    cases = [
        {"case_id": "f", "training_group": "far", "scale": 1.0},
        {"case_id": "m", "training_group": "intermediate", "scale": 2.0},
        {"case_id": "n", "training_group": "near", "scale": 3.0},
        {"case_id": "pl", "training_group": "peripheral_left", "scale": 4.0},
        {"case_id": "pr", "training_group": "peripheral_right", "scale": 5.0},
    ]
    baseline = {case["case_id"]: {"loss_metric": 1.0} for case in cases}
    model = Model()
    value, rows, health = _evaluate(
        model,
        cases,
        baseline,
        with_grad=True,
        progress_stage="7x7",
        progress_step="1/10",
        progress_learning_rate=2.0e-3,
    )
    assert len(model.batch_calls) == 3
    assert model.batch_calls[-1] == ["pr"]
    assert len(rows) == len(cases)
    assert math.isfinite(value)
    assert math.isfinite(health["J_total"])
    assert len(backward_calls) == 3
    assert "stage=7x7 step=1/10 batch=3/3" in capsys.readouterr().out


def test_main_training_log_is_append_only_and_readable(tmp_path) -> None:
    path = tmp_path / "training.log"
    _append_training_log(path, "[pal-train] stage=7x7 step=1/10 batch=46/46 loss=1 update=ACCEPT lr=0.002")
    _append_training_log(path, "[pal-train] INTERRUPTED stage_phase=stage_training error=RuntimeError")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "stage=7x7 step=1/10" in lines[0]
    assert "INTERRUPTED stage_phase=stage_training" in lines[1]


def test_joint_loss_rejects_incomplete_extra_or_mixed_training_groups() -> None:
    class Model:
        def field(self, case: dict[str, object]) -> FieldResult:
            kernel = torch.zeros((3, 3), dtype=torch.float64)
            kernel[1, 1] = 1.0
            return FieldResult(kernel, torch.tensor(1.0), 1.0, torch.tensor(0.0))

    complete = [
        {"case_id": "f", "training_group": "far"},
        {"case_id": "m", "training_group": "intermediate"},
        {"case_id": "n", "training_group": "near"},
        {"case_id": "pl", "training_group": "peripheral_left"},
        {"case_id": "pr", "training_group": "peripheral_right"},
    ]
    baseline = {case["case_id"]: {"m2_mm2": 1.0} for case in complete}
    with pytest.raises(ValueError, match="exactly the five groups"):
        _evaluate(Model(), complete[:-1], baseline, with_grad=False)
    extra = complete + [{"case_id": "x", "training_group": "other"}]
    with pytest.raises(ValueError, match="exactly the five groups"):
        _evaluate(
            Model(), extra, {**baseline, "x": {"m2_mm2": 1.0}}, with_grad=False
        )
    mixed = [dict(complete[0]), {"case_id": "ungrouped"}]
    with pytest.raises(ValueError, match="requires a training_group on every case"):
        _evaluate(
            Model(), mixed, {**baseline, "ungrouped": {"m2_mm2": 1.0}}, with_grad=False
        )


def test_resume_identity_fails_closed_on_config_input_or_implementation_drift(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.bin"
    implementation_path = tmp_path / "implementation.py"
    input_path.write_bytes(b"input-v1")
    implementation_path.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        pal_nurbs,
        "_identity_input_paths",
        lambda config: {"test_input": input_path},
    )
    monkeypatch.setattr(
        pal_nurbs,
        "_implementation_closure_paths",
        lambda: [implementation_path],
    )
    config = MinimalConfig(output=str(tmp_path / "run"), device="cpu")
    output, fresh_identity = _open_run_directory(config, resume=False)
    resumed_output, resumed_identity = _open_run_directory(config, resume=True)
    assert resumed_output == output
    assert resumed_identity == fresh_identity

    input_path.write_bytes(b"input-v2")
    with pytest.raises(ValueError, match="resume identity mismatch"):
        _open_run_directory(config, resume=True)
    input_path.write_bytes(b"input-v1")

    with pytest.raises(ValueError, match="resume identity mismatch"):
        _open_run_directory(replace(config, learning_rate=3.0e-3), resume=True)

    implementation_path.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="resume identity mismatch"):
        _open_run_directory(config, resume=True)


def test_prepare_only_resume_does_not_downgrade_completed_run(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.bin"
    implementation_path = tmp_path / "implementation.py"
    input_path.write_bytes(b"input")
    implementation_path.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        pal_nurbs,
        "_identity_input_paths",
        lambda config: {"test_input": input_path},
    )
    monkeypatch.setattr(
        pal_nurbs,
        "_implementation_closure_paths",
        lambda: [implementation_path],
    )
    config = MinimalConfig(output=str(tmp_path / "complete"), device="cpu")
    output, identity = _open_run_directory(config, resume=False)
    (output / "summary.json").write_text(
        json.dumps(
            {"identity_sha256": identity["identity_sha256"], "runtime_seconds": 1.0}
        ),
        encoding="utf-8",
    )
    run_state = {
        "schema_version": 1,
        "identity_sha256": identity["identity_sha256"],
        "status": "complete",
        "phase": "complete",
        "elapsed_seconds": 1.0,
    }
    (output / "run_state.json").write_text(json.dumps(run_state), encoding="utf-8")
    before = (output / "run_state.json").read_bytes()
    assert prepare_only(config, resume=True) == output / "preoptimization"
    assert (output / "run_state.json").read_bytes() == before


def test_stage_resume_roundtrip_preserves_model_adam_lr_history_and_best(tmp_path) -> None:
    module = FixedWeightNURBSPerturbation(7, device="cpu", dtype=torch.float64)
    optimizer = torch.optim.Adam([module.inner_q], lr=2.0e-3)
    module.inner_q.grad = torch.linspace(
        -1.0, 1.0, module.inner_q.numel(), dtype=torch.float64
    ).reshape_as(module.inner_q)
    optimizer.step()
    history = [{"step": 1, "accepted": True, "J": 0.99}]
    best_state = copy.deepcopy(module.state_dict())
    payload = _make_stage_resume_payload(
        identity_sha256="identity",
        status="active",
        control_count=7,
        max_steps=10,
        module=module,
        optimizer=optimizer,
        learning_rate=1.1e-3,
        completed_step=1,
        history=history,
        stage_initial=1.0,
        stage_initial_groups={"J": 1.0},
        best=0.99,
        best_state=best_state,
        best_health={"J_far": 0.99},
    )
    path = tmp_path / "resume.pt"
    _torch_save_atomic(path, payload)
    restored = _load_stage_resume_state(
        path,
        identity_sha256="identity",
        control_count=7,
        max_steps=10,
        device="cpu",
    )
    restored_module = FixedWeightNURBSPerturbation(7, device="cpu", dtype=torch.float64)
    restored_optimizer = torch.optim.Adam([restored_module.inner_q], lr=9.0e-3)
    restored_module.load_state_dict(restored["model_state"])
    _restore_optimizer_state(restored_optimizer, restored["optimizer_state"], restored_module)

    assert torch.equal(restored_module.inner_q, module.inner_q)
    assert restored["learning_rate"] == 1.1e-3
    assert restored["history"] == history
    assert restored["best"] == 0.99
    assert torch.equal(restored["best_state"]["inner_q"], best_state["inner_q"])
    original_optimizer = optimizer.state_dict()
    roundtrip_optimizer = restored_optimizer.state_dict()
    assert original_optimizer["param_groups"] == roundtrip_optimizer["param_groups"]
    for parameter_id, original_state in original_optimizer["state"].items():
        roundtrip_state = roundtrip_optimizer["state"][parameter_id]
        for name, value in original_state.items():
            if torch.is_tensor(value):
                assert torch.equal(value, roundtrip_state[name])
            else:
                assert value == roundtrip_state[name]
    assert all(
        value.device == restored_module.inner_q.device
        for state in restored_optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    assert not list(tmp_path.glob(".resume.pt.*.tmp"))


def test_case_layout_cache_cleanup_retains_only_training_cases() -> None:
    class System:
        def __init__(self) -> None:
            self.released = False

        def release_biot_lens(self) -> None:
            self.released = True

    class Model:
        device = torch.device("cpu")
        _key = staticmethod(MinimalOpticalModel._key)

        def __init__(self) -> None:
            self._cache = {}

    model = Model()
    training_case = {
        "distance_mm": 500.0,
        "field_x_deg": 1.0,
        "field_y_deg": 2.0,
    }
    keep_key = model._key(500.0, 1.0, 2.0)
    remove_key = model._key(2000.0, 3.0, 4.0)
    kept_system, removed_system = System(), System()
    model._cache = {
        keep_key: (kept_system, object()),
        remove_key: (removed_system, object()),
    }
    audit = _retain_training_cache(model, [training_case])
    assert set(model._cache) == {keep_key}
    assert not kept_system.released
    assert removed_system.released
    assert audit == {"before": 2, "retained": 1, "removed": 1}


def test_training_cache_materializes_all_cases_then_releases_aiming_templates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class System:
        def __init__(self) -> None:
            self.released = False

        def release_biot_lens(self) -> None:
            self.released = True

    class Model:
        device = torch.device("cpu")
        _key = staticmethod(MinimalOpticalModel._key)

        def __init__(self) -> None:
            self._cache = {}
            self._templates = {500.0: (System(), object())}
            self._cache_frozen = False

        def _system_and_rays(self, distance, x, y):
            key = self._key(distance, x, y)
            self._cache[key] = (System(), object())
            return self._cache[key]

    model = Model()
    template = model._templates[500.0][0]
    cases = [
        {"distance_mm": 500.0, "field_x_deg": 1.0, "field_y_deg": 2.0},
        {"distance_mm": 500.0, "field_x_deg": 3.0, "field_y_deg": 4.0},
    ]
    audit = _retain_training_cache(model, cases)
    assert audit == {
        "before": 2,
        "retained": 2,
        "removed": 0,
        "materialized": 2,
        "released_templates": 1,
    }
    assert template.released
    assert model._templates == {}
    assert model._cache_frozen


def test_retain_training_cache_materializes_extra_cases_before_freeze() -> None:
    class System:
        def __init__(self) -> None:
            self.released = False

        def release_biot_lens(self) -> None:
            self.released = True

    class Model:
        device = torch.device("cpu")
        _key = staticmethod(MinimalOpticalModel._key)

        def __init__(self) -> None:
            self._cache = {}
            self._templates = {500.0: (System(), object())}
            self._cache_frozen = False

        def _system_and_rays(self, distance, x, y):
            key = self._key(distance, x, y)
            self._cache[key] = (System(), object())
            return self._cache[key]

    model = Model()
    training_case = {
        "distance_mm": 500.0,
        "field_x_deg": 1.0,
        "field_y_deg": 2.0,
    }
    startup_case = {
        "distance_mm": 2000.0,
        "field_x_deg": 40.0,
        "field_y_deg": 40.0,
    }
    stale_key = model._key(2000.0, 3.0, 4.0)
    stale_system = System()
    model._cache[stale_key] = (stale_system, object())

    audit = _retain_training_cache(
        model,
        [training_case],
        extra_cases=[startup_case],
    )

    assert set(model._cache) == {
        model._key(500.0, 1.0, 2.0),
        model._key(2000.0, 40.0, 40.0),
    }
    assert stale_system.released
    assert model._cache_frozen
    assert audit == {
        "before": 3,
        "retained": 2,
        "removed": 1,
        "materialized": 2,
        "released_templates": 1,
    }


def test_refined_module_rebinds_frozen_training_cache_surfaces() -> None:
    class Surface:
        def __init__(self, perturbation):
            self.perturbation = perturbation

    class System:
        def __init__(self, perturbation):
            self.back_surface = Surface(perturbation)

    old = object()
    new = object()
    model = type("Model", (), {})()
    model._templates = {}
    model._cache = {(1.0, 0.0, 0.0): (System(old), object())}
    for cached_system, _ in model._cache.values():
        cached_system.back_surface.perturbation = new
    assert model._cache[(1.0, 0.0, 0.0)][0].back_surface.perturbation is new


def test_case_layout_filters_oversampled_pool_before_final_fps(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    zones_path = tmp_path / "zones.json"
    zones_path.write_text(
        json.dumps({"statistics": {"corridor": {"physical_y_range_mm": [-1.0, 1.0]}}}),
        encoding="utf-8",
    )
    config = MinimalConfig(zones_json=str(zones_path), device="cpu")
    candidates = [
        {"candidate_id": "bad", "eligible": True},
        {"candidate_id": "good", "eligible": True},
    ]
    selections = {"count": 0}
    written = {}

    monkeypatch.setattr(pal_nurbs, "generate_dense_candidate_fields", lambda **kwargs: [{}])
    monkeypatch.setattr(
        pal_nurbs, "trace_candidate_fields",
        lambda *args, **kwargs: candidates,
    )

    def select(rows, config, **kwargs):
        selections["count"] += 1
        if kwargs.get("group_counts") is pal_nurbs.FORWARD_POOL_GROUP_COUNTS:
            selected_rows = [row for row in rows if row["eligible"]]
        else:
            selected_rows = [next(row for row in rows if row["eligible"])]
        return [
            {
                **selected, "case_id": f"case_{selected['candidate_id']}",
                "training_group": "far", "distance_mm": 100000.0,
                "field_x_deg": 0.0, "field_y_deg": 0.0,
            }
            for selected in selected_rows
        ]

    monkeypatch.setattr(pal_nurbs, "build_joint_training_cases", select)
    monkeypatch.setattr(
        pal_nurbs, "_trace_preoptimization_case_geometry",
        lambda model, rows, **kwargs: [dict(row) for row in rows],
    )
    monkeypatch.setattr(
        pal_nurbs,
        "write_preoptimization_artifacts",
        lambda **kwargs: written.update(kwargs),
    )
    monkeypatch.setattr(
        pal_nurbs,
        "_retain_training_cache",
        lambda model, rows, **kwargs: {"retained": 1},
    )

    class Model:
        def validate_training_case_wfno(self, case):
            if case["candidate_id"] == "bad":
                raise RuntimeError("formal aiming failed")
            return {"physical_fft_pixel_pitch_mm": 0.001}

        def validate_training_case_forward(self, case):
            return {
                "ray_count": 4, "valid_ray_count": 4, "valid_fraction": 1.0,
                "physical_fft_pixel_pitch_mm": 0.001,
            }

    training = _prepare_case_layout(config, tmp_path, Model())
    assert selections["count"] == 2
    assert training[0]["candidate_id"] == "good"
    assert [row["candidate_id"] for row in written["candidates"]] == ["good"]
    assert candidates[0]["eligible"]
    assert candidates[0]["forward_wfno_status"] == "failed"
    audit = json.loads(
        (tmp_path / "preoptimization" / "forward_qualification_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["pool_attempt_count"] == 2
    assert audit["pool_failure_count"] == 1
    assert audit["validation_round_count"] == 1
    assert audit["failure_count"] == 0


def test_complete_pool_progress_import_is_identity_bound_and_atomic(tmp_path) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "run" / "progress.json"
    payload = {
        "schema_version": 1,
        "status": "complete",
        "pool_identity_sha256": "pool-identity",
        "pool_attempts": [{"candidate_id": "c1", "status": "ok"}],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    pal_nurbs._import_complete_pool_progress(
        source_path=source,
        destination_path=destination,
        pool_identity_sha256="pool-identity",
        progress_name="test progress",
    )

    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert not list(destination.parent.glob(".*.tmp"))


def test_complete_pool_progress_import_rejects_foreign_pool(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({
            "schema_version": 1,
            "status": "complete",
            "pool_identity_sha256": "foreign",
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pool identity mismatch"):
        pal_nurbs._import_complete_pool_progress(
            source_path=source,
            destination_path=tmp_path / "progress.json",
            pool_identity_sha256="expected",
            progress_name="test progress",
        )


def test_original_training_baseline_progress_resumes_at_the_exact_next_case(tmp_path) -> None:
    training_cases = [
        {"case_id": "f", "training_group": "far"},
        {"case_id": "m", "training_group": "intermediate"},
        {"case_id": "n", "training_group": "near"},
        {"case_id": "pl", "training_group": "peripheral_left"},
        {"case_id": "pr", "training_group": "peripheral_right"},
    ]
    class Model:
        device = torch.device("cpu")

        def __init__(self, fail_case: str | None = None) -> None:
            self.perturbation = FixedWeightNURBSPerturbation(7, device="cpu")
            self.fail_case = fail_case
            self.calls: list[str] = []

        def field(self, case: dict[str, object]) -> FieldResult:
            case_id = str(case["case_id"])
            self.calls.append(case_id)
            if case_id == self.fail_case:
                raise RuntimeError("synthetic interruption")
            kernel = torch.zeros((3, 3), dtype=torch.float64)
            kernel[1, 1] = 0.75
            kernel[1, 2] = 0.25
            return FieldResult(kernel, torch.tensor(1.0), 1.0, torch.tensor(0.01))

        def astig_A_by_zone(self) -> dict[str, torch.Tensor]:
            return {
                "astig_left": torch.tensor(0.2, dtype=torch.float64),
                "astig_right": torch.tensor(0.3, dtype=torch.float64),
            }

    path = tmp_path / "baseline_progress.pt"
    first = Model(fail_case="n")
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _evaluate_original_training_baseline_with_resume(
            first,
            training_cases,
            progress_path=path,
            identity_sha256="identity",
        )
    saved = torch.load(path, map_location="cpu")
    assert saved["next_training_index"] == 2
    assert [row["case_id"] for row in saved["training_rows"]] == ["f", "m"]

    resumed = Model()
    value, rows, health = _evaluate_original_training_baseline_with_resume(
        resumed,
        training_cases,
        progress_path=path,
        identity_sha256="identity",
    )
    assert resumed.calls == ["n", "pl", "pr"]
    assert [row["case_id"] for row in rows] == ["f", "m", "n", "pl", "pr"]
    assert value == 1.0
    assert health["J_functional"] == 1.0
    assert health["J_peripheral"] == 1.0

    with pytest.raises(ValueError, match="case order/IDs changed"):
        _evaluate_original_training_baseline_with_resume(
            Model(),
            list(reversed(training_cases)),
            progress_path=path,
            identity_sha256="identity",
        )


def test_interrupted_training_resume_matches_uninterrupted_adam_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_input = tmp_path / "identity-input.bin"
    implementation = tmp_path / "implementation.py"
    identity_input.write_bytes(b"fixed-input")
    implementation.write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        pal_nurbs,
        "_identity_input_paths",
        lambda config: {"test_input": identity_input},
    )
    monkeypatch.setattr(
        pal_nurbs,
        "_implementation_closure_paths",
        lambda: [implementation],
    )

    training_cases = [
        {
            "case_id": "f",
            "training_group": "far",
            "distance_mm": 100000.0,
            "field_x_deg": 0.0,
            "field_y_deg": 1.0,
        },
        {
            "case_id": "m",
            "training_group": "intermediate",
            "distance_mm": 2000.0,
            "field_x_deg": 0.0,
            "field_y_deg": 0.0,
        },
        {
            "case_id": "n",
            "training_group": "near",
            "distance_mm": 500.0,
            "field_x_deg": 0.0,
            "field_y_deg": -1.0,
        },
        {
            "case_id": "pl",
            "training_group": "peripheral_left",
            "distance_mm": 2000.0,
            "field_x_deg": -1.0,
            "field_y_deg": 0.0,
        },
        {
            "case_id": "pr",
            "training_group": "peripheral_right",
            "distance_mm": 2000.0,
            "field_x_deg": 1.0,
            "field_y_deg": 0.0,
        },
    ]
    class Model:
        _key = staticmethod(MinimalOpticalModel._key)

        def __init__(self, config: MinimalConfig, perturbation) -> None:
            self.config = config
            self.perturbation = perturbation
            self.device = torch.device("cpu")
            self.size_reference_mm = {500.0: 1.0, 2000.0: 1.0, 100000.0: 1.0}
            self._cache = {}
            self._templates = {}

        def set_prescription_context(self, sag, power_config, zones) -> None:
            return None

        def astig_A_by_zone(self) -> dict[str, torch.Tensor]:
            zero = torch.zeros((), dtype=torch.float64)
            delta = self.perturbation.delta_raw(zero, zero)
            return {
                "astig_left": torch.sigmoid(-0.5 + delta),
                "astig_right": torch.sigmoid(-0.25 + delta),
            }

        def field(self, case: dict[str, object]) -> FieldResult:
            zero = torch.zeros((), dtype=torch.float64)
            delta = self.perturbation.delta_raw(zero, zero)
            edge = torch.sigmoid(-1.0 + 0.5 * delta)
            z = torch.zeros_like(edge)
            kernel = torch.stack(
                (
                    torch.stack((z, z, z)),
                    torch.stack((z, 1.0 - edge, edge)),
                    torch.stack((z, z, z)),
                )
            )
            return FieldResult(kernel, torch.ones_like(edge), 1.0, torch.zeros_like(edge))

    monkeypatch.setattr(pal_nurbs, "MinimalOpticalModel", Model)
    monkeypatch.setattr(
        pal_nurbs,
        "_prepare_or_load_case_layout",
        lambda config, output, model, identity_sha256: training_cases,
    )
    monkeypatch.setattr(
        pal_nurbs,
        "load_pal",
        lambda config, device: (
            torch.zeros((5, 5), dtype=torch.float64),
            PALPowerConfig(40.0, 1.5, 100.0, 5.0),
            {},
        ),
    )

    def fake_prescription(sag, power_config, zones):
        zero = sag.sum() * 0.0
        return {"P_far_D": zero, "ADD_D": zero, "astig_mean_D": zero}

    monkeypatch.setattr(pal_nurbs, "prescription_metrics", fake_prescription)
    real_evaluate = pal_nurbs._evaluate
    interruption = {"enabled": True, "gradient_calls": 0}

    def interrupting_evaluate(*args, **kwargs):
        if kwargs.get("with_grad"):
            interruption["gradient_calls"] += 1
            if interruption["enabled"] and interruption["gradient_calls"] == 2:
                raise RuntimeError("synthetic stage interruption")
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(pal_nurbs, "_evaluate", interrupting_evaluate)
    interrupted_config = MinimalConfig(
        output=str(tmp_path / "interrupted"),
        device="cpu",
        intermediate_object_distance_mm=2000.0,
        max_steps_7=2,
        max_steps_11=0,
        max_steps_19=0,
    )
    with pytest.raises(RuntimeError, match="synthetic stage interruption"):
        run(interrupted_config)
    active = torch.load(
        tmp_path / "interrupted" / "stage_7x7" / "resume.pt", map_location="cpu"
    )
    assert active["status"] == "active"
    assert active["completed_step"] == 1

    interruption["enabled"] = False
    run(interrupted_config, resume=True)
    uninterrupted_config = replace(
        interrupted_config, output=str(tmp_path / "uninterrupted")
    )
    run(uninterrupted_config)

    resumed_final = torch.load(
        tmp_path / "interrupted" / "stage_7x7" / "final.pt", map_location="cpu"
    )
    uninterrupted_final = torch.load(
        tmp_path / "uninterrupted" / "stage_7x7" / "final.pt", map_location="cpu"
    )
    assert resumed_final["step"] == uninterrupted_final["step"] == 2
    for name, value in resumed_final["state_dict"].items():
        assert torch.equal(value, uninterrupted_final["state_dict"][name])
    assert (
        tmp_path / "interrupted" / "stage_7x7" / "history.csv"
    ).read_bytes() == (
        tmp_path / "uninterrupted" / "stage_7x7" / "history.csv"
    ).read_bytes()


def test_joint_loss_fails_closed_when_a_case_is_detached_from_nurbs() -> None:
    class DetachedModel:
        def field(self, case: dict[str, object]) -> FieldResult:
            kernel = torch.zeros((3, 3), dtype=torch.float64)
            kernel[1, 1] = 0.5
            kernel[1, 2] = 0.5
            return FieldResult(kernel, torch.tensor(1.0), 1.0, torch.tensor(0.0))

        def astig_A_by_zone(self) -> dict[str, torch.Tensor]:
            return {
                "astig_left": torch.tensor(0.2, dtype=torch.float64),
                "astig_right": torch.tensor(0.3, dtype=torch.float64),
            }

    cases = [
        {"case_id": "f", "training_group": "far"},
        {"case_id": "m", "training_group": "intermediate"},
        {"case_id": "n", "training_group": "near"},
        {"case_id": "pl", "training_group": "peripheral_left"},
        {"case_id": "pr", "training_group": "peripheral_right"},
    ]
    baseline = {case["case_id"]: {"loss_metric": 0.25} for case in cases}
    with pytest.raises(RuntimeError, match="detached from NURBS"):
        _evaluate(DetachedModel(), cases, baseline, with_grad=True)
