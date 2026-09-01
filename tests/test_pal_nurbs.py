from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
    _early_stopping_observation,
    _load_stage_resume_state,
    _make_stage_resume_payload,
    _open_run_directory,
    _evaluate_original_training_baseline_with_resume,
    _prepare_case_layout,
    _qualified_pool_case_for_saved_attempt,
    _release_inactive_case_cuda_cache,
    _retain_training_cache,
    _restore_optimizer_state,
    _stage_boundary_stop_reason,
    _training_stage_specs,
    _torch_save_atomic,
    _evaluate,
    load_pal,
    prepare_only,
    prescription_metrics,
    psf_second_moment_mm2,
    run,
)
from biot.e2e.regional_nurbs import FixedWeightNURBSPerturbation


def _nine_group_cases() -> list[dict[str, object]]:
    return [
        {"case_id": f"case_{index:02d}", "training_group": group, "scale": float(index)}
        for index, group in enumerate(
            pal_nurbs.FUNCTIONAL_GROUPS + pal_nurbs.PERIPHERAL_GROUPS,
            start=1,
        )
    ]


def _write_json_fixture(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _make_completed_parent_fixture(
    root: Path, *, steps: tuple[int, int, int],
) -> tuple[Path, MinimalConfig, dict[int, FixedWeightNURBSPerturbation]]:
    parent = root / "parent"
    parent.mkdir(parents=True)
    config = MinimalConfig(
        output=str(parent), device="cpu",
        max_steps_7=steps[0], max_steps_11=steps[1], max_steps_19=steps[2],
        max_extra_terminal_stage_steps=0,
    )
    identity_body = {
        "schema_version": pal_nurbs.RUN_IDENTITY_SCHEMA_VERSION,
        "method": pal_nurbs.METHOD_NAME,
        "fixture": "completed-parent",
    }
    identity_sha256 = pal_nurbs._canonical_json_sha256(identity_body)
    identity = {**identity_body, "identity_sha256": identity_sha256}
    _write_json_fixture(parent / "run_identity.json", identity)
    _write_json_fixture(parent / "config.json", pal_nurbs.asdict(config))
    _write_json_fixture(
        parent / "run_state.json",
        {
            "status": "complete", "phase": "complete",
            "identity_sha256": identity_sha256,
        },
    )

    module7 = FixedWeightNURBSPerturbation(7, device="cpu", dtype=torch.float64)
    with torch.no_grad():
        module7.inner_q.copy_(
            torch.linspace(-1.0e-4, 1.0e-4, module7.inner_q.numel()).reshape_as(module7.inner_q)
        )
    module11 = module7.refined(11)
    if steps[1] > 0:
        with torch.no_grad():
            module11.inner_q.add_(2.0e-5)
    module19 = module11.refined(19)
    if steps[2] > 0:
        with torch.no_grad():
            module19.inner_q.add_(1.0e-5)
    modules = {7: module7, 11: module11, 19: module19}
    terminal = [control for control, budget in zip((7, 11, 19), steps) if budget > 0][-1]
    stage_rows: list[dict[str, object]] = []
    for control, minimum in zip((7, 11, 19), steps):
        module = modules[control]
        actual = int(minimum)
        terminal_stage = control == terminal
        history = [
            {
                "step": step,
                "significant_improvement": bool(terminal_stage),
                "no_improvement_attempts": 0,
            }
            for step in range(1, actual + 1)
        ]
        stop_reason = "max_extra_reached" if terminal_stage else "minimum_completed"
        stage_summary = {
            "control_count": control,
            "is_terminal_stage": terminal_stage,
            "initial_J": 1.0,
            "initial_groups": {"J": 1.0},
            "best_J": 0.9,
            "best_groups": {"J": 0.9},
            "relative_stage_improvement": 0.1,
            "steps": actual,
            "minimum_steps": minimum,
            "maximum_steps": minimum,
            "actual_steps": actual,
            "extra_steps": 0,
            "early_stopping_patience": config.early_stopping_patience,
            "relative_improvement_threshold": config.relative_improvement_threshold,
            "no_improvement_attempts": 0,
            "stop_reason": stop_reason,
        }
        optimizer = torch.optim.Adam([module.inner_q], lr=config.learning_rate)
        stage_dir = parent / f"stage_{control}x{control}"
        pal_nurbs._save_checkpoint(
            stage_dir / "final.pt", module,
            identity_sha256=identity_sha256, J=0.9, step=actual,
        )
        _torch_save_atomic(
            stage_dir / "resume.pt",
            _make_stage_resume_payload(
                identity_sha256=identity_sha256,
                status="completed",
                control_count=control,
                minimum_steps=minimum,
                maximum_steps=minimum,
                terminal_control_count=terminal,
                early_stopping_patience=config.early_stopping_patience,
                relative_improvement_threshold=config.relative_improvement_threshold,
                max_extra_terminal_stage_steps=0,
                no_improvement_attempts=0,
                stop_reason=stop_reason,
                module=module,
                optimizer=optimizer,
                learning_rate=config.learning_rate,
                completed_step=actual,
                history=history,
                stage_initial=1.0,
                stage_initial_groups={"J": 1.0},
                best=0.9,
                best_state=module.state_dict(),
                best_health={"J_far": 0.9},
                stage_summary=stage_summary,
                optimizer_model_state=module.state_dict(),
            ),
        )
        stage_rows.append(stage_summary)
    _write_json_fixture(
        parent / "summary.json",
        {
            "identity_sha256": identity_sha256,
            "terminal_control_count": terminal,
            "final_control_count": 19,
            "actual_training_steps": sum(steps),
            "runtime_seconds": 1.0,
            "stages": stage_rows,
        },
    )
    _write_json_fixture(
        parent / "case_layout_state.json",
        {"schema_version": pal_nurbs.CASE_LAYOUT_STATE_SCHEMA_VERSION,
         "identity_sha256": identity_sha256},
    )
    _write_json_fixture(parent / "candidate_trace_progress.json", {"status": "complete"})
    _write_json_fixture(
        parent / "forward_qualification_progress.json", {"status": "complete"}
    )
    _write_json_fixture(
        parent / "final_phase_qualification_progress.json", {"status": "complete"}
    )
    zero = FixedWeightNURBSPerturbation(7, device="cpu", dtype=torch.float64)
    baseline_cases = _nine_group_cases()
    baseline_rows = [
        {
            **case,
            "loss_metric": 1.0,
            "astig_A_D": 1.0,
            "valid_fraction": 1.0,
            "edge_fraction": 0.0,
        }
        for case in baseline_cases
    ]
    baseline_health = {
        "objective_name": "fixture objective",
        "minimum_valid_fraction_ratio": 1.0,
        "maximum_edge_fraction": 0.0,
        **{
            f"J_{name}": 1.0
            for name in (*pal_nurbs.FUNCTIONAL_GROUPS, *pal_nurbs.PERIPHERAL_GROUPS)
        },
        "J_mid": 1.0,
        "J_functional": 1.0,
        "J_peripheral": 1.0,
        "J_total": 1.0,
    }
    _torch_save_atomic(
        parent / "baseline_state.pt",
        {
            "schema_version": pal_nurbs.BASELINE_STATE_SCHEMA_VERSION,
            "identity_sha256": identity_sha256,
            "control_count": 7,
            "model_state": zero.state_dict(),
            "training_case_ids": [str(case["case_id"]) for case in baseline_cases],
            "baseline_value": 1.0,
            "baseline_rows": baseline_rows,
            "baseline_health": baseline_health,
            "objective_config": {
                "group_weights": config.group_weights,
                "near_edge_astig_A_weight": config.near_edge_astig_A_weight,
                "weighted_mtf_loss_tolerance": config.weighted_mtf_loss_tolerance,
                "z4_rms_tolerance_mm": config.z4_rms_tolerance_mm,
                "astigmatism_tolerance_D": config.astigmatism_tolerance_D,
            },
            "baseline_power": {"P_far_D": 0.0, "ADD_D": 0.0},
            "rng_state": pal_nurbs._capture_rng_state(),
        },
    )
    diagnostics = parent / "gradient_diagnostics"
    diagnostic_records = []
    for label, checkpoint in (
        ("baseline_7x7", parent / "baseline_state.pt"),
        ("stage_7x7_final", parent / "stage_7x7" / "final.pt"),
    ):
        body = {
            "schema_version": pal_nurbs.GRADIENT_DIAGNOSTIC_SCHEMA_VERSION,
            "identity_sha256": identity_sha256,
            "label": label,
            "checkpoint_sha256": pal_nurbs._sha256_file(checkpoint),
        }
        path = diagnostics / f"{label}.json"
        _write_json_fixture(
            path, {**body, "diagnostic_sha256": pal_nurbs._canonical_json_sha256(body)}
        )
        diagnostic_records.append(
            {"label": label, "path": path.name, "sha256": pal_nurbs._sha256_file(path)}
        )
    manifest_body = {
        "schema_version": pal_nurbs.GRADIENT_DIAGNOSTIC_SCHEMA_VERSION,
        "identity_sha256": identity_sha256,
        "artifacts": diagnostic_records,
    }
    _write_json_fixture(
        diagnostics / "manifest.json",
        {
            **manifest_body,
            "manifest_sha256": pal_nurbs._canonical_json_sha256(manifest_body),
        },
    )
    return parent, config, modules


def test_current_phase_progress_uses_stable_source_key_before_renumbered_case_id() -> None:
    qualified_pool = [
        {
            "case_id": "far_01_Dinf",
            "training_group": "far",
            "candidate_id": "cand_a",
            "distance_mm": float("inf"),
            "field_x_deg": -9.0,
            "field_y_deg": 1.0,
        },
        {
            "case_id": "far_02_Dinf",
            "training_group": "far",
            "candidate_id": "cand_b",
            "distance_mm": float("inf"),
            "field_x_deg": -8.0,
            "field_y_deg": 2.0,
        },
    ]
    by_key = {
        pal_nurbs.qualified_source_key(row): row for row in qualified_pool
    }
    saved_attempt = {
        **qualified_pool[1],
        # A prior regional-FPS round assigned this now-colliding ordinal ID.
        "case_id": "far_01_Dinf",
    }

    resolved = _qualified_pool_case_for_saved_attempt(
        saved_attempt, qualified_pool, by_key
    )

    assert resolved is qualified_pool[1]


def test_current_phase_progress_does_not_fall_back_from_foreign_stable_key() -> None:
    qualified_pool = [
        {
            "case_id": "far_01_Dinf",
            "training_group": "far",
            "candidate_id": "cand_a",
            "distance_mm": float("inf"),
            "field_x_deg": -9.0,
            "field_y_deg": 1.0,
        }
    ]
    by_key = {
        pal_nurbs.qualified_source_key(row): row for row in qualified_pool
    }
    foreign_attempt = {
        **qualified_pool[0],
        "candidate_id": "foreign",
    }

    assert _qualified_pool_case_for_saved_attempt(
        foreign_attempt, qualified_pool, by_key
    ) is None


def test_main_training_phase_contract_is_nonlegacy_raw_psf() -> None:
    config = MinimalConfig(device="cpu")
    assert config.legacy_pupil_phase is False
    assert config.phase_reference == "biot_reference_sphere"
    assert config.remove_tilt is False
    with pytest.raises(ValueError, match="legacy_pupil_phase"):
        MinimalConfig(legacy_pupil_phase=True)
    with pytest.raises(ValueError, match="remove_tilt"):
        MinimalConfig(remove_tilt=True)
    with pytest.raises(ValueError, match="phase_reference"):
        MinimalConfig(phase_reference="image_plane_center")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_steps_7": -1}, "max_steps_7"),
        ({"max_steps_7": 1.5}, "max_steps_7"),
        ({"max_steps_7": True}, "max_steps_7"),
        ({"max_steps_11": -1}, "max_steps_11"),
        ({"max_steps_19": -1}, "max_steps_19"),
        (
            {"max_extra_terminal_stage_steps": -1},
            "max_extra_terminal_stage_steps",
        ),
        ({"early_stopping_patience": 0}, "early_stopping_patience"),
        ({"early_stopping_patience": -1}, "early_stopping_patience"),
        ({"early_stopping_patience": 1.5}, "early_stopping_patience"),
        ({"relative_improvement_threshold": 0.0}, "relative_improvement_threshold"),
        ({"relative_improvement_threshold": -1.0}, "relative_improvement_threshold"),
        ({"relative_improvement_threshold": math.inf}, "relative_improvement_threshold"),
        ({"relative_improvement_threshold": math.nan}, "relative_improvement_threshold"),
    ],
)
def test_training_budget_config_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        MinimalConfig(device="cpu", **kwargs)


def test_parent_fork_config_requires_paired_arguments_and_unmixed_imports(tmp_path) -> None:
    with pytest.raises(ValueError, match="supplied together"):
        MinimalConfig(parent_run=str(tmp_path / "parent"), device="cpu")
    with pytest.raises(ValueError, match="before start_stage"):
        MinimalConfig(
            parent_run=str(tmp_path / "parent"), start_stage=11,
            max_steps_7=1, max_steps_11=1, max_steps_19=0, device="cpu",
        )
    with pytest.raises(ValueError, match="start_stage training budget"):
        MinimalConfig(
            parent_run=str(tmp_path / "parent"), start_stage=11,
            max_steps_7=0, max_steps_11=0, max_steps_19=1, device="cpu",
        )
    with pytest.raises(ValueError, match="explicit evidence imports"):
        MinimalConfig(
            parent_run=str(tmp_path / "parent"), start_stage=11,
            max_steps_7=0, max_steps_11=1, max_steps_19=0,
            baseline_state_import="baseline.pt", device="cpu",
        )


@pytest.mark.parametrize(
    ("parent_steps", "start_stage", "allowed"),
    [
        ((50, 0, 0), 7, True),
        ((50, 0, 0), 11, True),
        ((50, 0, 0), 19, True),
        ((50, 25, 0), 7, False),
        ((50, 25, 0), 11, True),
        ((50, 25, 0), 19, True),
        ((50, 25, 10), 7, False),
        ((50, 25, 10), 11, False),
        ((50, 25, 10), 19, True),
    ],
)
def test_parent_fork_start_stage_matrix(
    tmp_path, parent_steps: tuple[int, int, int], start_stage: int, allowed: bool,
) -> None:
    case_root = tmp_path / f"p_{parent_steps[0]}_{parent_steps[1]}_{parent_steps[2]}_s{start_stage}"
    parent, _, _ = _make_completed_parent_fixture(case_root, steps=parent_steps)
    budgets = {
        7: (1, 0, 0),
        11: (0, 1, 0),
        19: (0, 0, 1),
    }[start_stage]
    child = MinimalConfig(
        output=str(case_root / "child"), parent_run=str(parent), start_stage=start_stage,
        device="cpu", max_steps_7=budgets[0], max_steps_11=budgets[1],
        max_steps_19=budgets[2],
    )
    before = {
        path.relative_to(parent).as_posix(): pal_nurbs._sha256_file(path)
        for path in parent.rglob("*") if path.is_file()
    }
    if allowed:
        context = pal_nurbs._validate_parent_run_source(child, device="cpu")
        assert context is not None
        assert context["terminal_control_count"] == next(
            control for control, budget in reversed(list(zip((7, 11, 19), parent_steps)))
            if budget > 0
        )
        assert context["start_stage"] == start_stage
    else:
        with pytest.raises(ValueError, match="cannot precede parent terminal stage"):
            pal_nurbs._validate_parent_run_source(child, device="cpu")
    after = {
        path.relative_to(parent).as_posix(): pal_nurbs._sha256_file(path)
        for path in parent.rglob("*") if path.is_file()
    }
    assert after == before


def test_parent_best_activation_uses_selected_final_and_fresh_adam(tmp_path) -> None:
    parent, _, modules = _make_completed_parent_fixture(tmp_path, steps=(50, 0, 0))
    child = MinimalConfig(
        output=str(tmp_path / "child"), parent_run=str(parent), start_stage=19,
        device="cpu", max_steps_7=0, max_steps_11=0, max_steps_19=1,
    )
    context = pal_nurbs._validate_parent_run_source(child, device="cpu")
    assert context is not None

    model = SimpleNamespace(perturbation=None, _templates={}, _cache={})
    activated = pal_nurbs._activate_parent_best(
        model, context, device=torch.device("cpu"),
    )
    _assert = pal_nurbs._assert_state_dict_equal
    _assert(activated.state_dict(), modules[19].state_dict(), context="test activation")
    optimizer = torch.optim.Adam([activated.inner_q], lr=child.learning_rate)
    assert optimizer.state == {}
    assert model.perturbation is activated


def test_parent_fork_rejects_nonfork_config_drift(tmp_path) -> None:
    parent, _, _ = _make_completed_parent_fixture(tmp_path, steps=(50, 25, 0))
    child = MinimalConfig(
        output=str(tmp_path / "child"), parent_run=str(parent), start_stage=11,
        device="cpu", max_steps_7=0, max_steps_11=1, max_steps_19=1,
        smooth_lambda=0.06,
    )
    with pytest.raises(ValueError, match="configuration mismatch.*smooth_lambda"):
        pal_nurbs._validate_parent_run_source(child, device="cpu")


def test_parent_fork_rejects_incomplete_parent_and_evidence(tmp_path) -> None:
    incomplete_root = tmp_path / "incomplete"
    parent, _, _ = _make_completed_parent_fixture(incomplete_root, steps=(1, 0, 0))
    state = json.loads((parent / "run_state.json").read_text(encoding="utf-8"))
    state.update({"status": "running", "phase": "stage_training"})
    _write_json_fixture(parent / "run_state.json", state)
    child = MinimalConfig(
        output=str(incomplete_root / "child"), parent_run=str(parent), start_stage=7,
        device="cpu", max_steps_7=1, max_steps_11=0, max_steps_19=0,
    )
    with pytest.raises(ValueError, match="must have complete run_state"):
        pal_nurbs._validate_parent_run_source(child, device="cpu")

    evidence_root = tmp_path / "bad_evidence"
    parent, _, _ = _make_completed_parent_fixture(evidence_root, steps=(1, 0, 0))
    _write_json_fixture(
        parent / "forward_qualification_progress.json", {"status": "running"},
    )
    child = replace(child, output=str(evidence_root / "child"), parent_run=str(parent))
    with pytest.raises(ValueError, match="forward_qualification_progress must be complete"):
        pal_nurbs._validate_parent_run_source(child, device="cpu")


def test_parent_fork_rejects_checkpoint_identity_and_intermediate_loss(
    tmp_path,
) -> None:
    identity_root = tmp_path / "bad_checkpoint"
    parent, _, _ = _make_completed_parent_fixture(identity_root, steps=(1, 1, 0))
    final_path = parent / "stage_11x11" / "final.pt"
    final = pal_nurbs._load_torch_mapping(final_path, map_location="cpu")
    final["identity_sha256"] = "foreign"
    _torch_save_atomic(final_path, final)
    child = MinimalConfig(
        output=str(identity_root / "child"), parent_run=str(parent), start_stage=11,
        device="cpu", max_steps_7=0, max_steps_11=1, max_steps_19=0,
    )
    with pytest.raises(ValueError, match="selected checkpoint identity mismatch"):
        pal_nurbs._validate_parent_run_source(child, device="cpu")

    missing_root = tmp_path / "missing_intermediate"
    parent, _, _ = _make_completed_parent_fixture(missing_root, steps=(1, 0, 0))
    (parent / "stage_11x11" / "resume.pt").unlink()
    child = replace(
        child, output=str(missing_root / "child"), parent_run=str(parent),
        start_stage=19, max_steps_11=0, max_steps_19=1,
    )
    with pytest.raises(FileNotFoundError, match="stage_11_resume"):
        pal_nurbs._validate_parent_run_source(child, device="cpu")


def test_parent_fork_run_uses_parent_best_and_preserves_parent(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, _, _ = _make_completed_parent_fixture(tmp_path, steps=(1, 1, 0))
    training_cases = _nine_group_cases()
    for index, case in enumerate(training_cases):
        case.update({
            "distance_mm": 500.0 + 100.0 * index,
            "field_x_deg": float(index - 5),
            "field_y_deg": float(5 - index),
        })

    class Model:
        _key = staticmethod(MinimalOpticalModel._key)

        def __init__(self, config: MinimalConfig, perturbation) -> None:
            self.config = config
            self.perturbation = perturbation
            self.device = torch.device("cpu")
            self.size_reference_mm = {}
            self._cache = {}
            self._templates = {}

        def set_prescription_context(self, sag, power_config, zones) -> None:
            return None

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

    def fake_evaluate(model, cases, baseline, *, with_grad, **kwargs):
        objective = 1.0 + model.perturbation.inner_q.square().mean()
        if with_grad:
            objective.backward()
        value = float(objective.detach().cpu())
        health = {
            "objective_name": "fixture objective",
            "minimum_valid_fraction_ratio": 1.0,
            "maximum_edge_fraction": 0.0,
            **{
                f"J_{name}": value
                for name in (*pal_nurbs.FUNCTIONAL_GROUPS, *pal_nurbs.PERIPHERAL_GROUPS)
            },
            "J_mid": value,
            "J_functional": value,
            "J_peripheral": value,
            "J_total": value,
        }
        return value, [], health

    monkeypatch.setattr(pal_nurbs, "_evaluate", fake_evaluate)

    def fake_prescription(sag, power_config, zones, *, baseline_sag=None):
        zero = sag.sum() * 0.0
        return {
            "P_far_D": zero,
            "ADD_D": zero,
            "astig_mean_D": zero,
            "lower_edge_max_abs_power_change_D": zero,
            "lower_edge_max_abs_astig_change_D": zero,
        }

    monkeypatch.setattr(pal_nurbs, "prescription_metrics", fake_prescription)
    before = {
        path.relative_to(parent).as_posix(): pal_nurbs._sha256_file(path)
        for path in parent.rglob("*") if path.is_file()
    }
    child = MinimalConfig(
        output=str(tmp_path / "child"), parent_run=str(parent), start_stage=11,
        device="cpu", max_steps_7=0, max_steps_11=1, max_steps_19=1,
        max_extra_terminal_stage_steps=0,
    )
    run(child)

    assert not (tmp_path / "child" / "stage_7x7").exists()
    with (parent / "stage_11x11" / "final.pt").open("rb") as handle:
        parent_final = torch.load(handle, map_location="cpu")
    with (tmp_path / "child" / "stage_11x11" / "initial.pt").open("rb") as handle:
        child_initial = torch.load(handle, map_location="cpu")
    pal_nurbs._assert_state_dict_equal(
        parent_final["state_dict"], child_initial["state_dict"],
        context="child initial/parent best",
    )
    with (tmp_path / "child" / "stage_11x11" / "resume.pt").open("rb") as handle:
        child_stage = torch.load(handle, map_location="cpu")
    assert child_stage["history"][0]["step"] == 1
    assert child_stage["optimizer_state"]["state"]

    summary = json.loads(
        (tmp_path / "child" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["actual_training_steps"] == 2
    assert summary["child_actual_training_steps_by_stage"] == {
        "7": 0, "11": 1, "19": 1,
    }
    assert summary["parent_actual_training_steps_by_stage"] == {
        "7": 1, "11": 1, "19": 0,
    }
    assert summary["lineage_actual_training_steps_by_stage"] == {
        "7": 1, "11": 2, "19": 1,
    }
    assert summary["optimizer_policy"] == "parent_best_fresh_adam"
    child_identity = json.loads(
        (tmp_path / "child" / "run_identity.json").read_text(encoding="utf-8")
    )
    assert child_identity["parent_lineage"]["selected_checkpoint_sha256"] == (
        pal_nurbs._sha256_file(parent / "stage_11x11" / "final.pt")
    )
    assert "PARENT_IMPORT" in (
        tmp_path / "child" / "training.log"
    ).read_text(encoding="utf-8")

    interruption = {"enabled": True, "gradient_calls": 0}

    def interrupting_evaluate(model, cases, baseline, *, with_grad, **kwargs):
        if with_grad:
            interruption["gradient_calls"] += 1
            if interruption["enabled"] and interruption["gradient_calls"] == 3:
                raise RuntimeError("synthetic child interruption")
        return fake_evaluate(
            model, cases, baseline, with_grad=with_grad, **kwargs,
        )

    monkeypatch.setattr(pal_nurbs, "_evaluate", interrupting_evaluate)
    interrupted = replace(
        child, output=str(tmp_path / "child_interrupted"), max_steps_19=2,
    )
    with pytest.raises(RuntimeError, match="synthetic child interruption"):
        run(interrupted)
    with (
        tmp_path / "child_interrupted" / "stage_19x19" / "resume.pt"
    ).open("rb") as handle:
        active = torch.load(handle, map_location="cpu")
    assert active["status"] == "active"
    assert active["completed_step"] == 1

    interruption.update({"enabled": False, "gradient_calls": 0})
    run(interrupted, resume=True)
    uninterrupted = replace(
        interrupted, output=str(tmp_path / "child_uninterrupted"),
    )
    run(uninterrupted)
    with (
        tmp_path / "child_interrupted" / "stage_19x19" / "final.pt"
    ).open("rb") as handle:
        resumed_final = torch.load(handle, map_location="cpu")
    with (
        tmp_path / "child_uninterrupted" / "stage_19x19" / "final.pt"
    ).open("rb") as handle:
        uninterrupted_final = torch.load(handle, map_location="cpu")
    pal_nurbs._assert_state_dict_equal(
        resumed_final["state_dict"], uninterrupted_final["state_dict"],
        context="resumed/uninterrupted child",
    )
    assert (
        tmp_path / "child_interrupted" / "stage_19x19" / "history.csv"
    ).read_bytes() == (
        tmp_path / "child_uninterrupted" / "stage_19x19" / "history.csv"
    ).read_bytes()
    after = {
        path.relative_to(parent).as_posix(): pal_nurbs._sha256_file(path)
        for path in parent.rglob("*") if path.is_file()
    }
    assert after == before


def test_terminal_stage_patience_counts_every_attempt_and_resets_only_on_strict_improvement() -> None:
    refreshed, relative, significant, counter = _early_stopping_observation(
        best_before=1.0,
        candidate=0.5,
        accepted=False,
        threshold=1.0e-4,
        no_improvement_attempts=2,
    )
    assert (refreshed, relative, significant, counter) == (False, 0.0, False, 3)

    refreshed, relative, significant, counter = _early_stopping_observation(
        best_before=1.0,
        candidate=0.99995,
        accepted=True,
        threshold=1.0e-4,
        no_improvement_attempts=3,
    )
    assert refreshed
    assert relative == pytest.approx(5.0e-5)
    assert not significant
    assert counter == 4

    refreshed, relative, significant, counter = _early_stopping_observation(
        best_before=1.0,
        candidate=0.9998,
        accepted=True,
        threshold=1.0e-4,
        no_improvement_attempts=4,
    )
    assert refreshed
    assert relative == pytest.approx(2.0e-4)
    assert significant
    assert counter == 0

    _, relative, significant, counter = _early_stopping_observation(
        best_before=2.0,
        candidate=1.0,
        accepted=True,
        threshold=0.5,
        no_improvement_attempts=0,
    )
    assert relative == 0.5
    assert not significant
    assert counter == 1


def test_terminal_stage_stop_order_suppresses_early_stopping_until_minimum() -> None:
    common = {
        "control_count": 19,
        "is_terminal_stage": True,
        "minimum_steps": 10,
        "maximum_steps": 60,
        "learning_rate": 2.0e-3,
        "minimum_learning_rate": 1.0e-6,
        "no_improvement_attempts": 9,
        "early_stopping_patience": 7,
    }
    assert _stage_boundary_stop_reason(completed_step=9, **common) is None
    assert _stage_boundary_stop_reason(completed_step=10, **common) == "early_stopping"
    assert _stage_boundary_stop_reason(
        completed_step=9,
        **{**common, "learning_rate": 5.0e-7},
    ) == "minimum_not_reached"
    assert _stage_boundary_stop_reason(
        completed_step=10,
        **{**common, "learning_rate": 5.0e-7, "no_improvement_attempts": 0},
    ) == "learning_rate_floor"
    assert _stage_boundary_stop_reason(
        completed_step=60,
        **{**common, "no_improvement_attempts": 0},
    ) == "max_extra_reached"


def test_extra_budget_binds_to_last_nonzero_training_stage() -> None:
    config = MinimalConfig(
        device="cpu",
        max_steps_7=50,
        max_steps_11=10,
        max_steps_19=0,
        max_extra_terminal_stage_steps=5,
    )
    assert _training_stage_specs(config) == (
        (7, 50, 50, False),
        (11, 10, 15, True),
        (19, 0, 0, False),
    )

    with pytest.raises(ValueError, match="at least one positive"):
        _training_stage_specs(
            replace(config, max_steps_7=0, max_steps_11=0, max_steps_19=0)
        )


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
    assert result.kernel.shape == (512, 512)
    assert result.raw_psf is not None
    assert result.raw_pixel_pitch_mm == pytest.approx(result.pixel_pitch_mm)
    m2 = psf_second_moment_mm2(result.kernel, pixel_pitch_mm=result.pixel_pitch_mm)
    m2.backward()
    assert torch.isfinite(result.kernel).all()
    assert bool((result.kernel >= 0).all())
    assert abs(float(result.kernel.sum()) - 1.0) <= 1e-10
    assert module.inner_q.grad is not None
    assert torch.isfinite(module.inner_q.grad).all()
    assert float(module.inner_q.grad.abs().max()) > 0.0


def test_raw_psf_batch_uses_one_true_batch_and_preserves_case_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object.__new__(MinimalOpticalModel)
    model.config = MinimalConfig(device="cpu", requested_np=16, fft_size_px=8)
    model.device = torch.device("cpu")
    model.sample_count = 4
    trace_calls: list[int] = []

    class System:
        def __init__(self, pitch: float) -> None:
            self.physical_fft_pixel_pitch_mm = pitch

    def system_and_rays(distance: float, x: float, y: float):
        del distance, y
        return System(0.01 + x * 0.001), object()

    model._system_and_rays = system_and_rays

    def trace_batch(systems, rays, *, phase_reference):
        assert phase_reference == "biot_reference_sphere"
        assert len(systems) == len(rays) == 3
        trace_calls.append(len(systems))
        return SimpleNamespace(
            phase_rad=torch.zeros((3, 4), dtype=torch.float64),
            valid=torch.ones((3, 4), dtype=torch.bool),
        )

    def fft_batch(phase, valid, **kwargs):
        assert phase.shape == valid.shape == (3, 4)
        assert kwargs["psf_size_px"] == 8
        psf = torch.zeros((3, 8, 8), dtype=torch.float64)
        psf[:, 4, 4] = 1.0
        return SimpleNamespace(psf=psf)

    monkeypatch.setattr(pal_nurbs, "trace_system_batch_to_image_with_phase", trace_batch)
    monkeypatch.setattr(pal_nurbs, "torch_fft_psf_from_phase", fft_batch)
    cases = [
        {"case_id": f"c{index}", "distance_mm": 500.0,
         "field_x_deg": float(index), "field_y_deg": 0.0}
        for index in range(3)
    ]
    result = model.raw_psf_batch(cases)
    assert trace_calls == [3]
    assert result.psf.shape == (3, 8, 8)
    assert torch.equal(result.valid_fraction, torch.ones(3, dtype=torch.float64))
    assert torch.allclose(
        result.pixel_pitch_mm, torch.tensor([0.01, 0.011, 0.012], dtype=torch.float64)
    )


def test_minimal_optical_model_close_releases_owned_systems_and_context() -> None:
    class System:
        def __init__(self) -> None:
            self.release_count = 0

        def release_biot_lens(self) -> None:
            self.release_count += 1

    cached = System()
    template = System()
    model = object.__new__(MinimalOpticalModel)
    model._cache = {(500.0, 0.0, 0.0): (cached, object())}
    model._templates = {500.0: (template, object())}
    model._pal_sag = torch.ones(1)
    model._pal_power_config = object()
    model._pal_zones = {"far": torch.ones(1, dtype=torch.bool)}

    model.close()

    assert cached.release_count == 1
    assert template.release_count == 1
    assert model._cache == {}
    assert model._templates == {}
    assert model._pal_sag is None
    assert model._pal_power_config is None
    assert model._pal_zones is None

    model.close()
    assert cached.release_count == 1
    assert template.release_count == 1


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
            return FieldResult(
                kernel, torch.ones_like(edge), 1.0, torch.zeros_like(edge),
                z4_defocus_mm2=edge,
            )

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
    summed_loss = sum(summed_model.field(case).z4_defocus_mm2 for case in cases)
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


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("weighted_mtf_loss_tolerance", 0.0),
        ("z4_rms_tolerance_mm", float("nan")),
        ("astigmatism_tolerance_D", -0.1),
    ],
)
def test_fixed_metric_tolerances_must_be_finite_and_positive(
    field_name: str, bad_value: float,
) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be finite and positive"):
        MinimalConfig(**{field_name: bad_value})
    assert len(pal_nurbs.DEFAULT_GROUP_WEIGHTS) == 9
    assert pal_nurbs.DEFAULT_GROUP_WEIGHTS["far"] == 0.24
    assert sum(pal_nurbs.DEFAULT_GROUP_WEIGHTS.values()) == pytest.approx(1.0)


def test_joint_loss_uses_explicit_nine_group_weights_and_routed_metrics() -> None:
    parameter = torch.nn.Parameter(torch.tensor(-0.5, dtype=torch.float64))

    class Model:
        def field(self, case: dict[str, object]) -> FieldResult:
            scale = float(case["scale"])
            edge = torch.sigmoid(parameter * scale)
            kernel = torch.zeros((9, 9), dtype=torch.float64)
            kernel[4, 4] = 1.0 - edge
            kernel[4, 5] = edge
            return FieldResult(
                kernel, torch.tensor(1.0), 0.001, torch.tensor(0.0),
                z4_defocus_mm2=edge,
            )

        def astig_A_by_zone(self) -> dict[str, torch.Tensor]:
            return {
                "astig_left": torch.sigmoid(parameter * 2.5),
                "astig_right": torch.sigmoid(parameter * 3.5),
                "near": torch.sigmoid(parameter * 1.5),
            }

    cases = _nine_group_cases()
    baseline = {
        str(case["case_id"]): {"loss_metric": 0.25, "astig_A_D": 0.2}
        for case in cases
    }
    value, rows, health = _evaluate(Model(), cases, baseline, with_grad=True)
    metrics = {str(row["training_group"]): str(row["loss_metric_name"]) for row in rows}
    assert metrics["far"] == "ahumada_weighted_mtf_loss"
    assert all(
        metrics[group].startswith("z4_defocus_mm2")
        for group in pal_nurbs.FUNCTIONAL_GROUPS
        if group != "far"
    )
    assert all(
        metrics[group] == "astig_A_D"
        for group in ("peripheral_left", "peripheral_right")
    )
    expected = sum(
        pal_nurbs.DEFAULT_GROUP_WEIGHTS[group] * health[f"J_{group}"]
        for group in pal_nurbs.FUNCTIONAL_GROUPS + pal_nurbs.PERIPHERAL_GROUPS
    )
    assert abs(value - expected) < 1e-15
    by_group = {str(row["training_group"]): row for row in rows}
    assert by_group["far"]["score"] == pytest.approx(
        by_group["far"]["weighted_mtf_loss"] / 0.10
    )
    assert by_group["corridor_upper"]["score"] == pytest.approx(
        by_group["corridor_upper"]["z4_defocus_mm2"] / (1.0e-4 ** 2)
    )
    assert by_group["near_edge_astig"]["score"] == pytest.approx(
        0.9 * by_group["near_edge_astig"]["z4_defocus_mm2"] / (1.0e-4 ** 2)
        + 0.1 * by_group["near_edge_astig"]["astig_A_D"] / 0.80
    )
    assert by_group["peripheral_left"]["score"] == pytest.approx(
        by_group["peripheral_left"]["astig_A_D"] / 0.80
    )
    different_baseline = {
        str(case["case_id"]): {"loss_metric": 1.0e9, "astig_A_D": 1.0e-9}
        for case in cases
    }
    repeated, _, _ = _evaluate(Model(), cases, different_baseline, with_grad=False)
    assert repeated == pytest.approx(value)
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad)
    assert float(parameter.grad) != 0.0


def test_group_gradient_diagnostic_reports_norms_cosines_and_preserves_state() -> None:
    module = FixedWeightNURBSPerturbation(7, device="cpu", dtype=torch.float64)

    class Model:
        config = MinimalConfig(device="cpu", case_batch_size=2, kernel_size_px=9)

        def field_batch(self, batch: list[dict[str, object]]) -> BatchFieldResult:
            scales = torch.as_tensor(
                [float(case["scale"]) for case in batch], dtype=torch.float64
            )
            edge = torch.sigmoid(module.inner_q.mean() * scales - 0.5)
            kernels = torch.zeros((len(batch), 9, 9), dtype=torch.float64)
            kernels[:, 4, 4] = 1.0 - edge
            kernels[:, 4, 5] = edge
            return BatchFieldResult(
                kernels=kernels,
                valid_fraction=torch.ones_like(edge),
                pixel_pitch_mm=torch.full_like(edge, 1.0e-3),
                edge_fraction=torch.zeros_like(edge),
                z4_defocus_mm2=edge.square(),
            )

        def astig_A_by_zone(self) -> dict[str, torch.Tensor]:
            value = torch.sigmoid(module.inner_q.mean() - 0.25)
            return {"astig_left": value, "astig_right": value * 1.1, "near": value * 0.5}

    module.inner_q.grad = torch.ones_like(module.inner_q)
    before_state = {name: value.clone() for name, value in module.state_dict().items()}
    payload = pal_nurbs._build_gradient_diagnostic(
        Model(),
        module,
        _nine_group_cases(),
        label="test",
        identity_sha256="identity",
        checkpoint_sha256="checkpoint",
        group_weights=pal_nurbs.DEFAULT_GROUP_WEIGHTS,
        near_edge_astig_A_weight=0.10,
        weighted_mtf_loss_tolerance=0.10,
        z4_rms_tolerance_mm=1.0e-4,
        astigmatism_tolerance_D=0.80,
    )
    assert payload["group_order"] == list(
        pal_nurbs.FUNCTIONAL_GROUPS + pal_nurbs.PERIPHERAL_GROUPS
    )
    assert len(payload["groups"]) == 9
    assert len(payload["cosine_matrix"]) == 9
    assert math.isfinite(payload["total_gradient_l2"])
    assert all(
        math.isfinite(record["gradient_l2"])
        for record in payload["groups"].values()
    )
    assert pal_nurbs._gradient_cosine(
        torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
    ) is None
    assert module.inner_q.grad is not None
    assert torch.equal(module.inner_q.grad, torch.ones_like(module.inner_q))
    assert all(
        torch.equal(before_state[name], value)
        for name, value in module.state_dict().items()
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
                pixel_pitch_mm=torch.full_like(edge, 0.001),
                edge_fraction=torch.zeros_like(edge),
                z4_defocus_mm2=edge,
            )

        def astig_A_by_zone(self) -> dict[str, torch.Tensor]:
            return {
                "astig_left": torch.sigmoid(parameter * 4.0),
                "astig_right": torch.sigmoid(parameter * 5.0),
                "near": torch.sigmoid(parameter * 3.5),
            }

    cases = _nine_group_cases()
    baseline = {
        str(case["case_id"]): {"loss_metric": 1.0, "astig_A_D": 1.0}
        for case in cases
    }
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
    assert len(model.batch_calls) == 4
    assert model.batch_calls[-1] == ["case_07"]
    assert len(rows) == len(cases)
    assert math.isfinite(value)
    assert math.isfinite(health["J_total"])
    assert len(backward_calls) == 5
    assert "stage=7x7 step=1/10 batch=4/4" in capsys.readouterr().out


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

    complete = _nine_group_cases()
    baseline = {str(case["case_id"]): {"loss_metric": 1.0} for case in complete}
    with pytest.raises(ValueError, match="exactly the nine groups"):
        _evaluate(Model(), complete[:-1], baseline, with_grad=False)
    extra = complete + [{"case_id": "x", "training_group": "other"}]
    with pytest.raises(ValueError, match="exactly the nine groups"):
        _evaluate(
            Model(), extra, {**baseline, "x": {"loss_metric": 1.0}}, with_grad=False
        )
    mixed = [dict(complete[0]), {"case_id": "ungrouped"}]
    with pytest.raises(ValueError, match="requires a training_group on every case"):
        _evaluate(
            Model(), mixed, {**baseline, "ungrouped": {"loss_metric": 1.0}}, with_grad=False
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
        minimum_steps=10,
        maximum_steps=10,
        terminal_control_count=19,
        early_stopping_patience=7,
        relative_improvement_threshold=1.0e-4,
        max_extra_terminal_stage_steps=50,
        no_improvement_attempts=0,
        stop_reason=None,
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
        minimum_steps=10,
        maximum_steps=10,
        terminal_control_count=19,
        early_stopping_patience=7,
        relative_improvement_threshold=1.0e-4,
        max_extra_terminal_stage_steps=50,
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

    legacy = copy.deepcopy(payload)
    legacy["schema_version"] = 2
    legacy_path = tmp_path / "legacy_resume.pt"
    _torch_save_atomic(legacy_path, legacy)
    with pytest.raises(ValueError, match="state schema mismatch"):
        _load_stage_resume_state(
            legacy_path,
            identity_sha256="identity",
            control_count=7,
            minimum_steps=10,
            maximum_steps=10,
            terminal_control_count=19,
            early_stopping_patience=7,
            relative_improvement_threshold=1.0e-4,
            max_extra_terminal_stage_steps=50,
            device="cpu",
        )


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
                **selected,
                "case_id": (
                    f"pool_{selected['candidate_id']}"
                    if kwargs.get("group_counts") is pal_nurbs.FORWARD_POOL_GROUP_COUNTS
                    else "final_01_Dinf"
                ),
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
    training_cases = _nine_group_cases()
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
            return FieldResult(
                kernel, torch.tensor(1.0), 0.001, torch.tensor(0.01),
                z4_defocus_mm2=torch.tensor(0.2, dtype=torch.float64),
            )

        def astig_A_by_zone(self) -> dict[str, torch.Tensor]:
            return {
                "astig_left": torch.tensor(0.2, dtype=torch.float64),
                "astig_right": torch.tensor(0.3, dtype=torch.float64),
                "near": torch.tensor(0.25, dtype=torch.float64),
            }

    path = tmp_path / "baseline_progress.pt"
    first = Model(fail_case="case_03")
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _evaluate_original_training_baseline_with_resume(
            first,
            training_cases,
            progress_path=path,
            identity_sha256="identity",
        )
    saved = torch.load(path, map_location="cpu")
    assert saved["next_training_index"] == 2
    assert [row["case_id"] for row in saved["training_rows"]] == ["case_01", "case_02"]

    resumed = Model()
    value, rows, health = _evaluate_original_training_baseline_with_resume(
        resumed,
        training_cases,
        progress_path=path,
        identity_sha256="identity",
    )
    assert resumed.calls == [f"case_{index:02d}" for index in range(3, 8)]
    assert [row["case_id"] for row in rows] == [f"case_{index:02d}" for index in range(1, 10)]
    assert math.isfinite(value) and value != 1.0
    assert health["J_functional"] != 1.0
    assert health["J_peripheral"] == pytest.approx((0.2 + 0.3) / (2.0 * 0.8))

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

    training_cases = _nine_group_cases()
    for index, case in enumerate(training_cases):
        group = str(case["training_group"])
        case.update({
            "distance_mm": (
                100000.0 if group in {"far", *pal_nurbs.PERIPHERAL_GROUPS}
                else 500.0 if group in {"near", "near_edge_astig"}
                else 2000.0
            ),
            "field_x_deg": float(index - 5),
            "field_y_deg": float(5 - index),
        })
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
                "near": torch.sigmoid(-0.75 + delta),
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
            return FieldResult(
                kernel, torch.ones_like(edge), 0.001, torch.zeros_like(edge),
                z4_defocus_mm2=edge,
            )

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

    def fake_prescription(sag, power_config, zones, *, baseline_sag=None):
        zero = sag.sum() * 0.0
        return {
            "P_far_D": zero,
            "ADD_D": zero,
            "astig_mean_D": zero,
            "lower_edge_max_abs_power_change_D": zero,
            "lower_edge_max_abs_astig_change_D": zero,
        }

    monkeypatch.setattr(pal_nurbs, "prescription_metrics", fake_prescription)
    real_evaluate = pal_nurbs._evaluate
    interruption = {"enabled": True, "gradient_calls": 0, "interrupt_at": 4}

    def interrupting_evaluate(*args, **kwargs):
        if kwargs.get("with_grad"):
            interruption["gradient_calls"] += 1
            if (
                interruption["enabled"]
                and interruption["gradient_calls"] == interruption["interrupt_at"]
            ):
                raise RuntimeError("synthetic stage interruption")
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(pal_nurbs, "_evaluate", interrupting_evaluate)
    interrupted_config = MinimalConfig(
        output=str(tmp_path / "interrupted"),
        device="cpu",
        intermediate_object_distance_mm=2000.0,
        max_steps_7=1,
        max_steps_11=1,
        max_steps_19=2,
        early_stopping_patience=1,
        relative_improvement_threshold=1.0,
        max_extra_terminal_stage_steps=5,
    )
    with pytest.raises(RuntimeError, match="synthetic stage interruption"):
        run(interrupted_config)
    active = torch.load(
        tmp_path / "interrupted" / "stage_19x19" / "resume.pt", map_location="cpu"
    )
    assert active["status"] == "active"
    assert active["completed_step"] == 1
    assert active["no_improvement_attempts"] == 1

    interruption["enabled"] = False
    run(interrupted_config, resume=True)
    uninterrupted_config = replace(
        interrupted_config, output=str(tmp_path / "uninterrupted")
    )
    run(uninterrupted_config)

    resumed_final = torch.load(
        tmp_path / "interrupted" / "stage_19x19" / "final.pt", map_location="cpu"
    )
    uninterrupted_final = torch.load(
        tmp_path / "uninterrupted" / "stage_19x19" / "final.pt", map_location="cpu"
    )
    assert resumed_final["step"] == uninterrupted_final["step"] == 2
    for name, value in resumed_final["state_dict"].items():
        assert torch.equal(value, uninterrupted_final["state_dict"][name])
    assert (
        tmp_path / "interrupted" / "stage_19x19" / "history.csv"
    ).read_bytes() == (
        tmp_path / "uninterrupted" / "stage_19x19" / "history.csv"
    ).read_bytes()
    resumed_summary = json.loads(
        (tmp_path / "interrupted" / "summary.json").read_text(encoding="utf-8")
    )
    assert [stage["actual_steps"] for stage in resumed_summary["stages"]] == [1, 1, 2]
    assert resumed_summary["minimum_training_steps"] == 4
    assert resumed_summary["actual_training_steps"] == 4
    assert resumed_summary["terminal_control_count"] == 19
    assert resumed_summary["extra_terminal_stage_steps"] == 0
    assert resumed_summary["training_stop_reason"] == "early_stopping"
    assert resumed_summary["stages"][-1]["no_improvement_attempts"] == 2
    resumed_log = (tmp_path / "interrupted" / "training.log").read_text(encoding="utf-8")
    assert "stage=19x19 step=2/7 minimum=2" in resumed_log
    assert "update=EARLY_STOPPING" in resumed_log
    assert "rel=" in resumed_log and "patience=2/1" in resumed_log
    best_checkpoint = torch.load(
        tmp_path / "interrupted" / "stage_19x19" / "best.pt", map_location="cpu"
    )
    for name, value in resumed_final["state_dict"].items():
        assert torch.equal(value, best_checkpoint["state_dict"][name])

    interruption.update({"enabled": True, "gradient_calls": 0, "interrupt_at": 2})
    extra_config = replace(
        interrupted_config,
        output=str(tmp_path / "extra_interrupted"),
        max_steps_7=0,
        max_steps_11=1,
        max_steps_19=0,
        early_stopping_patience=10,
        max_extra_terminal_stage_steps=2,
    )
    with pytest.raises(RuntimeError, match="synthetic stage interruption"):
        run(extra_config)
    extra_active = torch.load(
        tmp_path / "extra_interrupted" / "stage_11x11" / "resume.pt",
        map_location="cpu",
    )
    assert extra_active["completed_step"] == 1
    assert extra_active["no_improvement_attempts"] == 1

    interruption["enabled"] = False
    run(extra_config, resume=True)
    extra_uninterrupted = replace(
        extra_config, output=str(tmp_path / "extra_uninterrupted")
    )
    run(extra_uninterrupted)
    assert (
        tmp_path / "extra_interrupted" / "stage_11x11" / "history.csv"
    ).read_bytes() == (
        tmp_path / "extra_uninterrupted" / "stage_11x11" / "history.csv"
    ).read_bytes()
    extra_summary = json.loads(
        (tmp_path / "extra_interrupted" / "summary.json").read_text(encoding="utf-8")
    )
    terminal_summary = next(
        stage for stage in extra_summary["stages"] if stage["is_terminal_stage"]
    )
    assert terminal_summary["control_count"] == 11
    assert terminal_summary["actual_steps"] == 3
    assert terminal_summary["extra_steps"] == 2
    assert extra_summary["stages"][-1]["control_count"] == 19
    assert extra_summary["stages"][-1]["actual_steps"] == 0
    assert extra_summary["terminal_control_count"] == 11
    assert extra_summary["extra_terminal_stage_steps"] == 2
    assert extra_summary["training_stop_reason"] == "max_extra_reached"
    assert "update=MAX_EXTRA_REACHED" in (
        tmp_path / "extra_interrupted" / "training.log"
    ).read_text(encoding="utf-8")

    interruption["enabled"] = False
    floor_config = replace(
        extra_config,
        output=str(tmp_path / "minimum_not_reached"),
        max_steps_11=2,
        max_extra_terminal_stage_steps=0,
        minimum_learning_rate=1.5e-3,
        max_backtracks=0,
        step_sag_limit_mm=0.0,
    )
    with pytest.raises(pal_nurbs.MinimumTrainingBudgetError):
        run(floor_config)
    floor_state = json.loads(
        (tmp_path / "minimum_not_reached" / "run_state.json").read_text(encoding="utf-8")
    )
    assert floor_state["status"] == "failed"
    assert floor_state["phase"] == "minimum_not_reached"
    assert not (tmp_path / "minimum_not_reached" / "summary.json").exists()
    assert "update=MINIMUM_NOT_REACHED" in (
        tmp_path / "minimum_not_reached" / "training.log"
    ).read_text(encoding="utf-8")

    post_floor_config = replace(
        floor_config,
        output=str(tmp_path / "learning_rate_floor"),
        max_steps_11=1,
    )
    run(post_floor_config)
    post_floor_summary = json.loads(
        (tmp_path / "learning_rate_floor" / "summary.json").read_text(encoding="utf-8")
    )
    assert post_floor_summary["training_stop_reason"] == "learning_rate_floor"
    assert "update=LEARNING_RATE_FLOOR" in (
        tmp_path / "learning_rate_floor" / "training.log"
    ).read_text(encoding="utf-8")


def test_joint_loss_fails_closed_when_a_case_is_detached_from_nurbs() -> None:
    class DetachedModel:
        def field(self, case: dict[str, object]) -> FieldResult:
            kernel = torch.zeros((3, 3), dtype=torch.float64)
            kernel[1, 1] = 0.5
            kernel[1, 2] = 0.5
            return FieldResult(
                kernel, torch.tensor(1.0), 0.001, torch.tensor(0.0),
                z4_defocus_mm2=torch.tensor(0.25, dtype=torch.float64),
            )

        def astig_A_by_zone(self) -> dict[str, torch.Tensor]:
            return {
                "astig_left": torch.tensor(0.2, dtype=torch.float64),
                "astig_right": torch.tensor(0.3, dtype=torch.float64),
                "near": torch.tensor(0.25, dtype=torch.float64),
            }

    cases = _nine_group_cases()
    baseline = {
        str(case["case_id"]): {"loss_metric": 0.25, "astig_A_D": 0.25}
        for case in cases
    }
    with pytest.raises(RuntimeError, match="detached from NURBS"):
        _evaluate(DetachedModel(), cases, baseline, with_grad=True)
