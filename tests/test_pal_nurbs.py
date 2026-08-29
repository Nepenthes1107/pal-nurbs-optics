from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
import torch

from biot.e2e.pal_case_layout import DISTANCE_SPECS
from biot.e2e.pal_nurbs import (
    FieldResult,
    MinimalConfig,
    MinimalOpticalModel,
    _append_training_log,
    _baseline_metric_table,
    _evaluate,
    psf_second_moment_mm2,
)
from biot.e2e.regional_nurbs import FixedWeightNURBSPerturbation


def _toy_cases() -> list[dict[str, object]]:
    return [
        {
            "case_id": "D500_r00_c00",
            "distance_label": "D500",
            "zone": "near",
            "field_x_deg": 0.0,
            "field_y_deg": 0.0,
            "objective_weight": 0.5,
        },
        {
            "case_id": "D1000_r00_c00",
            "distance_label": "D1000",
            "zone": "corridor",
            "field_x_deg": 0.0,
            "field_y_deg": 0.0,
            "objective_weight": 0.3,
        },
        {
            "case_id": "Dinf_r00_c00",
            "distance_label": "Dinf",
            "zone": "far",
            "field_x_deg": 0.0,
            "field_y_deg": 0.0,
            "objective_weight": 0.2,
        },
    ]


class _ToyModel:
    def __init__(self, *, case_batch_size: int = 8) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        self.config = SimpleNamespace(case_batch_size=case_batch_size)

    def field(self, case: dict[str, object]) -> FieldResult:
        shift = torch.sigmoid(self.parameter + float(case["objective_weight"]))
        psf = torch.stack(
            (
                torch.stack((torch.zeros_like(shift), torch.zeros_like(shift), torch.zeros_like(shift))),
                torch.stack((torch.zeros_like(shift), 1.0 - shift, shift)),
                torch.stack((torch.zeros_like(shift), torch.zeros_like(shift), torch.zeros_like(shift))),
            )
        )
        return FieldResult(
            psf=psf,
            valid_fraction=torch.ones((), dtype=torch.float64),
            pixel_pitch_mm=1.0,
            edge_fraction=torch.zeros((), dtype=torch.float64),
            valid_mask=torch.ones(4, dtype=torch.bool),
        )

    def field_batch(self, cases: list[dict[str, object]]) -> FieldResult:
        results = [self.field(case) for case in cases]
        return FieldResult(
            psf=torch.stack([result.psf for result in results]),
            valid_fraction=torch.stack([result.valid_fraction for result in results]),
            pixel_pitch_mm=torch.tensor(
                [float(result.pixel_pitch_mm) for result in results], dtype=torch.float64
            ),
            edge_fraction=torch.stack([result.edge_fraction for result in results]),
            valid_mask=torch.stack([result.valid_mask for result in results]),
        )


class _DetachedModel:
    def __init__(self, *, case_batch_size: int = 8) -> None:
        self.config = SimpleNamespace(case_batch_size=case_batch_size)

    def field(self, case: dict[str, object]) -> FieldResult:
        psf = torch.zeros((3, 3), dtype=torch.float64)
        psf[1, 1] = 1.0
        return FieldResult(
            psf=psf,
            valid_fraction=torch.ones((), dtype=torch.float64),
            pixel_pitch_mm=1.0,
            edge_fraction=torch.zeros((), dtype=torch.float64),
            valid_mask=torch.ones(4, dtype=torch.bool),
        )

    def field_batch(self, cases: list[dict[str, object]]) -> FieldResult:
        results = [self.field(case) for case in cases]
        return FieldResult(
            psf=torch.stack([result.psf for result in results]),
            valid_fraction=torch.stack([result.valid_fraction for result in results]),
            pixel_pitch_mm=torch.ones(len(results), dtype=torch.float64),
            edge_fraction=torch.stack([result.edge_fraction for result in results]),
            valid_mask=torch.stack([result.valid_mask for result in results]),
        )


def test_fixed_distance_contract_is_exact() -> None:
    assert [spec.label for spec in DISTANCE_SPECS] == ["D500", "D1000", "Dinf"]
    assert [spec.object_distance_mm for spec in DISTANCE_SPECS[:2]] == [500.0, 1000.0]
    assert math.isinf(DISTANCE_SPECS[2].object_distance_mm)


def test_config_derives_fov_grid_from_bounds_and_step() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        MinimalConfig(fov_min_deg=-50.0, fov_max_deg=55.0, fov_step_deg=11.0)
    config = MinimalConfig(device="cpu", requested_np=32, fft_size_px=64)
    assert config.fov_step_deg == 11.0
    assert config.weights_json.endswith("multidistance_weights.json")
    assert config.max_accepted_steps == 50
    assert config.early_stopping_patience == 7
    assert config.case_batch_size == 8
    assert config.fov_min_deg == -55.0
    assert config.fov_max_deg == 55.0
    assert config.legacy_pupil_phase is False
    assert config.phase_reference == "biot_reference_sphere"
    assert config.remove_tilt is False
    with pytest.raises(ValueError, match="legacy_pupil_phase"):
        MinimalConfig(legacy_pupil_phase=True)
    with pytest.raises(ValueError, match="remove_tilt"):
        MinimalConfig(remove_tilt=True)
    with pytest.raises(ValueError, match="integer multiple"):
        MinimalConfig(fov_min_deg=-32.0)
    with pytest.raises(ValueError, match="case_batch_size"):
        MinimalConfig(case_batch_size=0)


def test_raw_psf_batch_reuses_verified_multidistance_field_batch() -> None:
    model = object.__new__(MinimalOpticalModel)
    calls: list[list[str]] = []
    expected = FieldResult(
        psf=torch.zeros((3, 8, 8), dtype=torch.float64),
        valid_fraction=torch.tensor([1.0, 0.75, 0.5], dtype=torch.float64),
        pixel_pitch_mm=torch.tensor([0.01, 0.011, 0.012], dtype=torch.float64),
        edge_fraction=torch.zeros(3, dtype=torch.float64),
        valid_mask=torch.ones((3, 4), dtype=torch.bool),
    )

    def field_batch(cases):
        calls.append([str(case["case_id"]) for case in cases])
        return expected

    model.field_batch = field_batch
    cases = [{"case_id": f"c{index}"} for index in range(3)]
    result = model.raw_psf_batch(cases)
    assert calls == [["c0", "c1", "c2"]]
    assert result.psf is expected.psf
    assert result.valid_fraction is expected.valid_fraction
    assert result.pixel_pitch_mm is expected.pixel_pitch_mm


def test_training_log_appends_durable_human_readable_records(tmp_path) -> None:
    path = tmp_path / "training.log"
    _append_training_log(path, "[pal-train] attempt=3 accepted=2/50 update=ACCEPT loss=0.8")
    _append_training_log(path, "[pal-train] INTERRUPTED phase=training_sweep error=KeyboardInterrupt")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "attempt=3 accepted=2/50" in lines[0]
    assert "INTERRUPTED phase=training_sweep" in lines[1]


def test_psf_second_moment_is_energy_normalized_and_differentiable() -> None:
    shift = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    coord = torch.linspace(-1.0, 1.0, 7, dtype=torch.float64)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    logits = -20.0 * ((xx - shift).square() + yy.square())
    psf = torch.softmax(logits.reshape(-1), dim=0).reshape(7, 7)
    moment = psf_second_moment_mm2(psf, pixel_pitch_mm=0.1)
    assert torch.all(psf >= 0.0)
    assert torch.allclose(psf.sum(), torch.ones((), dtype=torch.float64), atol=1.0e-12)
    assert torch.isfinite(moment)
    moment.backward()
    assert shift.grad is not None
    assert torch.isfinite(shift.grad)
    assert abs(float(shift.grad)) > 0.0


def test_weighted_m2_evaluation_accumulates_gradient_and_rows() -> None:
    model = _ToyModel()
    loss, rows, health = _evaluate(model, _toy_cases(), with_grad=True)
    assert math.isfinite(loss)
    assert len(rows) == 3
    assert math.isclose(sum(row["objective_weight"] for row in rows), 1.0)
    assert health["case_count"] == 3
    assert set(health["by_distance"]) == {"D500", "D1000", "Dinf"}
    assert model.parameter.grad is not None
    assert torch.isfinite(model.parameter.grad)
    assert abs(float(model.parameter.grad)) > 0.0
    assert all(row["weighted_m2_mm2"] >= 0.0 for row in rows)


def _repeated_toy_cases(count: int, weights: list[float]) -> list[dict[str, object]]:
    assert len(weights) == count
    return [
        {
            "case_id": f"D500_r00_c{index:02d}",
            "distance_label": "D500",
            "zone": "near",
            "field_x_deg": float(index),
            "field_y_deg": 0.0,
            "objective_weight": weights[index],
        }
        for index in range(count)
    ]


def test_evaluate_calls_backward_once_per_positive_batch_and_keeps_partial_last_batch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = _ToyModel(case_batch_size=4)
    hook_calls = 0

    def count_backward(_gradient: torch.Tensor) -> torch.Tensor:
        nonlocal hook_calls
        hook_calls += 1
        return _gradient

    model.parameter.register_hook(count_backward)
    cases = _repeated_toy_cases(9, [1.0 / 9.0] * 9)
    _, rows, _ = _evaluate(model, cases, with_grad=True)

    assert hook_calls == 3
    assert [row["case_id"] for row in rows] == [case["case_id"] for case in cases]
    output = capsys.readouterr().out
    assert "batch 3/3 cases 9-9/9 grad=True" in output


def test_all_zero_weight_batch_traces_but_skips_backward() -> None:
    class CountingToyModel(_ToyModel):
        def __init__(self) -> None:
            super().__init__(case_batch_size=4)
            self.batch_sizes: list[int] = []

        def field_batch(self, cases: list[dict[str, object]]) -> FieldResult:
            self.batch_sizes.append(len(cases))
            return super().field_batch(cases)

    model = CountingToyModel()
    hook_calls = 0

    def count_backward(_gradient: torch.Tensor) -> torch.Tensor:
        nonlocal hook_calls
        hook_calls += 1
        return _gradient

    model.parameter.register_hook(count_backward)
    cases = _repeated_toy_cases(8, [0.0] * 4 + [0.25] * 4)
    _, rows, _ = _evaluate(model, cases, with_grad=True)

    assert model.batch_sizes == [4, 4]
    assert hook_calls == 1
    assert all(row["weighted_loss"] == 0.0 for row in rows[:4])


def test_evaluation_progress_resumes_only_after_complete_batch(tmp_path) -> None:
    class InterruptAfterFirstBatch(_ToyModel):
        def __init__(self) -> None:
            super().__init__(case_batch_size=4)
            self.calls = 0

        def field_batch(self, cases: list[dict[str, object]]) -> FieldResult:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("intentional batch interruption")
            return super().field_batch(cases)

    cases = _repeated_toy_cases(9, [1.0 / 9.0] * 9)
    progress_path = tmp_path / "evaluation_progress.pt"
    interrupted = InterruptAfterFirstBatch()
    with pytest.raises(RuntimeError, match="intentional batch interruption"):
        _evaluate(
            interrupted,
            cases,
            with_grad=False,
            progress_path=progress_path,
            identity_sha256="batch-test",
        )
    saved = torch.load(progress_path, map_location="cpu")
    assert saved["next_case_index"] == 4
    assert len(saved["rows"]) == 4

    resumed = _ToyModel(case_batch_size=4)
    _, rows, _ = _evaluate(
        resumed,
        cases,
        with_grad=False,
        progress_path=progress_path,
        identity_sha256="batch-test",
    )
    assert [row["case_id"] for row in rows] == [case["case_id"] for case in cases]
    completed = torch.load(progress_path, map_location="cpu")
    assert completed["next_case_index"] == 9


def test_weighted_m2_evaluation_fails_when_case_is_detached() -> None:
    with pytest.raises(RuntimeError, match="detached"):
        _evaluate(_DetachedModel(), _toy_cases(), with_grad=True)


def test_zone_distance_baseline_normalization_makes_baseline_loss_one() -> None:
    model = _ToyModel()
    _, raw_rows, _ = _evaluate(model, _toy_cases(), with_grad=False)
    baseline_metrics = _baseline_metric_table(raw_rows, require_complete=False)
    model.parameter.grad = None
    loss, rows, _ = _evaluate(
        model, _toy_cases(), with_grad=True, baseline_metrics=baseline_metrics
    )
    assert loss == pytest.approx(1.0, abs=1.0e-12)
    assert all(row["normalized_metric"] == pytest.approx(1.0, abs=1.0e-12) for row in rows)


class _AstigToyModel(_ToyModel):
    def __init__(self, *, case_batch_size: int = 8) -> None:
        super().__init__(case_batch_size=case_batch_size)
        self.astig_calls = 0

    def astig_A_by_zone(self) -> dict[str, torch.Tensor]:
        self.astig_calls += 1
        value = self.parameter.square() + 1.0
        return {"astig_left": value, "astig_right": value + 1.0}


def test_astigmatism_zone_uses_m_over_a_astigmatism_a() -> None:
    model = _AstigToyModel()
    cases = [
        {
            "case_id": "D500_astig_left",
            "distance_label": "D500",
            "zone": "astig_left",
            "field_x_deg": 0.0,
            "field_y_deg": 0.0,
            "objective_weight": 0.5,
        },
        {
            "case_id": "D1000_astig_right",
            "distance_label": "D1000",
            "zone": "astig_right",
            "field_x_deg": 0.0,
            "field_y_deg": 0.0,
            "objective_weight": 0.5,
        },
    ]
    loss, rows, _ = _evaluate(model, cases, with_grad=True)
    expected = 0.5 * float(model.parameter.square() + 1.0) + 0.5 * float(model.parameter.square() + 2.0)
    assert loss == pytest.approx(expected, rel=1.0e-12)
    assert all(row["loss_metric_name"] == "astig_A_D" for row in rows)
    assert all(row["astig_A_D"] > 1.0 for row in rows)
    assert model.astig_calls == 1
    assert model.parameter.grad is not None
    assert torch.isfinite(model.parameter.grad)


def test_real_model_raw_psf_and_gradient_reach_7x7_pal() -> None:
    config = MinimalConfig(
        device="cpu",
        requested_np=32,
        fft_size_px=64,
        maximum_edge_fraction=0.5,
    )
    module = FixedWeightNURBSPerturbation(device="cpu", dtype=torch.float64)
    model = MinimalOpticalModel(config, module)
    try:
        result = model.field(
            {
                "case_id": "D500_r05_c05",
                "distance_label": "D500",
                "field_x_deg": 0.0,
                "field_y_deg": 0.0,
                "objective_weight": 1.0,
            }
        )
        assert result.psf.shape == (64, 64)
        assert torch.all(torch.isfinite(result.psf))
        assert torch.all(result.psf >= 0.0)
        assert torch.allclose(result.psf.sum(), torch.ones((), dtype=torch.float64), atol=1.0e-12)
        assert 0.0 < float(result.valid_fraction) <= 1.0
        moment = psf_second_moment_mm2(result.psf, pixel_pitch_mm=result.pixel_pitch_mm)
        moment.backward()
        assert module.inner_q.grad is not None
        assert torch.all(torch.isfinite(module.inner_q.grad))
        assert int((module.inner_q.grad.abs() > 0.0).sum()) >= 2
    finally:
        model.close()


def test_real_model_eight_case_batch_matches_scalar_psf_m2_and_pal_gradient() -> None:
    config = MinimalConfig(
        device="cpu",
        requested_np=32,
        fft_size_px=64,
        case_batch_size=8,
        maximum_edge_fraction=0.5,
    )
    module = FixedWeightNURBSPerturbation(device="cpu", dtype=torch.float64)
    model = MinimalOpticalModel(config, module)
    case_specs = (
        ("D500", 0.0, 0.0),
        ("D500", 3.0, 2.0),
        ("D1000", 8.0, -6.0),
        ("D1000", -4.0, 7.0),
        ("Dinf", -7.0, 5.0),
        ("Dinf", 2.0, -8.0),
        ("D500", 10.0, 0.0),
        ("D1000", -9.0, -3.0),
    )
    cases = [
        {
            "case_id": f"{label}_batch_{index}",
            "distance_label": label,
            "field_x_deg": field_x,
            "field_y_deg": field_y,
            "objective_weight": 0.125,
        }
        for index, (label, field_x, field_y) in enumerate(case_specs)
    ]
    try:
        scalar_results = [model.field(case) for case in cases]
        scalar_loss = sum(
            float(case["objective_weight"])
            * psf_second_moment_mm2(
                result.psf, pixel_pitch_mm=result.pixel_pitch_mm
            )
            for case, result in zip(cases, scalar_results)
        )
        scalar_loss.backward()
        assert module.inner_q.grad is not None
        scalar_gradient = module.inner_q.grad.detach().clone()
        module.inner_q.grad = None

        batched = model.field_batch(cases)
        batched_m2 = psf_second_moment_mm2(
            batched.psf, pixel_pitch_mm=batched.pixel_pitch_mm
        )
        batch_weights = torch.full((8,), 0.125, dtype=torch.float64)
        (batched_m2 * batch_weights).sum().backward()
        assert module.inner_q.grad is not None

        assert batched.psf.shape == (8, 64, 64)
        assert torch.equal(
            batched.valid_mask,
            torch.stack([result.valid_mask for result in scalar_results]),
        )
        assert torch.allclose(
            batched.psf,
            torch.stack([result.psf for result in scalar_results]),
            atol=2.0e-11,
            rtol=1.0e-11,
        )
        scalar_m2 = torch.stack(
            [
                psf_second_moment_mm2(
                    result.psf.detach(), pixel_pitch_mm=result.pixel_pitch_mm
                )
                for result in scalar_results
            ]
        )
        assert torch.allclose(batched_m2.detach(), scalar_m2, atol=2.0e-12, rtol=1.0e-11)
        assert torch.allclose(
            module.inner_q.grad,
            scalar_gradient,
            atol=1.0e-8,
            rtol=2.0e-8,
        )
    finally:
        model.close()
