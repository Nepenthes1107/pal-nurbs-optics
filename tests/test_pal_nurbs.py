from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from biot.e2e.pal_case_layout import DISTANCE_SPECS
from biot.e2e.pal_nurbs import (
    FieldResult,
    MinimalConfig,
    MinimalOpticalModel,
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
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))

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


class _DetachedModel:
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


def test_fixed_distance_contract_is_exact() -> None:
    assert [spec.label for spec in DISTANCE_SPECS] == ["D500", "D1000", "Dinf"]
    assert [spec.object_distance_mm for spec in DISTANCE_SPECS[:2]] == [500.0, 1000.0]
    assert math.isinf(DISTANCE_SPECS[2].object_distance_mm)


def test_config_requires_fixed_11_by_11_fov_grid() -> None:
    with pytest.raises(ValueError, match="fov_count=11"):
        MinimalConfig(fov_count=9)
    config = MinimalConfig(device="cpu", requested_np=32, fft_size_px=64)
    assert config.fov_count == 11
    assert config.weights_json.endswith("multidistance_weights.json")
    assert config.max_accepted_steps == 50
    assert config.early_stopping_patience == 7


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
    def astig_A_by_zone(self) -> dict[str, torch.Tensor]:
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
