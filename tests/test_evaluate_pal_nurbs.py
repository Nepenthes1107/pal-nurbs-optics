from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from scipy.signal import fftconvolve

import evaluate_pal_nurbs as evaluator


def test_weighted_mtf_is_finite_and_dc_normalized() -> None:
    axis = np.arange(130, dtype=np.float64) - 64.5
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    psf = np.exp(-(xx * xx + yy * yy) / (2.0 * 4.0**2))
    psf /= psf.sum()
    scores, sag, tan = evaluator._weighted_mtf(psf, 1.0e-3)
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    assert np.isfinite(sag).all() and np.isfinite(tan).all()
    assert abs(float(sag[0]) - 1.0) < 1e-12
    assert abs(float(tan[0]) - 1.0) < 1e-12


def test_evaluation_grid_has_three_distances_and_81_fields() -> None:
    class Config:
        pass

    config = Config()
    groups = evaluator._distance_cases(config)
    assert [item[0] for item in groups] == ["D500", "D1000", "Dinf"]
    assert all(len(item[2]) == 81 for item in groups)


def test_render_psf_has_fixed_shape_energy_and_physical_pitch(monkeypatch) -> None:
    monkeypatch.setattr(evaluator, "RAW_SIZE_PX", 8)
    monkeypatch.setattr(evaluator, "RENDER_SIZE_PX", 4)
    monkeypatch.setattr(evaluator, "CROP_PHYSICAL_SIZE_MM", 0.04)
    raw = np.zeros((8, 8), dtype=np.float64)
    raw[4, 4] = 1.0
    render, pitch = evaluator._render_psf(raw, 0.01)
    assert render.shape == (4, 4)
    assert np.isfinite(render).all()
    assert (render >= 0.0).all()
    assert abs(float(render.sum()) - 1.0) <= 1.0e-10
    assert pitch == pytest.approx(0.01)


def test_render_psf_rejects_insufficient_native_support(monkeypatch) -> None:
    monkeypatch.setattr(evaluator, "CROP_PHYSICAL_SIZE_MM", 1.0)
    raw = np.zeros((8, 8), dtype=np.float64)
    raw[4, 4] = 1.0
    with pytest.raises(ValueError, match="support is smaller"):
        evaluator._render_psf(raw, 0.01)


def test_condition_hdf5_resumes_exact_nodes_and_fails_on_corruption(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluator, "FIELD_VALUES", (-1.0, 1.0))
    monkeypatch.setattr(evaluator, "FIELD_COUNT", 4)
    monkeypatch.setattr(evaluator, "RAW_SIZE_PX", 8)
    monkeypatch.setattr(evaluator, "RENDER_SIZE_PX", 4)
    monkeypatch.setattr(evaluator, "CROP_PHYSICAL_SIZE_MM", 0.04)
    cases = [
        {"case_id": f"c{index}", "field_x_deg": x, "field_y_deg": y}
        for index, (x, y) in enumerate(((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)))
    ]

    class Model:
        def __init__(self, fail_after: int | None) -> None:
            self.calls = 0
            self.fail_after = fail_after

        def field(self, case):
            del case
            self.calls += 1
            if self.fail_after is not None and self.calls > self.fail_after:
                raise RuntimeError("synthetic interruption")
            raw = torch.zeros((8, 8), dtype=torch.float64)
            raw[4, 4] = 1.0
            common = {"valid_fraction": torch.tensor(0.75, dtype=torch.float64)}
            if hasattr(evaluator.pal, "DISTANCE_SPECS"):
                return SimpleNamespace(psf=raw, pixel_pitch_mm=0.01, **common)
            return SimpleNamespace(raw_psf=raw, raw_pixel_pitch_mm=0.01, **common)

    root = tmp_path / "psf_database"
    first = Model(fail_after=2)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        evaluator._build_condition_database(
            root,
            label="D500",
            distance=500.0,
            state="baseline",
            cases=cases,
            model=first,
            identity_sha256="identity",
            checkpoint_sha256="checkpoint",
            resume=False,
            completed_conditions=0,
        )
    partial = root / "D500_baseline.partial.h5"
    with h5py.File(partial, "r") as handle:
        assert handle["completed"][:].tolist() == [1, 1, 0, 0]

    resumed = Model(fail_after=None)
    final = evaluator._build_condition_database(
        root,
        label="D500",
        distance=500.0,
        state="baseline",
        cases=cases,
        model=resumed,
        identity_sha256="identity",
        checkpoint_sha256="checkpoint",
        resume=True,
        completed_conditions=0,
    )
    assert resumed.calls == 2
    assert final.name == "D500_baseline.h5"
    assert not partial.exists()
    with h5py.File(final, "r") as handle:
        assert set(handle.keys()) == {
            "field_xy_deg",
            "raw_psf",
            "render_psf",
            "raw_pixel_pitch_mm",
            "render_pixel_pitch_mm",
            "valid_fraction",
            "completed",
            "node_sha256",
        }
        assert handle["raw_psf"].shape == (4, 8, 8)
        assert handle["render_psf"].shape == (4, 4, 4)
        assert handle["completed"][:].tolist() == [1, 1, 1, 1]

    contract = evaluator._condition_contract(
        label="D500",
        distance=500.0,
        state="baseline",
        identity_sha256="identity",
        checkpoint_sha256="checkpoint",
    )
    with h5py.File(final, "r+") as handle:
        handle["valid_fraction"][0] = 0.5
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluator._validate_condition_file(final, contract, verify_render_contract=True)


def test_blur_scale_one_matches_reference_and_scale_four_is_display_only() -> None:
    chart = np.zeros((8, 8), dtype=np.float64)
    chart[3:5, 3:5] = 1.0
    psf = np.zeros((8, 8), dtype=np.float64)
    psf[4, 4] = 1.0
    chart_before = chart.copy()
    psf_before = psf.copy()
    historical = evaluator._normalize_display(fftconvolve(chart, psf, mode="same"))
    np.testing.assert_array_equal(evaluator._chart_tile(chart, psf, 1.0), historical)
    scaled = evaluator._chart_tile(chart, psf, 4.0)
    assert scaled.shape == chart.shape
    assert np.isfinite(scaled).all()
    np.testing.assert_array_equal(chart, chart_before)
    np.testing.assert_array_equal(psf, psf_before)


def test_evaluate_default_blur_scale_is_four() -> None:
    assert evaluator.evaluate.__kwdefaults__["blur_scale"] == 4.0
