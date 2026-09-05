from __future__ import annotations

from types import SimpleNamespace

import h5py
from matplotlib import font_manager
import numpy as np
import pytest
import torch
from scipy.signal import fftconvolve

import evaluate_pal_nurbs as evaluator
from biot.e2e.weighted_mtf import weighted_mtf_mean_torch_batch


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


def test_training_weighted_mtf_matches_evaluator_and_preserves_autograd() -> None:
    axis = np.arange(512, dtype=np.float64) - 255.5
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    psf = np.exp(-((xx / 3.0) ** 2 + (yy / 5.0) ** 2) / 2.0)
    psf /= psf.sum()
    expected = float(evaluator._weighted_mtf(psf, 1.0e-3)[0][2])
    tensor = torch.tensor(psf, dtype=torch.float64, requires_grad=True)
    actual = weighted_mtf_mean_torch_batch(
        tensor.unsqueeze(0),
        pixel_pitch_mm=torch.tensor([1.0e-3], dtype=torch.float64),
    )[0]
    assert float(actual.detach()) == pytest.approx(expected, rel=0.0, abs=1.0e-12)
    actual.backward()
    assert tensor.grad is not None
    assert bool(torch.isfinite(tensor.grad).all())
    assert int(torch.count_nonzero(tensor.grad)) > 0


def test_weighted_mtf_field_interpolation_preserves_nodes_and_domain() -> None:
    axis = np.linspace(-2.0, 2.0, 5, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    native = 0.5 + 0.02 * xx + 0.03 * yy + 0.01 * xx * yy

    x_fine, y_fine, fine = evaluator._interpolate_weighted_mtf_map(
        native, axis, axis, resolution=21,
    )

    assert fine.shape == (21, 21)
    assert x_fine[[0, -1]].tolist() == [-2.0, 2.0]
    assert y_fine[[0, -1]].tolist() == [-2.0, 2.0]
    np.testing.assert_allclose(fine[::5, ::5], native, rtol=0.0, atol=1.0e-12)


def test_weighted_mtf_field_interpolation_rejects_missing_samples() -> None:
    axis = np.linspace(-2.0, 2.0, 5, dtype=np.float64)
    native = np.ones((5, 5), dtype=np.float64)
    native[2, 3] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf.*forbidden"):
        evaluator._interpolate_weighted_mtf_map(native, axis, axis)
    native[2, 3] = 1.01
    with pytest.raises(ValueError, match=r"physical \[0,1\] range"):
        evaluator._interpolate_weighted_mtf_map(native, axis, axis)


def test_weighted_mtf_stage_adds_interpolated_pngs_to_manifest(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fields = (-15.0, -5.0, 5.0, 15.0)
    monkeypatch.setattr(evaluator, "FIELD_VALUES", fields)
    monkeypatch.setattr(evaluator, "FIELD_COUNT", 16)
    database = tmp_path / "psf_database"
    database.mkdir()
    for distance_index, label in enumerate(("D500", "D1000", "Dinf")):
        for state_index, state in enumerate(("baseline", "optimized")):
            with h5py.File(database / f"{label}_{state}.h5", "w") as handle:
                values = (
                    np.arange(16, dtype=np.float64)
                    + 10.0 * distance_index
                    + 2.0 * state_index
                )
                handle.create_dataset("raw_psf", data=values[:, None, None])
                handle.create_dataset(
                    "raw_pixel_pitch_mm", data=np.full(16, 0.001, dtype=np.float64),
                )

    monkeypatch.setattr(
        evaluator,
        "_directional_weighted_mtf",
        lambda psf, pitch, *, softmin_temperature: (
            np.asarray(
                [
                    0.2 + 0.005 * float(psf[0, 0]),
                    0.19 + 0.005 * float(psf[0, 0]),
                    0.18 + 0.005 * float(psf[0, 0]),
                    0.17 + 0.005 * float(psf[0, 0]),
                ]
            ),
            0.17 + 0.005 * float(psf[0, 0]),
            np.ones((4, 2)),
        ),
    )

    def write_plot(path, *args, **kwargs):
        path.write_bytes(b"synthetic png")

    monkeypatch.setattr(evaluator, "_plot_map", write_plot)
    monkeypatch.setattr(evaluator, "_plot_interpolated_weighted_mtf_map", write_plot)
    evaluator._run_weighted_mtf(
        tmp_path,
        config=SimpleNamespace(directional_softmin_temperature=0.02),
        identity_sha256="identity",
        database_sha256="database",
    )

    output = tmp_path / "weighted_mtf"
    expected_interpolated = {
        f"{label}_{state}_mean_interpolated.png"
        for label in ("D500", "D1000", "Dinf")
        for state in ("baseline", "optimized", "delta")
    }
    assert expected_interpolated == {
        path.name for path in output.glob("*_mean_interpolated.png")
    }
    manifest = evaluator._json(output / "weighted_mtf_manifest.json")
    assert manifest["config"]["field_map_interpolation"] == {
        "purpose": "display_only",
        "method": "cubic",
        "resolution": 200,
        "domain": [-15.0, 15.0],
        "extrapolation": False,
        "native_nodes_preserved_abs_tolerance": 1.0e-12,
    }
    assert len(manifest["files"]) == 108


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
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
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
            self.batch_sizes: list[int] = []

        def raw_psf_batch(self, batch):
            self.calls += 1
            if self.fail_after is not None and self.calls > self.fail_after:
                raise RuntimeError("synthetic interruption")
            size = len(batch)
            self.batch_sizes.append(size)
            raw = torch.zeros((size, 8, 8), dtype=torch.float64)
            raw[:, 4, 4] = 1.0
            return SimpleNamespace(
                psf=raw,
                pixel_pitch_mm=torch.full((size,), 0.01, dtype=torch.float64),
                valid_fraction=torch.full((size,), 0.75, dtype=torch.float64),
            )

    root = tmp_path / "psf_database"
    first = Model(fail_after=1)
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
            psf_batch_size=3,
        )
    partial = root / "D500_baseline.partial.h5"
    with h5py.File(partial, "r") as handle:
        assert handle["completed"][:].tolist() == [1, 1, 1, 0]

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
        psf_batch_size=3,
    )
    assert first.batch_sizes == [3]
    assert resumed.calls == 1
    assert resumed.batch_sizes == [1]
    progress = capsys.readouterr().out
    assert "[pal-eval] phase=psf_database condition=1/6 name=D500_baseline" in progress
    terminal = [line for line in progress.splitlines() if "fields=4/4" in line]
    assert terminal == [
        "[pal-eval] phase=psf_database condition=1/6 name=D500_baseline "
        "batch=1/1 fields=4/4 total=4/24 status=DONE"
    ]
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


def test_condition_resume_after_last_batch_reports_terminal_count_once(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(evaluator, "FIELD_VALUES", (-1.0, 1.0))
    monkeypatch.setattr(evaluator, "FIELD_COUNT", 4)
    monkeypatch.setattr(evaluator, "RAW_SIZE_PX", 8)
    monkeypatch.setattr(evaluator, "RENDER_SIZE_PX", 4)
    monkeypatch.setattr(evaluator, "CROP_PHYSICAL_SIZE_MM", 0.04)
    cases = [
        {"case_id": f"c{index}", "field_x_deg": x, "field_y_deg": y}
        for index, (x, y) in enumerate(
            ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))
        )
    ]

    class Model:
        def __init__(self) -> None:
            self.calls = 0

        def raw_psf_batch(self, batch):
            self.calls += 1
            raw = torch.zeros((len(batch), 8, 8), dtype=torch.float64)
            raw[:, 4, 4] = 1.0
            return SimpleNamespace(
                psf=raw,
                pixel_pitch_mm=torch.full((len(batch),), 0.01, dtype=torch.float64),
                valid_fraction=torch.full((len(batch),), 0.75, dtype=torch.float64),
            )

    root = tmp_path / "psf_database"
    validate_node = evaluator._validate_database_node
    monkeypatch.setattr(
        evaluator,
        "_validate_database_node",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("after final batch")),
    )
    with pytest.raises(RuntimeError, match="after final batch"):
        evaluator._build_condition_database(
            root, label="D500", distance=500.0, state="baseline", cases=cases,
            model=Model(), identity_sha256="identity", checkpoint_sha256="checkpoint",
            resume=False, completed_conditions=0, psf_batch_size=4,
        )
    monkeypatch.setattr(evaluator, "_validate_database_node", validate_node)
    capsys.readouterr()

    resumed = Model()
    evaluator._build_condition_database(
        root, label="D500", distance=500.0, state="baseline", cases=cases,
        model=resumed, identity_sha256="identity", checkpoint_sha256="checkpoint",
        resume=True, completed_conditions=0, psf_batch_size=4,
    )
    progress = capsys.readouterr().out.splitlines()
    assert resumed.calls == 0
    assert [line for line in progress if "fields=4/4" in line] == [
        "[pal-eval] phase=psf_database condition=1/6 name=D500_baseline "
        "fields=4/4 total=4/24 status=DONE"
    ]


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


def test_evaluate_default_psf_batch_size_is_eight() -> None:
    assert evaluator.evaluate.__kwdefaults__["psf_batch_size"] == 8


def test_evaluation_plot_font_is_bundled_and_resolvable() -> None:
    resolved = font_manager.findfont(
        font_manager.FontProperties(family=evaluator.PLOT_FONT_FAMILY),
        fallback_to_default=False,
    )
    assert font_manager.FontProperties(fname=resolved).get_name() == evaluator.PLOT_FONT_FAMILY


def test_averfang_distribution_uses_physical_mm_axes_and_local_style(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_mm = np.arange(-5.0, 6.0)
    physical_y_mm = x_mm[::-1]
    values = np.arange(121, dtype=np.float64).reshape(11, 11) - 60.0
    saved_font_family = list(evaluator.plt.rcParams["font.family"])
    closed = []
    real_close = evaluator.plt.close
    monkeypatch.setattr(evaluator.plt, "close", closed.append)

    path = tmp_path / "delta_power_D.png"
    evaluator._plot_averfang_distribution(
        path, values, x_mm, physical_y_mm,
        colorbar_label="Power (D)", symmetric=True,
    )

    assert path.is_file()
    assert len(closed) == 1
    figure = closed[0]
    axis, colorbar_axis = figure.axes
    assert axis.get_xlabel() == "X (mm)"
    assert axis.get_ylabel() == "Y (mm)"
    assert axis.get_aspect() == 1.0
    assert axis.images[0].origin == "lower"
    assert axis.images[0].get_extent() == [-2.0, 2.0, -2.0, 2.0]
    np.testing.assert_array_equal(axis.images[0].get_array(), values[3:-3, 3:-3][::-1])
    low, high = axis.images[0].get_clim()
    assert low == -high
    assert not axis.collections
    assert colorbar_axis.get_title() == "Power (D)"
    assert list(evaluator.plt.rcParams["font.family"]) == saved_font_family
    real_close(figure)


def test_resume_keeps_sealed_identity_across_legacy_classification_drift(
    tmp_path,
) -> None:
    path = tmp_path / "evaluation_identity.json"
    saved_body = {
        "schema_version": evaluator.EVAL_SCHEMA,
        "source_run_identity_sha256": "source",
        "checkpoint_sha256": "checkpoint",
        "source_identity_legacy_schema": False,
        "psf_database": {"batch_size": 8},
        "runtime": {"platform": "linux"},
    }
    saved = {
        **saved_body,
        "identity_sha256": evaluator._canonical_sha256(saved_body),
    }
    evaluator._write_json(path, saved)
    before = path.read_bytes()

    current_body = {**saved_body, "source_identity_legacy_schema": True}
    bound = evaluator._bind_evaluation_identity(
        path, identity_body=current_body, resume=True,
    )

    assert bound == saved
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="differing fields: checkpoint_sha256"):
        evaluator._bind_evaluation_identity(
            path,
            identity_body={**current_body, "checkpoint_sha256": "changed"},
            resume=True,
        )


def test_resume_rejects_modified_evaluation_identity(tmp_path) -> None:
    path = tmp_path / "evaluation_identity.json"
    body = {
        "schema_version": evaluator.EVAL_SCHEMA,
        "source_identity_legacy_schema": False,
    }
    evaluator._write_json(
        path, {**body, "identity_sha256": evaluator._canonical_sha256(body)},
    )
    payload = evaluator._json(path)
    payload["source_identity_legacy_schema"] = True
    evaluator._write_json(path, payload)
    with pytest.raises(ValueError, match="malformed or has been modified"):
        evaluator._bind_evaluation_identity(path, identity_body=body, resume=True)


def test_load_checkpoint_selects_completed_stage_and_checks_identity(tmp_path) -> None:
    stage = tmp_path / "stage_7x7"
    stage.mkdir()
    checkpoint = stage / "final.pt"
    with checkpoint.open("wb") as handle:
        torch.save(
            {
                "control_count": 7,
                "identity_sha256": "source-identity",
                "state_dict": {"inner_q": torch.zeros((5, 5), dtype=torch.float64)},
            },
            handle,
        )
    summary = {
        "final_control_count": 19,
        "stages": [
            {"control_count": 7, "actual_steps": 50},
            {"control_count": 11, "actual_steps": 25},
            {"control_count": 19, "actual_steps": 0},
        ],
    }

    selected, payload = evaluator._load_checkpoint(
        tmp_path, summary, torch.device("cpu"), checkpoint_stage=7,
        source_identity_sha256="source-identity",
    )
    assert selected == checkpoint
    assert payload["control_count"] == 7

    with pytest.raises(ValueError, match="identity does not match"):
        evaluator._load_checkpoint(
            tmp_path, summary, torch.device("cpu"), checkpoint_stage=7,
            source_identity_sha256="different-identity",
        )


def test_native_psf_batch_rejects_scalar_only_model() -> None:
    with pytest.raises(TypeError, match="must implement raw_psf_batch"):
        evaluator._native_psf_batch(SimpleNamespace(), [{"case_id": "c0"}])
