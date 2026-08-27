from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from basics import Material, Ray
import biot.e2e.system as e2e_system
from biot.e2e.pal_nurbs import FixedWeightNURBSPerturbation
from biot.e2e.psf_fft import effective_biot_pupil_sample_count
from biot.e2e.system import (
    FitSpec,
    LocalGradient3Surface,
    LocalGridSagSurface,
    build_fitted_e2e_system,
    make_aimed_pupil_rays,
    make_aimed_reference_ray,
    rotation_matrix_xyz,
    trace_system_to_image_with_phase,
)


GRAD3_XLSX = Path(__file__).resolve().parents[1] / "eye_image_glass_grad3.xlsx"
SMALL_FIT = FitSpec(control_shape=(9, 9), sample_shape=(21, 21), degree=3)


def _build_grad3(
    *, distance_mm: float | str = 500.0, field_x_deg: float = 0.0,
    field_y_deg: float = 0.0, perturbation=None,
):
    return build_fitted_e2e_system(
        GRAD3_XLSX,
        object_distance=distance_mm,
        field_x_deg=field_x_deg,
        field_y_deg=field_y_deg,
        wavelength_nm=555.0,
        fit_spec=SMALL_FIT,
        device="cpu",
        dtype=torch.float64,
        back_perturbation=perturbation,
    )


def test_grad3_loader_and_e2e_adapter_preserve_surface_contract() -> None:
    system, temp_path = _build_grad3()
    try:
        grin = [surface for surface in system.surfaces if isinstance(surface, LocalGradient3Surface)]
        assert len(grin) == 2
        first, second = (surface.source for surface in grin)
        assert first.type == second.type == "GRIN3"
        assert first.material_name == second.material_name == "grada"
        assert first.coeff is None and second.coeff is None
        assert first.delta_t == second.delta_t == pytest.approx(1.0)
        assert grin[0].integration_step_mm == grin[1].integration_step_mm == pytest.approx(5.0e-3)
        assert float(first.n0) == pytest.approx(1.368)
        assert float(first.Nr2) == pytest.approx(-0.001978)
        assert float(first.Nz1) == pytest.approx(0.049057)
        assert float(first.Nz2) == pytest.approx(-0.015427)
        assert float(second.n0) == pytest.approx(1.407)
        assert float(second.Nz2) == pytest.approx(-0.006605)
        first_exit = first.get_ior(
            torch.tensor(0.0), torch.tensor(0.0), torch.tensor(float(first.thickness))
        )
        assert float(first_exit) == pytest.approx(float(second.axial_ior()), abs=5.0e-7)
        assert system.stop_semi_diameter_mm == pytest.approx(1.5)
        assert isinstance(system.back_surface, LocalGridSagSurface)
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def test_original_gridsag_torch_form_matches_scipy_values_and_derivatives() -> None:
    system, temp_path = _build_grad3()
    try:
        assert isinstance(system.back_surface, LocalGridSagSurface)
        source = system.lens.surfaces[2]
        generator = torch.Generator(device="cpu").manual_seed(20260823)
        random_xy = -39.5 + 79.0 * torch.rand(
            (1000, 2), generator=generator, dtype=torch.float64
        )
        boundary_xy = torch.tensor(
            [
                [-40.0, 0.0],
                [40.0, 0.0],
                [0.0, -40.0],
                [0.0, 40.0],
                [-40.0, -40.0],
                [40.0, 40.0],
            ],
            dtype=torch.float64,
        )
        xy = torch.cat((random_xy, boundary_xy), dim=0)
        expected_sag, expected_negative_dx, expected_negative_dy, _ = source.get_surface(
            xy[:, 0], xy[:, 1]
        )
        got_sag, got_dx, got_dy = system.back_surface.sag_and_derivatives(
            xy[:, 0], xy[:, 1]
        )
        assert torch.allclose(got_sag, expected_sag, atol=1.0e-12, rtol=0.0)
        assert torch.allclose(got_dx, -expected_negative_dx, atol=1.0e-11, rtol=0.0)
        assert torch.allclose(got_dy, -expected_negative_dy, atol=1.0e-11, rtol=0.0)
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def test_gradient_material_fails_closed_and_model_glass_uses_buchdahl() -> None:
    with pytest.raises(TypeError, match="per-surface"):
        Material("grada").ior(torch.tensor(555.0, dtype=torch.float64))

    n_d, v_d = 1.565, 57.44
    material = Material(f"{n_d}/{v_d}")
    got = float(material.ior(torch.tensor(555.0, dtype=torch.float64)))
    assert got == pytest.approx(1.5667972939958934, abs=2.0e-12)


def test_finite_object_launch_uses_real_stop_and_common_object_point() -> None:
    system, temp_path = _build_grad3(distance_mm=500.0)
    try:
        reference = make_aimed_reference_ray(system, device="cpu", dtype=torch.float64)
        system.reference_ray = reference
        rays = make_aimed_pupil_rays(
            system,
            sample_count=3,
            pupil_radius_mm=None,
            field_x_deg=0.0,
            field_y_deg=0.0,
            device="cpu",
            dtype=torch.float64,
        )
        assert rays.launch_opl_mm is not None
        backward_t = -500.0 / rays.directions[:, 2]
        object_points = rays.origins_mm + backward_t[:, None] * rays.directions
        assert float((object_points - object_points[:1]).abs().max()) < 2.0e-9
        expected_opl = torch.linalg.norm(rays.origins_mm - object_points[:1], dim=-1)
        assert torch.allclose(rays.launch_opl_mm, expected_opl, atol=2.0e-10, rtol=0.0)
        assert float(rays.directions[:, :2].std(dim=0).max()) > 0.0
        with pytest.raises(ValueError, match="physical stop radius"):
            make_aimed_pupil_rays(
                system,
                sample_count=3,
                pupil_radius_mm=2.0,
                field_x_deg=0.0,
                field_y_deg=0.0,
                device="cpu",
                dtype=torch.float64,
            )
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def test_e2e_coordinate_break_uses_biot_vis_rotation_order() -> None:
    got = rotation_matrix_xyz(11.0, -7.0, 3.0, device="cpu", dtype=torch.float64)
    tx, ty, tz = (math.radians(value) for value in (11.0, -7.0, 3.0))
    rx = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, math.cos(tx), -math.sin(tx)], [0.0, math.sin(tx), math.cos(tx)]],
        dtype=torch.float64,
    )
    ry = torch.tensor(
        [[math.cos(ty), 0.0, math.sin(ty)], [0.0, 1.0, 0.0], [-math.sin(ty), 0.0, math.cos(ty)]],
        dtype=torch.float64,
    )
    rz = torch.tensor(
        [[math.cos(tz), -math.sin(tz), 0.0], [math.sin(tz), math.cos(tz), 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    assert torch.allclose(got, rz @ ry @ rx, atol=2.0e-15, rtol=0.0)
    assert not torch.allclose(got, rx @ ry @ rz, atol=1.0e-8, rtol=0.0)


def test_grad3_full_e2e_trace_is_finite_and_nurbs_gradient_reaches_image() -> None:
    perturbation = FixedWeightNURBSPerturbation(device="cpu", dtype=torch.float64)
    system, temp_path = _build_grad3(distance_mm=500.0, perturbation=perturbation)
    try:
        system.reference_ray = make_aimed_reference_ray(system, device="cpu", dtype=torch.float64)
        rays = make_aimed_pupil_rays(
            system,
            sample_count=3,
            pupil_radius_mm=None,
            field_x_deg=0.0,
            field_y_deg=0.0,
            device="cpu",
            dtype=torch.float64,
        )
        trace = trace_system_to_image_with_phase(
            system, rays, phase_reference="biot_reference_sphere"
        )
        assert bool(trace.valid.any())
        assert bool(torch.isfinite(trace.spots_mm[trace.valid]).all())
        assert bool(torch.isfinite(trace.phase_rad[trace.valid]).all())
        assert bool(torch.isfinite(trace.reference_opl_mm[trace.valid]).all())
        valid_spots = trace.spots_mm[trace.valid]
        centred = valid_spots - valid_spots.mean(dim=0, keepdim=True)
        loss = centred.square().sum(dim=-1).mean()
        loss.backward()
        gradient = perturbation.inner_q.grad
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert bool((gradient.abs() > 0.0).any())
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def test_grad3_activation_checkpoint_preserves_forward_and_nurbs_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = []
    gradients = []
    for enabled in (False, True):
        monkeypatch.setattr(e2e_system, "GRIN_ACTIVATION_CHECKPOINT", enabled)
        perturbation = FixedWeightNURBSPerturbation(device="cpu", dtype=torch.float64)
        system, temp_path = _build_grad3(
            distance_mm=500.0,
            field_x_deg=0.0,
            field_y_deg=0.0,
            perturbation=perturbation,
        )
        try:
            system.reference_ray = make_aimed_reference_ray(
                system, device="cpu", dtype=torch.float64,
            )
            rays = make_aimed_pupil_rays(
                system,
                sample_count=3,
                pupil_radius_mm=None,
                field_x_deg=0.0,
                field_y_deg=0.0,
                device="cpu",
                dtype=torch.float64,
            )
            trace = trace_system_to_image_with_phase(
                system, rays, phase_reference="biot_reference_sphere",
            )
            loss = (
                trace.phase_rad[trace.valid].square().mean()
                + trace.spots_mm[trace.valid].square().mean()
            )
            loss.backward()
            assert perturbation.inner_q.grad is not None
            outputs.append(
                (
                    trace.spots_mm.detach().clone(),
                    trace.phase_rad.detach().clone(),
                    trace.valid.detach().clone(),
                )
            )
            gradients.append(perturbation.inner_q.grad.detach().clone())
        finally:
            if temp_path is not None:
                Path(temp_path).unlink(missing_ok=True)

    assert torch.equal(outputs[0][2], outputs[1][2])
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])
    assert torch.equal(gradients[0], gradients[1])


@pytest.mark.parametrize(
    ("distance_mm", "field_x_deg", "field_y_deg"),
    (
        (500.0, 0.0, 0.0),
        (2000.0, 30.0, -20.0),
        (100000.0, -25.0, 15.0),
        ("Infinity", 20.0, 10.0),
    ),
)
def test_grad3_e2e_image_spots_and_phase_match_authoritative_biot_trace(
    distance_mm: float | str, field_x_deg: float, field_y_deg: float,
) -> None:
    system, temp_path = _build_grad3(
        distance_mm=distance_mm,
        field_x_deg=field_x_deg,
        field_y_deg=field_y_deg,
    )
    try:
        reference = make_aimed_reference_ray(system, device="cpu", dtype=torch.float64)
        system.reference_ray = reference
        rays = make_aimed_pupil_rays(
            system,
            sample_count=3,
            pupil_radius_mm=None,
            field_x_deg=0.0,
            field_y_deg=0.0,
            device="cpu",
            dtype=torch.float64,
        )
        e2e = trace_system_to_image_with_phase(system, rays, phase_reference="image_surface")
        wavelength_mm = torch.tensor(system.wavelength_nm * 1.0e-6, dtype=torch.float64)
        phase_init = (
            rays.launch_opl_mm - reference.launch_opl_mm.reshape(-1)[0]
        ) * (2.0 * torch.pi / wavelength_mm)
        def biot_ray() -> Ray:
            return Ray(
                rays.origins_mm[:, None, :],
                rays.directions[:, None, :],
                wavelength=torch.tensor(system.wavelength_nm, dtype=torch.float64),
                phase=phase_init[:, None],
                device=torch.device("cpu"),
            )

        _, biot_valid = system.lens.trace(
            biot_ray(), is_fixed=True, flag=False, OPD_flag=True
        )
        biot_sensor, biot_image_ray = system.lens.trace_eyesensor(
            biot_ray(), ignore_invalid=False, is_fixed=True, flag=False
        )
        biot_spots = biot_sensor.reshape(-1, 3)[:, :2].to(dtype=torch.float64)
        biot_phase = biot_image_ray.phase.reshape(-1).to(dtype=torch.float64)
        biot_valid = biot_valid.reshape(-1) & torch.isfinite(biot_spots).all(dim=-1)
        assert torch.equal(e2e.valid, biot_valid)
        assert bool(e2e.valid.any())
        assert torch.allclose(
            e2e.spots_mm[e2e.valid],
            biot_spots[biot_valid],
            atol=2.0e-11,
            rtol=0.0,
        )
        assert torch.allclose(
            e2e.phase_rad[e2e.valid],
            biot_phase[biot_valid],
            atol=3.0e-10,
            rtol=0.0,
        )
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("distance_mm", "field_x_deg", "field_y_deg"),
    (
        (2000.0, 30.0, -20.0),
        ("Infinity", 20.0, 10.0),
    ),
)
def test_grad3_e2e_reference_sphere_complex_pupil_matches_biot_fft_psf_i(
    distance_mm: float | str, field_x_deg: float, field_y_deg: float,
) -> None:
    requested_np = 8
    sample_count = effective_biot_pupil_sample_count(requested_np)
    system, temp_path = _build_grad3(
        distance_mm=distance_mm,
        field_x_deg=field_x_deg,
        field_y_deg=field_y_deg,
    )
    try:
        system.reference_ray = make_aimed_reference_ray(
            system, device="cpu", dtype=torch.float64
        )
        rays = make_aimed_pupil_rays(
            system,
            sample_count=sample_count,
            pupil_radius_mm=None,
            field_x_deg=0.0,
            field_y_deg=0.0,
            device="cpu",
            dtype=torch.float64,
        )
        e2e = trace_system_to_image_with_phase(
            system, rays, phase_reference="biot_reference_sphere"
        )
        assert bool(e2e.valid.all())

        axis = torch.linspace(-1.0, 1.0, sample_count, dtype=torch.float64)
        xx, yy = torch.meshgrid(axis, axis, indexing="xy")
        mask = xx.square() + yy.square() <= 1.0
        assert int(mask.sum()) == int(e2e.phase_rad.numel())
        e2e_pupil = torch.zeros(
            (sample_count, sample_count), dtype=torch.complex128
        )
        e2e_pupil[mask] = torch.exp(1j * e2e.phase_rad.reshape(-1))

        # fft_psf_i is the authoritative BIOT/BIOT_vis complex-pupil path.
        # Suppress its diagnostic prints so the regression remains readable.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            pupils, _ = system.lens.fft_psf_i(
                requested_np,
                64,
                torch.tensor(system.wavelength_nm, dtype=torch.float64),
                Hx=0.0,
                Hy=0.0,
                legacy_pupil_phase=False,
            )
        biot_pupil = torch.from_numpy(np.asarray(pupils[0])).to(torch.complex128)
        assert biot_pupil.shape == e2e_pupil.shape
        assert torch.equal(biot_pupil.abs() > 0.0, mask)

        # A single global piston is optically irrelevant and is the only
        # freedom allowed in this comparison.
        cross = (biot_pupil[mask] * e2e_pupil[mask].conj()).sum()
        aligned_e2e = e2e_pupil * torch.exp(1j * torch.angle(cross))
        assert torch.allclose(
            aligned_e2e[mask], biot_pupil[mask], atol=1.0e-8, rtol=0.0
        )
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)
