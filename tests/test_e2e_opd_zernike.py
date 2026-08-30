from __future__ import annotations

import numpy as np
import pytest
import torch

from optics import fit_wavefront_zernike
from biot.e2e.opd_zernike import (
    fit_low_order_opd_zernike_torch,
    low_order_osa_ansi_basis,
    normalized_pupil_disk_coordinates,
    z4_defocus_loss,
)
from biot.e2e.pal_nurbs import MinimalConfig, MinimalOpticalModel
from biot.e2e.regional_nurbs import FixedWeightNURBSPerturbation
from biot.e2e.system import trace_system_batch_to_image_with_phase


def test_low_order_opd_fit_recovers_coefficients_and_z4_gradient() -> None:
    sample_count = 31
    x, y = normalized_pupil_disk_coordinates(
        sample_count, device="cpu", dtype=torch.float64
    )
    basis = low_order_osa_ansi_basis(x, y)
    expected = torch.tensor(
        [2.0e-4, 3.0e-5, -4.0e-5, 8.0e-6, 1.5e-5, -7.0e-6],
        dtype=torch.float64,
        requires_grad=True,
    )
    opd = basis @ expected
    valid = torch.ones_like(opd, dtype=torch.bool)

    recovered = fit_low_order_opd_zernike_torch(
        opd, valid, sample_count=sample_count
    )
    torch.testing.assert_close(recovered, expected, rtol=0.0, atol=1.0e-15)

    loss = z4_defocus_loss(opd, valid, sample_count=sample_count)
    (gradient,) = torch.autograd.grad(loss, expected)
    analytic = 2.0 * float(expected[4].detach())
    assert float(gradient[4]) == pytest.approx(analytic, rel=1.0e-10, abs=1.0e-15)
    assert float(gradient[[0, 1, 2, 3, 5]].abs().max()) <= 1.0e-14

    epsilon = 1.0e-8
    plus = expected.detach().clone()
    minus = expected.detach().clone()
    plus[4] += epsilon
    minus[4] -= epsilon
    finite_difference = (
        z4_defocus_loss(basis @ plus, valid, sample_count=sample_count)
        - z4_defocus_loss(basis @ minus, valid, sample_count=sample_count)
    ) / (2.0 * epsilon)
    assert float(finite_difference) == pytest.approx(analytic, rel=1.0e-5)


def test_real_pal_corridor_and_near_opd_match_biot_numpy_and_keep_gradient() -> None:
    config = MinimalConfig(
        device="cpu", requested_np=64, fft_size_px=64, kernel_size_px=32
    )
    module = FixedWeightNURBSPerturbation(
        7, device="cpu", dtype=torch.float64
    )
    model = MinimalOpticalModel(config, module)
    cases = (
        {"case_id": "corridor", "distance_mm": 1000.0, "field_x_deg": 0.0, "field_y_deg": -15.0},
        {"case_id": "near", "distance_mm": 500.0, "field_x_deg": 0.0, "field_y_deg": -30.0},
    )
    try:
        systems, rays = [], []
        for case in cases:
            system, pupil_rays = model._system_and_rays(
                float(case["distance_mm"]),
                float(case["field_x_deg"]),
                float(case["field_y_deg"]),
            )
            systems.append(system)
            rays.append(pupil_rays)
        trace = trace_system_batch_to_image_with_phase(
            systems, rays, phase_reference=config.phase_reference
        )
        torch_coefficients = fit_low_order_opd_zernike_torch(
            trace.reference_opl_mm,
            trace.valid,
            sample_count=model.sample_count,
        )

        coordinates = np.linspace(-1.0, 1.0, model.sample_count)
        xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
        pupil = xx * xx + yy * yy <= 1.0
        numpy_coefficients = []
        for index in range(len(cases)):
            opd_grid = np.zeros_like(xx)
            valid_grid = np.zeros_like(pupil)
            values = trace.reference_opl_mm[index].detach().cpu().numpy()
            valid_values = trace.valid[index].detach().cpu().numpy()
            opd_grid[pupil] = values
            valid_grid[pupil] = valid_values
            rows, _ = fit_wavefront_zernike(
                opd_grid,
                xx,
                yy,
                valid_grid,
                config.wavelength_nm * 1.0e-6,
                n_max=2,
            )
            numpy_coefficients.append(
                [row["coefficient_opd_um"] * 1.0e-3 for row in rows]
            )
        expected = torch.as_tensor(numpy_coefficients, dtype=torch.float64)
        torch.testing.assert_close(
            torch_coefficients.detach(), expected, rtol=0.0, atol=1.0e-10
        )

        torch_coefficients[:, 4].square().sum().backward()
        assert module.inner_q.grad is not None
        assert bool(torch.isfinite(module.inner_q.grad).all())
        assert float(module.inner_q.grad.abs().max()) > 0.0
    finally:
        model.close()
