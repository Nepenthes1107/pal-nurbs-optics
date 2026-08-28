import numpy as np

import evaluate_pal_nurbs as evaluator


def test_weighted_mtf_is_finite_and_dc_normalized() -> None:
    axis = np.arange(130, dtype=np.float64) - 64.5
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    psf = np.exp(-(xx * xx + yy * yy) / (2.0 * 4.0**2))
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
