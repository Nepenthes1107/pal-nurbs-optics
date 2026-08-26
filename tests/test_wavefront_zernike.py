import unittest

import numpy as np
from prysm.polynomials import ansi_j_to_nm, zernike_nm

from optics import _ansi_nm_to_j, _real_zernike_nm, fit_wavefront_zernike


class TestWavefrontZernikeFit(unittest.TestCase):
    def setUp(self):
        coordinates = np.linspace(-1.0, 1.0, 81)
        self.x, self.y = np.meshgrid(coordinates, coordinates, indexing="xy")
        self.mask = np.hypot(self.x, self.y) <= 1.0
        self.rho = np.hypot(self.x, self.y)
        self.theta = np.arctan2(self.y, self.x)
        self.wavelength_mm = 555e-6

    def test_recovers_known_low_and_high_order_coefficients(self):
        expected_mm = {
            (0, 0): 2.0e-4,
            (1, 1): -3.0e-5,
            (1, -1): 4.0e-5,
            (2, 0): 1.5e-5,
            (3, 3): -8.0e-6,
        }
        opd = np.zeros_like(self.x)
        for (n, m), coefficient in expected_mm.items():
            opd += coefficient * _real_zernike_nm(n, m, self.rho, self.theta)

        coefficients, metrics = fit_wavefront_zernike(
            opd, self.x, self.y, self.mask, self.wavelength_mm, n_max=3
        )
        recovered = {(row["n"], row["m"]): row for row in coefficients}

        for term, expected in expected_mm.items():
            self.assertAlmostEqual(recovered[term]["coefficient_opd_um"], expected * 1e3, places=10)
            self.assertAlmostEqual(
                recovered[term]["coefficient_waves"], expected / self.wavelength_mm, places=10
            )
        self.assertLess(metrics["zernike_residual_rms_mm"], 1e-15)
        self.assertEqual(metrics["zernike_mode_count"], 10)
        self.assertEqual(metrics["zernike_basis"], "osa_ansi_real_zernike_nm_rms_normalized")
        self.assertEqual(metrics["zernike_indexing"], "ansi_j_zero_based")

    def test_basis_matches_prysm_osa_ansi_definition_and_indexing(self):
        for n in range(6):
            for m in range(-n, n + 1, 2):
                expected = zernike_nm(n, m, self.rho, self.theta, norm=True)
                actual = _real_zernike_nm(n, m, self.rho, self.theta)
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
                self.assertEqual(ansi_j_to_nm(_ansi_nm_to_j(n, m)), (n, m))

    def test_rejects_invalid_or_insufficient_input(self):
        zeros = np.zeros((3, 3), dtype=float)
        mask = np.ones((3, 3), dtype=bool)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            fit_wavefront_zernike(zeros, zeros, zeros, mask, self.wavelength_mm, n_max=-1)
        with self.assertRaisesRegex(ValueError, "no valid points"):
            fit_wavefront_zernike(zeros, zeros, zeros, np.zeros_like(mask), self.wavelength_mm, n_max=0)
        with self.assertRaisesRegex(ValueError, "only"):
            fit_wavefront_zernike(zeros, zeros, zeros, mask, self.wavelength_mm, n_max=4)
        non_finite = zeros.copy()
        non_finite[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            fit_wavefront_zernike(non_finite, zeros, zeros, mask, self.wavelength_mm, n_max=0)


if __name__ == "__main__":
    unittest.main()
