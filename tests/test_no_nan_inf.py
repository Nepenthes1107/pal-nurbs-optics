import unittest

import numpy as np

from optics import sanitize_and_energy_normalize_psf


class TestNoNanInf(unittest.TestCase):
    def test_sanitize_and_normalize_removes_nan_inf(self):
        psf = np.array(
            [
                [np.nan, np.inf, -1.0],
                [0.0, 2.0, 3.0],
            ],
            dtype=np.float64,
        )

        normalized = sanitize_and_energy_normalize_psf(psf)

        self.assertTrue(np.isfinite(normalized).all())
        self.assertTrue((normalized >= 0).all())
        self.assertAlmostEqual(float(normalized.sum()), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
