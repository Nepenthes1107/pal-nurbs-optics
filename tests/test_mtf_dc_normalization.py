import unittest

import numpy as np
import torch

from optics import Lensdata, compute_dc_normalized_mtf


class TestMTFDCNormalization(unittest.TestCase):
    def test_compute_dc_normalized_mtf_dc_is_one(self):
        psf = np.zeros((64, 64), dtype=np.float64)
        psf[32, 32] = 1.0

        mtf = compute_dc_normalized_mtf(psf)
        self.assertAlmostEqual(float(mtf[32, 32]), 1.0, places=12)

    def test_cal_mtf_runs_with_dc_normalized_pipeline(self):
        rng = np.random.default_rng(42)
        psf = rng.random((64, 64), dtype=np.float64)

        lens = Lensdata(device=torch.device("cpu"))
        mtf_t, mtf_s = lens.cal_mtf(psf, n_i=64, d_delta=0.001, f_stop=50, show_plot=False)

        self.assertTrue(np.isfinite(float(mtf_t)))
        self.assertTrue(np.isfinite(float(mtf_s)))


if __name__ == "__main__":
    unittest.main()
