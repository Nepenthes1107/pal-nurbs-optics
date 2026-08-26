import unittest

import numpy as np
import torch

from optics import Lensdata


class TestPSFNormalization(unittest.TestCase):
    def test_compute_psf_output_is_energy_normalized(self):
        lens = Lensdata(device=torch.device("cpu"))
        pupil = np.ones((32, 32), dtype=np.complex128)
        psf = lens._compute_psf([pupil], n_i=64, d_delta=0.001, f_stop=100, methods="fft", show_plot=False)

        self.assertTrue(np.isfinite(psf).all())
        self.assertTrue((psf >= 0).all())
        self.assertAlmostEqual(float(psf.sum()), 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
