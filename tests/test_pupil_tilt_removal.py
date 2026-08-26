import unittest

import numpy as np
import torch

from optics import (
    Lensdata,
    compute_dc_normalized_mtf,
    sanitize_and_energy_normalize_psf,
    summarize_complex_pupil_phase,
)


class TestReferencePupilPhaseDiagnostics(unittest.TestCase):
    def _make_tilted_pupil(self, n=128, tilt_x=80.0, tilt_y=-45.0, piston=1.25):
        coord = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
        x_grid, y_grid = torch.meshgrid(coord, coord, indexing="xy")
        mask = (x_grid**2 + y_grid**2) <= 1.0
        phase = piston + tilt_x * x_grid + tilt_y * y_grid
        pupil = torch.zeros((n, n), dtype=torch.complex128)
        pupil[mask] = torch.exp(1j * phase[mask])
        return pupil, x_grid, y_grid, mask

    def test_phase_summary_estimates_wrapped_complex_tilt(self):
        pupil, x_grid, y_grid, mask = self._make_tilted_pupil()

        metrics = summarize_complex_pupil_phase(pupil, x_grid, y_grid, mask, "raw")

        self.assertAlmostEqual(metrics["raw_tilt_x_rad_per_norm"], 80.0, places=8)
        self.assertAlmostEqual(metrics["raw_tilt_y_rad_per_norm"], -45.0, places=8)
        self.assertEqual(metrics["raw_mask_points"], int(mask.sum().item()))

    def test_reference_corrected_synthetic_psf_is_normalized_for_mtf(self):
        pupil = np.ones((32, 32), dtype=np.complex128)
        lens = Lensdata(device=torch.device("cpu"))

        psf = lens._compute_psf([pupil], n_i=64, d_delta=0.001, f_stop=100, methods="fft", show_plot=False)
        psf = sanitize_and_energy_normalize_psf(psf)
        mtf = compute_dc_normalized_mtf(psf)

        self.assertTrue(np.isfinite(psf).all())
        self.assertTrue((psf >= 0).all())
        self.assertAlmostEqual(float(psf.sum()), 1.0, places=8)
        self.assertAlmostEqual(float(mtf[mtf.shape[0] // 2, mtf.shape[1] // 2]), 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
