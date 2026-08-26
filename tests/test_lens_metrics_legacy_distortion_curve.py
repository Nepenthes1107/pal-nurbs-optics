import unittest
from pathlib import Path

import numpy as np
import torch

from lens_metrics_core import compute_distortion_curve, load_lens, sample_legacy_positive_fields


class TestLegacyDistortionCurve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.lens = load_lens(repo_root / "eye_image_glass.xlsx", device=torch.device("cpu"))

    def test_one_sided_positive_tangent_sampling(self):
        theta = sample_legacy_positive_fields(5.0, 5, tan_uniform=True)
        self.assertGreater(theta[0], 0.0)
        self.assertAlmostEqual(theta[-1], 5.0, places=7)
        self.assertTrue(np.all(np.diff(theta) > 0.0))

    def test_far_reference_uses_trace_paraxial_slope(self):
        result = compute_distortion_curve(self.lens, fov_deg=2.0, field_num=5, distortion_type="rotating_eye_far")
        columns = result["columns"]
        refs = result["data"][:, columns.index("reference_index")]
        self.assertTrue((refs == -1.0).all())
        theta0 = result["data"][0, columns.index("theta_deg")]
        dist0 = result["data"][0, columns.index("distortion")]
        self.assertEqual(theta0, 0.0)
        self.assertAlmostEqual(float(dist0), 0.0, places=12)
        self.assertGreater(result["metadata"]["trace_first_field_y_deg"], 0.0)
        self.assertEqual(result["metadata"]["far_reference_mode"], "trace_paraxial_slope")
        self.assertEqual(result["metadata"]["far_reference_fit_sample_count"], 5)
        self.assertIn("tan(theta_in)", result["metadata"]["magnification_policy"])
        self.assertIn("deviates from MATLAB magnif(1)", result["metadata"]["compatibility_deviation"])
        self.assertIn("trace-based paraxial slope", result["metadata"]["magnification_reference_policy"])

    def test_near_reference_uses_trace_height_slope(self):
        result = compute_distortion_curve(self.lens, fov_deg=2.0, field_num=5, distortion_type="rotating_eye_near")
        columns = result["columns"]
        refs = result["data"][:, columns.index("reference_index")]
        self.assertTrue((refs == -1.0).all())
        dist0 = result["data"][0, columns.index("distortion")]
        self.assertAlmostEqual(float(dist0), 0.0, places=12)
        self.assertEqual(result["metadata"]["near_reference_mode"], "trace_height_slope")
        self.assertEqual(result["metadata"]["near_reference_fit_sample_count"], 5)
        self.assertIn("fitted_actual_axis", result["metadata"]["magnification_policy"])
        self.assertIn("deviates from MATLAB magnif(1)", result["metadata"]["compatibility_deviation"])
        self.assertIn("trace-based height slope", result["metadata"]["magnification_reference_policy"])

    def test_near_magnification_uses_ideal_over_axis_corrected_actual(self):
        result = compute_distortion_curve(self.lens, fov_deg=2.0, field_num=5, distortion_type="rotating_eye_near")
        columns = result["columns"]
        mag = result["data"][:, columns.index("magnification")]
        actual = result["data"][:, columns.index("actual_height_mm")]
        ideal = result["data"][:, columns.index("ideal_height_mm")]
        axis = result["metadata"]["near_reference_actual_axis_mm"]
        expected = ideal / (actual - axis)
        expected[0] = result["data"][0, columns.index("magnification_reference")]
        np.testing.assert_allclose(mag, expected, rtol=1e-8, atol=1e-10)
        self.assertIn("near: height_ideal/(height_actual-fitted_actual_axis)", result["metadata"]["magnification_policy"])


if __name__ == "__main__":
    unittest.main()
