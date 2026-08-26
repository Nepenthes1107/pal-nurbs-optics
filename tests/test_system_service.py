import json
import unittest
from pathlib import Path

from biot.domain import Device
from biot.services import load_system_config, load_system_from_excel, save_system_config, summarize_system, validate_system


class TestSystemService(unittest.TestCase):
    def test_load_validate_and_roundtrip_system_config(self):
        repo_root = Path(__file__).resolve().parents[1]
        excel_path = repo_root / "eye_image_glass.xlsx"
        json_path = repo_root / "results" / "test_system_config.json"

        config = load_system_from_excel(excel_path, device=Device.CPU, np_pupil=32, ni_image=64)

        self.assertEqual(config.excel_path, excel_path)
        self.assertEqual(config.device, Device.CPU)
        self.assertEqual(config.np_pupil, 32)
        self.assertEqual(config.ni_image, 64)
        self.assertTrue(config.excel_sha256)
        self.assertEqual(validate_system(config), [])

        summary = summarize_system(config)
        self.assertIn("object_distance", summary)
        self.assertEqual(summary["device"], "cpu")

        save_system_config(config, json_path)
        restored = load_system_config(json_path)
        self.assertEqual(restored.excel_path, config.excel_path)
        self.assertEqual(restored.object_distance_mm, config.object_distance_mm)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "1.0")


if __name__ == "__main__":
    unittest.main()
