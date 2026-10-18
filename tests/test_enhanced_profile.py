
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestEnhancedProfile(unittest.TestCase):

    def test_enhanced_profile_exists_and_is_valid_json(self):
        path = ROOT / "outputs/reports/unified_site_profile_enhanced.json"
        self.assertTrue(path.is_file())

        with path.open(encoding="utf-8") as file:
            profile = json.load(file)

        self.assertEqual(
            profile["project"],
            "GeoSense Agent - Phase III",
        )
        self.assertEqual(
            profile["study_area"],
            "Autzen Stadium area, Eugene, Oregon, USA",
        )

    def test_bounding_box_is_documented(self):
        path = ROOT / "outputs/reports/unified_site_profile_enhanced.json"

        with path.open(encoding="utf-8") as file:
            profile = json.load(file)

        bbox = profile["bounding_box_analysis"]
        self.assertEqual(bbox["area_sq_ft"], 90000.0)
        self.assertTrue(bbox["coverage"]["lidar_covered"])
        self.assertGreater(
            bbox["metrics"]["ground_point_count"], 0
        )

    def test_unavailable_sources_are_disclosed(self):
        path = ROOT / "outputs/reports/unified_site_profile_enhanced.json"

        with path.open(encoding="utf-8") as file:
            profile = json.load(file)

        sources = profile["data_sources"]
        self.assertEqual(
            sources["phase_i_results_for_autzen"]["status"],
            "not supplied",
        )
        self.assertEqual(
            sources["phase_ii_results_for_autzen"]["status"],
            "not supplied",
        )
        self.assertTrue(profile["limitations"])


if __name__ == "__main__":
    unittest.main()
