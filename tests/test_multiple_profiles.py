
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "outputs/reports/site_profiles"


class TestMultipleProfiles(unittest.TestCase):

    def load_profile(self, name):
        path = PROFILE_DIR / f"{name}.json"
        self.assertTrue(path.is_file(), f"Missing profile: {path}")

        with path.open(encoding="utf-8") as file:
            return json.load(file)

    def test_both_profiles_exist(self):
        area1 = self.load_profile("autzen_area_1")
        area2 = self.load_profile("autzen_area_2")

        self.assertEqual(area1["location_name"], "autzen_area_1")
        self.assertEqual(area2["location_name"], "autzen_area_2")

    def test_profiles_have_different_bounds(self):
        area1 = self.load_profile("autzen_area_1")
        area2 = self.load_profile("autzen_area_2")

        bounds1 = area1["bounding_box"]["bounds"]
        bounds2 = area2["bounding_box"]["bounds"]

        self.assertNotEqual(bounds1, bounds2)

    def test_profiles_have_independent_measurements(self):
        area1 = self.load_profile("autzen_area_1")
        area2 = self.load_profile("autzen_area_2")

        metrics1 = area1["bounding_box"]["metrics"]
        metrics2 = area2["bounding_box"]["metrics"]

        self.assertNotEqual(
            metrics1["ground_point_count"],
            metrics2["ground_point_count"],
        )
        self.assertNotEqual(
            metrics1["slope"]["mean"],
            metrics2["slope"]["mean"],
        )

    def test_profile_index_lists_both_areas(self):
        path = PROFILE_DIR / "profiles_index.json"
        self.assertTrue(path.is_file())

        with path.open(encoding="utf-8") as file:
            index = json.load(file)

        names = {
            item["location_name"] for item in index["profiles"]
        }
        self.assertEqual(
            names, {"autzen_area_1", "autzen_area_2"}
        )


if __name__ == "__main__":
    unittest.main()
