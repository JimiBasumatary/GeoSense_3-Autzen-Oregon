
import json
import unittest

from src.unified_site_profile import build_profile


class TestUnifiedSiteProfile(unittest.TestCase):

    def test_profile_contains_required_sections(self):
        profile = build_profile()

        required_keys = [
            "project",
            "study_area",
            "lidar",
            "terrain",
            "streetview",
            "spatial_matching",
            "limitations",
        ]

        for key in required_keys:
            with self.subTest(key=key):
                self.assertIn(key, profile)

    def test_profile_contains_terrain_products(self):
        profile = build_profile()

        for product in ["dem", "hillshade", "slope", "aspect"]:
            with self.subTest(product=product):
                self.assertIn(product, profile["terrain"])

    def test_profile_contains_streetview_result(self):
        profile = build_profile()

        self.assertIn("image", profile["streetview"])
        self.assertIn("scene_composition", profile["streetview"])
        self.assertIn("walkability_score", profile["streetview"])

    def test_profile_json_is_saved(self):
        build_profile()

        with open(
            "outputs/reports/unified_site_profile_callable.json",
            encoding="utf-8",
        ) as file:
            saved_profile = json.load(file)

        self.assertIn("lidar", saved_profile)
        self.assertIn("streetview", saved_profile)


if __name__ == "__main__":
    unittest.main(verbosity=2)
