
import unittest
from unittest.mock import mock_open, patch

from src.phase3_lidar.streetview_semantic import analyze_location


class TestAnalyzeLocation(unittest.TestCase):

    def setUp(self):
        self.percentages_csv = (
            "image,width,height,road,sidewalk,building,"
            "vegetation,sky,car\n"
            "sample.jpg,1024,576,30,10,15,20,20,5\n"
        )

        self.scores_csv = (
            "filename,walkability_proxy,primary_rank,"
            "walkability_alternative,alternative_rank\n"
            "sample.jpg,72.5,1,68.0,2\n"
        )

    def test_scene_composition_has_six_classes(self):
        with patch(
            "builtins.open",
            mock_open(read_data=self.percentages_csv),
        ):
            with patch(
                "pathlib.Path.open",
                side_effect=[
                    mock_open(read_data=self.percentages_csv)(),
                    mock_open(read_data=self.scores_csv)(),
                ],
            ):
                result = analyze_location("sample.jpg")

        expected = {
            "road", "sidewalk", "building",
            "vegetation", "sky", "car",
        }
        self.assertEqual(
            set(result["scene_composition"].keys()),
            expected,
        )

    def test_walkability_score_is_numeric_and_in_range(self):
        with patch(
            "pathlib.Path.open",
            side_effect=[
                mock_open(read_data=self.percentages_csv)(),
                mock_open(read_data=self.scores_csv)(),
            ],
        ):
            result = analyze_location("sample.jpg")

        self.assertIsInstance(result["walkability_score"], float)
        self.assertGreaterEqual(result["walkability_score"], 0)
        self.assertLessEqual(result["walkability_score"], 100)

    def test_result_contains_requested_image(self):
        with patch(
            "pathlib.Path.open",
            side_effect=[
                mock_open(read_data=self.percentages_csv)(),
                mock_open(read_data=self.scores_csv)(),
            ],
        ):
            result = analyze_location("sample.jpg")

        self.assertEqual(result["image"], "sample.jpg")

    def test_unknown_image_raises_value_error(self):
        with patch(
            "pathlib.Path.open",
            return_value=mock_open(
                read_data=self.percentages_csv
            )(),
        ):
            with self.assertRaises(ValueError):
                analyze_location("missing.jpg")


if __name__ == "__main__":
    unittest.main(verbosity=2)
