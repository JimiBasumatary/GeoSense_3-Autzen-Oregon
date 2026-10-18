
import unittest
from unittest.mock import patch

from src.phase3_lidar.analyze_bbox_wgs84 import analyze_bbox_wgs84


class TestWGS84BoundingBoxAnalysis(unittest.TestCase):

    @patch("src.phase3_lidar.analyze_bbox_wgs84.analyze_bbox")
    @patch("src.phase3_lidar.analyze_bbox_wgs84.wgs84_bbox_to_lidar")
    def test_coordinates_are_converted_before_analysis(
        self, mock_convert, mock_analyze
    ):
        mock_convert.return_value = (
            635500.0, 850200.0, 639300.0, 853600.0
        )

        mock_analyze.return_value = {
            "bounds": {
                "xmin": 635500.0,
                "ymin": 850200.0,
                "xmax": 639300.0,
                "ymax": 853600.0,
            },
            "coverage": {"lidar_intersects": True},
            "metrics": {"ground_point_count": 100},
            "limitations": [],
        }

        result = analyze_bbox_wgs84(
            -123.0755, 44.0537, -123.0614, 44.0627
        )

        mock_convert.assert_called_once_with(
            -123.0755, 44.0537, -123.0614, 44.0627
        )
        mock_analyze.assert_called_once_with(
            (635500.0, 850200.0, 639300.0, 853600.0)
        )

        self.assertEqual(
            result["input_coordinate_system"], "EPSG:4326 (WGS84)"
        )
        self.assertEqual(
            result["analysis_coordinate_system"], "EPSG:2992"
        )
        self.assertEqual(
            result["metrics"]["ground_point_count"], 100
        )

    def test_invalid_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            analyze_bbox_wgs84(
                -123.0614, 44.0537, -123.0755, 44.0627
            )


if __name__ == "__main__":
    unittest.main()
