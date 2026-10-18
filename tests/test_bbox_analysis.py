
import unittest

from src.phase3_lidar.analyze_bbox import (
    analyze_bbox,
    get_coverage_status,
)


class TestBoundingBoxAnalysis(unittest.TestCase):

    def test_valid_bounding_box(self):
        result = analyze_bbox(
            (637000, 850500, 637300, 850800)
        )

        self.assertTrue(result["coverage"]["lidar_intersects"])
        self.assertGreater(
            result["metrics"]["ground_point_count"], 0
        )

    def test_invalid_bounding_box(self):
        with self.assertRaises(ValueError):
            analyze_bbox((637300, 850500, 637000, 850800))

    def test_box_outside_dataset(self):
        result = analyze_bbox((1, 1, 2, 2))

        self.assertFalse(result["coverage"]["lidar_intersects"])
        self.assertEqual(
            result["metrics"]["ground_point_count"], 0
        )

    def test_coverage_status_full(self):
        result = get_coverage_status(
            10, 10, 20, 20,
            0, 0, 30, 30,
        )

        self.assertEqual(result["status"], "full")
        self.assertTrue(result["fully_covered_bbox"])
        self.assertEqual(result["overlap_fraction"], 1.0)

    def test_coverage_status_partial(self):
        result = get_coverage_status(
            20, 20, 40, 40,
            0, 0, 30, 30,
        )

        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["intersects"])
        self.assertFalse(result["fully_covered_bbox"])
        self.assertAlmostEqual(
            result["overlap_fraction"], 0.25
        )

    def test_coverage_status_none(self):
        result = get_coverage_status(
            40, 40, 50, 50,
            0, 0, 30, 30,
        )

        self.assertEqual(result["status"], "none")
        self.assertFalse(result["intersects"])
        self.assertFalse(result["fully_covered_bbox"])
        self.assertEqual(result["overlap_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
