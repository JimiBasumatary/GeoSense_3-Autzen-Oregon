
import unittest

from src.phase3_lidar.slope_risk_indicator import (
    calculate_slope_indicator,
)


class TestSlopeRiskIndicator(unittest.TestCase):

    def test_slope_summary_has_valid_values(self):
        result = calculate_slope_indicator()

        self.assertGreater(result["valid_cell_count"], 0)
        self.assertGreaterEqual(result["slope_min_degrees"], 0)
        self.assertLessEqual(result["slope_max_degrees"], 90)
        self.assertGreaterEqual(result["slope_mean_degrees"], 0)

    def test_slope_class_percentages_sum_to_100(self):
        result = calculate_slope_indicator()

        total = sum(
            item["percentage_of_valid_cells"]
            for item in result["slope_classes"].values()
        )

        self.assertAlmostEqual(total, 100.0, places=2)

    def test_slope_classes_cover_all_valid_cells(self):
        result = calculate_slope_indicator()

        total_cells = sum(
            item["valid_cell_count"]
            for item in result["slope_classes"].values()
        )

        self.assertEqual(total_cells, result["valid_cell_count"])

    def test_indicator_discloses_limitations(self):
        result = calculate_slope_indicator()

        self.assertIn(
            "not validated",
            result["interpretation"],
        )
        self.assertGreater(len(result["limitations"]), 0)


if __name__ == "__main__":
    unittest.main()
