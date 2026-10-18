
import unittest

from src.phase3_lidar.canopy_occupancy import (
    calculate_canopy_occupancy,
)


class TestCanopyOccupancy(unittest.TestCase):

    def test_grid_dimensions_and_counts(self):
        result = calculate_canopy_occupancy()

        self.assertEqual(result["grid_dimensions"]["rows"], 100)
        self.assertEqual(result["grid_dimensions"]["columns"], 100)

        self.assertGreater(result["observed_cells"], 0)
        self.assertGreaterEqual(
            result["cells_with_class_5_vegetation"], 0
        )
        self.assertLessEqual(
            result["cells_with_class_5_vegetation"],
            result["observed_cells"],
        )

    def test_occupancy_is_within_valid_range(self):
        result = calculate_canopy_occupancy()
        percentage = result["vegetation_cell_occupancy_percent"]

        self.assertIsNotNone(percentage)
        self.assertGreaterEqual(percentage, 0)
        self.assertLessEqual(percentage, 100)

    def test_metric_is_not_mislabeled_as_canopy_cover(self):
        result = calculate_canopy_occupancy()

        self.assertIn(
            "not validated canopy-cover percentage",
            result["interpretation"],
        )

    def test_invalid_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            calculate_canopy_occupancy(
                bounds=(637300, 850500, 637000, 850800)
            )


if __name__ == "__main__":
    unittest.main()
