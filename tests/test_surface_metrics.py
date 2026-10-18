
import unittest
from unittest.mock import patch

import numpy as np

from src.phase3_lidar import surface_metrics


class TestSurfaceMetrics(unittest.TestCase):

    def test_area_and_point_density(self):
        result = surface_metrics.calculate_surface_metrics(
            (637000, 850500, 637300, 850800)
        )

        self.assertEqual(result["area_sq_ft"], 90000.0)

        counts = result["classified_point_counts"]
        density = result["point_density_per_sq_ft"]

        self.assertAlmostEqual(
            density["ground_class_2"],
            counts["ground_class_2"] / 90000.0,
        )
        self.assertAlmostEqual(
            density["high_vegetation_class_5"],
            counts["high_vegetation_class_5"] / 90000.0,
        )

    def test_invalid_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            surface_metrics.calculate_surface_metrics(
                (637300, 850500, 637000, 850800)
            )

    def test_point_density_is_not_cover_percentage(self):
        result = surface_metrics.calculate_surface_metrics(
            (637000, 850500, 637300, 850800)
        )

        self.assertNotIn(
            "canopy_cover_percent",
            result,
        )
        self.assertNotIn(
            "impervious_surface_percent",
            result,
        )
        self.assertIn(
            "Canopy-cover percentage and impervious-surface percentage "
            "are not calculated by this module.",
            result["limitations"],
        )


if __name__ == "__main__":
    unittest.main()
