
import unittest

from src.phase3_lidar.coordinate_utils import wgs84_bbox_to_lidar


class TestCoordinateUtils(unittest.TestCase):

    def test_autzen_bbox_conversion(self):
        result = wgs84_bbox_to_lidar(
            -123.0755,
            44.0537,
            -123.0614,
            44.0627,
        )

        self.assertEqual(len(result), 4)

        xmin, ymin, xmax, ymax = result

        self.assertLess(xmin, xmax)
        self.assertLess(ymin, ymax)

        # Broad checks for the expected Autzen-area coordinates.
        self.assertGreater(xmin, 600000)
        self.assertLess(xmax, 700000)
        self.assertGreater(ymin, 800000)
        self.assertLess(ymax, 900000)

    def test_reversed_bounds_raise_error(self):
        with self.assertRaises(ValueError):
            wgs84_bbox_to_lidar(
                -123.0614,
                44.0537,
                -123.0755,
                44.0627,
            )

    def test_invalid_latitude_raises_error(self):
        with self.assertRaises(ValueError):
            wgs84_bbox_to_lidar(
                -123.0755,
                95,
                -123.0614,
                96,
            )


if __name__ == "__main__":
    unittest.main()
