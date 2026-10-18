
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestLidarAnalyzer(unittest.TestCase):

    def test_required_inputs_exist(self):
        required = [
            ROOT / "data/lidar/raw/autzen-classified.copc.laz",
            ROOT / "data/lidar/processed/autzen_ground.laz",
            ROOT / "data/lidar/processed/autzen_dem.tif",
            ROOT / "outputs/street_imagery/segmentation_summary_fixed.csv",
        ]
        for path in required:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file(), f"Missing file: {path}")

    def test_generated_json_profile(self):
        path = ROOT / "outputs/reports/unified_site_profile.json"
        self.assertTrue(path.is_file())

        with path.open(encoding="utf-8") as file:
            profile = json.load(file)

        self.assertEqual(profile["project"], "GeoSense Agent - Phase III")
        self.assertEqual(
            profile["lidar"]["original_point_cloud"]["point_count"],
            10653336,
        )
        self.assertEqual(
            profile["lidar"]["ground_point_cloud"]["point_count"],
            6909405,
        )
        self.assertEqual(
            profile["street_imagery"]["segmentation"]["image_count"],
            5,
        )

    def test_terrain_rasters_share_dimensions(self):
        import rasterio

        names = [
            "autzen_dem.tif",
            "autzen_hillshade.tif",
            "autzen_slope.tif",
            "autzen_aspect.tif",
        ]
        dimensions = []
        for name in names:
            with rasterio.open(
                ROOT / "data/lidar/processed" / name
            ) as raster:
                dimensions.append(
                    (raster.width, raster.height, raster.crs)
                )

        self.assertTrue(all(item == dimensions[0] for item in dimensions))
        self.assertEqual(dimensions[0][0:2], (1143, 1553))

    def test_terrain_csv_exists(self):
        path = ROOT / "outputs/reports/terrain_metrics.csv"
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
