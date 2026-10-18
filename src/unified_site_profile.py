
"""Build a unified site profile from existing LiDAR and Street View outputs."""

from pathlib import Path
from datetime import datetime, timezone
import json
import sys

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lidar_analyzer import (
    analyze_bbox,
    lidar_summary,
    raster_summary,
)
from src.phase3_lidar.streetview_semantic import analyze_location


def build_profile(image_filename="141300064660795.jpg"):
    """Combine LiDAR metadata, terrain summaries, and one street image result."""
    processed = ROOT / "data" / "lidar" / "processed"
    raw_lidar = ROOT / "data" / "lidar" / "raw" / "autzen-classified.copc.laz"

    # Read LiDAR and terrain products using the existing analyzer functions.
    lidar = lidar_summary(raw_lidar)
    ground_lidar = lidar_summary(processed / "autzen_ground.laz")

    terrain = {
        "dem": raster_summary(processed / "autzen_dem.tif"),
        "hillshade": raster_summary(processed / "autzen_hillshade.tif"),
        "slope": raster_summary(processed / "autzen_slope.tif"),
        "aspect": raster_summary(processed / "autzen_aspect.tif"),
    }

    # Get the scene composition and preliminary walkability proxy.
    streetview = analyze_location(image_filename)
    bbox_bounds = (637000, 851000, 637100, 851100)
    bbox_profile = analyze_bbox(bbox_bounds)

    profile = {
        "project": "GeoSense Agent - Phase III",
        "study_area": "Autzen Stadium area, Eugene, Oregon, USA",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "profile_type": "Unified LiDAR and Street View profile",
        "lidar": {
            "original_point_cloud": lidar,
            "ground_point_cloud": ground_lidar,
        },
               "terrain": terrain,
        "bounding_box_analysis": bbox_profile,
        "streetview": streetview,
        "spatial_matching": {
            "status": "not_verified",
            "note": (
                "The selected street image has not been spatially matched "
                "to a LiDAR bounding box. Do not interpret these sources "
                "as measurements of the same exact location."
            ),
        },
        "limitations": [
            "Street scene classes are model predictions, not verified ground truth.",
            "Walkability score is a preliminary sample-relative proxy, not a validated safety score.",
            "LiDAR summary statistics describe the source dataset and are not site-specific unless a bounding-box analysis is performed.",
            "Street imagery and LiDAR have not been spatially matched.",
        ],
    }

    output_dir = ROOT / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "unified_site_profile_callable.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2, ensure_ascii=False)

    return profile


if __name__ == "__main__":
    result = build_profile()
    print("UNIFIED SITE PROFILE: SUCCESS")
    print("Image:", result["streetview"]["image"])
    print("Walkability proxy:", result["streetview"]["walkability_score"])
    print("Saved: outputs/reports/unified_site_profile_callable.json")
