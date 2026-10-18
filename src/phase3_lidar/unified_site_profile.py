
"""
Exercise 5: Unified LiDAR and street-view site profile.

The street-view image is an image-level sample. Its geographic
relationship to the LiDAR bounding box has not been verified.
"""

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.phase3_lidar.lidar_analyzer import analyze_bbox


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "outputs" / "reports"
STREETVIEW_DIR = ROOT / "outputs" / "streetview"


@dataclass
class EnhancedSiteProfile:
    latitude: float
    longitude: float
    avg_building_height_m: float | None
    canopy_cover_pct: float | None
    impervious_surface_pct: float | None
    slope_risk_score: float | None
    scene_composition: dict[str, float] | None
    walkability_score: float | None


def read_streetview_results(
    image_filename: str,
) -> tuple[dict[str, Any] | None, str]:
    """Read previously calculated street-view CSV results."""

    percentages_path = STREETVIEW_DIR / "class_percentages.csv"
    scores_path = STREETVIEW_DIR / "walkability_scores_24.csv"

    scene_composition = None
    score_row = None

    try:
        with percentages_path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            for row in csv.DictReader(file):
                if row.get("image") == image_filename:
                    scene_composition = {
                        key: float(value)
                        for key, value in row.items()
                        if key not in {"image", "width", "height"}
                        and value not in (None, "")
                    }
                    break

        with scores_path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            for row in csv.DictReader(file):
                if row.get("filename") == image_filename:
                    score_row = row
                    break

    except (OSError, csv.Error) as error:
        return None, f"Street-view CSV unavailable: {error}"

    if scene_composition is None:
        return None, f"No segmentation result for {image_filename}"

    if score_row is None:
        return None, f"No walkability score for {image_filename}"

    try:
        walkability_score = float(score_row["walkability_proxy"])
        alternative_score = float(
            score_row["walkability_alternative"]
        )
        primary_rank = int(score_row["primary_rank"])
        alternative_rank = int(score_row["alternative_rank"])
    except (KeyError, TypeError, ValueError) as error:
        return None, f"Invalid walkability CSV record: {error}"

    result = {
        "image": image_filename,
        "scene_composition": scene_composition,
        "walkability_score": walkability_score,
        "primary_rank": primary_rank,
        "walkability_alternative": alternative_score,
        "alternative_rank": alternative_rank,
        "score_type": "sample-relative preliminary proxy",
    }

    return result, (
        "image_level_result; geographic_match_not_verified"
    )


def build_profile(
    lat: float,
    lon: float,
    bbox_radius_deg: float = 0.005,
    image_filename: str = "141300064660795.jpg",
) -> dict[str, Any]:
    """Build a profile from LiDAR and existing image-level results."""

    lat = float(lat)
    lon = float(lon)
    bbox_radius_deg = float(bbox_radius_deg)

    if not -90 <= lat <= 90:
        raise ValueError("Latitude must be between -90 and 90")

    if not -180 <= lon <= 180:
        raise ValueError("Longitude must be between -180 and 180")

    if bbox_radius_deg <= 0:
        raise ValueError("bbox_radius_deg must be positive")

    if (
        lat - bbox_radius_deg < -90
        or lat + bbox_radius_deg > 90
    ):
        raise ValueError("Bounding box extends beyond valid latitude")

    if not image_filename or Path(image_filename).name != image_filename:
        raise ValueError("image_filename must be a filename, not a path")

    bbox = (
        lat - bbox_radius_deg,
        lon - bbox_radius_deg,
        lat + bbox_radius_deg,
        lon + bbox_radius_deg,
    )

    # LiDAR analysis uses a WGS84 bounding box.
    lidar_result = analyze_bbox(bbox)

    # Read existing CSV results without importing the segmentation model.
    streetview_result, streetview_status = read_streetview_results(
        image_filename
    )

    scene_composition = None
    walkability_score = None

    if streetview_result is not None:
        scene_composition = streetview_result["scene_composition"]
        walkability_score = streetview_result["walkability_score"]

    profile = EnhancedSiteProfile(
        latitude=lat,
        longitude=lon,
        avg_building_height_m=lidar_result.get(
            "avg_building_height_m"
        ),
        canopy_cover_pct=lidar_result.get("canopy_cover_pct"),
        impervious_surface_pct=lidar_result.get(
            "impervious_surface_pct"
        ),
        slope_risk_score=lidar_result.get("slope_risk_score"),
        scene_composition=scene_composition,
        walkability_score=walkability_score,
    )

    result = {
        "project": "GeoSense Agent 2.0 - Phase III Exercise 5",
        "study_area": "Autzen Stadium area, Eugene, Oregon, USA",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "profile": asdict(profile),
        "bounding_box_wgs84": {
            "min_lat": bbox[0],
            "min_lon": bbox[1],
            "max_lat": bbox[2],
            "max_lon": bbox[3],
        },
        "lidar_analysis": lidar_result,
        "streetview_analysis": streetview_result,
        "spatial_matching": {
            "status": streetview_status,
            "warning": (
                "The image's geographic location has not been verified "
                "against the LiDAR bounding box. Do not interpret image "
                "results as measurements at the profile coordinate."
            ),
        },
        "limitations": [
            "Building height is not calculated for this arbitrary bounding box.",
            "Canopy occupancy is an exploratory proxy, not validated canopy cover.",
            "Impervious surface percentage is not calculated as an area-based measure.",
            "A validated single slope-risk score is not calculated.",
            "Walkability is a preliminary image-based proxy.",
            "Street-view and LiDAR spatial matching is unverified.",
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "enhanced_site_profile_example.json"

    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("Profile saved:", output_path)
    return result


if __name__ == "__main__":
    # Example coordinate near Autzen Stadium, Eugene, Oregon.
    profile_result = build_profile(44.0582, -123.0687)

    print("Profile generation completed.")
    print(
        "Street-view status:",
        profile_result["spatial_matching"]["status"],
    )
    print("Profile fields:", list(profile_result["profile"].keys()))
