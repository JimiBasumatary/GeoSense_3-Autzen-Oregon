
from pathlib import Path
from datetime import datetime, timezone
import json
import sys

ROOT = Path(__file__).resolve().parent.parent

# Allow Python to import modules from the project root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase3_lidar.analyze_bbox import analyze_bbox


def build_enhanced_profile():
    report_dir = ROOT / "outputs" / "reports"

    original_path = report_dir / "unified_site_profile.json"
    bbox_path = report_dir / "bbox_analysis_example.json"
    output_path = report_dir / "unified_site_profile_enhanced.json"

    if not original_path.is_file():
        raise FileNotFoundError(
            f"Original profile not found: {original_path}"
        )

    # Reuse the existing profile.
    with original_path.open("r", encoding="utf-8") as file:
        original_profile = json.load(file)

    # Generate a fresh, reproducible bounding-box analysis.
    bounds = (637000, 850500, 637300, 850800)
    bbox_profile = analyze_bbox(bounds)

    # Save a separate copy of the bounding-box result.
    with bbox_path.open("w", encoding="utf-8") as file:
        json.dump(bbox_profile, file, indent=2, ensure_ascii=False)

    enhanced_profile = {
        "project": original_profile.get("project"),
        "study_area": original_profile.get("study_area"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "profile_type": "Enhanced site profile",
        "original_profile_file": str(
            original_path.relative_to(ROOT)
        ),
        "bounding_box_analysis_file": str(
            bbox_path.relative_to(ROOT)
        ),
        "data_sources": {
            "lidar": {
                "status": "available",
                "source": "Autzen classified LiDAR point cloud",
            },
            "terrain": {
                "status": "available",
                "products": ["DEM", "hillshade", "slope", "aspect"],
            },
            "street_imagery": {
                "status": "available",
                "image_count": original_profile.get(
                    "street_imagery", {}
                ).get("segmentation", {}).get("image_count", 0),
                "qualification": (
                    "Predictions are not manually validated ground truth; "
                    "not spatially matched to the LiDAR box."
                ),
            },
            "phase_i_results_for_autzen": {
                "status": "not supplied",
                "contribution": "Not included",
            },
            "phase_ii_results_for_autzen": {
                "status": "not supplied",
                "contribution": "Not included",
            },
        },
        "bounding_box_analysis": bbox_profile,
        "limitations": [
            "Bounding box coordinates are in EPSG:2992, not WGS84.",
            "Building-point elevation range is not building height.",
            "Canopy cover and impervious-surface area have not been calculated.",
            "Slope values are descriptive, not a validated risk score.",
            "Street imagery has not been spatially matched to the LiDAR box.",
            "Phase I and Phase II data for Autzen were not supplied.",
        ],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(enhanced_profile, file, indent=2, ensure_ascii=False)

    print("ENHANCED PROFILE: SUCCESS")
    print(f"Original profile preserved: {original_path}")
    print(f"Bounding-box results: {bbox_path}")
    print(f"Enhanced profile saved: {output_path}")


if __name__ == "__main__":
    build_enhanced_profile()
