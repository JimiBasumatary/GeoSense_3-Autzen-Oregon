
from pathlib import Path
import json

import laspy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data/lidar/processed/autzen_classified_hag.laz"
REPORT_DIR = ROOT / "outputs/reports"
REPORT_FILE = REPORT_DIR / "hag_quality_diagnostic.json"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Processed LiDAR not found: {INPUT_FILE}")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

cloud = laspy.read(INPUT_FILE)
hag = np.asarray(cloud["HeightAboveGround"], dtype=float)
classes = np.asarray(cloud.classification)

if not np.all(np.isfinite(hag)):
    raise ValueError("Non-finite HAG values found; investigate before summarising.")

class_stats = {}

for class_code in np.unique(classes):
    mask = classes == class_code
    values = hag[mask]

    class_stats[str(int(class_code))] = {
        "point_count": int(mask.sum()),
        "negative_hag_count": int(np.sum(values < 0)),
        "minimum_hag_ft": float(np.min(values)),
        "median_hag_ft": float(np.median(values)),
        "maximum_hag_ft": float(np.max(values)),
    }

report = {
    "input_file": str(INPUT_FILE.relative_to(ROOT)),
    "total_points": int(len(hag)),
    "non_finite_hag_count": int(np.sum(~np.isfinite(hag))),
    "negative_hag_count": int(np.sum(hag < 0)),
    "negative_hag_percentage": float(np.mean(hag < 0) * 100),
    "minimum_hag_ft": float(np.min(hag)),
    "median_hag_ft": float(np.median(hag)),
    "maximum_hag_ft": float(np.max(hag)),
    "class_statistics": class_stats,
    "interpretation": [
        "Negative HAG values require interpretation by classification and location.",
        "This report describes values; it does not prove physical accuracy.",
        "No LiDAR points or classifications are modified by this script.",
    ],
}

REPORT_FILE.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("HAG QUALITY DIAGNOSTIC COMPLETED")
print(f"Points: {len(hag):,}")
print(f"Negative HAG values: {np.sum(hag < 0):,}")
print(f"Negative HAG percentage: {np.mean(hag < 0) * 100:.2f}%")
print(f"Minimum HAG: {np.min(hag):.2f} ft")
print(f"Maximum HAG: {np.max(hag):.2f} ft")
print(f"Report: {REPORT_FILE}")
print("\nCLASS | POINTS | NEGATIVE | MIN FT | MEDIAN FT | MAX FT")

for code, stats in class_stats.items():
    print(
        f"{int(code):5d} | "
        f"{stats['point_count']:7,d} | "
        f"{stats['negative_hag_count']:8,d} | "
        f"{stats['minimum_hag_ft']:7.2f} | "
        f"{stats['median_hag_ft']:9.2f} | "
        f"{stats['maximum_hag_ft']:7.2f}"
    )
