
from pathlib import Path
import json
import numpy as np
import laspy
import pdal

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/lidar/raw/autzen-classified.copc.laz"
OUTPUT_DIR = ROOT / "data/lidar/processed"
REPORT_DIR = ROOT / "outputs/reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "autzen_classified_hag.laz"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input LiDAR not found: {INPUT_FILE}")

# Keep the source classifications. Calculate HAG without running SMRF
# over an already classified point cloud.
pipeline_definition = {
    "pipeline": [
        str(INPUT_FILE),
        {
            "type": "filters.hag_nn"
        },
        {
            "type": "writers.las",
            "filename": str(OUTPUT_FILE),
            "compression": "true",
            "extra_dims": "all"
        }
    ]
}

pipeline = pdal.Pipeline(json.dumps(pipeline_definition))
processed_count = pipeline.execute()

if not OUTPUT_FILE.exists():
    raise RuntimeError("PDAL did not create the output file.")

# Verify the saved output.
cloud = laspy.read(OUTPUT_FILE)
dimensions = list(cloud.point_format.dimension_names)

if "HeightAboveGround" not in dimensions:
    raise RuntimeError("HeightAboveGround dimension is missing.")

hag = np.asarray(cloud["HeightAboveGround"])
classes = np.asarray(cloud.classification)

report = {
    "input_file": str(INPUT_FILE.relative_to(ROOT)),
    "output_file": str(OUTPUT_FILE.relative_to(ROOT)),
    "processed_points": int(processed_count),
    "output_points": len(cloud.points),
    "output_size_bytes": OUTPUT_FILE.stat().st_size,
    "hag_dimension_present": "HeightAboveGround" in dimensions,
    "hag_min": float(np.min(hag)),
    "hag_max": float(np.max(hag)),
    "hag_mean": float(np.mean(hag)),
    "non_finite_hag_count": int(np.count_nonzero(~np.isfinite(hag))),
    "negative_hag_count": int(np.count_nonzero(hag < 0)),
    "classification_counts": {
        str(int(k)): int(np.count_nonzero(classes == k))
        for k in np.unique(classes)
    },
    "processing_notes": [
        "Original classification labels were retained by omitting SMRF.",
        "HAG was calculated using nearest-neighbour ground points.",
        "HAG values require further quality assessment."
    ]
}

report_path = REPORT_DIR / "pdal_pipeline_summary.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("HAG PIPELINE COMPLETED")
print(f"Processed points: {processed_count:,}")
print(f"Output points: {len(cloud.points):,}")
print(f"HAG dimension present: {'HeightAboveGround' in dimensions}")
print(f"HAG range: {np.min(hag):.2f} to {np.max(hag):.2f}")
print(f"Negative HAG values: {np.count_nonzero(hag < 0):,}")
print(f"Output: {OUTPUT_FILE}")
print(f"Report: {report_path}")
