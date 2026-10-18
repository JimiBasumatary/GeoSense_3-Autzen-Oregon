
from pathlib import Path
import json
import numpy as np
import laspy
import pdal

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/lidar/raw/autzen-classified.copc.laz"
ORIGINAL_HAG = ROOT / "data/lidar/processed/autzen_classified_hag.laz"
TEST_HAG = ROOT / "data/lidar/processed/autzen_classified_hag_count8_50ft_test.laz"
REPORT_DIR = ROOT / "outputs/reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Raw LiDAR not found: {INPUT_FILE}")

if not ORIGINAL_HAG.exists():
    raise FileNotFoundError(f"Original HAG file not found: {ORIGINAL_HAG}")

if TEST_HAG.exists():
    raise FileExistsError(
        f"Test output already exists: {TEST_HAG}\n"
        "Rename or remove it before rerunning this test."
    )

pipeline_definition = {
    "pipeline": [
        str(INPUT_FILE),
        {
            "type": "filters.hag_nn",
            "count": 8,
            "max_distance": 50,
            "allow_extrapolation": "true",
            "class": 2
        },
        {
            "type": "writers.las",
            "filename": str(TEST_HAG),
            "compression": "true",
            "extra_dims": "all"
        }
    ]
}

print("Running experimental HAG calculation...")
print("Settings: count=8, max_distance=50 source units (feet)")
pipeline = pdal.Pipeline(json.dumps(pipeline_definition))
processed_count = pipeline.execute()

if not TEST_HAG.exists():
    raise RuntimeError("PDAL did not create the test output.")

original = laspy.read(ORIGINAL_HAG)
tested = laspy.read(TEST_HAG)

if len(original.points) != len(tested.points):
    raise RuntimeError("Point counts differ; direct comparison is unsafe.")

if not np.array_equal(original.classification, tested.classification):
    raise RuntimeError("Classifications differ; investigate before comparison.")

if not np.array_equal(original.X, tested.X) or not np.array_equal(
    original.Y, tested.Y
) or not np.array_equal(original.Z, tested.Z):
    raise RuntimeError("Coordinates differ; direct comparison is unsafe.")

original_hag = np.asarray(original["HeightAboveGround"], dtype=np.float64)
test_hag = np.asarray(tested["HeightAboveGround"], dtype=np.float64)

def stats(values):
    finite = values[np.isfinite(values)]
    return {
        "count": int(len(values)),
        "non_finite": int(np.count_nonzero(~np.isfinite(values))),
        "negative": int(np.count_nonzero(values < 0)),
        "min": float(np.min(finite)) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
        "p90": float(np.percentile(finite, 90)) if len(finite) else None,
        "max": float(np.max(finite)) if len(finite) else None
    }

difference = test_hag - original_hag
finite_difference = difference[np.isfinite(difference)]

report = {
    "input_file": str(INPUT_FILE.relative_to(ROOT)),
    "original_hag_file": str(ORIGINAL_HAG.relative_to(ROOT)),
    "test_hag_file": str(TEST_HAG.relative_to(ROOT)),
    "settings": {
        "filter": "filters.hag_nn",
        "count": 8,
        "max_distance": 50,
        "max_distance_units": "source coordinate units; expected feet for EPSG:2992",
        "allow_extrapolation": True,
        "ground_class": 2
    },
    "processed_count": int(processed_count),
    "original_points": int(len(original.points)),
    "test_points": int(len(tested.points)),
    "original_hag_stats": stats(original_hag),
    "test_hag_stats": stats(test_hag),
    "test_minus_original_hag": {
        "median": float(np.median(finite_difference)),
        "mean": float(np.mean(finite_difference)),
        "p90_absolute_difference": float(
            np.percentile(np.abs(finite_difference), 90)
        ),
        "changed_by_more_than_1ft": int(
            np.count_nonzero(np.abs(finite_difference) > 1)
        )
    },
    "warning": (
        "Experimental comparison only. Similar point counts and summary "
        "statistics do not establish accuracy. Validate against known "
        "terrain or independent reference data before selecting a method."
    )
}

report_path = REPORT_DIR / "hag_count8_50ft_test_comparison.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nHAG ALTERNATIVE TEST COMPLETED")
print(f"Processed points: {processed_count:,}")
print(f"Original HAG stats: {json.dumps(report['original_hag_stats'])}")
print(f"Test HAG stats:     {json.dumps(report['test_hag_stats'])}")
print(f"Difference stats:   {json.dumps(report['test_minus_original_hag'])}")
print(f"Test file: {TEST_HAG}")
print(f"Comparison report: {report_path}")
print("\nThis is a diagnostic test, not proof that the alternative is more accurate.")
