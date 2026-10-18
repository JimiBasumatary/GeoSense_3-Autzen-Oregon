
import json
from pathlib import Path
from collections import Counter

import laspy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "lidar" / "raw" / "autzen-classified.copc.laz"
OUTPUT = ROOT / "outputs" / "reports" / "surface_class_inspection.json"


def inspect_classes():
    counts = Counter()

    with laspy.open(INPUT) as reader:
        header = reader.header
        point_count = header.point_count

        for chunk in reader.chunk_iterator(500_000):
            counts.update(
                np.asarray(chunk.classification).astype(int).tolist()
            )

    result = {
        "input_file": str(INPUT.relative_to(ROOT)),
        "total_points": int(point_count),
        "classification_counts": {
            str(code): int(count)
            for code, count in sorted(counts.items())
        },
        "interpretation": (
            "Classification counts describe the available point-cloud "
            "labels. They do not directly measure impervious surface area."
        ),
        "limitations": [
            "Ground-classified points can include paved and unpaved surfaces.",
            "Building points identify classified buildings, not all impervious surfaces.",
            "A validated surface-cover estimate requires suitable labels, "
            "rasterization or segmentation, and independent validation."
        ]
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("SURFACE CLASS INSPECTION")
    print("------------------------")
    print(f"Total points: {point_count:,}")

    for code, count in sorted(counts.items()):
        print(f"Class {code}: {count:,} points")

    print(f"\nReport saved to: {OUTPUT}")


if __name__ == "__main__":
    inspect_classes()
