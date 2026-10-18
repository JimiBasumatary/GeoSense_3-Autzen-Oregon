
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "data" / "streetview" / "images"
OUTPUT_DIR = ROOT / "outputs" / "streetview"

metadata_path = IMAGE_DIR / "image_metadata.csv"
percentages_path = OUTPUT_DIR / "class_percentages.csv"

metadata = pd.read_csv(metadata_path)
percentages = pd.read_csv(percentages_path)

# Check that the two datasets refer to the same images.
data = metadata.merge(
    percentages,
    left_on="filename",
    right_on="image",
    how="inner",
    validate="one_to_one",
)

if data.empty:
    raise SystemExit("ERROR: No matching image filenames were found.")

required = ["sidewalk", "vegetation", "car", "road"]
missing = [column for column in required if column not in data.columns]
if missing:
    raise SystemExit(f"Missing segmentation columns: {missing}")

# Normalize each indicator from 0 to 100 within the matched image sample.
# This gives a relative comparison, not an absolute walkability rating.
def minmax(series):
    low = series.min()
    high = series.max()
    if high == low:
        return pd.Series(50.0, index=series.index)
    return 100.0 * (series - low) / (high - low)

data["sidewalk_norm"] = minmax(data["sidewalk"])
data["vegetation_norm"] = minmax(data["vegetation"])
data["car_norm"] = minmax(data["car"])

# Primary experimental weighting.
data["walkability_proxy"] = (
    0.50 * data["sidewalk_norm"]
    + 0.30 * data["vegetation_norm"]
    + 0.20 * (100.0 - data["car_norm"])
)

# Sensitivity check: alternative weighting.
data["walkability_alternative"] = (
    0.70 * data["sidewalk_norm"]
    + 0.20 * data["vegetation_norm"]
    + 0.10 * (100.0 - data["car_norm"])
)

data["primary_rank"] = (
    data["walkability_proxy"]
    .rank(ascending=False, method="min")
    .astype(int)
)
data["alternative_rank"] = (
    data["walkability_alternative"]
    .rank(ascending=False, method="min")
    .astype(int)
)

output_columns = [
    "filename",
    "longitude",
    "latitude",
    "road",
    "sidewalk",
    "building",
    "vegetation",
    "sky",
    "car",
    "walkability_proxy",
    "primary_rank",
    "walkability_alternative",
    "alternative_rank",
]

output_path = OUTPUT_DIR / "walkability_scores_24.csv"
data[output_columns].sort_values("primary_rank").to_csv(
    output_path, index=False
)

print("\nPRELIMINARY WALKABILITY PROXY")
print("Matched images:", len(data))
print("Scores are relative to this sample; they are not validated ratings.\n")
print(
    data[[
        "filename",
        "walkability_proxy",
        "primary_rank",
        "walkability_alternative",
        "alternative_rank",
    ]].sort_values("primary_rank").to_string(index=False)
)
print("\nSaved:", output_path)
