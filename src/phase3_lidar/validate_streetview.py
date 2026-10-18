
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "data" / "streetview" / "images"
OUTPUT_DIR = ROOT / "outputs" / "streetview"
VALIDATION_DIR = OUTPUT_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

metadata_path = IMAGE_DIR / "image_metadata.csv"
percentages_path = OUTPUT_DIR / "class_percentages.csv"
scores_path = OUTPUT_DIR / "walkability_scores_24.csv"

metadata = pd.read_csv(metadata_path)
percentages = pd.read_csv(percentages_path)
scores = pd.read_csv(scores_path)

print("\n=== SPATIAL AND DATA VALIDATION ===")
print("Metadata records:", len(metadata))
print("Unique filenames:", metadata["filename"].nunique())
print("Segmentation records:", len(percentages))
print("Walkability score records:", len(scores))

# Check required coordinate fields.
metadata["longitude"] = pd.to_numeric(
    metadata["longitude"], errors="coerce"
)
metadata["latitude"] = pd.to_numeric(
    metadata["latitude"], errors="coerce"
)

valid_coords = (
    metadata["longitude"].between(-180, 180)
    & metadata["latitude"].between(-90, 90)
)
print("Valid coordinate records:", int(valid_coords.sum()))

# Check image files and dimensions.
image_checks = []

for _, row in metadata.iterrows():
    path = IMAGE_DIR / str(row["filename"])
    exists = path.is_file()
    readable = False
    width = height = None

    if exists:
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                width, height = im.size
            readable = True
        except Exception:
            pass

    image_checks.append({
        "filename": row["filename"],
        "file_exists": exists,
        "image_readable": readable,
        "width": width,
        "height": height,
        "longitude": row["longitude"],
        "latitude": row["latitude"],
    })

checks = pd.DataFrame(image_checks)
checks.to_csv(
    VALIDATION_DIR / "image_file_checks.csv", index=False
)

print("Files present:", int(checks["file_exists"].sum()))
print("Images readable:", int(checks["image_readable"].sum()))

# Verify filenames match across all processing stages.
meta_names = set(metadata["filename"].dropna().astype(str))
percent_names = set(percentages["image"].dropna().astype(str))
score_names = set(scores["filename"].dropna().astype(str))

print(
    "Metadata/segmentation filename match:",
    meta_names == percent_names
)
print(
    "Metadata/scoring filename match:",
    meta_names == score_names
)

# Check for duplicate filenames.
print("Duplicate metadata filenames:", int(metadata["filename"].duplicated().sum()))
print("Duplicate segmentation filenames:", int(percentages["image"].duplicated().sum()))
print("Duplicate score filenames:", int(scores["filename"].duplicated().sum()))

# Haversine distances between all valid image locations.
def haversine_m(lon1, lat1, lon2, lat2):
    radius = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = (
        np.sin(dp / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return 2 * radius * np.arcsin(np.sqrt(a))

valid = metadata.loc[valid_coords].copy()
distances = []

for i in range(len(valid)):
    for j in range(i + 1, len(valid)):
        a, b = valid.iloc[i], valid.iloc[j]
        distances.append({
            "image_1": a["filename"],
            "image_2": b["filename"],
            "distance_m": round(
                haversine_m(
                    a["longitude"], a["latitude"],
                    b["longitude"], b["latitude"]
                ),
                1,
            ),
        })

distance_path = VALIDATION_DIR / "pairwise_image_distances.csv"
distance_df = pd.DataFrame(
    distances,
    columns=["image_1", "image_2", "distance_m"]
)
distance_df.to_csv(distance_path, index=False)

print("\nPairwise distance rows saved:", len(distance_df))
if not distance_df.empty:
    print("Minimum distance (m):", distance_df["distance_m"].min())
    print("Maximum distance (m):", distance_df["distance_m"].max())

# Plot all valid locations, labelled with short index numbers.
fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(
    valid["longitude"],
    valid["latitude"],
    s=65,
    edgecolors="black",
    zorder=3,
)

for number, (_, row) in enumerate(valid.iterrows(), start=1):
    ax.annotate(
        str(number),
        (row["longitude"], row["latitude"]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )

ax.set_title("Mapillary Street-View Image Locations — Eugene, Oregon")
ax.set_xlabel("Longitude (decimal degrees)")
ax.set_ylabel("Latitude (decimal degrees)")
ax.grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
plot_path = VALIDATION_DIR / "image_location_plot.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# Save the plotted number-to-filename key.
location_key = valid[["filename", "longitude", "latitude"]].copy()
location_key.insert(
    0, "plot_label", range(1, len(location_key) + 1)
)
location_key.to_csv(
    VALIDATION_DIR / "image_location_plot_key.csv",
    index=False,
)

print("\nUpdated location plot saved:", plot_path)
print("Location label key saved:", VALIDATION_DIR / "image_location_plot_key.csv")
print("Validation files saved to:", VALIDATION_DIR)
