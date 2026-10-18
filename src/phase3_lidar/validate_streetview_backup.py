
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
scores_path = OUTPUT_DIR / "walkability_scores.csv"

metadata = pd.read_csv(metadata_path)
percentages = pd.read_csv(percentages_path)
scores = pd.read_csv(scores_path)

print("\n=== SPATIAL AND DATA VALIDATION ===")
print("Metadata records:", len(metadata))
print("Unique filenames:", metadata["filename"].nunique())

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
    exists = path.exists()
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

# Verify filenames match across all stages.
meta_names = set(metadata["filename"].astype(str))
percent_names = set(percentages["image"].astype(str))
score_names = set(scores["filename"].astype(str))

print("Metadata/segmentation filename match:", meta_names == percent_names)
print("Metadata/scoring filename match:", meta_names == score_names)

# Haversine distances between all image locations.
def haversine_m(lon1, lat1, lon2, lat2):
    radius = 6371000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2)**2
    return 2 * radius * np.arcsin(np.sqrt(a))

valid = metadata.loc[valid_coords].copy()
distances = []

for i in range(len(valid)):
    for j in range(i + 1, len(valid)):
        a, b = valid.iloc[i], valid.iloc[j]
        distances.append({
            "image_1": a["filename"],
            "image_2": b["filename"],
            "distance_m": round(haversine_m(
                a["longitude"], a["latitude"],
                b["longitude"], b["latitude"]
            ), 1),
        })

pd.DataFrame(distances).to_csv(
    VALIDATION_DIR / "pairwise_image_distances.csv", index=False
)

if distances:
    print("\nPairwise distances saved.")
    print(pd.DataFrame(distances).to_string(index=False))

# Plot the image locations; this is a coordinate plot, not a basemap.
fig, ax = plt.subplots(figsize=(12, 7))

ax.scatter(
    valid["longitude"],
    valid["latitude"],
    s=100,
    edgecolors="black",
    zorder=3
)

label_settings = {
    "191372532836148.jpg": ("Image 1", (10, 10)),
    "927422231427319.jpg": ("Image 2", (-75, 25)),
    "175910187778974.jpg": ("Image 3", (10, 25)),
    "2011242559028632.jpg": ("Image 4", (10, -25)),
}

for _, row in valid.iterrows():
    label, offset = label_settings[row["filename"]]
    ax.annotate(
        label,
        (row["longitude"], row["latitude"]),
        xytext=offset,
        textcoords="offset points",
        fontsize=10,
        arrowprops=dict(arrowstyle="-", lw=0.7),
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
    )

ax.set_title("Mapillary Street-View Image Locations — Eugene, Oregon")
ax.set_xlabel("Longitude (decimal degrees)")
ax.set_ylabel("Latitude (decimal degrees)")
ax.grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
fig.savefig(
    VALIDATION_DIR / "image_location_plot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("Updated location plot saved.")
print("\nSaved validation files to:", VALIDATION_DIR)
print("Location plot: image_location_plot.png")
