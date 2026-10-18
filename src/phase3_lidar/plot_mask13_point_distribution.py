
from pathlib import Path

import laspy
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from pyproj import Transformer

ROOT = Path(r"D:\GeoSense_Phase3")

LIDAR_PATH = (
    ROOT / "data" / "lidar" / "processed"
    / "autzen_classified_hag.laz"
)
IMAGE_PATH = (
    ROOT / "data" / "aerial"
    / "autzen_naip_georeferenced_fixed.tif"
)
MASK_PATH = (
    ROOT / "outputs" / "sam_georeferenced"
    / "individual_masks" / "mask_0013.png"
)
OUTPUT_PATH = (
    ROOT / "outputs" / "sam_georeferenced"
    / "mask13_hag_point_distribution.png"
)

print("Reading aerial image...")
with rasterio.open(IMAGE_PATH) as src:
    data = src.read()[:3]
    transform = src.transform
    image_crs = src.crs
    width = src.width
    height = src.height

rgb = np.moveaxis(data, 0, -1).astype(np.float32)
rgb8 = np.zeros(rgb.shape, dtype=np.uint8)

for band in range(3):
    values = rgb[:, :, band]
    low, high = np.percentile(values, [2, 98])
    if high <= low:
        high = low + 1
    rgb8[:, :, band] = (
        np.clip((values - low) / (high - low), 0, 1) * 255
    ).astype(np.uint8)

image = Image.fromarray(rgb8, mode="RGB")
mask = np.asarray(Image.open(MASK_PATH).convert("L")) > 0

print("Reading LiDAR...")
las = laspy.read(LIDAR_PATH)

classification = np.asarray(las.classification)
building = classification == 6

x = np.asarray(las.x)[building]
y = np.asarray(las.y)[building]
hag = np.asarray(las.HeightAboveGround)[building]

source_crs = las.header.parse_crs()
transformer = Transformer.from_crs(
    source_crs, image_crs, always_xy=True
)
xx, yy = transformer.transform(x, y)

inverse = ~transform
cols = np.floor(
    inverse.a * xx + inverse.b * yy + inverse.c
).astype(int)
rows = np.floor(
    inverse.d * xx + inverse.e * yy + inverse.f
).astype(int)

inside = (
    (cols >= 0) & (cols < width)
    & (rows >= 0) & (rows < height)
)

cols = cols[inside]
rows = rows[inside]
hag = hag[inside]

in_mask = mask[rows, cols]

# Keep a 20-pixel context around the mask.
mask_rows, mask_cols = np.where(mask)
r_min = max(0, int(mask_rows.min()) - 20)
r_max = min(height, int(mask_rows.max()) + 21)
c_min = max(0, int(mask_cols.min()) - 20)
c_max = min(width, int(mask_cols.max()) + 21)

# Crop the aerial image and draw the mask boundary.
crop = image.crop((c_min, r_min, c_max, r_max))
crop_mask = mask[r_min:r_max, c_min:c_max]
crop_array = np.array(crop).copy()

padded = np.pad(crop_mask, 1)
edges = (
    padded[1:-1, :-2]
    | padded[1:-1, 2:]
    | padded[:-2, 1:-1]
    | padded[2:, 1:-1]
) & ~crop_mask

crop_array[edges] = [255, 255, 0]
canvas = Image.fromarray(crop_array, mode="RGB")
draw = ImageDraw.Draw(canvas)

# Draw a deterministic sample of points.
local = (
    (cols >= c_min) & (cols < c_max)
    & (rows >= r_min) & (rows < r_max)
    & np.isfinite(hag)
)

indices = np.flatnonzero(local)
rng = np.random.default_rng(13)
sample_size = min(15000, len(indices))
indices = rng.choice(indices, sample_size, replace=False)

for i in indices:
    px = int(cols[i] - c_min)
    py = int(rows[i] - r_min)
    h = float(hag[i])

    if in_mask[i]:
        # Cyan inside mask, orange outside.
        colour = (
            (255, 40, 40) if h < 15
            else (255, 180, 0) if h < 40
            else (0, 255, 255)
        )
    else:
        colour = (255, 255, 255)

    draw.point((px, py), fill=colour)

# Enlarge the diagnostic crop for easier visual inspection.
canvas.thumbnail((1200, 900))
canvas.save(OUTPUT_PATH)

print("Mask 13 point-distribution plot created.")
print("Sampled points:", sample_size)
print("Output:", OUTPUT_PATH)
print(
    "Legend: yellow = mask boundary; red = HAG below 15 ft; "
    "orange = HAG 15-40 ft; cyan = HAG above 40 ft; "
    "white = points outside the mask."
)
print(
    "Caution: This is a diagnostic overlay, not proof of "
    "accurate image registration or a verified roof footprint."
)
