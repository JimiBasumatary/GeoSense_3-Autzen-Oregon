
from pathlib import Path

import laspy
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from pyproj import Transformer

ROOT = Path(r"D:\GeoSense_Phase3")

LIDAR_PATH = (
    ROOT / "data" / "lidar" / "raw"
    / "autzen-classified.copc.laz"
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
    / "mask13_ground_distribution.png"
)

print("Reading image and mask...")
with rasterio.open(IMAGE_PATH) as src:
    data = src.read()[:3]
    image_crs = src.crs
    transform = src.transform
    width, height = src.width, src.height

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
mask_rows, mask_cols = np.where(mask)

if len(mask_rows) == 0:
    raise ValueError("Mask 13 is empty.")

pad = 30
r_min = max(0, int(mask_rows.min()) - pad)
r_max = min(height, int(mask_rows.max()) + pad + 1)
c_min = max(0, int(mask_cols.min()) - pad)
c_max = min(width, int(mask_cols.max()) + pad + 1)

print("Reading LiDAR...")
las = laspy.read(LIDAR_PATH)
classes = np.asarray(las.classification)

ground = classes == 2
building = classes == 6

ground_xy = np.column_stack((
    np.asarray(las.x)[ground],
    np.asarray(las.y)[ground],
))
building_xy = np.column_stack((
    np.asarray(las.x)[building],
    np.asarray(las.y)[building],
))

transformer = Transformer.from_crs(
    las.header.parse_crs(), image_crs, always_xy=True
)

def to_pixels(points):
    xx, yy = transformer.transform(points[:, 0], points[:, 1])
    inverse = ~transform
    cols = np.floor(
        inverse.a * xx + inverse.b * yy + inverse.c
    ).astype(int)
    rows = np.floor(
        inverse.d * xx + inverse.e * yy + inverse.f
    ).astype(int)
    return cols, rows

ground_cols, ground_rows = to_pixels(ground_xy)
building_cols, building_rows = to_pixels(building_xy)

ground_inside = (
    (ground_cols >= c_min) & (ground_cols < c_max)
    & (ground_rows >= r_min) & (ground_rows < r_max)
)
building_inside = (
    (building_cols >= c_min) & (building_cols < c_max)
    & (building_rows >= r_min) & (building_rows < r_max)
)

ground_indices = np.flatnonzero(ground_inside)
building_indices = np.flatnonzero(building_inside)

print("Ground points in crop:", len(ground_indices))
print("Building points in crop:", len(building_indices))

# Draw the mask boundary in yellow.
crop = image.crop((c_min, r_min, c_max, r_max))
crop_mask = mask[r_min:r_max, c_min:c_max]
crop_array = np.array(crop).copy()

padded_mask = np.pad(crop_mask, 1)
edges = (
    padded_mask[1:-1, :-2]
    | padded_mask[1:-1, 2:]
    | padded_mask[:-2, 1:-1]
    | padded_mask[2:, 1:-1]
) & ~crop_mask
crop_array[edges] = [255, 255, 0]

canvas = Image.fromarray(crop_array, mode="RGB")
draw = ImageDraw.Draw(canvas)

rng = np.random.default_rng(13)

# Draw up to 15,000 ground points in blue.
n_ground = min(15000, len(ground_indices))
chosen_ground = rng.choice(
    ground_indices, n_ground, replace=False
)
for idx in chosen_ground:
    px = int(ground_cols[idx] - c_min)
    py = int(ground_rows[idx] - r_min)
    draw.point((px, py), fill=(0, 100, 255))

# Draw up to 10,000 building points in red.
n_building = min(10000, len(building_indices))
chosen_building = rng.choice(
    building_indices, n_building, replace=False
)
for idx in chosen_building:
    px = int(building_cols[idx] - c_min)
    py = int(building_rows[idx] - r_min)
    draw.point((px, py), fill=(255, 40, 40))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUTPUT_PATH)

print("Ground distribution image saved:", OUTPUT_PATH)
print("Legend: blue = class 2 ground; red = class 6 buildings;")
print("yellow = Mask 13 boundary.")
print(
    "Caution: Image registration is still unverified; this plot "
    "shows point locations, not proof of HAG accuracy."
)
