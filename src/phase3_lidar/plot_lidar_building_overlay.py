
from pathlib import Path

import laspy
import numpy as np
import rasterio
from pyproj import Transformer
from PIL import Image, ImageDraw

ROOT = Path(r"D:\GeoSense_Phase3")

LIDAR_PATH = (
    ROOT / "data" / "lidar" / "raw"
    / "autzen-classified.copc.laz"
)
IMAGE_PATH = (
    ROOT / "data" / "aerial"
    / "autzen_naip_georeferenced_fixed.tif"
)
OUTPUT_PATH = (
    ROOT / "outputs" / "sam_georeferenced"
    / "lidar_building_points_overlay.png"
)

print("Reading LiDAR...")
las = laspy.read(LIDAR_PATH)
classification = np.asarray(las.classification)
is_building = classification == 6

x = np.asarray(las.x)[is_building]
y = np.asarray(las.y)[is_building]

print("Building-class points:", len(x))

with rasterio.open(IMAGE_PATH) as src:
    raster = src.read()[:3]
    image_crs = src.crs
    transform = src.transform
    width = src.width
    height = src.height

rgb = np.moveaxis(raster, 0, -1).astype(np.float32)
rgb8 = np.zeros(rgb.shape, dtype=np.uint8)

# Use the same percentile stretch as the SAM input.
for band in range(3):
    values = rgb[:, :, band]
    low, high = np.percentile(values, [2, 98])
    if high <= low:
        high = low + 1
    rgb8[:, :, band] = (
        np.clip((values - low) / (high - low), 0, 1) * 255
    ).astype(np.uint8)

# Transform LiDAR coordinates to the raster CRS.
source_crs = las.header.parse_crs()
if source_crs is None or image_crs is None:
    raise ValueError("Could not identify LiDAR or image CRS.")

transformer = Transformer.from_crs(
    source_crs, image_crs, always_xy=True
)
xx, yy = transformer.transform(x, y)

# Map coordinates to pixel coordinates.
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

rows = rows[inside]
cols = cols[inside]

print("Building points inside image:", len(rows))

# Draw a deterministic sample so the image remains readable.
rng = np.random.default_rng(42)
sample_size = min(50000, len(rows))
chosen = rng.choice(len(rows), size=sample_size, replace=False)

overlay = Image.fromarray(rgb8, mode="RGB")
draw = ImageDraw.Draw(overlay)

# Mark point locations with small red dots.
for index in chosen:
    draw.point((int(cols[index]), int(rows[index])), fill=(255, 0, 0))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
overlay.save(OUTPUT_PATH)

print("Overlay saved:", OUTPUT_PATH)
print("Sampled points drawn:", sample_size)
print("Raster CRS:", image_crs)
print("Raster transform:", transform)
print(
    "Warning: This is a preliminary visual alignment check. "
    "The aerial image's georeferencing has not been independently "
    "validated against ground control or image features."
)
