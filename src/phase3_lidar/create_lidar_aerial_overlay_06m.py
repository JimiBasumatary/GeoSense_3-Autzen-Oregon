
from pathlib import Path

import laspy
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pyproj import Transformer

ROOT = Path(r"D:\GeoSense_Phase3")

LIDAR_PATH = (
    ROOT / "data" / "lidar" / "processed"
    / "autzen_classified_hag.laz"
)
IMAGE_PATH = (
    ROOT / "data" / "aerial"
    / "autzen_naip_06m_rgb.tif"
)
OUT_DIR = ROOT / "outputs" / "sam_georeferenced_06m"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "lidar_building_aerial_overlay_06m.png"

for path in [LIDAR_PATH, IMAGE_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

print("Reading aerial image...")
with rasterio.open(IMAGE_PATH) as src:
    image = src.read([1, 2, 3])
    image = np.moveaxis(image, 0, -1)
    transform = src.transform
    image_crs = src.crs
    bounds = src.bounds
    width, height = src.width, src.height

print("Reading LiDAR...")
las = laspy.read(LIDAR_PATH)
classes = np.asarray(las.classification)

# Use a deterministic sample so the plot stays readable.
building_indices = np.flatnonzero(classes == 6)
rng = np.random.default_rng(42)
if len(building_indices) > 150000:
    building_indices = rng.choice(
        building_indices, size=150000, replace=False
    )

x = np.asarray(las.x)[building_indices]
y = np.asarray(las.y)[building_indices]
hag = np.asarray(las["HeightAboveGround"])[building_indices]

source_crs = las.header.parse_crs()
if source_crs is None or image_crs is None:
    raise ValueError("CRS missing from LiDAR or aerial image.")

transformer = Transformer.from_crs(
    source_crs, image_crs, always_xy=True
)
xx, yy = transformer.transform(x, y)

valid = (
    np.isfinite(xx) & np.isfinite(yy)
    & (xx >= bounds.left) & (xx < bounds.right)
    & (yy >= bounds.bottom) & (yy < bounds.top)
)

xx, yy, hag = xx[valid], yy[valid], hag[valid]

# Stretch display values for a clearer aerial image.
display = image.astype(np.float32)
low, high = np.percentile(display, [2, 98])
if high > low:
    display = np.clip((display - low) / (high - low), 0, 1)

fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(
    display,
    extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
    origin="upper",
)

scatter = ax.scatter(
    xx, yy,
    c=hag,
    s=0.7,
    cmap="turbo",
    alpha=0.8,
    linewidths=0,
)

fig.colorbar(scatter, ax=ax, label="HAG (source units; expected feet)")
ax.set_title(
    "Autzen aerial image with LiDAR building-class points\n"
    "Visual alignment check — registration not independently verified"
)
ax.set_xlabel(f"X ({image_crs})")
ax.set_ylabel("Y")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=180)
plt.close(fig)

print("\nOVERLAY CREATED")
print("Building points plotted:", len(xx))
print("Image CRS:", image_crs)
print("Image bounds:", bounds)
print("Saved:", OUT_PATH)
print(
    "Caution: matching extent and visible overlap do not prove "
    "accurate image registration."
)
