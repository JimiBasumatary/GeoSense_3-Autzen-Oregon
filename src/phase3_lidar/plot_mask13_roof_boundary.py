
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image
from rasterio.plot import plotting_extent

ROOT = Path(r"D:\GeoSense_Phase3")

IMAGE_PATH = ROOT / "data" / "aerial" / "autzen_naip_georeferenced_fixed.tif"
MASK_PATH = ROOT / "outputs" / "sam_georeferenced" / "individual_masks" / "mask_0013.png"
OUT_PATH = ROOT / "outputs" / "sam_georeferenced" / "mask13_roof_boundary_overlay.png"

for path in [IMAGE_PATH, MASK_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

with rasterio.open(IMAGE_PATH) as src:
    image = np.moveaxis(src.read([1, 2, 3]), 0, -1)
    extent = plotting_extent(src)
    image_crs = src.crs
    width, height = src.width, src.height

mask = np.asarray(Image.open(MASK_PATH).convert("L")) > 0

if mask.shape != (height, width):
    raise ValueError(
        f"Mask shape {mask.shape} does not match image {(height, width)}"
    )

display = image.astype(np.float32)
low, high = np.percentile(display, [2, 98])
if high > low:
    display = np.clip((display - low) / (high - low), 0, 1)

fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(display, extent=extent, origin="upper")

# Display the segmentation region with transparency.
overlay = np.zeros((height, width, 4), dtype=float)
overlay[mask] = [1.0, 1.0, 0.0, 0.22]
ax.imshow(overlay, extent=extent, origin="upper")

# Draw the mask perimeter.
ax.contour(
    mask.astype(np.uint8),
    levels=[0.5],
    colors=["red"],
    linewidths=1.5,
    origin="upper",
    extent=extent,
)

ax.set_title(
    "SAM Mask 13 boundary on aerial image\n"
    f"CRS: {image_crs} — registration not independently verified"
)
ax.set_xlabel("Easting")
ax.set_ylabel("Northing")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=180)
plt.close(fig)

print("MASK 13 BOUNDARY OVERLAY CREATED")
print(f"Mask pixels: {int(mask.sum()):,}")
print(f"Image dimensions: {width} x {height}")
print(f"Image CRS: {image_crs}")
print(f"Saved: {OUT_PATH}")
print("Visual inspection is required; this does not prove registration.")
