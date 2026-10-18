
from pathlib import Path
import csv
import json
import time

import numpy as np
import rasterio
from PIL import Image
from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
)

ROOT = Path(r"D:\GeoSense_Phase3")
IMAGE_PATH = (
    ROOT / "data" / "aerial"
    / "autzen_naip_06m_rgb.tif"
)
CHECKPOINT = ROOT / "models" / "sam_vit_b_01ec64.pth"

OUT_DIR = ROOT / "outputs" / "sam_georeferenced_06m"
MASK_DIR = OUT_DIR / "individual_masks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

OVERLAY_PATH = OUT_DIR / "autzen_sam_overlay.jpg"
CSV_PATH = OUT_DIR / "autzen_sam_masks.csv"
METADATA_PATH = OUT_DIR / "autzen_sam_metadata.json"

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

# Read image bands and spatial metadata.
with rasterio.open(IMAGE_PATH) as src:
    raster = src.read()
    raster_crs = str(src.crs)
    raster_bounds = list(src.bounds)
    raster_transform = list(src.transform)
    width = src.width
    height = src.height

if raster.shape[0] < 3:
    raise ValueError("At least three image bands are required.")

# Convert the first three bands to RGB.
rgb = np.moveaxis(raster[:3], 0, -1).astype(np.float32)

# Scale each band to 8-bit for SAM. This is a visual stretch;
# the original GeoTIFF remains unchanged.
rgb8 = np.zeros(rgb.shape, dtype=np.uint8)
for band in range(3):
    values = rgb[:, :, band]
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        raise ValueError(f"Band {band + 1} contains no valid values.")

    low, high = np.percentile(valid, [2, 98])
    if high <= low:
        high = low + 1

    stretched = np.clip((values - low) / (high - low), 0, 1)
    rgb8[:, :, band] = (stretched * 255).astype(np.uint8)

print("Input:", IMAGE_PATH)
print("Dimensions:", width, "x", height)
print("CRS:", raster_crs)
print("Loading SAM ViT-B on CPU...")

sam = sam_model_registry["vit_b"](
    checkpoint=str(CHECKPOINT)
)
sam.to(device="cpu")
sam.eval()

generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    points_per_batch=4,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    min_mask_region_area=100,
)

print("Generating masks...")
start = time.perf_counter()
masks = generator.generate(rgb8)
elapsed = time.perf_counter() - start

print("Masks generated:", len(masks))
print("Segmentation seconds:", round(elapsed, 2))

# Save each mask in the raster's pixel grid.
records = []
for mask_id, item in enumerate(masks, start=1):
    seg = item["segmentation"]
    mask_path = MASK_DIR / f"mask_{mask_id:04d}.png"
    Image.fromarray(seg.astype(np.uint8) * 255).save(mask_path)

    x, y, w, h = [int(v) for v in item["bbox"]]
    records.append({
        "mask_id": mask_id,
        "area_pixels": int(item["area"]),
        "predicted_iou": float(item["predicted_iou"]),
        "stability_score": float(item["stability_score"]),
        "bbox_x": x,
        "bbox_y": y,
        "bbox_width": w,
        "bbox_height": h,
        "mask_file": str(mask_path),
    })

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)

# Create an overlay for visual inspection.
overlay = Image.fromarray(rgb8).convert("RGBA")
rng = np.random.default_rng(42)

for item in sorted(masks, key=lambda m: m["area"], reverse=True):
    colour = tuple(int(v) for v in rng.integers(40, 256, 3))
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    layer[item["segmentation"]] = (*colour, 75)
    overlay = Image.alpha_composite(
        overlay, Image.fromarray(layer, mode="RGBA")
    )

overlay.convert("RGB").save(OVERLAY_PATH, quality=95)

metadata = {
    "image": str(IMAGE_PATH),
    "width": width,
    "height": height,
    "crs": raster_crs,
    "bounds": raster_bounds,
    "affine_transform": raster_transform,
    "model": "SAM ViT-B",
    "device": "CPU",
    "points_per_side": 16,
    "number_of_masks": len(masks),
    "segmentation_seconds": round(elapsed, 2),
    "overlay": str(OVERLAY_PATH),
    "csv": str(CSV_PATH),
    "mask_directory": str(MASK_DIR),
    "warning": (
        "Masks are image segments, not verified rooftops. "
        "Spatial metadata describes the raster extent supplied by "
        "the image service; independent image-to-LiDAR registration "
        "has not yet been validated."
    ),
}

METADATA_PATH.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("\nRUN COMPLETE")
print("Overlay:", OVERLAY_PATH)
print("CSV:", CSV_PATH)
print("Masks:", MASK_DIR)
print("Metadata:", METADATA_PATH)
