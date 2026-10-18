
from pathlib import Path
import csv
import json
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
)

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------
ROOT = Path(r"D:\GeoSense_Phase3")
IMAGE_PATH = ROOT / "data" / "aerial" / "autzen_naip_preview.jpg"
CHECKPOINT = ROOT / "models" / "sam_vit_b_01ec64.pth"

OUT_DIR = ROOT / "outputs" / "sam_rooftop"
MASK_DIR = OUT_DIR / "individual_masks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

OVERLAY_PATH = OUT_DIR / "autzen_sam_overlay.jpg"
SUMMARY_PATH = OUT_DIR / "autzen_sam_masks.csv"
METADATA_PATH = OUT_DIR / "autzen_sam_run_metadata.json"

# --------------------------------------------------
# 2. Check inputs
# --------------------------------------------------
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Aerial image not found: {IMAGE_PATH}")

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"SAM checkpoint not found: {CHECKPOINT}")

image = np.array(Image.open(IMAGE_PATH).convert("RGB"))

print("Image:", IMAGE_PATH)
print("Image dimensions:", image.shape[1], "x", image.shape[0])
print("Device: CPU")
print("Loading SAM ViT-B...")

# --------------------------------------------------
# 3. Load model
# --------------------------------------------------
start_load = time.perf_counter()

sam = sam_model_registry["vit_b"](
    checkpoint=str(CHECKPOINT)
)
sam.to(device="cpu")
sam.eval()

load_seconds = time.perf_counter() - start_load
print(f"Model loaded in {load_seconds:.2f} seconds.")

# --------------------------------------------------
# 4. Generate automatic masks
# --------------------------------------------------
# Moderate settings for a first CPU test.
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    points_per_batch=4,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    min_mask_region_area=100,
)

print("Generating masks. CPU processing may take several minutes...")
start_segmentation = time.perf_counter()

masks = mask_generator.generate(image)

segmentation_seconds = time.perf_counter() - start_segmentation
print("Mask generation finished.")
print("Number of masks:", len(masks))
print(f"Segmentation time: {segmentation_seconds:.2f} seconds.")

# --------------------------------------------------
# 5. Save a visual overlay
# --------------------------------------------------
overlay = Image.fromarray(image).convert("RGBA")
rng = np.random.default_rng(42)

# Draw larger masks first so smaller regions remain visible.
ordered_masks = sorted(
    masks, key=lambda item: item["area"], reverse=True
)

for index, item in enumerate(ordered_masks, start=1):
    seg = item["segmentation"]
    colour = tuple(
        int(value) for value in rng.integers(40, 256, size=3)
    )

    colour_layer = np.zeros(
        (image.shape[0], image.shape[1], 4), dtype=np.uint8
    )
    colour_layer[seg] = (*colour, 75)

    overlay = Image.alpha_composite(
        overlay, Image.fromarray(colour_layer, mode="RGBA")
    )

overlay.convert("RGB").save(OVERLAY_PATH, quality=95)

# --------------------------------------------------
# 6. Save individual binary masks and CSV summary
# --------------------------------------------------
csv_fields = [
    "mask_id",
    "area_pixels",
    "predicted_iou",
    "stability_score",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
]

with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=csv_fields)
    writer.writeheader()

    for mask_id, item in enumerate(masks, start=1):
        seg = item["segmentation"]
        binary = (seg.astype(np.uint8) * 255)

        Image.fromarray(binary, mode="L").save(
            MASK_DIR / f"mask_{mask_id:04d}.png"
        )

        x, y, width, height = [
            int(value) for value in item["bbox"]
        ]

        writer.writerow({
            "mask_id": mask_id,
            "area_pixels": int(item["area"]),
            "predicted_iou": round(
                float(item["predicted_iou"]), 5
            ),
            "stability_score": round(
                float(item["stability_score"]), 5
            ),
            "bbox_x": x,
            "bbox_y": y,
            "bbox_width": width,
            "bbox_height": height,
        })

# --------------------------------------------------
# 7. Save run metadata
# --------------------------------------------------
metadata = {
    "image": str(IMAGE_PATH),
    "image_width_pixels": int(image.shape[1]),
    "image_height_pixels": int(image.shape[0]),
    "model": "SAM ViT-B",
    "device": "CPU",
    "points_per_side": 16,
    "pred_iou_thresh": 0.86,
    "stability_score_thresh": 0.92,
    "minimum_mask_region_area_pixels": 100,
    "number_of_masks": len(masks),
    "model_load_seconds": round(load_seconds, 2),
    "segmentation_seconds": round(segmentation_seconds, 2),
    "overlay": str(OVERLAY_PATH),
    "mask_csv": str(SUMMARY_PATH),
    "individual_masks_directory": str(MASK_DIR),
    "warning": (
        "SAM masks are image regions, not verified rooftop labels. "
        "Do not interpret pixel areas as ground areas until image "
        "georeferencing and projected pixel dimensions are verified."
    ),
}

METADATA_PATH.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("\nSAM RUN COMPLETE")
print("Overlay:", OVERLAY_PATH)
print("CSV summary:", SUMMARY_PATH)
print("Individual masks:", MASK_DIR)
print("Run metadata:", METADATA_PATH)
