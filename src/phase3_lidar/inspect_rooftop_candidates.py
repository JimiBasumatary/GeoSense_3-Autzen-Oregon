
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw

ROOT = Path(r"D:\GeoSense_Phase3")
OUT_DIR = ROOT / "outputs" / "sam_georeferenced"

IMAGE_PATH = (
    ROOT / "data" / "aerial"
    / "autzen_naip_georeferenced_fixed.tif"
)
CSV_PATH = OUT_DIR / "autzen_sam_masks.csv"
MASK_DIR = OUT_DIR / "individual_masks"
OUTPUT_PATH = OUT_DIR / "rooftop_candidates_detail.jpg"

CANDIDATE_IDS = [36, 61, 32]

df = pd.read_csv(CSV_PATH)
df = df[df["mask_id"].isin(CANDIDATE_IDS)]

with rasterio.open(IMAGE_PATH) as src:
    data = src.read()[:3]
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

original = Image.fromarray(rgb8, mode="RGB")

panel_width = 500
panel_height = 420
sheet = Image.new("RGB", (panel_width, panel_height * len(df)), "white")
draw = ImageDraw.Draw(sheet)

for panel, item in enumerate(df.itertuples(index=False)):
    mask_id = int(item.mask_id)
    mask_path = MASK_DIR / f"mask_{mask_id:04d}.png"

    if not mask_path.exists():
        print("Missing mask:", mask_path)
        continue

    x = int(item.bbox_x)
    y = int(item.bbox_y)
    w = int(item.bbox_width)
    h = int(item.bbox_height)

    # Add context around each mask.
    pad = 30
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(rgb8.shape[1], x + w + pad)
    bottom = min(rgb8.shape[0], y + h + pad)

    crop = original.crop((left, top, right, bottom))
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    crop_mask = mask[top:bottom, left:right]

    image_array = np.array(crop).copy()
    padded = np.pad(crop_mask, 1)
    edges = (
        padded[1:-1, :-2]
        | padded[1:-1, 2:]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
    ) & ~crop_mask

    image_array[edges] = [255, 0, 0]
    crop = Image.fromarray(image_array)
    crop.thumbnail((470, 350))

    y_offset = panel * panel_height
    sheet.paste(
        crop,
        (
            (panel_width - crop.width) // 2,
            y_offset + 5,
        ),
    )

    draw.text(
        (10, y_offset + 365),
        f"Mask {mask_id} | Area: {int(item.area_pixels)} pixels",
        fill="black",
    )
    draw.text(
        (10, y_offset + 385),
        f"IoU: {item.predicted_iou:.3f} | "
        f"Stability: {item.stability_score:.3f}",
        fill="black",
    )

sheet.save(OUTPUT_PATH, quality=95)

print("Candidate detail sheet created.")
print("Masks inspected:", len(df))
print("Output:", OUTPUT_PATH)
