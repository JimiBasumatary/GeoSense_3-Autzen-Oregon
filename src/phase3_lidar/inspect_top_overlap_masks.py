
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
CSV_PATH = OUT_DIR / "all_sam_masks_lidar_overlap.csv"
MASK_DIR = OUT_DIR / "individual_masks"
OUTPUT_PATH = OUT_DIR / "top_overlap_masks_contact_sheet.jpg"

df = pd.read_csv(CSV_PATH)
df = df.sort_values(
    "building_points_inside_mask", ascending=False
)

# Select the seven masks with the highest overlap counts.
df = df.head(7)

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

panel_width = 600
panel_height = 390
sheet = Image.new(
    "RGB", (panel_width, panel_height * len(df)), "white"
)
draw = ImageDraw.Draw(sheet)

for panel, item in enumerate(df.itertuples(index=False)):
    mask_id = int(item.mask_id)
    mask_path = MASK_DIR / f"mask_{mask_id:04d}.png"
    y_offset = panel * panel_height

    if not mask_path.exists():
        draw.text(
            (10, y_offset + 10),
            f"Mask {mask_id}: mask file missing",
            fill="black",
        )
        continue

    mask = np.asarray(
        Image.open(mask_path).convert("L")
    ) > 0

    # Find the mask's actual extent.
    ys, xs = np.where(mask)
    if len(xs) == 0:
        continue

    pad = 35
    left = max(0, int(xs.min()) - pad)
    top = max(0, int(ys.min()) - pad)
    right = min(rgb8.shape[1], int(xs.max()) + pad + 1)
    bottom = min(rgb8.shape[0], int(ys.max()) + pad + 1)

    crop = original.crop((left, top, right, bottom))
    crop_mask = mask[top:bottom, left:right]

    # Outline the SAM mask in red.
    array = np.array(crop).copy()
    padded = np.pad(crop_mask, 1)

    edges = (
        padded[1:-1, :-2]
        | padded[1:-1, 2:]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
    ) & ~crop_mask

    array[edges] = [255, 0, 0]
    crop = Image.fromarray(array)
    crop.thumbnail((570, 310))

    sheet.paste(
        crop,
        ((panel_width - crop.width) // 2, y_offset + 5),
    )

    hag_median = getattr(
        item, "hag_median_source_units", float("nan")
    )
    hag_text = (
        f"{hag_median:.2f} ft"
        if pd.notna(hag_median)
        else "unavailable"
    )

    draw.text(
        (10, y_offset + 325),
        f"Mask {mask_id} | Area: {int(item.mask_area_pixels)} px",
        fill="black",
    )
    draw.text(
        (10, y_offset + 345),
        f"Building points: {int(item.building_points_inside_mask)}"
        f" | Median HAG: {hag_text}",
        fill="black",
    )

sheet.save(OUTPUT_PATH, quality=95)

print("Top-overlap contact sheet created.")
print("Masks included:", len(df))
print("Output:", OUTPUT_PATH)
print(
    "Reminder: red outlines show SAM masks, not verified roof "
    "boundaries. Georeferencing remains unverified."
)
