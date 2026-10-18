
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import rasterio

ROOT = Path(r"D:\GeoSense_Phase3")
OUT_DIR = ROOT / "outputs" / "sam_georeferenced"

IMAGE_PATH = (
    ROOT / "data" / "aerial"
    / "autzen_naip_georeferenced_fixed.tif"
)
CSV_PATH = OUT_DIR / "autzen_sam_masks.csv"
MASK_DIR = OUT_DIR / "individual_masks"
OUTPUT_PATH = OUT_DIR / "largest_masks_contact_sheet.jpg"

if not CSV_PATH.exists():
    raise FileNotFoundError(CSV_PATH)

df = pd.read_csv(CSV_PATH)
df = df.sort_values("area_pixels", ascending=False).head(15)

# Read the same RGB stretch used for segmentation.
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

columns = 3
cell_width = 300
cell_height = 250
rows = (len(df) + columns - 1) // columns

sheet = Image.new(
    "RGB",
    (columns * cell_width, rows * cell_height),
    "white",
)
draw = ImageDraw.Draw(sheet)

for index, item in enumerate(df.itertuples(index=False)):
    mask_id = int(item.mask_id)
    mask_path = MASK_DIR / f"mask_{mask_id:04d}.png"

    if not mask_path.exists():
        print("Missing mask:", mask_path)
        continue

    x = int(item.bbox_x)
    y = int(item.bbox_y)
    w = int(item.bbox_width)
    h = int(item.bbox_height)

    crop = original.crop((x, y, x + w, y + h))
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    mask_crop = mask[y:y + h, x:x + w]

    image_array = np.array(crop).copy()

    # Mark the exterior boundary of the mask in red.
    padded = np.pad(mask_crop, 1)
    inside = mask_crop
    edge = (
        padded[1:-1, :-2]
        | padded[1:-1, 2:]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
    ) & ~inside

    image_array[edge] = [255, 0, 0]
    crop = Image.fromarray(image_array)
    crop.thumbnail((280, 190))

    col = index % columns
    row = index // columns
    left = col * cell_width
    top = row * cell_height

    sheet.paste(
        crop,
        (
            left + (cell_width - crop.width) // 2,
            top + 5,
        ),
    )

    draw.text(
        (left + 8, top + 200),
        f"Mask {mask_id} | {int(item.area_pixels)} pixels",
        fill="black",
    )
    draw.text(
        (left + 8, top + 218),
        f"IoU {item.predicted_iou:.2f} | "
        f"Stability {item.stability_score:.2f}",
        fill="black",
    )

sheet.save(OUTPUT_PATH, quality=95)

print("Contact sheet created.")
print("Masks displayed:", len(df))
print("Output:", OUTPUT_PATH)
