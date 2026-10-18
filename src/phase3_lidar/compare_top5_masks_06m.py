from pathlib import Path
import csv
import numpy as np
import rasterio
from PIL import Image, ImageDraw

root = Path(r"D:\GeoSense_Phase3")
folder = root / "outputs" / "sam_georeferenced_06m"
image_path = root / "data" / "aerial" / "autzen_naip_06m_rgb.tif"

mask_ids = [53, 22, 21, 18, 43]

with rasterio.open(image_path) as src:
    image = np.moveaxis(src.read([1, 2, 3]), 0, -1).astype(np.float32)

low, high = np.percentile(image, [2, 98])
if high <= low:
    high = low + 1

rgb = np.clip((image - low) / (high - low) * 255, 0, 255).astype(np.uint8)
base = Image.fromarray(rgb)

csv_path = folder / "all_sam_masks_lidar_overlap.csv"
with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
    records = {int(row["mask_id"]): row for row in csv.DictReader(f)}

panel_w, panel_h = 520, 420
header_h = 60
panels = []

for mask_id in mask_ids:
    mask_path = folder / "individual_masks" / f"mask_{mask_id:04d}.png"
    mask = Image.open(mask_path).convert("L")
    mask_array = np.asarray(mask) > 0
    ys, xs = np.where(mask_array)

    if len(xs) == 0:
        print(f"Skipping empty mask {mask_id}")
        continue

    padding = 35
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(base.width, int(xs.max()) + padding + 1)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(base.height, int(ys.max()) + padding + 1)

    crop = base.crop((x0, y0, x1, y1)).resize((panel_w, panel_h))
    mask_crop = mask.crop((x0, y0, x1, y1)).resize(
        (panel_w, panel_h), Image.Resampling.NEAREST
    )
    region = np.asarray(mask_crop) > 0

    overlay = np.asarray(crop).copy()
    overlay[region] = (
        0.55 * overlay[region] + 0.45 * np.array([255, 40, 40])
    ).astype(np.uint8)

    panel = Image.new("RGB", (panel_w, panel_h + header_h), "white")
    panel.paste(Image.fromarray(overlay), (0, header_h))

    draw = ImageDraw.Draw(panel)
    row = records[mask_id]
    draw.text(
        (8, 8),
        f"Mask {mask_id} | Building points: "
        f"{row['building_points_inside_mask']}",
        fill="black",
    )
    draw.text(
        (8, 30),
        f"Mask area: {row['mask_area_pixels']} pixels | Red = mask",
        fill="black",
    )
    panels.append(panel)

sheet = Image.new(
    "RGB",
    (panel_w * 2, (panel_h + header_h) * 3),
    (235, 235, 235),
)

for index, panel in enumerate(panels):
    x = (index % 2) * panel_w
    y = (index // 2) * (panel_h + header_h)
    sheet.paste(panel, (x, y))

output_path = folder / "top5_candidate_mask_comparison.jpg"
sheet.save(output_path, quality=92)

print("Saved:", output_path)
print("Candidates shown:", len(panels))
print("Image size:", sheet.size)