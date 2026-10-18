
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs" / "reports" / "terrain_metrics.csv"
OUTPUT = ROOT / "docs" / "exercise5" / "figures" / "Fig_09_terrain_metrics.png"

with SOURCE.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.reader(f))

if not rows:
    raise ValueError("The terrain metrics CSV is empty.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

width = 1500
row_height = 38
visible_rows = rows[:15]
height = 130 + row_height * len(visible_rows)

image = Image.new("RGB", (width, height), "#f4f7fb")
draw = ImageDraw.Draw(image)

try:
    title_font = ImageFont.truetype("arial.ttf", 30)
    body_font = ImageFont.truetype("consola.ttf", 17)
except OSError:
    title_font = body_font = ImageFont.load_default()

draw.text((35, 20), "GeoSense Agent — Terrain Metrics CSV", fill="#183153", font=title_font)
draw.text((35, 65), f"Source: {SOURCE.relative_to(ROOT)}", fill="#555555", font=body_font)

y = 105
for row_index, row in enumerate(visible_rows):
    line = "  |  ".join(row)
    if row_index == 0:
        draw.rectangle((25, y - 3, width - 25, y + 30), fill="#d9eaf7")
    draw.text((35, y), line[:160], fill="#183153" if row_index == 0 else "#222222", font=body_font)
    y += row_height

draw.text((35, height - 25), f"Preview: {len(visible_rows)} of {len(rows)} CSV rows", fill="#555555", font=body_font)
image.save(OUTPUT)

print(f"Terrain CSV evidence created: {OUTPUT}")
print(f"CSV rows including header: {len(rows)}")
print(f"Preview rows: {len(visible_rows)}")
