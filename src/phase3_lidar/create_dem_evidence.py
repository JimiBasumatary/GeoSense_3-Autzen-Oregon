from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs" / "reports" / "dem_metadata.txt"
OUTPUT = ROOT / "docs" / "exercise5" / "figures" / "Fig_03_dem_metadata.png"

if not SOURCE.exists():
    raise FileNotFoundError(f"DEM metadata not found: {SOURCE}")

text = SOURCE.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()

keywords = (
    "Driver:",
    "Size is",
    "Origin =",
    "Pixel Size =",
    "ID[\"EPSG\",2992]",
    "ID[\"EPSG\",6360]",
    "Min=",
    "Max=",
    "NoData Value",
    "STATISTICS_MEAN=",
    "STATISTICS_STDDEV=",
    "STATISTICS_VALID_PERCENT="
)

selected = []
for line in lines:
    if any(key in line for key in keywords):
        selected.append(line.strip())

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

width = 1500
line_height = 34
height = 150 + line_height * max(len(selected), 1)

image = Image.new("RGB", (width, height), "#f4f7fb")
draw = ImageDraw.Draw(image)

try:
    title_font = ImageFont.truetype("arial.ttf", 30)
    body_font = ImageFont.truetype("consola.ttf", 19)
except OSError:
    title_font = body_font = ImageFont.load_default()

draw.text(
    (35, 25),
    "GeoSense Agent — DEM Metadata",
    fill="#183153",
    font=title_font
)
draw.text(
    (35, 75),
    "Source: outputs/reports/dem_metadata.txt",
    fill="#555555",
    font=body_font
)

y = 125
for line in selected:
    draw.text((40, y), line[:140], fill="#222222", font=body_font)
    y += line_height

image.save(OUTPUT)

print(f"DEM evidence created: {OUTPUT}")
print(f"Metadata lines included: {len(selected)}")