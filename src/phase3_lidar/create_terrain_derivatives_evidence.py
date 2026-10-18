from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "outputs" / "reports"
OUTPUT = ROOT / "docs" / "exercise5" / "figures" / "Fig_04_terrain_derivatives.png"

files = [
    ("HILLSHADE", REPORTS / "hillshade_metadata.txt"),
    ("SLOPE", REPORTS / "slope_metadata.txt"),
    ("ASPECT", REPORTS / "aspect_metadata.txt"),
]

for label, path in files:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} metadata: {path}")

width = 1500
height = 480
image = Image.new("RGB", (width, height), "#f4f7fb")
draw = ImageDraw.Draw(image)

try:
    title_font = ImageFont.truetype("arial.ttf", 30)
    heading_font = ImageFont.truetype("arial.ttf", 22)
    body_font = ImageFont.truetype("consola.ttf", 17)
except OSError:
    title_font = heading_font = body_font = ImageFont.load_default()

draw.text(
    (35, 20),
    "GeoSense Agent — Terrain Derivatives",
    fill="#183153",
    font=title_font,
)

columns = [35, 535, 1035]

for (label, path), x in zip(files, columns):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    draw.rectangle((x, 80, x + 430, 430), outline="#b8c7d9", width=2)
    draw.text((x + 15, 95), label, fill="#183153", font=heading_font)

    selected = []
    for line in lines:
        line = line.strip()
        if (
            line.startswith("Driver:")
            or line.startswith("Size is")
            or line.startswith("Origin =")
            or line.startswith("Pixel Size =")
            or "ID[\"EPSG\",2992]" in line
            or line.startswith("Min=")
            or line.startswith("NoData Value")
            or line.startswith("STATISTICS_MEAN=")
            or line.startswith("STATISTICS_VALID_PERCENT=")
        ):
            selected.append(line)

    y = 145
    for line in selected[:10]:
        draw.text((x + 15, y), line[:48], fill="#222222", font=body_font)
        y += 27

draw.text(
    (35, 445),
    "Evidence generated from the saved GDAL metadata files; values are not independently recalculated.",
    fill="#555555",
    font=body_font,
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
image.save(OUTPUT)

print(f"Terrain derivatives evidence created: {OUTPUT}")
print("Sources: hillshade_metadata.txt, slope_metadata.txt, aspect_metadata.txt")