from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "outputs" / "reports" / "exercise5_test_results.txt"
OUT = ROOT / "docs" / "exercise5" / "figures" / "Fig_10_pytest_results.png"

text = LOG.read_text(encoding="utf-8", errors="replace")
summary = next(
    (line.strip() for line in reversed(text.splitlines())
     if "passed" in line and "subtests" in line),
    None
)

if summary is None:
    raise ValueError("Could not find the final pytest summary in the log.")

OUT.parent.mkdir(parents=True, exist_ok=True)

image = Image.new("RGB", (1200, 360), "#f4f7fb")
draw = ImageDraw.Draw(image)

try:
    title_font = ImageFont.truetype("arial.ttf", 36)
    body_font = ImageFont.truetype("consola.ttf", 24)
except OSError:
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()

draw.text((45, 35), "GeoSense Agent — Phase III", fill="#183153", font=title_font)
draw.text((45, 100), "Exercise 5: Automated Test Results", fill="#183153", font=body_font)
draw.text((45, 175), summary, fill="#167447", font=body_font)
draw.text((45, 245), "Source: saved pytest log", fill="#555555", font=body_font)

image.save(OUT)
print(f"Screenshot evidence image created: {OUT}")
print(f"Test summary: {summary}")