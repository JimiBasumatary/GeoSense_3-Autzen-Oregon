
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs" / "reports" / "unified_site_profile.json"
OUTPUT = ROOT / "docs" / "exercise5" / "figures" / "Fig_08_unified_site_profile.png"

profile = json.loads(SOURCE.read_text(encoding="utf-8"))

lidar = profile["lidar"]
terrain = profile["terrain"]
street = profile["street_imagery"]["segmentation"]

dem = terrain["dem"]
slope = terrain["slope"]
ground = lidar["ground_point_cloud"]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
image = Image.new("RGB", (1200, 720), "#f4f7fb")
draw = ImageDraw.Draw(image)

try:
    title_font = ImageFont.truetype("arial.ttf", 34)
    heading_font = ImageFont.truetype("arial.ttf", 24)
    body_font = ImageFont.truetype("consola.ttf", 21)
except OSError:
    title_font = heading_font = body_font = ImageFont.load_default()

draw.text((45, 30), "GeoSense Agent — Unified Site Profile", fill="#183153", font=title_font)
draw.text((45, 85), profile["study_area"], fill="#444444", font=body_font)

draw.text((45, 155), "LiDAR", fill="#167447", font=heading_font)
draw.text((65, 200), f"Original points: {lidar['original_point_cloud']['point_count']:,}", fill="#222222", font=body_font)
draw.text((65, 240), f"Ground points: {ground['point_count']:,}", fill="#222222", font=body_font)

draw.text((45, 310), "Terrain", fill="#167447", font=heading_font)
draw.text((65, 355), f"DEM: {dem['width_pixels']} x {dem['height_pixels']} pixels", fill="#222222", font=body_font)
draw.text((65, 395), f"DEM mean elevation: {dem['mean']:.2f} ft", fill="#222222", font=body_font)
draw.text((65, 435), f"Mean slope: {slope['mean']:.4f} degrees", fill="#222222", font=body_font)

draw.text((45, 505), "Street-level imagery", fill="#167447", font=heading_font)
draw.text((65, 550), f"Images summarized: {street['image_count']}", fill="#222222", font=body_font)
draw.text((65, 590), f"Corrected overlays: {profile['street_imagery']['corrected_overlay_count']}", fill="#222222", font=body_font)
draw.text((45, 655), "Segmentation values are model predictions; imagery is not spatially matched to LiDAR.", fill="#555555", font=body_font)

image.save(OUTPUT)
print(f"Profile evidence created: {OUTPUT}")
print(f"Source profile: {SOURCE}")
