
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(r"D:\GeoSense_Phase3")

image_path = ROOT / "data" / "streetview" / "images" / "141300064660795.jpg"
output_dir = ROOT / "outputs" / "streetview" / "validation"

mask_path = output_dir / "141300064660795_manual_reference.png"
overlay_path = output_dir / "141300064660795_manual_reference_overlay.png"

if not image_path.exists():
    raise FileNotFoundError(f"Original image not found: {image_path}")

output_dir.mkdir(parents=True, exist_ok=True)

image = Image.open(image_path).convert("RGB")
width, height = image.size

# Class IDs follow the main GeoSense segmentation workflow:
# 0=road, 1=sidewalk, 2=building, 8=vegetation,
# 10=sky, 13=car, 255=unlabelled.
mask = Image.new("L", (width, height), 255)
draw = ImageDraw.Draw(mask)

# These are approximate draft polygons.
# They must be visually inspected and corrected before accuracy testing.

# Sky: upper part of the scene
draw.polygon(
    [(0, 0), (width - 1, 0), (width - 1, 130),
     (800, 145), (650, 125), (450, 150), (250, 125), (0, 145)],
    fill=10
)

# Vegetation: approximate tree and greenery regions
draw.polygon(
    [(0, 115), (180, 105), (310, 150), (300, 300),
     (190, 350), (0, 330)],
    fill=8
)
draw.polygon(
    [(700, 120), (850, 130), (width - 1, 150),
     (width - 1, 370), (850, 320), (760, 250)],
    fill=8
)

# Road: conservative estimate of the visible paved roadway.
# Leave uncertain edges unlabelled.
draw.polygon(
    [
        (285, 285),
        (390, 255),
        (520, 255),
        (655, 275),
        (770, 330),
        (850, 405),
        (220, 405),
        (250, 350),
    ],
    fill=0
)

# Exclude the visible car from the road reference.
# It remains unlabelled rather than being assumed to be road.
draw.polygon(
    [
        (405, 275),
        (435, 266),
        (505, 270),
        (535, 290),
        (542, 340),
        (520, 367),
        (435, 367),
        (407, 345),
    ],
    fill=255
)

# Leave the dashboard and windshield foreground unlabelled.
draw.polygon(
    [(0, 425), (1023, 425), (1023, 575), (0, 575)],
    fill=255
)
# Sidewalk: leave unlabelled unless the pedestrian path is clearly visible.
# Building: leave unlabelled unless a building boundary is clearly visible.
# Cars: leave unlabelled unless individual vehicles can be outlined reliably.

mask.save(mask_path)

# Create a coloured overlay using the project's class palette.
palette = {
    0: (128, 128, 128),   # road
    1: (255, 215, 0),     # sidewalk
    2: (255, 99, 71),     # building
    8: (34, 139, 34),     # vegetation
    10: (30, 144, 255),   # sky
    13: (186, 85, 211),   # car
}

mask_array = np.array(mask)
image_array = np.array(image)
colour_array = np.zeros_like(image_array)

for class_id, colour in palette.items():
    colour_array[mask_array == class_id] = colour

labelled = mask_array != 255
overlay_array = image_array.copy()
overlay_array[labelled] = (
    0.55 * image_array[labelled] +
    0.45 * colour_array[labelled]
).astype(np.uint8)

Image.fromarray(overlay_array).save(overlay_path)

print("DRAFT REFERENCE MASK CREATED")
print("Original image:", image_path)
print("Mask:", mask_path)
print("Overlay:", overlay_path)
print("Image dimensions:", image.size)
print("Labelled pixels:", int(labelled.sum()))
print("Unlabelled pixels:", int((~labelled).sum()))
print("Class IDs:", np.unique(mask_array).tolist())
print("WARNING: Draft polygons only; inspect and correct before accuracy evaluation.")
