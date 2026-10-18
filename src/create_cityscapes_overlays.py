
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

INPUT_DIR = "data/street_imagery/raw"
OUTPUT_DIR = "outputs/street_imagery"
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "overlays")

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)

# Standard Cityscapes colours for the 19 model classes
palette = np.array([
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32],    # bicycle
], dtype=np.uint8)

summary_path = os.path.join(OUTPUT_DIR, "segmentation_summary.csv")
summary = pd.read_csv(summary_path)

for image_path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg"))):
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(MASK_DIR, f"{image_id}_mask.png")

    if not os.path.exists(mask_path):
        print("Missing mask:", image_id)
        continue

    image = Image.open(image_path).convert("RGB")
    mask = np.array(Image.open(mask_path).convert("L"))

    if mask.shape != (image.height, image.width):
        print("Dimension mismatch:", image_id)
        continue

    if mask.max() >= len(palette):
        print("Unexpected class ID:", image_id, mask.max())
        continue

    colour_mask = Image.fromarray(palette[mask])
    overlay = Image.blend(image, colour_mask, alpha=0.45)

    overlay.save(os.path.join(
        OVERLAY_DIR, f"{image_id}_overlay_cityscapes.png"
    ))
    print("Created consistent-colour overlay:", image_id)

print("Finished. Original outputs have been preserved.")
