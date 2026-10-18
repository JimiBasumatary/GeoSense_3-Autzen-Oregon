
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

INPUT_DIR = "data/street_imagery/raw"
OUTPUT_DIR = "outputs/street_imagery"
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "overlays")

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)

MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Loading model:", MODEL_NAME)

processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))

if not image_paths:
    raise FileNotFoundError(f"No JPG images found in {INPUT_DIR}")

# Cityscapes-style colour palette
rng = np.random.default_rng(42)
num_classes = model.config.num_labels
palette = rng.integers(40, 256, size=(num_classes, 3), dtype=np.uint8)
palette[0] = [128, 64, 128]  # road, for the standard Cityscapes label map
palette[1] = [244, 35, 232]  # sidewalk, for the standard Cityscapes label map

summary_rows = []

for image_path in image_paths:
    filename = os.path.basename(image_path)
    image_id = os.path.splitext(filename)[0]

    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            logits = F.interpolate(
                outputs.logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        mask_rgb = palette[prediction]
        mask_image = Image.fromarray(mask_rgb)
        mask_image.save(os.path.join(MASK_DIR, f"{image_id}_mask.png"))

        overlay = Image.blend(image, mask_image, alpha=0.45)
        overlay.save(os.path.join(OVERLAY_DIR, f"{image_id}_overlay.png"))

        row = {"image_id": image_id, "width": width, "height": height}

        for class_id, label in model.config.id2label.items():
            label = str(label)
            percent = float(np.mean(prediction == int(class_id)) * 100)
            row[f"{label}_percent"] = round(percent, 3)

        summary_rows.append(row)
        print(f"Processed: {filename} ({width}x{height})")

    except Exception as exc:
        print(f"Failed: {filename}: {exc}")

if summary_rows:
    summary_path = os.path.join(OUTPUT_DIR, "segmentation_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print("\nSuccessful images:", len(summary_rows), "of", len(image_paths))
    print("Summary CSV:", summary_path)
    print("Masks folder:", MASK_DIR)
    print("Overlays folder:", OVERLAY_DIR)
else:
    print("No images were successfully processed.")
