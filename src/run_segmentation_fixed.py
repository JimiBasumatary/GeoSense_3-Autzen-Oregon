
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
MASK_DIR = os.path.join(OUTPUT_DIR, "class_id_masks")
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "overlays_cityscapes")

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)

MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

palette = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70],
    [102, 102, 156], [190, 153, 153], [153, 153, 153],
    [250, 170, 30], [220, 220, 0], [107, 142, 35],
    [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230],
    [119, 11, 32],
], dtype=np.uint8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

label_map = {int(k): str(v) for k, v in model.config.id2label.items()}
image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))
summary = []

print("Device:", device)
print("Images found:", len(image_paths))

for path in image_paths:
    image_id = os.path.splitext(os.path.basename(path))[0]
    image = Image.open(path).convert("RGB")
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

    # Save actual integer class IDs: values 0 through 18
    Image.fromarray(prediction, mode="L").save(
        os.path.join(MASK_DIR, f"{image_id}_class_ids.png")
    )

    colour_mask = Image.fromarray(palette[prediction], mode="RGB")
    overlay = Image.blend(image, colour_mask, alpha=0.45)
    overlay.save(os.path.join(OVERLAY_DIR, f"{image_id}_overlay.png"))

    row = {"image_id": image_id, "width": width, "height": height}
    for class_id, label in label_map.items():
        row[f"{label}_percent"] = round(
            float(np.mean(prediction == class_id) * 100), 3
        )
    summary.append(row)

    print(
        f"{image_id}: mask IDs {prediction.min()}–{prediction.max()} "
        f"({len(np.unique(prediction))} classes present)"
    )

summary_path = os.path.join(OUTPUT_DIR, "segmentation_summary_fixed.csv")
pd.DataFrame(summary).to_csv(summary_path, index=False)

print("\nCompleted:", len(summary), "images")
print("Class-ID masks:", MASK_DIR)
print("Corrected overlays:", OVERLAY_DIR)
print("Summary:", summary_path)
