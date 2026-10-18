
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageColor
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
)


# ---------------------------------------------------------
# 1. PROJECT PATHS AND MODEL CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data" / "streetview" / "images"
OUTPUT_DIR = ROOT / "outputs" / "streetview"
MASK_DIR = OUTPUT_DIR / "masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

# Cityscapes class IDs used in this exercise.
TARGET_CLASSES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    8: "vegetation",
    10: "sky",
    13: "car",
}

# Overlay colours for the selected classes.
COLORS = {
    0: "#808080",
    1: "#FFD700",
    2: "#FF6347",
    8: "#228B22",
    10: "#1E90FF",
    13: "#BA55D3",
}


# ---------------------------------------------------------
# 2. LOCATION ANALYSIS FUNCTION
# ---------------------------------------------------------

def analyze_location(image_filename):
    """
    Return scene composition and a previously calculated
    preliminary walkability proxy for a processed image.

    This function reads existing output files. It does not
    rerun segmentation or recalculate the walkability score.
    """

    percentages_path = OUTPUT_DIR / "class_percentages.csv"
    scores_path = OUTPUT_DIR / "walkability_scores_24.csv"

    if not percentages_path.is_file():
        raise FileNotFoundError(
            f"Segmentation CSV not found: {percentages_path}"
        )

    if not scores_path.is_file():
        raise FileNotFoundError(
            f"Walkability scores not found: {scores_path}. "
            "Run score_walkability.py first."
        )

    # Find the image's scene composition.
    scene_composition = None

    with percentages_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row.get("image") == image_filename:
                scene_composition = {
                    class_name: float(row[class_name])
                    for class_name in TARGET_CLASSES.values()
                }
                break

    if scene_composition is None:
        raise ValueError(
            f"No segmentation result found for {image_filename}"
        )

    # Find the previously calculated walkability proxy.
    walkability_score = None
    primary_rank = None
    alternative_score = None
    alternative_rank = None

    with scores_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row.get("filename") == image_filename:
                walkability_score = float(
                    row["walkability_proxy"]
                )
                primary_rank = int(row["primary_rank"])
                alternative_score = float(
                    row["walkability_alternative"]
                )
                alternative_rank = int(
                    row["alternative_rank"]
                )
                break

    if walkability_score is None:
        raise ValueError(
            f"No walkability score found for {image_filename}"
        )

    return {
        "image": image_filename,
        "scene_composition": scene_composition,
        "walkability_score": walkability_score,
        "primary_rank": primary_rank,
        "walkability_alternative": alternative_score,
        "alternative_rank": alternative_rank,
        "score_type": "sample-relative preliminary proxy",
    }


# ---------------------------------------------------------
# 3. SEGMENTATION WORKFLOW
# ---------------------------------------------------------

def main():

    # Create output directories.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGE_DIR.is_dir():
        raise SystemExit(
            f"Image directory does not exist: {IMAGE_DIR}"
        )

    # Find supported image files.
    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    # Exclude non-image metadata files automatically by extension.
    if not image_paths:
        raise SystemExit(
            f"No images found in {IMAGE_DIR}"
        )

    # Choose the available processing device.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device.type.upper())
    print("Images found:", len(image_paths))
    print("Loading model:", MODEL_NAME)
    print(
        "The first run may download model weights "
        "from Hugging Face."
    )

    # Load processor and pretrained model.
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME
    )

    model.eval()
    model.to(device)

    results = []

    # Process each image independently.
    for index, image_path in enumerate(image_paths, start=1):

        print(
            f"\n[{index}/{len(image_paths)}] "
            f"Processing {image_path.name}"
        )

        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")

            inputs = processor(
                images=image,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                outputs = model(**inputs)

                logits = F.interpolate(
                    outputs.logits,
                    size=(image.height, image.width),
                    mode="bilinear",
                    align_corners=False,
                )

                prediction = (
                    logits.argmax(dim=1)[0]
                    .cpu()
                    .numpy()
                )

            # Calculate the percentage of pixels in each target class.
            total_pixels = prediction.size
            percentages = {}

            for class_id, class_name in TARGET_CLASSES.items():
                count = int(
                    np.count_nonzero(prediction == class_id)
                )

                percentages[class_name] = round(
                    100.0 * count / total_pixels,
                    3,
                )

            # Save the predicted class-ID mask.
            mask_path = (
                MASK_DIR / f"{image_path.stem}_mask.png"
            )

            Image.fromarray(
                prediction.astype(np.uint8)
            ).save(mask_path)

            # Create a transparent colour overlay.
            color_array = np.zeros(
                (image.height, image.width, 4),
                dtype=np.uint8,
            )

            for class_id, hex_color in COLORS.items():
                rgb = ImageColor.getrgb(hex_color)
                selected = prediction == class_id

                color_array[selected, :3] = rgb
                color_array[selected, 3] = 150

            color_overlay = Image.fromarray(
                color_array,
                mode="RGBA",
            )

            overlay = Image.alpha_composite(
                image.convert("RGBA"),
                color_overlay,
            )

            overlay_path = (
                OVERLAY_DIR
                / f"{image_path.stem}_overlay.png"
            )

            overlay.save(overlay_path)

            # Store results for JSON and CSV output.
            record = {
                "image": image_path.name,
                "width": image.width,
                "height": image.height,
                "model": MODEL_NAME,
                "class_percentages": percentages,
                "mask": str(mask_path.relative_to(ROOT)),
                "overlay": str(overlay_path.relative_to(ROOT)),
            }

            results.append(record)

            print("Class percentages:")

            for name, percentage in percentages.items():
                print(f"  {name}: {percentage:.3f}%")

            print("Saved mask:", mask_path)
            print("Saved overlay:", overlay_path)

        except Exception as error:
            print(
                f"ERROR processing {image_path.name}: {error}"
            )
            raise

    # Save complete segmentation results as JSON.
    json_path = OUTPUT_DIR / "segmentation_results.json"

    with json_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    # Save class percentages as CSV.
    csv_path = OUTPUT_DIR / "class_percentages.csv"

    fieldnames = [
        "image",
        "width",
        "height",
        *TARGET_CLASSES.values(),
    ]

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            row = {
                "image": result["image"],
                "width": result["width"],
                "height": result["height"],
                **result["class_percentages"],
            }

            writer.writerow(row)

    print("\nSEGMENTATION FINISHED")
    print("Images processed:", len(results))
    print("JSON:", json_path)
    print("CSV:", csv_path)
    print("Masks:", MASK_DIR)
    print("Overlays:", OVERLAY_DIR)

    print(
        "\nNote: segmentation results describe model predictions; "
        "they are not ground-truth labels."
    )


# ---------------------------------------------------------
# 4. SCRIPT ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
