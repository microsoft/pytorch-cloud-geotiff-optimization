# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


def process_vaihingen_dataset(raw_path):

    root = Path(raw_path)

    images_source_dir = root / "ISPRS_semantic_labeling_Vaihingen" / "top"
    masks_source_dir = root / "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE"
    geoinfo_source_dir = root / "Vaihingen_dsm_tiles_geoinfo"

    if not images_source_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_source_dir}")
    if not masks_source_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_source_dir}")
    if not geoinfo_source_dir.exists():
        raise FileNotFoundError(f"Geoinfo directory not found: {geoinfo_source_dir}")

    images = list(images_source_dir.glob("*"))
    images = [
        f for f in images if f.is_file() and f.suffix.lower() in [".tif", ".tiff"]
    ]

    masks = list(masks_source_dir.glob("*"))
    masks = [f for f in masks if f.is_file() and f.suffix.lower() in [".tif", ".tiff"]]

    geoinfo_files = list(geoinfo_source_dir.glob("*"))
    geoinfo_files = [
        f for f in geoinfo_files if f.is_file() and f.suffix.lower() == ".tfw"
    ]

    if not images:
        raise ValueError(f"No .tif/.tiff images found in {images_source_dir}")
    if not masks:
        raise ValueError(f"No .tif/.tiff masks found in {masks_source_dir}")
    if not geoinfo_files:
        raise ValueError(f"No .tfw geoinfo files found in {geoinfo_source_dir}")

    print(
        f"Images: {len(images)}, Masks: {len(masks)}, Geoinfo files: {len(geoinfo_files)}"
    )

    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for img in tqdm(images, desc="Processing images"):
        shutil.copy(img, images_dir / img.name)

    for mask in tqdm(masks, desc="Processing masks"):
        shutil.copy(mask, masks_dir / mask.name)

    for geoinfo in tqdm(geoinfo_files, desc="Processing geoinfo files"):
        shutil.copy(geoinfo, images_dir / geoinfo.name)

    for item in root.iterdir():
        if item.name not in ["images", "masks"]:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    print("Processing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Vaihingen dataset")
    parser.add_argument("--path", required=True, help="Path to raw dataset")

    args = parser.parse_args()
    process_vaihingen_dataset(args.path)
