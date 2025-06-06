# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


def process_potsdam_dataset(raw_path):

    root = Path(raw_path)

    files = list(root.glob("**/*"))
    print(f"Total files found: {len(files)}")

    files = [f for f in files if f.is_file()]
    print(f"Files after filtering directories: {len(files)}")

    images = [f for f in files if "_RGBIR" in f.name]
    masks = [f for f in files if "_label" in f.name]
    print(f"Images: {len(images)}, Masks: {len(masks)}")

    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for img in tqdm(images, desc="Processing images"):
        new_name = img.name.replace("_RGBIR", "")
        shutil.copy(img, images_dir / new_name)

    for mask in tqdm(masks, desc="Processing masks"):
        new_name = mask.name.replace("_label", "")
        shutil.copy(mask, masks_dir / new_name)

    for item in root.iterdir():
        if item.name not in ["images", "masks"]:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    print("Processing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Potsdam dataset")
    parser.add_argument("--path", required=True, help="Path to raw dataset")

    args = parser.parse_args()
    process_potsdam_dataset(args.path)
