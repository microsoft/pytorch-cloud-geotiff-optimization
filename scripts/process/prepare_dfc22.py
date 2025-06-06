# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


def process_dfc22_dataset(raw_path):

    root = Path(raw_path)

    files = list(root.glob("**/*.tif"))
    print(f"Total files found: {len(files)}")

    files = [f for f in files if "RGEALTI" not in str(f)]
    print(f"Files after filtering RGEALTI: {len(files)}")

    images = [f for f in files if "UrbanAtlas" not in str(f)]
    masks = [f for f in files if "UrbanAtlas" in str(f)]
    print(f"Images: {len(images)}, Masks: {len(masks)}")

    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for img, mask in tqdm(zip(images, masks), total=len(images)):
        new_mask_name = mask.name.replace("_UA2012", "")
        shutil.copy(img, images_dir / img.name)
        shutil.copy(mask, masks_dir / new_mask_name)

    for item in root.iterdir():
        if item.name not in ["images", "masks"]:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    print("Processing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DFC22 dataset")
    parser.add_argument("--path", required=True, help="Path to raw dataset")

    args = parser.parse_args()
    process_dfc22_dataset(args.path)
