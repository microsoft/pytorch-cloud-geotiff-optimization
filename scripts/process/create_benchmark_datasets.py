# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import rasterio as rio
import numpy as np
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

class_mapping = {
    (255, 255, 255): 1,  # Impervious surfaces
    (0, 0, 255): 2,  # Building
    (0, 255, 255): 3,  # Low vegetation
    (0, 255, 0): 4,  # Tree
    (255, 255, 0): 5,  # Car
    (255, 0, 0): 6,  # Clutter/background
}

DATASET_CONFIG = {
    "vaihingen": {
        "needs_encoding": True,
        "crs": "EPSG:32632",
        "transform_pattern": "*{area}*.tfw",
    },
    "potsdam": {
        "needs_encoding": True,
        "crs": "EPSG:32633",
        "transform_pattern": "{file_name}.tfw",
    },
    "dfc-22": {"needs_encoding": False, "crs": None, "transform_pattern": None},
}

PROCESSING_CONFIG = {
    "default": {"compress": "DEFLATE", "zlevel": 6, "cog_profile": "deflate"},
    "optimal_local": {"compress": "none", "cog_profile": "raw"},
    "optimal_remote": {"compress": "LERC_ZSTD", "zlevel": 22, "cog_profile": "raw"},
}


def encode_mask(mask_path, output_path, dataset_type):
    """Encode RGB mask to single-channel for vaihingen/potsdam"""
    config = DATASET_CONFIG[dataset_type]

    with rio.open(mask_path) as src:
        mask_data = src.read()
        profile = src.profile.copy()
        if config["crs"]:
            profile.update(crs=config["crs"])

    # Handle transform files
    transform_path = None
    if config["transform_pattern"]:
        if dataset_type == "vaihingen":
            area = mask_path.stem.split("_")[-1]
            transform_files = list(
                (mask_path.parent.parent / "images").glob(f"*{area}*.tfw")
            )
            if transform_files:
                transform_path = transform_files[0]
        elif dataset_type == "potsdam":
            file_name = mask_path.stem
            transform_path = mask_path.parent.parent / "images" / f"{file_name}.tfw"

    if transform_path and transform_path.exists():
        with open(transform_path) as f:
            tfw = [float(line.strip()) for line in f.readlines()]
        transform = rio.transform.Affine(tfw[0], tfw[1], tfw[4], tfw[2], tfw[3], tfw[5])
        profile.update(transform=transform)

        # Update corresponding image with transform/CRS
        img_path = mask_path.parent.parent / "images" / mask_path.name
        if img_path.exists():
            with rio.open(img_path, "r+") as src:
                src.transform = profile["transform"]
                src.crs = profile["crs"]

    # Create single-channel mask
    new_mask = np.zeros((mask_data.shape[1], mask_data.shape[2]), dtype=np.uint8)

    for rgb, class_value in class_mapping.items():
        rgb_match = (
            (mask_data[0] == rgb[0])
            & (mask_data[1] == rgb[1])
            & (mask_data[2] == rgb[2])
        )
        new_mask[rgb_match] = class_value

    profile.update(count=1, dtype="uint8", nodata=0)

    with rio.open(output_path, "w", **profile) as dst:
        dst.write(new_mask, 1)


def process_files(files, processing_type):
    """Process files with given compression settings"""
    config = PROCESSING_CONFIG[processing_type]

    for file_path in tqdm(files, desc=f"Processing {processing_type}"):
        with rio.open(file_path) as src:
            profile = src.profile.copy()
            data = src.read()

        profile.update(
            compress=config["compress"],
            blockxsize=512,
            blockysize=512,
            tiled=True,
            interleave="pixel",
            num_threads="all_cpus",
        )

        if "zlevel" in config:
            profile.update(zlevel=config["zlevel"])

        temp_path = file_path.with_suffix(".temp.tif")

        with rio.open(temp_path, "w", **profile) as dst:
            dst.write(data)

        cog_profile = cog_profiles.get(config["cog_profile"])
        cog_profile.update(
            blockxsize=512,
            blockysize=512,
            compress=config["compress"],
            tiled=True,
            interleave="pixel",
            num_threads="all_cpus",
        )

        if "zlevel" in config:
            cog_profile.update(zlevel=config["zlevel"])

        cog_translate(temp_path, file_path, cog_profile, quiet=True)
        temp_path.unlink()


def setup_directories(raw_path):
    """Create output directories and copy raw data"""
    raw_path = Path(raw_path)
    parent_dir = raw_path.parent

    output_dirs = {}
    for proc_type in ["default", "optimal_local", "optimal_remote"]:
        output_dir = parent_dir / proc_type
        output_dir.mkdir(exist_ok=True)
        (output_dir / "images").mkdir(exist_ok=True)
        (output_dir / "masks").mkdir(exist_ok=True)
        output_dirs[proc_type] = output_dir

    return output_dirs


def copy_and_encode_data(raw_path, output_dirs, dataset_type):
    """Copy raw data to output directories and encode masks if needed"""
    raw_path = Path(raw_path)
    images = list((raw_path / "images").glob("*.tif"))
    masks = list((raw_path / "masks").glob("*.tif"))

    config = DATASET_CONFIG[dataset_type]

    for proc_type, output_dir in output_dirs.items():
        print(f"Copying data for {proc_type}...")

        # Copy images
        for img in tqdm(images, desc="Copying images"):
            shutil.copy2(img, output_dir / "images" / img.name)

        # Copy/encode masks
        for mask in tqdm(masks, desc="Processing masks"):
            output_mask = output_dir / "masks" / mask.name
            if config["needs_encoding"]:
                encode_mask(mask, output_mask, dataset_type)
            else:
                shutil.copy2(mask, output_mask)


def main():
    parser = argparse.ArgumentParser(
        description="Process dataset with multiple optimization profiles"
    )
    parser.add_argument("--raw_path", help="Path to raw dataset directory")
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    dataset_type = raw_path.parent.name

    if dataset_type not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. Must be one of {list(DATASET_CONFIG.keys())}"
        )

    print(f"Processing {dataset_type} dataset...")

    # Setup directories
    output_dirs = setup_directories(raw_path)

    # Copy and encode data
    copy_and_encode_data(raw_path, output_dirs, dataset_type)

    # Process each optimization type
    for proc_type, output_dir in output_dirs.items():
        print(f"Optimizing for {proc_type}...")

        images = list((output_dir / "images").glob("*.tif"))
        masks = list((output_dir / "masks").glob("*.tif"))

        process_files(images, proc_type)
        process_files(masks, proc_type)

    print("Processing complete!")


if __name__ == "__main__":
    main()
