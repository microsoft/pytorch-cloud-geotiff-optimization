# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import planetary_computer as pc
import pystac_client
import stackstac
import rioxarray
import numpy as np
import rasterio
from tqdm import tqdm
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 imagery with various compression settings"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory where images will be saved",
    )
    args = parser.parse_args()

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
    )

    area_of_interest = {
        "coordinates": [
            [
                [35.93742049088772, -0.7293990623006721],
                [35.93742049088772, -1.9960903674445376],
                [37.623493987775305, -1.9960903674445376],
                [37.623493987775305, -0.7293990623006721],
                [35.93742049088772, -0.7293990623006721],
            ]
        ],
        "type": "Polygon",
    }

    time_of_interest = "2024-01-01/2024-12-31"

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=area_of_interest,
        datetime=time_of_interest,
        query={"eo:cloud_cover": {"lt": 5}},
    )

    items = search.item_collection()
    print(f"Returned {len(items)} Items")

    compression_types = [
        "webp",
        "lerc",
        "jpeg2000",
        "jpeg",
        "lzw",
        "deflate_1",
        "deflate_6",
        "deflate_9",
        "none",
    ]

    block_sizes = [128, 256, 512, 1024]

    def download_item(item, root_output_dir):
        os.makedirs(root_output_dir, exist_ok=True)

        for compression_type in tqdm(compression_types):
            for block_size in tqdm(block_sizes):
                output_dir = os.path.join(
                    root_output_dir, compression_type, f"block_{block_size}"
                )
                os.makedirs(output_dir, exist_ok=True)

                output_file = os.path.join(output_dir, f"{item.id}.tif")

                if os.path.exists(output_file):
                    continue

                all_channels = []
                for band in tqdm(["B02", "B03", "B04", "B08"]):
                    with rasterio.open(pc.sign(item.assets[band].href)) as src:
                        profile = src.profile
                        channel = src.read()
                        all_channels.append(channel)
                data = np.concatenate(all_channels, axis=0)

                del profile["nodata"]
                profile["count"] = 4
                profile["tiled"] = True
                profile["interleave"] = "pixel"

                profile["blockxsize"] = block_size
                profile["blockysize"] = block_size

                for key in ["compress", "predictor", "level", "quality", "max_z_error"]:
                    if key in profile:
                        del profile[key]

                current_profile = profile.copy()
                current_data = data.copy()

                needs_8bit = compression_type in [
                    "webp",
                    "jpeg",
                    "jpeg2000",
                    "jpegxl",
                    "lerc",
                ]
                if needs_8bit and current_profile.get("dtype") != "uint8":
                    min_val = np.min(current_data)
                    max_val = np.max(current_data)
                    if max_val > min_val:
                        current_data = np.clip(
                            ((current_data - min_val) / (max_val - min_val) * 255),
                            0,
                            255,
                        ).astype(np.uint8)
                    else:
                        current_data = np.zeros_like(current_data).astype(np.uint8)
                    current_profile["dtype"] = "uint8"

                if compression_type == "none":
                    current_profile["compress"] = "NONE"
                elif compression_type == "lzw":
                    current_profile["compress"] = "LZW"
                    current_profile["predictor"] = 2
                elif compression_type == "deflate_1":
                    current_profile["compress"] = "DEFLATE"
                    current_profile["zlevel"] = 1
                elif compression_type == "deflate_6":
                    current_profile["compress"] = "DEFLATE"
                    current_profile["zlevel"] = 6
                elif compression_type == "deflate_9":
                    current_profile["compress"] = "DEFLATE"
                    current_profile["zlevel"] = 9
                elif compression_type == "webp":
                    current_profile["compress"] = "WEBP"
                    current_profile["quality"] = 75
                elif compression_type == "jpeg":
                    current_profile["compress"] = "JPEG"
                    current_profile["quality"] = 85
                elif compression_type == "jpeg2000":
                    current_profile["compress"] = "JPEG2000"
                    current_profile["quality"] = 80
                elif compression_type == "jpegxl":
                    current_profile["compress"] = "JXL"
                    current_profile["quality"] = 80
                elif compression_type == "lerc":
                    current_profile["compress"] = "LERC"
                    current_profile["max_z_error"] = 0.001
                current_profile["num_threads"] = "all_cpus"

                with rasterio.open(output_file, "w", **current_profile) as f:
                    f.write(current_data)

    for item in tqdm(items[:10]):
        download_item(item, args.output_dir)
