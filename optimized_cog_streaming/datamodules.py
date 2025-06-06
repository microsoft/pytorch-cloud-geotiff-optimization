# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os.path as osp
import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader
import rasterio
import os
from tqdm import tqdm


def collate_fn(batch):
    images = []
    masks = []
    for item in batch:
        for img, mask in item:
            images.append(img)
            masks.append(mask)
    return torch.stack(images), torch.stack(masks)


def construct_paths(base_path, filenames, is_remote=False, sas_token=None):
    if is_remote:
        return [f"{base_path}/images/{fn}{sas_token}" for fn in filenames], [
            f"{base_path}/masks/{fn}{sas_token}" for fn in filenames
        ]
    else:
        return [osp.join(base_path, "images", fn) for fn in filenames], [
            osp.join(base_path, "masks", fn) for fn in filenames
        ]


def get_bytes_per_pixel(dtype):
    """Convert numpy/rasterio data type to bytes per pixel."""
    dtype_str = str(dtype)
    if "uint8" in dtype_str:
        return 1
    elif "uint16" in dtype_str:
        return 2
    elif "uint32" in dtype_str:
        return 4
    elif "int8" in dtype_str:
        return 1
    elif "int16" in dtype_str:
        return 2
    elif "int32" in dtype_str:
        return 4
    elif "float16" in dtype_str:
        return 2
    elif "float32" in dtype_str:
        return 4
    elif "float64" in dtype_str:
        return 8
    else:
        # Default to 4 bytes if unknown
        return 4


def get_image_sizes(uris):
    sizes = []
    block_size = None
    dtype = None
    bad_files = list()
    for uri in uris:
        try:
            with rasterio.open(uri) as src:
                sizes.append(src.shape)
                if dtype is None:
                    dtype = src.dtypes[0]  # Get data type from first band
                if block_size is None:
                    block_size = src.profile.get("blockysize", 256), src.profile.get(
                        "blockxsize", 256
                    )
                else:
                    block_y = src.profile.get("blockysize", 256)
                    block_x = src.profile.get("blockxsize", 256)
                    if block_y != block_size[0] or block_x != block_size[1]:
                        print(
                            f"Warning: Inconsistent block sizes detected. Using the first one: {block_size}"
                        )
        except Exception as e:
            print(f"Error opening {uri}: {e}")
            bad_files.append(uri)

    return sizes, block_size, dtype, bad_files


class RandomBlockSampler(Sampler):
    def __init__(
        self, image_fns, image_sizes, patch_size, length, block_size=None, num_threads=1
    ):
        self.image_fns = image_fns
        self.image_sizes = image_sizes
        self.patch_size = patch_size
        self.block_size = block_size
        self.length = length
        self.num_threads = num_threads
        self.N = len(image_fns)

    def __len__(self):
        assert (
            self.length % self.num_threads == 0
        ), "length must be divisible by num_threads"
        return self.length // self.num_threads

    def __iter__(self):
        indices = np.random.choice(self.N, size=self.length, replace=True)
        indices = [int(i) for i in indices]

        samples = []
        for i in indices:
            if i >= len(self.image_sizes):
                print(
                    f"Warning: Index {i} out of bounds for image_sizes (len={len(self.image_sizes)})"
                )
                continue

            image_size = self.image_sizes[i]
            height, width = image_size[0], image_size[1]

            # Ensure patch fits within image dimensions
            if width <= self.patch_size or height <= self.patch_size:
                print(
                    f"Warning: Image {i} ({width}x{height}) is smaller than patch size {self.patch_size}. Skipping."
                )
                continue

            if self.block_size is None:
                x = np.random.randint(0, width - self.patch_size)
                y = np.random.randint(0, height - self.patch_size)
            else:
                block_y, block_x = self.block_size

                # Calculate how many complete blocks we can fit
                num_blocks_y = max(1, (height - self.patch_size) // block_y)
                num_blocks_x = max(1, (width - self.patch_size) // block_x)

                if num_blocks_y <= 0 or num_blocks_x <= 0:
                    # If we can't fit a complete block, just use random sampling
                    x = np.random.randint(0, width - self.patch_size)
                    y = np.random.randint(0, height - self.patch_size)
                else:
                    block_y = np.random.randint(0, num_blocks_y)
                    block_x = np.random.randint(0, num_blocks_x)

                    block_start_y = block_y * self.block_size[0]
                    block_start_x = block_x * self.block_size[1]

                    if self.patch_size <= self.block_size[0]:
                        max_y_offset = min(
                            self.block_size[0] - self.patch_size,
                            height - block_start_y - self.patch_size,
                        )
                        y = block_start_y + np.random.randint(
                            0, max(1, max_y_offset + 1)
                        )
                    else:
                        y = min(block_start_y, height - self.patch_size)

                    if self.patch_size <= self.block_size[1]:
                        max_x_offset = min(
                            self.block_size[1] - self.patch_size,
                            width - block_start_x - self.patch_size,
                        )
                        x = block_start_x + np.random.randint(
                            0, max(1, max_x_offset + 1)
                        )
                    else:
                        x = min(block_start_x, width - self.patch_size)

            # Final safety check
            x = min(x, width - self.patch_size)
            y = min(y, height - self.patch_size)

            samples.append((i, y, x, self.patch_size))

        if self.num_threads == 1:
            for sample in samples:
                yield sample
        else:
            for i in range(0, len(samples), self.num_threads):
                yield samples[i : i + self.num_threads]


class SegmentationDataModule:
    def __init__(
        self,
        config_name,
        config_paths,
        config_params,
        train_filenames,
        val_filenames,
        batch_size,
        device,
        dataset="vaihingen",
    ):
        self.config_name = config_name
        self.config_paths = config_paths
        self.config_params = config_params
        self.train_filenames = train_filenames
        self.val_filenames = val_filenames
        self.batch_size = batch_size
        self.dataset = dataset
        self.device = device

        self.train_loader = None
        self.val_loader = None
        self.val_batch = None

    def setup(self):
        from optimized_cog_streaming.datasets import SegmentationDataset

        params = self.config_params[self.config_name]
        base_path = self.config_paths[self.dataset][self.config_name]
        use_remote = self.config_name.startswith("remote")

        sas_token = os.getenv("AZURE_SAS") if use_remote else None

        train_image_files, train_mask_files = construct_paths(
            base_path, self.train_filenames, use_remote, sas_token
        )
        val_image_files, val_mask_files = construct_paths(
            base_path, self.val_filenames, use_remote, sas_token
        )

        train_image_sizes, train_block_size, train_dtype, train_bad_files = (
            get_image_sizes(train_image_files)
        )
        val_image_sizes, _, val_dtype, val_bad_files = get_image_sizes(val_image_files)

        if len(train_bad_files) > 0:
            bad_file_positions = [
                train_image_files.index(file) for file in train_bad_files
            ]
            train_image_files = [
                file
                for i, file in enumerate(train_image_files)
                if i not in bad_file_positions
            ]
            train_mask_files = [
                file
                for i, file in enumerate(train_mask_files)
                if i not in bad_file_positions
            ]
            train_image_sizes = [
                size
                for i, size in enumerate(train_image_sizes)
                if i not in bad_file_positions
            ]

        if len(val_bad_files) > 0:
            bad_file_positions = [val_image_files.index(file) for file in val_bad_files]
            val_image_files = [
                file
                for i, file in enumerate(val_image_files)
                if i not in bad_file_positions
            ]
            val_mask_files = [
                file
                for i, file in enumerate(val_mask_files)
                if i not in bad_file_positions
            ]
            val_image_sizes = [
                size
                for i, size in enumerate(val_image_sizes)
                if i not in bad_file_positions
            ]

        t_block_size = None
        if params["sampler_type"] == "block":
            t_block_size = train_block_size

        # Create dataset objects
        train_dataset = SegmentationDataset(
            train_image_files,
            train_mask_files,
            num_threads=params["num_threads"],
            use_augmentation=True,
        )

        val_dataset = SegmentationDataset(
            val_image_files,
            val_mask_files,
            num_threads=1,
            use_augmentation=False,
        )

        # Create sampler objects
        train_sampler = RandomBlockSampler(
            train_image_files,
            train_image_sizes,
            patch_size=params["patch_size"],
            length=1_002_400,
            block_size=t_block_size,
            num_threads=params["num_threads"],
        )

        val_sampler = RandomBlockSampler(
            val_image_files,
            val_image_sizes,
            patch_size=params["patch_size"],
            length=256,
            block_size=t_block_size,
            num_threads=1,
        )

        t_collate_fn = None
        if params["num_threads"] > 1:
            t_collate_fn = collate_fn

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.batch_size,
            num_workers=params["num_workers"],
            prefetch_factor=params["prefetch_factor"],
            collate_fn=t_collate_fn,
            persistent_workers=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=4,
            num_workers=64,
            prefetch_factor=params["prefetch_factor"],
            collate_fn=t_collate_fn,
            persistent_workers=True,
        )

        val_images, val_masks = [], []
        for batch in tqdm(self.val_loader):
            images, masks = batch
            val_images.append(images.to(self.device) / 255.0)
            val_masks.append(masks.to(self.device).squeeze(1))
        self.val_batch = torch.cat(val_images), torch.cat(val_masks)

        # Clean up validation data loader after loading data
        self.val_loader = None

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def get_val_batch(self):
        return self.val_batch
