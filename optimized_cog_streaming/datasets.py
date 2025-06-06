# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
import rasterio
import kornia.augmentation as K


class SegmentationDataset(Dataset):
    def __init__(self, image_fns, mask_fns, num_threads=1, use_augmentation=True):
        self.image_fns = image_fns
        self.mask_fns = mask_fns
        self.num_threads = num_threads
        self.use_augmentation = use_augmentation

        if use_augmentation:
            self.augmentations = K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                K.RandomRotation(degrees=90, p=0.5, align_corners=True),
                same_on_batch=False,
                data_keys=["input", "mask"],
            )

    def load_single(self, idx):
        i, y, x, size = idx

        if i >= len(self.image_fns) or i >= len(self.mask_fns):
            raise IndexError(
                f"Index {i} out of bounds (len images={len(self.image_fns)}, len masks={len(self.mask_fns)})"
            )

        window = rasterio.windows.Window(x, y, size, size)

        image_path = self.image_fns[i]
        mask_path = self.mask_fns[i]

        try:
            with rasterio.open(image_path) as f:
                image_height, image_width = f.shape
                # Ensure window is within bounds
                if x + size > image_width or y + size > image_height:
                    valid_size_x = min(size, image_width - x)
                    valid_size_y = min(size, image_height - y)
                    valid_size = min(valid_size_x, valid_size_y)
                    window = rasterio.windows.Window(x, y, valid_size, valid_size)
                    image = f.read(window=window)
                    # Pad if necessary
                    if valid_size < size:
                        padded_image = np.zeros((3, size, size), dtype=image.dtype)
                        padded_image[:, :valid_size, :valid_size] = image
                        image = padded_image
                else:
                    image = f.read([1, 2, 3], window=window)
        except Exception as e:
            print(f"Error reading image {image_path} at window {window}: {e}")
            # Return zeros if image can't be read
            image = np.zeros((3, size, size), dtype=np.float32)

        try:
            with rasterio.open(mask_path) as f:
                mask_height, mask_width = f.shape
                # Ensure window is within bounds
                if x + size > mask_width or y + size > mask_height:
                    valid_size_x = min(size, mask_width - x)
                    valid_size_y = min(size, mask_height - y)
                    valid_size = min(valid_size_x, valid_size_y)
                    window = rasterio.windows.Window(x, y, valid_size, valid_size)
                    mask = f.read(window=window, out_shape=(1, valid_size, valid_size))
                    # Pad if necessary
                    if valid_size < size:
                        padded_mask = np.zeros((1, size, size), dtype=mask.dtype)
                        padded_mask[:, :valid_size, :valid_size] = mask
                        mask = padded_mask
                else:
                    mask = f.read(window=window, out_shape=(1, size, size))
        except Exception as e:
            print(f"Error reading mask {mask_path} at window {window}: {e}")
            # Return zeros if mask can't be read
            mask = np.zeros((1, size, size), dtype=np.int64)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        return image, mask

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            image, mask = self.load_single(idx)

            if self.use_augmentation:
                # Add batch dimension for Kornia
                image_batch = image.unsqueeze(0)
                mask_batch = mask.unsqueeze(0)

                # Apply same augmentation to both image and mask
                augmented = self.augmentations(image_batch, mask_batch)

                # Remove batch dimension
                image = augmented[0].squeeze(0)
                mask = augmented[1].squeeze(0)

            return image, mask

        elif isinstance(idx, list):
            futures = []
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                for sub_idx in idx:
                    assert isinstance(sub_idx, tuple)
                    futures.append(executor.submit(self.load_single, sub_idx))
            results = [future.result() for future in futures]

            if self.use_augmentation:
                augmented_results = []
                for img, mask in results:
                    # Add batch dimension for Kornia
                    img_batch = img.unsqueeze(0)
                    mask_batch = mask.unsqueeze(0)

                    # Apply same augmentation to both image and mask
                    augmented = self.augmentations(img_batch, mask_batch)

                    # Remove batch dimension
                    img = augmented[0].squeeze(0)
                    mask = augmented[1].squeeze(0)
                    augmented_results.append((img, mask))

                return augmented_results

            return results


class TileDataset(Dataset):
    def __init__(self, image_fns, num_threads=1):
        self.image_fns = image_fns
        self.num_threads = num_threads

        if self.num_threads > 1:
            self.pool = ThreadPoolExecutor(max_workers=self.num_threads)
        else:
            self.pool = None

    def load_single(self, idx):
        i, y, x, size = idx
        window = rasterio.windows.Window(x, y, size, size)

        path = self.image_fns[i]

        with rasterio.open(path) as f:
            data = f.read(window=window)
        data = torch.from_numpy(data).float()

        return data

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.load_single(idx)
        elif isinstance(idx, list):
            futures = []
            for sub_idx in idx:
                assert isinstance(sub_idx, tuple)
                futures.append(self.pool.submit(self.load_single, sub_idx))
            results = [future.result() for future in futures]
            return results

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True)

    def __del__(self):
        self.close()
