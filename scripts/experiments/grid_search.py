# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import json
import csv
import time as time_module
import threading
import signal
import statistics
import gc

import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import pynvml
import yaml

from dotenv import load_dotenv

load_dotenv()

from optimized_cog_streaming.datamodules import (
    get_image_sizes,
    RandomBlockSampler,
    get_bytes_per_pixel,
)
from optimized_cog_streaming.datasets import TileDataset


def collate_fn(batch):
    batch = [item for sublist in batch for item in sublist]
    return torch.stack(batch, dim=0)


def evaluate_dataloader_config(
    urls,
    image_sizes,
    block_size,
    sampler_type,
    patch_size,
    batch_size,
    num_workers,
    num_threads,
    persistent_workers,
    prefetch_factor,
    length,
    max_iterations,
    device,
    model,
    dtype=None,
):
    """
    Evaluate a specific dataloader configuration and return mb_per_second.
    """
    # Clear all caches before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force garbage collection
    gc.collect()

    print(f"Device: {device}")

    t_block_size = None
    if sampler_type == "block":
        t_block_size = block_size

    t_collate_fn = None
    if num_threads > 1:
        t_collate_fn = collate_fn

    dataset = TileDataset(
        urls,
        num_threads=num_threads,
    )

    sampler = RandomBlockSampler(
        urls,
        image_sizes,
        patch_size=patch_size,
        length=length,
        block_size=t_block_size,
        num_threads=num_threads,
    )

    dl = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=t_collate_fn,
    )

    gpu_util_samples = []
    stop_monitor = threading.Event()
    gpu_id = device.index if device.type == "cuda" else None

    def monitor_gpu():
        if gpu_id is None or pynvml is None:
            return

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

            while not stop_monitor.is_set():
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util_samples.append(util.gpu)
                except pynvml.NVMLError:
                    pass
                time_module.sleep(0.1)

            pynvml.nvmlShutdown()
        except Exception as e:
            print(f"GPU monitoring error: {e}")

    monitor_thread = None
    if device.type == "cuda" and pynvml is not None:
        monitor_thread = threading.Thread(target=monitor_gpu)
        monitor_thread.daemon = True
        monitor_thread.start()

    start_time = time_module.time()
    num_pixels = 0
    bytes_per_pixel = get_bytes_per_pixel(dtype) if dtype is not None else 4
    iteration_count = 0
    timed_out = False

    try:

        def timeout_handler(signum, frame):
            timeout_duration = 60 * 3
            raise TimeoutError(
                f"Data loading timed out after {timeout_duration} seconds"
            )

        signal.signal(signal.SIGALRM, timeout_handler)
        timeout_duration = 60 * 3
        signal.alarm(timeout_duration)

        print(f"Processing up to {max_iterations} iterations...")
        for batch in tqdm(dl):
            if device.type != "cpu":
                batch = batch.to(device)

            # Ensure batch has 3 channels for ResNet
            bs, c, h, w = batch.shape
            if c == 1:
                batch = batch.repeat(1, 3, 1, 1)
            elif c > 3:
                batch = batch[:, :3, :, :]

            # Run through model like bayesian search
            with torch.no_grad():
                _ = model(batch)

            num_pixels += bs * c * h * w
            iteration_count += 1

            if iteration_count >= max_iterations:
                break

        signal.alarm(0)
    except TimeoutError as e:
        print(f"Warning: {e}")
        timed_out = True
        signal.alarm(0)
    except Exception as e:
        print(f"Error during data loading: {e}")
        return None, 0
    finally:
        if monitor_thread is not None:
            stop_monitor.set()
            monitor_thread.join(timeout=1.0)

        dataset.close()
        del dl
        if device.type != "cpu":
            torch.cuda.empty_cache()

    end_time = time_module.time()
    elapsed_time = end_time - start_time

    # Calculate throughput even if timed out, as long as some data was processed
    if num_pixels > 0 and elapsed_time > 0:
        mb_per_second = (num_pixels * bytes_per_pixel) / elapsed_time / 1e6
    else:
        mb_per_second = 0

    avg_gpu_util = 0
    if gpu_util_samples:
        avg_gpu_util = sum(gpu_util_samples) / len(gpu_util_samples)
        print(f"Average GPU utilization: {avg_gpu_util:.2f}%")

    print(
        f"Configuration: patch_size={patch_size}, batch_size={batch_size}, num_workers={num_workers}, num_threads={num_threads}"
    )
    print(f"MB per second: {mb_per_second:.2f}")
    print(f"Completed {iteration_count} iterations in {elapsed_time:.2f} seconds")
    if timed_out:
        print(
            f"Note: Configuration timed out but processed {iteration_count} iterations"
        )

    # Clear caches after evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force garbage collection
    gc.collect()

    return mb_per_second, avg_gpu_util


def calculate_statistics(values):
    """
    Calculate the mean and standard deviation for a list of values.
    """
    if len(values) < 2:
        return values[0], 0

    mean = statistics.mean(values)
    stdev = statistics.stdev(values)

    return mean, stdev


def run_grid_search(
    var1,
    var2,
    default_values,
    all_values,
    args,
    device,
    all_urls_by_compression,
    all_image_sizes_by_compression,
    model,
):
    """
    Run a 2D grid search for the specified variables.
    """
    var1_values = all_values[var1]
    var2_values = all_values[var2]

    # Create output directory
    mode = "Local" if args.use_local else "Remote"
    output_dir = os.path.join(args.output_dir, f"{var1}_vs_{var2}", mode)
    os.makedirs(output_dir, exist_ok=True)

    # Create CSV file to write results
    csv_file = f"{output_dir}/results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [var1, var2, "mb_per_second", "mb_per_second_stdev", "gpu_utilization"]
        )

    # Create JSON file for all configuration details
    json_file = f"{output_dir}/all_results.json"
    all_results = []

    # Total number of combinations to evaluate
    total_combinations = len(var1_values) * len(var2_values)
    current_combination = 0

    print(f"Running 2D grid search for {var1} vs {var2}")
    print(f"Total combinations to evaluate: {total_combinations}")

    # Run grid search
    for val1 in var1_values:
        for val2 in var2_values:
            current_combination += 1
            print(
                f"\nEvaluating combination {current_combination}/{total_combinations}"
            )
            print(f"{var1}={val1}, {var2}={val2}")

            # Set up configuration
            config = default_values.copy()
            config[var1] = val1
            config[var2] = val2

            # Construct the data key from compression and block_size
            data_key = f"{config['compression']}_{config['block_size']}"

            # Get URLs for this compression and block size combination
            trial_urls = all_urls_by_compression[data_key]
            trial_image_sizes, trial_block_size, trial_dtype = (
                all_image_sizes_by_compression[data_key]
            )

            # Run each configuration 5 times
            mb_per_second_values = []
            gpu_util_values = []

            num_runs = 5
            for run in range(num_runs):
                print(f"\nRun {run+1}/{num_runs}")
                try:
                    mb_per_second, gpu_util = evaluate_dataloader_config(
                        urls=trial_urls,
                        image_sizes=trial_image_sizes,
                        block_size=trial_block_size,
                        sampler_type=config["sampler_type"],
                        patch_size=config["patch_size"],
                        batch_size=config["batch_size"],
                        num_workers=config["num_workers"],
                        num_threads=config["num_threads"],
                        persistent_workers=config["persistent_workers"],
                        prefetch_factor=config["prefetch_factor"],
                        length=1024,
                        max_iterations=args.training_iters,
                        device=device,
                        model=model,
                        dtype=trial_dtype,
                    )

                    # Handle None return values properly
                    if mb_per_second is not None:
                        mb_per_second_values.append(mb_per_second)
                        gpu_util_values.append(gpu_util)

                except Exception as e:
                    print(f"Error during evaluation run {run+1}: {e}")

                # Force cleanup between runs
                if device.type != "cpu":
                    torch.cuda.empty_cache()

                # Small delay between runs to ensure resources are freed
                time_module.sleep(1)

            # Calculate statistics if we have any successful runs
            if mb_per_second_values:
                # Calculate mean and standard deviation
                mb_mean, mb_stdev = calculate_statistics(mb_per_second_values)
                gpu_mean = (
                    sum(gpu_util_values) / len(gpu_util_values)
                    if gpu_util_values
                    else 0
                )

                print(f"MB/s: {mb_mean:.2f} ± {mb_stdev:.2f}")

                # Write to CSV
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([val1, val2, mb_mean, mb_stdev, gpu_mean])

                # Add to results list
                result = {
                    var1: val1,
                    var2: val2,
                    "mb_per_second": mb_mean,
                    "mb_per_second_stdev": mb_stdev,
                    "mb_per_second_values": mb_per_second_values,
                    "gpu_utilization": gpu_mean,
                    "full_config": config,
                }
                all_results.append(result)
            else:
                # Write failed result to CSV
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([val1, val2, 0, 0, 0])

                # Add failed result to results list
                result = {
                    var1: val1,
                    var2: val2,
                    "mb_per_second": 0,
                    "mb_per_second_stdev": 0,
                    "mb_per_second_values": [],
                    "gpu_utilization": 0,
                    "full_config": config,
                }
                all_results.append(result)

            # Save JSON after each evaluation
            with open(json_file, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nGrid search completed. Results saved to {output_dir}/")
    return all_results


if __name__ == "__main__":
    # Define the parameter space
    all_values = {
        "compression": [
            "deflate_1",
            "deflate_6",
            "deflate_9",
            "jpeg",
            "jpeg2000",
            "lerc",
            "lzw",
            "none",
            "webp",
        ],
        "block_size": [128, 256, 512, 1024],
        "sampler_type": ["random", "block"],
        "patch_size": [128, 256, 512, 1024],
        "num_workers": [1, 2, 4, 8, 16, 32, 64],
        "num_threads": [1, 2, 4, 8, 16, 32],
        "prefetch_factor": [1, 2, 4, 8, 16],
    }

    # Define default values
    default_values = {
        "compression": "none",
        "block_size": 512,
        "sampler_type": "block",
        "patch_size": 256,
        "batch_size": 64,
        "num_workers": 6,
        "num_threads": 1,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }

    # Add command line arguments
    parser = argparse.ArgumentParser(
        description="2D Grid Search for DataLoader Parameters"
    )

    parser.add_argument(
        "--var1",
        type=str,
        required=True,
        choices=list(all_values.keys()),
        help="First variable for grid search",
    )

    parser.add_argument(
        "--var2",
        type=str,
        required=True,
        choices=list(all_values.keys()),
        help="Second variable for grid search",
    )

    parser.add_argument(
        "--use_local",
        action="store_true",
        help="Use local files instead of downloading",
    )

    parser.add_argument(
        "--training-iters",
        type=int,
        default=100,
        help="Maximum number of iterations to evaluate each configuration",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use for data loading (default: 0, -1 for CPU)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="grid_results",
        help="Directory where results should be saved",
    )

    args = parser.parse_args()

    # Validate variables are different
    if args.var1 == args.var2:
        raise ValueError("The two variables must be different")

    # Set up device for GPU data loading
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU for data loading")

    # Initialize NVML for GPU monitoring if using GPU
    if device.type == "cuda" and pynvml is None:
        print(
            "Warning: pynvml not found. Install with 'pip install pynvml' for GPU monitoring."
        )

    # Load pre-trained ResNet18 model once
    print("Loading pre-trained ResNet18 model...")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)

    # Load datasets configuration
    with open("configs/search.yaml", "r") as f:
        datasets_config = yaml.safe_load(f)

    base_files = datasets_config["sentinel-2"]["files"]
    local_root = datasets_config["sentinel-2"]["local"]
    remote_root = datasets_config["sentinel-2"]["remote"]
    if remote_root.endswith("/"):
        remote_root = remote_root[:-1]

    # Load the dataset with all compression types
    all_urls_by_compression = {}
    all_image_sizes_by_compression = {}

    if args.use_local:
        # Process each compression type with different block sizes
        for compression in all_values["compression"]:
            for block_size in all_values["block_size"]:
                config_dir = os.path.join(
                    local_root, compression, f"block_{block_size}"
                )

                # Skip if directory doesn't exist
                if not os.path.exists(config_dir):
                    print(f"Directory not found: {config_dir}")
                    continue

                # Use the files from datasets.yaml
                files = [
                    os.path.join(config_dir, f)
                    for f in base_files
                    if os.path.exists(os.path.join(config_dir, f))
                ]

                if not files:
                    print(f"No files found in {config_dir}")
                    continue

                # Use compression_blocksize as key for consistency with other logic
                config_key = f"{compression}_{block_size}"
                all_urls_by_compression[config_key] = files

                # Get image sizes and block sizes for this configuration
                try:
                    image_sizes, actual_block_size, dtype, bad_files = get_image_sizes(
                        files
                    )
                    if len(bad_files) > 0:
                        files = [file for file in files if file not in bad_files]
                        all_urls_by_compression[config_key] = files

                    all_image_sizes_by_compression[config_key] = (
                        image_sizes,
                        actual_block_size,
                        dtype,
                    )
                    print(
                        f"Loaded {len(files)} files for {config_key}, block size: {actual_block_size}, dtype: {dtype}"
                    )
                except Exception as e:
                    print(f"Error getting image sizes for {config_key}: {e}")
    else:
        # Process each compression type with different block sizes
        for compression in all_values["compression"]:
            for block_size in all_values["block_size"]:
                # Build URLs for this configuration
                urls = [
                    f"{remote_root}/{compression}/block_{block_size}/{filename}{os.getenv('AZURE_SAS')}"
                    for filename in base_files
                ]

                # Use compression_blocksize as key for consistency with other logic
                config_key = f"{compression}_{block_size}"
                all_urls_by_compression[config_key] = urls

                # Get image sizes and block sizes for this configuration
                try:
                    image_sizes, actual_block_size, dtype, bad_files = get_image_sizes(
                        urls
                    )
                    if len(bad_files) > 0:
                        urls = [url for url in urls if url not in bad_files]
                        all_urls_by_compression[config_key] = urls

                    all_image_sizes_by_compression[config_key] = (
                        image_sizes,
                        actual_block_size,
                        dtype,
                    )
                    print(
                        f"Loaded {len(urls)} files for {config_key}, block size: {actual_block_size}, dtype: {dtype}"
                    )
                except Exception as e:
                    print(f"Error getting image sizes for {config_key}: {e}")

    # Update default values based on mode (using best hyperparameters from Bayesian search)
    if args.use_local:
        # Best local mode hyperparameters from Bayesian search
        default_values["compression"] = "none"
        default_values["block_size"] = 512
        default_values["sampler_type"] = "block"  # tiled = YES
        default_values["patch_size"] = 512
        default_values["batch_size"] = 256
        default_values["num_workers"] = 4
        default_values["num_threads"] = 1
        default_values["prefetch_factor"] = 2
    else:
        # Best remote mode hyperparameters from Bayesian search
        default_values["compression"] = "lerc"
        default_values["block_size"] = 512
        default_values["sampler_type"] = "block"  # tiled = YES
        default_values["patch_size"] = 1024
        default_values["batch_size"] = 64
        default_values["num_workers"] = 64
        default_values["num_threads"] = 1
        default_values["prefetch_factor"] = 8

    # Run the grid search
    results = run_grid_search(
        args.var1,
        args.var2,
        default_values,
        all_values,
        args,
        device,
        all_urls_by_compression,
        all_image_sizes_by_compression,
        model,
    )
