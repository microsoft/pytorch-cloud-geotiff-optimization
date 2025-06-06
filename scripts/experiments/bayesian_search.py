# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import signal
import argparse
import os
import pickle
import time as time_module
import threading

import optuna
from optuna.samplers import TPESampler
from optuna.exceptions import TrialPruned
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from tqdm import tqdm
import optuna.visualization as vis
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
    compression,
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
    import gc

    gc.collect()

    print("Configuration:")
    print(f"Sampler type: {sampler_type}")
    print(f"Compression: {compression}")
    print(f"Patch size: {patch_size}")
    print(f"Batch size: {batch_size}")
    print(f"Number of workers: {num_workers}")
    print(f"Number of threads: {num_threads}")
    print(f"Persistent workers: {persistent_workers}")
    print(f"Prefetch factor: {prefetch_factor}")
    print(f"Max iterations: {max_iterations}")
    print(f"Device: {device}")

    # Determine block size for sampler
    t_block_size = None
    if sampler_type == "block":
        t_block_size = block_size

    # Determine collate function
    t_collate_fn = None
    if num_threads > 1:
        t_collate_fn = collate_fn

    # Create the dataset
    dataset = TileDataset(
        urls,
        num_threads=num_threads,
    )

    # Create the sampler
    sampler = RandomBlockSampler(
        urls,
        image_sizes,
        patch_size=patch_size,
        length=length,
        block_size=t_block_size,
        num_threads=num_threads,
    )

    print(f"Creating DataLoader with {num_workers} workers...")
    dl = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=t_collate_fn,
    )

    # GPU utilization monitoring setup
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

    # Start GPU monitoring if using GPU
    monitor_thread = None
    if device.type == "cuda" and pynvml is not None:
        monitor_thread = threading.Thread(target=monitor_gpu)
        monitor_thread.daemon = True
        monitor_thread.start()

    # Time the loading of batches
    start_time = time_module.time()
    num_pixels = 0
    bytes_per_pixel = get_bytes_per_pixel(dtype) if dtype is not None else 4
    iteration_count = 0
    timed_out = False

    try:
        # Set timeout for the entire operation
        def timeout_handler(signum, frame):
            timeout_duration = 60 * 3
            raise TimeoutError(
                f"Data loading timed out after {timeout_duration} seconds"
            )

        signal.signal(signal.SIGALRM, timeout_handler)
        timeout_duration = 60 * 3
        signal.alarm(timeout_duration)

        # Load batches until max_iterations is reached
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

            with torch.no_grad():
                _ = model(batch)

            num_pixels += bs * c * h * w
            iteration_count += 1

            # Stop after reaching max_iterations
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
        # Stop GPU monitoring
        if monitor_thread is not None:
            stop_monitor.set()
            monitor_thread.join(timeout=1.0)

    end_time = time_module.time()
    elapsed_time = end_time - start_time

    # Calculate throughput even if timed out, as long as some data was processed
    if num_pixels > 0 and elapsed_time > 0:
        mb_per_second = (num_pixels * bytes_per_pixel) / elapsed_time / 1e6
    else:
        mb_per_second = 0

    # Calculate average GPU utilization if available
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
    import gc

    gc.collect()

    return mb_per_second, avg_gpu_util


def objective(trial):
    """
    Optuna objective function to maximize megabytes per second.
    """
    # Define the hyperparameters to optimize with reasonable ranges
    compression_options = [
        "deflate_1",
        "deflate_6",
        "deflate_9",
        "jpeg",
        "jpeg2000",
        "lerc",
        "lzw",
        "none",
        "webp",
    ]
    sampler_options = ["random", "block"]
    patch_options = [128, 256, 512, 1024]
    worker_options = [1, 2, 4, 8, 16, 32, 64]
    thread_options = [1, 2, 4, 8, 16]
    prefetch_options = [1, 2, 4, 8, 16]
    block_size_options = [128, 256, 512, 1024]

    # Use a large length to ensure dataloader never runs out of data
    length = 100_000

    compression = trial.suggest_categorical("compression", compression_options)
    block_size_param = trial.suggest_categorical("block_size", block_size_options)
    sampler_type = trial.suggest_categorical("sampler_type", sampler_options)
    patch_size = trial.suggest_categorical("patch_size", patch_options)
    batch_size = 256 if args.local else 64
    num_workers = trial.suggest_categorical("num_workers", worker_options)
    num_threads = trial.suggest_categorical("num_threads", thread_options)
    # Always use persistent workers
    persistent_workers = True
    prefetch_factor = trial.suggest_categorical("prefetch_factor", prefetch_options)

    # Get URLs for this compression type and block size
    compression_block_key = f"{compression}_{block_size_param}"
    if compression_block_key not in all_urls_by_config:
        # Trial with this combination does not have data
        raise TrialPruned(
            f"No data for compression={compression}, block_size={block_size_param}"
        )

    trial_urls = all_urls_by_config[compression_block_key]
    trial_image_sizes, trial_block_size, trial_dtype = all_image_sizes_by_config[
        compression_block_key
    ]

    try:
        mb_per_second, gpu_util = evaluate_dataloader_config(
            urls=trial_urls,
            image_sizes=trial_image_sizes,
            block_size=trial_block_size,
            sampler_type=sampler_type,
            patch_size=patch_size,
            batch_size=batch_size,
            num_workers=num_workers,
            num_threads=num_threads,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            length=length,
            compression=compression,
            max_iterations=args.training_iters,
            device=device,
            model=model,
            dtype=trial_dtype,
        )

        # Store GPU utilization as additional user attribute
        trial.set_user_attr("gpu_utilization", gpu_util)

        if mb_per_second is None:
            raise TrialPruned("Configuration failed with error")

        return mb_per_second
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 0.0


if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser(
        description="DataLoader Hyperparameter Optimization for Throughput"
    )

    # Study configuration
    parser.add_argument(
        "--trials", type=int, default=100, help="Number of optimization trials"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="throughput_optimization",
        help="Base name for the study",
    )

    # Data source configuration
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local files instead of downloading",
    )

    # Training/evaluation configuration
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
        default="optuna_studies",
        help="Directory where results should be saved",
    )

    args = parser.parse_args()

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

    # Load the dataset
    if args.local:
        # Use local files
        all_urls_by_config = {}
        all_image_sizes_by_config = {}

        # Define compression options and block sizes to search
        compression_options = [
            "deflate_1",
            "deflate_6",
            "deflate_9",
            "jpeg",
            "jpeg2000",
            "lerc",
            "lzw",
            "none",
            "webp",
        ]
        block_size_options = [128, 256, 512, 1024]

        # Process each combination
        for compression in compression_options:
            for block_size in block_size_options:
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

                config_key = f"{compression}_{block_size}"
                all_urls_by_config[config_key] = files

                # Get image sizes and block sizes for this configuration
                try:
                    image_sizes, actual_block_size, dtype, bad_files = get_image_sizes(
                        files
                    )
                    if len(bad_files) > 0:
                        files = [file for file in files if file not in bad_files]
                        all_urls_by_config[config_key] = files

                    all_image_sizes_by_config[config_key] = (
                        image_sizes,
                        actual_block_size,
                        dtype,
                    )
                    print(
                        f"Inspected {len(files)} image sizes for {config_key}, block size: {actual_block_size}, dtype: {dtype}"
                    )
                except Exception as e:
                    print(f"Error getting image sizes for {config_key}: {e}")
                    all_image_sizes_by_config[config_key] = ([], None, None)
    else:
        # Use remote files
        all_urls_by_config = {}
        all_image_sizes_by_config = {}

        # Define compression options and block sizes to search
        compression_options = [
            "deflate_1",
            "deflate_6",
            "deflate_9",
            "jpeg",
            "jpeg2000",
            "lerc",
            "lzw",
            "none",
            "webp",
        ]
        block_size_options = [128, 256, 512, 1024]

        # Process each combination
        for compression in compression_options:
            for block_size in block_size_options:
                # Build URLs for this configuration
                urls = [
                    f"{remote_root}/{compression}/block_{block_size}/{filename}{os.getenv('AZURE_SAS')}"
                    for filename in base_files
                ]
                config_key = f"{compression}_{block_size}"
                all_urls_by_config[config_key] = urls

                # Get image sizes and block sizes for this configuration
                try:
                    image_sizes, actual_block_size, dtype, bad_files = get_image_sizes(
                        urls
                    )
                    if len(bad_files) > 0:
                        urls = [url for url in urls if url not in bad_files]
                        all_urls_by_config[config_key] = urls

                    all_image_sizes_by_config[config_key] = (
                        image_sizes,
                        actual_block_size,
                        dtype,
                    )
                    print(
                        f"Loaded {len(urls)} files for {config_key}, block size: {actual_block_size}, dtype: {dtype}"
                    )
                except Exception as e:
                    print(f"Error getting image sizes for {config_key}: {e}")
                    # Use default values if we can't get sizes
                    all_image_sizes_by_config[config_key] = ([], None, None)

    # Create output directory for study results
    local_remote = "Local" if args.local else "Remote"
    output_dir = os.path.join(args.output_dir, args.study_name, local_remote)
    os.makedirs(output_dir, exist_ok=True)

    # Create the study with the TPE sampler (Bayesian optimization)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=TPESampler(),
        load_if_exists=True,
    )

    # Run the optimization
    print(f"Starting optimization with {args.trials} trials...")
    print(
        f"Using {'local' if args.local else 'remote'} files with compression and block size as search parameters"
    )
    study.optimize(objective, n_trials=args.trials)

    # Print the best parameters
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (MB per second): {trial.value:.2f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the study object for later analysis
    with open(f"{output_dir}/study.pkl", "wb") as f:
        pickle.dump(study, f)

    # Save the results as a CSV for easy analysis
    df = study.trials_dataframe()
    df.to_csv(f"{output_dir}/results.csv", index=False)

    # Generate visualizations if plotly is available
    try:
        # Optimization history plot
        fig = vis.plot_optimization_history(study)
        fig.write_image(f"{output_dir}/optimization_history.png")

        # Parameter importance plot
        fig = vis.plot_param_importances(study)
        fig.write_image(f"{output_dir}/param_importances.png")

        # Slice plot
        fig = vis.plot_slice(study)
        fig.write_image(f"{output_dir}/slice.png")

        # Contour plot
        fig = vis.plot_contour(study)
        fig.write_image(f"{output_dir}/contour.png")
    except ImportError:
        print("Plotly is not available. Skipping visualizations.")

    # Save the best configuration
    best_config = {
        "compression": trial.params["compression"],
        "block_size": trial.params["block_size"],
        "sampler_type": trial.params["sampler_type"],
        "patch_size": trial.params["patch_size"],
        "batch_size": 256 if args.local else 64,
        "num_workers": trial.params["num_workers"],
        "num_threads": trial.params["num_threads"],
        "persistent_workers": True,
        "prefetch_factor": trial.params["prefetch_factor"],
        "mb_per_second": trial.value,
        "gpu_utilization": trial.user_attrs.get("gpu_utilization", 0),
    }

    with open(f"{output_dir}/{args.study_name}_best_config.pkl", "wb") as f:
        pickle.dump(best_config, f)

    # Save GPU utilization data for all trials
    gpu_util_df = pd.DataFrame(
        [
            {
                **t.params,
                "mb_per_second": t.value,
                "gpu_utilization": t.user_attrs.get("gpu_utilization", 0),
            }
            for t in study.trials
        ]
    )
    gpu_util_df.to_csv(
        f"{output_dir}/{args.study_name}_gpu_utilization.csv", index=False
    )

    print(f"\nAll results saved to {output_dir}/")
