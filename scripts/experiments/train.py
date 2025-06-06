# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import json
from pathlib import Path
import time
from urllib.parse import urlparse
import torch
import numpy as np
import segmentation_models_pytorch as smp
import threading
import warnings
from dotenv import load_dotenv
import pynvml
from tqdm import tqdm
from optimized_cog_streaming.config import TrainingConfig
from optimized_cog_streaming.datamodules import SegmentationDataModule
from azure.storage.blob import ContainerClient

warnings.filterwarnings(
    "ignore", message="Default grid_sample and affine_grid behavior has changed"
)

load_dotenv()


def load_configs(datasets_path, configs_path):
    """Load training configurations and return backward-compatible structure."""
    config = TrainingConfig.load_from_yaml(datasets_path, configs_path)
    return (
        config.get_config_paths(),
        config.get_config_params(),
        config.get_dataset_classes(),
    )


datasets_path = Path("configs/train.yaml").absolute()
configs_path = Path("configs/configs.yaml").absolute()
CONFIG_PATHS, CONFIG_PARAMS, DATASET_CLASSES = load_configs(
    str(datasets_path), str(configs_path)
)


def calculate_iou(pred, target, num_classes):
    pred = pred.argmax(dim=1)
    ious = []

    start_class = 0

    for cls in range(start_class, num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        iou = (intersection / union).item() if union > 0 else 0.0
        ious.append(iou)
    return sum(ious) / num_classes


def train_and_evaluate(
    config_name,
    train_filenames,
    val_filenames,
    batch_size,
    max_time,
    device,
    dataset,
    num_classes,
):
    print(f"\nRunning experiment with configuration: {config_name}")

    datamodule = SegmentationDataModule(
        config_name=config_name,
        config_paths=CONFIG_PATHS,
        config_params=CONFIG_PARAMS,
        train_filenames=train_filenames,
        val_filenames=val_filenames,
        batch_size=batch_size,
        device=device,
        dataset=dataset,
    )

    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_batch = datamodule.get_val_batch()

    val_images, val_masks = val_batch

    if num_classes == 6:  # Vaihingen/Potsdam
        val_masks = val_masks - 1
        val_masks[val_masks == -1] = 5
    elif dataset == "dfc-22":
        val_masks = val_masks - 1

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=-1 if dataset == "dfc-22" else -100
    )

    results = {
        "timestamps": [],
        "val_iou": [],
        "config": config_name,
        "params": CONFIG_PARAMS[config_name],
    }

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
                time.sleep(1)

            pynvml.nvmlShutdown()
        except Exception as e:
            print(f"GPU monitoring error: {e}")

    monitor_thread = None

    start_time = time.time()
    iterations = 0
    validation_interval = 5

    print(f"Starting training with {config_name}...")

    try:
        while time.time() - start_time < max_time:
            for images, masks in tqdm(
                train_loader, desc="Training", total=len(train_loader)
            ):

                images = images.to(device) / 255.0
                masks = masks.to(device).squeeze(1)

                if num_classes == 6:
                    masks = masks - 1
                    masks[masks == -1] = 5
                elif dataset == "dfc-22":
                    masks = masks - 1

                if iterations == 0 and device.type == "cuda" and pynvml is not None:
                    monitor_thread = threading.Thread(target=monitor_gpu)
                    monitor_thread.daemon = True
                    monitor_thread.start()

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, masks)

                loss.backward()
                optimizer.step()

                iterations += 1

                if iterations % validation_interval == 0:

                    current_time = time.time() - start_time

                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(val_images)
                        val_iou = calculate_iou(val_outputs, val_masks, num_classes)

                    results["timestamps"].append(current_time)
                    results["val_iou"].append(val_iou)
                    print(f"Validation IoU: {val_iou:.4f}")

                    model.train()

                if time.time() - start_time >= max_time:
                    break
    finally:
        if monitor_thread is not None:
            stop_monitor.set()
            monitor_thread.join(timeout=1.0)

    # Calculate average GPU utilization
    avg_gpu_util = 0
    if gpu_util_samples:
        avg_gpu_util = sum(gpu_util_samples) / len(gpu_util_samples)
        print(f"Average GPU utilization: {avg_gpu_util:.2f}%")

    results["avg_gpu_utilization"] = avg_gpu_util
    results["gpu_utilization_samples"] = gpu_util_samples

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train segmentation model and track IoU over time"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="vaihingen",
        choices=["vaihingen", "potsdam", "dfc-22"],
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--max_time", type=int, default=600, help="Maximum training time in seconds"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (-1 for CPU)")
    parser.add_argument(
        "--val_split", type=float, default=0.25, help="Validation split ratio"
    )

    args = parser.parse_args()

    results_dir = f"results_train_{args.dataset}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to {results_dir}")

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU for training")

    local_path = CONFIG_PATHS[args.dataset]["local-default"]
    if local_path:
        all_image_files = list(set([e.name for e in Path(local_path).glob("**/*.tif")]))
    else:
        remote_path = CONFIG_PATHS[args.dataset]["remote-default"]
        if not remote_path:
            raise ValueError(
                f"No local or remote path found for dataset {args.dataset}"
            )
        sas_token = os.getenv("AZURE_SAS")
        if not sas_token:
            raise ValueError("AZURE_SAS environment variable not set")
        parsed_url = urlparse(remote_path)
        account_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        path_parts = parsed_url.path.strip("/").split("/")
        container_name = path_parts[0]
        prefix = "/".join(path_parts[1:]) + "/" if len(path_parts) > 1 else ""
        container = ContainerClient(
            account_url=account_url, container_name=container_name, credential=sas_token
        )
        all_image_files = list(
            set(
                [
                    blob.name.split("/")[-1]
                    for blob in container.list_blobs(name_starts_with=prefix)
                ]
            )
        )

    print(f"Using {len(all_image_files)} predefined images")

    np.random.seed(42)
    indices = np.random.permutation(len(all_image_files))
    val_split = int(args.val_split * len(all_image_files))
    val_indices = indices[:val_split]
    train_indices = indices[val_split:]

    train_filenames = [all_image_files[i] for i in train_indices]
    val_filenames = [all_image_files[i] for i in val_indices]

    print(
        f"Using {len(train_filenames)} files for training and {len(val_filenames)} for validation"
    )

    configs = ["remote-optimal", "remote-default", "local-default", "local-optimal"]
    all_results = {}

    for config_name in configs:

        batch_size = 128 if config_name.startswith("local") else 64
        print(f"Using batch size {batch_size} for {config_name}")

        num_classes = DATASET_CLASSES[args.dataset]
        print(f"Using {num_classes} classes for dataset {args.dataset}")

        results = train_and_evaluate(
            config_name=config_name,
            train_filenames=train_filenames,
            val_filenames=val_filenames,
            batch_size=batch_size,
            max_time=args.max_time,
            device=device,
            dataset=args.dataset,
            num_classes=num_classes,
        )

        all_results[config_name] = results

        output_file = os.path.join(results_dir, f"{config_name}.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results for {config_name} saved to {output_file}")

    combined_output_file = os.path.join(results_dir, "all_results.json")
    with open(combined_output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"All experiments completed. Final results saved to {combined_output_file}")
