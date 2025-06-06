#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import sys
import subprocess
from pathlib import Path

__version__ = "0.1.0"


def get_script_path(script_name):
    """Get the full path to a script in the scripts/experiments directory."""
    script_path = Path(__file__).parent.parent / "scripts" / "experiments" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    return str(script_path)


def version(args):
    """Print version information."""
    print(f"ocogs {__version__}")
    print("Optimized COG Streaming toolkit for Earth Observation data")
    return 0


def bayesian_search(args):
    """Run Bayesian hyperparameter optimization."""
    script_path = get_script_path("bayesian_search.py")
    cmd = [sys.executable, script_path]

    if args.trials:
        cmd.extend(["--trials", str(args.trials)])
    if args.study_name:
        cmd.extend(["--study_name", args.study_name])
    if args.local:
        cmd.append("--local")
    if args.training_iters:
        cmd.extend(["--training-iters", str(args.training_iters)])
    if args.gpu is not None:
        cmd.extend(["--gpu", str(args.gpu)])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])

    return subprocess.run(cmd)


def grid_search(args):
    """Run 2D grid search for parameter optimization."""
    script_path = get_script_path("grid_search.py")
    cmd = [sys.executable, script_path]

    cmd.extend(["--var1", args.var1])
    cmd.extend(["--var2", args.var2])
    if args.use_local:
        cmd.append("--use_local")
    if args.training_iters:
        cmd.extend(["--training-iters", str(args.training_iters)])
    if args.gpu is not None:
        cmd.extend(["--gpu", str(args.gpu)])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])

    return subprocess.run(cmd)


def train(args):
    """Train segmentation model with optimized data loading."""
    script_path = get_script_path("train.py")
    cmd = [sys.executable, script_path]

    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    if args.max_time:
        cmd.extend(["--max_time", str(args.max_time)])
    if args.gpu is not None:
        cmd.extend(["--gpu", str(args.gpu)])
    if args.val_split:
        cmd.extend(["--val_split", str(args.val_split)])

    return subprocess.run(cmd)


def main():
    """Main entry point for the ocogs command."""
    parser = argparse.ArgumentParser(
        description="Optimized COG Streaming toolkit for Earth Observation data",
        prog="ocogs",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Version subcommand
    version_parser = subparsers.add_parser("version", help="Show version information")
    version_parser.set_defaults(func=version)

    # Bayesian search subcommand
    bayesian_parser = subparsers.add_parser(
        "bayesian_search", help="Run Bayesian hyperparameter optimization"
    )
    bayesian_parser.add_argument(
        "--trials", type=int, default=100, help="Number of optimization trials"
    )
    bayesian_parser.add_argument(
        "--study_name",
        type=str,
        default="throughput_optimization",
        help="Base name for the study",
    )
    bayesian_parser.add_argument(
        "--local", action="store_true", help="Use local files instead of remote"
    )
    bayesian_parser.add_argument(
        "--training-iters",
        type=int,
        default=100,
        help="Maximum number of iterations to evaluate each configuration",
    )
    bayesian_parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use for data loading (default: 0, -1 for CPU)",
    )
    bayesian_parser.add_argument(
        "--output-dir",
        type=str,
        default="optuna_studies",
        help="Directory where results should be saved",
    )
    bayesian_parser.set_defaults(func=bayesian_search)

    # Grid search subcommand
    grid_parser = subparsers.add_parser(
        "grid_search", help="Run 2D grid search for parameter optimization"
    )
    grid_parser.add_argument(
        "--var1",
        type=str,
        required=True,
        choices=[
            "compression",
            "block_size",
            "sampler_type",
            "patch_size",
            "num_workers",
            "num_threads",
            "prefetch_factor",
        ],
        help="First variable for grid search",
    )
    grid_parser.add_argument(
        "--var2",
        type=str,
        required=True,
        choices=[
            "compression",
            "block_size",
            "sampler_type",
            "patch_size",
            "num_workers",
            "num_threads",
            "prefetch_factor",
        ],
        help="Second variable for grid search",
    )
    grid_parser.add_argument(
        "--use_local", action="store_true", help="Use local files instead of remote"
    )
    grid_parser.add_argument(
        "--training-iters",
        type=int,
        default=100,
        help="Maximum number of iterations to evaluate each configuration",
    )
    grid_parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use for data loading (default: 0, -1 for CPU)",
    )
    grid_parser.add_argument(
        "--output-dir",
        type=str,
        default="grid_results",
        help="Directory where results should be saved",
    )
    grid_parser.set_defaults(func=grid_search)

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train", help="Train segmentation model with optimized data loading"
    )
    train_parser.add_argument(
        "--dataset",
        type=str,
        default="vaihingen",
        choices=["vaihingen", "potsdam", "dfc-22"],
        help="Dataset to use for training",
    )
    train_parser.add_argument(
        "--max_time", type=int, default=600, help="Maximum training time in seconds"
    )
    train_parser.add_argument("--gpu", type=int, default=0, help="GPU ID (-1 for CPU)")
    train_parser.add_argument(
        "--val_split", type=float, default=0.25, help="Validation split ratio"
    )
    train_parser.set_defaults(func=train)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute the selected command
    try:
        result = args.func(args)
        return result.returncode if hasattr(result, "returncode") else result
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit(main())
