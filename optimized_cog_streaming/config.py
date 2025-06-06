# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Optional, Dict
from pydantic import BaseModel
import yaml


class DatasetInfo(BaseModel):
    """Configuration for a single dataset."""

    classes: int
    local_default: Optional[str] = None
    local_optimal: Optional[str] = None
    remote_default: Optional[str] = None
    remote_optimal: Optional[str] = None

    class Config:
        populate_by_name = True
        alias_generator = lambda field_name: field_name.replace("_", "-")


class ProcessingParams(BaseModel):
    """Processing parameters for training configurations."""

    num_workers: int
    num_threads: int
    prefetch_factor: int
    sampler_type: str
    patch_size: int


class TrainingConfig(BaseModel):
    """Training configuration that loads from YAML files."""

    datasets: Dict[str, DatasetInfo]
    processing: Dict[str, ProcessingParams]

    @classmethod
    def load_from_yaml(
        cls, datasets_path: str = "datasets.yaml", configs_path: str = "configs.yaml"
    ):
        """Load configuration from YAML files."""
        with open(datasets_path, "r") as f:
            dataset_config = yaml.safe_load(f)

        with open(configs_path, "r") as f:
            processing_config = yaml.safe_load(f)

        # Parse datasets
        datasets = {}
        for name, info in dataset_config.items():
            datasets[name] = DatasetInfo(**info)

        # Parse processing configs with special handling for 'default'
        processing = {}
        for name, params in processing_config.items():
            params_with_flags = params.copy()

            if name == "default":
                # Create both local-default and remote-default from default
                processing["local-default"] = ProcessingParams(**params_with_flags)
                remote_params = params_with_flags.copy()
                processing["remote-default"] = ProcessingParams(**remote_params)
            else:
                processing[name] = ProcessingParams(**params_with_flags)

        return cls(datasets=datasets, processing=processing)

    def get_config_paths(self) -> Dict[str, Dict[str, str]]:
        """Get CONFIG_PATHS structure for backward compatibility."""
        config_paths = {}
        for dataset_name, dataset_info in self.datasets.items():
            config_paths[dataset_name] = {}
            for config_type in [
                "local-default",
                "local-optimal",
                "remote-default",
                "remote-optimal",
            ]:
                attr_name = config_type.replace("-", "_")
                if hasattr(dataset_info, attr_name):
                    value = getattr(dataset_info, attr_name)
                    if value is not None:
                        config_paths[dataset_name][config_type] = value
        return config_paths

    def get_config_params(self) -> Dict[str, Dict]:
        """Get CONFIG_PARAMS structure for backward compatibility."""
        return {name: params.model_dump() for name, params in self.processing.items()}

    def get_dataset_classes(self) -> Dict[str, int]:
        """Get DATASET_CLASSES structure for backward compatibility."""
        return {name: info.classes for name, info in self.datasets.items()}
