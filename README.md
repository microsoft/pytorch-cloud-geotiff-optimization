# Optimized COG Streaming

<div style="max-width: 800px">
  <img src="https://i.postimg.cc/Bb45W5Z8/news.png" alt="Optimized COG Streaming" style="width: 100%">
</div>
<div style="margin-bottom: 1rem;"></div>

A toolkit for optimizing Earth Observation COG streaming in PyTorch. Achieves 20x throughput and 90% GPU utilization through optimized data loading and compression.

## Quick Start

### 1. Environment Setup

```bash
mamba create -n ocog python=3.13.3
conda activate ocog
pip install -r requirements.txt
pip install -e .
```

### 2. Configuration

Configure dataset paths in the config files:

**For Hyperparameter Search** (`configs/search.yaml`):
```yaml
sentinel-2:
  local: ""  # LOCAL DEFAULT PATH
  remote: ""  # REMOTE DEFAULT URL
  files:
    - S2A_MSIL2A_20241029T074031_R092_T36MZD_20241029T111159.tif
    - S2A_MSIL2A_20241029T074031_R092_T37MBV_20241029T111159.tif
    - S2A_MSIL2A_20241228T074331_R092_T36MZE_20241228T110550.tif
    - S2B_MSIL2A_20241014T073759_R092_T37MBT_20241014T102258.tif
    - S2B_MSIL2A_20241014T073759_R092_T37MBU_20241014T102258.tif
    - S2B_MSIL2A_20241017T074829_R135_T36MZC_20241017T100313.tif
    - S2B_MSIL2A_20241024T073909_R092_T36MZC_20241024T095754.tif
    - S2B_MSIL2A_20241213T074229_R092_T37MCU_20241213T094330.tif
    - S2B_MSIL2A_20241216T075239_R135_T36MZD_20241216T094931.tif
    - S2B_MSIL2A_20241226T075239_R135_T36MZD_20241226T094735.tif
```

**For Training** (`configs/train.yaml`):
```yaml
vaihingen:
  local-default: "" # LOCAL DEFAULT PATH
  local-optimal: "" # LOCAL OPTIMAL PATH
  remote-default: "" # REMOTE DEFAULT URL
  remote-optimal: "" # REMOTE OPTIMAL URL
  classes: 6
potsdam:
  local-default: "" # LOCAL DEFAULT PATH
  local-optimal: "" # LOCAL OPTIMAL PATH
  remote-default: "" # REMOTE DEFAULT URL
  remote-optimal: "" # REMOTE OPTIMAL URL
  classes: 6
dfc-22:
  local-default: "" # LOCAL DEFAULT PATH
  local-optimal: "" # LOCAL OPTIMAL PATH
  remote-default: "" # REMOTE DEFAULT URL
  remote-optimal: "" # REMOTE OPTIMAL URL
  classes: 15
```

For remote access, create `.env` file:
```
AZURE_SAS=?your_sas_token
```

## Usage

The `ocogs` command provides a unified interface for all toolkit functionality:

```bash
# Show available commands
ocogs --help

# Show version
ocogs --version

# Get help for specific commands
ocogs bayesian_search --help
ocogs grid_search --help
ocogs train --help
```

### Hyperparameter Optimization

**Bayesian Search** - Find optimal configurations:
```bash
# Local optimization
ocogs bayesian_search --trials 50 --local --training-iters 100

# Remote optimization  
ocogs bayesian_search --trials 100 --training-iters 200
```

**Grid Search** - Compare specific parameters:
```bash
ocogs grid_search \
    --var1 compression --var2 block_size \
    --use_local --training-iters 100
```

### Training

Train segmentation models with optimized data loading:
```bash
ocogs train \
    --dataset vaihingen \
    --max_time 600 \
    --gpu 0
```

## Dataset Preparation

### Download Test Data (Sentinel-2)

```bash
python scripts/acquire/sentinel_2.py --output_dir /path/to/sentinel2
```

Configure the sentinel-2 paths in `configs/search.yaml` by filling in the `local` and `remote` values.

### Benchmark Datasets

**1. Download datasets:**
- **DFC-22**: [IEEE Dataport](https://ieee-dataport.org/competitions/data-fusion-contest-2022-dfc2022#files) (`labeled_train.zip`)
- **Vaihingen**: [Download link](https://seafile.projekt.uni-hannover.de/f/6a06a837b1f349cfa749/) (*Contact organizers for password*)
- **Potsdam**: [Download link](https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/) (*Contact organizers for password*)

**2. Extract and organize:**
```bash
# Extract archives
cd /path/to/dataset
scripts/acquire/extract.sh .

# Prepare dataset structure
python scripts/process/prepare_vaihingen.py --path /path/to/vaihingen/raw
python scripts/process/prepare_potsdam.py --path /path/to/potsdam/raw  
python scripts/process/prepare_dfc22.py --path /path/to/dfc-22/raw
```

**3. Create optimized versions:**
```bash
python scripts/process/create_benchmark_datasets.py --raw_path /path/to/dataset/raw
```

This creates three versions:
- `default/`: DEFLATE compression, 512x512 tiles
- `optimal_local/`: No compression for local I/O
- `optimal_remote/`: LERC_ZSTD compression for cloud streaming

## Configuration Reference

### Processing Parameters (`configs/configs.yaml`)

```yaml
default:
  num_workers: 4      # DataLoader workers
  num_threads: 1      # Per-sample threads  
  prefetch_factor: 2  # Batches to prefetch
  sampler_type: random # random|block
  patch_size: 256     # Training patch size

local-optimal:
  sampler_type: block # Block-aligned sampling
  
remote-optimal:
  num_workers: 64     # More workers for network I/O
  prefetch_factor: 8  # Higher prefetch for latency
```

### Dataset Paths (`configs/train.yaml`)

Each dataset supports four configurations:
- `local-default`: Local files, standard compression
- `local-optimal`: Local files, optimized for speed
- `remote-default`: Remote files, standard compression
- `remote-optimal`: Remote files, optimized for streaming


## Data Attribution

This repository uses the following datasets:

### DFC-22 Dataset
The Data Fusion Contest 2022 (DFC-22) dataset is provided by IEEE GRSS, Université Bretagne-Sud, ONERA, and ESA Φ-lab. 

If you use this data, please cite:
1. 2022 IEEE GRSS Data Fusion Contest. Online: https://www.grss-ieee.org/technical-committees/image-analysis-and-data-fusion/
2. Castillo-Navarro, J., Le Saux, B., Boulch, A. and Lefèvre, S.. Semi-supervised semantic segmentation in Earth Observation: the MiniFrance suite, dataset analysis and multi-task network study. Mach Learn (2021). https://doi.org/10.1007/s10994-020-05943-y
3. Hänsch, R.; Persello, C.; Vivone, G.; Castillo Navarro, J.; Boulch, A.; Lefèvre, S.; Le Saux, B. : 2022 IEEE GRSS Data Fusion Contest: Semi-Supervised Learning [Technical Committees], IEEE Geoscience and Remote Sensing Magazine, March 2022

#### Usage conditions
The data are provided for research purposes and must be identified as "grss_dfc_2022" in any scientific publication.

### ISPRS Vaihingen Dataset
The Vaihingen dataset is part of the ISPRS 2D Semantic Labeling Benchmark. If you use this data, please cite:
- Cramer, M., 2010. The DGPF test on digital aerial camera evaluation – overview and test design. Photogrammetrie – Fernerkundung – Geoinformation 2(2010):73-82.

And include the following acknowledgement:
"The Vaihingen data set was provided by the German Society for Photogrammetry, Remote Sensing and Geoinformation (DGPF) [Cramer, 2010]: http://www.ifp.uni-stuttgart.de/dgpf/DKEP-Allg.html."

#### Usage conditions
1. The data must not be used for other than research purposes. Any other use is prohibited.
2. The data must not be distributed to third parties. Any person interested in the data may obtain them via ISPRS WG III/4.
3. The German Association of Photogrammetry, Remote Sensing and GeoInformation (DGPF) should be informed about any published papers whose results are based on the Vaihingen test data.

### ISPRS Potsdam Dataset
The Potsdam dataset is part of the ISPRS 2D Semantic Labeling Benchmark. If you use this data, please cite:
- ISPRS 2D Semantic Labeling - Potsdam: https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx

The dataset consists of 38 patches of true orthophotos (TOP) and digital surface models (DSM) with a ground sampling distance of 5 cm. The data is provided in different channel compositions (IRRG, RGB, RGBIR) as TIFF files.

#### Usage conditions
Based on similar ISPRS test datasets, this data is intended for research purposes only and should not be redistributed. Researchers interested in the data should obtain it directly from the ISPRS benchmark website.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## License

This project is licensed under the [MIT License](LICENSE).