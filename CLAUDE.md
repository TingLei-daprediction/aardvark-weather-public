# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aardvark Weather is a deep learning model for weather forecasting, combining multi-modal observational data (satellite, surface stations, reanalysis) with a neural network architecture. The model is trained in stages: an **encoder** processes observations, a **processor** refines the representation, and a **decoder** generates forecasts.

## Common Commands

**Environment** (conda env named `aardvark-env`, Python 3.8, PyTorch 2.3 / CUDA 12.1):
```bash
conda env create -f environment.yml
conda activate aardvark-env
```

**Formatting** — `black==24.4.2` is the configured formatter. Format changed files before broader edits:
```bash
black aardvark scripts
```

**Inference / demos** (the only fully runnable entry points with the shipped sample data):
```bash
jupyter lab notebooks/forecast_demo.ipynb        # full model forecast from sample data
jupyter lab notebooks/e2e_finetune_demo.ipynb    # end-to-end finetuned station forecast
jupyter lab notebooks/data_demo.ipynb            # inspect a data timeslice
```

**Training** — all three core stages run through `aardvark/train_module.py`, selected by `--mode`:
- `--mode assimilation` → train the **encoder** (`--decoder vit_assimilation`)
- `--mode forecast` → pre-train/train the **processor** (`--decoder vit`)
- `--mode downscaling` → train the **decoder**
Processor finetuning is `finetune.py`; end-to-end finetuning is `e2e_train.py`.
Illustrative single-node invocations live in `training/*.sh` (e.g. `train_processor_4u_sfc.sh`,
`train_decoder_4u_sfc.sh`, `e2e_train_4u_sfc.sh`). SLURM multi-GPU launchers are `training/dd.sh`
(2 nodes × 2 GPUs) and `training/dd-2GPU.sh`. **These reference local data pipelines and compute
infrastructure not in the repo — treat them as documentation, not runnable, unless data paths are provided.**

**Data validation** — when changing preprocessing outputs, run the relevant helper under `scripts/`
(e.g. `python scripts/check_obs_memmaps.py ...`, `scripts/run_check_obs_netcdfs.sh`). There is no
formal test runner; validate with the narrowest runnable artifact (notebook, `check_*` script, or a
small shape/construction smoke test).

## Key CLI Parameters (`train_module.py`)

- `--era5_mode`: one of `4u`, `sfc`, `4u_sfc` — selects which ERA5 fields are included and drives channel counts.
- `--time_freq`: `1D` (daily) or `6H`; `--frequency`: lead-time step in hours (6 or 24).
- `--in_channels` / `--out_channels` / `--int_channels`: must match the `era5_mode` and modality set — see `expected_in_channels_assimilation()` before changing.
- `--diff`: predict ERA5 differences (1) vs. raw fields (0).
- Per-instrument channel overrides exist (`--amsua_channels`, `--amsub_channels`, `--iasi_channels`, `--ascat_channels`, `--hirs_channels`); leave `None` to use defaults from `data_shapes.py`.
- DDP: `--master_port`, `--backend` (nccl/gloo), `--world_size`.

## Architecture Overview

The model follows a three-stage training pipeline:

1. **Encoder (ConvCNP)** - Located in `models.py` and `architectures.py`
   - Ingests multi-modal satellite data and surface observations
   - Uses DeepSet networks (`set_convs.py`) to aggregate observations at different spatial locations
   - Outputs a latent representation over a fixed grid (256×128 internal grid)
   - Input modalities: AMSU-A, AMSU-B, HIRS, IASI, ASCAT, ICOADS, HadISD, IGRA

2. **Processor** - Transformer-like module that refines the encoded state
   - Pre-trained on reanalysis data, then finetuned with observations
   - Controlled by parameters like `in_channels`, `out_channels`, `int_channels` that vary by era5_mode

3. **Decoder (UNet)** - Located in `unet_wrap_padding.py`
   - Converts processor output to spatially-gridded weather predictions
   - Can be trained for specific lead times and pressure levels

## File Organization

**Core Model Files:**
- `aardvark/models.py` - Main model classes: `ConvCNPWeather`, `Processor`, `Decoder`
- `aardvark/architectures.py` - Basic building blocks (MLP, positional encodings)
- `aardvark/set_convs.py` - DeepSet convolution layers for observation aggregation
- `aardvark/vit.py` - Vision Transformer components used in processor
- `aardvark/unet_wrap_padding.py` - UNet decoder implementation with padding

**Training & Loss:**
- `aardvark/train_module.py` - Main training loop orchestration with DDP support
- `aardvark/trainer.py` - DDPTrainer class for distributed training
- `aardvark/loss_functions.py` - Weighted RMSE and pressure-weighted losses
- `aardvark/finetune.py` - Processor finetuning code
- `aardvark/e2e_train.py` - End-to-end finetuning of full model
- `aardvark/e2e_model.py` - Wrapper for end-to-end inference

**Data Loading:**
- `aardvark/loader.py` - `WeatherDataset` base class, handles observation and ERA5 loading
- `aardvark/loader_utils_new.py` - Data loading utilities for specific modalities
- `aardvark/data_shapes.py` - Metadata about data shapes and channel counts
- `aardvark/misc_downscaling_functionality.py` - Spatial downscaling utilities

**Utilities:**
- `aardvark/utils.py` - General utilities
- `aardvark/generate_initial_condition_single.py` - Initial condition generation for forecasting

## Key Concepts

**Channels and Shapes:**
- Different satellite instruments have fixed channel counts: AMSU-A (13), AMSU-B (12), HIRS (26)
- Observations are aggregated into (density, value) pairs by DeepSet, doubling channel count
- `expected_in_channels_assimilation()` in `train_module.py` computes expected channel counts given configuration
- The `era_mode` parameter controls which ERA5 fields are included (e.g., `4u` uses 4 upper atmosphere levels)

**Time Frequencies:**
- Data can be loaded at 6-hour (`6H`) or daily (`1D`) resolution
- Controlled by `time_freq` parameter throughout the stack
- Different time frequencies require corresponding data preprocessing

**Lead Times:**
- Distance between initial condition and target forecast time
- Decoder trained separately for different lead times
- End-to-end finetuning can specialize for specific lead times

**DDP Training:**
- Code uses PyTorch DistributedDataParallel for multi-GPU training
- `ddp_setup()` in `train_module.py` initializes process groups with NCCL backend
- Rank and world size must be set correctly for distributed training

## Common Development Tasks

**Understanding Channel Flow:**
- Start in `train_module.py` with `expected_in_channels_assimilation()` to see how channels are computed
- Trace through loader to see which observations contribute which channels
- Check `models.py` ConvCNPWeather init to see how it expects channels

**Adding a New Observation Modality:**
- Add loading code in `loader.py` (model after existing `load_amsua()`, `load_igra()` methods)
- Update channel count in `expected_in_channels_assimilation()`
- Add corresponding DeepSet convolution in `ConvCNPWeather.__init__` (see `self.amsua_setconvs`)
- Update forward pass in `ConvCNPWeather` to include the new modality

**Modifying Loss Functions:**
- Loss functions live in `loss_functions.py` (e.g., `WeightedRmseLoss`, `PressureWeightedRmseLoss`)
- Applied in trainer's step function
- Check which loss is instantiated in `train_module.py` main loop

**Data Format:**
- Observations are PyTorch sparse tensors (COO format) with shape [batch, channels, n_observations]
- Grid data (ERA5) is dense tensors with shape [batch, channels, height, width]
- Heights and widths vary: ERA5 at res=1 uses 256×128 internal grid, sample data at 0.5° resolution

## Important Notes

- **Training scripts are illustrative**: The shell scripts in `training/` and many training commands in Python files reference local data pipelines not included in the repo. The code is provided for transparency of the methodology, not as a fully runnable training pipeline without additional data and infrastructure setup.
- **Sample data is limited**: `data/sample_data_final.pkl` contains a small subset for demonstration via Jupyter notebooks.
- **Grid coordinates are precomputed**: Longitude/latitude grid coordinates are loaded from `.npy` files in `data/grid_lon_lat/` and normalized to [0,1] range for the model.
- **Normalization factors**: Mean and std for prediction targets are stored in `data/norm_factors/` for denormalization during inference.

## Entry Points

**For Forecasting (Inference):**
- See `notebooks/forecast_demo.ipynb` - load complete model and generate predictions
- `e2e_model.py` - wrapper for end-to-end inference with finetuned weights

**For Training:**
- `train_module.py` - main entry point, uses DDP for multi-GPU training
- Expects command-line args for era_mode, lead_time, checkpoint paths, etc.

**For Finetuning:**
- `finetune.py` - processor finetuning against observation data
- `e2e_train.py` - full model end-to-end finetuning

## Dependencies

Pinned in `environment.yml` (conda env `aardvark-env`):
- Python 3.8, PyTorch 2.3.0 + torchvision/torchaudio (CUDA 12.1, cudnn 8.9)
- NumPy 1.24, Pandas 2.0, xarray, scipy for data handling
- `timm` (ViT components), `wandb` (experiment logging), `huggingface-hub`/`safetensors` (weights)
- Geo/data stack: cartopy, eccodes/cfgrib, netcdf4, zarr, h5py for reading source datasets
- `black==24.4.2` for formatting (pip section)

## Git Considerations

The `.gitignore` excludes `__pycache__/`, `*.pyc`, and `*.pyo` files. Large data files (memmaps, netCDF data) and model checkpoints are not tracked; these are managed externally via HuggingFace or local data pipelines.
