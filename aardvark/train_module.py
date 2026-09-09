"""
NB: this script is for illustration purposes only and is not runnable as our
full dataset is not provided as part of the submission, due to size constraints.
Many of the relevant paths to the data have been thus replaced by dummy paths.
"""

import os
import sys
import pickle
import argparse
import random
import math
import shutil

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group


from trainer import DDPTrainer
from loss_functions import WeightedRmseLoss, PressureWeightedRmseLoss, RmseLoss
from misc_downscaling_functionality import ConvCNPWeatherOnToOff, DownscalingRmseLoss
from loader import *
from models import *
from unet_wrap_padding import *
from grid_config import (
    load_grid_config,
    set_active_config,
    DEFAULT_CONFIG_PATH,
    assert_grid_files_consistent,
)
from month_manifest import format_months, read_month_manifest


sys.path.append("../npw/data")
torch.set_float32_matmul_precision("medium")


def ddp_setup(rank, world_size, master_port, backend):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    init_process_group(backend=backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)


def start_date(name):
    if name == "train":
        return "2007-01-02"
    elif name == "val":
        return "2019-01-01"
    elif name == "test":
        return "2018-01-01"
    else:
        raise Exception(f"Unrecognised split name {name}")


def end_date(name):
    if name == "train":
        return "2017-12-31"
    elif name == "val":
        return "2019-11-01"
    elif name == "test":
        return "2018-12-21"
    else:
        raise Exception(f"Unrecognised split name {name}")


def expected_in_channels_assimilation(
    amsua_channels,
    amsub_channels,
    iasi_channels,
    ascat_channels,
    hirs_channels,
    disable_igra,
    two_frames,
    climatology_channels=24,
    obs_set="all",
):
    # convDeepSet encoders output density + value per channel (2x).
    aux_total = 4 + climatology_channels + 5  # elev vars + climatology + aux time channels
    if obs_set == "rtma_surface":
        # RTMA Phase 1: HadISD surface obs (5 vars x 2) + aux only; single frame.
        return 2 * 5 + aux_total
    amsua = 2 * amsua_channels
    amsub = 2 * amsub_channels
    hirs = 2 * hirs_channels
    sat = 2 * 2
    icoads = 2 * 5
    hadisd = 2 * 5  # tas, tds, psl, u, v (all 5 HadISD vars now encoded; see docs/plan_hadisd_v_drop.md)
    igra = 0 if disable_igra else 2 * 24
    ascat = ascat_channels
    iasi = iasi_channels

    obs_total = amsua + amsub + hirs + sat + icoads + hadisd + igra + ascat + iasi
    if two_frames:
        return obs_total * 2 + aux_total
    return obs_total + aux_total


def expected_in_channels_forecast(era5_mode, include_year=False):
    """
    Forecast loader y_context = era5/IC fields + elev(4) + time channels.
    ForecastLoader currently uses 4 time channels (no year).
    """
    base = 30 if era5_mode == "4u_sfc" else 24
    time_ch = 5 if include_year else 4
    return base + 4 + time_ch


def main(rank, world_size, output_dir, args):
    """
    Primary training script for the encoder, processor and decoder modules.
    """
    if args.assim_val_stride < 1:
        raise ValueError("--assim_val_stride must be positive")
    if (
        args.time_freq == "1H"
        and args.assim_val_stride > 1
        and math.gcd(args.assim_val_stride, 24) != 1
    ):
        raise ValueError(
            "--assim_val_stride must be coprime with 24 for 1H data so validation "
            "does not alias onto a subset of UTC hours"
        )
    master_port = args.master_port
    lead_time = args.lead_time
    era5_mode = args.era5_mode
    if args.obs_set == "rtma_surface" and bool(args.two_frames):
        raise ValueError("--obs_set rtma_surface currently supports only --two_frames 0")
    weights_dir = args.weights_dir
    ddp_setup(rank, world_size, master_port, args.backend)
    if args.seed is not None:
        # Keep model initialization identical across A/B runs. NumPy/Python receive a rank offset
        # so rank-local sampling remains reproducible without duplicating its random stream.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed + rank)
        random.seed(args.seed + rank)

    # Install the grid config (era5_x/era5_y file names, inner ViT grid) for this process,
    # before any dataset or model is built. int_x/int_y come from the YAML unless overridden
    # on the CLI.
    grid_cfg = load_grid_config(args.grid_config)
    set_active_config(grid_cfg)
    if args.int_x is None:
        args.int_x = grid_cfg["int_x"]
    if args.int_y is None:
        args.int_y = grid_cfg["int_y"]
#clt
    if torch.cuda.is_available() :
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"


    # Instantiate loss function
    if args.loss == "lw_rmse":
        lf = WeightedRmseLoss(
            args.data_path,
            args.aux_data_path,
            start_ind=args.start_ind,
            end_ind=args.end_ind,
            weight_per_variable=bool(args.weight_per_variable),
        )
    elif args.loss == "lw_rmse_pressure_weighted":
        lf = PressureWeightedRmseLoss(
            era5_mode, args.data_path, args.aux_data_path
        )
    elif args.loss == "rmse":
        lf = RmseLoss(
            start_ind=0,
            end_ind=args.end_ind - args.start_ind,
            debug_nan_checks=bool(args.debug_nan_checks),
        )
    elif args.loss == "downscaling_rmse":
        lf = DownscalingRmseLoss()

    # Setup datasets

    # Case 1: training encoder
    if args.mode == "assimilation":
        train_dataset = WeatherDatasetAssimilation(
            device=device_name,
            hadisd_mode="train",
            start_date=args.assim_train_start_date,
            end_date=args.assim_train_end_date,
            lead_time=0,
            era5_mode=args.era5_mode,            var_start=args.start_ind,
            var_end=args.end_ind,
            diff=bool(args.diff),
            two_frames=bool(args.two_frames),
            data_path=args.data_path,
            aux_data_path=args.aux_data_path,
            disable_igra=bool(args.disable_igra),
            time_freq=args.time_freq,
            obs_set=args.obs_set,
            obs_norm_mode=args.obs_norm_mode,
            background_mode=args.background_mode,
            selected_months=args.assim_train_months_resolved,
            sample_stride=1,
        )
        val_dataset = WeatherDatasetAssimilation(
            device=device_name,
            hadisd_mode="train",
            start_date=args.assim_val_start_date,
            end_date=args.assim_val_end_date,
            lead_time=0,
            era5_mode=args.era5_mode,            var_start=args.start_ind,
            var_end=args.end_ind,
            diff=bool(args.diff),
            two_frames=bool(args.two_frames),
            data_path=args.data_path,
            aux_data_path=args.aux_data_path,
            disable_igra=bool(args.disable_igra),
            time_freq=args.time_freq,
            obs_set=args.obs_set,
            obs_norm_mode=args.obs_norm_mode,
            background_mode=args.background_mode,
            selected_months=args.assim_val_months_resolved,
            sample_stride=args.assim_val_stride,
        )

    # Case 2: training processor
    elif args.mode == "forecast":
        if args.ic == "aardvark":
            train_dataset = FineTuneForecastLoaderNew(
                device=device_name,
                mode="train",
                lead_time=lead_time,
                era5_mode=era5_mode,                frequency=args.frequency,
                diff=bool(args.diff),
                aardvark_ic_path=args.aardvark_ic_path,
                random_lt=True,
                data_path=args.data_path,
                aux_data_path=args.aux_data_path,
            )
            val_dataset = FineTuneForecastLoaderNew(
                device=device_name,
                mode="val",
                lead_time=lead_time,
                era5_mode=era5_mode,                frequency=args.frequency,
                diff=bool(args.diff),
                aardvark_ic_path=args.aardvark_ic_path,
                data_path=args.data_path,
                aux_data_path=args.aux_data_path,
            )
        else:
            train_dataset = ForecastLoader(
                device=device_name,
                mode="train",
                lead_time=lead_time,
                era5_mode=era5_mode,                frequency=args.frequency,
                diff=bool(args.diff),
                u_only=False,
                random_lt=False,
                start_date=args.forecast_train_start_date,
                end_date=args.forecast_train_end_date,
                data_path=args.data_path,
                aux_data_path=args.aux_data_path,
            )
            val_dataset = ForecastLoader(
                device=device_name,
                mode="val",
                lead_time=lead_time,
                era5_mode=era5_mode,                frequency=args.frequency,
                diff=bool(args.diff),
                u_only=False,
                random_lt=False,
                start_date=args.forecast_val_start_date,
                end_date=args.forecast_val_end_date,
                data_path=args.data_path,
                aux_data_path=args.aux_data_path,
            )

    # Case 3: training decoder
    elif args.mode == "downscaling":

        train_dataset = ForecasterDatasetDownscaling(
            start_date="2007-01-02",
            end_date="2017-12-31",
            lead_time=args.lead_time,
            hadisd_var=args.var,
            mode="train",
            device=device_name,
            forecast_path=None,
            era5_mode=args.era5_mode,
            data_path=args.data_path,
            aux_data_path=args.aux_data_path,
            time_freq=args.time_freq,
        )

        val_dataset = ForecasterDatasetDownscaling(
            start_date="2019-01-01",
            end_date="2019-12-21",
            lead_time=args.lead_time,
            hadisd_var=args.var,
            mode="train",
            device=device_name,
            forecast_path=None,
            era5_mode=args.era5_mode,
            data_path=args.data_path,
            aux_data_path=args.aux_data_path,
            time_freq=args.time_freq,
        )

        try:
            os.mkdir(f"{output_dir}lt_{args.lead_time}")
        except FileExistsError:
            pass

        output_dir = f"{output_dir}lt_{args.lead_time}/"

    # Instantiate model

    if args.mode == "downscaling":
        model = ConvCNPWeatherOnToOff(
            in_channels=args.in_channels,
            out_channels=args.end_ind - args.start_ind,
            int_channels=args.int_channels,
            device=device_name,            decoder=args.decoder,
            mode=args.mode,
            film=bool(args.film),
            data_path=args.model_data_path,
            cmd_init_ls=args.cmd_init_ls,
        )
        # Cross-check: model vs dataset grid files describe the same grid (shape + values).
        assert_grid_files_consistent(args.model_data_path, args.data_path)
    else:
        amsua_channels = args.amsua_channels
        if amsua_channels is None:
            amsua_channels = 11 if args.time_freq != "6H" else 13
        amsub_channels = args.amsub_channels
        if amsub_channels is None:
            amsub_channels = 5 if args.time_freq != "6H" else 12
        iasi_channels = args.iasi_channels
        if iasi_channels is None:
            iasi_channels = 45 if args.time_freq != "6H" else 52
        ascat_channels = args.ascat_channels
        if ascat_channels is None:
            ascat_channels = 15 if args.time_freq != "6H" else 17
        hirs_channels = args.hirs_channels
        if hirs_channels is None:
            hirs_channels = 20 if args.time_freq != "6H" else 26
        expected_in_channels = expected_in_channels_assimilation(
            amsua_channels,
            amsub_channels,
            iasi_channels,
            ascat_channels,
            hirs_channels,
            disable_igra=bool(args.disable_igra),
            two_frames=bool(args.two_frames),
            climatology_channels=getattr(train_dataset, "climatology_channels", 24),
            obs_set=args.obs_set,
        )
        expected_model_in_channels = None
        if args.mode == "assimilation":
            expected_model_in_channels = expected_in_channels
            if args.in_channels is None:
                args.in_channels = expected_model_in_channels
            elif args.in_channels != expected_model_in_channels:
                raise ValueError(
                    f"in_channels={args.in_channels} does not match expected "
                    f"{expected_model_in_channels} for current settings"
                )
        elif args.mode == "forecast":
            expected_forecast_in = expected_in_channels_forecast(args.era5_mode)
            expected_model_in_channels = expected_forecast_in
            if args.in_channels is None:
                args.in_channels = expected_model_in_channels
            elif args.in_channels != expected_model_in_channels:
                raise ValueError(
                    f"in_channels={args.in_channels} does not match expected "
                    f"{expected_model_in_channels} for forecast settings"
                )
        if args.mode == "assimilation" and rank == 0:
            print(
                "[INFO] assimilation input channels: "
                f"expected={expected_model_in_channels} "
                f"configured={args.in_channels} "
                f"climatology_channels={getattr(train_dataset, 'climatology_channels', 24)} "
                f"climatology_path={getattr(train_dataset, 'climatology_path', 'unknown')}",
                flush=True,
            )
        model_out_channels = args.out_channels
        if args.mode != "forecast" or model_out_channels is None:
            model_out_channels = args.end_ind - args.start_ind
        model = ConvCNPWeather(
            in_channels=args.in_channels,
            out_channels=model_out_channels,
            int_channels=args.int_channels,
            device=device_name,            gnp=bool(0),
            decoder=args.decoder,
            mode=args.mode,
            film=bool(args.film),
            two_frames=bool(args.two_frames),
            data_path=args.model_data_path,
            amsua_channels=amsua_channels,
            amsub_channels=amsub_channels,
            hirs_channels=hirs_channels,
            expected_in_channels=expected_model_in_channels,
            debug_nan_checks=bool(args.debug_nan_checks),
            cmd_init_ls=args.cmd_init_ls,
            int_x=args.int_x,
            int_y=args.int_y,
            obs_set=args.obs_set,
        )

        # Cross-check: the model's grid files (model_data_path) and the dataset's grid files
        # (data_path) must describe the same data grid -- shape and actual coordinate values.
        assert_grid_files_consistent(args.model_data_path, args.data_path)

    # Instantiate loaders
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    print(
        f"[INFO] train_dataset len={len(train_dataset)} "
        f"val_dataset len={len(val_dataset)}",
        flush=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
    )

    # Instantiate trainer

    trainer = DDPTrainer(
        model,
        rank,
        train_loader,
        val_loader,
        lf,
        output_dir,
        args.lr,
        train_sampler,
        weight_decay=args.weight_decay,
        weights_path=weights_dir,
        resume_training=bool(args.resume_training),
        tune_film=args.film,
    )

    # Train model

    trainer.train(n_epochs=args.epoch)
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir")
    parser.add_argument("--mode")
    parser.add_argument("--weights_dir")
    parser.add_argument(
        "--resume_training",
        type=int,
        default=0,
        help="Restore optimizer/scheduler/epoch state from --weights_dir and continue training.",
    )
    parser.add_argument("--in_channels", type=int)
    parser.add_argument("--out_channels", type=int)
    parser.add_argument("--int_channels", type=int)
    parser.add_argument("--loss")
    parser.add_argument("--ic")
    parser.add_argument("--decoder")
    parser.add_argument("--film")
    parser.add_argument("--aardvark_ic_path")
    parser.add_argument("--two_frames", type=int, default=0)
    parser.add_argument("--weight_per_variable", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--master_port", default="12345")
    parser.add_argument("--backend", default="nccl", help="DDP backend (nccl or gloo)")
    parser.add_argument("--world_size", type=int, default=None, help="Override world size")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lead_time", type=int)
    parser.add_argument(
        "--era5_mode",
        default="4u_sfc",
        choices=["4u", "sfc", "4u_sfc", "rtma_ok_sfc"],
    )
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument(
        "--grid_config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the grid-config YAML (era5_x/era5_y file names, int_x/int_y).",
    )
    parser.add_argument(
        "--int_x",
        type=int,
        default=None,
        help="Inner ViT grid width (Grid B); overrides grid_config. Even and >= nlon.",
    )
    parser.add_argument(
        "--int_y",
        type=int,
        default=None,
        help="Inner ViT grid height (Grid B); overrides grid_config. Even and >= nlat.",
    )
    parser.add_argument("--frequency", type=int, default=6)
    parser.add_argument("--diff", type=int, default=1)
    parser.add_argument("--start_ind", type=int, default=0)
    parser.add_argument("--end_ind", type=int, default=24)
    parser.add_argument("--disable_igra", type=int, default=0)
    parser.add_argument(
        "--obs_set",
        default="all",
        choices=["all", "rtma_surface"],
        help="Observation set: 'all' = full Aardvark modalities; "
        "'rtma_surface' = surface obs only (tas, sh, psl, u, v).",
    )
    parser.add_argument(
        "--obs_norm_mode",
        default="static",
        choices=["static", "monthly"],
        help="Surface-observation normalization: 'static' keeps existing norm files; "
        "'monthly' uses station-aligned mean/std vectors for each RTMA month.",
    )
    parser.add_argument(
        "--background_mode",
        default="daily_00z",
        choices=["daily_00z", "hourly"],
        help="RTMA background cadence: persist the daily 00 UTC first guess (default) "
        "or use the first guess valid at each sample hour.",
    )
    parser.add_argument("--time_freq", default="1D")
    parser.add_argument("--amsua_channels", type=int, default=None)
    parser.add_argument("--amsub_channels", type=int, default=None)
    parser.add_argument("--iasi_channels", type=int, default=None)
    parser.add_argument("--ascat_channels", type=int, default=None)
    parser.add_argument("--hirs_channels", type=int, default=None)
    parser.add_argument("--debug_nan_checks", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base random seed; omitted preserves the existing stochastic behavior.",
    )
    parser.add_argument(
        "--cmd_init_ls",
        type=float,
        default=0.001,
        help="Initial learnable ConvDeepSet length scale in normalized lat/lon units.",
    )
    parser.add_argument("--assim_train_start_date", default="2007-01-02")
    parser.add_argument("--assim_train_end_date", default="2017-12-31")
    parser.add_argument("--assim_val_start_date", default="2019-01-01")
    parser.add_argument("--assim_val_end_date", default="2019-12-31")
    parser.add_argument(
        "--assim_val_stride",
        type=int,
        default=1,
        help="Keep every Nth assimilation validation timestamp. For 1H data, N must be "
        "coprime with 24 to retain every UTC hour (default: every timestamp).",
    )
    parser.add_argument(
        "--assim_train_months_file",
        default=None,
        help="Optional text file listing selected assimilation training months, one exact "
        "YYYY-MM per line. Blank lines and lines beginning with # are ignored.",
    )
    parser.add_argument(
        "--assim_val_months_file",
        default=None,
        help="Optional text file listing selected assimilation validation months, one exact "
        "YYYY-MM per line. For single-timestamp inference this may be omitted.",
    )
    parser.add_argument("--forecast_train_start_date", default="2007-01-02")
    parser.add_argument("--forecast_train_end_date", default="2017-12-31")
    parser.add_argument("--forecast_val_start_date", default="2019-01-01")
    parser.add_argument("--forecast_val_end_date", default="2019-12-31")
    parser.add_argument("--data_path", default="path_to_data/")
    parser.add_argument("--aux_data_path", default="path_to_auxiliary_data/")
    parser.add_argument("--model_data_path", default="../data/")
    parser.add_argument("--downscaling_train_start_date", default="1979-01-01")
    parser.add_argument("--downscaling_train_end_date", default="2017-12-31")
    parser.add_argument("--downscaling_context", default="era5")
    parser.add_argument("--downscaling_lead_time", type=int)
    parser.add_argument("--var", default=None)
    args = parser.parse_args()

    if (
        args.assim_train_months_file or args.assim_val_months_file
    ) and args.mode != "assimilation":
        parser.error("assimilation month manifests are supported only with --mode assimilation")

    try:
        train_months = (
            read_month_manifest(
                args.assim_train_months_file, "--assim_train_months_file"
            )
            if args.assim_train_months_file
            else None
        )
        val_months = (
            read_month_manifest(args.assim_val_months_file, "--assim_val_months_file")
            if args.assim_val_months_file
            else None
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    # Preserve canonical values in config.pkl; workers receive these directly rather than
    # reparsing files whose contents could change after the run starts.
    args.assim_train_months_resolved = (
        format_months(train_months) if train_months else None
    )
    args.assim_val_months_resolved = format_months(val_months) if val_months else None

    if train_months and val_months:
        overlap = sorted(set(train_months) & set(val_months))
        if overlap:
            print(
                "[WARN] assimilation training and validation month manifests overlap: "
                + " ".join(format_months(overlap))
                + "; validation will not measure independent generalization.",
                flush=True,
            )

    torch.device("cuda")

    # Create results directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Keep the source manifests with the run in addition to recording their resolved values.
    for source, destination_name in (
        (args.assim_train_months_file, "selected_train_months.txt"),
        (args.assim_val_months_file, "selected_val_months.txt"),
    ):
        if source:
            destination = os.path.join(output_dir, destination_name)
            if os.path.abspath(source) != os.path.abspath(destination):
                shutil.copy2(source, destination)

    # Save config
    with open(output_dir + "/config.pkl", "wb") as f:
        pickle.dump(vars(args), f)

    world_size = args.world_size or torch.cuda.device_count()
    if args.backend == "gloo" and world_size == 1:
        main(0, 1, output_dir, args)
    else:
        mp.spawn(main, args=[world_size, output_dir, args], nprocs=world_size)
