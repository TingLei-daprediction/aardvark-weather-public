"""
NB: this script is for illustration purposes only and is not runnable as our
full dataset is not provided as part of the submission, due to size constraints.
Many of the relevant paths to the data have been thus replaced by dummy paths.
"""

import os
import sys
import pickle
import argparse

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
    disable_igra,
    two_frames,
):
    # convDeepSet encoders output density + value per channel (2x).
    amsua = 2 * amsua_channels
    amsub = 2 * amsub_channels
    hirs = 2 * 26
    sat = 2 * 2
    icoads = 2 * 5
    hadisd = 2 * 4
    igra = 0 if disable_igra else 2 * 24
    ascat = ascat_channels
    iasi = iasi_channels

    obs_total = amsua + amsub + hirs + sat + icoads + hadisd + igra + ascat + iasi
    aux_total = 4 + 24 + 5  # elev vars + climatology + aux time channels
    if two_frames:
        return obs_total * 2 + aux_total
    return obs_total + aux_total


def main(rank, world_size, output_dir, args):
    """
    Primary training script for the encoder, processor and decoder modules.
    """

    master_port = args.master_port
    lead_time = args.lead_time
    era5_mode = args.era5_mode
    weights_dir = args.weights_dir
    ddp_setup(rank, world_size, master_port, args.backend)
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
            args.res,
            args.data_path,
            args.aux_data_path,
            start_ind=args.start_ind,
            end_ind=args.end_ind,
            weight_per_variable=bool(args.weight_per_variable),
        )
    elif args.loss == "lw_rmse_pressure_weighted":
        lf = PressureWeightedRmseLoss(
            args.res, era5_mode, args.data_path, args.aux_data_path
        )
    elif args.loss == "rmse":
        lf = RmseLoss()
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
            era5_mode="4u",
            res=args.res,
            var_start=args.start_ind,
            var_end=args.end_ind,
            diff=bool(args.diff),
            two_frames=bool(args.two_frames),
            data_path=args.data_path,
            aux_data_path=args.aux_data_path,
            disable_igra=bool(args.disable_igra),
            time_freq=args.time_freq,
        )
        val_dataset = WeatherDatasetAssimilation(
            device=device_name,
            hadisd_mode="train",
            start_date=args.assim_val_start_date,
            end_date=args.assim_val_end_date,
            lead_time=0,
            era5_mode="4u",
            res=args.res,
            var_start=args.start_ind,
            var_end=args.end_ind,
            diff=bool(args.diff),
            two_frames=bool(args.two_frames),
            data_path=args.data_path,
            aux_data_path=args.aux_data_path,
            disable_igra=bool(args.disable_igra),
            time_freq=args.time_freq,
        )

    # Case 2: training processor
    elif args.mode == "forecast":
        if args.ic == "aardvark":
            train_dataset = FineTuneForecastLoaderNew(
                device=device_name,
                mode="train",
                lead_time=lead_time,
                era5_mode=era5_mode,
                res=args.res,
                frequency=args.frequency,
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
                era5_mode=era5_mode,
                res=args.res,
                frequency=args.frequency,
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
                era5_mode=era5_mode,
                res=args.res,
                frequency=args.frequency,
                diff=bool(args.diff),
                u_only=False,
                random_lt=False,
                data_path=args.data_path,
                aux_data_path=args.aux_data_path,
            )
            val_dataset = ForecastLoader(
                device=device_name,
                mode="val",
                lead_time=lead_time,
                era5_mode=era5_mode,
                res=args.res,
                frequency=args.frequency,
                diff=bool(args.diff),
                u_only=False,
                random_lt=False,
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
            device=device_name,
            res=args.res,
            decoder=args.decoder,
            mode=args.mode,
            film=bool(args.film),
            data_path=args.model_data_path,
        )
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
        expected_in_channels = expected_in_channels_assimilation(
            amsua_channels,
            amsub_channels,
            iasi_channels,
            ascat_channels,
            disable_igra=bool(args.disable_igra),
            two_frames=bool(args.two_frames),
        )
        if args.in_channels is None:
            args.in_channels = expected_in_channels
        elif args.in_channels != expected_in_channels:
            raise ValueError(
                f"in_channels={args.in_channels} does not match expected "
                f"{expected_in_channels} for current settings"
            )
        model = ConvCNPWeather(
            in_channels=args.in_channels,
            out_channels=args.end_ind - args.start_ind,
            int_channels=args.int_channels,
            device=device_name,
            res=args.res,
            gnp=bool(0),
            decoder=args.decoder,
            mode=args.mode,
            film=bool(args.film),
            two_frames=bool(args.two_frames),
            data_path=args.model_data_path,
            amsua_channels=amsua_channels,
            amsub_channels=amsub_channels,
        )

    # Instantiate loaders
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

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
    parser.add_argument("--era5_mode", default="4u")
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--res", type=int, default=1)
    parser.add_argument("--frequency", type=int, default=6)
    parser.add_argument("--diff", type=int, default=1)
    parser.add_argument("--start_ind", type=int, default=0)
    parser.add_argument("--end_ind", type=int, default=24)
    parser.add_argument("--disable_igra", type=int, default=0)
    parser.add_argument("--time_freq", default="6H")
    parser.add_argument("--amsua_channels", type=int, default=None)
    parser.add_argument("--amsub_channels", type=int, default=None)
    parser.add_argument("--iasi_channels", type=int, default=None)
    parser.add_argument("--ascat_channels", type=int, default=None)
    parser.add_argument("--assim_train_start_date", default="2007-01-02")
    parser.add_argument("--assim_train_end_date", default="2017-12-31")
    parser.add_argument("--assim_val_start_date", default="2019-01-01")
    parser.add_argument("--assim_val_end_date", default="2019-12-31")
    parser.add_argument("--data_path", default="path_to_data/")
    parser.add_argument("--aux_data_path", default="path_to_auxiliary_data/")
    parser.add_argument("--model_data_path", default="../data/")
    parser.add_argument("--downscaling_train_start_date", default="1979-01-01")
    parser.add_argument("--downscaling_train_end_date", default="2017-12-31")
    parser.add_argument("--downscaling_context", default="era5")
    parser.add_argument("--downscaling_lead_time", type=int)
    parser.add_argument("--var", default=None)
    args = parser.parse_args()

    torch.device("cuda")

    # Create results directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Save config
    with open(output_dir + "/config.pkl", "wb") as f:
        pickle.dump(vars(args), f)

    world_size = args.world_size or torch.cuda.device_count()
    if args.backend == "gloo" and world_size == 1:
        main(0, 1, output_dir, args)
    else:
        mp.spawn(main, args=[world_size, output_dir, args], nprocs=world_size)
