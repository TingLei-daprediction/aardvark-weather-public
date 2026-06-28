import sys

import numpy as np
import torch
import torch.nn as nn

from architectures import MLP
from set_convs import convDeepSet
from unet_wrap_padding import *
from vit import *
from grid_config import model_grid_x_path, model_grid_y_path

sys.path.append("../")


class ConvCNPWeather(nn.Module):
    """
    ConvCNP class used for the encoder and processor modules
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        int_channels,
        device,
        data_path="../data/",
        gnp=False,
        mode="assimilation",
        decoder=None,
        film=False,
        two_frames=False,
        amsua_channels=13,
        amsub_channels=12,
        hirs_channels=26,
        expected_in_channels=None,
        debug_nan_checks=False,
        cmd_init_ls=0.001,
        int_x=256,
        int_y=128,
    ):

        super().__init__()

        self.device = device

        if (
            expected_in_channels is not None
            and mode == "assimilation"
            and in_channels != expected_in_channels
        ):
            raise ValueError(
                f"in_channels={in_channels} does not match expected "
                f"{expected_in_channels} for current settings"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.int_channels = int_channels
        self.decoder = decoder
        self.int_x = int_x  # inner ViT grid width (Grid B), configurable
        self.int_y = int_y  # inner ViT grid height (Grid B), configurable
        self.data_path = data_path
        self.mode = mode
        self.film = film
        self.two_frames = two_frames
        self.amsua_channels = amsua_channels
        self.amsub_channels = amsub_channels
        self.hirs_channels = hirs_channels
        self.debug_nan_checks = debug_nan_checks
        self.cmd_init_ls = float(cmd_init_ls)

        N_SAT_VARS = 2  # clt_hard-wired number of satellite vars used by encoder_sat
        N_ICOADS_VARS = 5  # clt_hard-wired number of ICOADS vars used by encoder_icoads
        N_HADISD_VARS = 5  # clt_hard-wired number of HadISD vars used by encoder_hadisd

        # Load data/target grid (Grid A) longitude-latitude locations. These files are the
        # single source of truth for the data-grid size and extent.
        self.era5_x = (
            torch.from_numpy(
                np.load(model_grid_x_path(self.data_path))
            ).float()
            / 360
        )
        self.era5_y = (
            torch.from_numpy(
                np.load(model_grid_y_path(self.data_path))
            ).float()
            / 360
        )

        # Grid A dimensions, derived from the grid files (not hard-wired).
        self.nlon = int(self.era5_x.shape[0])
        self.nlat = int(self.era5_y.shape[0])

        # Observation-aggregation grid (Grid A). Built from the actual grid coordinates so
        # the extent matches the (possibly regional) domain -- previously hard-wired to the
        # global linspace(0,360,240)/(-90,90,121), which smears obs across the whole globe
        # for a regional domain.
        # Kept on CPU here; forward() moves int_grid to the batch device
        # (see `self.int_grid = [i.to(task["y_target"].device) ...]`).
        self.int_grid = [
            self.era5_x.float(),
            self.era5_y.float(),
        ]

        self.int_grid = [self.int_grid[0].unsqueeze(0), self.int_grid[1].unsqueeze(0)]

        # Validate the configured inner grid (Grid B). Only the assimilation/ViT-assimilation
        # path upsamples to int_x/int_y, so these constraints apply there only.
        if self.decoder == "vit_assimilation":
            if self.int_x % 2 or self.int_y % 2:
                raise ValueError(
                    f"[grid] inner grid (int_x={self.int_x}, int_y={self.int_y}) must have "
                    "even dimensions (odd sizes break stride-2/pooling in the backbone)."
                )
            if self.int_x < self.nlon or self.int_y < self.nlat:
                raise ValueError(
                    f"[grid] inner grid (int_x={self.int_x}, int_y={self.int_y}) must be >= "
                    f"data grid (nlon={self.nlon}, nlat={self.nlat}); it should oversample, "
                    "not undersample, Grid A."
                )

        # Create input setconvs for each data modality
        self.ascat_setconvs = convDeepSet(
            self.cmd_init_ls, "OnToOn", density_channel=True, device=self.device
        )
        self.amsua_setconvs = [
            convDeepSet(self.cmd_init_ls, "OnToOn", density_channel=True, device=self.device)
            for _ in range(self.amsua_channels)
        ]
        self.amsub_setconvs = [
            convDeepSet(self.cmd_init_ls, "OnToOn", density_channel=True, device=self.device)
            for _ in range(self.amsub_channels)
        ]
        self.hirs_setconvs = [
            convDeepSet(self.cmd_init_ls, "OnToOn", density_channel=True, device=self.device)
            for _ in range(self.hirs_channels)
        ]

        self.sat_setconvs = [
            convDeepSet(self.cmd_init_ls, "OnToOn", density_channel=True, device=self.device)
            for _ in range(N_SAT_VARS)
        ]
        self.hadisd_setconvs = [
            convDeepSet(self.cmd_init_ls, "OffToOn", density_channel=True, device=self.device)
            for _ in range(N_HADISD_VARS)
        ]
        self.icoads_setconvs = [
            convDeepSet(self.cmd_init_ls, "OffToOn", density_channel=True, device=self.device)
            for _ in range(N_ICOADS_VARS)
        ]
        self.igra_setconvs = [
            convDeepSet(self.cmd_init_ls, "OffToOn", density_channel=True, device=self.device)
            for _ in range(24)
        ]

        self.sc_out = convDeepSet(
            self.cmd_init_ls, "OnToOff", density_channel=False, device=self.device
        )

        # Instantiate the decoder. Here decoder refers to decoder in a convCNP (i.e the ViT backbone)
        if self.decoder == "vit":
            self.decoder_lr = ViT(
                in_channels=in_channels,
                out_channels=out_channels,
                h_channels=512,  # clt_hard-wired ViT hidden width
                depth=16,  # clt_hard-wired ViT depth
                patch_size=5,  # clt_hard-wired ViT patch size
                per_var_embedding=True,
                img_size=[self.nlon, self.nlat],  # data/target grid (Grid A)
            )

        elif self.decoder == "vit_assimilation":
            self.decoder_lr = ViT(
                in_channels=self.in_channels,
                out_channels=out_channels,
                h_channels=512,  # clt_hard-wired ViT hidden width
                depth=8,  # clt_hard-wired ViT depth
                patch_size=3,  # clt_hard-wired ViT patch size
                per_var_embedding=False,
                img_size=[self.int_x, self.int_y],  # inner ViT grid (Grid B)
            )

        self.mlp = MLP(
            in_channels=out_channels,
            out_channels=out_channels,
            h_channels=128,  # clt_hard-wired MLP hidden width
            h_layers=4,  # clt_hard-wired MLP depth
        )
        self.break_next = False

    def encoder_hadisd(self, task, prefix):
        """
        Data preprocessing for HadISD
        """

        encodings = []
        for channel in range(len(self.hadisd_setconvs)):
            encodings.append(
                self.hadisd_setconvs[channel](
                    x_in=[
                        task["x_context_hadisd_{}".format(prefix)][channel][:, 0, :],
                        task["x_context_hadisd_{}".format(prefix)][channel][:, 1, :],
                    ],
                    wt=task["y_context_hadisd_{}".format(prefix)][channel].unsqueeze(1),
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_sat(self, task, prefix):
        """
        Data preprocessing for Gridsat
        """

        encodings = []
        for channel in range(task["sat_{}".format(prefix)].shape[1]):
            encodings.append(
                self.sat_setconvs[channel](
                    x_in=task["sat_x_{}".format(prefix)],
                    wt=task["sat_{}".format(prefix)][:, channel : channel + 1, ...],
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_icoads(self, task, prefix):
        """
        Data preprocessing for ICOADS
        """

        encodings = []
        for channel in range(5):
            encodings.append(
                self.icoads_setconvs[channel](
                    x_in=task["icoads_x_{}".format(prefix)],
                    wt=task["icoads_{}".format(prefix)][:, channel, :].unsqueeze(1),
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)

        return encodings

    def encoder_amsua(self, task, prefix):
        """
        Data preprocessing for AMSU-A
        """

        encodings = []
        task["amsua_{}".format(prefix)][..., -1] = np.nan
        task["amsua_{}".format(prefix)][task["amsua_{}".format(prefix)] == 0] = np.nan
        for i in range(self.amsua_channels):
            encodings.append(
                self.amsua_setconvs[i](
                    x_in=task["amsua_x_{}".format(prefix)],
                    wt=task["amsua_{}".format(prefix)].permute(0, 3, 2, 1)[
                        :, i : i + 1, ...
                    ],
                    x_out=self.int_grid,
                )
            )

        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_amsub(self, task, prefix):
        """
        Data preprocessing for AMSU-B
        """

        encodings = []
        task["amsub_{}".format(prefix)][task["amsub_{}".format(prefix)] == 0] = np.nan
        for i in range(self.amsub_channels):
            encodings.append(
                self.amsub_setconvs[i](
                    x_in=task["amsub_x_{}".format(prefix)],
                    wt=task["amsub_{}".format(prefix)].permute(0, 3, 1, 2)[
                        :, i : i + 1, ...
                    ],
                    x_out=self.int_grid,
                )
            )

        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_hirs(self, task, prefix):
        """
        Data preprocessing for HIRS
        """

        encodings = []

        task["hirs_{}".format(prefix)][task["hirs_{}".format(prefix)] == 0] = np.nan
        for i in range(self.hirs_channels):
            encodings.append(
                self.hirs_setconvs[i](
                    x_in=task["hirs_x_{}".format(prefix)],
                    wt=task["hirs_{}".format(prefix)].permute(0, 3, 1, 2)[
                        :, i : i + 1, ...
                    ],
                    x_out=self.int_grid,
                )
            )

        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_igra(self, task, prefix):
        """
        Data preprocessing for IGRA
        """

        encodings = []
        for channel in range(24):
            encodings.append(
                self.igra_setconvs[channel](
                    x_in=task["igra_x_{}".format(prefix)],
                    wt=task["igra_{}".format(prefix)][:, channel, :].unsqueeze(1),
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)

        return encodings

    def encoder_ascat(self, task, prefix):
        """
        Data preprocessing for ASCAT
        """

        task["ascat_{}".format(prefix)][
            torch.isnan(task["ascat_{}".format(prefix)])
        ] = 0
        e = nn.functional.interpolate(
            task["ascat_{}".format(prefix)].permute(0, 3, 1, 2), size=(self.nlon, self.nlat)
        )
        e = torch.flip(e, dims=[-1])
        return e

    def encoder_iasi(self, task, prefix):
        """
        Data preprocessing for IASI
        """

        task["iasi_{}".format(prefix)][torch.isnan(task["iasi_{}".format(prefix)])] = 0
        e = nn.functional.interpolate(
            task["iasi_{}".format(prefix)].permute(0, 3, 1, 2), size=(self.nlon, self.nlat)
        )
        e = torch.flip(e, dims=[-1])
        return e

    def forward(self, task, film_index):

        # Setup input
        if self.mode == "assimilation":

            self.int_grid = [i.to(task["y_target"].device) for i in self.int_grid]
            elev = nn.functional.interpolate(
                torch.flip(task["era5_elev_current"].permute(0, 1, 3, 2), dims=[2]),
                size=(self.int_grid[0].shape[1], self.int_grid[1].shape[1]),
            )
#cltorg bug            elev = torch.flip(task["era5_elev_current"].permute(0, 1, 3, 2), dims=[2])

            def igra_encoding(prefix):
                if f"igra_{prefix}" in task and f"igra_x_{prefix}" in task:
                    return self.encoder_igra(task, prefix)
                batch = task["y_target"].shape[0]
                return torch.zeros(
                    (batch, 24, elev.shape[2], elev.shape[3]),
                    device=elev.device,
                )

            if not self.two_frames:
                encodings = [
                    self.encoder_iasi(task, "current"),
                    self.encoder_ascat(task, "current"),
                    self.encoder_hadisd(task, "current"),
                    self.encoder_icoads(task, "current"),
                    self.encoder_sat(task, "current"),
                    self.encoder_amsua(task, "current"),
                    self.encoder_amsub(task, "current"),
                    igra_encoding("current"),
                    self.encoder_hirs(task, "current"),
                    elev,
                    task["climatology_current"],
                    torch.ones(
                        (
                            elev.shape[0],
                            task["aux_time_current"].shape[1],
                            elev.shape[2],
                            elev.shape[3],
                        ),
                        device=elev.device,
                    )
                    * task["aux_time_current"].unsqueeze(-1).unsqueeze(-1),
                ]
                if not getattr(self, "_debug_encoding_shapes", False):
                    for i, enc in enumerate(encodings):
                        print(f"[DEBUG] encodings[{i}] shape: {tuple(enc.shape)}")
                    self._debug_encoding_shapes = True
            else:
                # Option to pass two timesteps (t=-1 and t=0) as input
                encodings = [
                    self.encoder_iasi(task, "current"),
                    self.encoder_ascat(task, "current"),
                    self.encoder_hadisd(task, "current"),
                    self.encoder_icoads(task, "current"),
                    self.encoder_sat(task, "current"),
                    self.encoder_amsua(task, "current"),
                    self.encoder_amsub(task, "current"),
                    igra_encoding("current"),
                    self.encoder_hirs(task, "current"),
                    self.encoder_iasi(task, "prev"),
                    self.encoder_ascat(task, "prev"),
                    self.encoder_hadisd(task, "prev"),
                    self.encoder_icoads(task, "prev"),
                    self.encoder_sat(task, "prev"),
                    self.encoder_amsua(task, "prev"),
                    self.encoder_amsub(task, "prev"),
                    igra_encoding("prev"),
                    self.encoder_hirs(task, "prev"),
                    elev,
                    task["climatology_current"],
                    torch.ones(
                        (
                            elev.shape[0],
                            task["aux_time_current"].shape[1],
                            elev.shape[2],
                            elev.shape[3],
                        ),
                        device=elev.device,
                    )
                    * task["aux_time_current"].unsqueeze(-1).unsqueeze(-1),
                ]
            if self.debug_nan_checks and not getattr(self, "_debug_encoding_nans", False):
                for i, enc in enumerate(encodings):
                    enc_nan = torch.isnan(enc).sum().item()
                    enc_inf = torch.isinf(enc).sum().item()
                    if enc_nan or enc_inf:
                        print(
                            f"[DEBUG] encodings[{i}] nan={enc_nan} inf={enc_inf} "
                            f"min={enc.min().item():.6g} max={enc.max().item():.6g}"
                        )
                self._debug_encoding_nans = True
            spatial = [enc.shape[-2:] for enc in encodings]
            if len(set(spatial)) != 1:
                try:
                    import torch.distributed as dist

                    rank = (
                        dist.get_rank()
                        if dist.is_available() and dist.is_initialized()
                        else -1
                    )
                except Exception:
                    rank = -1
                print(
                    f"[DEBUG][rank {rank}] encoding spatial mismatch: "
                    f"{[tuple(enc.shape) for enc in encodings]}"
                )
                raise RuntimeError(f"Encoding spatial mismatch: {spatial}")
            x = torch.cat(encodings, dim=1)
            if self.debug_nan_checks and not getattr(self, "_debug_concat_nans", False):
                x_nan = torch.isnan(x).sum().item()
                x_inf = torch.isinf(x).sum().item()
                print(
                    f"[DEBUG] concat nan={x_nan} inf={x_inf} "
                    f"min={x.min().item():.6g} max={x.max().item():.6g}"
                )
                self._debug_concat_nans = True

        else:
            x = task["y_context"]

        if x.shape[-1] > x.shape[-2]:
            x = x.permute(0, 1, 3, 2)

        # Run ViT backbone
        if self.decoder == "vit":
            x = self.decoder_lr(x, lead_times=task["lt"])
            x = x.permute(0, 3, 1, 2)
        else:
            x = nn.functional.interpolate(x, size=(self.int_x, self.int_y))
            x = self.decoder_lr(x, film_index=(task["lt"] * 0) + 1)
        if self.debug_nan_checks and not getattr(self, "_debug_decoder_nans", False):
            x_nan = torch.isnan(x).sum().item()
            x_inf = torch.isinf(x).sum().item()
            print(
                f"[DEBUG] decoder nan={x_nan} inf={x_inf} "
                f"min={x.min().item():.6g} max={x.max().item():.6g}"
            )
            self._debug_decoder_nans = True

        # Process outputs

        if np.logical_and(
            self.mode == "assimilation", self.decoder == "vit_assimilation"
        ):
            x = nn.functional.interpolate(x.permute(0, 3, 1, 2), size=(self.nlon, self.nlat))
            return x.permute(0, 3, 2, 1)

        elif self.mode == "forecast":
            x = nn.functional.interpolate(x, size=(self.nlon, self.nlat)).permute(0, 2, 3, 1)
            return x.permute(0, 2, 1, 3)

        return x
