import os
import time as timelib
from time import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from loader_utils_new import *
from data_shapes import *
from grid_config import (
    loader_grid_x_path,
    loader_grid_y_path,
    elev_vars_path,
    norm_mean_path,
    norm_std_path,
    era5_memmap_path,
)


def grid_dims_from_files(data_path):
    """Return (nlon, nlat) for the data/target grid (Grid A) from the canonical grid files.

    The era5_x file holds the longitudes (nlon) and era5_y the latitudes (nlat); their names
    come from the active grid config (see ``grid_config``). These files are the single source
    of truth for the data-grid size; every other grid-dependent array is validated against
    them via :func:`assert_grid_match`.
    """
    lon = np.load(loader_grid_x_path(data_path), mmap_mode="r")
    lat = np.load(loader_grid_y_path(data_path), mmap_mode="r")
    return int(np.asarray(lon).shape[0]), int(np.asarray(lat).shape[0])


def assert_grid_match(name, spatial_dims, expected):
    """Assert a 2-D spatial shape matches an explicit expected (orientation-sensitive) order.

    Loaders store grid fields in differing axis orders (e.g. raw ``elev_vars`` is
    (channels, nlat, nlon) but post-loader tensors are (channels, nlon, nlat)). Pass the
    expected ``(d0, d1)`` so a swapped lat/lon array is caught, not hidden.
    """
    dims = tuple(int(d) for d in spatial_dims)
    exp = tuple(int(e) for e in expected)
    if dims != exp:
        raise ValueError(
            f"[grid] {name} spatial dims {dims} do not match expected {exp}; "
            "check the era5_x/era5_y grid files and this array's lat/lon axis order."
        )


def assert_memmap_size(name, path, shape, itemsize=4):
    """Assert an on-grid memmap file's byte size matches the expected shape exactly.

    ``np.memmap(mode="r", shape=...)`` raises only when the file is *smaller* than required;
    a *larger* file is silently mapped to a prefix, so a wrong-resolution (e.g. non-OK-domain)
    observation file would otherwise go undetected. This catches size/resolution mismatches.
    Note: it cannot detect a same-byte-count lat/lon axis swap (the memmap stores no
    coordinates); orientation is enforced by convention via the shape we pass.
    """
    expected = int(np.prod(shape)) * itemsize
    actual = os.path.getsize(path)
    if actual != expected:
        raise ValueError(
            f"[grid] {name} file {path} size {actual} bytes does not match expected "
            f"{expected} bytes for shape {tuple(shape)} (itemsize {itemsize}); the file's "
            "grid/resolution likely differs from era5_x/era5_y."
        )


class WeatherDataset(Dataset):
    """
    Base weather dataset class
    """

    def __init__(
        self,
        device,
        hadisd_mode,
        start_date,
        end_date,
        lead_time,
        era5_mode="train",
        filter_dates=None,
        diff=None,
        data_path=None,
        aux_data_path=None,
        disable_igra=False,
        time_freq="6H",
        obs_set="all",
    ):

        super().__init__()

        # Setup
        self.device = device
        self.mode = hadisd_mode
        self.data_path = data_path or "path_to_data/"
        self.aux_data_path = aux_data_path or "path_to_auxiliary_data/"
        self.start_date = start_date
        self.end_date = end_date
        self.lead_time = lead_time
        self.era5_mode = era5_mode
        self.filter_dates = filter_dates
        self.diff = diff
        self.disable_igra = disable_igra
        self.time_freq = time_freq
        # obs_set selects which observation modalities are loaded/encoded.
        # "all" = full Aardvark set; "rtma_surface" = surface obs only (tas, sh, psl, u, v).
        self.obs_set = obs_set
        self.surface_only = obs_set == "rtma_surface"
        # Per-month file layout (ERA5-type indexing) for the RTMA surface 15-min path. When
        # active, obs and target are stored one file per month and addressed by month_frame();
        # the offset machinery (build_offsets) is bypassed -- alignment is deterministic.
        self.step_minutes, self.frames_per_day, self.freq_tag = parse_time_freq(self.time_freq)
        self.monthly = self.surface_only and self.time_freq == "15min"
        # 15-min is only wired through the per-month rtma_surface path; other obs sets still use
        # the global-obs/offset logic and have no 15-min layout. Fail clearly rather than later.
        if self.time_freq == "15min" and not self.monthly:
            raise ValueError(
                "time_freq=15min is currently supported only with --obs_set rtma_surface"
            )
        self.offsets = None if self.monthly else build_offsets(self.time_freq)

        # Data/target grid (Grid A) size, derived from the canonical grid files. Set before
        # any modality loading so the observation/ERA5 memmap shapes and the elevation/grid
        # assertions can all be validated against it.
        self.nlon, self.nlat = grid_dims_from_files(self.data_path)

        # Date indexing
        self.dates = pd.date_range(start_date, end_date, freq=self.time_freq)
        if self.filter_dates == "start":
            self.index = np.array([i for i, d in enumerate(self.dates) if d.month < 7])
        elif self.filter_dates == "end":
            self.index = np.array([i for i, d in enumerate(self.dates) if d.month >= 7])
        else:
            self.index = np.array(range(len(self.dates)))
        if self.index.size == 0:
            print(
                f"[WARN] Empty date index: start={self.start_date} end={self.end_date} "
                f"time_freq={self.time_freq} filter_dates={self.filter_dates}"
            )
        # offsets is None on the per-month path (alignment is deterministic, no offset table).
        if self.offsets is not None:
            for key, mapping in self.offsets.items():
                if self.start_date not in mapping:
                    print(
                        f"[WARN] start_date {self.start_date} not in offsets for {key}; "
                        "dataset may be empty or misaligned."
                    )

        # Load the input modalities. For obs_set="rtma_surface" only the surface obs
        # (HadISD per-variable path) are loaded; the global modalities are skipped -- required
        # because the regional (OK) setup does not provide those global obs files.
        if not self.surface_only:
            if not self.disable_igra:
                print("Loading IGRA")
                self.load_igra()

            print("Loading AMSU-A")
            self.load_amsua()

            print("Loading AMSU-B")
            self.load_amsub()

            print("Loading ICOADS")
            self.load_icoads()

            print("Loading IASI")
            self.load_iasi()

            print("Loading GEO")
            self.load_sat_data()

        print("Loading HADISD")
        self.load_hadisd(self.mode)

        if not self.surface_only:
            print("Loading ASCAT")
            self.load_ascat_data()
            self.load_hirs_data()

        # Load the ground truth data for training
        print("Loading ERA5")
        if self.monthly:
            # Per-month target files keyed by (year, month); see _load_era5_monthly.
            self.era5_sfc = self._load_era5_monthly()
        else:
            self.era5_sfc = [
                self.load_era5(year)
                for year in range(int(start_date[:4]), int(end_date[:4]) + 1)
            ]

        # Internal grid to longitude latitude correspondence
        self.era5_x = [
            self.to_tensor(
                np.load(loader_grid_x_path(self.data_path))
            )
            / LATLON_SCALE_FACTOR,
            self.to_tensor(
                np.load(loader_grid_y_path(self.data_path))
            )
            / LATLON_SCALE_FACTOR,
        ]
        assert_grid_match(
            "era5_x/era5_y",
            (self.era5_x[0].shape[0], self.era5_x[1].shape[0]),
            (self.nlon, self.nlat),
        )

        # Orography. Raw file convention is (channels, nlat, nlon); after permute/flip the
        # loader tensor is (channels, nlon, nlat).
        raw_elev = np.load(elev_vars_path(self.data_path))
        assert_grid_match("elev_vars (raw)", raw_elev.shape[1:], (self.nlat, self.nlon))
        self.era5_elev = self.to_tensor(raw_elev)
        self.era5_elev = torch.flip(self.era5_elev.permute(0, 2, 1), [-1])
        assert_grid_match(
            "era5_elev (loader)", self.era5_elev.shape[1:], (self.nlon, self.nlat)
        )
        xx, yy = torch.meshgrid(self.era5_x[0], self.era5_x[1])
        self.era5_lonlat = torch.stack([xx, yy])

        # Climatology slot. On the rtma_surface 15-min path this slot is fed a NORMALIZED daily
        # 00z BACKGROUND/prior (date-specific, NOT a multi-year climatology) from per-month files;
        # otherwise it is the conventional multi-year day-of-year climatology.
        if self.monthly:
            self.background = self._load_background_monthly()
            self.climatology_channels = self._background_channels
        else:
            climatology_path = self.data_path + "era5/climatology_data.mmap"
            if not os.path.exists(climatology_path):
                climatology_path = self.data_path + "climatology_data.mmap"
            self.climatology_shape = get_climatology_shape(
                climatology_path, self.nlon, self.nlat
            )
            self.climatology_channels = self.climatology_shape[2]
            self.climatology_path = climatology_path
            self.climatology = np.memmap(
                climatology_path,
                dtype="float32",
                mode="r",
                shape=self.climatology_shape,
            )

        # Setup normalisation factors
        if self.diff:
            self.era5_mean_spatial = np.load(
                self.aux_data_path + "era5_spatial_means.npy"
            )[0, ...]
            self.means = np.load(self.aux_data_path + "era5_avdiff_means.npy")[
                :, np.newaxis, np.newaxis, ...
            ]
            self.stds = np.load(self.aux_data_path + "era5_avdiff_stds.npy")[
                :, np.newaxis, np.newaxis, ...
            ]
        else:
            self.means = np.load(
                norm_mean_path(self.aux_data_path, self.era5_mode)
            )[:, np.newaxis, np.newaxis, ...]
            self.stds = np.load(
                norm_std_path(self.aux_data_path, self.era5_mode)
            )[:, np.newaxis, np.newaxis, ...]

    def _infer_time_dim(self, path, fixed_shape):
        file_bytes = os.path.getsize(path)
        denom = 4
        for dim in fixed_shape:
            denom *= dim
        if file_bytes % denom != 0:
            raise ValueError(f"File size not divisible by expected frame size: {path}")
        return file_bytes // denom

    def _era5_grid_axes(self):
        lon = np.load(loader_grid_x_path(self.data_path))
        lat = np.load(loader_grid_y_path(self.data_path))
        return lon.astype(np.float32), lat.astype(np.float32)

    def load_icoads(self):
        """
        Load the ICOADS data
        """

        icoads_y_path = self.data_path + "icoads/1999_2021_icoads_y.mmap"
        icoads_y_shape = list(
            ICOADS_Y_SHAPE_1D if self.time_freq != "6H" else ICOADS_Y_SHAPE
        )
        self.icoads_y = np.memmap(
            icoads_y_path,
            dtype="float32",
            mode="r",
            shape=tuple(icoads_y_shape),
        )

        icoads_x_path = self.data_path + "icoads/1999_2021_icoads_x.mmap"
        icoads_x_shape = list(
            ICOADS_X_SHAPE_1D if self.time_freq != "6H" else ICOADS_X_SHAPE
        )
        static_shape = (icoads_x_shape[2], icoads_x_shape[1])
        file_bytes = os.path.getsize(icoads_x_path)
        if file_bytes == np.prod(static_shape) * 4:
            icoads_x = np.memmap(
                icoads_x_path,
                dtype="float32",
                mode="r",
                shape=static_shape,
            )
            self.icoads_x = icoads_x / LATLON_SCALE_FACTOR
            self.icoads_x_is_static = True
        else:
            icoads_x = np.memmap(
                icoads_x_path,
                dtype="float32",
                mode="r",
                shape=tuple(icoads_x_shape),
            )
            self.icoads_x = icoads_x / LATLON_SCALE_FACTOR
            self.icoads_x_is_static = False
        self.icoads_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_icoads.npy")
        )
        self.icoads_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_icoads.npy")
        )
        window = 365 * (4 if self.time_freq == "6H" else 1)
        self.icoads_means = self.to_tensor(
            np.nanmean(self.icoads_y[-window:, ...], axis=(0, 2))[:, np.newaxis]
        )
        self.icoads_stds = self.to_tensor(
            np.nanstd(self.icoads_y[-window:, ...], axis=(0, 2))[:, np.newaxis]
        )
        self.icoads_index_offset = self.offsets["icoads"][self.start_date]
        return

    def load_igra(self):
        """
        Load the IGRA data
        """

        igra_y_path = self.data_path + "igra/1999_2021_igra_y.mmap"
        igra_y_shape = list(
            IGRA_Y_SHAPE_1D if self.time_freq != "6H" else IGRA_Y_SHAPE
        )
        self.igra_y = np.memmap(
            igra_y_path,
            dtype="float32",
            mode="r",
            shape=tuple(igra_y_shape),
        )

        self.igra_x = np.copy(
            np.memmap(
                self.data_path + "igra/1999_2021_igra_x.mmap",
                dtype="float32",
                mode="r",
                shape=IGRA_X_SHAPE,
            )
        )
        self.igra_x = self.igra_x / LATLON_SCALE_FACTOR

        self.igra_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_igra.npy")
        )
        self.igra_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_igra.npy")
        )

        self.igra_index_offset = self.offsets["igra"][self.start_date]

        return

    def load_amsua(self):
        """
        Load the AMSU-A data
        """

        amsua_path = self.data_path + "amsua/2007_2021_amsua.mmap"
        amsua_shape = list(AMSUA_Y_SHAPE)
        if self.time_freq != "6H":
            # Daily gridded obs on Grid A; AMSU-A layout is (time, nlat, nlon, channels).
            amsua_shape = list(AMSUA_Y_SHAPE_1D)
            amsua_shape[1] = self.nlat
            amsua_shape[2] = self.nlon
        if self.time_freq != "6H":
            assert_memmap_size("amsua", amsua_path, amsua_shape)
        self.amsua_y = np.memmap(
            amsua_path,
            dtype="float32",
            mode="r",
            shape=tuple(amsua_shape),
        )
        self.amsua_index_offset = self.offsets["amsua"][self.start_date]

        if self.time_freq != "6H":
            lon, lat = self._era5_grid_axes()
            lon = ((lon + 360) % 360) / LATLON_SCALE_FACTOR
            lat = lat / LATLON_SCALE_FACTOR
            self.amsua_x = [lon, lat]
        else:
            xx = np.linspace(-180, 179, 360, dtype=np.float32)
            xx = ((xx + 360) % 360) / LATLON_SCALE_FACTOR
            yy = np.linspace(90, -90, 180, dtype=np.float32) / LATLON_SCALE_FACTOR
            self.amsua_x = [xx, yy]

        self.amsua_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_amsua.npy")
        )
        self.amsua_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_amsua.npy")
        )

        return

    def load_amsub(self):
        """
        Load the AMSU-B data
        """

        amsub_path = self.data_path + "amsub_mhs/2007_2021_amsub.mmap"
        amsub_shape = list(AMSUB_Y_SHAPE)
        if self.time_freq != "6H":
            # Daily gridded obs on Grid A; AMSU-B layout is (time, nlon, nlat, channels).
            amsub_shape = list(AMSUB_Y_SHAPE_1D)
            amsub_shape[1] = self.nlon
            amsub_shape[2] = self.nlat
        if self.time_freq != "6H":
            assert_memmap_size("amsub", amsub_path, amsub_shape)
        self.amsub_y = np.memmap(
            amsub_path,
            dtype="float32",
            mode="r",
            shape=tuple(amsub_shape),
        )
        self.amsub_index_offset = self.offsets["amsub"][self.start_date]

        if self.time_freq != "6H":
            lon, lat = self._era5_grid_axes()
            lon = ((lon + 360) % 360) / LATLON_SCALE_FACTOR
            lat = lat / LATLON_SCALE_FACTOR
            self.amsub_x = [lon, lat]
        else:
            xx = np.linspace(0, 359, 360, dtype=np.float32)
            xx = ((xx + 360) % 360) / LATLON_SCALE_FACTOR
            yy = np.linspace(90, -90, 181, dtype=np.float32) / LATLON_SCALE_FACTOR
            self.amsub_x = [xx, yy]

        self.amsub_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_amsub.npy")
        )
        self.amsub_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_amsub.npy")
        )

        return

    def load_ascat_data(self):
        """
        Load the ASCAT data
        """

        ascat_path = self.data_path + "ascat/2007_2021_ascat.mmap"
        ascat_shape = list(ASCAT_Y_SHAPE)
        if self.time_freq != "6H":
            # Daily gridded obs on Grid A; ASCAT layout is (time, nlon, nlat, channels).
            ascat_shape = list(ASCAT_Y_SHAPE_1D)
            ascat_shape[1] = self.nlon
            ascat_shape[2] = self.nlat
        if self.time_freq != "6H":
            assert_memmap_size("ascat", ascat_path, ascat_shape)
        self.ascat_y = np.memmap(
            ascat_path,
            dtype="float32",
            mode="r",
            shape=tuple(ascat_shape),
        )
        self.ascat_index_offset = self.offsets["ascat"][self.start_date]

        if self.time_freq != "6H":
            lon, lat = self._era5_grid_axes()
            lon = ((lon + 360) % 360) / LATLON_SCALE_FACTOR
            lat = lat / LATLON_SCALE_FACTOR
            self.ascat_x = [lon, np.copy(lat[::-1])]
        else:
            xx = np.linspace(0, 359, 360, dtype=np.float32)
            xx = ((xx + 360) % 360) / LATLON_SCALE_FACTOR
            yy = np.linspace(-90, 90, 181, dtype=np.float32) / LATLON_SCALE_FACTOR
            self.ascat_x = [xx, np.copy(yy[::-1])]

        self.ascat_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_ascat.npy")
        )
        self.ascat_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_ascat.npy")
        )

        return

    def load_hirs_data(self):
        """
        Load the HIRS data
        """

        hirs_path = self.data_path + "hirs/2007_2021_hirs.mmap"
        hirs_shape = list(HIRS_Y_SHAPE)
        if self.time_freq != "6H":
            # Daily gridded obs on Grid A; HIRS layout is (time, nlon, nlat, channels).
            hirs_shape = list(HIRS_Y_SHAPE_1D)
            hirs_shape[1] = self.nlon
            hirs_shape[2] = self.nlat
        if self.time_freq != "6H":
            assert_memmap_size("hirs", hirs_path, hirs_shape)
        self.hirs_y = np.memmap(
            hirs_path,
            dtype="float32",
            mode="r",
            shape=tuple(hirs_shape),
        )
        self.hirs_index_offset = self.offsets["ascat"][self.start_date]

        if self.time_freq != "6H":
            lon, lat = self._era5_grid_axes()
            lon = ((lon + 360) % 360) / LATLON_SCALE_FACTOR
            lat = lat / LATLON_SCALE_FACTOR
            self.hirs_x = [lon, np.copy(lat[::-1])]
        else:
            xx = np.linspace(0, 359, 360, dtype=np.float32)
            xx = ((xx + 360) % 360) / LATLON_SCALE_FACTOR
            yy = np.linspace(-90, 90, 181, dtype=np.float32) / LATLON_SCALE_FACTOR
            self.hirs_x = [xx, np.copy(yy[::-1])]

        self.hirs_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/hirs_means.npy")
        )
        self.hirs_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/hirs_stds.npy")
        )

        return

    def load_sat_data(self):
        """
        Load the GRIDSAT data
        """

        sat_path = self.data_path + "gridsat/gridsat_data.mmap"
        sat_shape = list(GRIDSAT_Y_SHAPE)
        if self.time_freq != "6H":
            sat_shape = list(GRIDSAT_Y_SHAPE_1D)
        self.sat_y = np.memmap(
            sat_path,
            dtype="float32",
            mode="r",
            shape=tuple(sat_shape),
        )

        xx = np.load(self.data_path + "gridsat/sat_x.npy") / LATLON_SCALE_FACTOR
        yy = np.load(self.data_path + "gridsat/sat_y.npy") / LATLON_SCALE_FACTOR
        self.sat_x = [xx, yy]
        self.sat_index_offset = self.offsets["sat"][self.start_date]

        self.sat_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_sat.npy")
        )
        self.sat_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_sat.npy")
        )

        return

    def load_iasi(self):
        """
        Load the IASI data
        """

        iasi_path = self.data_path + "2007_2021_iasi_subset.mmap"
        iasi_shape = list(IASI_Y_SHAPE)
        if self.time_freq != "6H":
            # Daily gridded obs on Grid A; IASI layout is (time, nlon, nlat, channels).
            iasi_shape = list(IASI_Y_SHAPE_1D)
            iasi_shape[1] = self.nlon
            iasi_shape[2] = self.nlat
        if self.time_freq != "6H":
            assert_memmap_size("iasi", iasi_path, iasi_shape)
        self.iasi = np.memmap(
            iasi_path,
            dtype="float32",
            mode="r",
            shape=tuple(iasi_shape),
        )
        self.iasi_index_offset = self.offsets["ascat"][self.start_date]

        if self.time_freq != "6H":
            lon, lat = self._era5_grid_axes()
            lon = ((lon + 360) % 360) / LATLON_SCALE_FACTOR
            lat = lat / LATLON_SCALE_FACTOR
            self.iasi_x = [lon, np.copy(lat[::-1])]
        else:
            xx = np.linspace(0, 359, 360, dtype=np.float32)
            xx = ((xx + 360) % 360) / LATLON_SCALE_FACTOR
            yy = np.linspace(-90, 90, 181, dtype=np.float32) / LATLON_SCALE_FACTOR
            self.iasi_x = [xx, np.copy(yy[::-1])]

        self.iasi_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_iasi.npy")
        )
        self.iasi_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_iasi.npy")
        )

        return

    def _month_keys(self):
        """List of (year, month) tuples spanned by the run's date range (per-month files)."""
        periods = pd.period_range(self.start_date, self.end_date, freq="M")
        return [(p.year, p.month) for p in periods]

    def _era5_month_path(self, year, month):
        return os.path.join(
            self.data_path,
            "era5",
            f"era5_{self.era5_mode}_1_{self.freq_tag}_{year}-{month:02d}.memmap",
        )

    def _load_era5_monthly(self):
        """Open per-month ERA5 target memmaps keyed by (year, month).

        Channel count is inferred once from the first month (assuming its frame count is
        correct), then EVERY month's frame count is hard-asserted to
        ``days_in_month * frames_per_day`` so a missing/partial month fails loudly rather than
        silently shifting rows (obs/target alignment depends on it).
        """
        era5 = {}
        channels = None
        per_frame = self.nlon * self.nlat * 4
        for (year, month) in self._month_keys():
            path = self._era5_month_path(year, month)
            nbytes = os.path.getsize(path)
            frames_expected = days_in_month(year, month) * self.frames_per_day
            if channels is None:
                denom = frames_expected * per_frame
                if denom == 0 or nbytes % denom != 0:
                    raise ValueError(
                        f"{path}: size {nbytes} not divisible by frame block {denom} "
                        f"(frames={frames_expected}, nlon={self.nlon}, nlat={self.nlat})"
                    )
                channels = nbytes // denom
            if nbytes != frames_expected * channels * per_frame:
                raise AssertionError(
                    f"{path}: {nbytes // (channels * per_frame)} frames != expected "
                    f"{frames_expected} (days_in_month*{self.frames_per_day}); "
                    "obs/target alignment requires an exact match"
                )
            era5[(year, month)] = np.memmap(
                path,
                dtype="float32",
                mode="r",
                shape=(frames_expected, channels, self.nlon, self.nlat),
            )
        self.era5_channels = channels  # used to cross-check the 00z background channel count
        return era5

    def _background_month_path(self, year, month):
        return os.path.join(
            self.data_path,
            "era5",
            f"background_{self.era5_mode}_1_{year}-{month:02d}.memmap",
        )

    def _load_background_monthly(self):
        """Open per-month daily 00z BACKGROUND memmaps keyed by (year, month).

        This populates the encoder's "climatology" input slot on the rtma_surface 15-min path,
        but it is a date-specific daily 00z background/prior -- NOT a multi-year climatology.
        Each month is ``(days_in_month, C, nlon, nlat)`` with ONE 00z frame per day. The field is
        expected to be **already normalized at build time with the TARGET mean/std** (loader
        feeds it as-is, like the climatology slot today). Hard-asserts (review point 6):
        frames == days_in_month, C == target channels, spatial == (nlon, nlat).
        """
        background = {}
        target_channels = getattr(self, "era5_channels", None)
        per_frame = self.nlon * self.nlat * 4
        channels = None
        for (year, month) in self._month_keys():
            path = self._background_month_path(year, month)
            nbytes = os.path.getsize(path)
            days = days_in_month(year, month)
            if channels is None:
                denom = days * per_frame
                if denom == 0 or nbytes % denom != 0:
                    raise ValueError(
                        f"{path}: size {nbytes} not divisible by (days*nlon*nlat*4)={denom} "
                        f"(days={days}, nlon={self.nlon}, nlat={self.nlat})"
                    )
                channels = nbytes // denom
                if target_channels is not None and channels != target_channels:
                    raise AssertionError(
                        f"{path}: background channels {channels} != target channels "
                        f"{target_channels}; the 00z background must match the target fields"
                    )
            if nbytes != days * channels * per_frame:
                raise AssertionError(
                    f"{path}: {nbytes // (channels * per_frame)} frames != expected {days} "
                    "(days_in_month); the 00z background must have exactly one frame per day"
                )
            background[(year, month)] = np.memmap(
                path,
                dtype="float32",
                mode="r",
                shape=(days, channels, self.nlon, self.nlat),
            )
        self._background_channels = channels
        return background

    def _load_obs_monthly(self, var, stations):
        """Open per-month surface-obs value memmaps for one variable, keyed by (year, month).

        Each month file is ``(days_in_month * frames_per_day, stations)``; the frame count is
        hard-asserted so obs and target stay aligned.
        """
        out = {}
        per_frame = stations * 4
        for (year, month) in self._month_keys():
            path = os.path.join(
                self.data_path,
                "hadisd_processed",
                f"{var}_vals_{year}-{month:02d}.memmap",
            )
            nbytes = os.path.getsize(path)
            frames_expected = days_in_month(year, month) * self.frames_per_day
            if nbytes != frames_expected * per_frame:
                raise AssertionError(
                    f"{path}: {nbytes // per_frame} frames != expected {frames_expected} "
                    f"(days_in_month*{self.frames_per_day}); check the 15-min month file"
                )
            out[(year, month)] = np.memmap(
                path, dtype="float32", mode="r", shape=(frames_expected, stations)
            )
        return out

    def load_hadisd(self, mode):
        """
        Load the HADISD data
        """

        self.hadisd_x = []
        self.hadisd_alt = []
        self.hadisd_y = []
        # RTMA surface set uses specific humidity (sh) in place of HadISD dewpoint (tds);
        # the per-variable loading/encoding path is otherwise identical (option A reuse).
        if getattr(self, "surface_only", False):
            hadisd_vars = ["tas", "sh", "psl", "u", "v"]
        else:
            hadisd_vars = ["tas", "tds", "psl", "u", "v"]
        for var in hadisd_vars:
            lon = lon_to_0_360(
                np.load(
                    self.data_path + "hadisd_processed/{}_lon_{}.npy".format(var, mode)
                )
            )
            lat = np.load(
                self.data_path + "hadisd_processed/{}_lat_{}.npy".format(var, mode)
            )
            alt = np.load(
                self.data_path + "hadisd_processed/{}_alt_{}.npy".format(var, mode)
            )
            if self.monthly:
                # Per-month value files: {var}_vals_<YYYY>-<MM>.memmap. Coords above stay
                # static (single file); only the values are per-month.
                stations = lon.shape[0]
                vals = self._load_obs_monthly(var, stations)
            else:
                vals_path = (
                    self.data_path + "hadisd_processed/{}_vals_{}.memmap".format(var, self.mode)
                )
                if self.time_freq == "6H":
                    shape = get_hadisd_shape(mode)
                else:
                    stations = lon.shape[0]
                    time_dim = self._infer_time_dim(vals_path, [stations])
                    shape = (time_dim, stations)
                vals = np.memmap(
                    vals_path,
                    dtype="float32",
                    mode="r",
                    shape=shape,
                )

            self.hadisd_x.append(np.stack([lon, lat], axis=-1) / LATLON_SCALE_FACTOR)
            self.hadisd_alt.append(alt)
            self.hadisd_y.append(vals)

        if not self.monthly:
            self.hadisd_index_offset = self.offsets["hadisd"][self.start_date]

        self.hadisd_means = [
            self.to_tensor(
                np.load(
                    self.aux_data_path + "norm_factors/mean_hadisd_{}.npy".format(var)
                )
            )
            for var in hadisd_vars
        ]
        self.hadisd_stds = [
            self.to_tensor(
                np.load(
                    self.aux_data_path + "norm_factors/std_hadisd_{}.npy".format(var)
                )
            )
            for var in hadisd_vars
        ]

        return

    def load_era5(self, year):
        """
        Load the ERA5 training data
        """

        if year % 4 == 0:
            days = 366
        else:
            days = 365
        d = days * (4 if self.time_freq == "6H" else 1)

        if self.era5_mode == "sfc":
            levels = 4
        elif self.era5_mode == "13u":
            levels = 69
        elif self.era5_mode in ("4u_sfc", "rtma_ok_sfc"):
            # rtma_ok_sfc = RTMA OK Phase 1 surface target (5 channels); channel count is
            # inferred from the memmap file size rather than hard-coded.
            levels = None
        else:
            levels = 24

        x = self.nlon
        y = self.nlat
        freq_tag = "6" if self.time_freq == "6H" else "1d"
        memmap_path = era5_memmap_path(
            self.data_path, self.era5_mode, freq_tag, year
        )
        if levels is None:
            nbytes = os.path.getsize(memmap_path)
            denom = d * x * y * 4
            if nbytes % denom != 0:
                raise ValueError(f"File size not divisible by expected frame size: {memmap_path}")
            levels = nbytes // denom
        mmap = np.memmap(
            memmap_path,
            dtype="float32",
            mode="r",
            shape=(d, levels, x, y),
        )
        return mmap

    def norm_era5(self, x):

        x = (x - self.means) / self.stds
        return x

    def unnorm_era5(self, x):

        x = x * self.stds + self.means
        return x

    def norm_data(self, x, means, stds):
        return (x - means) / stds

    def norm_hadisd(self, x):
        for i in range(5):
            x[i] = (x[i] - self.hadisd_means[i]) / self.hadisd_stds[i]
        return x

    def __len__(self):
        return self.index.shape[0] - 1 - 1

    def to_tensor(self, arr):
        return torch.from_numpy(arr).float().to(self.device)

    def get_time_aux(self, current_date):
        """
        Return the auxiliary temporal channels given a date
        """

        doy = current_date.dayofyear
        year = (current_date.year - 2007) / 15
        # Fractional hour so sub-hourly (e.g. 15-min) steps are distinguishable. Backward
        # compatible: 6H/1D timestamps have minute==0, so this equals the integer hour.
        time_of_day = current_date.hour + current_date.minute / 60
        return np.array(
            [
                np.cos(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.sin(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.cos(np.pi * 2 * time_of_day / 24),
                np.sin(np.pi * 2 * time_of_day / 24),
                year,
            ]
        )


class WeatherDatasetAssimilation(WeatherDataset):
    """
    Encoder training loader
    """

    def __init__(
        self,
        device,
        hadisd_mode,
        start_date,
        end_date,
        lead_time,
        era5_mode="sfc",
        filter_dates=None,
        var_start=0,
        var_end=24,
        diff=False,
        two_frames=False,
        data_path=None,
        aux_data_path=None,
        disable_igra=False,
        time_freq="6H",
        obs_set="all",
    ):

        super().__init__(
            device,
            hadisd_mode,
            start_date,
            end_date,
            lead_time,
            era5_mode,
            filter_dates=filter_dates,
            diff=diff,
            data_path=data_path,
            aux_data_path=aux_data_path,
            disable_igra=disable_igra,
            time_freq=time_freq,
            obs_set=obs_set,
        )

        # Setup

        self.var_start = var_start
        self.var_end = var_end
        self.diff = diff
        self.two_frames = two_frames
        if getattr(self, "monthly", False) and self.two_frames:
            raise NotImplementedError(
                "monthly per-month files (rtma_surface + 15min) do not support two_frames: "
                "the two-frame / year-boundary path assumes per-year ERA5 memmaps."
            )

    def load_era5_time(self, index):
        """
        ERA5 ground truth data loading
        """

        date = self.dates[index]
        if self.monthly:
            (year, month), frame_in_month = month_frame(date, self.step_minutes)
            era5 = self.era5_sfc[(year, month)][frame_in_month, ...]
        else:
            year = date.year
            if self.time_freq == "6H":
                doy = (date.dayofyear - 1) * 4 + (date.hour // 6)
            else:
                doy = date.dayofyear - 1
            era5 = self.era5_sfc[year - int(self.start_date[:4])][doy, ...]
        era5 = np.copy(era5)
        if not getattr(self, "_debug_era5_shapes", False):
            print(
                "[DEBUG] era5_mode",
                self.era5_mode,
                "time_freq",
                self.time_freq,
                "era5_frame_shape",
                era5.shape,
                "means_shape",
                getattr(self, "means", None).shape,
                "stds_shape",
                getattr(self, "stds", None).shape,
            )
            self._debug_era5_shapes = True
        if self.diff:
            era5 = era5 - self.era5_mean_spatial
        era5 = self.norm_era5(era5[np.newaxis, ...])[0, ...]
        return era5

    def load_year_end(self, year, doy):
        data_1 = self.era5_sfc[year - int(self.start_date[:4])][doy : doy + 1, ...]
        missing = self.lead_time - data_1.shape[0] + 1
        data_2 = self.era5_sfc[year - int(self.start_date[:4]) + 1][:missing, ...]
        data = np.concatenate([data_1, data_2])
        return data

    def load_era5_slice(self, index):
        """
        ERA5 ground truth data loading
        """

        date = self.dates[index]
        year = date.year
        if self.time_freq == "6H":
            doy = (date.dayofyear - 1) * 4
        else:
            doy = date.dayofyear - 1

        next_date = self.dates[index + 1]
        next_year = next_date.year

        if next_year != year:
            era5 = self.load_year_end(year, doy)
        else:
            era5 = self.era5_sfc[year - int(self.start_date[:4])][doy : doy + 1, ...]

        era5 = self.norm_era5(np.copy(era5))
        return era5

    def __getitem__(self, index):

        if self.two_frames:
            # Case 1: loading t=0 and t=-1
            index = index + 1
            current = self.get_index(index, "current")
            prev = self.get_index(index - 1, "prev")
            current["y_target"] = current["y_target_current"]

            return {**current, **prev}
        else:
            # Case 2: loading t=0
            current = self.get_index(index, "current")
            current["y_target"] = current["y_target_current"]

            return {**current}

    def unnorm_pred(self, x):
        dev = x.device
        x = x.detach().cpu().numpy()

        x = (
            x
            * self.stds[np.newaxis, ...].transpose(0, 2, 3, 1)[
                ..., self.var_start : self.var_end
            ]
            + self.means[np.newaxis, ...].transpose(0, 2, 3, 1)[
                ..., self.var_start : self.var_end
            ]
        )
        if bool(self.diff):
            x = (
                x
                + self.era5_mean_spatial[np.newaxis, ...].transpose(0, 3, 2, 1)[
                    ..., self.var_start : self.var_end
                ]
            )
        return torch.from_numpy(x).float().to(dev)

    def unnorm_base_context(self, x):
        dev = x.device
        x = x.detach().cpu().numpy()
        x = x * self.stds[np.newaxis, ...] + self.means[np.newaxis, ...]
        return torch.from_numpy(x).float().to(dev)

    def get_index(self, index, prefix):
        """
        Load data for the relevant index respecting different offsets depending on the modality
        """

        index = self.index[index]
        date = self.dates[index]

        # HadISD (always loaded -- the surface obs used by both obs_set modes)
        x_context_hadisd = self.hadisd_x
        if self.monthly:
            # Per-month values: each hadisd_y entry is a {(year, month): memmap} dict; read the
            # SAME (month, frame) the target uses so obs and target stay aligned.
            (m_year, m_month), frame_in_month = month_frame(date, self.step_minutes)
            y_context_hadisd = [
                v[(m_year, m_month)][frame_in_month, :] for v in self.hadisd_y
            ]
        else:
            y_context_hadisd = [
                i[index + self.hadisd_index_offset, :] for i in self.hadisd_y
            ]
        x_context_hadisd = [self.to_tensor(i).permute(1, 0) for i in x_context_hadisd]
        y_context_hadisd = [self.to_tensor(i) for i in y_context_hadisd]
        y_context_hadisd = self.norm_hadisd(y_context_hadisd)

        # ERA5
        era5 = self.to_tensor(self.load_era5_time(index))
        era5_target = era5.permute(2, 1, 0)
        era5_x = self.era5_x

        # AUxiliary variables
        aux_time = self.to_tensor(self.get_time_aux(date))
        if self.monthly:
            # Normalized daily 00z background/prior (reuses the climatology slot; NOT a multi-year
            # climatology). All 96 intraday samples of a day share that day's 00z field.
            climatology = self.background[(date.year, date.month)][date.day - 1, ...]
        elif self.time_freq == "6H":
            climatology = self.climatology[date.hour // 6, date.dayofyear - 1, ...]
        else:
            climatology = self.climatology[0, date.dayofyear - 1, ...]

        task = {
            "x_context_hadisd_{}".format(prefix): x_context_hadisd,
            "y_context_hadisd_{}".format(prefix): y_context_hadisd,
            "climatology_{}".format(prefix): self.to_tensor(climatology),
            "y_target_{}".format(prefix): era5_target[
                ..., self.var_start : self.var_end
            ],
            "era5_x_{}".format(prefix): era5_x,
            "era5_elev_{}".format(prefix): self.era5_elev,
            "era5_lonlat_{}".format(prefix): self.era5_lonlat,
            "aux_time_{}".format(prefix): aux_time,
            "lt": torch.Tensor([self.var_start]),
        }

        # Global observation modalities -- skipped entirely for obs_set="rtma_surface".
        if not self.surface_only:
            # ICOADS
            icoads_y = self.icoads_y[index + self.icoads_index_offset, ...]
            if getattr(self, "icoads_x_is_static", False):
                icoads_x = [self.icoads_x[:, 0], self.icoads_x[:, 1]]
            else:
                icoads_x = self.icoads_x[index + self.icoads_index_offset, ...]
                icoads_x = [icoads_x[0, :], icoads_x[1, :]]
            icoads_x = [self.to_tensor(i) for i in icoads_x]
            icoads_y = self.to_tensor(icoads_y)
            icoads_y = self.norm_data(icoads_y, self.icoads_means, self.icoads_stds)

            # GRIDSAT
            sat_y = self.sat_y[index + self.sat_index_offset, ...]
            sat_x = [self.to_tensor(i) for i in self.sat_x]
            sat_y = self.to_tensor(sat_y)
            sat_y = self.norm_data(sat_y, self.sat_means, self.sat_stds)

            # AMSU-A
            amsua_y = self.to_tensor(self.amsua_y[index + self.amsua_index_offset, ...])
            amsua_y[amsua_y < -998] = torch.nan
            amsua_x = [self.to_tensor(i) for i in self.amsua_x]
            amsua_y[amsua_y < -998] = np.nan
            amsua_y = self.norm_data(amsua_y, self.amsua_means, self.amsua_stds)

            # AMSU-B
            amsub_y = self.to_tensor(self.amsub_y[index + self.amsub_index_offset, ...])
            amsub_y[amsub_y < -998] = torch.nan
            amsub_x = [self.to_tensor(i) for i in self.amsub_x]
            amsub_y[amsub_y < -998] = np.nan
            amsub_y = self.norm_data(amsub_y, self.amsub_means, self.amsub_stds)

            # IASI
            iasi_y = self.to_tensor(self.iasi[index + self.iasi_index_offset, ...])
            iasi_x = [self.to_tensor(i) for i in self.iasi_x]
            iasi_y = self.norm_data(iasi_y, self.iasi_means, self.iasi_stds)

            # ASCAT
            ascat_y = self.to_tensor(self.ascat_y[index + self.ascat_index_offset, ...])
            ascat_x = [self.to_tensor(i) for i in self.ascat_x]
            ascat_y[..., 4][ascat_y[..., 4] < -990] = np.nan
            ascat_y = self.norm_data(ascat_y, self.ascat_means, self.ascat_stds)

            # HIRS
            hirs_y = self.to_tensor(self.hirs_y[index + self.hirs_index_offset, ...])
            hirs_y[hirs_y < -998] = np.nan
            hirs_x = [self.to_tensor(i) for i in self.hirs_x]
            hirs_y = self.norm_data(hirs_y, self.hirs_means, self.hirs_stds)

            task.update(
                {
                    "sat_x_{}".format(prefix): sat_x,
                    "sat_{}".format(prefix): sat_y,
                    "icoads_x_{}".format(prefix): icoads_x,
                    "icoads_{}".format(prefix): icoads_y,
                    "amsua_{}".format(prefix): amsua_y,
                    "amsua_x_{}".format(prefix): amsua_x,
                    "amsub_{}".format(prefix): amsub_y,
                    "amsub_x_{}".format(prefix): amsub_x,
                    "iasi_{}".format(prefix): iasi_y,
                    "iasi_x_{}".format(prefix): iasi_x,
                    "ascat_{}".format(prefix): ascat_y,
                    "ascat_x_{}".format(prefix): ascat_x,
                    "hirs_{}".format(prefix): hirs_y,
                    "hirs_x_{}".format(prefix): hirs_x,
                }
            )

            # IGRA (optional)
            if not self.disable_igra:
                igra_y = self.to_tensor(self.igra_y[index + self.igra_index_offset, ...])
                igra_x = [self.igra_x[:, 0], self.igra_x[:, 1]]
                igra_x = [self.to_tensor(i) for i in igra_x]
                igra_y = self.norm_data(igra_y, self.igra_means, self.igra_stds)
                task.update(
                    {
                        "igra_x_{}".format(prefix): igra_x,
                        "igra_{}".format(prefix): igra_y,
                    }
                )

        return task


class HadISDDataset(Dataset):
    """
    HadISD dataset for decoder training
    """

    def __init__(
        self,
        var,
        mode,
        device,
        start_date,
        end_date,
        data_path=None,
        aux_data_path=None,
        time_freq="6H",
    ):
        super().__init__()

        # Setup
        if not mode in ["train", "val", "test"]:
            raise Exception(f"mode is {mode}. Must be train, val, or test.")

        self.var = var
        self.mode = mode
        self.start_date = start_date
        self.device = device
        self.data_path = data_path or "path_to_data/"
        self.aux_data_path = aux_data_path or "path_to_auxiliary_data/"
        self.time_freq = time_freq
        self.offsets = build_offsets(self.time_freq)
        dates = pd.date_range(start_date, end_date, freq=self.time_freq)
        self.index = np.array(range(len(dates)))

        # Load the hadISD data
        self.load_hadisd()

    def load_hadisd(self):
        """
        Load the raw HadISD data
        """

        data_path = self.data_path
        aux_data_path = self.aux_data_path
        var = self.var
        mode = self.mode

        lon = lon_to_0_360(
            np.load(data_path + f"hadisd_processed/{var}_lon_{mode}.npy")
        )
        lat = np.load(data_path + f"hadisd_processed/{var}_lat_{mode}.npy")
        vals_path = data_path + f"hadisd_processed/{var}_vals_{mode}.memmap"
        if self.time_freq == "6H":
            shape = get_hadisd_shape(mode)
        else:
            stations = lon.shape[0]
            time_dim = self._infer_time_dim(vals_path, [stations])
            shape = (time_dim, stations)
        vals = np.memmap(
            vals_path,
            dtype="float32",
            mode="r",
            shape=shape,
        )
        self.hadisd_x = np.stack([lon, lat], axis=-1) / LATLON_SCALE_FACTOR
        self.hadisd_alt = np.load(
            data_path + f"hadisd_processed/{var}_alt_{mode}_final.npy"
        )
        self.hadisd_y = vals

        self.hadisd_index_offset = self.offsets["hadisd"][self.start_date]
        self.hadisd_means = self.to_tensor(
            np.load(aux_data_path + f"norm_factors/mean_hadisd_{var}.npy")
        )
        self.hadisd_stds = self.to_tensor(
            np.load(aux_data_path + f"norm_factors/std_hadisd_{var}.npy")
        )
        return

    def norm_hadisd(self, x):
        return (x - self.hadisd_means) / self.hadisd_stds

    def unnorm_pred(self, x):
        return self.hadisd_means + self.hadisd_stds * x

    def __len__(self):
        return self.index.shape[0] - 2

    def to_tensor(self, arr):
        return torch.from_numpy(np.array(arr)).float().to(self.device)

    def _infer_time_dim(self, path, fixed_shape):
        file_bytes = os.path.getsize(path)
        denom = 4
        for dim in fixed_shape:
            denom *= dim
        if file_bytes % denom != 0:
            raise ValueError(f"File size not divisible by expected frame size: {path}")
        return file_bytes // denom

    def __getitem__(self, index):
        index = self.index[index]

        # Get longitude-latitude locations
        x_target = self.to_tensor(self.hadisd_x).permute(1, 0)

        # Get altitude and normalise
        m_alt = np.expand_dims(np.load("path_to_mean_alt.npy"), 1)
        s_alt = np.expand_dims(np.load("path_to_std_alt.npy"), 1)
        alt_target = self.to_tensor((self.hadisd_alt - m_alt) / s_alt)[:, :]

        # Get observations
        y_target = self.norm_hadisd(
            self.to_tensor(self.hadisd_y[index + self.hadisd_index_offset, :])
        )

        assert x_target.shape[0] == 2
        n_stations = x_target.shape[1]
        assert alt_target.shape[1] == n_stations
        assert y_target.shape[0] == n_stations

        return {"x": x_target, "altitude": alt_target, "y": y_target}


class AardvarkICDataset(Dataset):
    """
    Helper dataset to handle initial condition loading for decoder training
    """

    def __init__(
        self,
        device,
        start_date,
        end_date,
        lead_time=0,
        era5_mode="4u",
        data_path=None,
        aux_data_path=None,
        encoder_predictions_path=None,
        time_freq="6H",
    ):
        super().__init__()

        # Setup
        self.data_path = data_path or "path_to_data/"
        self.aux_data_path = aux_data_path or "path_to_auxiliary_data/"
        self.encoder_predictions_path = (
            encoder_predictions_path or "path_to_encoder_predictions/"
        )
        self.era5_mode = era5_mode
        self.time_freq = time_freq
        offset_factor = 4 if self.time_freq == "6H" else 1
        channels = 30 if self.era5_mode == "4u_sfc" else 24
        # Encoder-prediction memmaps are stored on the data/target grid (Grid A).
        self.nlon, self.nlat = grid_dims_from_files(self.data_path)

        if lead_time == 0:
            # If leadtime is 0 load the output of the encoder
            if start_date == "2007-01-02" and end_date == "2017-12-31":
                ic_fname = "ic_train.mmap"
            elif start_date == "2019-01-01" and end_date == "2019-12-01":
                ic_fname = "ic_val.mmap"
            elif start_date == "2018-01-01" and end_date == "2018-12-31":
                ic_fname = "ic_test.mmap"
            else:
                print((start_date, end_date))
                raise Exception("Invalid start and end date")

            dates = pd.date_range(start_date, end_date, freq=self.time_freq)

            self.data = np.memmap(
                self.encoder_predictions_path + ic_fname,
                dtype="float32",
                mode="r",
                shape=(len(dates), self.nlat, self.nlon, channels),  # shape of the output
            )
        else:
            # if leadtime >0 load the forecast generated from the encoder prediction
            if start_date == "2007-01-02" and end_date == "2017-12-31":
                ic_fname = f"ic_train_{lead_time}.mmap"

            elif start_date == "2019-01-01" and end_date == "2019-12-01":
                ic_fname = f"ic_val_{lead_time}.mmap"
            elif start_date == "2018-01-01" and end_date == "2018-12-31":
                ic_fname = f"ic_test_{lead_time}.mmap"
            else:
                print((start_date, end_date))
                raise Exception("Invalid start and end date.")

            dates = pd.date_range(start_date, end_date, freq=self.time_freq)[
                (lead_time) * offset_factor :
            ]
            ic_shape = (len(dates), self.nlat, self.nlon, channels)

            self.data = np.memmap(
                self.data_path + "forecast_finetune/" + ic_fname,
                dtype="float32",
                mode="r",
                shape=ic_shape,
            )

        self.device = device

        # Normalisation
        mean_factors_path = norm_mean_path(self.aux_data_path, self.era5_mode)
        std_factors_path = norm_std_path(self.aux_data_path, self.era5_mode)
        self.means = np.load(mean_factors_path)[:, np.newaxis, np.newaxis, ...]
        self.stds = np.load(std_factors_path)[:, np.newaxis, np.newaxis, ...]

    def __getitem__(self, index):
        # Load Aardvark prediction and normalise
        data_raw = np.transpose(np.copy(self.data[index, :, :, :]), (2, 1, 0))
        data = (data_raw - self.means) / self.stds
        return torch.from_numpy(data).to(self.device)


class WeatherDatasetDownscaling(Dataset):
    """
    Main decoder training dataset. Uses AardvarkICDataset and HadISDDataset to
    handle processor output and station data
    """

    def __init__(
        self,
        device,
        hadisd_mode,
        start_date,
        end_date,
        context_mode,
        era5_mode="sfc",
        hadisd_var="tas",
        lead_time=1,
        data_path=None,
        aux_data_path=None,
        time_freq="6H",
    ):
        # The context mode determines whether we make use of ERA5 or our own ICs.
        if not context_mode in ["era5", "aardvark"]:
            raise Exception(
                f"context_mode must be era5 or aardvark, got {context_mode}"
            )

        super().__init__()

        # Setup
        self.lead_time = lead_time

        self.device = device
        self.data_path = data_path or "path_to_data/"
        self.aux_data_path = aux_data_path or "path_to_auxiliary_data/"
        self.start_date = start_date
        self.end_date = end_date
        self.era5_mode = era5_mode
        self.context_mode = context_mode
        self.time_freq = time_freq
        self.offset_factor = 4 if self.time_freq == "6H" else 1

        self.dates = pd.date_range(start_date, end_date, freq=self.time_freq)
        self.index = np.array(range(len(self.dates)))

        # Load ERA5 data for pre-training
        self.era5_sfc = [
            self.load_era5(year)
            for year in range(int(start_date[:4]), int(end_date[:4]) + 1)
        ]

        raw_era5_lon = np.load(loader_grid_x_path(self.data_path))
        raw_era5_lat = np.load(loader_grid_y_path(self.data_path))
        self.era5_x = [
            self.to_tensor(raw_era5_lon) / LATLON_SCALE_FACTOR,
            self.to_tensor(raw_era5_lat) / LATLON_SCALE_FACTOR,
        ]
        self.nlon = self.era5_x[0].shape[0]
        self.nlat = self.era5_x[1].shape[0]

        # Load orography. Raw file is (channels, nlat, nlon); post-permute (channels, nlon, nlat).
        elev_path = elev_vars_path(self.data_path)
        raw_elev = np.load(elev_path)
        assert_grid_match("elev_vars (raw)", raw_elev.shape[1:], (self.nlat, self.nlon))
        self.era5_elev = self.to_tensor(raw_elev).permute(0, 2, 1)
        assert_grid_match(
            "era5_elev (loader)", self.era5_elev.shape[1:], (self.nlon, self.nlat)
        )

        # Normalisation
        mean_factors_path = norm_mean_path(self.aux_data_path, era5_mode)
        std_factors_path = norm_std_path(self.aux_data_path, era5_mode)
        self.means = np.load(mean_factors_path)[:, np.newaxis, np.newaxis, ...]
        self.stds = np.load(std_factors_path)[:, np.newaxis, np.newaxis, ...]

        # HadISD data
        self.hadisd_data = HadISDDataset(
            var=hadisd_var,
            mode=hadisd_mode,
            device=device,
            start_date=start_date,
            end_date=end_date,
            data_path=self.data_path,
            aux_data_path=self.aux_data_path,
            time_freq=self.time_freq,
        )

        if context_mode == "aardvark":
            # Load the Aardvark encoder predictions
            self.aardvark_data = AardvarkICDataset(
                device,
                start_date,
                end_date,
                lead_time,
                era5_mode=era5_mode,
                data_path=self.data_path,
                aux_data_path=self.aux_data_path,
                time_freq=self.time_freq,
            )

    def load_era5(self, year):
        """
        Load the raw ERA5 data
        """

        if year % 4 == 0:
            d = 366 * self.offset_factor
        else:
            d = 365 * self.offset_factor

        if self.era5_mode == "sfc":
            levels = 4
        elif self.era5_mode == "13u":
            levels = 69
        elif self.era5_mode in ("4u_sfc", "rtma_ok_sfc"):
            # rtma_ok_sfc = RTMA OK Phase 1 surface target (5 channels); channel count is
            # inferred from the memmap file size rather than hard-coded.
            levels = None
        else:
            levels = 24

        x = self.nlon
        y = self.nlat
        freq_tag = "6" if self.time_freq == "6H" else "1d"
        memmap_path = era5_memmap_path(
            self.data_path, self.era5_mode, freq_tag, year
        )
        if levels is None:
            nbytes = os.path.getsize(memmap_path)
            denom = d * x * y * 4
            if nbytes % denom != 0:
                raise ValueError(f"File size not divisible by expected frame size: {memmap_path}")
            levels = nbytes // denom
        mmap = np.memmap(
            memmap_path,
            dtype="float32",
            mode="r",
            shape=(d, levels, x, y),
        )
        return mmap

    def norm_era5(self, x):
        x = (x - self.means) / self.stds
        return x

    def unnorm_era5(self, x):
        x = x * self.stds + self.means
        return x

    def unnorm_pred(self, x):
        return self.hadisd_data.unnorm_pred(x)

    def norm_data(self, x, means, stds):
        return (x - means) / stds

    def __len__(self):
        return self.index.shape[0] - (self.lead_time) * self.offset_factor

    def to_tensor(self, arr):
        return torch.from_numpy(np.array(arr)).float().to(self.device)

    def get_time_aux(self, current_date):
        """
        Get auxiliary time variables for a given date
        """

        doy = current_date.dayofyear
        year = (current_date.year - 2007) / 15
        # Fractional hour so sub-hourly (e.g. 15-min) steps are distinguishable. Backward
        # compatible: 6H/1D timestamps have minute==0, so this equals the integer hour.
        time_of_day = current_date.hour + current_date.minute / 60
        return np.array(
            [
                np.cos(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.sin(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.cos(np.pi * 2 * time_of_day / 24),
                np.sin(np.pi * 2 * time_of_day / 24),
                year,
            ]
        )

    def load_era5_time(self, index):
        """
        Load ERA5 training data
        """

        date = self.dates[index]
        year = date.year
        if self.time_freq == "6H":
            hour = date.hour
            doy = (date.dayofyear - 1) * 4 + (hour // 6)
        else:
            doy = date.dayofyear - 1

        era5 = self.era5_sfc[year - int(self.start_date[:4])][doy, ...]
        era5 = np.copy(era5)
        era5 = self.norm_era5(era5[np.newaxis, ...])[0, ...]
        return era5

    def load_year_end(self, year, doy):
        data_1 = self.era5_sfc[year - int(self.start_date[:4])][doy : doy + 1, ...]
        missing = self.lead_time - data_1.shape[0] + 1
        data_2 = self.era5_sfc[year - int(self.start_date[:4]) + 1][:missing, ...]
        data = np.concatenate([data_1, data_2])
        return data

    def __getitem__(self, index):

        index = self.index[index]
        date = self.dates[index + self.offset_factor * self.lead_time]

        # Get HadISD data
        hadisd_slice = self.hadisd_data[index + self.offset_factor * self.lead_time]

        # Get lon-lat
        x_context = self.era5_x
        n_lon = x_context[0].shape[0]
        n_lat = x_context[1].shape[0]

        # Get auxiliary time
        aux_time = torch.reshape(self.to_tensor(self.get_time_aux(date)), (-1, 1, 1))

        # Load the context (either aardvark or ERA5 for use in pre-training)
        if self.context_mode == "era5":
            y_context_obs = self.to_tensor(
                self.load_era5_time(index + self.offset_factor * self.lead_time)
            )

        elif self.context_mode == "aardvark":
            y_context_obs = self.aardvark_data[index]

        else:
            raise Exception

        y_context = torch.cat(
            [
                y_context_obs,
                self.era5_elev.permute(0, 2, 1),
                aux_time.repeat(1, n_lon, n_lat),
            ]
        )

        assert y_context.shape[1] == n_lon
        assert y_context.shape[2] == n_lat

        x = hadisd_slice["x"]
        alt = hadisd_slice["altitude"]
        y = hadisd_slice["y"]

        return {
            "x_target": x,
            "alt_target": alt,
            "y_target": y,
            "y_context": y_context,
            "x_context": x_context,
            "aux_time": aux_time,
            "lt": torch.Tensor([0]),
        }


class ForecasterDatasetDownscaling(Dataset):
    """
    Dataset to generate decoder predictions from pre-saved Aardvark forecasts
    """

    def __init__(
        self,
        start_date,
        end_date,
        lead_time,
        hadisd_var,
        mode,
        device,
        forecast_path,
        era5_mode="4u",
        region="global",
        data_path=None,
        aux_data_path=None,
        time_freq="6H",
    ):
        super().__init__()

        # Setup

        if not mode in ["train", "val", "test"]:
            raise Exception(f"Mode is {mode}. Must be either train, val, or test")

        self.device = device
        self.start_date = start_date
        self.end_date = end_date
        self.lead_time = lead_time
        self.mode = mode
        self.era5_mode = era5_mode
        self.data_path = data_path or "path_to_data/"
        self.aux_data_path = aux_data_path or "path_to_auxiliary_data/"
        self.time_freq = time_freq
        self.offset = np.timedelta64(lead_time, "D").astype("timedelta64[ns]")
        self.offset_factor = 4 if self.time_freq == "6H" else 1
        self.channels = 30 if self.era5_mode == "4u_sfc" else 24

        self.dates = pd.date_range(start_date, end_date, freq=self.time_freq)[:-30]

        # Normalisation
        self.means = np.load(
            norm_mean_path(self.aux_data_path, self.era5_mode)
        )
        self.stds = np.load(
            norm_std_path(self.aux_data_path, self.era5_mode)
        )

        # Load auxiliary data
        self.load_npy_file()
        raw_era5_lon = np.load(loader_grid_x_path(self.data_path))
        raw_era5_lat = np.load(loader_grid_y_path(self.data_path))
        self.era5_x = [
            self.to_tensor(raw_era5_lon) / LATLON_SCALE_FACTOR,
            self.to_tensor(raw_era5_lat) / LATLON_SCALE_FACTOR,
        ]
        self.nlon = self.era5_x[0].shape[0]
        self.nlat = self.era5_x[1].shape[0]
        elev_path = elev_vars_path(self.data_path)
        raw_elev = np.load(elev_path)
        assert_grid_match("elev_vars (raw)", raw_elev.shape[1:], (self.nlat, self.nlon))
        self.era5_elev = self.to_tensor(raw_elev).permute(0, 2, 1)
        assert_grid_match(
            "era5_elev (loader)", self.era5_elev.shape[1:], (self.nlon, self.nlat)
        )

        # Load hadISD
        self.hadisd_data = HadISDDataset(
            var=hadisd_var,
            mode="train",
            device=device,
            start_date=start_date,
            end_date=end_date,
            data_path=self.data_path,
            aux_data_path=self.aux_data_path,
            time_freq=self.time_freq,
        )

        # Subset to region
        self.region = region
        if self.region != "global":
            self.mask = np.load(
                self.data_path + f"hadisd_processed/tas_mask_train_{region}.npy"
            )

    def date_range(self):
        return np.arange(
            start=np.datetime64(self.start_date).astype("datetime64[ns]"),
            stop=np.datetime64(self.end_date).astype("datetime64[ns]"),
            step=np.timedelta64(1, "D").astype("timedelta64[ns]"),
        )

    def load_npy_file(self):
        """
        Load the pre-saved Aardvark forecasts
        """

        dates = pd.date_range(self.start_date, self.end_date, freq=self.time_freq)

        if self.mode == "train":
            dates = dates[: -(10 * self.offset_factor)]  # Need 10 day offset at end of year

        self.Y_context = np.memmap(
            "path_to_forecasts/forecast_{}.mmap".format(self.mode),
            dtype="float32",
            mode="r",
            shape=(len(dates), self.nlat, self.nlon, self.channels, 11),
        )

        return

    def norm_era5(self, x):
        return (x - self.means) / self.stds

    def norm_hadisd(self, x):
        return self.hadisd_data.norm_hadisd(x)

    def unnorm_pred(self, x):
        return self.hadisd_data.unnorm_pred(x)

    def __len__(self):
        return len(self.dates) - (10 * self.offset_factor)  # Need 10 day offset at end of year

    def to_tensor(self, arr):
        return torch.from_numpy(np.array(arr)).float().to(self.device)

    def get_time_aux(self, index):
        """
        Get the auxiliary time variables
        """

        current_date = (self.dates + self.offset)[index]
        doy = current_date.dayofyear
        year = (current_date.year - 2007) / 15
        # Fractional hour so sub-hourly (e.g. 15-min) steps are distinguishable. Backward
        # compatible: 6H/1D timestamps have minute==0, so this equals the integer hour.
        time_of_day = current_date.hour + current_date.minute / 60
        return np.array(
            [
                np.cos(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.sin(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.cos(np.pi * 2 * time_of_day / 24),
                np.sin(np.pi * 2 * time_of_day / 24),
                year,
            ]
        )

    def __getitem__(self, index):

        # Load target data
        hadisd_slice = self.hadisd_data[index + self.offset_factor * self.lead_time]

        x_context = self.era5_x
        n_lon = x_context[0].shape[0]
        n_lat = x_context[1].shape[0]

        # Load auxiliary time
        aux_time = torch.reshape(self.to_tensor(self.get_time_aux(index)), (-1, 1, 1))

        # Load input
        y_context = self.norm_era5(self.Y_context[index, ..., self.lead_time])
        y_context = torch.cat(
            [
                self.to_tensor(y_context).permute(2, 1, 0),
                self.era5_elev.permute(0, 2, 1),
                aux_time.repeat(1, n_lon, n_lat),
            ]
        )

        assert y_context.shape[1] == n_lon
        assert y_context.shape[2] == n_lat

        # Handle region masking
        if self.region != "global":
            hadisd_slice["y"][self.mask] = np.nan

        return {
            "x_target": hadisd_slice["x"],
            "alt_target": hadisd_slice["altitude"],
            "y_target": hadisd_slice["y"],
            "y_context": y_context,
            "x_context": x_context,
            "aux_time": aux_time,
            "lt": torch.Tensor([0]),
        }


class ForecastLoader(Dataset):
    """
    Loader for finetuning the processor module
    """

    def __init__(
        self,
        device,
        mode,
        lead_time,
        era5_mode="sfc",
        frequency=24,
        norm=True,
        diff=False,
        rollout=False,
        random_lt=False,
        u_only=False,
        ic_path=None,
        finetune_step=None,
        finetune_eval_every=100,
        eval_steps=False,
        start_date=None,
        end_date=None,
        data_path=None,
        aux_data_path=None,
    ):

        super().__init__()

        # Setup
        self.device = device
        self.mode = mode
        self.data_path = data_path or "data_path/"
        self.aux_data_path = aux_data_path or "path_to_auxiliary_data/"

        self.lead_time = lead_time
        self.era5_mode = era5_mode
        self.frequency = frequency
        self.norm = norm
        self.diff = diff
        self.rollout = rollout
        self.random_lt = random_lt
        self.u_only = u_only
        self.ic_path = ic_path

        self.finetune_step = finetune_step
        self.finetune_eval_every = finetune_eval_every
        self.eval_steps = eval_steps
        channels = 30 if self.era5_mode == "4u_sfc" else 24

        # Data/target grid (Grid A) size from the canonical grid files; used for IC/ERA5
        # memmap shapes below and asserted against the orography array.
        self.nlon, self.nlat = grid_dims_from_files(self.data_path)

        if self.frequency == 6:
            self.lead_time = self.lead_time * 4
            freq = "6H"

        else:
            freq = "1D"

        if start_date is not None or end_date is not None:
            if not start_date or not end_date:
                raise ValueError("Both start_date and end_date are required for ForecastLoader.")
            self.dates = pd.date_range(start_date, end_date, freq=freq)
        else:
            if self.mode == "train":
                self.dates = pd.date_range("1979-01-01", "2017-12-31", freq=freq)
            elif self.mode == "tune":
                self.dates = pd.date_range("2018-01-01", "2018-12-31", freq=freq)
            elif self.mode == "test":
                self.dates = pd.date_range("2018-01-01", "2018-12-31", freq=freq)
            elif self.mode == "val":
                self.dates = pd.date_range("2019-01-01", "2019-12-31", freq=freq)

        # Load the predictions from the previous leadtime to be the new context set
        if self.finetune_step is not None:

            if self.mode == "train":
                if start_date is not None and end_date is not None:
                    self.dates = pd.date_range(start_date, end_date, freq=freq)
                else:
                    self.dates = pd.date_range("2007-01-02", "2017-12-31", freq=freq)
                ic_shape = (
                    len(self.dates) - max(0, (self.finetune_step - 1) * 4),
                    self.nlat,
                    self.nlon,
                    channels,
                )
            elif self.mode == "val":
                if start_date is not None and end_date is not None:
                    self.dates = pd.date_range(start_date, end_date, freq=freq)
                else:
                    self.dates = pd.date_range("2019-01-01", "2019-12-31", freq=freq)
                ic_shape = (
                    len(self.dates) - max(0, (self.finetune_step - 1) * 4),
                    self.nlat,
                    self.nlon,
                    channels,
                )
            elif self.mode == "test":
                if start_date is not None and end_date is not None:
                    self.dates = pd.date_range(start_date, end_date, freq=freq)
                else:
                    self.dates = pd.date_range("2018-01-01", "2018-12-31", freq=freq)
                ic_shape = (
                    len(self.dates) - max(0, (self.finetune_step - 1) * 4),
                    self.nlat,
                    self.nlon,
                    channels,
                )

            if self.finetune_step > 1:
                print(ic_shape)
                self.ic = np.memmap(
                    self.ic_path
                    + "ic_{}_{}.mmap".format(self.mode, self.finetune_step - 1),
                    dtype="float32",
                    mode="r",
                    shape=ic_shape,
                )
            elif self.ic_path is not None:

                self.ic = np.memmap(
                    self.ic_path + "ic_{}.mmap".format(self.mode),
                    dtype="float32",
                    mode="r",
                    shape=ic_shape,
                )

        elif self.ic_path is not None:
            if self.mode == "train":
                if start_date is not None and end_date is not None:
                    self.dates = pd.date_range(start_date, end_date, freq=freq)
                else:
                    self.dates = pd.date_range("2007-01-02", "2017-12-31", freq=freq)
            ic_shape = (len(self.dates), self.nlat, self.nlon, channels)

            self.ic = np.memmap(
                self.ic_path + "/ic_{}.mmap".format(self.mode),
                dtype="float32",
                mode="r",
                shape=ic_shape,
            )

        # Orography. Raw file is (channels, nlat, nlon).
        self.era5_elev = np.float32(
            np.load(elev_vars_path(self.data_path))
        )
        assert_grid_match(
            "elev_vars (raw)", self.era5_elev.shape[1:], (self.nlat, self.nlon)
        )
        elev_mean = self.era5_elev.mean(axis=(1, 2))[:, np.newaxis, np.newaxis]
        elev_std = self.era5_elev.std(axis=(1, 2))[:, np.newaxis, np.newaxis]
        self.era5_elev = (self.era5_elev - elev_mean) / elev_std
        # Align to (channels, lon, lat) like ERA5 fields used by ForecastLoader.
        self.era5_elev = np.transpose(self.era5_elev, (0, 2, 1))
        assert_grid_match(
            "era5_elev (loader)", self.era5_elev.shape[1:], (self.nlon, self.nlat)
        )

        # ERA5 ground truth data for training
        self.era5_sfc = [
            self.load_era5(year)
            for year in range(int(self.dates[0].year), int(self.dates[-1].year) + 1)
        ]

        # Noramalisation factors
        self.means = (
            self.to_tensor(
                np.load(norm_mean_path(self.data_path, self.era5_mode))
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )
        self.stds = (
            self.to_tensor(
                np.load(norm_std_path(self.data_path, self.era5_mode))
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )
        if self.diff:
            self.diff_means = (
                self.to_tensor(
                    np.load(
                        self.data_path
                        + "norm_factors/mean_diff_{}_1.npy".format(
                            self.era5_mode
                        )
                    )
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            self.diff_stds = (
                self.to_tensor(
                    np.load(
                        self.data_path
                        + "norm_factors/std_diff_{}_1.npy".format(
                            self.era5_mode
                        )
                    )
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            self.diff_means_1 = (
                self.to_tensor(
                    np.load(
                        self.data_path
                        + "norm_factors/mean_diff_{}_1_6h.npy".format(
                            self.era5_mode
                        )
                    )
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            self.diff_stds_1 = (
                self.to_tensor(
                    np.load(
                        self.data_path
                        + "norm_factors/std_diff_{}_1_6h.npy".format(
                            self.era5_mode
                        )
                    )
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            self.diff_means_2 = (
                self.to_tensor(
                    np.load(
                        self.data_path
                        + "norm_factors/mean_diff_{}_1_12h.npy".format(
                            self.era5_mode
                        )
                    )
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            self.diff_stds_2 = (
                self.to_tensor(
                    np.load(
                        self.data_path
                        + "norm_factors/std_diff_{}_1_12h.npy".format(
                            self.era5_mode
                        )
                    )
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            self.means_dict = {
                0: self.diff_means,
                2: self.diff_means_2,
                3: self.diff_means_1,
            }
            self.stds_dict = {
                0: self.diff_stds,
                2: self.diff_stds_2,
                3: self.diff_stds_1,
            }
        else:
            self.means_dict = {}
            self.stds_dict = {}

    def __len__(self):
        if np.logical_and(self.eval_steps, self.mode == "train"):
            return self.finetune_eval_every * 12 * 4

        return self.dates.shape[0] - self.lead_time

    def to_tensor(self, arr):

        return torch.from_numpy(arr).float().to(self.device)

    def norm_era5(self, x):
        x = (x - self.means) / self.stds
        return x

    def norm_era5_tendency(self, x, lt_offset):

        x = (x - self.means_dict[lt_offset]) / self.stds_dict[lt_offset]
        return x

    def unnorm_pred(self, x):
        x = x * self.diff_stds.unsqueeze(0) + self.diff_means.unsqueeze(0)
        return x

    def unnorm_base_context(self, x):
        x = x * self.stds.unsqueeze(0) + self.means.unsqueeze(0)
        return x

    def load_era5(self, year):
        """
        Load ERA5 data for training
        """

        if year % 4 == 0:
            d = 366
        else:
            d = 365

        if self.frequency == 6:
            d = d * 4

        if self.era5_mode == "sfc":
            levels = 4
        elif self.era5_mode == "13u":
            levels = 69
        elif self.era5_mode in ("4u_sfc", "rtma_ok_sfc"):
            # rtma_ok_sfc = RTMA OK Phase 1 surface target (5 channels); channel count is
            # inferred from the memmap file size rather than hard-coded.
            levels = None
        else:
            levels = 24

        x = self.nlon
        y = self.nlat

        freq_tag = "6" if self.frequency == 6 else "1d"
        memmap_path = era5_memmap_path(
            self.data_path, self.era5_mode, freq_tag, year
        )
        if levels is None:
            nbytes = os.path.getsize(memmap_path)
            denom = d * x * y * 4
            if nbytes % denom != 0:
                raise ValueError(
                    f"File size not divisible by expected frame size: {memmap_path}"
                )
            levels = nbytes // denom
        mmap = np.memmap(
            memmap_path,
            dtype="float32",
            mode="r",
            shape=(d, levels, x, y),
        )
        return mmap

    def load_era5_time(self, index):
        """
        Load ERA5 data for training
        """
        date = self.dates[index]
        year = date.year
        doy = date.dayofyear - 1
        hour = date.hour
        if self.frequency == 6:
            era5 = self.era5_sfc[year - int(self.dates[0].year)][
                doy * 4 + hour // 6, ...
            ]
        else:
            era5 = self.era5_sfc[year - int(self.dates[0].year)][doy, ...]

        return np.copy(era5)

    def make_time_channels(self, index, x, y):
        """
        Make auxiliary time channels
        """

        date = self.dates[index]
        hour = date.hour
        doy = date.dayofyear - 1
        if date.year % 4 == 0:
            n_days = 366
        else:
            n_days = 365
        hour_sin = np.sin(hour * np.pi / 12) * np.float32(np.ones((1, x, y)))
        hour_cos = np.cos(hour * np.pi / 12) * np.float32(np.ones((1, x, y)))
        doy_sin = np.sin(doy * 2 * np.pi / n_days) * np.float32(np.ones((1, x, y)))
        doy_cos = np.cos(doy * 2 * np.pi / n_days) * np.float32(np.ones((1, x, y)))
        return np.concatenate([hour_sin, hour_cos, doy_sin, doy_cos])

    def __getitem__(self, index):

        # Option to offset to random leadtime
        lt_offset = 0
        if self.random_lt:
            lt_offset = np.random.choice([0, 2, 3])

        # Load ground truth data
        y_target = self.to_tensor(
            self.load_era5_time(index + self.lead_time - lt_offset)
        )

        # Load either initial condition or ERA5 depending on task
        if self.ic_path is not None:
            era5_ts0 = self.ic[index].copy().transpose(2, 1, 0)

        else:
            era5_ts0 = self.load_era5_time(index)

        # Auxiliary time
        time = self.make_time_channels(index, era5_ts0.shape[1], era5_ts0.shape[2])
        era5_ts0 = self.to_tensor(
            np.concatenate([era5_ts0, self.era5_elev, time], axis=0)
        )
        y_context = era5_ts0.permute(0, 2, 1)[:, ...]

        # Normalisation
        if self.diff:
            channels = 30 if self.era5_mode == "4u_sfc" else 24
            y_target = (y_target - era5_ts0[:channels, ...]).permute(2, 1, 0)
            y_target = self.norm_era5_tendency(y_target, lt_offset)
            y_context[:channels, ...] = self.norm_era5(y_context[:channels, ...])

        else:
            if self.norm:
                channels = 30 if self.era5_mode == "4u_sfc" else 24
                y_context[:channels, ...] = self.norm_era5(y_context[:channels, ...])
                y_target = self.norm_era5(y_target)
            y_target = y_target.permute(2, 1, 0)

        if self.rollout:
            # Option to return entire timeseries of target data
            targets = []
            for t in range(self.lead_time + 1):
                t = self.to_tensor(self.load_era5_time(index + t))
                targets.append(t.permute(2, 1, 0))
            targets = torch.stack(targets, dim=-1)[..., ::4]

            return {
                "y_context": y_context.permute(0, 2, 1),
                "y_target": y_target,
                "targets": targets,
                "lt": self.to_tensor(np.array([lt_offset])),
            }

        else:
            return {
                "y_context": y_context.permute(0, 2, 1),
                "y_target": y_target[..., :],
                "lt": self.to_tensor(np.array([lt_offset])),
                "target_index": self.to_tensor(np.array([index])),
            }


class WeatherDatasetE2E(WeatherDataset):
    """
    Dataset for running Aardvark end-to-end
    """

    def __init__(
        self,
        device,
        hadisd_mode,
        start_date,
        end_date,
        lead_time,
        mode,
        hadisd_var,
        max_steps_per_epoch=None,
        era5_mode="sfc",
        filter_dates=None,
        var_start=0,
        var_end=24,
        diff=False,
        two_frames=False,
        region="global",
    ):

        super().__init__(
            device,
            hadisd_mode,
            start_date,
            end_date,
            lead_time,
            era5_mode,
            filter_dates=filter_dates,
            diff=diff,
        )

        # Setup
        self.var_start = var_start
        self.var_end = var_end
        self.diff = diff
        self.two_frames = two_frames
        self.region = region
        self.lead_time = lead_time
        self.mode = mode
        self.max_steps_per_epoch = max_steps_per_epoch

        # Initialise encoder dataset
        self.assimilation_dataset = WeatherDatasetAssimilation(
            device="cuda",
            hadisd_mode="train",
            start_date=start_date,
            end_date=end_date,
            lead_time=0,
            era5_mode="4u",
            var_start=0,
            var_end=24,
            diff=False,
            two_frames=False,
        )

        # Initialise forecast dataset
        self.forecast_dataset = ForecastLoader(
            device="cuda",
            mode=mode,
            lead_time=lead_time,
            era5_mode=era5_mode,
            frequency=6,
            diff=True,
            u_only=False,
            random_lt=False,
        )

        # Initialise downscaling dataset
        self.downscaling_dataset = ForecasterDatasetDownscaling(
            start_date=start_date,
            end_date=end_date,
            lead_time=lead_time,
            hadisd_var=hadisd_var,
            mode=mode,
            device=device,
            forecast_path=None,
            era5_mode=era5_mode,
            region=region,
        )

    def __len__(self):
        if self.max_steps_per_epoch:
            return self.max_steps_per_epoch
        return len(self.downscaling_dataset) - 40  # Need 10 day offset at end of year

    def __getitem__(self, index):

        if self.max_steps_per_epoch:
            index = np.random.choice(
                np.arange(len(self.downscaling_dataset) - 40)
            )  # Need 10 day offset at end of year

        # Get data for each of the three datasets
        assimilation = self.assimilation_dataset.__getitem__(index)
        forecast = self.forecast_dataset.__getitem__(index)
        downscaling = self.downscaling_dataset.__getitem__(index)

        # Create task
        task = {
            "assimilation": assimilation,
            "forecast": forecast,
            "downscaling": downscaling,
            "index": torch.tensor(index),
        }

        # Add y target to allow for end to end finetuning if needed
        task["y_target"] = task["downscaling"]["y_target"]

        return task

    def unnorm_pred(self, x):

        dev = x.device
        x = x.detach().cpu().numpy()

        x = (
            x
            * self.stds[np.newaxis, ...].transpose(0, 2, 3, 1)[
                ..., self.var_start : self.var_end
            ]
            + self.means[np.newaxis, ...].transpose(0, 2, 3, 1)[
                ..., self.var_start : self.var_end
            ]
        )
        if bool(self.diff):
            x = (
                x
                + self.era5_mean_spatial[np.newaxis, ...].transpose(0, 3, 2, 1)[
                    ..., self.var_start : self.var_end
                ]
            )
        return torch.from_numpy(x).float().to(dev)
