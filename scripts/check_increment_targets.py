"""Synthetic regression checks for hourly increment preprocessing and assimilation.

Run: python scripts/check_increment_targets.py
Requires the repository's PyTorch environment. No external weather data or GPU needed.
"""

import calendar
import contextlib
import copy
from datetime import timedelta
import gc
import io
import json
import socket
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "aardvark"))
sys.path.insert(0, str(REPO / "scripts"))
from background_normalization import file_sha256, load_background_norms
from grid_config import (
    load_grid_config,
    set_active_config,
    era5_month_path,
    background_input_path,
    increment_month_path,
    increment_norm_path,
)
from increment_data import (
    CHANNEL_ORDER,
    atomic_json,
    build_increment,
    check_target_checkpoint,
    check_training_months,
    frame_layout,
    load_increment_norms,
    read_json,
    sidecar_path,
    validate_product,
    verify_increment_input,
    verify_increment_sources,
    verify_subtraction,
    write_source_sidecar,
)
from compute_increment_norms import compute_statistics, write_factors
from loader import (
    WeatherDatasetAssimilation,
    ForecastLoader,
    WeatherDatasetDownscaling,
    ForecasterDatasetDownscaling,
    WeatherDatasetE2E,
)
from loader_utils_new import month_frame
from loss_functions import RmseLoss
from physical_metrics import (
    require_even_validation,
    physical_metric_sums,
    finalize_physical_metrics,
)


class OffsetPrediction(torch.nn.Module):
    def forward(self, task, film_index=0):
        return task["y_target"] + 0.5


def distributed_eval(
    rank, world_size, root, config_path, init_uri, output_dir, target_mode
):
    from trainer import DDPTrainer

    set_active_config(load_grid_config(config_path))
    backend = "gloo"
    if sys.platform == "win32":
        # The workstation hostname can resolve to a disconnected corporate interface.
        # Bind Gloo itself to loopback, not just the rendezvous store.
        def loopback_gloo(store, rank, size, timeout):
            options = dist.ProcessGroupGloo._Options()
            options._devices = [
                dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")
            ]
            options._timeout = timeout
            return dist.ProcessGroupGloo(store, rank, size, options)

        dist.Backend.register_backend("gloo_loopback", loopback_gloo, devices=["cpu"])
        backend = "gloo_loopback"
    dist.init_process_group(
        backend,
        init_method=init_uri,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=45),
    )
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            dataset = make_dataset(
                root,
                "2022-01-01 00:00",
                end="2022-01-01 03:00",
                target_mode=target_mode,
            )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)
        trainer = DDPTrainer.__new__(DDPTrainer)
        trainer.model = OffsetPrediction()
        trainer.val_loader = trainer.train_loader = loader
        trainer.loss_function = RmseLoss(start_ind=0, end_ind=5)
        trainer.rank, trainer.epoch, trainer.best_loss = rank, 0, 1000
        trainer.save_path = str(output_dir) + "/"
        trainer.test_loader = None
        trainer.background_normalization = dataset.background_normalization
        trainer.target_normalization = dataset.target_normalization
        trainer.eval_epoch(False, 0)
    finally:
        dist.destroy_process_group()


def make_dataset(root, start="2022-01-01 00:00", end=None, **options):
    kwargs = dict(
        device="cpu",
        hadisd_mode="train",
        start_date=start,
        end_date=end or start,
        lead_time=0,
        era5_mode="rtma_ok_sfc",
        var_start=0,
        var_end=5,
        diff=False,
        time_freq="1H",
        obs_set="rtma_surface",
        background_mode="hourly",
        target_mode="increment",
        background_input="raw",
        include_physical_targets=True,
        data_path=str(root) + "/",
        aux_data_path=str(root) + "/",
    )
    kwargs.update(options)
    return WeatherDatasetAssimilation(**kwargs)


class IncrementChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Repository-local temp root also works in restricted developer sandboxes.
        (REPO / "work").mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(
            dir=REPO / "work", prefix="increment-check-"
        )
        cls.root = Path(cls.temp.name)
        cls.config = load_grid_config()
        cls.config["grid_files"].update(
            loader_x="lon.npy", loader_y="lat.npy", model_x="lon.npy", model_y="lat.npy"
        )
        cls.config["data_files"]["elev_vars"] = "elev.npy"
        cls.config_path = cls.root / "grid.yaml"
        cls.config_path.write_text(yaml.safe_dump(cls.config))
        set_active_config(cls.config)
        np.save(cls.root / "lon.npy", np.array([260, 261, 262], dtype="float32"))
        np.save(cls.root / "lat.npy", np.array([34, 35], dtype="float32"))
        elevation = np.zeros((4, 2, 3), dtype="float32")
        elevation[2] = np.sin(np.deg2rad(np.array([34, 35], dtype="float32")))[:, None]
        np.save(cls.root / "elev.npy", elevation)
        norms = cls.root / "norm_factors"
        norms.mkdir()
        cls.analysis_mean = np.array([280, 0.01, 100000, 2, -2], dtype="float32")
        cls.analysis_std = np.array([10, 0.005, 1000, 5, 5], dtype="float32")
        np.save(norms / "mean_rtma_ok_sfc_1.npy", cls.analysis_mean)
        np.save(norms / "std_rtma_ok_sfc_1.npy", cls.analysis_std)
        obs = cls.root / "hadisd_processed"
        obs.mkdir()
        for var in CHANNEL_ORDER:
            np.save(norms / f"mean_hadisd_{var}.npy", np.float32(0))
            np.save(norms / f"std_hadisd_{var}.npy", np.float32(1))
        cls.months = [(2022, 1), (2022, 2), (2023, 1), (2024, 2)]
        rng = np.random.default_rng(13)
        cls.shapes = {}
        for year, month in cls.months:
            frames = calendar.monthrange(year, month)[1] * 24
            shape = (frames, 5, 3, 2)
            cls.shapes[year, month] = shape
            tag = f"{year}-{month:02d}"
            for var in CHANNEL_ORDER:
                for axis, values in (("lon", [260]), ("lat", [34]), ("alt", [1])):
                    np.save(
                        obs / f"{var}_{axis}_train-{tag}.npy",
                        np.array(values, dtype="float32"),
                    )
                np.zeros((frames, 1), dtype="float32").tofile(
                    obs / f"{var}_vals_1h_{tag}.memmap"
                )
            analysis = (
                cls.analysis_mean[None, :, None, None]
                + rng.normal(size=shape).astype("float32")
                * cls.analysis_std[None, :, None, None]
            )
            delta = (
                rng.normal(size=shape).astype("float32")
                * cls.analysis_std[None, :, None, None]
                * 0.2
            )
            background = analysis - delta
            # Permanent analysis masks do not count against the 99% threshold.
            analysis[:, :, 0, 0] = np.nan
            background[:, :, 0, 0] = np.nan
            # One additional missing background hour must affect metrics, not analysis loss.
            background[0, :, 1, 0] = np.nan
            a = Path(era5_month_path(str(cls.root), "rtma_ok_sfc", "1h", year, month))
            b = Path(
                background_input_path(
                    str(cls.root), "rtma_ok_sfc", year, month, "hourly", "raw"
                )
            )
            a.parent.mkdir(exist_ok=True)
            analysis.tofile(a)
            background.tofile(b)
            write_source_sidecar(a, shape, year, month)
            write_source_sidecar(b, shape, year, month)
            normalized = (
                background - cls.analysis_mean[None, :, None, None]
            ) / cls.analysis_std[None, :, None, None]
            normalized.tofile(
                background_input_path(
                    str(cls.root), "rtma_ok_sfc", year, month, "hourly", "normalized"
                )
            )
        cls.train_months = cls.root / "train_months.txt"
        cls.train_months.write_text("2022-01\n2022-02\n")
        cls.run_cli(
            "build_hourly_increments.py",
            "--months",
            *[f"{y}-{m:02d}" for y, m in cls.months],
        )
        cls.run_cli(
            "compute_increment_norms.py", "--months_file", str(cls.train_months)
        )
        _, _, cls.metadata = load_increment_norms(str(cls.root), "rtma_ok_sfc")

    @classmethod
    def run_cli(cls, script, *args):
        command = [
            sys.executable,
            str(REPO / "scripts" / script),
            "--data_root",
            str(cls.root),
            "--grid_config",
            str(cls.config_path),
            *args,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        cls.temp.cleanup()

    def setUp(self):
        set_active_config(self.config)

    def test_sign_and_written_integrity(self):
        shape = self.shapes[2022, 1]
        a, b, out = [self.root / name for name in ("sign-a", "sign-b", "sign-inc")]
        analysis = np.arange(np.prod(shape), dtype="float32").reshape(shape)
        analysis.tofile(a)
        (analysis - 1).tofile(b)
        with self.assertRaisesRegex(ValueError, "missing sidecar"):
            build_increment(a, b, out, shape, 2022, 1)
        metadata = build_increment(a, b, out, shape, 2022, 1, allow_legacy=True)
        self.assertEqual(metadata["channel_order_source"], "override")
        written = np.fromfile(out, dtype="float32").reshape(shape)
        np.testing.assert_array_equal(written, 1)
        np.testing.assert_array_equal(analysis - 1 + written, analysis)
        written[-1] = -1
        with self.assertRaisesRegex(ValueError, "bit-exact"):
            verify_subtraction(analysis, analysis - 1, written)

    def test_channel_orientation_and_atomic_failure(self):
        shape = self.shapes[2022, 1]
        a, b, out = [self.root / name for name in ("bad-a", "bad-b", "bad-inc")]
        np.ones(shape, dtype="float32").tofile(a)
        np.zeros(shape, dtype="float32").tofile(b)
        write_source_sidecar(a, shape, 2022, 1)
        write_source_sidecar(b, shape, 2022, 1)
        meta = read_json(sidecar_path(a))
        meta["channel_order"] = CHANNEL_ORDER[::-1]
        atomic_json(sidecar_path(a), meta)
        with self.assertRaisesRegex(ValueError, "channel_order"):
            build_increment(a, b, out, shape, 2022, 1)
        meta["channel_order"] = CHANNEL_ORDER
        meta["grid_orientation"]["origin"] = "NW"
        atomic_json(sidecar_path(a), meta)
        with self.assertRaisesRegex(ValueError, "grid_orientation"):
            build_increment(a, b, out, shape, 2022, 1)
        invalid = np.zeros(shape, dtype="float32")
        invalid[-1] = np.inf
        invalid.tofile(b)
        write_source_sidecar(a, shape, 2022, 1)
        write_source_sidecar(b, shape, 2022, 1)
        with self.assertRaisesRegex(ValueError, "infinities"):
            build_increment(a, b, out, shape, 2022, 1)
        self.assertFalse(out.exists())
        self.assertEqual(list(self.root.glob("bad-inc.*.tmp")), [])

    def test_statistics_and_nan_masks(self):
        self.assertTrue(all(c >= 0.99 for c in self.metadata["coverage"]))
        self.assertTrue(all(f > 0.16 for f in self.metadata["analysis_nan_fraction"]))
        inputs, arrays = [], []
        for y, m in [(2022, 1), (2022, 2)]:
            shape = self.shapes[y, m]
            path = increment_month_path(str(self.root), "rtma_ok_sfc", "1h", y, m)
            a = era5_month_path(str(self.root), "rtma_ok_sfc", "1h", y, m)
            b = background_input_path(
                str(self.root), "rtma_ok_sfc", y, m, "hourly", "raw"
            )
            values = np.fromfile(path, dtype="float32").reshape(shape)
            analysis, bg = [
                np.fromfile(p, dtype="float32").reshape(shape) for p in (a, b)
            ]
            np.testing.assert_array_equal(
                np.isnan(values), np.isnan(analysis) | np.isnan(bg)
            )
            arrays.append(values)
            inputs.append((path, a, shape))
        expected = np.concatenate(arrays).astype("float64")
        np.testing.assert_allclose(
            self.metadata["mean"], np.nanmean(expected, axis=(0, 2, 3)), rtol=1e-6
        )
        np.testing.assert_allclose(
            self.metadata["std"], np.nanstd(expected, axis=(0, 2, 3)), rtol=1e-6
        )
        with self.assertRaisesRegex(ValueError, "coverage"):
            compute_statistics(inputs, self.analysis_std, minimum_coverage=1)
        with self.assertRaisesRegex(ValueError, "floor"):
            compute_statistics(inputs, self.analysis_std, std_floor_fraction=1)

    def test_loader_boundaries_roundtrip_and_loss(self):
        for y, m in [(2022, 1), (2022, 2), (2024, 2)]:
            last = calendar.monthrange(y, m)[1]
            time = f"{y}-{m:02d}-{last:02d} 23:00"
            import pandas as pd

            self.assertEqual(month_frame(pd.Timestamp(time), 60)[1], last * 24 - 1)
            with contextlib.redirect_stdout(io.StringIO()):
                raw = make_dataset(self.root, time)
                normalized = make_dataset(
                    self.root, time, background_input="normalized"
                )
            task = next(iter(torch.utils.data.DataLoader(raw, batch_size=1)))
            other = next(iter(torch.utils.data.DataLoader(normalized, batch_size=1)))
            prediction = raw.physical_analysis(task["y_target"], task)
            np.testing.assert_allclose(
                prediction.numpy(),
                task["physical_analysis_current"].numpy(),
                rtol=1e-6,
                atol=0.02,
            )
            np.testing.assert_allclose(
                prediction.numpy(),
                normalized.physical_analysis(other["y_target"], other).numpy(),
                rtol=1e-6,
                atol=0.02,
            )
            zero = torch.zeros_like(task["y_target"])
            expected = raw.physical_background(task) + torch.as_tensor(
                raw.means.transpose(2, 1, 0)
            )
            torch.testing.assert_close(
                raw.physical_analysis(zero, task), expected, equal_nan=True
            )
            with self.assertRaisesRegex(ValueError, "increment factors"):
                raw.unnorm_base_context(zero)
            loss = RmseLoss(start_ind=0, end_ind=5)(
                task["y_target"], task["y_target"] + 1, None
            )
            self.assertAlmostEqual(loss.item(), 1.0)
            self.assertNotEqual(raw.means[0, 0, 0], raw.background_means[0, 0, 0])

    def test_analysis_loss_mask_unchanged(self):
        with contextlib.redirect_stdout(io.StringIO()):
            dataset = make_dataset(self.root, target_mode="analysis")
        task = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
        truth = task["physical_analysis_current"].numpy()
        background = dataset.physical_background(task).numpy()
        self.assertTrue(np.isfinite(task["y_target"][0, 0, 1]).all())
        self.assertTrue(np.isnan(background[0, 0, 1]).all())
        sums = physical_metric_sums(truth + 1, truth, background)
        self.assertTrue((sums[1] < sums[2]).all())
        np.testing.assert_allclose(finalize_physical_metrics(sums)["rmse"], 1)

    def test_training_does_not_read_metric_truth(self):
        for mode in ("analysis", "increment"):
            with contextlib.redirect_stdout(io.StringIO()):
                dataset = make_dataset(
                    self.root, target_mode=mode, include_physical_targets=False
                )
            self.assertEqual(dataset.analysis_truth, {})
            with mock.patch(
                "loader.np.memmap", side_effect=AssertionError("extra training read")
            ):
                task = dataset[0]
            self.assertFalse(any(key.startswith("physical_") for key in task))

    def test_source_sidecars_reject_stale_increments(self):
        for source in ("analysis", "background"):
            path = (
                era5_month_path(str(self.root), "rtma_ok_sfc", "1h", 2022, 1)
                if source == "analysis"
                else background_input_path(
                    str(self.root), "rtma_ok_sfc", 2022, 1, "hourly", "raw"
                )
            )
            sidecar = sidecar_path(path)
            original = read_json(sidecar)
            changed = copy.deepcopy(original)
            changed["sha256"] = "0" * 64
            try:
                atomic_json(sidecar, changed)
                with self.assertRaisesRegex(
                    ValueError, source + " source hash mismatch"
                ):
                    make_dataset(self.root, include_physical_targets=False)
            finally:
                atomic_json(sidecar, original)
        inc_path = increment_month_path(str(self.root), "rtma_ok_sfc", "1h", 2022, 1)
        product = read_json(sidecar_path(inc_path))
        analysis = era5_month_path(str(self.root), "rtma_ok_sfc", "1h", 2022, 1)
        source_meta = read_json(sidecar_path(analysis))
        try:
            sidecar_path(analysis).unlink()
            with self.assertRaisesRegex(ValueError, "missing sidecar"):
                verify_increment_sources(
                    product, analysis, None, self.shapes[2022, 1], 2022, 1
                )
            product["sources"]["analysis"]["channel_order_source"] = "override"
            with self.assertWarnsRegex(UserWarning, "UNVERIFIED legacy analysis"):
                verify_increment_sources(
                    product, analysis, None, self.shapes[2022, 1], 2022, 1
                )
        finally:
            atomic_json(sidecar_path(analysis), source_meta)

    def test_metric_rejects_bad_predictions_after_reduction(self):
        truth = np.ones((2, 2, 3, 5), dtype="float32")
        background = truth.copy()
        background[0, 0, 0] = np.nan
        prediction = truth.copy()
        prediction[0, 0, 0] = np.nan
        finalize_physical_metrics(physical_metric_sums(prediction, truth, background))
        good_rank = physical_metric_sums(truth, truth, background)
        for value in (np.nan, np.inf, -np.inf):
            prediction[1, 0, 0] = value
            bad_rank = physical_metric_sums(prediction, truth, background)
            np.testing.assert_array_equal(bad_rank[1], good_rank[1])
            with self.assertRaisesRegex(ValueError, "non-finite predictions"):
                finalize_physical_metrics(good_rank + bad_rank)

    def test_load_enforces_hard_statistics_limits(self):
        aux = self.root / "separate-aux"
        (aux / "norm_factors").mkdir(parents=True)
        np.save(aux / "norm_factors/std_rtma_ok_sfc_1.npy", self.analysis_std)
        mean_path, std_path = [
            Path(increment_norm_path(str(aux), "rtma_ok_sfc", s))
            for s in ("mean", "std")
        ]
        mean, std = [
            np.asarray(self.metadata[key], dtype="float32") for key in ("mean", "std")
        ]
        metadata = copy.deepcopy(self.metadata)
        metadata["minimum_coverage"] = 0.5
        metadata["coverage"] = [0.5] * 5
        write_factors(mean_path, std_path, mean, std, metadata)
        with self.assertRaisesRegex(ValueError, "coverage"):
            load_increment_norms(str(aux), "rtma_ok_sfc")
        metadata = copy.deepcopy(self.metadata)
        metadata["std_floor_fraction"] = 1e-6
        write_factors(mean_path, std_path, mean, self.analysis_std * 1e-4, metadata)
        with self.assertRaisesRegex(ValueError, "floor"):
            load_increment_norms(str(aux), "rtma_ok_sfc")
        with self.assertRaisesRegex(ValueError, "invalid chunk size"):
            compute_statistics([], self.analysis_std, minimum_coverage=0.5)
        self.run_cli(
            "compute_increment_norms.py",
            "--months_file",
            str(self.train_months),
            "--aux_data_path",
            str(aux),
            "--overwrite",
        )
        actual_mean, actual_std, _ = load_increment_norms(str(aux), "rtma_ok_sfc")
        np.testing.assert_array_equal(actual_mean[:, 0, 0], mean)
        np.testing.assert_array_equal(actual_std[:, 0, 0], std)

    def test_preflight_bad_date_fails_without_traceback(self):
        script = self.root / "bad-date.sh"
        script.write_text(
            f'python train_module.py --data_path "{self.root}" --aux_data_path "{self.root}" '
            f'--model_data_path "{self.root}" --grid_config "{self.config_path}" '
            "--obs_set rtma_surface --time_freq 1H --target_mode increment --diff 0 "
            "--background_mode hourly --assim_train_start_date 2022-02-31 "
            "--assim_train_end_date 2022-02-28 --assim_val_start_date 2023-01-01 "
            "--assim_val_end_date 2023-01-01\n"
        )
        result = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/check_ok_run_files.py"),
                "--train_script",
                str(script),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("date error", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_preflight_valid_analysis_and_increment(self):
        for mode in ("analysis", "increment"):
            script = self.root / f"preflight-{mode}.sh"
            script.write_text(
                f'python train_module.py --data_path "{self.root}" --aux_data_path "{self.root}" '
                f'--model_data_path "{self.root}" --grid_config "{self.config_path}" '
                f"--target_mode {mode} --era5_mode rtma_ok_sfc --loss rmse --epoch 1 "
                "--obs_set rtma_surface --time_freq 1H --diff 0 --background_input raw "
                "--background_mode hourly --assim_train_start_date 2022-01-01 "
                "--assim_train_end_date 2022-01-01 --assim_val_start_date 2023-01-01 "
                "--assim_val_end_date 2023-01-01\n"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts/check_ok_run_files.py"),
                    "--train_script",
                    str(script),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_validation_requires_physical_fields(self):
        from trainer import DDPTrainer

        trainer = DDPTrainer.__new__(DDPTrainer)
        trainer.model = OffsetPrediction()
        trainer.val_loader = SimpleNamespace(
            dataset=SimpleNamespace(
                monthly=True, diff=False, include_physical_targets=False
            )
        )
        with self.assertRaisesRegex(ValueError, "include_physical_targets=True"):
            trainer.eval_epoch(False, 0)

    def test_trainer_constructor_target_checks(self):
        from trainer import DDPTrainer

        class CpuModel(torch.nn.Linear):
            decoder = "linear"

            def to(self, *args, **kwargs):
                if args and isinstance(args[0], int):
                    return self
                return super().to(*args, **kwargs)

        with contextlib.redirect_stdout(io.StringIO()):
            train = make_dataset(self.root, include_physical_targets=False)
            validation = make_dataset(self.root)
        train_loader = torch.utils.data.DataLoader(train)
        val_loader = torch.utils.data.DataLoader(validation)
        destination = self.root / "constructor-run"
        destination.mkdir()

        def construct(val=val_loader):
            return DDPTrainer(
                CpuModel(1, 1),
                0,
                train_loader,
                val,
                RmseLoss(0, 5),
                str(destination) + "/",
                1e-3,
                None,
                weight_decay=0,
            )

        # Only CUDA placement/wrapping is replaced; provenance/persistence run unchanged.
        with mock.patch(
            "trainer.DDP",
            side_effect=lambda model, **kwargs: SimpleNamespace(module=model),
        ):
            construct()
            target_path = destination / "target_normalization.json"
            saved = read_json(target_path)
            self.assertEqual(
                saved["effective_sha256"], self.metadata["effective_sha256"]
            )
            construct()
            wrong = copy.deepcopy(saved)
            wrong["mean_file_sha256"] = "0" * 64
            atomic_json(target_path, wrong)
            with self.assertRaisesRegex(ValueError, "normalization mismatch"):
                construct()
            self.assertEqual(read_json(target_path), wrong)
            atomic_json(target_path, saved)
            bad_validation = SimpleNamespace(
                dataset=SimpleNamespace(target_normalization=wrong)
            )
            with self.assertRaisesRegex(ValueError, "normalization mismatch"):
                construct(bad_validation)

    def test_invalid_configurations_and_month_coverage(self):
        for options in [
            dict(diff=True),
            dict(background_mode="daily_00z"),
            dict(time_freq="15min"),
            dict(obs_set="all"),
        ]:
            with self.assertRaisesRegex(ValueError, "increment target requires"):
                make_dataset(self.root, **options)
        for loader in (
            ForecastLoader,
            WeatherDatasetDownscaling,
            ForecasterDatasetDownscaling,
            WeatherDatasetE2E,
        ):
            import inspect

            required = {
                name: None
                for name, p in inspect.signature(loader).parameters.items()
                if p.default is inspect.Parameter.empty
            }
            with self.assertRaisesRegex(ValueError, "unsupported"):
                loader(**required, target_mode="increment")
        with contextlib.redirect_stdout(io.StringIO()):
            covered = make_dataset(self.root)
            uncovered = make_dataset(self.root, "2023-01-01 00:00")
        covered.check_statistics_training_months()
        with self.assertRaisesRegex(ValueError, "not covered"):
            uncovered.check_statistics_training_months()
        with self.assertRaisesRegex(ValueError, "not covered"):
            check_training_months(["2022-03"], self.metadata)
        require_even_validation(365, 4, monthly_rtma=False)
        for ranks in (2, 4):
            with self.assertRaisesRegex(ValueError, "divisible"):
                require_even_validation(365, ranks, monthly_rtma=True)

    def test_checkpoint_rejection_for_every_purpose(self):
        from trainer import DDPTrainer

        # All load purposes share this actual trainer gate, before loading any weights.
        trainer = DDPTrainer.__new__(DDPTrainer)
        trainer.model = torch.nn.Linear(1, 1)
        trainer.background_normalization = None
        trainer.background_norm_manifest = None
        trainer.target_normalization = self.metadata
        path = self.root / "checkpoint"
        for checkpoint in (
            {"model_state_dict": trainer.model.state_dict()},
            {
                "model_state_dict": trainer.model.state_dict(),
                "target_normalization": {"target_mode": "analysis"},
            },
        ):
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                trainer._load_weights_if_provided(path)
        torch.save(
            {
                "model_state_dict": trainer.model.state_dict(),
                "target_normalization": self.metadata,
            },
            path,
        )
        trainer._load_weights_if_provided(path)
        check_target_checkpoint({"target_normalization": self.metadata}, self.metadata)
        for key in (
            "mean_file_sha256",
            "std_file_sha256",
            "effective_sha256",
            "training_months",
            "channel_order_source",
        ):
            changed = copy.deepcopy(self.metadata)
            changed[key] = "wrong"
            with self.assertRaisesRegex(ValueError, "mismatch"):
                check_target_checkpoint(
                    {"target_normalization": changed}, self.metadata
                )
        with self.assertRaisesRegex(ValueError, "target_mode"):
            check_target_checkpoint({"target_normalization": self.metadata}, None)
        changed = copy.deepcopy(self.metadata)
        changed["increment_sources"]["2022-01"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source metadata"):
            check_target_checkpoint({"target_normalization": changed}, self.metadata)

    def test_startup_recorded_hash_and_optional_rehash(self):
        path = self.root / "tampered-inc"
        original = Path(
            increment_month_path(str(self.root), "rtma_ok_sfc", "1h", 2022, 1)
        )
        path.write_bytes(original.read_bytes())
        meta = read_json(sidecar_path(original))
        atomic_json(sidecar_path(path), meta)
        shape = self.shapes[2022, 1]
        with open(path, "r+b") as stream:
            stream.seek(40)
            stream.write(np.float32(17).tobytes())
        verify_increment_input(path, shape, 2022, 1, self.metadata)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            verify_increment_input(path, shape, 2022, 1, self.metadata, rehash=True)
        meta["sha256"] = "0" * 64
        atomic_json(sidecar_path(path), meta)
        with self.assertRaisesRegex(ValueError, "statistics source"):
            verify_increment_input(path, shape, 2022, 1, self.metadata)

    def test_distributed_trainer_metrics_and_saved_outputs(self):
        for target_mode in ("analysis", "increment"):
            results = []
            for world_size in (1, 2):
                output = self.root / f"eval-{target_mode}-{world_size}"
                output.mkdir()
                # Loopback TCP avoids FileStore locking differences on Windows.
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", 0))
                    port = probe.getsockname()[1]
                rendezvous = f"tcp://127.0.0.1:{port}?use_libuv=0"
                mp.spawn(
                    distributed_eval,
                    args=(
                        world_size,
                        str(self.root),
                        str(self.config_path),
                        rendezvous,
                        str(output),
                        target_mode,
                    ),
                    nprocs=world_size,
                    join=True,
                )
                results.append(read_json(output / "physical_analysis_metrics.json"))
                name = (
                    "unnorm_analysis_preds_0.npy"
                    if target_mode == "increment"
                    else "unnorm_preds_0.npy"
                )
                self.assertTrue((output / name).is_file())
                if target_mode == "increment":
                    self.assertTrue((output / "unnorm_increment_preds_0.npy").is_file())
            np.testing.assert_allclose(
                results[0]["rmse"], results[1]["rmse"], rtol=1e-12
            )
            self.assertEqual(results[0]["valid_count"], results[1]["valid_count"])
            expected_scale = (
                np.asarray(self.metadata["std"])
                if target_mode == "increment"
                else self.analysis_std
            )
            np.testing.assert_allclose(
                results[0]["rmse"], expected_scale * 0.5, rtol=0.01
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
