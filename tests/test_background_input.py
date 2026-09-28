"""CPU regression checks: python -m unittest discover -s tests -v."""

import calendar
import contextlib
import copy
import gc
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "aardvark"))

from background_normalization import (
    check_checkpoint_normalization,
    compare_manifests,
    file_sha256,
    load_background_norms,
    normalize_background_frame,
    vector_sha256,
)
from grid_config import background_input_path, load_grid_config, set_active_config
from loader import WeatherDatasetAssimilation


class BackgroundInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.aux = self.root / "aux"
        for folder in (
            self.data / "era5",
            self.data / "hadisd_processed",
            self.data / "norm_factors",
            self.aux / "norm_factors",
            self.aux / "custom",
        ):
            folder.mkdir(parents=True)
        self.mean = np.array([2.0, 4.0, 8.0, 16.0, 32.0], dtype=np.float64)
        self.std = np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float64)
        np.save(self.aux / "custom/mean.npy", self.mean)
        np.save(self.aux / "custom/std.npy", self.std)
        # Deliberately different defaults: incorrect norm-root/template selection must fail parity.
        np.save(self.data / "norm_factors/mean_rtma_ok_sfc_1.npy", self.mean + 99)
        np.save(self.data / "norm_factors/std_rtma_ok_sfc_1.npy", self.std * 7)
        self.yaml = self.root / "grid.yaml"
        self.yaml.write_text(
            "data_files:\n  norm_mean: custom/mean.npy\n  norm_std: custom/std.npy\n",
            encoding="utf-8",
        )
        set_active_config(load_grid_config(str(self.yaml)))
        np.save(self.data / "era5/era5_x_1.npy", np.array([257.0, 258.0]))
        lat = np.array([34.0, 35.0, 36.0])
        np.save(self.data / "era5/era5_y_1.npy", lat)
        elev = np.zeros((4, 3, 2), dtype=np.float32)
        elev[2] = np.sin(np.deg2rad(lat))[:, None]
        np.save(self.data / "era5/elev_vars_1.npy", elev)
        self.raw_paths = []
        for month in (1, 2):
            days = calendar.monthrange(2022, month)[1]
            frames = days * 24
            shape = (frames, 5, 2, 3)
            target = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 32
            target.tofile(
                self.data / f"era5/era5_rtma_ok_sfc_1_1h_2022-{month:02d}.memmap"
            )
            for mode, count in (("daily_00z", days), ("hourly", frames)):
                path = Path(
                    background_input_path(
                        str(self.data), "rtma_ok_sfc", 2022, month, mode, "raw"
                    )
                )
                # Distinct month, frame, channel, and spatial values catch incorrect indexing.
                raw = (
                    month * 100
                    + np.arange(count * 30, dtype=np.float32).reshape(count, 5, 2, 3)
                    / 8
                )
                raw.tofile(path)
                self.raw_paths.append(path)
            for var in ("tas", "sh", "psl", "u", "v"):
                for component, value in (("lon", 257.0), ("lat", 34.0), ("alt", 0.0)):
                    np.save(
                        self.data
                        / f"hadisd_processed/{var}_{component}_train-2022-{month:02d}.npy",
                        np.array([value]),
                    )
                np.zeros((frames, 1), dtype=np.float32).tofile(
                    self.data
                    / f"hadisd_processed/{var}_vals_1h_2022-{month:02d}.memmap"
                )
                np.save(
                    self.aux / f"norm_factors/mean_hadisd_{var}.npy", np.array([0.0])
                )
                np.save(
                    self.aux / f"norm_factors/std_hadisd_{var}.npy", np.array([1.0])
                )

    def tearDown(self):
        set_active_config(load_grid_config())
        gc.collect()
        self.temp.cleanup()

    def dataset(self, mode="daily_00z", **kwargs):
        options = dict(
            device="cpu",
            hadisd_mode="train",
            start_date="2022-01-31 23:00",
            end_date="2022-02-01 01:00",
            lead_time=0,
            era5_mode="rtma_ok_sfc",
            var_end=5,
            data_path=str(self.data) + "/",
            aux_data_path=str(self.aux) + "/",
            time_freq="1H",
            obs_set="rtma_surface",
            background_mode=mode,
        )
        options.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return WeatherDatasetAssimilation(**options)

    def offline(self, mode):
        subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/normalize_background.py"),
                "--data_dir",
                str(self.data),
                "--aux_data_path",
                str(self.aux),
                "--grid_config",
                str(self.yaml),
                "--background_mode",
                mode,
                "--months",
                "2022-01",
                "2022-02",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_offline_and_loader_exact_parity_all_modes_and_month_boundary(self):
        # Nonrepresentable float64 factors catch accidental float64 promotion in the loader.
        self.mean += np.array(
            [0.123456789, 0.014536789, 0.413567891, 0.135789123, 0.187654321]
        )
        self.std += np.array(
            [0.135792468, 0.198765432, 0.278912345, 0.313579246, 0.512345678]
        )
        np.save(self.aux / "custom/mean.npy", self.mean)
        np.save(self.aux / "custom/std.npy", self.std)
        hashes = {p: file_sha256(p) for p in self.raw_paths}
        for mode in ("daily_00z", "hourly"):
            self.offline(mode)
            # Omit input option to exercise backward-compatible default.
            normalized, raw = self.dataset(mode), self.dataset(
                mode, background_input="raw"
            )
            for i, (month, day, hour) in enumerate(((1, 31, 23), (2, 1, 0), (2, 1, 1))):
                with contextlib.redirect_stdout(io.StringIO()):
                    old_task, new_task = normalized[i], raw[i]
                actual = new_task["climatology_current"]
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(tuple(actual.shape), (5, 2, 3))
                torch.testing.assert_close(
                    actual, old_task["climatology_current"], rtol=0, atol=0
                )
                frame = day - 1 if mode == "daily_00z" else (day - 1) * 24 + hour
                values = (
                    month * 100
                    + (frame * 30 + np.arange(30, dtype=np.float32).reshape(5, 2, 3))
                    / 8
                )
                expected = (
                    values - self.mean.astype(np.float32)[:, None, None]
                ) / self.std.astype(np.float32)[:, None, None]
                np.testing.assert_array_equal(actual.numpy(), expected)
                # Simple fixed-weight consumer catches any downstream layout/value difference.
                consumer = torch.nn.Conv2d(5, 2, 1, bias=False)
                with torch.no_grad():
                    consumer.weight.fill_(0.25)
                    torch.testing.assert_close(
                        consumer(actual[None]),
                        consumer(old_task["climatology_current"][None]),
                        rtol=0,
                        atol=0,
                    )
                np.testing.assert_array_equal(
                    raw[i]["climatology_current"].numpy(), expected
                )
            del raw, normalized
        self.assertEqual(hashes, {p: file_sha256(p) for p in self.raw_paths})

    def test_raw_requires_no_normalized_files_and_never_falls_back(self):
        raw = self.dataset(background_input="raw")
        self.assertEqual(raw[0]["climatology_current"].shape, (5, 2, 3))
        del raw
        self.offline("daily_00z")
        self.raw_paths[0].unlink()
        with self.assertRaisesRegex(FileNotFoundError, "background_input=raw"):
            self.dataset(background_input="raw")
        default = self.dataset()
        del default

    def test_offline_default_norm_paths_still_work(self):
        subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/normalize_background.py"),
                "--data_dir",
                str(self.data),
                "--months",
                "2022-01",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        path = background_input_path(str(self.data), "rtma_ok_sfc", 2022, 1)
        normalized = np.fromfile(path, dtype=np.float32).reshape(31, 5, 2, 3)
        raw = np.fromfile(self.raw_paths[0], dtype=np.float32).reshape(31, 5, 2, 3)
        expected = (raw - (self.mean + 99).astype(np.float32)[None, :, None, None]) / (
            (self.std * 7).astype(np.float32)[None, :, None, None]
        )
        np.testing.assert_array_equal(normalized, expected)

    def test_preflight_parses_input_environment_including_regression_launcher(self):
        sys.path.insert(0, str(REPO / "scripts"))
        from check_ok_run_files import parse_train_script

        for name in (
            "dev-mul_checkpoint-infer.sh",
            "training/test-val-regression-rtma_ok_sfc-1GPU.sh",
        ):
            for option in ("raw", "normalized"):
                with mock.patch.dict("os.environ", {"BACKGROUND_INPUT": option}):
                    _, flags = parse_train_script(REPO / name)
                self.assertEqual(flags["background_input"], option)

    def test_reject_options_before_reading_data(self):
        for options in (
            {"background_input": "typo"},
            {"background_input": "raw", "diff": True},
            {"background_input": "raw", "time_freq": "1D"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.dataset(**options)

    def test_norm_shapes_values_and_float32(self):
        with contextlib.redirect_stdout(io.StringIO()):
            mean, std, _ = load_background_norms(str(self.aux), "rtma_ok_sfc", 5)
        np.testing.assert_array_equal(
            normalize_background_frame(np.broadcast_to(mean, (5, 2, 3)), mean, std), 0
        )
        np.testing.assert_array_equal(
            normalize_background_frame(
                np.broadcast_to(mean + std, (5, 2, 3)), mean, std
            ),
            1,
        )
        for bad in (
            np.ones((5, 1)),
            np.ones(4),
            np.zeros(5),
            np.full(5, np.nan),
            np.full(5, 1e-100),
        ):
            np.save(self.aux / "custom/std.npy", bad)
            with self.subTest(std=bad), self.assertRaises(ValueError):
                load_background_norms(str(self.aux), "rtma_ok_sfc", 5)

    def test_checkpoint_provenance_and_legacy_policy(self):
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, manifest = load_background_norms(
                str(self.aux), "rtma_ok_sfc", 5, "raw"
            )
        checkpoint = self.root / "epoch_1"
        checkpoint.write_bytes(b"legacy checkpoint fixture")
        check_checkpoint_normalization(
            {"background_normalization": manifest}, manifest, checkpoint
        )
        moved = dict(manifest, mean_path="elsewhere", background_input="normalized")
        compare_manifests(moved, manifest)
        changed = copy.deepcopy(manifest)
        changed["mean"][0] += 1
        changed["effective_sha256"] = vector_sha256(
            changed["mean"], changed["std"], changed["channel_order"]
        )
        with self.assertRaisesRegex(ValueError, "mismatch"):
            check_checkpoint_normalization(
                {"background_normalization": changed}, manifest, checkpoint
            )
        with self.assertRaisesRegex(ValueError, "reviewed"):
            check_checkpoint_normalization({}, manifest, checkpoint)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            check_checkpoint_normalization(
                {}, dict(manifest, background_input="normalized"), checkpoint
            )
            self.assertTrue(any("UNVERIFIED" in str(w.message) for w in recorded))
        external = self.root / "legacy.json"
        external.write_text(
            json.dumps(
                {
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "background_normalization": manifest,
                }
            )
        )
        check_checkpoint_normalization({}, manifest, checkpoint, external)
        checkpoint.write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "checksum"):
            check_checkpoint_normalization({}, manifest, checkpoint, external)

    def test_trainer_checkpoint_roundtrip_and_mismatch_before_weight_loading(self):
        from trainer import DDPTrainer

        with contextlib.redirect_stdout(io.StringIO()):
            _, _, manifest = load_background_norms(
                str(self.aux), "rtma_ok_sfc", 5, "raw"
            )
        # Exercise the actual save/load methods on CPU without a GPU DDP allocation.
        trainer = DDPTrainer.__new__(DDPTrainer)
        trainer.model = torch.nn.Linear(2, 1)
        trainer.opt = torch.optim.Adam(trainer.model.parameters())
        trainer.best_loss = 1.0
        trainer.save_path = str(self.root)
        trainer.background_normalization = manifest
        trainer.background_norm_manifest = None
        trainer._save_last_checkpoint(2, 1.0)
        path = self.root / "checkpoint_last"
        saved = torch.load(path, map_location="cpu")
        self.assertEqual(saved["background_normalization"], manifest)
        self.assertEqual(saved["epoch"], 2)
        original = trainer.model.weight.detach().clone()
        with torch.no_grad():
            trainer.model.weight.zero_()
        trainer._load_weights_if_provided(path)
        torch.testing.assert_close(trainer.model.weight, original, rtol=0, atol=0)
        # Check failure precedes loading model weights.
        trainer.background_normalization = copy.deepcopy(manifest)
        trainer.background_normalization["mean"][0] += 1
        trainer.background_normalization["effective_sha256"] = vector_sha256(
            trainer.background_normalization["mean"],
            manifest["std"],
            manifest["channel_order"],
        )
        with torch.no_grad():
            trainer.model.weight.zero_()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            trainer._load_weights_if_provided(path)
        self.assertEqual(torch.count_nonzero(trainer.model.weight).item(), 0)

    def test_hourly_diagnostics_both_representations(self):
        self.offline("hourly")
        reports = []
        for input_mode in ("normalized", "raw"):
            output = self.root / (input_mode + ".csv")
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts/diagnose_hourly_background.py"),
                    "--data_root",
                    str(self.data),
                    "--aux_data_path",
                    str(self.aux),
                    "--grid_config",
                    str(self.yaml),
                    "--background_input",
                    input_mode,
                    "--start",
                    "2022-01",
                    "--end",
                    "2022-02",
                    "--sample_every_days",
                    "15",
                    "--csv_out",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            reports.append(output.read_text())
        # Exactly representable fixture factors make inverse normalization exact here too.
        self.assertEqual(reports[0], reports[1])


if __name__ == "__main__":
    unittest.main()
