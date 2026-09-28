"""Background normalization and checkpoint provenance, independent of PyTorch."""

import hashlib
import io
import json
from pathlib import Path
import warnings

import numpy as np

from grid_config import norm_mean_path, norm_std_path


FORMAT = "aardvark-background-normalization-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_background_input(background_input, monthly=True, diff=False):
    if background_input not in ("normalized", "raw"):
        raise ValueError("background_input must be 'normalized' or 'raw'")
    if background_input == "raw" and (not monthly or diff):
        raise ValueError("raw background requires the monthly RTMA path with diff=0")


def _vectors(mean, std, channels=None):
    mean, std = np.asarray(mean), np.asarray(std)
    if mean.ndim != 1 or mean.size == 0 or mean.shape != std.shape:
        raise ValueError("background mean/std must be matching nonempty (C,) vectors")
    if mean.dtype.kind not in "fiu" or std.dtype.kind not in "fiu":
        raise ValueError("background mean/std must contain real numeric values")
    if channels is not None and mean.size != channels:
        raise ValueError("background mean/std channel count does not match background")
    if (
        not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(std))
        or np.any(std <= 0)
    ):
        raise ValueError(
            "background means must be finite; std must be finite and positive"
        )
    # Cast BEFORE broadcasting, matching normalize_background.py's arithmetic.
    mean, std = mean.astype(np.float32), std.astype(np.float32)
    if (
        not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(std))
        or np.any(std <= 0)
    ):
        raise ValueError(
            "background factors are not representable as finite positive-std float32"
        )
    return mean, std


def vector_sha256(mean, std, channel_order):
    digest = hashlib.sha256(json.dumps(channel_order, separators=(",", ":")).encode())
    digest.update(np.asarray(mean, dtype="<f4").tobytes())
    digest.update(np.asarray(std, dtype="<f4").tobytes())
    return digest.hexdigest()


def load_background_norms(
    aux_data_path,
    era5_mode,
    channels=None,
    background_input="normalized",
    background_mode="daily_00z",
):
    paths = [
        Path(norm_mean_path(aux_data_path, era5_mode)).resolve(),
        Path(norm_std_path(aux_data_path, era5_mode)).resolve(),
    ]
    contents = [p.read_bytes() for p in paths]
    mean, std = _vectors(
        *(np.load(io.BytesIO(content), allow_pickle=False) for content in contents),
        channels=channels,
    )
    order = (
        ["tas", "sh", "psl", "u", "v"]
        if era5_mode == "rtma_ok_sfc" and mean.size == 5
        else ["{}:ch{}".format(era5_mode, i) for i in range(mean.size)]
    )
    manifest = {
        "format": FORMAT,
        "era5_mode": era5_mode,
        "dtype": "float32",
        "channel_order": order,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "mean_path": str(paths[0]),
        "std_path": str(paths[1]),
        "mean_file_sha256": hashlib.sha256(contents[0]).hexdigest(),
        "std_file_sha256": hashlib.sha256(contents[1]).hexdigest(),
        "effective_sha256": vector_sha256(mean, std, order),
        "background_input": background_input,
        "background_mode": background_mode,
    }
    print(
        "[background normalization] " + json.dumps(manifest, sort_keys=True), flush=True
    )
    return mean[:, None, None], std[:, None, None], manifest


def normalize_background_frame(raw, mean, std):
    """Return a new float32 array; never modify the source memmap."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or mean.shape != (raw.shape[0], 1, 1) or std.shape != mean.shape:
        raise ValueError("expected background (C, nlon, nlat) and factors (C, 1, 1)")
    return (raw - mean) / std


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError("invalid background normalization manifest format")
    mean, std = _vectors(manifest.get("mean"), manifest.get("std"))
    order = manifest.get("channel_order")
    if (
        not isinstance(order, list)
        or len(order) != mean.size
        or any(not isinstance(x, str) for x in order)
    ):
        raise ValueError("invalid normalization channel order")
    if manifest.get("dtype") != "float32" or manifest.get(
        "effective_sha256"
    ) != vector_sha256(mean, std, order):
        raise ValueError("invalid normalization vector hash/dtype")
    return manifest


def compare_manifests(expected, current):
    validate_manifest(expected)
    validate_manifest(current)
    if expected["effective_sha256"] != current["effective_sha256"] or expected.get(
        "era5_mode"
    ) != current.get("era5_mode"):
        raise ValueError(
            "checkpoint background normalization mismatch: mean/std or channels changed"
        )
    for key in ("mean_file_sha256", "std_file_sha256"):
        if expected.get(key) != current.get(key):
            warnings.warn(
                "Normalization file hash changed, but effective float32 factors agree: "
                + key
            )


def check_checkpoint_normalization(
    checkpoint, current, checkpoint_path, legacy_manifest=None
):
    if current is None:
        return
    expected = (
        checkpoint.get("background_normalization")
        if isinstance(checkpoint, dict)
        else None
    )
    if expected is None and legacy_manifest:
        with open(legacy_manifest, encoding="utf-8") as stream:
            external = json.load(stream)
        if external.get("checkpoint_sha256") != file_sha256(checkpoint_path):
            raise ValueError(
                "legacy normalization manifest checkpoint checksum mismatch"
            )
        expected = external.get("background_normalization")
        if expected is None:
            raise ValueError("legacy manifest is missing background_normalization")
    if expected is None:
        if current["background_input"] == "raw":
            raise ValueError(
                "Raw mode requires checkpoint normalization metadata or a reviewed "
                "--background_norm_manifest tied to this checkpoint's SHA256"
            )
        warnings.warn(
            "UNVERIFIED background normalization: legacy checkpoint has no metadata; "
            "configured factors cannot establish normalized memmap provenance"
        )
        return
    compare_manifests(expected, current)
