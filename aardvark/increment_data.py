"""Hourly increment products and target provenance (no PyTorch dependency)."""

import calendar
import json
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np

from background_normalization import _vectors, file_sha256, vector_sha256
from grid_config import increment_norm_path, norm_std_path
from month_manifest import parse_month

CHANNEL_ORDER = ["tas", "sh", "psl", "u", "v"]
ORIENTATION = {
    "origin": "SW",
    "axis_order": ["time", "channel", "lon", "lat"],
    "lon_direction": "ascending",
    "lat_direction": "ascending",
}
PRODUCT_FORMAT = "aardvark-grid-product-v1"
TARGET_FORMAT = "aardvark-target-normalization-v1"
MIN_VALID_COVERAGE = 0.99
MIN_STD_FRACTION = 1e-3


def sidecar_path(path):
    return Path(str(path) + ".json")


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def frame_layout(year, month, hourly=True):
    return {
        "month": f"{year:04d}-{month:02d}",
        "cadence": "1H" if hourly else "daily_00z",
        "ordering": "UTC chronological",
        "first_hour": 0,
    }


def write_source_sidecar(path, shape, year, month, hourly=True):
    atomic_json(
        sidecar_path(path),
        {
            "format": PRODUCT_FORMAT,
            "kind": "source",
            "dtype": "float32",
            "shape": list(shape),
            "frame_count": shape[0],
            "channel_order": CHANNEL_ORDER,
            "grid_orientation": ORIENTATION,
            "time_layout": frame_layout(year, month, hourly),
            "channel_order_source": "preprocessing",
            "sha256": file_sha256(path),
            "tool": "prep_urma_ok_hourly",
            "tool_version": 1,
        },
    )


def validate_product(path, shape, order, layout, allow_legacy=False, rehash=False):
    path = Path(path)
    if path.stat().st_size != int(np.prod(shape)) * 4:
        raise ValueError(f"{path}: wrong file size for float32 shape {shape}")
    metadata_path = sidecar_path(path)
    if not metadata_path.exists():
        if not allow_legacy:
            raise ValueError(
                f"{path}: missing sidecar; use --allow_legacy_inputs explicitly"
            )
        return {"sha256": file_sha256(path), "channel_order_source": "override"}
    meta = read_json(metadata_path)
    for key, expected in (
        ("format", PRODUCT_FORMAT),
        ("dtype", "float32"),
        ("shape", list(shape)),
        ("frame_count", shape[0]),
        ("channel_order", list(order)),
        ("grid_orientation", ORIENTATION),
        ("time_layout", layout),
    ):
        if meta.get(key) != expected:
            raise ValueError(f"{path}: sidecar {key} mismatch")
    if not isinstance(meta.get("sha256"), str) or len(meta["sha256"]) != 64:
        raise ValueError(f"{path}: missing SHA256")
    if rehash and file_sha256(path) != meta["sha256"]:
        raise ValueError(f"{path}: SHA256 mismatch")
    return meta


def verify_subtraction(analysis, background, increment):
    frames = sorted(
        set(np.linspace(0, len(analysis) - 1, min(7, len(analysis)), dtype=int))
    )
    for frame in frames:
        a, b, actual = analysis[frame], background[frame], increment[frame]
        expected = np.subtract(a, b, dtype=np.float32)
        mask = np.isnan(a) | np.isnan(b)
        if not np.array_equal(np.isnan(actual), mask):
            raise ValueError(f"increment NaN mask mismatch at frame {frame}")
        if not np.array_equal(
            actual[~mask].view(np.uint32), expected[~mask].view(np.uint32)
        ):
            raise ValueError(f"increment subtraction is not bit-exact at frame {frame}")
    return [int(frame) for frame in frames]


def build_increment(
    analysis_path,
    background_path,
    output_path,
    shape,
    year,
    month,
    order=CHANNEL_ORDER,
    allow_legacy=False,
    overwrite=False,
    chunk_frames=24,
):
    if chunk_frames < 1 or shape[0] != calendar.monthrange(year, month)[1] * 24:
        raise ValueError(
            "increments require complete hourly months and positive chunk size"
        )
    if len(order) != shape[1] or len(set(order)) != len(order):
        raise ValueError("expected channel order must match the channel count")
    paths = [Path(p).resolve() for p in (analysis_path, background_path, output_path)]
    if len(set(paths)) != 3:
        raise ValueError("increment output must not collide with source files")
    analysis_path, background_path, output_path = paths
    if (output_path.exists() or sidecar_path(output_path).exists()) and not overwrite:
        raise FileExistsError(output_path)
    layout = frame_layout(year, month)
    source_meta = [
        validate_product(p, shape, order, layout, allow_legacy, rehash=True)
        for p in paths[:2]
    ]
    analysis = np.memmap(analysis_path, dtype="float32", mode="r", shape=shape)
    background = np.memmap(background_path, dtype="float32", mode="r", shape=shape)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=output_path.parent, prefix=output_path.name + ".", suffix=".tmp"
    )
    os.close(fd)
    out = witness = None
    try:
        out = np.memmap(temporary, dtype="float32", mode="w+", shape=shape)
        for start in range(0, shape[0], chunk_frames):
            a, b = (
                analysis[start : start + chunk_frames],
                background[start : start + chunk_frames],
            )
            if np.isinf(a).any() or np.isinf(b).any():
                raise ValueError("increment inputs contain infinities")
            delta = np.subtract(a, b, dtype=np.float32)
            if np.isinf(delta).any():
                raise ValueError("increment subtraction overflow")
            out[start : start + chunk_frames] = delta
        out.flush()
        del out
        out = None
        witness = np.memmap(temporary, dtype="float32", mode="r", shape=shape)
        checked = verify_subtraction(analysis, background, witness)
        del witness
        witness = None
        metadata = {
            "format": PRODUCT_FORMAT,
            "kind": "increment",
            "dtype": "float32",
            "shape": list(shape),
            "frame_count": shape[0],
            "channel_order": list(order),
            "grid_orientation": ORIENTATION,
            "time_layout": layout,
            "sign_convention": "analysis_raw - background_raw",
            "channel_order_source": (
                "override"
                if any(m.get("channel_order_source") == "override" for m in source_meta)
                else "preprocessing"
            ),
            "sources": {
                name: {
                    "path": str(p),
                    "sha256": m["sha256"],
                    "channel_order_source": m["channel_order_source"],
                }
                for name, p, m in zip(
                    ("analysis", "background"), paths[:2], source_meta
                )
            },
            "sha256": file_sha256(temporary),
            "verified_frames": checked,
            "tool": "build_hourly_increment",
            "tool_version": 1,
        }
        # Remove stale metadata before replacing data: an interrupted replacement fails closed.
        sidecar_path(output_path).unlink(missing_ok=True)
        os.replace(temporary, output_path)
        atomic_json(sidecar_path(output_path), metadata)
        return metadata
    finally:
        for mapping in (out, witness, analysis, background):
            if mapping is not None:
                mapping._mmap.close()
        del analysis, background
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_target_mode(target_mode, monthly, time_freq, background_mode, diff):
    if target_mode not in ("analysis", "increment"):
        raise ValueError("target_mode must be analysis or increment")
    if target_mode == "increment" and (
        not monthly or time_freq != "1H" or background_mode != "hourly" or diff
    ):
        raise ValueError(
            "increment target requires monthly RTMA surface assimilation, 1H, hourly background and diff=0"
        )


def load_increment_norms(aux_path, era5_mode, channels=None):
    paths = [
        Path(increment_norm_path(aux_path, era5_mode, s)).resolve()
        for s in ("mean", "std")
    ]
    meta = read_json(sidecar_path(paths[0]))
    mean, std = _vectors(
        *(np.load(p, allow_pickle=False) for p in paths), channels=channels
    )
    if meta.get("format") != TARGET_FORMAT or meta.get("target_mode") != "increment":
        raise ValueError("invalid increment statistics metadata")
    order = meta.get("channel_order")
    if (
        order != CHANNEL_ORDER
        or len(order) != len(mean)
        or meta.get("dtype") != "float32"
        or meta.get("era5_mode") != era5_mode
    ):
        raise ValueError("increment normalization channel order mismatch")
    months = meta.get("training_months")
    if not isinstance(months, list) or not months or len(months) != len(set(months)):
        raise ValueError("increment statistics require unique nonempty training months")
    for month in months:
        parse_month(month, "increment statistics")
        if month not in meta.get("increment_sources", {}):
            raise ValueError("increment statistics missing source month: " + month)
    if meta.get("channel_order_source") not in ("preprocessing", "override"):
        raise ValueError("increment statistics missing channel-order provenance")
    coverage = np.asarray(meta.get("coverage", []), dtype=float)
    if (
        coverage.shape != mean.shape
        or not np.isfinite(coverage).all()
        or np.any(coverage < MIN_VALID_COVERAGE)
        or np.any(coverage > 1)
    ):
        raise ValueError("increment statistics coverage is below its threshold")
    analysis_std_path = norm_std_path(aux_path, era5_mode)
    analysis_std = np.load(analysis_std_path, allow_pickle=False)
    _, analysis_std = _vectors(np.zeros_like(analysis_std), analysis_std, len(mean))
    if meta.get("analysis_std_sha256") != file_sha256(analysis_std_path):
        raise ValueError(
            "analysis std changed since increment statistics were computed"
        )
    if np.any(std < MIN_STD_FRACTION * analysis_std):
        raise ValueError("increment std below analysis-relative floor")
    for key, path in zip(("mean_file_sha256", "std_file_sha256"), paths):
        if meta.get(key) != file_sha256(path):
            raise ValueError("increment normalization factor hash mismatch")
    if meta.get("effective_sha256") != vector_sha256(mean, std, order):
        raise ValueError("increment effective normalization hash mismatch")
    return mean[:, None, None], std[:, None, None], meta


def check_training_months(months, metadata):
    missing = set(months) - set(metadata.get("training_months", []))
    if missing:
        raise ValueError(
            "training months not covered by increment statistics: "
            + " ".join(sorted(missing))
        )


def verify_increment_sources(
    product, analysis_path, background_path, shape, year, month, rehash=False
):
    """Check current source sidecars against increment provenance without reading frames.

    Missing legacy sidecars can only be tolerated for explicitly overridden sources.
    Existing sidecars must agree even if a legacy override was used originally.
    """
    for name, path in (("analysis", analysis_path), ("background", background_path)):
        if path is None:
            continue
        source = product.get("sources", {}).get(name, {})
        if not source.get("sha256"):
            raise ValueError(f"increment missing {name} source hash")
        override = (
            source.get("channel_order_source", product.get("channel_order_source"))
            == "override"
        )
        if not sidecar_path(path).exists() and override:
            if Path(path).stat().st_size != int(np.prod(shape)) * 4:
                raise ValueError(f"{path}: source size mismatch")
            warnings.warn(
                f"UNVERIFIED legacy {name} sidecar: {path}; increment source used an explicit override"
            )
            if rehash and file_sha256(path) != source["sha256"]:
                raise ValueError(f"{path}: {name} source hash mismatch")
            continue
        current = validate_product(
            path,
            shape,
            product["channel_order"],
            frame_layout(year, month),
            rehash=rehash,
        )
        if current["sha256"] != source["sha256"]:
            raise ValueError(f"{path}: {name} source hash mismatch; rebuild increments")


def verify_increment_input(path, shape, year, month, metadata, rehash=False):
    product = validate_product(
        path, shape, metadata["channel_order"], frame_layout(year, month), rehash=rehash
    )
    if (
        product.get("kind") != "increment"
        or product.get("sign_convention") != "analysis_raw - background_raw"
    ):
        raise ValueError(f"{path}: invalid increment sign/kind")
    key = f"{year:04d}-{month:02d}"
    recorded = metadata.get("increment_sources", {}).get(key)
    if key in metadata["training_months"] and (
        not recorded or recorded["sha256"] != product["sha256"]
    ):
        raise ValueError(f"{path}: increment hash does not match statistics source")
    if product.get("channel_order_source") not in ("preprocessing", "override"):
        raise ValueError(f"{path}: missing channel-order provenance")
    return product


def check_target_checkpoint(checkpoint, current):
    expected = (
        checkpoint.get("target_normalization") if isinstance(checkpoint, dict) else None
    )
    current_mode = current.get("target_mode", "analysis") if current else "analysis"
    if expected is None:
        if current_mode == "increment":
            raise ValueError(
                "increment mode requires matching checkpoint target_normalization for every loading purpose; start from scratch"
            )
        return
    if expected.get("target_mode") != current_mode:
        raise ValueError("checkpoint target_mode mismatch")
    if current_mode == "increment":
        for key in (
            "format",
            "mean_file_sha256",
            "std_file_sha256",
            "effective_sha256",
            "channel_order",
            "training_months",
            "channel_order_source",
        ):
            if expected.get(key) is None or expected.get(key) != current.get(key):
                raise ValueError("checkpoint target normalization mismatch: " + key)
        # Paths can move between machines; recorded source identities must still agree.
        for month in current["training_months"]:
            for key in ("sha256", "channel_order_source"):
                expected_source = expected.get("increment_sources", {}).get(month, {})
                current_source = current.get("increment_sources", {}).get(month, {})
                if (
                    not current_source.get(key)
                    or expected_source.get(key) != current_source[key]
                ):
                    raise ValueError(
                        "checkpoint target source metadata mismatch: " + month
                    )
