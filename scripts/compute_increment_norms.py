"""Compute training-only increment factors from verified complete hourly products."""

import argparse
import calendar
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "aardvark"))
from background_normalization import _vectors, file_sha256, vector_sha256
from grid_config import (
    increment_month_path,
    increment_norm_path,
    load_grid_config,
    set_active_config,
    loader_grid_x_path,
    loader_grid_y_path,
    norm_std_path,
)
from increment_data import (
    CHANNEL_ORDER,
    TARGET_FORMAT,
    MIN_VALID_COVERAGE,
    MIN_STD_FRACTION,
    atomic_json,
    frame_layout,
    sidecar_path,
    validate_product,
)
from month_manifest import read_month_manifest


def compute_statistics(
    inputs,
    analysis_std,
    chunk_frames=24,
    minimum_coverage=MIN_VALID_COVERAGE,
    std_floor_fraction=MIN_STD_FRACTION,
):
    if (
        chunk_frames < 1
        or not MIN_VALID_COVERAGE <= minimum_coverage <= 1
        or not np.isfinite(std_floor_fraction)
        or std_floor_fraction < MIN_STD_FRACTION
    ):
        raise ValueError("invalid chunk size, minimum coverage or std floor")
    channels = len(analysis_std)
    count = np.zeros(channels, dtype=np.int64)
    analysis_count = np.zeros(channels, dtype=np.int64)
    mean = np.zeros(channels, dtype=np.float64)
    m2 = np.zeros(channels, dtype=np.float64)
    total_points = 0
    for increment_path, analysis_path, shape in inputs:
        inc = np.memmap(increment_path, dtype="float32", mode="r", shape=shape)
        ana = np.memmap(analysis_path, dtype="float32", mode="r", shape=shape)
        try:
            for start in range(0, shape[0], chunk_frames):
                delta, truth = (
                    np.array(inc[start : start + chunk_frames]),
                    np.array(ana[start : start + chunk_frames]),
                )
                if np.isinf(delta).any() or np.isinf(truth).any():
                    raise ValueError("statistics inputs contain infinities")
                if np.any(np.isfinite(delta) & ~np.isfinite(truth)):
                    raise ValueError("increment valid where analysis is missing")
                analysis_count += np.isfinite(truth).sum(axis=(0, 2, 3))
                total_points += delta.shape[0] * shape[2] * shape[3]
                # Merge channel-wise centered moments, avoiding E[x^2]-E[x]^2 cancellation.
                for c in range(channels):
                    values = np.asarray(delta[:, c], dtype=np.float64)
                    values = values[np.isfinite(values)]
                    n = values.size
                    if not n:
                        continue
                    batch_mean = values.mean()
                    batch_m2 = np.square(values - batch_mean).sum()
                    difference = batch_mean - mean[c]
                    combined = count[c] + n
                    m2[c] += batch_m2 + difference**2 * count[c] * n / combined
                    mean[c] += difference * n / combined
                    count[c] = combined
        finally:
            inc._mmap.close()
            ana._mmap.close()
            del inc, ana
    coverage = np.divide(
        count,
        analysis_count,
        out=np.zeros(channels, dtype=float),
        where=analysis_count > 0,
    )
    if np.any(coverage < minimum_coverage):
        raise ValueError(
            f"increment coverage below {minimum_coverage} of analysis-valid points: {coverage}"
        )
    std = np.sqrt(m2 / count)
    mean, std = _vectors(mean, std, channels)
    if np.any(std < std_floor_fraction * analysis_std):
        raise ValueError(
            "increment std below analysis-relative floor; factors were not clamped"
        )
    diagnostics = {
        "valid_count": count.tolist(),
        "analysis_valid_count": analysis_count.tolist(),
        "coverage": coverage.tolist(),
        "sample_count": total_points,
        "analysis_nan_fraction": (1 - analysis_count / total_points).tolist(),
        "increment_nan_fraction": (1 - count / total_points).tolist(),
    }
    return mean, std, diagnostics


def write_factors(mean_path, std_path, mean, std, metadata):
    temporary = []
    try:
        for path, values in ((Path(mean_path), mean), (Path(std_path), std)):
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(
                dir=path.parent, prefix=path.name + ".", suffix=".tmp"
            )
            with os.fdopen(fd, "wb") as stream:
                np.save(stream, values)
            temporary.append((name, path))
        metadata.update(
            mean_path=str(Path(mean_path).resolve()),
            std_path=str(Path(std_path).resolve()),
            mean_file_sha256=file_sha256(temporary[0][0]),
            std_file_sha256=file_sha256(temporary[1][0]),
            effective_sha256=vector_sha256(mean, std, metadata["channel_order"]),
        )
        sidecar_path(mean_path).unlink(missing_ok=True)
        for name, path in temporary:
            os.replace(name, path)
        atomic_json(sidecar_path(mean_path), metadata)
    finally:
        for name, _ in temporary:
            if os.path.exists(name):
                os.unlink(name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True)
    p.add_argument(
        "--aux_data_path", help="Factor input/output root (default: data_root)"
    )
    p.add_argument(
        "--months_file",
        required=True,
        help="Intended training months; all frames included",
    )
    p.add_argument("--era5_mode", default="rtma_ok_sfc")
    p.add_argument("--grid_config", default=str(REPO / "aardvark/grid_config_ok.yaml"))
    p.add_argument("--chunk_frames", type=int, default=24)
    p.add_argument("--minimum_coverage", type=float, default=0.99)
    p.add_argument("--std_floor_fraction", type=float, default=1e-3)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    aux_root = args.aux_data_path or args.data_root
    set_active_config(
        load_grid_config(None if args.grid_config == "default" else args.grid_config)
    )
    months = read_month_manifest(args.months_file, "--months_file")
    mean_path, std_path = [
        Path(increment_norm_path(aux_root, args.era5_mode, s)) for s in ("mean", "std")
    ]
    if not args.overwrite and any(
        p.exists() for p in (mean_path, std_path, sidecar_path(mean_path))
    ):
        raise FileExistsError(
            "increment factors already exist; use --overwrite explicitly"
        )
    analysis_std = np.load(norm_std_path(aux_root, args.era5_mode), allow_pickle=False)
    _, analysis_std = _vectors(np.zeros_like(analysis_std), analysis_std)
    nlon = np.load(loader_grid_x_path(args.data_root)).size
    nlat = np.load(loader_grid_y_path(args.data_root)).size
    inputs, sources, provenance = [], {}, set()
    order = None
    for year, month in months:
        shape = (
            calendar.monthrange(year, month)[1] * 24,
            len(analysis_std),
            nlon,
            nlat,
        )
        path = Path(
            increment_month_path(args.data_root, args.era5_mode, "1h", year, month)
        )
        meta = validate_product(
            path, shape, CHANNEL_ORDER, frame_layout(year, month), rehash=True
        )
        if (
            meta.get("kind") != "increment"
            or meta.get("sign_convention") != "analysis_raw - background_raw"
        ):
            raise ValueError("invalid increment product")
        order = meta["channel_order"]
        analysis = Path(meta["sources"]["analysis"]["path"])
        if (
            analysis.stat().st_size != np.prod(shape) * 4
            or file_sha256(analysis) != meta["sources"]["analysis"]["sha256"]
        ):
            raise ValueError("analysis witness differs from increment source")
        inputs.append((path, analysis, shape))
        sources[f"{year:04d}-{month:02d}"] = {
            "path": str(path.resolve()),
            "sha256": meta["sha256"],
            "channel_order_source": meta["channel_order_source"],
        }
        provenance.add(meta["channel_order_source"])
    mean, std, diagnostics = compute_statistics(
        inputs,
        analysis_std,
        args.chunk_frames,
        args.minimum_coverage,
        args.std_floor_fraction,
    )
    metadata = {
        "format": TARGET_FORMAT,
        "target_mode": "increment",
        "era5_mode": args.era5_mode,
        "dtype": "float32",
        "channel_order": order,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "training_months": sorted(sources),
        "increment_sources": sources,
        "channel_order_source": (
            "override" if "override" in provenance else "preprocessing"
        ),
        "minimum_coverage": args.minimum_coverage,
        "std_floor_fraction": args.std_floor_fraction,
        "analysis_std_sha256": file_sha256(norm_std_path(aux_root, args.era5_mode)),
        "tool": "compute_increment_norms",
        "tool_version": 1,
        **diagnostics,
    }
    write_factors(mean_path, std_path, mean, std, metadata)
    print("[OK] increment statistics " + str(metadata))


if __name__ == "__main__":
    main()
