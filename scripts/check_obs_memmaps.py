"""
Check observation memmaps and report inferred shapes/time lengths.

This script uses file sizes to infer the time dimension and compares to
the expected shapes in aardvark/data_shapes.py.
"""

import argparse
import os
from pathlib import Path

import numpy as np

from aardvark import data_shapes


def bytes_per_item():
    return 4  # float32


def file_size(path):
    return os.path.getsize(path)


def infer_time(file_bytes, fixed_dims):
    denom = bytes_per_item()
    for d in fixed_dims:
        denom *= d
    if file_bytes % denom != 0:
        return None
    return file_bytes // denom


def report_memmap(name, path, expected_shape, time_first=True):
    if not path.exists():
        print(f"[MISSING] {name}: {path}")
        return

    size = file_size(path)
    exp_bytes = np.prod(expected_shape) * bytes_per_item()
    status = "OK" if size == exp_bytes else "SIZE_MISMATCH"

    if time_first:
        fixed = expected_shape[1:]
    else:
        fixed = expected_shape[:-1]
    inferred_time = infer_time(size, fixed)

    print(f"[{status}] {name}: {path}")
    print(f"  expected_shape={expected_shape} expected_bytes={exp_bytes}")
    print(f"  actual_bytes={size} inferred_time={inferred_time}")


def report_npy(name, path):
    if not path.exists():
        print(f"[MISSING] {name}: {path}")
        return
    arr = np.load(path)
    print(f"[OK] {name}: {path} shape={arr.shape} dtype={arr.dtype}")


def main():
    p = argparse.ArgumentParser(description="Check observation memmaps and sizes")
    p.add_argument("--data_path", required=True, help="Root of observation memmaps")
    args = p.parse_args()

    root = Path(args.data_path)

    # Station data
    report_memmap(
        "IGRA_Y",
        root / "igra/1999_2021_igra_y.mmap",
        data_shapes.IGRA_Y_SHAPE,
    )
    report_memmap(
        "IGRA_X",
        root / "igra/1999_2021_igra_x.mmap",
        data_shapes.IGRA_X_SHAPE,
    )
    report_memmap(
        "ICOADS_Y",
        root / "icoads/1999_2021_icoads_y.mmap",
        data_shapes.ICOADS_Y_SHAPE,
    )
    report_memmap(
        "ICOADS_X",
        root / "icoads/1999_2021_icoads_x.mmap",
        data_shapes.ICOADS_X_SHAPE,
    )

    # Gridded obs
    report_memmap(
        "AMSU-A",
        root / "amsua/2007_2021_amsua.mmap",
        data_shapes.AMSUA_Y_SHAPE,
    )
    report_memmap(
        "AMSU-B",
        root / "amsub_mhs/2007_2021_amsub.mmap",
        data_shapes.AMSUB_Y_SHAPE,
    )
    report_memmap(
        "ASCAT",
        root / "ascat/2007_2021_ascat.mmap",
        data_shapes.ASCAT_Y_SHAPE,
    )
    report_memmap(
        "HIRS",
        root / "hirs/2007_2021_hirs.mmap",
        data_shapes.HIRS_Y_SHAPE,
    )
    report_memmap(
        "GRIDSAT",
        root / "gridsat/gridsat_data.mmap",
        data_shapes.GRIDSAT_Y_SHAPE,
    )
    report_memmap(
        "IASI",
        root / "2007_2021_iasi_subset.mmap",
        data_shapes.IASI_Y_SHAPE,
    )

    # Satellite grid coords
    report_npy("GRIDSAT_X", root / "gridsat/sat_x.npy")
    report_npy("GRIDSAT_Y", root / "gridsat/sat_y.npy")


if __name__ == "__main__":
    main()
