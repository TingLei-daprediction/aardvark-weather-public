"""Create and compare compact JSON regression summaries (Python 3.8+)."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from check_val_regression import SHAPE_ONLY_FILES, VALUE_FILES


FORMAT = "aardvark-val-summary-v1"
BLOCKS = 64
SAMPLES = 256
METRICS = VALUE_FILES[:3]


def summarize_array(array, metric=False):
    flat = np.asarray(array).reshape(-1)
    if flat.dtype.kind not in "fiu":
        raise ValueError("expected a real numeric array")
    result = {"shape": list(array.shape)}
    # Preserve nonfinite locations as well as counts; never emit invalid JSON.
    mask = np.zeros(flat.size, dtype=np.uint8)
    mask[np.isnan(flat)] = 1
    mask[np.isposinf(flat)] = 2
    mask[np.isneginf(flat)] = 3
    result["nonfinite"] = {
        "nan": int(np.count_nonzero(mask == 1)),
        "posinf": int(np.count_nonzero(mask == 2)),
        "neginf": int(np.count_nonzero(mask == 3)),
        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
    }
    if metric:
        result["values"] = [float(x) if np.isfinite(x) else None for x in flat]
        return result
    indices = np.linspace(0, flat.size - 1, min(SAMPLES, flat.size), dtype=int)
    result["samples"] = [
        float(flat[i]) if np.isfinite(flat[i]) else None for i in indices
    ]
    result["blocks"] = []
    for block in np.array_split(flat, min(BLOCKS, flat.size) or 1):
        finite = block[np.isfinite(block)].astype(np.float64)
        if not finite.size:
            result["blocks"].append(None)
            continue
        result["blocks"].append(
            {
                "min": float(finite.min()),
                "max": float(finite.max()),
                "mean": float(finite.mean()),
                "std": float(finite.std()),
                "rms": float(np.sqrt(np.mean(finite * finite))),
            }
        )
    return result


def summarize(run_dir, provenance):
    files = {}
    for name in VALUE_FILES + SHAPE_ONLY_FILES:
        array = np.load(Path(run_dir) / name, mmap_mode="r", allow_pickle=False)
        files[name] = (
            summarize_array(array, metric=name in METRICS)
            if name in VALUE_FILES
            else {"shape": list(array.shape)}
        )
    return {"format": FORMAT, "provenance": provenance, "files": files}


def compare(ref, new, rtol, atol, path="files"):
    """Return mismatches; structure and integers match exactly, floats tolerantly."""
    if type(ref) is not type(new):
        return [path + ": type mismatch"]
    if isinstance(ref, dict):
        if ref.keys() != new.keys():
            return [path + ": fields differ"]
        return [
            failure
            for key in ref
            for failure in compare(ref[key], new[key], rtol, atol, path + "." + key)
        ]
    if isinstance(ref, list):
        if len(ref) != len(new):
            return [path + ": lengths differ"]
        return [
            failure
            for i, (a, b) in enumerate(zip(ref, new))
            for failure in compare(a, b, rtol, atol, "{}[{}]".format(path, i))
        ]
    if isinstance(ref, float):
        matches = (
            math.isfinite(ref)
            and math.isfinite(new)
            and abs(new - ref) <= atol + rtol * abs(ref)
        )
    else:
        matches = ref == new
    return [] if matches else ["{}: {} -> {}".format(path, ref, new)]


def read_summary(path):
    with open(path, encoding="utf-8") as stream:
        summary = json.load(stream)
    if not isinstance(summary, dict):
        raise ValueError("expected a summary object: " + str(path))
    if summary.get("format") != FORMAT:
        raise ValueError("unsupported summary format: " + str(path))
    files = summary.get("files", {})
    if set(files) != set(VALUE_FILES + SHAPE_ONLY_FILES):
        raise ValueError("missing or unexpected output entries: " + str(path))
    # Reject incomplete summaries even when both inputs are incomplete alike.
    for name, entry in files.items():
        if not isinstance(entry, dict):
            raise ValueError("invalid file entry: " + name)
        shape = entry.get("shape")
        if not isinstance(shape, list) or any(
            type(n) is not int or n < 0 for n in shape
        ):
            raise ValueError("invalid shape: " + name)
        size = math.prod(shape)
        expected = {"shape"}
        if name in VALUE_FILES:
            expected |= (
                {"nonfinite", "values"}
                if name in METRICS
                else {"nonfinite", "samples", "blocks"}
            )
            counts = entry.get("nonfinite", {})
            if set(counts) != {"nan", "posinf", "neginf", "mask_sha256"}:
                raise ValueError("invalid nonfinite fields: " + name)
            for key in ("nan", "posinf", "neginf"):
                if type(counts[key]) is not int or not 0 <= counts[key] <= size:
                    raise ValueError("invalid nonfinite count: " + name)
            digest = counts["mask_sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(
                c not in "0123456789abcdef" for c in digest
            ):
                raise ValueError("invalid nonfinite mask hash: " + name)
            lengths = (
                {"values": size}
                if name in METRICS
                else {
                    "samples": min(SAMPLES, size),
                    "blocks": min(BLOCKS, size) or 1,
                }
            )
            for key, length in lengths.items():
                if not isinstance(entry.get(key), list) or len(entry[key]) != length:
                    raise ValueError("invalid " + key + ": " + name)
                for value in entry[key]:
                    if value is None:
                        continue
                    if key == "blocks":
                        if not isinstance(value, dict) or set(value) != {
                            "min", "max", "mean", "std", "rms",
                        }:
                            raise ValueError("invalid block statistics: " + name)
                        values = value.values()
                    else:
                        values = [value]
                    if any(
                        type(x) is not float or not math.isfinite(x) for x in values
                    ):
                        raise ValueError("invalid numeric summary: " + name)
        if set(entry) != expected:
            raise ValueError("invalid fields: " + name)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="post-process a completed run")
    create.add_argument("run_dir")
    create.add_argument("output", help="JSON text file; refuses to overwrite")
    create.add_argument("--commit", default="unknown")
    create.add_argument("--checkpoint", default="unknown")
    check = commands.add_parser("compare", help="compare two JSON text files")
    check.add_argument("reference")
    check.add_argument("candidate")
    check.add_argument("--rtol", type=float, default=1e-6)
    check.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()
    try:
        if args.command == "create":
            summary = summarize(
                args.run_dir, {"commit": args.commit, "checkpoint": args.checkpoint}
            )
            encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
            with open(args.output, "x", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded + "\n")
            print("[OK] wrote " + args.output)
            return 0
        if any(not math.isfinite(x) or x < 0 for x in (args.rtol, args.atol)):
            raise ValueError("tolerances must be finite and nonnegative")
        failures = compare(
            read_summary(args.reference),
            read_summary(args.candidate),
            args.rtol,
            args.atol,
        )
        for failure in failures[:30]:
            print("[FAIL] " + failure)
        if failures:
            print("[FAIL] {} mismatches".format(len(failures)))
            return 1
        print("[OK] compact summaries match within tolerance")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("[ERROR] " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
