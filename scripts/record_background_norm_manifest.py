"""Record reviewed normalization provenance for a legacy checkpoint.

This cannot recover historical statistics. Supply the norm files verified against
the original run; --reviewed_source records the evidence for that decision.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aardvark"))
from background_normalization import file_sha256, load_background_norms
from grid_config import load_grid_config, set_active_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--aux_data_path", required=True)
    parser.add_argument("--grid_config", default="default")
    parser.add_argument("--era5_mode", default="rtma_ok_sfc")
    parser.add_argument(
        "--background_mode", choices=["daily_00z", "hourly"], default="daily_00z"
    )
    parser.add_argument(
        "--reviewed_source",
        required=True,
        help="Evidence identifying the original run's norm factors",
    )
    parser.add_argument(
        "--output", help="Default: <checkpoint>.normalization.json; never overwrites"
    )
    args = parser.parse_args()
    if not args.reviewed_source.strip():
        parser.error(
            "--reviewed_source must describe the original normalization evidence"
        )
    checkpoint_hash = file_sha256(args.checkpoint)
    set_active_config(
        load_grid_config(None if args.grid_config == "default" else args.grid_config)
    )
    _, _, manifest = load_background_norms(
        args.aux_data_path,
        args.era5_mode,
        background_input="raw",
        background_mode=args.background_mode,
    )
    output = args.output or args.checkpoint + ".normalization.json"
    with open(output, "x", encoding="utf-8") as stream:
        json.dump(
            {
                "checkpoint_sha256": checkpoint_hash,
                "reviewed_source": args.reviewed_source,
                "background_normalization": manifest,
            },
            stream,
            indent=2,
        )
    print("Recorded reviewed legacy normalization: " + output)


if __name__ == "__main__":
    main()
