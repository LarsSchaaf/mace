#!/usr/bin/env python
"""Assign a persistent ``config_id`` to every frame of an xyz file.

The noise-resilient training logger tracks the per-config bootstrap weight by
``config_id``. If your xyz does not already contain one, training will fall back
to the frame index, but that index is only unique *within a single file*. Use
this utility to stamp an explicit, file-spanning, persistent id into the
``info`` block of every frame so the ids are stable and globally unique across
your train / valid / test files.

Examples
--------
Stamp ids 0..N-1 into a single file (in place is not done; writes a new file):

    mace_assign_config_ids --input train.xyz --output train_ids.xyz

Stamp ids across several files with a continuous, non-overlapping counter:

    mace_assign_config_ids \\
        --input train.xyz valid.xyz test.xyz \\
        --output train_ids.xyz valid_ids.xyz test_ids.xyz

Use a custom key and overwrite ids that already exist:

    mace_assign_config_ids -i data.xyz -o data_ids.xyz \\
        --key my_id --overwrite
"""

import argparse

import ase.io


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", "-i", nargs="+", required=True, help="One or more input xyz files."
    )
    parser.add_argument(
        "--output",
        "-o",
        nargs="+",
        required=True,
        help="Output xyz file(s); must match the number of inputs.",
    )
    parser.add_argument(
        "--key",
        default="config_id",
        help="info key to write the id under (default: config_id). "
        "Must match --config_id_key passed to mace_run_train.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First id to assign (default: 0). Ids are assigned with a single "
        "continuous counter across all input files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an id that is already present (default: keep existing).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.input) != len(args.output):
        raise SystemExit(
            f"Got {len(args.input)} input(s) but {len(args.output)} output(s); "
            "they must match one-to-one."
        )

    counter = args.start
    for in_path, out_path in zip(args.input, args.output):
        frames = ase.io.read(in_path, index=":")
        if not isinstance(frames, list):
            frames = [frames]
        n_written = 0
        for atoms in frames:
            if args.overwrite or atoms.info.get(args.key) is None:
                atoms.info[args.key] = counter
                n_written += 1
            counter += 1
        ase.io.write(out_path, frames)
        print(
            f"{in_path} -> {out_path}: wrote '{args.key}' to {n_written}/{len(frames)} frames."
        )
    print(f"Next free id: {counter}")


if __name__ == "__main__":
    main()
