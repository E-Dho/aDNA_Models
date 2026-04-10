#!/usr/bin/env python3
"""Collapse Group ID values to their country prefix for reuse in plot_latents.py."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_PATTERN = r"^(?P<country>[^_]+)_.*$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collapse values like 'Hungary_LateAvar.AG' to 'Hungary' in a CSV so "
            "plot_latents.py can color by country-level Group IDs."
        )
    )
    parser.add_argument("--input_csv", required=True, help="Input CSV containing Group ID values")
    parser.add_argument("--output_csv", required=True, help="Path for the rewritten CSV")
    parser.add_argument(
        "--group_column",
        default="Group ID",
        help="Name of the Group ID column to collapse",
    )
    parser.add_argument(
        "--output_column",
        default="Group ID Country",
        help="Output column name when not overwriting the original Group ID column",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Regex used to extract the country prefix; must define a 'country' group",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the original Group ID column instead of creating a new one",
    )
    parser.add_argument(
        "--na_value",
        default="NA",
        help="Replacement value for missing Group ID entries",
    )
    parser.add_argument(
        "--input_sep",
        default=None,
        help="Input delimiter. Defaults to tab for .tsv/.anno, otherwise comma.",
    )
    parser.add_argument(
        "--output_sep",
        default=None,
        help="Output delimiter. Defaults to the input delimiter.",
    )
    return parser.parse_args()


def collapse_group_ids(series: pd.Series, pattern: re.Pattern[str], na_value: str) -> pd.Series:
    filled = series.fillna(na_value).astype(str)

    def collapse(value: str) -> str:
        match = pattern.match(value)
        if match is None:
            return value
        return match.group("country")

    return filled.map(collapse)


def infer_sep(path: Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    if path.suffix in {".tsv", ".anno"}:
        return "\t"
    return ","


def main() -> None:
    args = parse_args()
    pattern = re.compile(args.pattern)
    if "country" not in pattern.groupindex:
        raise ValueError("The regex passed to --pattern must define a named group 'country'")

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    input_sep = infer_sep(input_path, args.input_sep)
    output_sep = infer_sep(output_path, args.output_sep) if args.output_sep is not None else input_sep

    frame = pd.read_csv(input_path, sep=input_sep, low_memory=False)
    if args.group_column not in frame.columns:
        raise ValueError(f"Column '{args.group_column}' not found in {input_path}")

    collapsed = collapse_group_ids(frame[args.group_column], pattern=pattern, na_value=args.na_value)
    target_column = args.group_column if args.overwrite else args.output_column
    frame = frame.copy()
    frame[target_column] = collapsed

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, sep=output_sep, index=False)

    changed = int((frame[args.group_column].fillna(args.na_value).astype(str) != collapsed).sum())
    print(f"Wrote {output_path}")
    print(f"Source column: {args.group_column}")
    print(f"Target column: {target_column}")
    print(f"Rows changed: {changed}")
    print("Top collapsed labels:")
    print(collapsed.value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
