#!/usr/bin/env python3
"""Merge per-(n, L) sweep outputs into a single grid JSON (and optional plot)."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import sys

DIR_PATTERN = re.compile(r"^n(?P<num>\d+)_L(?P<len>\d+)$")

PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.append(str(PKG_ROOT))


def _iter_result_dirs(run_dir: Path) -> Iterable[Tuple[Path, int, int]]:
    """Yield (path, n, L) for each matching subdirectory."""
    for entry in sorted(run_dir.iterdir()):
        if not entry.is_dir():
            continue
        match = DIR_PATTERN.match(entry.name)
        if not match:
            continue
        yield entry, int(match.group("num")), int(match.group("len"))


def _safe_float(value, default: float = math.inf) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def merge_run_directory(run_dir: Path) -> Dict[str, List]:
    """
    Combine every `results.json` under `run_dir/n{n}_L{L}` into a single table
    with explicit `num_taxa` and `sequence_length` columns.
    """
    column_order: List[str] = ["num_taxa", "sequence_length"]
    seen_columns = set(column_order)
    merged_rows: List[Dict] = []

    for dir_path, n_val, seq_len in _iter_result_dirs(run_dir):
        results_path = dir_path / "results.json"
        if not results_path.exists():
            continue

        with results_path.open("r") as fh:
            data = json.load(fh)

        file_columns = data.get("columns")
        row_entries = data.get("rows")
        if not isinstance(file_columns, list) or not isinstance(row_entries, list):
            continue  # skip malformed files

        for col in file_columns:
            if col not in seen_columns:
                seen_columns.add(col)
                column_order.append(col)

        for row in row_entries:
            if not isinstance(row, dict):
                continue
            merged_row = {col: row.get(col) for col in file_columns}
            merged_row["num_taxa"] = n_val
            merged_row["sequence_length"] = seq_len
            merged_rows.append(merged_row)

    if not merged_rows:
        raise ValueError(f"No per-configuration results found under {run_dir}")

    merged_rows.sort(
        key=lambda r: (
            r.get("sequence_length", 0),
            r.get("num_taxa", 0),
            _safe_float(r.get("p")),
        )
    )

    return {"columns": column_order, "rows": merged_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge n/L sweep results into a single grid file."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Experiment directory containing n*_L*/results.json subfolders.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Destination for the merged JSON (defaults to <run_dir>/results_grid_merged.json).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If provided, regenerate the faceted partition-agreement plot.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Optional override for the plot output path (defaults to <run_dir>/partition_agreement.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    merged = merge_run_directory(run_dir)

    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else run_dir / "results_grid_merged.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as fh:
        json.dump(merged, fh, indent=2, allow_nan=True)
    print(f"Wrote merged grid with {len(merged['rows'])} rows to {output_json}")

    if args.plot:
        plot_path = (
            args.plot_path.expanduser().resolve()
            if args.plot_path
            else run_dir / "partition_agreement.png"
        )
        os.environ.setdefault("MPLBACKEND", "Agg")
        from src.utils.plotting import plot_taxa_sweep, _extract_model_name
        
        # Extract model name from directory
        model_name = _extract_model_name(run_dir.name)
        
        # Read config for subtitle with all parameters
        config_path = run_dir / "sweep_config.json"
        if config_path.exists():
            with config_path.open("r") as f:
                sweep_config = json.load(f)
            seq_len = sweep_config["sequence_length_values"][0]
            mu = sweep_config["mutation_rate"]
            ne = sweep_config.get("tree_params", {}).get("pop_size", "N/A")
            bootstrap_reps = sweep_config.get("bootstrap_reps", "N/A")
            subtitle = f"$L = {seq_len}$, $\\mu = {mu}$, $N_e = {ne}$, {bootstrap_reps} bootstrap reps"
        else:
            # Fallback: Get sequence length from merged data
            seq_lengths = sorted(set(r.get("sequence_length", 0) for r in merged["rows"]))
            if seq_lengths:
                seq_len = seq_lengths[0]
                subtitle = f"$L = {seq_len}$"
            else:
                subtitle = None
        
        plot_taxa_sweep(
            json_path=str(output_json),
            output_path=str(plot_path),
            model_name=model_name,
            subtitle=subtitle,
        )
        print(f"Wrote taxa sweep plot to {plot_path}")


if __name__ == "__main__":
    main()
















