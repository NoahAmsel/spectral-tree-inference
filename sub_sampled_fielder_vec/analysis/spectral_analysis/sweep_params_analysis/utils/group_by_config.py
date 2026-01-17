"""Group experimental results by configuration (N, L)."""

from typing import List, Dict, Any
from collections import defaultdict


def group_by_config(rows: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
    """Group results by (num_taxa, sequence_length), sorted by p.

    Args:
        rows: List of result dictionaries

    Returns:
        Dictionary mapping (N, L) -> list of rows sorted by p
    """
    grouped = defaultdict(list)

    for row in rows:
        key = (row["num_taxa"], row["sequence_length"])
        grouped[key].append(row)

    # Sort each group by sampling rate
    for key in grouped:
        grouped[key].sort(key=lambda r: r["p"])

    return dict(grouped)
