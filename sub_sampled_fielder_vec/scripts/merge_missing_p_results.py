#!/usr/bin/env python3
"""
Merge missing p-value results into the original results directory.

This script updates the original results.json files with computed values
from the missing p-value experiments, replacing guardrail_low entries
with actual computed data.
"""

import json
import os
from pathlib import Path

# The p values that were computed separately
MISSING_P_VALUES = [
    0.001,
    0.0016378937069540646,
    0.0026826957952797246,
    0.004393970560760791,
    0.005179474679231213,
    0.0062,
    0.0071968567300115215,
    0.0072,
]

# Base paths
RESULTS_BASE = Path("/Users/itaygonnen/Python_Repos/spectral-tree-inference/sub_sampled_fielder_vec/results")
ORIGINAL_DIR = RESULTS_BASE / "20251130-193600-balanced_tree_mu_01"

# Source directories for missing p-value results
MISSING_P_DIRS = [
    RESULTS_BASE / "20251130-212159-balanced_tree_mu_01_8192_missing_p_values",
    RESULTS_BASE / "20251201-095800-balanced_tree_mu_01_n512_L500-n2048_L10000_missing_p_values",
    RESULTS_BASE / "20251201-112924-balanced_tree_mu_01_n4096_L500-1000-5000_missing_p_values",
]


def load_json_with_special_values(filepath):
    """Load JSON file, handling NaN and Infinity values."""
    with open(filepath, 'r') as f:
        content = f.read()
    # JSON doesn't support NaN/Infinity, but Python's json module can be tricked
    # by replacing these with valid Python representations
    return json.loads(content.replace('NaN', 'null').replace('Infinity', '"__INFINITY__"').replace('-Infinity', '"__NEG_INFINITY__"'))


def save_json_with_special_values(data, filepath):
    """Save JSON file, properly handling NaN and Infinity values."""
    # First convert to JSON string
    json_str = json.dumps(data, indent=2)
    # Replace our placeholders with the original special values
    json_str = json_str.replace('"__INFINITY__"', 'Infinity')
    json_str = json_str.replace('"__NEG_INFINITY__"', '-Infinity')
    # Handle null -> NaN for numeric fields (we need to be careful here)
    # Actually, we should preserve the structure as it was
    with open(filepath, 'w') as f:
        f.write(json_str)


def find_row_by_p(rows, p_value, tolerance=1e-10):
    """Find a row by its p value with floating point tolerance."""
    for i, row in enumerate(rows):
        if row.get('p') is not None and abs(row['p'] - p_value) < tolerance:
            return i, row
    return None, None


def merge_results(original_path, missing_path):
    """Merge missing p-value results into original results."""
    print(f"  Loading original: {original_path}")
    print(f"  Loading missing: {missing_path}")
    
    # Load both files
    original_data = load_json_with_special_values(original_path)
    missing_data = load_json_with_special_values(missing_path)
    
    updates_made = 0
    
    # For each row in the missing results
    for missing_row in missing_data['rows']:
        p_value = missing_row.get('p')
        if p_value is None:
            continue
            
        # Check if this is one of the missing p values
        is_missing_p = any(abs(p_value - mp) < 1e-10 for mp in MISSING_P_VALUES)
        if not is_missing_p:
            continue
            
        # Find the corresponding row in original
        idx, original_row = find_row_by_p(original_data['rows'], p_value)
        
        if idx is not None:
            # Check if the original has guardrail data and missing has computed data
            original_source = original_row.get('result_source', '')
            missing_source = missing_row.get('result_source', '')
            
            if 'guardrail' in str(original_source) and missing_source == 'computed':
                print(f"    Updating p={p_value}: {original_source} -> {missing_source}")
                original_data['rows'][idx] = missing_row
                updates_made += 1
            elif missing_source == 'computed':
                print(f"    Updating p={p_value}: {original_source} -> {missing_source}")
                original_data['rows'][idx] = missing_row
                updates_made += 1
    
    if updates_made > 0:
        # Sort rows by p value
        original_data['rows'].sort(key=lambda x: x.get('p', 0) if x.get('p') is not None else 0)
        save_json_with_special_values(original_data, original_path)
        print(f"    Saved {updates_made} updates to {original_path}")
    else:
        print(f"    No updates needed")
    
    return updates_made


def main():
    """Main merge process."""
    total_updates = 0
    
    # Find all n_L directories to process
    for missing_dir in MISSING_P_DIRS:
        if not missing_dir.exists():
            print(f"WARNING: Missing directory not found: {missing_dir}")
            continue
            
        print(f"\nProcessing source: {missing_dir.name}")
        
        # List all n*_L* subdirectories
        for subdir in sorted(missing_dir.iterdir()):
            if not subdir.is_dir() or not subdir.name.startswith('n'):
                continue
                
            missing_results = subdir / "results.json"
            original_results = ORIGINAL_DIR / subdir.name / "results.json"
            
            if not missing_results.exists():
                print(f"  Skipping {subdir.name}: no results.json in missing dir")
                continue
                
            if not original_results.exists():
                print(f"  Skipping {subdir.name}: no results.json in original dir")
                continue
                
            print(f"\n  Merging {subdir.name}:")
            updates = merge_results(original_results, missing_results)
            total_updates += updates
    
    print(f"\n{'='*60}")
    print(f"Total updates made: {total_updates}")


if __name__ == "__main__":
    main()
















