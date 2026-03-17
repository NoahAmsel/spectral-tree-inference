#!/usr/bin/env python3
"""
One-time script to clean up incomplete cache entries.

This removes any cache directories that don't have a .complete sentinel file,
which indicates they were created by interrupted experiments.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.persistent_cache import clean_incomplete_caches

if __name__ == "__main__":
    print("Scanning cache for incomplete entries...")
    removed = clean_incomplete_caches()
    print(f"\n✓ Cleanup complete: {removed} incomplete cache entries removed")

    if removed == 0:
        print("  Your cache is already clean!")
    else:
        print(f"  Freed up disk space by removing {removed} incomplete directories")
