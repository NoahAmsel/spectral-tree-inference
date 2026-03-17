"""Quick script to check what leverage-related columns are available."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from analysis.leveraged_sampling_analysis import load_single_run_dataframe

# Load a sample run
RUN_DIR = Path(__file__).parent.parent.parent.parent / "results" / "20260117-155215-balanced_binary_leveraged"

if RUN_DIR.exists():
    df = load_single_run_dataframe(RUN_DIR)

    print("All columns in results:")
    print("="*80)
    for col in sorted(df.columns):
        print(f"  {col}")

    print("\n" + "="*80)
    print("\nLeverage-related columns:")
    print("="*80)
    leverage_cols = [c for c in df.columns if 'leverage' in c.lower()]
    for col in leverage_cols:
        print(f"  {col}")

    print("\n" + "="*80)
    print("\nPhase1-related columns:")
    print("="*80)
    phase1_cols = [c for c in df.columns if 'phase1' in c.lower()]
    for col in phase1_cols:
        print(f"  {col}")
else:
    print(f"Directory not found: {RUN_DIR}")
    print("Please update RUN_DIR in this script to point to your leveraged results.")
