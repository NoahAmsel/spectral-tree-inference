# Logging and Progress Tracking Guide

## Overview

The experiment system now includes comprehensive logging and progress tracking to help you monitor long-running experiments, especially on cloud VMs.

## Log Files Created

### 1. Main Log File: `experiment.log`
**Location**: `results/<timestamp>-<run_name>/experiment.log`

**Contents**:
- All log messages with timestamps
- Progress updates from middle-out processing
- Round-by-round completion times
- Guardrails triggers
- Final summary statistics

**Format**:
```
[HH:MM:SS] COMPONENT | LEVEL | Message
```

**Example**:
```
[00:19:41] EXPERIMENT_RUNNER | INFO | Log file created: /path/to/experiment.log
[00:19:41] BOOTSTRAP_SWEEP | INFO | n=8192, L=500 preparing experiment data…
[00:19:45] MIDDLE_OUT | INFO | Phase 1: Computing middle seed p[12]=0.01
[00:19:45] MIDDLE_OUT | INFO |   This will take ~60s for 10 bootstrap reps...
[00:20:47] MIDDLE_OUT | INFO | Middle result (took 62.3s): partition_agreement_M=85.2%
[00:20:47] MIDDLE_OUT | INFO |   💾 Checkpoint saved: progress_checkpoint.json
[00:20:47] MIDDLE_OUT | INFO |
Round 1: offset=1
[00:20:47] MIDDLE_OUT | INFO |   High side: indices [13, 14, 15, 16] (p=[0.0139, 0.0165, 0.0193, 0.0268])
[00:20:47] MIDDLE_OUT | INFO |   Low side: indices [11, 10, 9, 8] (p=[0.01, 0.0082, 0.0072, 0.00620])
[00:20:47] MIDDLE_OUT | INFO |   Processing 8 p-values in parallel...
[00:20:47] MIDDLE_OUT | INFO |   Estimated time: ~60s per p-value (with 8 workers in parallel)
[00:20:47] MIDDLE_OUT | INFO |   [Workers are running silently - check back in ~60s]
[00:21:50] MIDDLE_OUT | INFO |   ✓ Completed in 63.2s
[00:21:50] MIDDLE_OUT | INFO |     p[13]=0.0139: partition_agreement_M=92.5%
[00:21:50] MIDDLE_OUT | INFO |     p[14]=0.0165: partition_agreement_M=96.8%
...
```

### 2. Progress Checkpoint: `progress_checkpoint.json`
**Location**: `results/<timestamp>-<run_name>/progress_checkpoint.json`

**Purpose**: Crash recovery and real-time progress monitoring

**Contents**:
```json
{
  "timestamp": "2025-11-25 00:21:50",
  "round": 2,
  "elapsed_seconds": 185.3,
  "p_values": [0.0001, 0.000139, ..., 1.0],
  "results": [
    {"p": 0.0001, "status": "pending"},
    {"p": 0.000139, "status": "skipped"},
    {
      "p": 0.01,
      "status": "completed",
      "sign_agreement": 92.3,
      "partition_agreement_M": 85.2,
      "partition_agreement_S": 83.1,
      "dot_product": 0.912
    },
    ...
  ]
}
```

**Statuses**:
- `pending`: Not yet computed
- `completed`: Successfully computed
- `skipped`: Skipped due to guardrails

**Updates**: Saved after every round (every ~60s with 8 workers)

## Monitoring Your Experiment

### On the VM (while experiment runs):

**Watch the log file in real-time**:
```bash
# Terminal 1: Run experiment
python3 main_taxa_sweep.py

# Terminal 2: Follow the log
tail -f results/<timestamp>-*/experiment.log
```

**Check progress checkpoint**:
```bash
# View latest checkpoint
cat results/<timestamp>-*/progress_checkpoint.json | jq '.'

# Count completed p-values
cat results/<timestamp>-*/progress_checkpoint.json | jq '.results[] | select(.status=="completed") | .p'

# Check elapsed time
cat results/<timestamp>-*/progress_checkpoint.json | jq '.elapsed_seconds'
```

### After VM Disconnect/Crash:

**Find your experiment**:
```bash
ls -lt results/
# Look for directory with your run_name
```

**Check if it's still running**:
```bash
ps aux | grep python3 | grep main_taxa_sweep
```

**Review what completed**:
```bash
cd results/<your-experiment-dir>

# Check log file
tail -100 experiment.log

# Check checkpoint
cat progress_checkpoint.json | jq '.results[] | {p, status}'
```

## Example Output During Run

### What You'll See on stdout (debug mode):

```
================================================================================
SPECTRAL TREE INFERENCE - Sub-sampled STDR Experiment
================================================================================
Experiment Type:    Grid Search
Configurations:     n=[8192], L=[500, 1000]
Mutation Rate:      μ = 0.1
P-values:           25 values from 1.00e-04 to 1.00e+00
Bootstrap Reps:     10
Display Mode:       debug
Run Name:           8192_500_1000_mu_01_parallel
================================================================================

EXPERIMENT_RUNNER | INFO | Log file created: /path/to/results/.../experiment.log
EXPERIMENT_RUNNER | INFO | All progress will be logged to this file
BOOTSTRAP_SWEEP | INFO | n=8192, L=500 preparing experiment data…
BOOTSTRAP_SWEEP | INFO | Generating tree and sequences...
BOOTSTRAP_SWEEP | INFO | Computing full similarity matrix...
BOOTSTRAP_SWEEP | INFO | Computing reference Fiedler vector...
BOOTSTRAP_SWEEP | INFO | Using middle-out parallel processing with 8 workers
MIDDLE_OUT | INFO | Starting middle-out processing: 25 p-values, middle_idx=12
MIDDLE_OUT | INFO | Progress checkpoints will be saved to: .../progress_checkpoint.json

MIDDLE_OUT | INFO | Phase 1: Computing middle seed p[12]=0.01
MIDDLE_OUT | INFO |   This will take ~60s for 10 bootstrap reps...
MIDDLE_OUT | INFO | Middle result (took 62.3s): partition_agreement_M=85.2%
MIDDLE_OUT | INFO |   💾 Checkpoint saved: .../progress_checkpoint.json

MIDDLE_OUT | INFO |
Round 1: offset=1
MIDDLE_OUT | INFO |   High side: indices [13, 14, 15, 16]
MIDDLE_OUT | INFO |   Low side: indices [11, 10, 9, 8]
MIDDLE_OUT | INFO |   Processing 8 p-values in parallel...
MIDDLE_OUT | INFO |   Estimated time: ~60s per p-value (with 8 workers in parallel)
MIDDLE_OUT | INFO |   [Workers are running silently - check back in ~60s]

... ~60s wait ...

MIDDLE_OUT | INFO |   ✓ Completed in 63.2s
MIDDLE_OUT | INFO |     p[13]=0.0139: partition_agreement_M=92.5%
MIDDLE_OUT | INFO |     p[14]=0.0165: partition_agreement_M=96.8%
...
MIDDLE_OUT | INFO |   💾 Checkpoint saved: .../progress_checkpoint.json

MIDDLE_OUT | INFO |
Round 2: offset=5
...

MIDDLE_OUT | INFO |
================================================================================
MIDDLE_OUT | INFO | Middle-out processing complete!
MIDDLE_OUT | INFO |   Total time: 5.2 minutes (312.1s)
MIDDLE_OUT | INFO |   Computed: 15/25 p-values
MIDDLE_OUT | INFO |   Skipped (guardrails): 10 p-values
MIDDLE_OUT | INFO |   Speedup vs sequential: ~5.3x
MIDDLE_OUT | INFO |
================================================================================
```

## Time Estimates

For your configuration (n=8192, 10 bootstrap reps, 8 workers):

- **Single p-value** (sequential): ~60s
- **Round with 8 p-values** (parallel): ~60s (8x speedup!)
- **Full experiment** (25 p-values, with guardrails): ~5-7 minutes
- **Without middle-out** (sequential, all 25): ~25 minutes

## Troubleshooting

### Experiment seems stuck?
Check the log file - it shows estimated wait times:
```
[Workers are running silently - check back in ~60s]
```

This is normal! Workers are computing in parallel processes and can't send live updates.

### VM crashed?
1. SSH back in
2. Check `experiment.log` - see what completed
3. Check `progress_checkpoint.json` - see exact progress
4. Restart from scratch (no automatic resume yet, but you know what completed)

### Want to reduce logging verbosity?
Change `display_mode="progress"` in your config (instead of "debug")
- stdout will be quieter (only progress bars)
- `experiment.log` will still have everything

## Files Summary

For each experiment run in `results/<timestamp>-<run_name>/`:

| File | Purpose | Updated |
|------|---------|---------|
| `experiment.log` | Complete log with timestamps | Real-time |
| `progress_checkpoint.json` | Crash recovery data | Every round (~60s) |
| `results_grid.json` | Final numerical results | On completion |
| `config.json` | Experiment configuration | At start |
| `fiedler_ref_*.npy` | Reference vectors | On completion |
| `plot_*.png` | Visualizations | On completion |

## Best Practices

1. **Always run with `display_mode="debug"`** for cloud VMs - you need the logs!

2. **Use `nohup` or `tmux`** for long runs:
   ```bash
   # Option 1: nohup
   nohup python3 main_taxa_sweep.py > output.txt 2>&1 &

   # Option 2: tmux (recommended)
   tmux new -s experiment
   python3 main_taxa_sweep.py
   # Ctrl+B, then D to detach
   # Later: tmux attach -t experiment
   ```

3. **Monitor from another terminal**:
   ```bash
   watch -n 10 'tail -20 results/*/experiment.log'
   ```

4. **Save your run directory path**:
   ```bash
   python3 main_taxa_sweep.py | tee run_output.txt
   # Outputs the results directory at the end
   ```
