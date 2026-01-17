# Spectral Analysis Framework

Characterizes the **Spectral Concentration Phase Transition** in subsampled Laplacians.

## Three Operating Regimes

1. **noise_bulk**: Signal buried in noise (partition_agreement_M ≤ 55%)
   - λ₂(L_S) within Marchenko-Pastur noise bulk
   - Random partitions, high IPR (localized eigenvector)

2. **spectral_emergence**: BBP transition (55% < agreement < 95%)
   - λ₂(L_S) "pops" out of noise
   - Partition agreement: 50% → 100%
   - IPR drops (delocalization)

3. **perturbation_plateau**: Davis-Kahan stability (agreement ≥ 95%)
   - L_S is valid spectral proxy for L_M
   - DK ratio < 0.5
   - Perfect partition agreement

## Usage

```bash
# From sub_sampled_fielder_vec/ directory
python -m spectral_analysis.sweep_params_analysis.run_analysis \
  results/20251130-193600-balanced_tree_mu_01
```

## Input Requirements

The `results_grid_merged.json` file must contain:
- `num_taxa`, `sequence_length`, `p`
- `partition_agreement_M`, `std`
- `lambda_2_S`, `lambda_3_S`, `lambda_2_M`, `lambda_3_M`
- `mean_dk_ratio_S`, `mean_ipr_S`

## Outputs

Analysis results are saved to `results/{run_name}/analysis_{timestamp}/`:

### Files
- `regime_classification.csv` - Per-configuration regime labels
- `scaling_law_fit.txt` - Regression: p_crit ~ f(N, L)

### Plots (5 files total, covering all 20 configurations)
- **Graph A**: `A_phase_boundary.png` - Combined scatter plot (p vs L, color by N, size by agreement)
- **Graph B**: `B_eigenvalue_pop.png` - λ₂(p), λ₃(p) trajectories in 5×4 grid
  - Shaded gray "noise ocean" (0 to λ₃)
  - Bold red λ₂ line (signal emerging)
  - Vertical markers: blue (λ₂ > λ₃), green (p_crit)
- **Graph C**: `C_stability_curve.png` - Davis-Kahan ratio vs p in 5×4 grid
- **Graph D**: `D_ipr_delocalization.png` - IPR delocalization curve in 5×4 grid
  - Red dashed baseline at 1/N (perfect delocalization)
  - X-axis synchronized with Graphs B & C
- **Graph E**: `E_scaling_law.png` - p_crit vs L validation
  - One curve per N value
  - Theory overlays: p ∝ L^(-0.5), L^(-1), fitted exponent

## Notation

- **Subscripts**: `_S` = subsampled, `_M` = full matrix
- **Variables**: p (sampling rate), N (num_taxa), L (sequence_length)
- **Eigenvalues**: λ₂ (Fiedler), λ₃ (third eigenvalue)
