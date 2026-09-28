# Native Steady 1D Benchmark Summary

## Scope

- Dedicated program: `run_legacy_hf_pde_steady1d_suite.py`
- Native tensor contract: `[B, C, N]`; no replicated auxiliary spatial dimension.
- PDE cases per training mode: 9 (five baseline and four high-nonlinearity cases).
- Models per case: FNO, CFNO, HF-FNO, and HF-CFNO.
- Training modes: PhysicsOnly, Hybrid, and DataOnly.
- Every model was trained for 1200 epochs on CUDA.
- Training/validation/test sample counts: 32/8/4.
- Baseline and high-nonlinearity grids: 64 and 96 points.
- The nominal native-1D budget is `round(sqrt(500000)) = 707` real parameters.
- Actual counts are 705--719. For every matched pair, FNO is not smaller than HF-FNO and CFNO is not smaller than HF-CFNO.

## Hard Constraints and Well-Posedness

Poisson remains a two-point Dirichlet problem. Its output is projected as a linear endpoint lift plus `4 xi (1-xi) N_theta`, so both endpoint values are exact. The corresponding linear problem is unique.

The original nonlinear steady boundary-value formulations must not be treated as globally unique without an additional theorem or parameter restriction. This is especially important for steady Burgers: a source field alone does not determine the solution, and even a nonlinear two-point boundary-value problem can have multiple branches.

For this benchmark, Burgers, Allen-Cahn, and reaction-diffusion were therefore defined as spatial Cauchy problems. Their signed left value and signed left slope are input channels, and the hard form is

`u_theta(xi) = u(0) + L u_x(0) xi + xi^2 N_theta(xi)`.

For Burgers, `u' = q` and `q' = (a u q - f)/nu` form a smooth first-order initial-value system for `nu > 0`. The signed initial jet selects one local branch uniquely, and the manufactured global solution guarantees that this branch exists over the benchmark interval. KdV analogously receives its signed left value, slope, and curvature and uses an `xi^3` correction.

## Aggregate Results

The table reports the median test relative L2 error over the nine cases in each mode. These errors show that the very small square-root parameter budget is challenging; the benchmark does not support a claim that every HFB model is uniformly superior.

| Mode | FNO | CFNO | HF-FNO | HF-CFNO |
|---|---:|---:|---:|---:|
| PhysicsOnly | 3.24 | 3.79 | 5.47 | 2.73 |
| Hybrid | 2.69 | 3.87 | 4.10 | 2.49 |
| DataOnly | 2.79 | 2.86 | 2.92 | 2.28 |

Paired test-relative-L2 comparisons:

| Mode | HF-FNO better than FNO | HF-CFNO better than CFNO |
|---|---:|---:|
| PhysicsOnly | 4/9 | 6/9 |
| Hybrid | 4/9 | 5/9 |
| DataOnly | 5/9 | 6/9 |

HF-CFNO is the more consistent enhanced branch in these runs. Its geometric-mean relative-L2 ratio against CFNO is 0.66, 0.76, and 0.67 for PhysicsOnly, Hybrid, and DataOnly, respectively. HF-FNO is inconsistent: its corresponding ratios against FNO are 1.61, 1.59, and 1.47. The residual ranking does not always agree with the field-error ranking, so both plots and metrics should be inspected for each case.

## Figure Inventory

The output hierarchy mirrors the figure types found in the three reference roots. It includes, as applicable:

- `prediction_exact_error.png` and `prediction_exact_error_line.png`
- `residual_distribution.png`
- `spectrum_analysis.png`
- `frequency_component_abs_error.png`
- normalized and unnormalized training convergence figures
- separate objective, data, and physics convergence figures
- case-level four-model convergence, spectral-error, and relative-L2 comparisons
- PDE-level `frequency_component_error_comparison.png`
- training-mode-level `global_test_rel_l2_heatmap.png`

The machine-readable `plot_manifest.json` verifies 1332 required reference-aligned files with zero missing files. Additional diagnostic variants bring the total to 2296 PNG files.

## Reproducibility Files

- `benchmark_config.json`: full setup, hard-constraint definitions, and parameter policy.
- `global_comparison.csv` and `global_comparison.json`: all 108 model records.
- `completion_report.json`: training and plotting completion status.
- `plot_manifest.json`: required figure inventory and missing-file check.
- Each model directory contains `best_checkpoint.pt`, `history.csv`, `metrics.json`, `sample_fields.npz`, and its complete figure set.
