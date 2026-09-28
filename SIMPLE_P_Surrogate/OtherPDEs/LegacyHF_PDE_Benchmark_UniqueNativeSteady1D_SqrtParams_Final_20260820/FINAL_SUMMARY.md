# Native Steady-1D 10K-Parameter Rerun

## Scope and completion

- Cases: Poisson/Steady1D; baseline and high-nonlinear KdV, Allen-Cahn, Burgers, and Reaction-Diffusion.
- Training modes: PhysicsOnly, Hybrid, and DataOnly.
- Models: FNO, CFNO, HF-FNO, and HF-CFNO.
- Training: 1200 epochs on CUDA for every model, with mode-specific validation checkpoint selection.
- Data: 32 training, 8 validation, and 4 test functions. Baseline cases use 64 points; high-nonlinear cases use 96 points.
- Parameter budget: 9,991--10,170 real parameters depending on case and model. The non-HF member is never smaller than its matched HF member.
- Tensor contract: native `[B,C,N]`; no replicated auxiliary spatial dimension.
- Constraints: signed Cauchy/Dirichlet data are passed explicitly and imposed by hard projections. No boundary loss is used.

All 108 runs completed. There are 108 checkpoints, histories, metric files, and field archives. The plotting inventory verifies all 1,332 required files with zero missing; 2,296 PNG files are present including additional diagnostics.

## Parameter-scale comparison

The 10K models improve test relative L2 over the matched approximately 700-parameter models in 96 of 108 runs.

| Training mode | Improved runs | Total | Median 10K/700 error ratio |
|---|---:|---:|---:|
| PhysicsOnly | 30 | 36 | 0.63 |
| Hybrid | 33 | 36 | 0.50 |
| DataOnly | 33 | 36 | 0.49 |

Capacity is useful, but its effect depends strongly on the training objective. The median test relative L2 is 1.86 for PhysicsOnly, 1.29 for Hybrid, and 1.26 for DataOnly; the corresponding median normalized PDE MSE values are 1.59, 2.26, and 5.04.

## HFB ablation

The table counts paired field-error wins of HF-FNO over FNO and HF-CFNO over CFNO.

| Training mode | Baseline wins | High-nonlinear wins |
|---|---:|---:|
| PhysicsOnly | 4/10 | 6/8 |
| Hybrid | 6/10 | 8/8 |
| DataOnly | 9/10 | 8/8 |

HFB is most consistent when data anchor the intended branch. In every high-nonlinear Hybrid and DataOnly pair, the HFB model has lower field error than its conventional counterpart. The advantage is less systematic under pure physics optimization, so HFB should be interpreted as a response to sparse-data spectral bias rather than a general repair of PINO training dynamics.

## Interpretation

PhysicsOnly models can reduce the discretized PDE objective while retaining large field errors or selecting an undesirable optimization basin. Increasing capacity and enforcing complete signed hard constraints do not remove this behavior.

DataOnly training controls the sampled branch much better, especially with HFB, but produces the largest median PDE residual. This is consistent with insufficient derivative smoothness between supervised samples.

Hybrid training gives the clearest compromise: data constrain branch selection while the PDE term improves local consistency. The high-nonlinear HFB pairs are uniformly better in field error, although their residuals and absolute-error distributions must still be checked rather than inferred from relative L2 alone.

## Files

- `global_comparison.csv`: all 108 10K-model records, including all-point mean/maximum field errors and PDE residuals.
- `comparison_vs_707params.csv`: all 108 matched 10K-versus-approximately-700 comparisons.
- `plot_manifest.json`: required-figure inventory; 1,332/1,332 present.
- `completion_report.json`: training and post-processing completion status.
- Every training-mode/case directory contains `RESULTS_10K.md` and all model/result/convergence/spectrum figures.
