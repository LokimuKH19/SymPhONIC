# Burgers / HighNonlinear_Steady1D / Hybrid - 10K rerun

- Profile: high_nonlinear
- Epochs: 1200
- Device: CUDA
- Training mode: Data and PDE losses are optimized jointly.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_CFNO** with test relative L2 **1.3768**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10127 | 1195 | 1.6353 | 2.5160 | 1.0667 | 2.9590 | 43.3743 | 280.7962 | 0.564 |
| CFNO | 10170 | 1199 | 3.0354 | 27.7285 | 1.6620 | 6.5671 | 111.9084 | 2867.6128 | 1.310 |
| HF-FNO | 10125 | 1200 | 1.3818 | 2.1101 | 0.9775 | 2.4024 | 41.0589 | 113.3142 | 0.519 |
| HF-CFNO | 10169 | 1200 | 1.3768 | 1.4130 | 0.8918 | 2.5576 | 33.1784 | 169.6741 | 0.552 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
