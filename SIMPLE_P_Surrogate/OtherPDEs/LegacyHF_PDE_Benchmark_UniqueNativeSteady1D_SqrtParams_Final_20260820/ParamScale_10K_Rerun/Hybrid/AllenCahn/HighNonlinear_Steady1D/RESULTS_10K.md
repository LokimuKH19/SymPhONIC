# AllenCahn / HighNonlinear_Steady1D / Hybrid - 10K rerun

- Profile: high_nonlinear
- Epochs: 1200
- Device: CUDA
- Training mode: Data and PDE losses are optimized jointly.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_CFNO** with test relative L2 **1.2599**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10127 | 1199 | 1.9874 | 180.9067 | 1.3489 | 4.2716 | 18.4876 | 161.1476 | 0.448 |
| CFNO | 10170 | 881 | 3.8633 | 2220.2319 | 1.8573 | 6.1721 | 39.9152 | 433.6477 | 1.226 |
| HF-FNO | 10125 | 539 | 1.2991 | 1.9632 | 0.7890 | 2.0612 | 1.2114 | 6.3160 | 0.317 |
| HF-CFNO | 10169 | 659 | 1.2599 | 0.7343 | 0.8003 | 1.9379 | 1.0976 | 6.8627 | 0.281 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
