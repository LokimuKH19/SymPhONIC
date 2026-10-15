# ReactionDiffusion / HighNonlinear_Steady1D / Hybrid - 10K rerun

- Profile: high_nonlinear
- Epochs: 1200
- Device: CUDA
- Training mode: Data and PDE losses are optimized jointly.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_FNO** with test relative L2 **1.2451**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10127 | 980 | 1.9318 | 53.2235 | 0.9928 | 3.2757 | 7.9010 | 20.7321 | 0.409 |
| CFNO | 10170 | 1188 | 3.9236 | 340.8360 | 2.0932 | 7.2025 | 18.0513 | 167.5160 | 0.980 |
| HF-FNO | 10125 | 1140 | 1.2451 | 2.3326 | 0.8635 | 2.3961 | 4.6052 | 19.0339 | 0.402 |
| HF-CFNO | 10169 | 1183 | 1.2584 | 2.6418 | 0.9416 | 2.4048 | 4.2626 | 18.1740 | 0.234 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
