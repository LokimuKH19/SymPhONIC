# Poisson / Steady1D / Hybrid - 10K rerun

- Profile: baseline
- Epochs: 1200
- Device: CUDA
- Training mode: Data and PDE losses are optimized jointly.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_CFNO** with test relative L2 **0.1604**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10007 | 1200 | 0.2142 | 0.3377 | 0.0472 | 0.1013 | 38.5687 | 309.9762 | 0.562 |
| CFNO | 10029 | 1200 | 0.2688 | 0.1996 | 0.0490 | 0.1358 | 17.9153 | 49.8910 | 0.571 |
| HF-FNO | 10003 | 1183 | 0.1839 | 0.1571 | 0.0614 | 0.2697 | 54.3314 | 512.7230 | 0.441 |
| HF-CFNO | 10029 | 1195 | 0.1604 | 0.1697 | 0.0976 | 0.2427 | 33.7895 | 182.1835 | 0.569 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
