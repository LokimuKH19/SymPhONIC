# Burgers / Steady1D / Hybrid - 10K rerun

- Profile: baseline
- Epochs: 1200
- Device: CUDA
- Training mode: Data and PDE losses are optimized jointly.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **CFNO** with test relative L2 **1.1572**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10007 | 410 | 1.4021 | 2.8208 | 0.2652 | 0.5888 | 2.3543 | 7.0751 | 1.019 |
| CFNO | 10029 | 811 | 1.1572 | 2.2600 | 0.5591 | 1.4187 | 4.1275 | 17.9716 | 0.299 |
| HF-FNO | 10003 | 374 | 1.1696 | 1.3851 | 0.5395 | 1.0686 | 3.6818 | 23.2358 | 0.210 |
| HF-CFNO | 10029 | 608 | 1.2922 | 1.6437 | 0.4080 | 1.0359 | 2.7854 | 9.7445 | 0.891 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
