# Burgers / Steady1D / DataOnly - 10K rerun

- Profile: baseline
- Epochs: 1200
- Device: CUDA
- Training mode: Data-only optimization; the PDE residual is diagnostic only.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_CFNO** with test relative L2 **1.1286**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10007 | 615 | 1.4651 | 34.3936 | 0.3379 | 0.7892 | 5.2004 | 14.6259 | 1.050 |
| CFNO | 10029 | 1014 | 1.5771 | 5.0625 | 0.5353 | 1.4377 | 4.5832 | 21.0466 | 0.414 |
| HF-FNO | 10003 | 670 | 1.1621 | 4.1743 | 0.4465 | 1.0956 | 3.6031 | 15.3180 | 0.210 |
| HF-CFNO | 10029 | 750 | 1.1286 | 2.9981 | 0.5575 | 1.1258 | 4.1255 | 12.1679 | 0.776 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
