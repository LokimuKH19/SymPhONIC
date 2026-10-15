# KdV / Steady1D / Hybrid - 10K rerun

- Profile: baseline
- Epochs: 1200
- Device: CUDA
- Training mode: Data and PDE losses are optimized jointly.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_FNO** with test relative L2 **0.9039**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10005 | 1195 | 1.3403 | 3.2534 | 0.6621 | 1.6670 | 14.6993 | 82.0832 | 0.498 |
| CFNO | 10027 | 1177 | 1.5667 | 5.6698 | 0.6326 | 1.3672 | 10.0760 | 30.1999 | 0.717 |
| HF-FNO | 10001 | 1175 | 0.9039 | 4.8587 | 0.5971 | 1.3952 | 9.4797 | 36.7923 | 0.246 |
| HF-CFNO | 10027 | 1177 | 1.6258 | 8.5766 | 0.7963 | 1.9665 | 16.2701 | 96.5587 | 0.478 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
