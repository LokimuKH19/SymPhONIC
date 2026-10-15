# AllenCahn / Steady1D / Hybrid - 10K rerun

- Profile: baseline
- Epochs: 1200
- Device: CUDA
- Training mode: Data and PDE losses are optimized jointly.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_FNO** with test relative L2 **1.0021**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10007 | 552 | 1.0881 | 0.4807 | 0.5950 | 1.2292 | 0.7900 | 3.1711 | 0.954 |
| CFNO | 10029 | 936 | 1.1095 | 0.3825 | 0.5477 | 2.0778 | 0.5078 | 2.0395 | 0.270 |
| HF-FNO | 10003 | 513 | 1.0021 | 0.2324 | 0.5349 | 1.3319 | 0.4782 | 2.0474 | 0.167 |
| HF-CFNO | 10029 | 803 | 1.1968 | 0.2981 | 0.5708 | 2.3854 | 0.6767 | 10.0273 | 0.812 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
