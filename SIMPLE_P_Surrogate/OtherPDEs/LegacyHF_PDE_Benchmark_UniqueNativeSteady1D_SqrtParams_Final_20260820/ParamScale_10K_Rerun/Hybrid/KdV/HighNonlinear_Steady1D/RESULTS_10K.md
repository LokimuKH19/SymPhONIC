# KdV / HighNonlinear_Steady1D / Hybrid - 10K rerun

- Profile: high_nonlinear
- Epochs: 1200
- Device: CUDA
- Training mode: Data and PDE losses are optimized jointly.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_FNO** with test relative L2 **1.4465**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 9999 | 1195 | 3.0101 | 27.9483 | 1.7561 | 4.8047 | 414.1827 | 6897.2095 | 0.450 |
| CFNO | 10168 | 1183 | 3.3301 | 5.6209 | 2.7117 | 4.9112 | 286.6967 | 1989.9775 | 0.712 |
| HF-FNO | 9991 | 1156 | 1.4465 | 5.7641 | 0.9626 | 3.7023 | 221.8082 | 1053.0757 | 0.323 |
| HF-CFNO | 10167 | 1156 | 1.9680 | 7.3129 | 1.4447 | 3.8161 | 310.7960 | 2697.6174 | 0.427 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
