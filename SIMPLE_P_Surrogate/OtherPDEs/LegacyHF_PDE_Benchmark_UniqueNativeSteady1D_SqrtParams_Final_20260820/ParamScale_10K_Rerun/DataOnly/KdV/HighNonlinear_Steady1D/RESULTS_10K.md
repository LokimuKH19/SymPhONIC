# KdV / HighNonlinear_Steady1D / DataOnly - 10K rerun

- Profile: high_nonlinear
- Epochs: 1200
- Device: CUDA
- Training mode: Data-only optimization; the PDE residual is diagnostic only.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_FNO** with test relative L2 **1.3761**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 9999 | 1200 | 3.1150 | 15.3073 | 1.9581 | 4.8081 | 298.5325 | 1905.3904 | 0.462 |
| CFNO | 10168 | 1183 | 3.3887 | 15.3005 | 2.7830 | 4.8309 | 370.4476 | 3114.3164 | 0.709 |
| HF-FNO | 9991 | 1037 | 1.3761 | 16.6069 | 0.9704 | 3.7079 | 329.8500 | 1613.5923 | 0.300 |
| HF-CFNO | 10167 | 1174 | 1.9188 | 14.5646 | 1.5016 | 3.9670 | 307.7734 | 1583.4099 | 0.401 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
