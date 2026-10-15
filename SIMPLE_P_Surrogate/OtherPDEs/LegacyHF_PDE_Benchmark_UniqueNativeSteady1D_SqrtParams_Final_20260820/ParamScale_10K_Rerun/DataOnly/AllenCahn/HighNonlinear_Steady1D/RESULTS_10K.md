# AllenCahn / HighNonlinear_Steady1D / DataOnly - 10K rerun

- Profile: high_nonlinear
- Epochs: 1200
- Device: CUDA
- Training mode: Data-only optimization; the PDE residual is diagnostic only.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_FNO** with test relative L2 **1.1174**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10127 | 1180 | 1.4523 | 28.4877 | 0.9679 | 3.0769 | 2.0675 | 10.4330 | 0.462 |
| CFNO | 10170 | 1200 | 1.8116 | 121.8675 | 1.1227 | 3.2083 | 4.0920 | 19.0424 | 0.792 |
| HF-FNO | 10125 | 813 | 1.1174 | 5.3250 | 0.6190 | 2.3563 | 0.9886 | 6.4939 | 0.431 |
| HF-CFNO | 10169 | 1079 | 1.1499 | 2.2705 | 0.5852 | 2.3053 | 0.9692 | 5.9009 | 0.449 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
