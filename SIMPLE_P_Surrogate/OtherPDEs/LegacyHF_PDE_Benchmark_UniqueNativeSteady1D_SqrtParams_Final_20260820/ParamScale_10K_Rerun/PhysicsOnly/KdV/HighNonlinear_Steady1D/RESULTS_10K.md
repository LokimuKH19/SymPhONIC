# KdV / HighNonlinear_Steady1D / PhysicsOnly - 10K rerun

- Profile: high_nonlinear
- Epochs: 1200
- Device: CUDA
- Training mode: Pure-physics optimization; field truth is used only for evaluation.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_FNO** with test relative L2 **2.6740**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 9999 | 1188 | 5.2297 | 2.1430 | 3.7568 | 5.6098 | 151.5239 | 675.8062 | 0.539 |
| CFNO | 10168 | 880 | 5.9302 | 3.7824 | 4.6459 | 7.5035 | 204.8117 | 910.5391 | 0.786 |
| HF-FNO | 9991 | 1193 | 2.6740 | 1.3207 | 1.5038 | 4.0671 | 226.2435 | 5530.2485 | 0.489 |
| HF-CFNO | 10167 | 1063 | 3.2128 | 1.8838 | 2.2274 | 4.0935 | 167.0291 | 1441.6179 | 0.610 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
