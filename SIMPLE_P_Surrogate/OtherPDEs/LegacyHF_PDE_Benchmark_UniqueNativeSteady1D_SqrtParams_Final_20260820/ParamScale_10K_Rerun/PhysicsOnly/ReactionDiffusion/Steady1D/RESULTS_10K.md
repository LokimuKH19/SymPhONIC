# ReactionDiffusion / Steady1D / PhysicsOnly - 10K rerun

- Profile: baseline
- Epochs: 1200
- Device: CUDA
- Training mode: Pure-physics optimization; field truth is used only for evaluation.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_FNO** with test relative L2 **1.0143**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10007 | 783 | 1.0294 | 0.7955 | 0.3988 | 0.9131 | 0.7755 | 3.4951 | 0.605 |
| CFNO | 10029 | 1140 | 1.6939 | 1.0724 | 0.6888 | 1.2588 | 1.0336 | 4.7897 | 0.446 |
| HF-FNO | 10003 | 461 | 1.0143 | 0.8502 | 0.6405 | 1.2358 | 1.6347 | 30.5800 | 0.160 |
| HF-CFNO | 10029 | 574 | 1.6825 | 0.9696 | 0.8927 | 1.7696 | 1.3578 | 12.5355 | 0.794 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
