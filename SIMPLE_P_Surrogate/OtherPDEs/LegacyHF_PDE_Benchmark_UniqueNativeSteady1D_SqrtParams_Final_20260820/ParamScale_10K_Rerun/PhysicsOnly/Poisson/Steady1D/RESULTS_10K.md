# Poisson / Steady1D / PhysicsOnly - 10K rerun

- Profile: baseline
- Epochs: 1200
- Device: CUDA
- Training mode: Pure-physics optimization; field truth is used only for evaluation.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **CFNO** with test relative L2 **0.5149**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10007 | 1199 | 0.5958 | 0.0160 | 0.3482 | 0.6249 | 13.6925 | 184.1828 | 1.286 |
| CFNO | 10029 | 1200 | 0.5149 | 0.0074 | 0.3542 | 0.6458 | 7.0527 | 18.1392 | 0.219 |
| HF-FNO | 10003 | 1199 | 0.6469 | 0.0069 | 0.3686 | 0.7483 | 27.9965 | 331.7014 | 0.735 |
| HF-CFNO | 10029 | 1195 | 0.5245 | 0.0039 | 0.3513 | 0.6640 | 8.8763 | 51.3791 | 1.598 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
