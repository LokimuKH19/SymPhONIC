# AllenCahn / HighNonlinear_Steady1D / PhysicsOnly - 10K rerun

- Profile: high_nonlinear
- Epochs: 1200
- Device: CUDA
- Training mode: Pure-physics optimization; field truth is used only for evaluation.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_CFNO** with test relative L2 **1.8515**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10127 | 1009 | 2.4981 | 173.6372 | 1.4906 | 4.7269 | 21.9835 | 188.3044 | 0.555 |
| CFNO | 10170 | 747 | 4.0440 | 3080.1433 | 1.9747 | 6.3899 | 41.8497 | 483.0096 | 1.294 |
| HF-FNO | 10125 | 709 | 3.9760 | 12.0196 | 1.4952 | 26.6751 | 436.4745 | 37451.2813 | 0.968 |
| HF-CFNO | 10169 | 850 | 1.8515 | 2.3556 | 1.1782 | 6.7681 | 9.9478 | 579.8315 | 0.410 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
