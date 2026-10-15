# ReactionDiffusion / HighNonlinear_Steady1D / PhysicsOnly - 10K rerun

- Profile: high_nonlinear
- Epochs: 1200
- Device: CUDA
- Training mode: Pure-physics optimization; field truth is used only for evaluation.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_CFNO** with test relative L2 **2.5969**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10127 | 661 | 3.0998 | 497.2843 | 1.2341 | 3.7000 | 9.8941 | 37.6241 | 0.632 |
| CFNO | 10170 | 1193 | 5.0175 | 1130.3370 | 2.9434 | 9.9086 | 44.5574 | 339.7480 | 1.073 |
| HF-FNO | 10125 | 1198 | 3.5591 | 7.0618 | 1.6021 | 19.4980 | 18.6551 | 979.6062 | 1.113 |
| HF-CFNO | 10169 | 1140 | 2.5969 | 5.1824 | 1.4740 | 9.5629 | 8.5255 | 275.8940 | 0.479 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
