# Burgers / HighNonlinear_Steady1D / PhysicsOnly - 10K rerun

- Profile: high_nonlinear
- Epochs: 1200
- Device: CUDA
- Training mode: Pure-physics optimization; field truth is used only for evaluation.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_FNO** with test relative L2 **1.7075**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10127 | 951 | 1.8640 | 1.8780 | 0.9018 | 2.6478 | 31.5386 | 133.7514 | 0.576 |
| CFNO | 10170 | 775 | 2.6669 | 2.9909 | 1.2612 | 3.2776 | 33.7174 | 147.1213 | 0.947 |
| HF-FNO | 10125 | 1197 | 1.7075 | 0.9965 | 1.0085 | 4.1525 | 63.6226 | 3902.7732 | 0.549 |
| HF-CFNO | 10169 | 547 | 1.9529 | 1.5937 | 0.8761 | 2.7024 | 35.3828 | 1302.3004 | 0.715 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
