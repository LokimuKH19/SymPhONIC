# Burgers / Steady1D / PhysicsOnly - 10K rerun

- Profile: baseline
- Epochs: 1200
- Device: CUDA
- Training mode: Pure-physics optimization; field truth is used only for evaluation.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **FNO** with test relative L2 **1.0910**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10007 | 503 | 1.0910 | 1.4224 | 0.5084 | 1.7165 | 1.7450 | 4.8509 | 0.777 |
| CFNO | 10029 | 360 | 1.2455 | 1.1912 | 0.5126 | 1.6704 | 2.2879 | 9.5828 | 0.276 |
| HF-FNO | 10003 | 493 | 1.1938 | 0.6505 | 0.3829 | 1.4045 | 4.4967 | 110.6812 | 0.195 |
| HF-CFNO | 10029 | 253 | 1.1232 | 1.0452 | 0.4907 | 1.8536 | 2.7019 | 11.4721 | 0.627 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
