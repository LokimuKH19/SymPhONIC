# KdV / Steady1D / PhysicsOnly - 10K rerun

- Profile: baseline
- Epochs: 1200
- Device: CUDA
- Training mode: Pure-physics optimization; field truth is used only for evaluation.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **CFNO** with test relative L2 **2.4086**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10005 | 1197 | 2.7182 | 1.8329 | 1.7633 | 3.3230 | 7.2693 | 25.6932 | 0.814 |
| CFNO | 10027 | 1195 | 2.4086 | 1.2866 | 1.5826 | 3.3573 | 6.6191 | 28.9966 | 0.711 |
| HF-FNO | 10001 | 749 | 2.7901 | 1.1558 | 1.1167 | 2.7522 | 45.1354 | 1116.3927 | 0.483 |
| HF-CFNO | 10027 | 709 | 2.7988 | 1.7690 | 1.4825 | 3.0702 | 18.3591 | 255.2596 | 0.595 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
