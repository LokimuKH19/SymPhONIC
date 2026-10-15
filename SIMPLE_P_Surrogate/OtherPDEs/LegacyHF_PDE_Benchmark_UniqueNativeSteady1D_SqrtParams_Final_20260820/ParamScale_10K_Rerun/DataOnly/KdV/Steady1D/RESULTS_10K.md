# KdV / Steady1D / DataOnly - 10K rerun

- Profile: baseline
- Epochs: 1200
- Device: CUDA
- Training mode: Data-only optimization; the PDE residual is diagnostic only.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_CFNO** with test relative L2 **1.1134**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10005 | 1193 | 1.2557 | 58.1704 | 0.7323 | 1.4746 | 50.6278 | 284.1289 | 0.450 |
| CFNO | 10027 | 1154 | 1.5218 | 56.2569 | 0.6133 | 2.0166 | 37.9381 | 199.1054 | 0.638 |
| HF-FNO | 10001 | 736 | 1.1339 | 225.1940 | 0.6260 | 1.4069 | 46.2658 | 184.7197 | 0.415 |
| HF-CFNO | 10027 | 1051 | 1.1134 | 45.5595 | 0.6321 | 1.5741 | 32.8909 | 190.5428 | 0.487 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
