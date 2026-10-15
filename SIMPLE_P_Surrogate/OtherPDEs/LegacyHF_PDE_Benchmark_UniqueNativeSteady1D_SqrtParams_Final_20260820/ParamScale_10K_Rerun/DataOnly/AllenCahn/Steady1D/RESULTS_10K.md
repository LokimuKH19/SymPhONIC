# AllenCahn / Steady1D / DataOnly - 10K rerun

- Profile: baseline
- Epochs: 1200
- Device: CUDA
- Training mode: Data-only optimization; the PDE residual is diagnostic only.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_CFNO** with test relative L2 **0.9361**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10007 | 626 | 0.9741 | 1.5479 | 0.3932 | 1.2223 | 0.7620 | 8.3144 | 0.742 |
| CFNO | 10029 | 742 | 1.2043 | 1.5046 | 0.7226 | 3.0636 | 1.0534 | 11.7941 | 0.332 |
| HF-FNO | 10003 | 439 | 1.0530 | 0.2540 | 0.6625 | 1.5102 | 0.5604 | 1.8854 | 0.187 |
| HF-CFNO | 10029 | 959 | 0.9361 | 0.2536 | 0.7984 | 2.0540 | 0.7497 | 6.7371 | 0.704 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
