# Burgers / HighNonlinear_Steady1D / DataOnly - 10K rerun

- Profile: high_nonlinear
- Epochs: 1200
- Device: CUDA
- Training mode: Data-only optimization; the PDE residual is diagnostic only.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **HF_FNO** with test relative L2 **1.3657**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10127 | 1195 | 1.6002 | 3.3161 | 1.1440 | 2.9664 | 37.6369 | 263.0903 | 0.531 |
| CFNO | 10170 | 1189 | 3.6162 | 77.6332 | 1.8978 | 5.9416 | 125.2582 | 3148.0999 | 1.452 |
| HF-FNO | 10125 | 1197 | 1.3657 | 3.1686 | 0.8526 | 2.3861 | 38.9497 | 219.2089 | 0.468 |
| HF-CFNO | 10169 | 1175 | 1.3966 | 2.2105 | 0.9060 | 2.4406 | 35.8159 | 133.8771 | 0.591 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
