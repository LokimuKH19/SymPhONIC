# ReactionDiffusion / Steady1D / Hybrid - 10K rerun

- Profile: baseline
- Epochs: 1200
- Device: CUDA
- Training mode: Data and PDE losses are optimized jointly.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **CFNO** with test relative L2 **0.8506**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10007 | 498 | 0.9658 | 2.2332 | 0.2118 | 0.6224 | 0.9540 | 6.8780 | 0.880 |
| CFNO | 10029 | 710 | 0.8506 | 0.8971 | 0.2015 | 0.8041 | 0.6432 | 3.7068 | 0.216 |
| HF-FNO | 10003 | 651 | 0.8809 | 1.7090 | 0.2789 | 0.7694 | 0.6361 | 1.8073 | 0.147 |
| HF-CFNO | 10029 | 801 | 0.9486 | 1.0545 | 0.4449 | 1.0755 | 0.9760 | 3.1947 | 0.728 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
