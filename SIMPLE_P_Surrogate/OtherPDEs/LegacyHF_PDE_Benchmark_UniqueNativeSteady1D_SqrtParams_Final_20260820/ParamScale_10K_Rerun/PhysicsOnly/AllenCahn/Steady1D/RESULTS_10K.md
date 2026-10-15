# AllenCahn / Steady1D / PhysicsOnly - 10K rerun

- Profile: baseline
- Epochs: 1200
- Device: CUDA
- Training mode: Pure-physics optimization; field truth is used only for evaluation.
- Boundary/initial data: exact hard projection; no boundary loss.
- Error and residual summaries below use every point in the saved field archive.

Best field result: **CFNO** with test relative L2 **1.3979**.

| Model | Params | Best epoch | Test rel. L2 | PDE MSE | Mean abs. error | Max abs. error | Mean abs. residual | Max abs. residual | 10K/700 rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 10007 | 382 | 1.6997 | 23.0913 | 0.7293 | 1.5106 | 0.8161 | 6.8804 | 1.178 |
| CFNO | 10029 | 1198 | 1.3979 | 1.9681 | 1.1643 | 2.6002 | 0.8309 | 4.8930 | 0.340 |
| HF-FNO | 10003 | 385 | 1.4477 | 0.5822 | 1.0766 | 3.4717 | 1.4181 | 29.4732 | 0.237 |
| HF-CFNO | 10029 | 807 | 1.4893 | 0.7160 | 0.9729 | 2.0797 | 0.6781 | 5.4419 | 0.722 |

The directory also contains individual normalized and absolute loss curves, prediction/reference/error plots, PDE-residual plots, Fourier-error plots, and four-model comparison figures.

See the root PARAMETER_SCALE_10K_SUMMARY.md for the cross-case interpretation.
