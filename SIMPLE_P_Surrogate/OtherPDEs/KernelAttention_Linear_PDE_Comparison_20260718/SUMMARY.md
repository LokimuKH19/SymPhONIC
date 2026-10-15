# Spectral Kernel Attention Comparison

All variants share the same CFNO backbone and contain exactly one attention layer.
Data-free selection uses validation PDE residual; data-driven selection uses validation data MSE; hybrid selection uses their weighted sum.

| PDE | Case | Mode | Winner | Test relative L2 | Test PDE MSE |
|---|---|---|---|---:|---:|
| Poisson | Steady1D | DataDriven | HF_ATTN | 3.756001e-01 | 5.379836e-01 |
| Poisson | Steady1D | Hybrid | C_ATTN | 4.086040e-01 | 7.156375e-02 |
| Poisson | Steady1D | DataFree | HF_ATTN | 5.040471e-01 | 3.733088e-03 |
| Poisson | Steady2D | DataDriven | C_ATTN | 1.732497e-02 | 1.134302e-02 |
| Poisson | Steady2D | Hybrid | C_ATTN | 1.029768e-02 | 5.858351e-04 |
| Poisson | Steady2D | DataFree | C_ATTN | 1.155639e-02 | 2.897975e-04 |
| Wave | Transient1D | DataDriven | HF_ATTN | 2.809666e-01 | 1.240323e+04 |
| Wave | Transient1D | Hybrid | HF_ATTN | 7.560100e-02 | 1.267594e+01 |
| Wave | Transient1D | DataFree | HF_ATTN | 8.711208e-02 | 1.327955e+01 |
