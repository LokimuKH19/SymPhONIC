# Spectral Kernel Attention Comparison

All variants share the same CFNO backbone and contain exactly one attention layer.
Data-free selection uses validation PDE residual; data-driven selection uses validation data MSE; hybrid selection uses their weighted sum.

| PDE | Case | Mode | Winner | Test relative L2 | Test PDE MSE |
|---|---|---|---|---:|---:|
| AllenCahn | HighNonlinear_Transient1D | DataDriven | F_ATTN | 1.781803e-01 | 9.796436e-02 |
| AllenCahn | HighNonlinear_Transient1D | Hybrid | F_ATTN | 1.656565e-01 | 3.095548e-02 |
| AllenCahn | HighNonlinear_Transient1D | DataFree | F_ATTN | 1.981678e-01 | 1.783409e-02 |
| Burgers | HighNonlinear_Transient1D | DataDriven | HF_ATTN | 1.516536e-01 | 3.642068e-02 |
| Burgers | HighNonlinear_Transient1D | Hybrid | HF_ATTN | 1.526203e-01 | 2.826963e-02 |
| Burgers | HighNonlinear_Transient1D | DataFree | F_ATTN | 1.499142e-01 | 1.804504e-02 |
| KdV | HighNonlinear_Transient1D | DataDriven | F_ATTN | 3.172281e-01 | 3.291241e-01 |
| KdV | HighNonlinear_Transient1D | Hybrid | F_ATTN | 2.975934e-01 | 1.756955e-01 |
| KdV | HighNonlinear_Transient1D | DataFree | HF_ATTN | 4.247671e-01 | 2.341357e-01 |
| ReactionDiffusion | HighNonlinear_Transient1D | DataDriven | F_ATTN | 8.485056e-02 | 4.266070e-02 |
| ReactionDiffusion | HighNonlinear_Transient1D | Hybrid | C_ATTN | 8.199708e-02 | 1.318687e-02 |
| ReactionDiffusion | HighNonlinear_Transient1D | DataFree | C_ATTN | 8.212914e-02 | 9.165440e-03 |
