# PINO Full-Field PDE Residual Statistics

- Root: `D:\Ansys\SymPhONIC\OtherPDEs\LegacyHF_PDE_Benchmark_PINO_20260705`
- `sample_fields.npz` files analyzed: 68
- Cases: 17
- Residual definition: `L(prediction) - source`, recomputed from the saved field arrays.
- Scope: every saved grid point is included; no PDE interior crop is applied.
- Boundary derivatives use the same `np.gradient(..., edge_order=2)` full-field finite-difference evaluator as the plotting/evaluation path.

## Aggregate By Model

| variant | cases | mean(mean abs) | median(mean abs) | mean(RMS) | median(RMS) | mean(max abs) | median(max abs) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FNO | 17 | 1.285591e+01 | 4.235585e-01 | 2.011795e+01 | 5.543757e-01 | 1.273099e+02 | 2.031255e+00 |
| CFNO | 17 | 5.606870e+00 | 3.354808e-01 | 8.194357e+00 | 4.477685e-01 | 5.176861e+01 | 1.101767e+00 |
| HF_FNO | 17 | 9.560923e+00 | 4.082765e-01 | 3.731769e+01 | 1.181902e+00 | 5.342128e+02 | 8.153570e+00 |
| HF_CFNO | 17 | 3.972698e+00 | 2.923015e-01 | 9.073654e+00 | 8.421340e-01 | 9.189547e+01 | 5.133528e+00 |

## Best Variant Per Case

| PDE | case | best mean(abs) | best RMS | best max(abs) |
|---|---|---|---|---|
| AllenCahn | Steady1D | HF_FNO (3.023014e-02) | HF_FNO (3.684393e-02) | CFNO (7.473050e-02) |
| AllenCahn | Steady2D | CFNO (1.389578e-02) | CFNO (1.713193e-02) | CFNO (6.282696e-02) |
| AllenCahn | Transient1D | CFNO (1.782393e-02) | CFNO (2.388858e-02) | CFNO (8.774655e-02) |
| Burgers | Steady1D | CFNO (3.813433e-01) | CFNO (4.787131e-01) | CFNO (1.101767e+00) |
| Burgers | Steady2D | HF_CFNO (1.210319e-01) | HF_CFNO (1.573183e-01) | CFNO (4.267484e-01) |
| Burgers | Transient1D | CFNO (1.251829e-01) | HF_CFNO (1.644830e-01) | CFNO (5.285052e-01) |
| KdV | HighNonlinear_Steady1D | HF_CFNO (1.622024e+01) | HF_CFNO (3.092397e+01) | HF_CFNO (1.816636e+02) |
| KdV | HighNonlinear_Steady2D | HF_CFNO (1.389302e+01) | HF_CFNO (2.395047e+01) | HF_CFNO (2.427189e+02) |
| KdV | Steady1D | CFNO (4.406180e-01) | CFNO (7.580403e-01) | CFNO (2.882848e+00) |
| KdV | Steady2D | CFNO (1.951030e-01) | CFNO (4.294324e-01) | FNO (2.031255e+00) |
| KdV | Transient1D | CFNO (9.655764e+00) | CFNO (1.465418e+01) | HF_CFNO (8.580484e+01) |
| Poisson | Steady1D | CFNO (4.933109e+00) | CFNO (5.855455e+00) | CFNO (2.026549e+01) |
| Poisson | Steady2D | CFNO (5.402830e-01) | CFNO (8.630648e-01) | CFNO (6.421397e+00) |
| ReactionDiffusion | Steady1D | FNO (2.696962e-01) | FNO (3.763974e-01) | HF_FNO (8.044075e-01) |
| ReactionDiffusion | Steady2D | CFNO (5.094037e-02) | CFNO (6.400584e-02) | CFNO (1.958735e-01) |
| ReactionDiffusion | Transient1D | CFNO (3.761552e-02) | CFNO (4.766069e-02) | CFNO (1.814082e-01) |
| Wave | Transient1D | HF_CFNO (1.596788e+01) | CFNO (2.726617e+01) | CFNO (3.040794e+02) |

## Largest Max-Abs Residuals

| PDE | case | variant | mean(abs) | RMS | max(abs) |
|---|---|---|---:|---:|---:|
| Wave | Transient1D | HF_FNO | 5.947108e+01 | 3.558433e+02 | 7.167654e+03 |
| Wave | Transient1D | HF_CFNO | 1.596788e+01 | 5.905707e+01 | 9.161688e+02 |
| Poisson | Steady1D | HF_FNO | 4.254586e+01 | 1.422448e+02 | 8.584333e+02 |
| Wave | Transient1D | FNO | 3.771824e+01 | 6.225003e+01 | 5.851303e+02 |
| KdV | Transient1D | FNO | 5.031498e+01 | 6.694383e+01 | 4.090215e+02 |
| Poisson | Steady1D | FNO | 3.497849e+01 | 7.258337e+01 | 3.937662e+02 |
| KdV | HighNonlinear_Steady2D | HF_FNO | 1.559567e+01 | 3.188171e+01 | 3.693608e+02 |
| Wave | Transient1D | CFNO | 1.921365e+01 | 2.726617e+01 | 3.040794e+02 |
| KdV | HighNonlinear_Steady2D | FNO | 5.199665e+01 | 6.839238e+01 | 2.875519e+02 |
| Poisson | Steady2D | FNO | 1.393245e+01 | 2.928007e+01 | 2.828341e+02 |

## Largest Mean-Abs Residuals

| PDE | case | variant | mean(abs) | RMS | max(abs) |
|---|---|---|---:|---:|---:|
| Wave | Transient1D | HF_FNO | 5.947108e+01 | 3.558433e+02 | 7.167654e+03 |
| KdV | HighNonlinear_Steady2D | FNO | 5.199665e+01 | 6.839238e+01 | 2.875519e+02 |
| KdV | Transient1D | FNO | 5.031498e+01 | 6.694383e+01 | 4.090215e+02 |
| Poisson | Steady1D | HF_FNO | 4.254586e+01 | 1.422448e+02 | 8.584333e+02 |
| KdV | HighNonlinear_Steady2D | CFNO | 4.071880e+01 | 5.514867e+01 | 2.555424e+02 |
| Wave | Transient1D | FNO | 3.771824e+01 | 6.225003e+01 | 5.851303e+02 |
| Poisson | Steady1D | FNO | 3.497849e+01 | 7.258337e+01 | 3.937662e+02 |
| KdV | HighNonlinear_Steady1D | FNO | 2.716292e+01 | 3.910287e+01 | 1.932311e+02 |
| Wave | Transient1D | CFNO | 1.921365e+01 | 2.726617e+01 | 3.040794e+02 |
| KdV | HighNonlinear_Steady1D | CFNO | 1.848323e+01 | 3.287450e+01 | 1.892462e+02 |

## Output Files

- `full_field_pde_residual_statistics.csv`: one row per `PDE/case/model`.
- `full_field_pde_residual_case_ranking.csv`: per-case model ranking by mean(abs), RMS and max(abs).
- `full_field_pde_residual_model_summary.csv`: aggregate statistics by model family.
- `full_field_pde_residual_case_winners.csv`: best model per case under each residual metric.
- `full_field_pde_residual_statistics.json`: machine-readable copy of the same analysis.
