# PINO Full-Field PDE Residual Statistics

- Root: `D:\Ansys\SymPhONIC\OtherPDEs\LegacyHF_PDE_Benchmark_PINO_20260705`
- `sample_fields.npz` files analyzed: 68
- Cases: 17
- Field error definition: `prediction - exact`, computed over the same saved full field.
- Residual definition: `L(prediction) - source`, recomputed from the saved field arrays.
- Scope: every saved grid point is included; no PDE interior crop is applied.
- Boundary derivatives use the same `np.gradient(..., edge_order=2)` full-field finite-difference evaluator as the plotting/evaluation path.

## Aggregate By Model

| variant | cases | mean field mean(abs) | median field RMS | mean residual mean(abs) | median residual RMS | mean residual max(abs) |
|---|---:|---:|---:|---:|---:|---:|
| FNO | 17 | 3.614662e-01 | 3.223517e-01 | 1.285591e+01 | 5.543757e-01 | 1.273099e+02 |
| CFNO | 17 | 2.800037e-01 | 3.008292e-01 | 5.606870e+00 | 4.477685e-01 | 5.176861e+01 |
| HF_FNO | 17 | 4.431150e-01 | 2.688733e-01 | 9.560923e+00 | 1.181902e+00 | 5.342128e+02 |
| HF_CFNO | 17 | 4.436647e-01 | 1.893427e-01 | 3.972698e+00 | 8.421340e-01 | 9.189547e+01 |

## Best Variant Per Case

| PDE | case | best field mean(abs) | best field RMS | best residual mean(abs) | best residual RMS | best residual max(abs) |
|---|---|---|---|---|---|---|
| AllenCahn | Steady1D | HF_CFNO (9.865619e-02) | HF_CFNO (1.440140e-01) | HF_FNO (3.023014e-02) | HF_FNO (3.684393e-02) | CFNO (7.473050e-02) |
| AllenCahn | Steady2D | CFNO (5.302035e-02) | CFNO (7.784579e-02) | CFNO (1.389578e-02) | CFNO (1.713193e-02) | CFNO (6.282696e-02) |
| AllenCahn | Transient1D | CFNO (2.848111e-03) | CFNO (3.874400e-03) | CFNO (1.782393e-02) | CFNO (2.388858e-02) | CFNO (8.774655e-02) |
| Burgers | Steady1D | FNO (5.643269e-01) | CFNO (8.822565e-01) | CFNO (3.813433e-01) | CFNO (4.787131e-01) | CFNO (1.101767e+00) |
| Burgers | Steady2D | HF_FNO (1.969904e-01) | HF_FNO (2.688733e-01) | HF_CFNO (1.210319e-01) | HF_CFNO (1.573183e-01) | CFNO (4.267484e-01) |
| Burgers | Transient1D | HF_FNO (6.412424e-03) | HF_FNO (9.034473e-03) | CFNO (1.251829e-01) | HF_CFNO (1.644830e-01) | CFNO (5.285052e-01) |
| KdV | HighNonlinear_Steady1D | FNO (5.299247e-01) | FNO (6.529674e-01) | HF_CFNO (1.622024e+01) | HF_CFNO (3.092397e+01) | HF_CFNO (1.816636e+02) |
| KdV | HighNonlinear_Steady2D | HF_CFNO (3.343973e-01) | HF_CFNO (4.001225e-01) | HF_CFNO (1.389302e+01) | HF_CFNO (2.395047e+01) | HF_CFNO (2.427189e+02) |
| KdV | Steady1D | CFNO (9.236243e-01) | CFNO (1.073663e+00) | CFNO (4.406180e-01) | CFNO (7.580403e-01) | CFNO (2.882848e+00) |
| KdV | Steady2D | FNO (2.912168e-01) | FNO (4.122772e-01) | CFNO (1.951030e-01) | CFNO (4.294324e-01) | FNO (2.031255e+00) |
| KdV | Transient1D | CFNO (3.062909e-02) | CFNO (3.919220e-02) | CFNO (9.655764e+00) | CFNO (1.465418e+01) | HF_CFNO (8.580484e+01) |
| Poisson | Steady1D | CFNO (3.437919e-01) | CFNO (3.781119e-01) | CFNO (4.933109e+00) | CFNO (5.855455e+00) | CFNO (2.026549e+01) |
| Poisson | Steady2D | HF_CFNO (1.437677e-03) | HF_CFNO (1.844604e-03) | CFNO (5.402830e-01) | CFNO (8.630648e-01) | CFNO (6.421397e+00) |
| ReactionDiffusion | Steady1D | HF_CFNO (3.871041e-01) | FNO (4.946587e-01) | FNO (2.696962e-01) | FNO (3.763974e-01) | HF_FNO (8.044075e-01) |
| ReactionDiffusion | Steady2D | CFNO (1.111335e-01) | CFNO (1.489206e-01) | CFNO (5.094037e-02) | CFNO (6.400584e-02) | CFNO (1.958735e-01) |
| ReactionDiffusion | Transient1D | HF_CFNO (2.637599e-03) | HF_CFNO (3.352006e-03) | CFNO (3.761552e-02) | CFNO (4.766069e-02) | CFNO (1.814082e-01) |
| Wave | Transient1D | HF_CFNO (8.381571e-02) | HF_CFNO (1.167469e-01) | HF_CFNO (1.596788e+01) | CFNO (2.726617e+01) | CFNO (3.040794e+02) |

## Largest Field Mean-Abs Errors

| PDE | case | variant | field mean(abs) | field RMS | field max(abs) | residual mean(abs) |
|---|---|---|---:|---:|---:|---:|
| KdV | HighNonlinear_Steady1D | HF_CFNO | 2.543273e+00 | 2.646664e+00 | 3.695511e+00 | 1.622024e+01 |
| KdV | Steady1D | HF_FNO | 2.191976e+00 | 2.317799e+00 | 3.310894e+00 | 1.582130e+00 |
| KdV | HighNonlinear_Steady1D | HF_FNO | 1.930727e+00 | 2.027581e+00 | 3.069191e+00 | 1.808277e+01 |
| KdV | HighNonlinear_Steady2D | FNO | 1.718161e+00 | 1.874278e+00 | 3.234755e+00 | 5.199665e+01 |
| KdV | Steady1D | HF_CFNO | 1.715527e+00 | 1.843466e+00 | 2.718544e+00 | 2.051332e+00 |
| KdV | Steady1D | FNO | 1.095664e+00 | 1.229629e+00 | 2.016319e+00 | 7.022652e-01 |
| Burgers | Steady1D | HF_CFNO | 9.568560e-01 | 1.114746e+00 | 1.931569e+00 | 1.448474e+00 |
| KdV | Steady1D | CFNO | 9.236243e-01 | 1.073663e+00 | 1.842917e+00 | 4.406180e-01 |
| Burgers | Steady1D | CFNO | 6.746149e-01 | 8.822565e-01 | 1.692378e+00 | 3.813433e-01 |
| KdV | Steady2D | HF_FNO | 6.434666e-01 | 8.191719e-01 | 1.944189e+00 | 4.082765e-01 |

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

- `full_field_pde_residual_with_field_error_statistics.csv`: one row per `PDE/case/model`.
- `full_field_pde_residual_with_field_error_case_ranking.csv`: per-case model ranking by field error and PDE residual metrics.
- `full_field_pde_residual_with_field_error_model_summary.csv`: aggregate field-error and residual statistics by model family.
- `full_field_pde_residual_with_field_error_case_winners.csv`: best model per case under field-error and residual metrics.
- `full_field_pde_residual_with_field_error_statistics.json`: machine-readable copy of the same analysis.
