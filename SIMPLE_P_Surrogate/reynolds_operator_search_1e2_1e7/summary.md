# Reynolds operator search summary

Methods: FNO, CFNO, legacy HF-CFNO with replicate cavity padding, and subgrid vorticity-gated HF-CFNO without explicit L_gate alignment.

|re_label|method|status|best_pde|best_pde_iter|final_pde|final_L_cont|final_L_mom|interior_max_speed|interior_max_abs_vorticity|final_gate_mean|ghia_re100_rmse|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Re1e02|cfno|ok|1.2997e+00|4922|1.3215e+00|2.5104e-01|1.0705e+00|6.8481e-01|3.3911e+01|nan|1.8904e-01|
|Re1e02|fno|ok|1.8824e+00|4910|1.9658e+00|1.9688e-01|1.7689e+00|6.8352e-01|3.1624e+01|nan|1.9721e-01|
|Re1e02|hf_cfno|ok|7.7197e-03|4936|1.4347e-02|3.7698e-03|1.0577e-02|9.0058e-01|3.6097e+01|2.6449e-01|1.9486e-02|
|Re1e02|subgrid_hf_cfno|ok|5.3354e-03|4913|1.3127e-02|3.6261e-03|9.5014e-03|8.9097e-01|3.6063e+01|5.5670e-02|2.8052e-02|
|Re1e03|cfno|ok|5.4089e-02|4999|5.4089e-02|1.3552e-02|4.0537e-02|4.2583e-01|3.4224e+01|nan|nan|
|Re1e03|fno|ok|9.0738e-02|4717|9.5458e-02|3.0440e-02|6.5019e-02|4.2620e-01|3.5399e+01|nan|nan|
|Re1e03|hf_cfno|ok|2.6308e-04|4928|7.8466e-04|1.3452e-04|6.5014e-04|8.3709e-01|4.5848e+01|2.6987e-01|nan|
|Re1e03|subgrid_hf_cfno|ok|4.6705e-04|4861|2.0890e-03|1.3355e-04|1.9554e-03|8.3306e-01|4.4171e+01|5.5943e-02|nan|
|Re1e04|cfno|ok|9.9707e-03|4932|1.0027e-02|4.4283e-03|5.5986e-03|3.5342e-02|9.1226e+00|nan|nan|
|Re1e04|fno|ok|8.1669e-03|4517|8.2822e-03|2.2722e-03|6.0100e-03|1.9178e-01|2.5498e+01|nan|nan|
|Re1e04|hf_cfno|ok|6.1438e-05|4588|1.9300e-04|2.0004e-05|1.7300e-04|4.6613e-01|4.1445e+01|2.7752e-01|nan|
|Re1e04|subgrid_hf_cfno|ok|1.3644e-04|4551|1.9267e-04|2.1845e-05|1.7082e-04|3.6912e-01|3.9759e+01|5.5288e-02|nan|
|Re1e05|cfno|ok|1.2338e-04|1644|1.3731e-04|4.8336e-05|8.8977e-05|4.1104e-03|3.5449e-01|nan|nan|
|Re1e05|fno|ok|1.4051e-04|4712|1.4096e-04|4.6096e-05|9.4865e-05|1.1384e-03|6.0699e-02|nan|nan|
|Re1e05|hf_cfno|ok|1.4594e-06|4984|2.1628e-06|2.1501e-07|1.9478e-06|8.3136e-02|5.2528e+00|2.7863e-01|nan|
|Re1e05|subgrid_hf_cfno|ok|4.3161e-06|4888|5.2613e-06|1.1561e-06|4.1052e-06|8.5397e-02|5.5086e+00|5.6050e-02|nan|
|Re1e06|cfno|ok|1.4061e-06|4952|1.4063e-06|4.6595e-07|9.4039e-07|1.0236e-04|3.6011e-03|nan|nan|
|Re1e06|fno|ok|1.4477e-06|4998|1.4499e-06|4.6212e-07|9.8778e-07|1.7472e-04|1.5015e-02|nan|nan|
|Re1e06|hf_cfno|ok|1.1580e-06|4998|1.1581e-06|4.8828e-07|6.6982e-07|5.2193e-04|2.6947e-02|2.6815e-01|nan|
|Re1e06|subgrid_hf_cfno|ok|1.1503e-06|4998|1.1506e-06|4.8799e-07|6.6259e-07|2.7838e-03|2.1033e-01|5.7170e-02|nan|
|Re1e07|cfno|ok|1.7665e-08|4997|1.7697e-08|5.8234e-09|1.1873e-08|2.8307e-05|3.0212e-03|nan|nan|
|Re1e07|fno|ok|5.6534e-08|4969|6.0979e-08|3.2566e-08|2.8413e-08|7.8098e-05|1.1658e-02|nan|nan|
|Re1e07|hf_cfno|ok|1.2025e-08|4966|1.2040e-08|5.1317e-09|6.9083e-09|1.4148e-04|7.4005e-03|2.6792e-01|nan|
|Re1e07|subgrid_hf_cfno|ok|1.2960e-08|4990|1.2985e-08|5.3283e-09|7.6566e-09|2.2493e-03|1.4954e-01|5.7245e-02|nan|

Metric notes:

- `best_pde` is `min(L_cont + L_mom)` over the run.
- `final_pde` is the residual at the final iteration.
- `ghia_re100_rmse` is only valid and only emitted for Re=100.
- Plain FNO/CFNO use the same Fourier coordinate features as the HF variants.
- Cross-Re absolute PDE residuals are not physical accuracy scores here. At
  high Re, low residual can coincide with collapsed interior speed and
  vorticity; compare field diagnostics as well as residuals.

Aggregate figures:

- `BestPDE_vs_Re.png`
- `FinalPDE_vs_Re.png`
- `GateMean_vs_Re.png`
- `MaxVorticity_vs_Re.png`
- `MetricHeatmap_best_pde.png`
