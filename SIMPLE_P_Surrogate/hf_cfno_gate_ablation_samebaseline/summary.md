# HF-CFNO gate ablation summary

All cases use the same entry point, seed, grid, width, depth, learning rate, and 5000 iterations unless the run script parameter is changed.

|case|gate_mode|high_gate_init|gate_alignment_weight|best_pde|best_pde_iter|final_pde|final_loss|ghia_rmse|final_gate_mean|final_spatial_gate_mean|
|---|---|---|---|---|---|---|---|---|---|---|
|old_hf_cfno|legacy|-1.000000e+00|0.000000e+00|7.276341e-03|3908|4.672285e-02|4.672285e-02|2.453497e-02|2.640355e-01|1.000000e+00|
|new_gate_weak|subgrid|-2.000000e+00|1.000000e-02|8.685461e-03|3690|1.306252e-02|1.416187e-02|2.861944e-02|5.558644e-02|4.702114e-01|
|new_gate_stronger|subgrid|-1.000000e+00|5.000000e-02|6.605286e-03|3283|9.542748e-02|9.967653e-02|2.202578e-02|1.183192e-01|4.470367e-01|
|new_gate_no_align|subgrid|-2.000000e+00|0.000000e+00|5.335437e-03|4913|1.312747e-02|1.312747e-02|2.805190e-02|5.566972e-02|4.707451e-01|
|new_gate_with_fuse_but_gated|subgrid_gated_fuse|-2.000000e+00|1.000000e-02|6.967789e-03|4195|3.283374e-02|3.383604e-02|3.396497e-02|5.335193e-02|4.665052e-01|

Metric notes:

- `best_pde` is `min(L_cont + L_mom)` over the run.
- `final_loss` is `L_cont + L_mom + gate_alignment_weight * L_gate` at the final iteration.
- `ghia_rmse` is computed from the saved Re=100 centerline comparison file.
- `final_gate_mean` is the effective residual amplitude `lambda*g`; `final_spatial_gate_mean` is the physical spatial gate `g`.
