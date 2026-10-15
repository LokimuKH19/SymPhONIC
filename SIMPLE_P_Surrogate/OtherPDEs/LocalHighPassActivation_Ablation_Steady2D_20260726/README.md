# Local high-pass activation ablation

No statistically significant inner-activation effect was observed in the completed paired runs; strict equivalence within the practical margin was not proven.

Completed 20 runs (10 paired observations); the planned design contained 24 runs.
Testing was stopped early at the user's request.

Both arms retain `depthwise -> pointwise -> mix`; only GELU versus identity changes.
All runs use 1000 epochs, a 64x64 grid, paired initialization seeds, and hybrid training.
The equivalence margin is +/-5% in test relative L2.

See `activation_ablation_summary.json` and `activation_ablation_runs.csv` for exact values.
