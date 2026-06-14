# HF-CFNO gate ablation

This folder keeps the same-baseline HF-CFNO gate tests requested for the cavity
case. Every run uses:

```text
operator = hf_cfno
output_mode = streamfunction
N = 129
modes = 16
high_modes = 32
width = 24
depth = 5
lr = 1e-3
seed = 10492
max_iter = 5000
```

Run all cases with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_gate_ablation.ps1
```

The cases are:

```text
old_hf_cfno:
  --gate-mode legacy
  --boundary-mode legacy_mixed
  --high-gate-init -1.0
  --gate-alignment-weight 0.0

new_gate_weak:
  --gate-mode subgrid
  --boundary-mode replicate
  --high-gate-init -2.0
  --gate-alignment-weight 1e-2

new_gate_stronger:
  --gate-mode subgrid
  --boundary-mode replicate
  --high-gate-init -1.0
  --gate-alignment-weight 0.05

new_gate_no_align:
  --gate-mode subgrid
  --boundary-mode replicate
  --high-gate-init -2.0
  --gate-alignment-weight 0.0

new_gate_with_fuse_but_gated:
  --gate-mode subgrid_gated_fuse
  --boundary-mode replicate
  --high-gate-init -2.0
  --gate-alignment-weight 1e-2
```

Each case writes field diagnostics, gate maps, loss curves, history CSV, and
Ghia Re=100 centerline comparison into its own subfolder. Logs are in `logs/`.
After the runs finish, `summarize_gate_ablation.py` writes:

```text
summary.csv
summary.md
AblationPDEResidual.png
AblationEffectiveGateMean.png
AblationSpatialGateMean.png
AblationMetricBars.png
```

The archive `_aborted_detached_gate_bug_20260613_164051` is kept only as a
debug record. It captured the pre-fix run where `SpatialGateMap` was detached
before `L_gate`, so the alignment term was logged but not trainable. The
official results in this folder were rerun after fixing that autograd path.
