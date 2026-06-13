# HF-CFNO Same-Baseline Comparison

## Fixed Settings

All runs below use the same solver entry point:

```text
python CFNO_Coupled_uvp.py
```

Common settings:

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

The comparison reports:

```text
PDE residual = L_cont + L_mom
total loss   = L_cont + L_mom + gate_alignment_weight * L_gate
```

The current results are stored in:

```text
hf_cfno_gate_ablation_samebaseline/
```

This folder also contains an archived aborted run from before the autograd fix.
Those archived files are kept only as debugging evidence and are not part of
the official comparison.

## Gate Autograd Fix

During the ablation, `new_gate_weak` and `new_gate_no_align` initially followed
nearly identical trajectories. The cause was that `SpatialGateMap` had been
stored as a detached diagnostic tensor before forming `L_gate`. That made the
alignment term a logged metric rather than a trainable physical constraint.

The corrected implementation keeps `last_spatial_gate` on the autograd graph
until `L_gate` is formed, while diagnostic summaries detach only when converting
to CPU values. A gradient smoke test confirmed nonzero gradients for the
vorticity-gate threshold, slope, and velocity-probe parameters.

## Runs

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

## Results

| Case | Gate mode | Best iter | Best PDE | Final PDE | Final total loss | Ghia RMSE | Effective gate mean | Spatial gate mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| old_hf_cfno | legacy | 3908 | 7.276e-3 | 4.672e-2 | 4.672e-2 | 2.453e-2 | 2.640e-1 | 1.000 |
| new_gate_weak | subgrid | 3690 | 8.685e-3 | 1.306e-2 | 1.416e-2 | 2.862e-2 | 5.559e-2 | 4.702e-1 |
| new_gate_stronger | subgrid | 3283 | 6.605e-3 | 9.543e-2 | 9.968e-2 | 2.203e-2 | 1.183e-1 | 4.470e-1 |
| new_gate_no_align | subgrid | 4913 | 5.335e-3 | 1.313e-2 | 1.313e-2 | 2.805e-2 | 5.567e-2 | 4.707e-1 |
| new_gate_with_fuse_but_gated | subgrid_gated_fuse | 4195 | 6.968e-3 | 3.283e-2 | 3.384e-2 | 3.396e-2 | 5.335e-2 | 4.665e-1 |

The full numeric table is:

```text
hf_cfno_gate_ablation_samebaseline/summary.csv
```

The aggregate plots are:

```text
hf_cfno_gate_ablation_samebaseline/AblationPDEResidual.png
hf_cfno_gate_ablation_samebaseline/AblationEffectiveGateMean.png
hf_cfno_gate_ablation_samebaseline/AblationSpatialGateMean.png
hf_cfno_gate_ablation_samebaseline/AblationMetricBars.png
```

## Interpretation

The no-alignment subgrid gate gives the lowest best PDE residual in this set,
but its Ghia centerline RMSE is worse than the original HF-CFNO. This means the
physics residual alone is still not enough to certify the flow profile.

The weak alignment setting (`gate_alignment_weight = 1e-2`) is now truly
trainable after the autograd fix, but it worsens best PDE residual and does not
improve Ghia RMSE. At this strength it acts more like a regularizer than a
useful physical controller.

The stronger setting (`high_gate_init = -1`, `gate_alignment_weight = 0.05`)
has the best Ghia RMSE in this ablation and a better best PDE residual than the
original HF-CFNO. However, its final PDE residual is poor because the trajectory
has late spikes. If this variant is pursued, it needs early stopping, a learning
rate schedule, or a staged alignment weight.

The gated-fuse variant restores the original fuse capacity but keeps every
high/fuse residual under `lambda*g`. It reaches a best PDE residual close to the
original HF-CFNO, but it does not recover the original Ghia accuracy. This
supports the suspicion that the original HF-CFNO's advantage partly comes from
an ungated fuse residual path, not only from high-frequency capacity.

## Plotting Check

Each official output directory contains:

```text
U.npy
V.npy
P.npy
Speed.npy / Speed.png
Vorticity.npy / Vorticity.png
Streamlines.png
centerline.csv
GhiaCenterline.png
ghia_re100_centerline_error.csv
history.csv
Loss.png
Gate.png
GateMap.npy / GateMap.png
SpatialGate.png
SpatialGateMap.npy / SpatialGateMap.png
PhysicalGateTarget.npy / PhysicalGateTarget.png
```

The solver uses `axis 0 = y`, `axis 1 = x`; vorticity is computed as
`dV/dx - dU/dy`, with `dV/dx` along axis 1 and `dU/dy` along axis 0.
