# HF-CFNO Subgrid Vorticity Gate

## Motivation

The previous high-frequency branch acted mostly as extra capacity: a spectral band
operator and a local high-pass operator were added to the low-frequency CFNO
state with a global scalar gate. That is useful numerically, but the gate does
not know where high-frequency corrections are physically needed.

The revised HF-CFNO treats the high-pass branch as a subgrid correction. The
low-frequency operator predicts the resolved state, while the high-pass residual
is activated by a continuous local vortical-activity indicator. This gives the
gate a general interpretation that is not specific to the lid-driven cavity.

## Filtered Decomposition

For a hidden state z_l on a grid with spacing dx, dy, define a local filter
G_Delta with filter width

```text
Delta = k_f sqrt(dx dy),
```

where k_f is the odd high-pass filter size. A learned 1x1 projection P maps the
hidden state to a velocity-like carrier field

```text
q_l = P z_l = (u_l, v_l).
```

Then

```text
q_bar_l = G_Delta q_l,
q_prime_l = q_l - q_bar_l.
```

Here q_bar_l is the resolved carrier and q_prime_l is the subgrid-scale
fluctuation represented by the high-pass branch.

## Vorticity Activity

For 2D flow, the out-of-plane vorticity is

```text
omega(q) = d v / d x - d u / d y.
```

The local subgrid activity is

```text
a_l(x, y) =
    Delta sqrt( omega(q_bar_l)^2 + c_sg omega(q_prime_l)^2 + epsilon ).
```

The first term activates high-frequency correction in strong resolved vortical
regions. The second term activates it when the hidden carrier already contains
under-resolved high-pass vortical content. The coefficient c_sg is a tunable
subgrid weight.

To avoid dependence on the arbitrary scale of the hidden state, the activity is
normalized on each sample and layer:

```text
eta_l(x, y) = a_l(x, y) / ( mean_xy a_l + epsilon ).
```

This makes the indicator dimensionless and keeps the method portable across
grid sizes and flow parameters.

## Continuous Gate

The spatial high-pass gate is

```text
g_l(x, y) = sigmoid( s_l ( eta_l(x, y) - tau_l ) ),
```

where s_l > 0 is the gate slope and tau_l > 0 is the activation threshold. Both
are differentiable trainable parameters, represented through softplus so they
remain positive.

The final high-pass amplitude also includes a global cap

```text
lambda_l = sigmoid(alpha_l).
```

The HF-CFNO block becomes

```text
z_{l+1}
  = L_l(z_l) + lambda_l g_l(x, y) H_l(z_l),
```

where L_l is the low-frequency CFNO/FNO operator and H_l is the fused high-pass
residual from the spectral band and local high-pass paths.

The physical interpretation belongs to `g_l(x, y)`. The factor `lambda_l` is a
separate residual-amplitude cap that prevents the high-pass correction from
overriding the resolved operator early in training.

The high-pass residual can be parameterized in two physically gated ways. The
compact version uses a learned high-pass fusion from the spectral band and
local paths:

```text
H_l(z_l) = W_l( concat[B_l(z_l), C_l(z_l)] ).
```

The expressive gated-fuse version restores the original 1x1 fuse capacity but
keeps the entire high-frequency bypass inside the same physical gate:

```text
H_l(z_l)
  = B_l(z_l)
    + C_l(z_l)
    + F_l( concat[L_l(z_l), B_l(z_l), C_l(z_l)] ),

z_{l+1}
  = L_l(z_l) + lambda_l g_l(x, y) H_l(z_l).
```

This keeps the method interpretation intact: every high-frequency residual
term, including the learned fuse term, is active only where the continuous
vortical-activity gate permits it.

Because the derivatives, average filter, normalization, softplus, and sigmoid
are differentiable, gradients flow through the gate and into the projection,
threshold, slope, and upstream hidden features.

## Optional Physical Alignment

When physical velocity is available during training, the latent gate can be
anchored to the actual predicted vorticity. For a predicted velocity field
u = (u, v), define

```text
omega_phys = d v / d x - d u / d y,
a_phys = Delta sqrt( omega_phys^2 + epsilon ),
eta_phys = a_phys / ( mean_xy a_phys + epsilon ),
g_phys = sigmoid( s ( eta_phys - tau ) ).
```

The training objective may include

```text
L_gate = mean_xy ( g_latent - stopgrad(g_phys) )^2.
```

Here `g_latent` is the spatial gate before multiplication by the global residual
cap `lambda_l`. The stop-gradient keeps this term from changing the flow merely
to make the target easier. It instead asks the high-pass bypass gate to follow
the current physical vortical activity. In a supervised setting, `g_phys` can be
computed from labeled velocity; in a physics-only setting, it can be computed
from the current predicted velocity.

## Boundary Handling

The old local high-pass path used circular padding in one direction because the
operator was originally written for a rotor problem. The revised implementation
parameterizes the padding mode:

```text
boundary_mode in {replicate, reflect, circular}
```

For the cavity experiment, `replicate` is used by default so opposite walls are
not connected. Rotor or annular-periodic cases can still use `circular`.

## Implementation Correspondence

The current code keeps two gate diagnostics:

```text
SpatialGateMap = mean_l g_l(x, y),
GateMap        = mean_l lambda_l g_l(x, y).
```

`SpatialGateMap` is the physical vortical-activity gate used in `L_gate`.
`GateMap` is the effective high-pass residual amplitude actually multiplying
the fused high-frequency residual. Both are useful: the first checks the
methodology, while the second checks how strongly the residual branch is allowed
to affect the solution.

For training, `SpatialGateMap` is kept on the autograd graph until `L_gate` is
formed. Diagnostic summaries and saved maps detach only when converting to CPU
arrays. This is essential: detaching the spatial gate before `L_gate` would make
the alignment term a logged metric rather than a trainable physical constraint.

The training CSV stores both sets of statistics:

```text
iter,L_cont,L_mom,loss,L_gate,
gate_mean,gate_min,gate_max,
spatial_gate_mean,spatial_gate_min,spatial_gate_max
```

The implementation lives in:

```text
VorticitySubgridGate2d
HFCFNOBlock / HFFNOBlock
HF_CFNO2d_small.high_pass_spatial_gate_map()
HF_CFNO2d_small.high_pass_gate_map()
NeuralCavityPressure.physical_vorticity_gate()
```

For ablation against the original HF-CFNO, the code also exposes

```text
gate_mode = legacy
```

which recovers the earlier high-frequency path:

```text
z_{l+1}
  = L_l(z_l)
    + sigmoid(alpha_l) ( B_l(z_l) + C_l(z_l) )
    + F_l( concat[L_l(z_l), B_l(z_l), C_l(z_l)] ).
```

Here `B_l` is the spectral high-band branch, `C_l` is the local high-pass
branch, and `F_l` is the ungated 1x1 fusion path. This branch is intentionally
kept as a baseline because it has much stronger high-frequency capacity, even
though its fusion path has weaker physical interpretability.

The expressive physically gated alternative is selected with

```text
gate_mode = subgrid_gated_fuse
```

which uses the same `B_l`, `C_l`, and `F_l` branches as the legacy expression
but multiplies their combined high-frequency residual by
`lambda_l g_l(x, y)`. Therefore it tests whether the original HF-CFNO advantage
comes from fuse capacity itself or from an ungated residual escape path.

The original rotor-style boundary treatment can be reproduced with

```text
boundary_mode = legacy_mixed
```

which means circular padding in the first spatial direction and replicate
padding in the second spatial direction.

## Same-Baseline Protocol

All variants must be tested through the same entry point:

```text
CFNO_Coupled_uvp.py
```

Use the same solver and training settings:

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

The exact original legacy reproduction command settings are:

```text
--gate-mode legacy
--boundary-mode legacy_mixed
--high-gate-init -1.0
--gate-alignment-weight 0.0
```

This checks compatibility with the original rotor-origin HF-CFNO implementation.
It should not be the only fair cavity comparison because it keeps the original
mixed circular/replicate padding.

The boundary-controlled legacy command settings are:

```text
--gate-mode legacy
--boundary-mode replicate
--high-gate-init -1.0
--gate-alignment-weight 0.0
```

This is the preferred baseline when isolating the high-frequency mechanism under
the same non-periodic cavity padding used by the subgrid-gated variant.

The subgrid-gated command settings are:

```text
--gate-mode subgrid
--boundary-mode replicate
--high-gate-init -2.0
--gate-alignment-weight 1e-2
```

The expressive gated-fuse command settings are:

```text
--gate-mode subgrid_gated_fuse
--boundary-mode replicate
--high-gate-init -2.0
--gate-alignment-weight 1e-2
```

Report both the PDE residual and the total training objective:

```text
PDE residual = L_cont + L_mom
total loss   = L_cont + L_mom + gate_alignment_weight * L_gate
```

The legacy baseline has `L_gate = 0`, so its PDE residual and total loss are
identical. The subgrid-gated variant has an extra alignment term, so PDE
residual is the fair solver-accuracy comparison and total loss is the fair
training-objective comparison.

## Field-Level Diagnostics

Residual loss alone is not a sufficient physical-quality metric for the
physics-only cavity solve. The solver now saves field-level diagnostics:

```text
U.npy
V.npy
P.npy
Speed.npy
Vorticity.npy
Streamlines.png
Speed.png
Vorticity.png
centerline.csv
ghia_re100_centerline_error.csv
GhiaCenterline.png
history.csv
Loss.png
Gate.png
GateMap.npy
GateMap.png
SpatialGate.png
SpatialGateMap.npy
SpatialGateMap.png
PhysicalGateTarget.npy
PhysicalGateTarget.png
```

The field layout follows the solver convention `axis 0 = y`, `axis 1 = x`.
Thus vorticity is plotted as

```text
omega = dV/dx - dU/dy
```

## Reynolds Sweep Extension

The same solver entry point also supports Reynolds-number sweeps:

```text
--re Re
```

with

```text
mu = rho U_lid L / Re.
```

For the Re sweep, plain FNO and plain CFNO use the same Fourier coordinate
features as the HF variants. This is necessary because the forcing input is a
constant lid-velocity field; without coordinate features, a plain translation
equivariant FNO/CFNO has too little positional information for a fair cavity
comparison.

The Re-sweep subgrid method uses

```text
--gate-mode subgrid
--boundary-mode replicate
--high-gate-init -2.0
--gate-alignment-weight 0.0
```

This isolates the continuous vorticity-gated high-pass mechanism without adding
an auxiliary physical-gate alignment loss. The legacy HF-CFNO comparison uses
the legacy high-pass/fuse path with `replicate` cavity padding rather than the
old rotor-origin mixed circular padding.

The Re sweep also shows an important limitation. For high Reynolds numbers, the
physics-only residual can become degenerate: when `mu` is very small, weak
interior velocity fields can make both the convective and viscous residuals
small. Therefore cross-Re absolute PDE residuals should not be interpreted as
physical quality by themselves. A lower residual at larger Re may indicate a
collapsed interior flow rather than a more accurate cavity solution.

This limitation is not solved by the present HF-CFNO design. The legacy
high-pass branch and the subgrid-vorticity gate change how the network
represents high-frequency corrections, but they do not change the physics-only
objective. With

```text
R_m = rho (u dot grad) u + grad p - mu Laplacian(u),
mu = rho U_lid L / Re,
```

an interior velocity field of amplitude epsilon and nearly constant pressure
can make the residual terms scale like

```text
continuity        O(epsilon),
convection        O(epsilon^2),
viscous diffusion O(mu epsilon / Delta x^2).
```

At large Re, the small `mu` factor makes the viscous penalty weak, and reducing
the interior velocity also suppresses the convective residual. Because the lid
boundary condition is hard-imposed while the residual is measured on the
cropped interior, the loss does not by itself guarantee correct momentum
transport from the moving wall into the cavity.

The vorticity gate remains useful as an interpretable high-pass activation
mechanism: it can preserve more localized vortical activity than a plain
low-frequency operator. However, it is not an independent physical constraint.
If the learned field collapses and its vorticity becomes small, the gate signal
also weakens. High-Re cavity training therefore needs a non-degenerate
objective, such as staged Re continuation, scale-balanced residual terms,
wall-shear or momentum-flux constraints, energy/enstrophy constraints,
interior velocity anchoring, or reference data where available.

The Re-sweep summary therefore reports interior speed and interior vorticity
after removing two boundary layers from the diagnostic arrays.

where `dV/dx` is taken along axis 1 and `dU/dy` is taken along axis 0.
The centerline diagnostics use `U[:, mid]` for `u(x=0.5, y)` and `V[mid, :]`
for `v(x, y=0.5)`.

For the Re=100 lid-driven cavity, `GhiaCenterline.png` overlays the model
centerline velocities against the standard Ghia et al. benchmark. This is the
preferred diagnostic for judging whether a low residual is physically useful or
merely a weak/trivial field.

## Interpretation

The gate is analogous to a differentiable LES indicator:

- high vorticity means strong local rotation and likely smaller active scales;
- high subgrid vorticity means unresolved oscillatory content is already present;
- the high-pass branch is residual-only, so it corrects the resolved operator
  instead of replacing it;
- the gate is continuous, trainable, and grid-aware through Delta.
