# Reynolds Search Analysis

## Scope

This sweep tests four operator choices on Re = 1e2 ... 1e7 with the same
physics-only cavity training setup:

```text
FNO
CFNO
HF-CFNO legacy high-pass mechanism with replicate cavity padding
Subgrid-HF-CFNO with continuous vorticity gate and no explicit L_gate alignment
```

All runs completed with exit code 0 and empty stderr logs.

## Negative Result: Residual-Flow Decoupling

The Reynolds sweep exposes a real failure mode of the current physics-only
training objective. As Re increases, the reported PDE residual can decrease by
orders of magnitude, while the predicted flow becomes less physical. This must
not be reported as improved physical accuracy.

The steady momentum residual minimized here is

```text
R_m = rho (u dot grad) u + grad p - mu Laplacian(u),
mu = rho U_lid L / Re.
```

If the learned interior velocity has amplitude epsilon and the pressure becomes
nearly constant, then the residual terms scale roughly as

```text
continuity        O(epsilon),
convection        O(epsilon^2),
viscous diffusion O(mu epsilon / Delta x^2).
```

Therefore, when Re is large and `mu` is small, a weak interior field can produce
a very small residual even if it does not represent the lid-driven cavity flow.
The hard boundary condition still imposes the moving lid, but the residual is
evaluated on the cropped interior and does not by itself require the correct
momentum transfer from the lid into the cavity. The optimizer can therefore
prefer a numerically quiet field: nearly constant pressure, weak interior
velocity, weak interior vorticity, and small residual.

This is not a problem that the current HF design can solve by itself. The
legacy HF-CFNO and the subgrid-gated HF-CFNO only change the parameterization
of the solution and the high-frequency residual path. They are still trained
against the same degenerate scalar residual. At high Re, reducing the physical
activity also weakens the vortical signal that would activate the high-pass
branch, so a vorticity gate can preserve more local activity but cannot force a
physically correct cavity circulation without an additional non-degenerate
constraint.

Use these results mainly to compare methods at the same Re and to inspect field
statistics such as interior speed and interior vorticity. Cross-Re absolute PDE
residual values are not reliable physical-quality measures in this setup.

## Best PDE Residual Winners

| Re | Best method | Best PDE | Interior max speed | Interior max abs vorticity |
|---:|---|---:|---:|---:|
| 1e2 | subgrid_hf_cfno | 5.335e-3 | 8.910e-1 | 3.606e1 |
| 1e3 | hf_cfno | 2.631e-4 | 8.371e-1 | 4.585e1 |
| 1e4 | hf_cfno | 6.144e-5 | 4.661e-1 | 4.145e1 |
| 1e5 | hf_cfno | 1.459e-6 | 8.314e-2 | 5.253e0 |
| 1e6 | subgrid_hf_cfno | 1.150e-6 | 2.784e-3 | 2.103e-1 |
| 1e7 | hf_cfno | 1.202e-8 | 1.415e-4 | 7.401e-3 |

## Observations

At Re=100, the two HF-CFNO variants dominate plain FNO and CFNO. The subgrid
gate gives the best PDE residual, while legacy HF-CFNO has the better Ghia
Re=100 centerline RMSE.

From Re=1e3 to Re=1e5, legacy HF-CFNO is the strongest by PDE residual and also
keeps much higher interior vorticity than plain FNO/CFNO. This suggests that
the legacy high-pass capacity is still useful when the cavity solve remains
moderately active.

At Re=1e6 and Re=1e7, all methods reach extremely small PDE residuals, but the
interior field statistics reveal collapse. Plain FNO/CFNO have almost no
interior velocity or vorticity. Legacy HF-CFNO also collapses strongly. The
subgrid-HF-CFNO retains the most interior vortical activity at Re=1e6 and
Re=1e7, despite not always winning the scalar PDE residual.

This means the high-Re residual trend is misleading: lower residual does not
mean a better cavity solution. It mainly reflects that the dimensional residual
becomes easier to minimize when viscosity is small and the network suppresses
interior motion.

The subgrid effective gate mean stays around 0.055 across Re, while the legacy
HF-CFNO effective gate stays around 0.27. The subgrid model's high-Re difference
therefore comes from the spatially structured vorticity gate and residual path,
not from a larger global high-pass amplitude.

## Methodological Takeaway

The subgrid gate is not a universal scalar-residual winner, and it should not be
presented as a cure for the high-Re residual degeneracy. Its value appears in
preserving localized vortical activity under a physically interpretable
high-pass path. For high-Re cavity tests, the next methodology step should be a
non-degenerate objective: for example staged continuation in Re, nondimensional
or scale-balanced residual terms, wall-shear or momentum-flux constraints,
energy/enstrophy constraints, interior velocity anchoring, or a
benchmark/reference term where available. Without that, the PDE residual alone
can reward collapsed high-Re fields.
