# Reynolds operator search

This folder runs a same-baseline Reynolds-number sweep for the lid-driven
cavity PINO-style solve.

Reynolds numbers:

```text
1e2, 1e3, 1e4, 1e5, 1e6, 1e7
```

Common settings:

```text
operator-specific method below
output_mode = streamfunction
N = 129
modes = 16
high_modes = 32
width = 24
depth = 5
lr = 1e-3
seed = 10492
max_iter = 5000
rho = 1
lid_velocity = 1
mu = 1 / Re
```

Methods:

```text
fno:
  Plain FNO with the same Fourier coordinate features as the HF variants.

cfno:
  Plain CFNO with the same Fourier coordinate features as the HF variants.

hf_cfno:
  Legacy HF-CFNO high-pass mechanism with replicate cavity padding.

subgrid_hf_cfno:
  New subgrid vorticity-gated HF-CFNO, no explicit L_gate alignment.
```

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_reynolds_search.ps1
```

The script writes each case to:

```text
Re*/method/
```

and logs to:

```text
logs/
```

After all runs, `summarize_reynolds_search.py` writes:

```text
summary.csv
summary.md
BestPDE_vs_Re.png
FinalPDE_vs_Re.png
GateMean_vs_Re.png
MaxVorticity_vs_Re.png
MetricHeatmap_best_pde.png
```

Ghia Re=100 comparison is only saved for Re=100. For other Reynolds numbers,
the solver saves centerline plots without attaching the Re=100 benchmark.
