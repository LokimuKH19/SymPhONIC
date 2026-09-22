# OtherPDEs Directory Guide

This directory benchmarks FNO, CFNO, and their high-frequency-enhanced variants on several PDE families. It contains data-only, physics-only, and hybrid training experiments, together with ablations for spectral attention, local high-pass activation, and native one-dimensional neural operators.

> The directory preserves both the legacy multidimensional experiments and the later native-1D experiments. Dated legacy result directories are reproducibility artifacts and should not be overwritten by new runs.

The Chinese version of this guide is available in `Readme.md`.

## Quick Navigation

| Goal | Recommended entry point |
|---|---|
| Inspect the legacy FNO/CFNO/HF multi-PDE comparison | `LegacyHF_PDE_Benchmark_*_20260705/` |
| Run the legacy steady-1D, steady-2D, and transient-1D benchmark | `run_legacy_hf_pde_suite.py` |
| Run true native `[B,C,N]` steady-1D operators | `run_legacy_hf_pde_steady1d_suite.py` |
| Inspect the native-1D 700-parameter and 10K-parameter ablations | `LegacyHF_PDE_Benchmark_UniqueNativeSteady1D_SqrtParams_Final_20260820/` |
| Compare low- and high-frequency spectral attention | `KernelAttention_*_PDE_Comparison_20260718/` |
| Inspect the inner-activation ablation of the local high-pass block | `LocalHighPassActivation_Ablation_Steady2D_20260726/` |
| Redraw existing legacy figures without retraining | `redraw_legacy_pde_results.py` |
| Recompute full-field PDE residual and field-error statistics from NPZ files | `summarize_pino_full_field_residuals.py` |

## Directory Tree

The tree below shows the meaningful project structure. Repeated model-level output files are represented by their common pattern, which is documented later in this guide.

```text
OtherPDEs/
|-- Readme.md
|-- Readme_EN.md
|-- NeuroOperators.py
|-- NeuralOperators.py
|-- run_legacy_hf_pde_suite.py
|-- run_legacy_hf_pde_steady1d_suite.py
|-- run_steady1d_gpu_isolated.ps1
|-- run_steady1d_10k_ablation_gpu_isolated.ps1
|-- run_attention_pde_modes.py
|-- run_local_highpass_activation_ablation.py
|-- redraw_legacy_pde_results.py
|-- summarize_pino_full_field_residuals.py
|-- test_spectral_kernel_attention.py
|-- test_local_highpass_activation.py
|-- benchmark_logs/
|-- KernelAttention_Linear_PDE_Comparison_20260718/
|   |-- DataDriven/
|   |-- DataFree/
|   |-- Hybrid/
|   `-- SUMMARY.md
|-- KernelAttention_Nonlinear_PDE_Comparison_20260718/
|   |-- DataDriven/
|   |-- DataFree/
|   |-- Hybrid/
|   `-- SUMMARY.md
|-- LegacyHF_PDE_Benchmark_DataOnly_20260705/
|   |-- AllenCahn/
|   |-- Burgers/
|   |-- KdV/
|   |-- Poisson/
|   |-- ReactionDiffusion/
|   `-- Wave/
|-- LegacyHF_PDE_Benchmark_Hybrid_20260705/
|   `-- <same PDE hierarchy as DataOnly>
|-- LegacyHF_PDE_Benchmark_PhysicsOnly_20260705/
|   `-- <currently completed physics-only cases>
|-- LegacyHF_PDE_Benchmark_UniqueNativeSteady1D_SqrtParams_Final_20260820/
|   |-- PhysicsOnly/
|   |-- Hybrid/
|   |-- DataOnly/
|   |-- CrossModeComparisons/
|   |-- ParamScale_10K_Rerun/
|   `-- FINAL_SUMMARY.md
|-- LocalHighPassActivation_Ablation_Steady2D_20260726/
|   |-- Poisson/Steady2D/
|   |-- Burgers/HighNonlinear_Steady2D/
|   `-- README.md
`-- __pycache__/
```

## Core Programs

### `NeuroOperators.py`

The principal neural-operator architecture library in this directory. It contains:

- two-dimensional FNO, CFNO, HF-FNO, and HF-CFNO implementations;
- Fourier and Chebyshev/DCT spectral convolutions;
- the bidirectional multiband spectral convolution `MultiBandSpectralConv2d`;
- the local high-pass block `LocalHighPassBlock2d`;
- Fourier low-frequency, Chebyshev low-frequency, and Fourier high-frequency spectral attention;
- `SpectralAttentionOperator2d`;
- high-frequency Fourier features, boundary-aware padding, switchable activations, and supporting utilities.

Common operator changes should be implemented here and accompanied by the relevant tests.

### `NeuralOperators.py`

A compatibility import shim containing only:

```python
from NeuroOperators import *
```

It preserves the historical module spelling used by older scripts. All actual implementations belong in `NeuroOperators.py`; models should not be duplicated here.

### `run_legacy_hf_pde_suite.py`

The main legacy multi-PDE benchmark program. It:

- constructs manufactured/reference solutions, source terms, and training data;
- runs steady-1D, steady-2D, and transient-1D cases;
- supports Poisson, Wave, KdV, Allen-Cahn, Burgers, and Reaction-Diffusion equations;
- supports baseline and `high_nonlinear` profiles;
- compares FNO, CFNO, HF-FNO, and HF-CFNO, and can also run spectral-attention variants;
- supports `data`, `pde`, and `hybrid` training modes;
- imposes boundary conditions through hard projections rather than a boundary loss;
- writes checkpoints, histories, field NPZ files, error figures, residual figures, spectra, and model-comparison tables.

This is the legacy multidimensional framework. Its historical 1D cases are still represented through the two-dimensional operator framework. Use the dedicated native-1D program below for true one-dimensional operators.

Frequently used arguments include:

```text
--profile baseline|high_nonlinear
--training-mode data|pde|hybrid
--variants FNO,CFNO,HF_FNO,HF_CFNO
--pdes <PDE filter>
--cases <case filter>
--target-params <parameter budget>
--resume
```

Formal benchmark runs are required to use at least 1000 epochs.

### `run_legacy_hf_pde_steady1d_suite.py`

The dedicated native steady-1D benchmark, with tensor contract `[B,C,N]` and no replicated pseudo-2D axis. It provides:

- native `SpectralConv1d`, Chebyshev/CFNO, HF-FNO, and HF-CFNO implementations;
- one-sided real-signal Fourier analysis and Hermitian-conjugate reconstruction for the positive/negative-frequency branches;
- joint width/head-hidden searches for closely matched model sizes;
- two-point Dirichlet hard constraints for Poisson;
- signed left value, slope, and curvature inputs for KdV;
- signed left value and slope inputs for Burgers, Allen-Cahn, and Reaction-Diffusion, which are formulated as spatial initial-value problems;
- PhysicsOnly, Hybrid, and DataOnly modes;
- decoupled training and `--plots-only` post-processing;
- automatic checks for parameter counts, CUDA gradients, hard constraints, Hermitian frequency pairing, and figure completeness.

This program generates `LegacyHF_PDE_Benchmark_UniqueNativeSteady1D_SqrtParams_Final_20260820/`.

### `run_steady1d_gpu_isolated.ps1`

The process-isolated GPU launcher for the approximately 700-parameter native-1D benchmark. It:

- starts one Python process per training mode, case, and model;
- releases process-level CUDA and host memory after every model;
- retries failed models with progressively smaller micro-batches;
- skips completed runs by checking for `metrics.json`;
- performs final aggregation and full plot rendering.

### `run_steady1d_10k_ablation_gpu_isolated.ps1`

The parameter-scale ablation launcher for all native steady-1D cases. By default it runs:

- 9 PDE cases;
- 3 training modes;
- 4 models;
- 1200 epochs per model;
- an approximately 10K parameter budget;
- output under `ParamScale_10K_Rerun/` in the native-1D result root, while automatically skipping completed runs.

### `run_attention_pde_modes.py`

The three-mode spectral-kernel-attention orchestrator. It calls `run_legacy_hf_pde_suite.py` for:

- `F_ATTN`: Fourier low-frequency attention;
- `C_ATTN`: Chebyshev low-frequency attention;
- `HF_ATTN`: Fourier high-frequency attention;
- DataDriven, DataFree, and Hybrid training.

All variants share the same CFNO backbone and contain exactly one attention layer. The program also summarizes the learned gate, source/state RMS ratio, field error, and PDE residual across modes.

### `run_local_highpass_activation_ablation.py`

The steady-2D ablation for the inner activation of `LocalHighPassBlock2d`. It compares:

- `gelu`;
- `identity`, which disables only the inner activation;
- HF-FNO and HF-CFNO;
- paired random seeds and matched initializations.

It analyzes test relative L2, frequency-component error, convergence, paired relative changes, statistical significance, and practical equivalence.

### `redraw_legacy_pde_results.py`

Redraws figures from existing `sample_fields.npz`, `history.csv`, and `metrics.json` files without retraining. It implements:

- Times New Roman typography;
- 12 px axis labels and 10.5 px legends/colorbars;
- normalization by the largest loss component in the first five epochs;
- prediction, theory, error, and residual distributions;
- Fourier-coefficient errors and radial-wavenumber statistics for higher-dimensional fields;
- model-level, case-level, and PDE-level comparisons.

The script's default root names come from an earlier project stage. For the current result directories, pass the actual DataOnly, Hybrid, and PhysicsOnly roots explicitly through `--roots`.

### `summarize_pino_full_field_residuals.py`

Reads every `sample_fields.npz` under a benchmark root and recomputes full-grid statistics for:

- mean, mean absolute, MSE, RMS, P95, and maximum of `prediction - theory`;
- PDE residuals of the prediction and exact solution;
- source-scale-normalized residuals;
- per-case rankings, winners, and model-family aggregates.

It does not train a model. It is responsible for the `full_field_pde_residual_*` reports stored in the PhysicsOnly result root.

### Test Programs

#### `test_spectral_kernel_attention.py`

Checks:

- shapes and gradients for all three spectral-attention sources;
- rejection of constant fields by the high-frequency source;
- the requirement that `SpectralAttentionOperator2d` contains exactly one attention layer.

#### `test_local_highpass_activation.py`

Checks:

- preservation of the pointwise-path gradient in `identity` mode;
- equal parameter counts for GELU and identity variants;
- different forward maps under identical weights.

## Experiment and Result Directories

### `benchmark_logs/`

Historical standard-output, standard-error, resume, and watcher logs for the legacy benchmark. Filenames normally encode the training mode, PDE/profile, date, and resume stage. `run_hybrid_high_after_baseline_20260706.ps1` is the historical launcher that chained hybrid baseline and high-nonlinearity jobs.

Use this directory for diagnosing old runs, not as a destination for new results.

### `KernelAttention_Linear_PDE_Comparison_20260718/`

Spectral-attention comparisons on linear PDEs:

```text
DataDriven|DataFree|Hybrid/
|-- Poisson/Steady1D/{F_ATTN,C_ATTN,HF_ATTN}/
|-- Poisson/Steady2D/{F_ATTN,C_ATTN,HF_ATTN}/
`-- Wave/Transient1D/{F_ATTN,C_ATTN,HF_ATTN}/
```

`SUMMARY.md`, `cross_mode_comparison.*`, and the `cross_mode_*.png` files summarize field error, PDE residual, and attention-gate diagnostics across the three training modes.

### `KernelAttention_Nonlinear_PDE_Comparison_20260718/`

Spectral-attention comparisons on high-nonlinearity transient-1D PDEs:

```text
DataDriven|DataFree|Hybrid/
|-- AllenCahn/HighNonlinear_Transient1D/{F_ATTN,C_ATTN,HF_ATTN}/
|-- Burgers/HighNonlinear_Transient1D/{F_ATTN,C_ATTN,HF_ATTN}/
|-- KdV/HighNonlinear_Transient1D/{F_ATTN,C_ATTN,HF_ATTN}/
`-- ReactionDiffusion/HighNonlinear_Transient1D/{F_ATTN,C_ATTN,HF_ATTN}/
```

The root `SUMMARY.md` reports the winner and attention diagnostics for every PDE and training mode.

### `LegacyHF_PDE_Benchmark_DataOnly_20260705/`

Legacy data-only results. The optimization objective contains only field-data error; the PDE residual is retained as a diagnostic. The directory currently contains 27 cases:

- Poisson: Steady1D and Steady2D;
- Wave: Transient1D;
- KdV, Allen-Cahn, Burgers, and Reaction-Diffusion: baseline and high-nonlinearity Steady1D, Steady2D, and Transient1D cases.

Every case compares FNO, CFNO, HF-FNO, and HF-CFNO.

### `LegacyHF_PDE_Benchmark_Hybrid_20260705/`

Legacy hybrid data-plus-physics results. Its 27-case hierarchy matches DataOnly. The objective is the data error plus a weighted PDE residual.

This directory is the primary legacy source for studying the branch-anchoring role of data and the smoothing/regularizing role of the physics loss.

### `LegacyHF_PDE_Benchmark_PhysicsOnly_20260705/`

Legacy physics-only results. The optimization objective contains only the PDE residual; the theoretical field is used for evaluation and plotting, not as a training target. The current directory contains 17 cases:

- baseline cases from all six PDE families;
- high-nonlinearity Steady1D and Steady2D KdV cases;
- not every high-nonlinearity extension present in DataOnly and Hybrid.

In addition to the standard benchmark summaries, this root contains `full_field_pde_residual_*` and `full_field_pde_residual_with_field_error_*` reports over all grid points.

### `LegacyHF_PDE_Benchmark_UniqueNativeSteady1D_SqrtParams_Final_20260820/`

The native steady-1D ablation with `[B,C,N]` models and hard-constraint inputs designed to make the mathematical branch explicit.

```text
PhysicsOnly|Hybrid|DataOnly/
|-- Poisson/Steady1D/
|-- KdV/{Steady1D,HighNonlinear_Steady1D}/
|-- AllenCahn/{Steady1D,HighNonlinear_Steady1D}/
|-- Burgers/{Steady1D,HighNonlinear_Steady1D}/
`-- ReactionDiffusion/{Steady1D,HighNonlinear_Steady1D}/
```

Important contents:

- `FINAL_SUMMARY.md`: summary of the approximately 700-parameter experiment;
- `PhysicsOnly/`, `Hybrid/`, and `DataOnly/`: 108 model results across the three modes;
- `CrossModeComparisons/`: comparisons of the same PDE/case/model across training modes;
- `ParamScale_10K_Rerun/`: the approximately 10K-parameter ablation for all nine native steady-1D cases, containing 108 models;
- `Readme.txt`: the brief historical note explaining the 10K follow-up;
- `plot_manifest.json`: required-figure completeness verification;
- `completion_report.json`: training and post-processing completion state.

`ParamScale_10K_Rerun/PARAMETER_SCALE_10K_SUMMARY.md` is the 10K experiment summary. All 27 training-mode/case directories contain `RESULTS_10K.md`. The completion inventory verifies all 108 models and all 1,332 required plots.

### `LocalHighPassActivation_Ablation_Steady2D_20260726/`

The steady-2D ablation of the inner GELU in `LocalHighPassBlock2d`:

```text
Poisson/Steady2D/{HF_FNO,HF_CFNO}/{gelu,identity}/seed_*/
Burgers/HighNonlinear_Steady2D/{HF_FNO,HF_CFNO}/{gelu,identity}/seed_*/
```

The planned design contained 24 runs and was stopped at 20 completed runs at the user's request. No statistically significant inner-activation effect was observed, although strict equivalence within the +/-5% practical margin was not established. See the local `README.md` and `activation_ablation_summary.json` for exact results.

### `__pycache__/`

Automatically generated Python bytecode cache. It is neither source code nor an experiment result and can be ignored.

## Result-Directory Conventions

### Legacy FNO/CFNO/HF benchmark hierarchy

```text
<training-root>/
`-- <PDE>/
    `-- <Case>/
        |-- FNO/
        |-- CFNO/
        |-- HF_FNO/
        |-- HF_CFNO/
        |-- model_comparison_metrics.csv
        |-- comparison_convergence*.png
        |-- comparison_frequency_component_abs_error.png
        `-- comparison_test_rel_l2.png
```

### Standard model-directory files

| File | Meaning |
|---|---|
| `best_checkpoint.pt` | Model selected by the validation objective of the corresponding training mode |
| `history.csv` | Per-epoch train/validation/test objective, data loss, PDE loss, and diagnostics |
| `metrics.json` | Parameter count, best epoch, test errors, residual statistics, and run configuration |
| `sample_fields.npz` | Prediction, exact/theory, source, error, residual, and related field arrays |
| `prediction_exact_error.png` | Prediction, theoretical solution, and error distribution |
| `prediction_exact_error_line.png` | One-dimensional line representation |
| `residual_distribution.png` | PDE residual distributions of prediction and theory |
| `spectrum_analysis.png` | Prediction/theory spectra and spectral error |
| `frequency_component_abs_error.csv/.png` | Absolute Fourier-coefficient error by wavenumber |
| `training_convergence*.png` | Normalized and absolute objective/data/physics convergence curves |
| `plot_normalization.json` | Convergence normalization scale and plotting configuration |

### Higher-level summary files

| Level | Typical outputs |
|---|---|
| Case | Four-model convergence, test relative L2, and frequency-error comparisons |
| PDE | `frequency_component_error_comparison.png` |
| Training-mode root | `global_comparison.csv/.json`, winner summaries, and global heatmaps |
| Cross-mode root | `CrossModeComparisons/`, cross-mode CSV/JSON tables, and heatmaps |

## Names and Training Modes

| Name | Meaning |
|---|---|
| FNO | Fourier Neural Operator |
| CFNO | Chebyshev/cosine-domain FNO variant |
| HF-FNO | FNO enhanced with high-frequency bands, a local high-pass path, and high-frequency features |
| HF-CFNO | Corresponding high-frequency-enhanced CFNO |
| F_ATTN | Fourier low-frequency spectral attention |
| C_ATTN | Chebyshev low-frequency spectral attention |
| HF_ATTN | Fourier high-frequency spectral attention |
| DataOnly/DataDriven | Training only with field-data error |
| PhysicsOnly/DataFree | Training only with the PDE residual |
| Hybrid | Joint field-data and PDE-residual training |

## Scope of the Current Conclusions

- A hard mask or hard boundary projection guarantees the prescribed boundary/initial jet, but does not guarantee that optimization selects the intended physical branch.
- The main supported role of HFB is to mitigate spectral bias and improve high-frequency generalization under sparse supervision. It should not be interpreted as a complete solution to PINO training dynamics.
- Data-only training provides stronger branch anchoring, but can produce poor derivative smoothness and large PDE residuals.
- Physics-only training constrains local physical structure, but does not adequately control branches or low-residual spurious solutions.
- Hybrid training is currently used to balance data-driven branch anchoring with physics-based smoothing and regularization, especially for nonlinear PDEs with several potential branches.

## Maintenance Recommendations

1. Add shared model architectures to `NeuroOperators.py` and add or update the corresponding tests.
2. Add general multidimensional PDE cases to `run_legacy_hf_pde_suite.py`.
3. Add native steady-1D cases to `run_legacy_hf_pde_steady1d_suite.py`.
4. Create a new result root for every new experiment; do not overwrite dated historical directories.
5. Keep training and figure regeneration separate when possible. If NPZ and history files already exist, prefer post-processing without retraining.
6. Formal result roots should retain the configuration, checkpoints, histories, metrics, field NPZ files, figure manifest, and an experiment summary.
