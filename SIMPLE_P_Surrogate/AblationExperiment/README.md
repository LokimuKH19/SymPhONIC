# AblationExperiment

Code and results for Section 3.2: ablation of spectral and local high-frequency branches on the strongly nonlinear steady 2D KdV benchmark.

## Experiment

Seven configurations (Core, spectral branch, local branch, full HFB, two parameter-matched FNO controls, and Core + high-frequency scaling), ten paired seeds (20260924–20260933), and two training objectives: **140 runs**. The fixed dataset contains 32 training, 8 validation, and 4 test functions on a 96 × 96 grid. Training uses 1200 epochs / 2400 updates. Protocol files record architecture sizes, data seed, checkpoint selection and hardware.

Core retains the shared HFB architecture, including both low-frequency paths. The high-frequency scaling control adapts the `featscale2` module from [Khodakarami et al.](https://github.com/SiaK4/HFS_ResUNet/blob/main/Models/ResUnet_HFS.py) to this core; it is a module comparison. Shared legacy dependencies are retained under `sources/` to preserve the original computation; this package runs only the KdV ablation.

## Contents

- `kdv_ablation.py`, `run_all.py`: architectures, dataset reconstruction, audit and training.
- `KdV_Physics/`, `KdV_Hybrid/`: original protocols and per-seed metrics, training histories, and saved predictions/reference fields.
- `absolute_metrics.py`: field RMSE, high-frequency RMSE and paired bootstrap intervals from saved predictions.
- `summarize.py`: original relative-error and timing summaries.
- `plot_absolute_results.py`: RMSE panels and radial error spectra.
- `plot_six_panel_means.py`: six panels of field RMSE, high-frequency RMSE and normalized PDE MSE, with open-diamond means.
- `absolute_error_revision/`, `six_panel_means_20260926/`: calculated results and figures.

Model checkpoints, duplicated generated training datasets, document-editing scripts and manuscript files are omitted. `sources/reference_sample_fields.npz` supports the dataset consistency check.

## Usage

Python 3.11; install dependencies with `pip install -r requirements.txt`. Original training used PyTorch 2.3.0 + CUDA 12.1 on an NVIDIA GeForce RTX 4060 Laptop GPU. Install the appropriate CUDA build of PyTorch for GPU training. Times New Roman should be installed to reproduce the figure typography (9 pt, equivalent to 12 CSS px).

Recalculate and plot the supplied results (no GPU required):

```bash
python summarize.py
python absolute_metrics.py
python plot_absolute_results.py
python plot_six_panel_means.py
```

Repeat all training runs (CUDA required):

```bash
python run_all.py
```

Fresh runs and reconstructed datasets are written to `reruns/`, preserving the supplied results; completed jobs there are skipped on restart. For summaries of fresh runs, use their `KdV_Physics` and `KdV_Hybrid` folders in a separate copy of this package in place of the supplied folders.

Field RMSE is computed per test function and averaged over four functions; high-frequency RMSE uses the same grid normalization after retaining Fourier indices with |kx| ≥ 12 or |ky| ≥ 12. Normalized PDE MSE is the stored `selected.pde_mse`, with no ×1000 display factor. Ten-seed spreads are sample standard deviations; paired bootstrap intervals use 20,000 resamples and are conditional on the fixed test functions. Training time counts synchronized optimization steps and excludes validation and preprocessing. See figure captions for the distinction between standard-deviation bars and boxplot whiskers.

Packaging changes only local import resolution, fresh-run output location and portable font discovery; numerical model/training logic and supplied experiment results are preserved. `SHA256SUMS.txt` lists the distributed files.
