from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


DEFAULT_ROOTS = (
    "LegacyHF_PDE_Benchmark_PINO_20260705",
    "LegacyHF_PDE_Benchmark_Hybrid_20260705",
    "LegacyHF_PDE_Benchmark_Normalized_20260705",
)

# First-row colors are the primary model/curve colors. Second-row colors are
# reserved for paired or auxiliary curves. The low-contrast third row is not used.
PRIMARY = ("#3c7fb1", "#e64532", "#44a05c", "#b53289")
SECONDARY = ("#90bfd5", "#f5a65b", "#a0d292", "#f296ac")
MODEL_ORDER = ("FNO", "CFNO", "HF_FNO", "HF_CFNO")
MODEL_COLOR = dict(zip(MODEL_ORDER, PRIMARY))
SERIES_COLOR = {
    "train": PRIMARY[0],
    "validation": PRIMARY[1],
    "test": PRIMARY[2],
    "theory": PRIMARY[0],
    "prediction": PRIMARY[1],
    "abs error": PRIMARY[2],
}
SERIES_STYLE = {"train": "-", "validation": "--", "test": ":"}
EPS = np.finfo(np.float64).tiny


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 10.5,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "figure.titlesize": 13,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.8,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


@dataclass(frozen=True)
class CaseInfo:
    pde: str
    case: str
    model: str

    @property
    def is_transient(self) -> bool:
        return "Transient1D" in self.case

    @property
    def is_steady_1d(self) -> bool:
        return "Steady1D" in self.case

    @property
    def is_steady_2d(self) -> bool:
        return "Steady2D" in self.case

    @property
    def domain_x(self) -> float:
        return 2.0 * math.pi if self.pde == "KdV" else 1.0

    @property
    def domain_y(self) -> float:
        return 2.0 * math.pi if self.pde == "KdV" else 1.0

    @property
    def final_time(self) -> float:
        if self.pde == "Wave":
            return 0.8
        if self.pde == "KdV":
            return 0.006 if self.case.startswith("HighNonlinear_") else 0.01
        if self.pde == "AllenCahn":
            return 0.5
        return 0.3

    @property
    def extent(self) -> tuple[float, float, float, float]:
        if self.is_transient:
            return (0.0, self.final_time, 0.0, self.domain_x)
        if self.is_steady_1d:
            return (0.0, 1.0, 0.0, self.domain_x)
        return (0.0, self.domain_y, 0.0, self.domain_x)

    @property
    def horizontal_label(self) -> str:
        if self.is_transient:
            return "t"
        if self.is_steady_1d:
            return "auxiliary coordinate"
        return "y"

    @property
    def spectral_label(self) -> str:
        return "radial spatial wavenumber" if self.is_steady_2d else "spatial wavenumber"


PANEL_SPECS = (
    ("objective", "train_loss", "val_loss", "test_loss", "total objective"),
    ("data", "train_mse", "val_mse", "test_mse", "data MSE"),
    ("physics", "train_pde_mse", "val_pde_mse", "test_pde_mse", "normalized PDE residual MSE"),
    (
        "absolute",
        "train_pde_mse_absolute",
        "val_pde_mse_absolute",
        "test_pde_mse_absolute",
        "absolute PDE residual MSE",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redraw legacy PDE benchmark plots without retraining.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--roots", nargs="*", default=list(DEFAULT_ROOTS))
    parser.add_argument("--model-limit", type=int, default=None, help="Process only N model folders for visual QA.")
    return parser.parse_args()


def read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty history: {path}")
    columns: dict[str, np.ndarray] = {}
    for key in rows[0]:
        values = []
        for row in rows:
            value = row.get(key, "")
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(float("nan"))
        columns[key] = np.asarray(values, dtype=np.float64)
    return columns


def loss_columns(history: dict[str, np.ndarray]) -> list[str]:
    return [key for key in history if key != "epoch" and ("loss" in key.lower() or "mse" in key.lower())]


def first_five_loss_scale(history: dict[str, np.ndarray]) -> tuple[float, list[str], int]:
    columns = loss_columns(history)
    count = min(5, len(history.get("epoch", [])))
    values = []
    for key in columns:
        current = np.abs(history[key][:count])
        values.extend(current[np.isfinite(current)].tolist())
    scale = max(values, default=1.0)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return float(scale), columns, count


def safe_log_values(values: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.abs(np.asarray(values, dtype=np.float64)) / max(float(scale), EPS)
    finite_positive = scaled[np.isfinite(scaled) & (scaled > 0.0)]
    floor = max(float(finite_positive.min()) * 0.1, 1e-16) if finite_positive.size else 1e-16
    return np.where(np.isfinite(scaled), np.maximum(scaled, floor), np.nan)


def save_figure(fig: plt.Figure, *paths: Path) -> int:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    plt.close(fig)
    return len(paths)


def style_axis(ax: plt.Axes, *, legend: bool = False) -> None:
    ax.grid(True, which="both", color="#c8c8c8", alpha=0.45, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=10.5)
    if legend:
        ax.legend(fontsize=10.5, frameon=True, edgecolor="#c0c0c0")


def add_colorbar(fig: plt.Figure, image, ax: plt.Axes, label: str) -> None:
    colorbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.035)
    colorbar.ax.tick_params(labelsize=10.5)
    colorbar.set_label(label, fontsize=10.5)


def case_info(model_dir: Path, root: Path) -> CaseInfo:
    relative = model_dir.relative_to(root)
    if len(relative.parts) != 3:
        raise ValueError(f"Expected PDE/case/model layout, got {relative}")
    return CaseInfo(*relative.parts)


def draw_prediction_fields(info: CaseInfo, fields: dict[str, np.ndarray], out_dir: Path) -> int:
    pred = np.asarray(fields["prediction"], dtype=np.float64)
    truth = np.asarray(fields["exact"], dtype=np.float64)
    error = np.abs(pred - truth)
    vmin = float(min(np.nanmin(pred), np.nanmin(truth)))
    vmax = float(max(np.nanmax(pred), np.nanmax(truth)))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.35), constrained_layout=True)
    entries = (
        (pred, "Prediction", "viridis", vmin, vmax, "u"),
        (truth, "Theory", "viridis", vmin, vmax, "u"),
        (error, "Absolute error", "magma", None, None, "|prediction - theory|"),
    )
    for ax, (array, title, cmap, lower, upper, cbar_label) in zip(axes, entries):
        image = ax.imshow(
            array,
            origin="lower",
            aspect="auto",
            extent=info.extent,
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel(info.horizontal_label, fontsize=12)
        ax.set_ylabel("x", fontsize=12)
        ax.tick_params(labelsize=10.5)
        add_colorbar(fig, image, ax, cbar_label)
    fig.suptitle(f"{info.pde} / {info.case} / {info.model}")
    count = save_figure(fig, out_dir / "prediction_exact_error.png")

    if info.is_steady_1d:
        center = pred.shape[1] // 2
        x = np.linspace(0.0, info.domain_x, pred.shape[0])
        fig, ax = plt.subplots(figsize=(7.6, 4.5), constrained_layout=True)
        ax.plot(x, truth[:, center], label="Theory", color=PRIMARY[0])
        ax.plot(x, pred[:, center], label="Prediction", color=PRIMARY[1], linestyle="--")
        ax.plot(x, error[:, center], label="Absolute error", color=PRIMARY[2], linestyle=":")
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("u", fontsize=12)
        ax.set_title(f"{info.pde} / {info.case} / {info.model} center line")
        style_axis(ax, legend=True)
        count += save_figure(fig, out_dir / "prediction_exact_error_line.png")
    return count


def draw_residual_fields(info: CaseInfo, fields: dict[str, np.ndarray], out_dir: Path) -> int:
    pred = np.asarray(fields["residual_prediction"], dtype=np.float64)
    truth = np.asarray(fields["residual_exact"], dtype=np.float64)
    finite = np.concatenate([np.abs(pred[np.isfinite(pred)]), np.abs(truth[np.isfinite(truth)])])
    vmax = max(float(np.percentile(finite, 99.0)) if finite.size else 0.0, 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.35), constrained_layout=True)
    for ax, array, title in (
        (axes[0], pred, "Prediction residual"),
        (axes[1], truth, "Theory residual"),
    ):
        image = ax.imshow(
            array,
            origin="lower",
            aspect="auto",
            extent=info.extent,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel(info.horizontal_label, fontsize=12)
        ax.set_ylabel("x", fontsize=12)
        ax.tick_params(labelsize=10.5)
        add_colorbar(fig, image, ax, "PDE residual")
    fig.suptitle(f"{info.pde} / {info.case} / {info.model}")
    return save_figure(fig, out_dir / "residual_distribution.png")


def radial_shell_mean(values: np.ndarray) -> np.ndarray:
    nx, ny = values.shape
    kx = np.fft.fftshift(np.fft.fftfreq(nx) * nx)
    ky = np.fft.fftshift(np.fft.fftfreq(ny) * ny)
    radius = np.rint(np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)).astype(np.int64)
    sums = np.bincount(radius.ravel(), weights=values.ravel())
    counts = np.bincount(radius.ravel())
    return sums / np.maximum(counts, 1)


def spectral_components(info: CaseInfo, pred: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if info.is_steady_2d:
        pred_ft = np.fft.fftshift(np.fft.fft2(pred, norm="forward"))
        truth_ft = np.fft.fftshift(np.fft.fft2(truth, norm="forward"))
        pred_mag = radial_shell_mean(np.abs(pred_ft))
        truth_mag = radial_shell_mean(np.abs(truth_ft))
        coefficient_error = radial_shell_mean(np.abs(pred_ft - truth_ft))
    else:
        # For transient 1D, transform x at every stored time and average over time.
        # For steady 1D, the same operation averages over the replicated helper axis.
        pred_ft = np.fft.rfft(pred, axis=0, norm="forward")
        truth_ft = np.fft.rfft(truth, axis=0, norm="forward")
        pred_mag = np.mean(np.abs(pred_ft), axis=1)
        truth_mag = np.mean(np.abs(truth_ft), axis=1)
        coefficient_error = np.mean(np.abs(pred_ft - truth_ft), axis=1)
    count = min(len(pred_mag), len(truth_mag), len(coefficient_error))
    return {
        "component": np.arange(count, dtype=np.int64),
        "prediction_magnitude": pred_mag[:count],
        "theory_magnitude": truth_mag[:count],
        "absolute_coefficient_error": coefficient_error[:count],
    }


def write_spectral_csv(path: Path, spectrum: dict[str, np.ndarray]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(spectrum.keys())
        for row in zip(*(spectrum[key] for key in spectrum)):
            writer.writerow(row)


def draw_spectrum(info: CaseInfo, fields: dict[str, np.ndarray], out_dir: Path) -> tuple[int, dict[str, np.ndarray]]:
    spectrum = spectral_components(info, fields["prediction"], fields["exact"])
    k = spectrum["component"]
    fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
    for key, label, color, style in (
        ("theory_magnitude", "Theory magnitude", PRIMARY[0], "-"),
        ("prediction_magnitude", "Prediction magnitude", PRIMARY[1], "--"),
        ("absolute_coefficient_error", "Absolute coefficient error", PRIMARY[2], ":"),
    ):
        ax.semilogy(k, np.maximum(spectrum[key], EPS), label=label, color=color, linestyle=style)
    ax.set_xlabel(info.spectral_label, fontsize=12)
    ax.set_ylabel("mean Fourier coefficient magnitude", fontsize=12)
    ax.set_title(f"{info.pde} / {info.case} / {info.model} spectrum")
    style_axis(ax, legend=True)
    count = save_figure(fig, out_dir / "spectrum_analysis.png")

    fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
    ax.semilogy(
        k,
        np.maximum(spectrum["absolute_coefficient_error"], EPS),
        color=PRIMARY[3],
        label="|prediction FFT - theory FFT|",
    )
    ax.set_xlabel(info.spectral_label, fontsize=12)
    ax.set_ylabel("absolute Fourier coefficient error", fontsize=12)
    ax.set_title(f"{info.pde} / {info.case} / {info.model} spectral error")
    style_axis(ax, legend=True)
    count += save_figure(fig, out_dir / "frequency_component_abs_error.png")
    write_spectral_csv(out_dir / "frequency_component_abs_error.csv", spectrum)
    return count, spectrum


def available_panels(history: dict[str, np.ndarray]) -> list[tuple[str, str, str, str, str]]:
    return [panel for panel in PANEL_SPECS if panel[1] in history and panel[2] in history]


def draw_history_panel(
    ax: plt.Axes,
    history: dict[str, np.ndarray],
    panel: tuple[str, str, str, str, str],
    scale: float,
) -> None:
    _, train_key, val_key, test_key, ylabel = panel
    epochs = history["epoch"]
    for label, key in (("train", train_key), ("validation", val_key), ("test", test_key)):
        if key not in history:
            continue
        ax.semilogy(
            epochs,
            safe_log_values(history[key], scale),
            label=label,
            color=SERIES_COLOR[label],
            linestyle=SERIES_STYLE[label],
        )
    ax.set_xlabel("epoch", fontsize=12)
    ax.set_ylabel(f"normalized {ylabel}", fontsize=12)
    style_axis(ax, legend=True)


def draw_model_convergence(
    info: CaseInfo,
    history: dict[str, np.ndarray],
    scale: float,
    out_dir: Path,
) -> int:
    panels = available_panels(history)
    if not panels:
        return 0
    fig, axes = plt.subplots(1, len(panels), figsize=(5.7 * len(panels), 4.45), constrained_layout=True)
    for ax, panel in zip(np.atleast_1d(axes), panels):
        ax.set_title(panel[0])
        draw_history_panel(ax, history, panel, scale)
    fig.suptitle(f"{info.pde} / {info.case} / {info.model}; first-five scale = {scale:.3e}")
    count = save_figure(fig, out_dir / "training_convergence.png")

    for panel in panels:
        suffix = panel[0]
        fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
        draw_history_panel(ax, history, panel, scale)
        ax.set_title(f"{info.pde} / {info.case} / {info.model} / {suffix}")
        paths = [out_dir / f"training_convergence_{suffix}.png"]
        if suffix == "physics":
            paths.append(out_dir / "training_convergence_normalized.png")
        count += save_figure(fig, *paths)
    return count


def load_metrics(model_dir: Path) -> dict:
    path = model_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def model_label(model_dir: Path) -> str:
    metrics = load_metrics(model_dir)
    params = metrics.get("params")
    return f"{model_dir.name} ({int(params):,})" if params is not None else model_dir.name


def draw_case_comparison(case_dir: Path, models: list[dict]) -> int:
    if not models:
        return 0
    shared_scale = max(float(item["scale"]) for item in models)
    available = []
    for panel in PANEL_SPECS:
        if any(panel[2] in item["history"] for item in models):
            available.append(panel)
    count = 0

    def draw_comparison_panel(ax: plt.Axes, panel: tuple[str, str, str, str, str]) -> None:
        _, _, val_key, _, ylabel = panel
        for item in models:
            history = item["history"]
            if val_key not in history:
                continue
            model = item["model_dir"].name
            ax.semilogy(
                history["epoch"],
                safe_log_values(history[val_key], shared_scale),
                color=MODEL_COLOR.get(model, PRIMARY[0]),
                label=model_label(item["model_dir"]),
            )
        ax.set_xlabel("epoch", fontsize=12)
        ax.set_ylabel(f"normalized validation {ylabel}", fontsize=12)
        style_axis(ax, legend=True)

    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(5.9 * len(available), 4.75), constrained_layout=True)
        for ax, panel in zip(np.atleast_1d(axes), available):
            ax.set_title(panel[0])
            draw_comparison_panel(ax, panel)
        fig.suptitle(f"{case_dir.parent.name} / {case_dir.name}; shared first-five scale = {shared_scale:.3e}")
        count += save_figure(fig, case_dir / "comparison_convergence.png")

    for panel in available:
        suffix = panel[0]
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        draw_comparison_panel(ax, panel)
        ax.set_title(f"{case_dir.parent.name} / {case_dir.name} / {suffix}")
        paths = [case_dir / f"comparison_convergence_{suffix}.png"]
        if suffix == "physics":
            paths.append(case_dir / "comparison_convergence_normalized.png")
        count += save_figure(fig, *paths)

    names = []
    rel_l2 = []
    colors = []
    for item in models:
        metrics = load_metrics(item["model_dir"])
        value = metrics.get("test_rel_l2")
        if value is None:
            continue
        model = item["model_dir"].name
        names.append(model)
        rel_l2.append(max(float(value), EPS))
        colors.append(MODEL_COLOR.get(model, PRIMARY[0]))
    if rel_l2:
        fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
        ax.bar(names, rel_l2, color=colors, edgecolor="#404040", linewidth=0.6)
        ax.set_yscale("log")
        ax.set_xlabel("model", fontsize=12)
        ax.set_ylabel("test relative L2", fontsize=12)
        ax.set_title(f"{case_dir.parent.name} / {case_dir.name} model comparison")
        style_axis(ax)
        count += save_figure(fig, case_dir / "comparison_test_rel_l2.png")

    spectral_models = [item for item in models if item.get("spectrum") is not None]
    if spectral_models:
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        for item in spectral_models:
            spectrum = item["spectrum"]
            model = item["model_dir"].name
            ax.semilogy(
                spectrum["component"],
                np.maximum(spectrum["absolute_coefficient_error"], EPS),
                color=MODEL_COLOR.get(model, PRIMARY[0]),
                label=model_label(item["model_dir"]),
            )
        info = spectral_models[0]["info"]
        ax.set_xlabel(info.spectral_label, fontsize=12)
        ax.set_ylabel("absolute Fourier coefficient error", fontsize=12)
        ax.set_title(f"{case_dir.parent.name} / {case_dir.name} spectral error comparison")
        style_axis(ax, legend=True)
        count += save_figure(fig, case_dir / "comparison_frequency_component_abs_error.png")

    metadata = {
        "normalization": "All plotted loss components are divided by the shared case scale.",
        "first_epochs": min(int(item["first_count"]) for item in models),
        "model_first_five_scales": {item["model_dir"].name: item["scale"] for item in models},
        "shared_case_scale": shared_scale,
    }
    (case_dir / "comparison_normalization.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return count


def draw_equation_spectral_comparison(pde_dir: Path, cases: dict[Path, list[dict]]) -> int:
    available = [(case_dir, models) for case_dir, models in sorted(cases.items()) if any(m.get("spectrum") is not None for m in models)]
    if not available:
        return 0
    columns = min(3, len(available))
    rows = int(math.ceil(len(available) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6.1 * columns, 4.45 * rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (case_dir, models) in zip(axes_flat, available):
        for item in models:
            spectrum = item.get("spectrum")
            if spectrum is None:
                continue
            model = item["model_dir"].name
            ax.semilogy(
                spectrum["component"],
                np.maximum(spectrum["absolute_coefficient_error"], EPS),
                color=MODEL_COLOR.get(model, PRIMARY[0]),
                label=model,
            )
        info = models[0]["info"]
        ax.set_title(case_dir.name)
        ax.set_xlabel(info.spectral_label, fontsize=12)
        ax.set_ylabel("absolute Fourier coefficient error", fontsize=12)
        style_axis(ax, legend=True)
    for ax in axes_flat[len(available) :]:
        ax.remove()
    fig.suptitle(f"{pde_dir.name}: frequency-component error comparison")
    return save_figure(fig, pde_dir / "frequency_component_error_comparison.png")


def draw_global_heatmap(root: Path, all_models: list[dict]) -> int:
    records = []
    for item in all_models:
        metrics = load_metrics(item["model_dir"])
        value = metrics.get("test_rel_l2")
        if value is None:
            continue
        info = item["info"]
        records.append((f"{info.pde}/{info.case}", info.model, max(float(value), EPS)))
    if not records:
        return 0
    cases = sorted({record[0] for record in records})
    models = [model for model in MODEL_ORDER if any(record[1] == model for record in records)]
    matrix = np.full((len(cases), len(models)), np.nan)
    case_index = {name: index for index, name in enumerate(cases)}
    model_index = {name: index for index, name in enumerate(models)}
    for case, model, value in records:
        if model in model_index:
            matrix[case_index[case], model_index[model]] = value
    finite = matrix[np.isfinite(matrix)]
    vmin = max(float(finite.min()) if finite.size else 1e-6, EPS)
    vmax = max(float(finite.max()) if finite.size else 1.0, vmin * (1.0 + 1e-12))
    fig_height = max(6.0, 0.36 * len(cases) + 2.2)
    fig, ax = plt.subplots(figsize=(8.2, fig_height), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_xticks(np.arange(len(models)), labels=models)
    ax.set_yticks(np.arange(len(cases)), labels=cases)
    ax.set_xlabel("model", fontsize=12)
    ax.set_ylabel("PDE / case", fontsize=12)
    ax.set_title("Global test relative L2 comparison")
    ax.tick_params(labelsize=10.5)
    add_colorbar(fig, image, ax, "test relative L2")
    return save_figure(fig, root / "global_test_rel_l2_heatmap.png")


def process_root(root: Path, model_limit: int | None) -> dict:
    histories = sorted(root.rglob("history.csv"))
    if model_limit is not None:
        histories = histories[:model_limit]
    generated = 0
    errors = []
    all_models: list[dict] = []
    case_models: dict[Path, list[dict]] = {}
    for index, history_path in enumerate(histories, start=1):
        model_dir = history_path.parent
        try:
            info = case_info(model_dir, root)
            history = read_history(history_path)
            scale, columns, first_count = first_five_loss_scale(history)
            with np.load(model_dir / "sample_fields.npz") as archive:
                fields = {key: np.array(archive[key], copy=True) for key in archive.files}
            generated += draw_prediction_fields(info, fields, model_dir)
            generated += draw_residual_fields(info, fields, model_dir)
            spectrum_count, spectrum = draw_spectrum(info, fields, model_dir)
            generated += spectrum_count
            generated += draw_model_convergence(info, history, scale, model_dir)
            metadata = {
                "normalization": "Every plotted loss component is divided by the largest absolute loss/MSE component in the first five epochs (or all available epochs when fewer than five).",
                "first_epochs_used": first_count,
                "loss_columns": columns,
                "model_first_five_scale": scale,
            }
            (model_dir / "plot_normalization.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            item = {
                "model_dir": model_dir,
                "info": info,
                "history": history,
                "scale": scale,
                "first_count": first_count,
                "spectrum": spectrum,
            }
            all_models.append(item)
            case_models.setdefault(model_dir.parent, []).append(item)
            print(f"[{root.name}] {index}/{len(histories)} {info.pde}/{info.case}/{info.model}", flush=True)
        except Exception as exc:  # Continue so one damaged artifact cannot hide all other results.
            errors.append({"model_dir": str(model_dir), "error": repr(exc)})
            print(f"ERROR {model_dir}: {exc!r}", flush=True)

    for case_dir, models in sorted(case_models.items()):
        models.sort(key=lambda item: MODEL_ORDER.index(item["model_dir"].name) if item["model_dir"].name in MODEL_ORDER else 99)
        generated += draw_case_comparison(case_dir, models)

    pde_cases: dict[Path, dict[Path, list[dict]]] = {}
    for case_dir, models in case_models.items():
        pde_cases.setdefault(case_dir.parent, {})[case_dir] = models
    for pde_dir, cases in sorted(pde_cases.items()):
        generated += draw_equation_spectral_comparison(pde_dir, cases)
    generated += draw_global_heatmap(root, all_models)

    report = {
        "root": str(root),
        "models_processed": len(all_models),
        "cases_processed": len(case_models),
        "figures_written": generated,
        "errors": errors,
        "font": "Times New Roman",
        "axis_label_size": 12,
        "legend_and_colorbar_size": 10.5,
        "primary_palette": list(PRIMARY),
        "secondary_palette": list(SECONDARY),
        "loss_normalization": "per-model first-five all-loss maximum; case comparisons use the largest model scale",
    }
    (root / "redraw_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def resolve_roots(base_dir: Path, roots: Iterable[str]) -> list[Path]:
    resolved = []
    for value in roots:
        path = Path(value)
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        resolved.append(path)
    return resolved


def main() -> None:
    args = parse_args()
    roots = resolve_roots(args.base_dir.resolve(), args.roots)
    reports = [process_root(root, args.model_limit) for root in roots]
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
