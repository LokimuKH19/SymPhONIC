from __future__ import annotations

import argparse
import csv
import json
import math
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

import redraw_legacy_pde_results as redraw
import run_legacy_hf_pde_suite as benchmark


ACTIVATIONS = ("gelu", "identity")
VARIANTS = ("HF_FNO", "HF_CFNO")
ACTIVATION_COLORS = {"gelu": "#3c7fb1", "identity": "#e64532"}
ACTIVATION_SHADES = {"gelu": "#90bfd5", "identity": "#f5a65b"}
MODEL_COLORS = {"HF_FNO": "#44a05c", "HF_CFNO": "#b53289"}
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
        "lines.linewidth": 1.9,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired LocalHighPassBlock2d inner-activation ablation.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("LocalHighPassActivation_Ablation_Steady2D_20260726"),
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--test-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--high-modes", type=int, default=16)
    parser.add_argument("--target-params", type=int, default=500000)
    parser.add_argument("--seeds", default="20260726,20260727,20260728")
    parser.add_argument("--training-mode", choices=("data", "pde", "hybrid"), default="hybrid")
    parser.add_argument("--physics-weight", type=float, default=0.05)
    parser.add_argument("--equivalence-margin", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1000:
        raise ValueError("Activation ablation must run at least 1000 epochs.")
    args.seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if len(args.seeds) < 3:
        raise ValueError("Use at least three paired seeds for the activation ablation.")
    if not 0.0 < args.equivalence_margin < 1.0:
        raise ValueError("--equivalence-margin must be between 0 and 1.")
    return args


def suite_namespace(args: argparse.Namespace, *, profile: str, cases: str) -> Namespace:
    return Namespace(
        output_root=str(args.output_root),
        epochs=args.epochs,
        grid=args.grid,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        lr=args.lr,
        modes=args.modes,
        high_modes=args.high_modes,
        depth=4,
        target_params=args.target_params,
        seed=args.seeds[0],
        variants=",".join(VARIANTS),
        attention_rank=8,
        attention_gate_init=-2.0,
        pdes="",
        cases=cases,
        profile=profile,
        training_mode=args.training_mode,
        physics_weight=args.physics_weight,
        local_activation="gelu",
        append_existing_global=False,
        resume=args.resume,
        cpu=args.cpu,
    )


def selected_specs(args: argparse.Namespace) -> list[benchmark.CaseSpec]:
    baseline = suite_namespace(args, profile="baseline", cases="Poisson/Steady2D")
    nonlinear = suite_namespace(args, profile="high_nonlinear", cases="Burgers/HighNonlinear_Steady2D")
    specs = benchmark.build_case_specs(baseline) + benchmark.build_case_specs(nonlinear)
    if len(specs) != 2:
        raise RuntimeError(f"Expected two steady 2D cases, got {[spec.slug for spec in specs]}")
    return specs


def metric_value(metrics: dict, key: str) -> float:
    value = metrics.get(key, float("nan"))
    return float(value) if value is not None else float("nan")


def spectral_relative_error(npz_path: Path) -> float:
    with np.load(npz_path) as archive:
        pred = np.asarray(archive["prediction"], dtype=np.float64)
        truth = np.asarray(archive["exact"], dtype=np.float64)
    pred_ft = np.fft.fft2(pred, norm="ortho")
    truth_ft = np.fft.fft2(truth, norm="ortho")
    numerator = np.linalg.norm((pred_ft - truth_ft).ravel())
    denominator = max(float(np.linalg.norm(truth_ft.ravel())), EPS)
    return float(numerator / denominator)


def postprocess_run(spec: benchmark.CaseSpec, variant: str, activation: str, seed: int, out_dir: Path) -> float:
    history = redraw.read_history(out_dir / "history.csv")
    scale, columns, first_count = redraw.first_five_loss_scale(history)
    with np.load(out_dir / "sample_fields.npz") as archive:
        fields = {key: np.array(archive[key], copy=True) for key in archive.files}
    label = f"{variant} / {activation} / seed {seed}"
    info = redraw.CaseInfo(spec.pde, spec.case, label)
    redraw.draw_prediction_fields(info, fields, out_dir)
    redraw.draw_residual_fields(info, fields, out_dir)
    redraw.draw_spectrum(info, fields, out_dir)
    redraw.draw_model_convergence(info, history, scale, out_dir)
    metadata = {
        "normalization": "All loss curves use the maximum over every loss/MSE component in the first five epochs.",
        "first_epochs_used": first_count,
        "loss_columns": columns,
        "model_first_five_scale": scale,
    }
    (out_dir / "plot_normalization.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return scale


def train_or_load(
    args: argparse.Namespace,
    spec: benchmark.CaseSpec,
    variant: str,
    activation: str,
    paired_seed: int,
    model_seed: int,
    width: int,
    data: dict[str, torch.Tensor],
    out_dir: Path,
    device: torch.device,
) -> tuple[dict, float]:
    metrics_path = out_dir / "metrics.json"
    if args.resume and metrics_path.exists() and (out_dir / "history.csv").exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(f"Resume {spec.slug}/{variant}/{activation}/seed_{paired_seed}", flush=True)
    else:
        metrics = benchmark.train_one_model(
            spec,
            variant,
            width,
            data,
            out_dir,
            epochs=args.epochs,
            device=device,
            seed=model_seed,
            training_mode=args.training_mode,
            physics_weight=args.physics_weight,
            local_activation=activation,
        )
    metrics.update(
        {
            "pde": spec.pde,
            "case": spec.case,
            "paired_seed": paired_seed,
            "model_seed": model_seed,
            "local_activation": activation,
            "spectral_relative_error": spectral_relative_error(out_dir / "sample_fields.npz"),
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    scale = postprocess_run(spec, variant, activation, paired_seed, out_dir)
    return metrics, scale


def row_from_metrics(metrics: dict, scale: float) -> dict:
    return {
        "pde": metrics["pde"],
        "case": metrics["case"],
        "variant": metrics["variant"],
        "activation": metrics["local_activation"],
        "paired_seed": metrics["paired_seed"],
        "model_seed": metrics["model_seed"],
        "params": metrics["params"],
        "width": metrics["width"],
        "best_epoch": metrics["best"]["epoch"],
        "test_rel_l2": metric_value(metrics, "test_rel_l2"),
        "test_mse": metric_value(metrics, "test_mse"),
        "test_pde_mse": metric_value(metrics, "test_pde_mse"),
        "test_pde_mse_absolute": metric_value(metrics, "test_pde_mse_absolute"),
        "residual_rms": metric_value(metrics.get("residual_prediction", {}), "rms"),
        "spectral_relative_error": metric_value(metrics, "spectral_relative_error"),
        "first_five_loss_scale": scale,
        "elapsed_s": metric_value(metrics, "elapsed_s"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def paired_arrays(rows: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = float(row[metric])
        if np.isfinite(value) and value > 0.0:
            by_seed[int(row["paired_seed"])][row["activation"]] = value
    complete = [(seed, values) for seed, values in sorted(by_seed.items()) if all(name in values for name in ACTIVATIONS)]
    gelu = np.asarray([values["gelu"] for _, values in complete], dtype=np.float64)
    identity = np.asarray([values["identity"] for _, values in complete], dtype=np.float64)
    return gelu, identity, [seed for seed, _ in complete]


def equivalence_statistics(gelu: np.ndarray, identity: np.ndarray, margin: float) -> dict:
    if len(gelu) < 2:
        return {"n": int(len(gelu))}
    log_ratio = np.log(identity / gelu)
    mean = float(log_ratio.mean())
    sem = float(stats.sem(log_ratio))
    if sem == 0.0:
        ci_low = ci_high = mean
    else:
        ci_low, ci_high = stats.t.interval(0.95, len(log_ratio) - 1, loc=mean, scale=sem)
    lower = math.log(1.0 - margin)
    upper = math.log(1.0 + margin)
    p_lower = float(stats.ttest_1samp(log_ratio, lower, alternative="greater").pvalue)
    p_upper = float(stats.ttest_1samp(log_ratio, upper, alternative="less").pvalue)
    difference_p = float(stats.ttest_1samp(log_ratio, 0.0).pvalue)
    relative = identity / gelu - 1.0
    return {
        "n": int(len(log_ratio)),
        "gelu_mean": float(gelu.mean()),
        "gelu_std": float(gelu.std(ddof=1)),
        "identity_mean": float(identity.mean()),
        "identity_std": float(identity.std(ddof=1)),
        "identity_minus_gelu_mean_percent": float(relative.mean() * 100.0),
        "identity_minus_gelu_median_percent": float(np.median(relative) * 100.0),
        "mean_absolute_paired_change_percent": float(np.mean(np.abs(relative)) * 100.0),
        "log_ratio_95ci_percent": [float((math.exp(ci_low) - 1.0) * 100.0), float((math.exp(ci_high) - 1.0) * 100.0)],
        "paired_difference_p": difference_p,
        "equivalence_margin_percent": margin * 100.0,
        "equivalence_tost_p": max(p_lower, p_upper),
        "equivalent_within_margin": bool(max(p_lower, p_upper) < 0.05),
    }


def group_rows(rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["pde"], row["case"], row["variant"])].append(row)
    return dict(grouped)


def draw_paired_metric(root: Path, grouped: dict[tuple[str, str, str], list[dict]]) -> None:
    groups = sorted(grouped)
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 4.7), constrained_layout=True)
    for ax, key in zip(np.atleast_1d(axes), groups):
        gelu, identity, seeds = paired_arrays(grouped[key], "test_rel_l2")
        for index, seed in enumerate(seeds):
            ax.plot(
                [0, 1],
                [gelu[index], identity[index]],
                color="#90bfd5",
                marker="o",
                markersize=5,
                alpha=0.85,
                label="paired seeds" if index == 0 else None,
            )
        ax.scatter([0], [gelu.mean()], color=ACTIVATION_COLORS["gelu"], marker="D", s=55, zorder=4, label="GELU mean")
        ax.scatter([1], [identity.mean()], color=ACTIVATION_COLORS["identity"], marker="D", s=55, zorder=4, label="Identity mean")
        ax.set_xticks([0, 1], ["GELU", "Identity"])
        ax.set_yscale("log")
        ax.set_xlabel("inner activation", fontsize=12)
        ax.set_ylabel("test relative L2", fontsize=12)
        ax.set_title(f"{key[0]} / {key[1]}\n{key[2]}")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend(fontsize=10.5)
    fig.suptitle("Local high-pass inner-activation paired ablation")
    fig.savefig(root / "activation_ablation_test_rel_l2.png")
    plt.close(fig)


def draw_relative_change(root: Path, grouped: dict[tuple[str, str, str], list[dict]], margin: float) -> None:
    labels = []
    means = []
    errors = []
    colors = []
    for key in sorted(grouped):
        gelu, identity, _ = paired_arrays(grouped[key], "test_rel_l2")
        relative = (identity / gelu - 1.0) * 100.0
        labels.append(f"{key[0]}\n{key[1]}\n{key[2]}")
        means.append(float(relative.mean()))
        errors.append(float(relative.std(ddof=1)) if len(relative) > 1 else 0.0)
        colors.append(MODEL_COLORS[key[2]])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 5.1), constrained_layout=True)
    ax.bar(x, means, yerr=errors, color=colors, edgecolor="#404040", linewidth=0.6, capsize=4)
    ax.axhspan(-margin * 100.0, margin * 100.0, color="#90bfd5", alpha=0.18, label=f"practical margin +/-{margin * 100:.0f}%")
    ax.axhline(0.0, color="#404040", linewidth=0.9)
    ax.set_xticks(x, labels)
    ax.set_xlabel("case and model", fontsize=12)
    ax.set_ylabel("Identity minus GELU test L2 (%)", fontsize=12)
    ax.set_title("Paired activation effect; mean +/- seed standard deviation")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=10.5)
    fig.savefig(root / "activation_ablation_relative_change.png")
    plt.close(fig)


def convergence_key(history: dict[str, np.ndarray], training_mode: str) -> str:
    if training_mode == "data":
        return "val_mse"
    if training_mode == "pde":
        return "val_pde_mse"
    return "val_loss"


def draw_convergence(root: Path, rows: list[dict], grouped: dict[tuple[str, str, str], list[dict]], training_mode: str) -> None:
    row_lookup = {
        (row["pde"], row["case"], row["variant"], row["activation"], int(row["paired_seed"])): row
        for row in rows
    }
    groups = sorted(grouped)
    fig, axes = plt.subplots(1, len(groups), figsize=(5.4 * len(groups), 4.8), constrained_layout=True)
    for ax, key in zip(np.atleast_1d(axes), groups):
        group = grouped[key]
        shared_scale = max(float(row["first_five_loss_scale"]) for row in group)
        for activation in ACTIVATIONS:
            curves = []
            epochs = None
            for seed in sorted({int(row["paired_seed"]) for row in group}):
                row = row_lookup[(key[0], key[1], key[2], activation, seed)]
                run_dir = Path(row["run_dir"])
                history = redraw.read_history(run_dir / "history.csv")
                metric = convergence_key(history, training_mode)
                epochs = history["epoch"]
                curves.append(redraw.safe_log_values(history[metric], shared_scale))
            stack = np.stack(curves)
            mean = stack.mean(axis=0)
            std = stack.std(axis=0, ddof=1 if len(stack) > 1 else 0)
            lower = np.maximum(mean - std, mean * 0.05)
            upper = mean + std
            ax.semilogy(epochs, mean, color=ACTIVATION_COLORS[activation], label=activation.upper())
            ax.fill_between(epochs, lower, upper, color=ACTIVATION_SHADES[activation], alpha=0.28)
        ax.set_xlabel("epoch", fontsize=12)
        ax.set_ylabel("normalized validation loss", fontsize=12)
        ax.set_title(f"{key[0]} / {key[1]}\n{key[2]}")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend(fontsize=10.5)
    fig.suptitle("Activation ablation convergence; shared first-five scale per panel")
    fig.savefig(root / "activation_ablation_convergence.png")
    plt.close(fig)


def draw_frequency_comparison(root: Path, grouped: dict[tuple[str, str, str], list[dict]]) -> None:
    groups = sorted(grouped)
    fig, axes = plt.subplots(1, len(groups), figsize=(5.4 * len(groups), 4.8), constrained_layout=True)
    for ax, key in zip(np.atleast_1d(axes), groups):
        for activation in ACTIVATIONS:
            spectra = []
            components = None
            activation_rows = sorted(
                (row for row in grouped[key] if row["activation"] == activation),
                key=lambda row: int(row["paired_seed"]),
            )
            for row in activation_rows:
                path = Path(row["run_dir"]) / "frequency_component_abs_error.csv"
                with path.open("r", encoding="utf-8", newline="") as handle:
                    data = list(csv.DictReader(handle))
                current_components = np.asarray([int(float(item["component"])) for item in data], dtype=np.int64)
                current_error = np.asarray([float(item["absolute_coefficient_error"]) for item in data], dtype=np.float64)
                if components is None:
                    components = current_components
                count = min(len(components), len(current_components), len(current_error))
                components = components[:count]
                spectra = [spectrum[:count] for spectrum in spectra]
                spectra.append(current_error[:count])
            if not spectra:
                continue
            stack = np.stack(spectra)
            mean = stack.mean(axis=0)
            std = stack.std(axis=0, ddof=1 if len(stack) > 1 else 0)
            lower = np.maximum(mean - std, mean * 0.05)
            upper = mean + std
            ax.semilogy(components, np.maximum(mean, EPS), color=ACTIVATION_COLORS[activation], label=activation.upper())
            ax.fill_between(components, lower, upper, color=ACTIVATION_SHADES[activation], alpha=0.28)
        ax.set_xlabel("radial spatial wavenumber", fontsize=12)
        ax.set_ylabel("absolute Fourier coefficient error", fontsize=12)
        ax.set_title(f"{key[0]} / {key[1]}\n{key[2]}")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend(fontsize=10.5)
    fig.suptitle("Local high-pass activation spectral-error ablation")
    fig.savefig(root / "activation_ablation_frequency_component_error.png")
    plt.close(fig)


def write_summary(root: Path, args: argparse.Namespace, rows: list[dict]) -> dict:
    grouped = group_rows(rows)
    group_stats = []
    all_gelu = []
    all_identity = []
    for key, values in sorted(grouped.items()):
        gelu, identity, seeds = paired_arrays(values, "test_rel_l2")
        result = {"pde": key[0], "case": key[1], "variant": key[2], "seeds": seeds}
        result.update(equivalence_statistics(gelu, identity, args.equivalence_margin))
        group_stats.append(result)
        all_gelu.extend(gelu.tolist())
        all_identity.extend(identity.tolist())
    aggregate = equivalence_statistics(
        np.asarray(all_gelu, dtype=np.float64),
        np.asarray(all_identity, dtype=np.float64),
        args.equivalence_margin,
    )
    completed_pairs = int(aggregate.get("n", 0))
    expected_pairs = len(grouped) * len(args.seeds)
    if aggregate.get("equivalent_within_margin"):
        conclusion = f"GELU and identity are statistically equivalent within +/-{args.equivalence_margin * 100:.0f}% for this ablation."
    elif aggregate.get("paired_difference_p", 1.0) < 0.05:
        conclusion = "The inner activation has a statistically detectable effect in this ablation."
    else:
        conclusion = (
            "No statistically significant inner-activation effect was observed in the completed paired runs; "
            "strict equivalence within the practical margin was not proven."
        )
    summary = {
        "design": {
            "cases": ["Poisson/Steady2D", "Burgers/HighNonlinear_Steady2D"],
            "models": list(VARIANTS),
            "activations": list(ACTIVATIONS),
            "paired_seeds": list(args.seeds),
            "epochs": args.epochs,
            "grid": args.grid,
            "training_mode": args.training_mode,
            "physics_weight": args.physics_weight,
            "controlled_difference": "Only LocalHighPassBlock2d inner activation changes; pointwise and all parameters remain active.",
        },
        "group_statistics": group_stats,
        "aggregate_statistics": aggregate,
        "completion": {
            "completed_runs": len(rows),
            "expected_runs": len(grouped) * len(args.seeds) * len(ACTIVATIONS),
            "completed_pairs": completed_pairs,
            "expected_pairs": expected_pairs,
            "stopped_early": completed_pairs < expected_pairs,
        },
        "conclusion": conclusion,
    }
    (root / "activation_ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Local high-pass activation ablation",
        "",
        conclusion,
        "",
        f"Completed {len(rows)} runs ({completed_pairs} paired observations); the planned design contained {expected_pairs * 2} runs.",
        "Testing was stopped early at the user's request.",
        "",
        "Both arms retain `depthwise -> pointwise -> mix`; only GELU versus identity changes.",
        f"All runs use {args.epochs} epochs, a {args.grid}x{args.grid} grid, paired initialization seeds, and {args.training_mode} training.",
        f"The equivalence margin is +/-{args.equivalence_margin * 100:.0f}% in test relative L2.",
        "",
        "See `activation_ablation_summary.json` and `activation_ablation_runs.csv` for exact values.",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    draw_paired_metric(root, grouped)
    draw_relative_change(root, grouped, args.equivalence_margin)
    draw_convergence(root, rows, grouped, args.training_mode)
    draw_frequency_comparison(root, grouped)
    return summary


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    specs = selected_specs(args)
    config = {
        "args": {**vars(args), "output_root": str(root), "seeds": list(args.seeds)},
        "device": str(device),
        "torch": torch.__version__,
        "cases": [spec.slug for spec in specs],
        "controlled_ablation": "local_activation=gelu versus identity; pointwise remains active",
        "plot_style": {
            "font": "Times New Roman",
            "axis_label_size": 12,
            "legend_and_colorbar_size": 10.5,
            "colors": ACTIVATION_COLORS,
        },
    }
    (root / "ablation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows = []
    for case_index, spec in enumerate(specs, start=1):
        split_args = suite_namespace(
            args,
            profile="high_nonlinear" if spec.case.startswith("HighNonlinear_") else "baseline",
            cases=f"{spec.pde}/{spec.case}",
        )
        data_seed = args.seeds[0] + case_index * 1009
        data = benchmark.split_data(spec, split_args, data_seed)
        widths = benchmark.match_widths(spec, VARIANTS)
        case_dir = root / spec.pde / spec.case
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "case_design.json").write_text(
            json.dumps(
                {
                    "spec": benchmark.asdict(spec),
                    "data_seed": data_seed,
                    "width_matching": widths,
                    "paired_seeds": list(args.seeds),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        for variant_index, variant in enumerate(VARIANTS, start=1):
            for paired_seed in args.seeds:
                model_seed = paired_seed + case_index * 1009 + variant_index * 97
                for activation in ACTIVATIONS:
                    out_dir = case_dir / variant / activation / f"seed_{paired_seed}"
                    print(
                        f"\n[{spec.slug}] {variant} activation={activation} paired_seed={paired_seed} device={device}",
                        flush=True,
                    )
                    metrics, scale = train_or_load(
                        args,
                        spec,
                        variant,
                        activation,
                        paired_seed,
                        model_seed,
                        widths[variant]["width"],
                        data,
                        out_dir,
                        device,
                    )
                    row = row_from_metrics(metrics, scale)
                    row["run_dir"] = str(out_dir)
                    rows.append(row)
                    write_csv(root / "activation_ablation_runs.csv", rows)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    summary = write_summary(root, args, rows)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Activation ablation complete: {root}", flush=True)


if __name__ == "__main__":
    main()
