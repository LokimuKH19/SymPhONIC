import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
MODE_DIRS = {
    "data": "DataDriven",
    "pde": "DataFree",
    "hybrid": "Hybrid",
}
VARIANTS = ("F_ATTN", "C_ATTN", "HF_ATTN")


def run_streamed(command, log_path):
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {log_path}")


def read_rows(root):
    rows = []
    for mode, folder in MODE_DIRS.items():
        path = root / folder / "global_comparison.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["training_mode"] = mode
                rows.append(row)
    return rows


def numeric(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def save_heatmap(root, rows, metric, title, filename):
    cases = []
    for row in rows:
        label = f"{row['pde']}/{row['case']} | {MODE_DIRS[row['training_mode']]}"
        if label not in cases:
            cases.append(label)
    matrix = np.full((len(cases), len(VARIANTS)), np.nan)
    for row in rows:
        label = f"{row['pde']}/{row['case']} | {MODE_DIRS[row['training_mode']]}"
        matrix[cases.index(label), VARIANTS.index(row["variant"])] = numeric(row, metric)
    display = np.log10(np.maximum(matrix, 1e-30))
    fig, ax = plt.subplots(figsize=(8.5, max(5.0, 0.48 * len(cases))), constrained_layout=True)
    image = ax.imshow(display, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(VARIANTS)), VARIANTS)
    ax.set_yticks(np.arange(len(cases)), cases)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=f"log10({metric})", shrink=0.85)
    fig.savefig(root / filename, dpi=220)
    plt.close(fig)


def save_attention_bars(root, rows, metric, ylabel, filename, log_scale=False):
    cases = sorted({f"{row['pde']}/{row['case']}" for row in rows})
    modes = tuple(MODE_DIRS)
    x = np.arange(len(cases))
    width = 0.085
    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    offset_index = 0
    for mode in modes:
        for variant in VARIANTS:
            values = []
            for case in cases:
                match = next(
                    row
                    for row in rows
                    if f"{row['pde']}/{row['case']}" == case
                    and row["training_mode"] == mode
                    and row["variant"] == variant
                )
                values.append(numeric(match, metric))
            offset = (offset_index - 4.0) * width
            ax.bar(x + offset, values, width, label=f"{MODE_DIRS[mode]} / {variant}")
            offset_index += 1
    ax.set_xticks(x, cases)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(root / filename, dpi=220)
    plt.close(fig)


def write_summary(root, rows):
    fields = sorted({key for row in rows for key in row})
    with (root / "cross_mode_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / "cross_mode_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    winners = []
    grouped = sorted({(row["pde"], row["case"], row["training_mode"]) for row in rows})
    for pde, case, mode in grouped:
        candidates = [
            row
            for row in rows
            if row["pde"] == pde and row["case"] == case and row["training_mode"] == mode
        ]
        winner = min(candidates, key=lambda row: numeric(row, "test_rel_l2"))
        winners.append(
            {
                "pde": pde,
                "case": case,
                "training_mode": mode,
                "winner": winner["variant"],
                "test_rel_l2": numeric(winner, "test_rel_l2"),
                "test_pde_mse": numeric(winner, "test_pde_mse"),
            }
        )
    with (root / "cross_mode_winners.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(winners[0]))
        writer.writeheader()
        writer.writerows(winners)

    lines = [
        "# Spectral Kernel Attention Comparison",
        "",
        "All variants share the same CFNO backbone and contain exactly one attention layer.",
        "Data-free selection uses validation PDE residual; data-driven selection uses validation data MSE; hybrid selection uses their weighted sum.",
        "",
        "| PDE | Case | Mode | Winner | Test relative L2 | Test PDE MSE |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in winners:
        lines.append(
            f"| {row['pde']} | {row['case']} | {MODE_DIRS[row['training_mode']]} | "
            f"{row['winner']} | {row['test_rel_l2']:.6e} | {row['test_pde_mse']:.6e} |"
        )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    save_heatmap(
        root,
        rows,
        "test_rel_l2",
        "Attention comparison: test relative L2",
        "cross_mode_test_rel_l2.png",
    )
    save_heatmap(
        root,
        rows,
        "test_pde_mse",
        "Attention comparison: normalized PDE residual",
        "cross_mode_test_pde_mse.png",
    )
    save_attention_bars(
        root,
        rows,
        "attention_gate",
        "learned attention gate",
        "cross_mode_attention_gate.png",
    )
    save_attention_bars(
        root,
        rows,
        "attention_source_to_state_rms",
        "attention source / state RMS",
        "cross_mode_attention_source_ratio.png",
        log_scale=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="KernelAttention_PDE_Comparison_20260718")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--target-params", type=int, default=500000)
    parser.add_argument("--attention-rank", type=int, default=8)
    parser.add_argument("--attention-gate-init", type=float, default=-2.0)
    parser.add_argument("--physics-weight", type=float, default=0.05)
    parser.add_argument("--profile", choices=("baseline", "high_nonlinear"), default="baseline")
    parser.add_argument("--pdes", default="Poisson,Wave")
    parser.add_argument(
        "--cases",
        default="Poisson/Steady1D,Poisson/Steady2D,Wave/Transient1D",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1000:
        raise ValueError("At least 1000 epochs are required.")

    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for mode, folder in MODE_DIRS.items():
        mode_root = output_root / folder
        command = [
            sys.executable,
            str(ROOT / "run_legacy_hf_pde_suite.py"),
            "--output-root",
            str(mode_root),
            "--epochs",
            str(args.epochs),
            "--grid",
            str(args.grid),
            "--target-params",
            str(args.target_params),
            "--attention-rank",
            str(args.attention_rank),
            "--attention-gate-init",
            str(args.attention_gate_init),
            "--variants",
            ",".join(VARIANTS),
            "--pdes",
            args.pdes,
            "--cases",
            args.cases,
            "--profile",
            args.profile,
            "--training-mode",
            mode,
            "--physics-weight",
            str(args.physics_weight),
        ]
        if args.resume:
            command.append("--resume")
        (mode_root / "command.txt").parent.mkdir(parents=True, exist_ok=True)
        (mode_root / "command.txt").write_text(subprocess.list2cmdline(command), encoding="utf-8")
        print(f"\n##### Starting {folder} #####", flush=True)
        run_streamed(command, output_root / f"{folder}.log")
    write_summary(output_root, read_rows(output_root))
    print(f"\nAttention comparison complete: {output_root}")


if __name__ == "__main__":
    main()
