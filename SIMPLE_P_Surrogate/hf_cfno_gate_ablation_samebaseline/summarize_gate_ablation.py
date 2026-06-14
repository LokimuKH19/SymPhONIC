from __future__ import annotations

import math
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

CASES = [
    {
        "case": "old_hf_cfno",
        "gate_mode": "legacy",
        "boundary_mode": "legacy_mixed",
        "high_gate_init": -1.0,
        "gate_alignment_weight": 0.0,
    },
    {
        "case": "new_gate_weak",
        "gate_mode": "subgrid",
        "boundary_mode": "replicate",
        "high_gate_init": -2.0,
        "gate_alignment_weight": 1e-2,
    },
    {
        "case": "new_gate_stronger",
        "gate_mode": "subgrid",
        "boundary_mode": "replicate",
        "high_gate_init": -1.0,
        "gate_alignment_weight": 0.05,
    },
    {
        "case": "new_gate_no_align",
        "gate_mode": "subgrid",
        "boundary_mode": "replicate",
        "high_gate_init": -2.0,
        "gate_alignment_weight": 0.0,
    },
    {
        "case": "new_gate_with_fuse_but_gated",
        "gate_mode": "subgrid_gated_fuse",
        "boundary_mode": "replicate",
        "high_gate_init": -2.0,
        "gate_alignment_weight": 1e-2,
    },
]


def find_one(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def load_history(path: Path) -> np.ndarray:
    history = np.genfromtxt(path, delimiter=",", names=True)
    if history.shape == ():
        history = np.array([history], dtype=history.dtype)
    return history


def ghia_metrics(path: Path) -> dict[str, float]:
    # The first header contains a comma and is not quoted by np.savetxt.
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    comps = data[:, 0].astype(int)
    errors = data[:, 4].astype(float)

    metrics = {
        "ghia_rmse": float(np.sqrt(np.mean(errors**2))),
        "ghia_mae": float(np.mean(np.abs(errors))),
    }
    for comp, label in [(0, "u_xmid"), (1, "v_ymid")]:
        mask = comps == comp
        if np.any(mask):
            comp_errors = errors[mask]
            metrics[f"ghia_{label}_rmse"] = float(np.sqrt(np.mean(comp_errors**2)))
            metrics[f"ghia_{label}_mae"] = float(np.mean(np.abs(comp_errors)))
        else:
            metrics[f"ghia_{label}_rmse"] = math.nan
            metrics[f"ghia_{label}_mae"] = math.nan
    return metrics


def fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6e}"
    return str(value)


def summarize_case(config: dict[str, object]) -> dict[str, object]:
    case = str(config["case"])
    out_dir = ROOT / case
    row: dict[str, object] = dict(config)
    row["output_dir"] = str(out_dir)

    history_path = find_one(out_dir, "*history.csv")
    ghia_path = find_one(out_dir, "*ghia_re100_centerline_error.csv")
    if history_path is None:
        row["status"] = "missing_history"
        return row

    history = load_history(history_path)
    pde = history["L_cont"] + history["L_mom"]
    loss = history["loss"]
    best_pde_idx = int(np.argmin(pde))
    best_loss_idx = int(np.argmin(loss))
    final = history[-1]

    row.update(
        {
            "status": "ok",
            "history_file": str(history_path),
            "final_iter": int(final["iter"]),
            "final_L_cont": float(final["L_cont"]),
            "final_L_mom": float(final["L_mom"]),
            "final_pde": float(final["L_cont"] + final["L_mom"]),
            "final_loss": float(final["loss"]),
            "final_L_gate": float(final["L_gate"]),
            "final_gate_mean": float(final["gate_mean"]),
            "final_gate_min": float(final["gate_min"]),
            "final_gate_max": float(final["gate_max"]),
            "final_spatial_gate_mean": float(final["spatial_gate_mean"]),
            "final_spatial_gate_min": float(final["spatial_gate_min"]),
            "final_spatial_gate_max": float(final["spatial_gate_max"]),
            "best_pde_iter": int(history["iter"][best_pde_idx]),
            "best_pde": float(pde[best_pde_idx]),
            "best_pde_loss": float(loss[best_pde_idx]),
            "best_pde_L_gate": float(history["L_gate"][best_pde_idx]),
            "best_loss_iter": int(history["iter"][best_loss_idx]),
            "best_loss": float(loss[best_loss_idx]),
            "best_loss_pde": float(pde[best_loss_idx]),
        }
    )

    if ghia_path is not None:
        row["ghia_file"] = str(ghia_path)
        row.update(ghia_metrics(ghia_path))
    else:
        row["ghia_file"] = ""
        row.update(
            {
                "ghia_rmse": math.nan,
                "ghia_mae": math.nan,
                "ghia_u_xmid_rmse": math.nan,
                "ghia_u_xmid_mae": math.nan,
                "ghia_v_ymid_rmse": math.nan,
                "ghia_v_ymid_mae": math.nan,
            }
        )
    return row


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    columns = [
        "case",
        "gate_mode",
        "high_gate_init",
        "gate_alignment_weight",
        "best_pde",
        "best_pde_iter",
        "final_pde",
        "final_loss",
        "ghia_rmse",
        "final_gate_mean",
        "final_spatial_gate_mean",
    ]
    lines = [
        "# HF-CFNO gate ablation summary",
        "",
        "All cases use the same entry point, seed, grid, width, depth, learning rate, and 5000 iterations unless the run script parameter is changed.",
        "",
        "|" + "|".join(columns) + "|",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(col, "")) for col in columns) + "|")
    lines.extend(
        [
            "",
            "Metric notes:",
            "",
            "- `best_pde` is `min(L_cont + L_mom)` over the run.",
            "- `final_loss` is `L_cont + L_mom + gate_alignment_weight * L_gate` at the final iteration.",
            "- `ghia_rmse` is computed from the saved Re=100 centerline comparison file.",
            "- `final_gate_mean` is the effective residual amplitude `lambda*g`; `final_spatial_gate_mean` is the physical spatial gate `g`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(rows: list[dict[str, object]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for row in ok_rows:
        history = load_history(Path(str(row["history_file"])))
        pde = history["L_cont"] + history["L_mom"]
        ax.semilogy(history["iter"], pde, label=str(row["case"]))
    ax.set_xlabel("iteration")
    ax.set_ylabel("PDE residual")
    ax.set_title("HF-CFNO gate ablation: PDE residual")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(ROOT / "AblationPDEResidual.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for row in ok_rows:
        history = load_history(Path(str(row["history_file"])))
        ax.plot(history["iter"], history["gate_mean"], label=f"{row['case']} effective")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean effective gate")
    ax.set_title("HF-CFNO gate ablation: effective high-pass amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(ROOT / "AblationEffectiveGateMean.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for row in ok_rows:
        history = load_history(Path(str(row["history_file"])))
        ax.plot(history["iter"], history["spatial_gate_mean"], label=str(row["case"]))
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean spatial gate")
    ax.set_title("HF-CFNO gate ablation: vorticity spatial gate")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(ROOT / "AblationSpatialGateMean.png", dpi=200)
    plt.close(fig)

    labels = [str(row["case"]) for row in ok_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar(x, [float(row["best_pde"]) for row in ok_rows], color="#4477aa")
    axes[0].set_yscale("log")
    axes[0].set_title("Best PDE residual")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].grid(True, axis="y", which="both", alpha=0.25)
    axes[1].bar(x, [float(row["ghia_rmse"]) for row in ok_rows], color="#aa7744")
    axes[1].set_title("Ghia centerline RMSE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.savefig(ROOT / "AblationMetricBars.png", dpi=200)
    plt.close(fig)


def main() -> None:
    rows = [summarize_case(config) for config in CASES]
    write_csv(rows, ROOT / "summary.csv")
    write_markdown(rows, ROOT / "summary.md")
    write_plots(rows)
    print(f"Wrote {ROOT / 'summary.csv'}")
    print(f"Wrote {ROOT / 'summary.md'}")


if __name__ == "__main__":
    main()
