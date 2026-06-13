from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def load_config() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    config_path = ROOT / "case_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        return config["reynolds"], config["methods"]
    reynolds = [
        {"Label": "Re1e02", "Value": "1e2"},
        {"Label": "Re1e03", "Value": "1e3"},
        {"Label": "Re1e04", "Value": "1e4"},
        {"Label": "Re1e05", "Value": "1e5"},
        {"Label": "Re1e06", "Value": "1e6"},
        {"Label": "Re1e07", "Value": "1e7"},
    ]
    methods = [
        {"Name": "fno"},
        {"Name": "cfno"},
        {"Name": "hf_cfno"},
        {"Name": "subgrid_hf_cfno"},
    ]
    return reynolds, methods


def find_one(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def load_history(path: Path) -> np.ndarray:
    history = np.genfromtxt(path, delimiter=",", names=True)
    if history.shape == ():
        history = np.array([history], dtype=history.dtype)
    return history


def ghia_rmse(path: Path | None) -> float:
    if path is None or not path.exists():
        return math.nan
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data[None, :]
        err = data[:, 4].astype(float)
        return float(np.sqrt(np.mean(err**2)))
    except Exception:
        return math.nan


def field_metrics(out_dir: Path, prefix: str) -> dict[str, float]:
    result = {
        "max_speed": math.nan,
        "mean_speed": math.nan,
        "interior_max_speed": math.nan,
        "interior_mean_speed": math.nan,
        "max_abs_vorticity": math.nan,
        "mean_abs_vorticity": math.nan,
        "interior_max_abs_vorticity": math.nan,
        "interior_mean_abs_vorticity": math.nan,
        "max_abs_u": math.nan,
        "max_abs_v": math.nan,
        "max_abs_p": math.nan,
    }
    speed_path = find_one(out_dir, "*_Speed.npy")
    omega_path = find_one(out_dir, "*_Vorticity.npy")
    u_path = find_one(out_dir, "*_U.npy")
    v_path = find_one(out_dir, "*_V.npy")
    p_path = find_one(out_dir, "*_P.npy")
    if speed_path is not None:
        speed = np.load(speed_path)
        result["max_speed"] = float(np.nanmax(speed))
        result["mean_speed"] = float(np.nanmean(speed))
        interior = speed[2:-2, 2:-2] if min(speed.shape) > 4 else speed
        result["interior_max_speed"] = float(np.nanmax(interior))
        result["interior_mean_speed"] = float(np.nanmean(interior))
    if omega_path is not None:
        omega = np.load(omega_path)
        result["max_abs_vorticity"] = float(np.nanmax(np.abs(omega)))
        result["mean_abs_vorticity"] = float(np.nanmean(np.abs(omega)))
        interior = omega[2:-2, 2:-2] if min(omega.shape) > 4 else omega
        result["interior_max_abs_vorticity"] = float(np.nanmax(np.abs(interior)))
        result["interior_mean_abs_vorticity"] = float(np.nanmean(np.abs(interior)))
    if u_path is not None:
        result["max_abs_u"] = float(np.nanmax(np.abs(np.load(u_path))))
    if v_path is not None:
        result["max_abs_v"] = float(np.nanmax(np.abs(np.load(v_path))))
    if p_path is not None:
        result["max_abs_p"] = float(np.nanmax(np.abs(np.load(p_path))))
    return result


def summarize_case(re_cfg: dict[str, object], method_cfg: dict[str, object]) -> dict[str, object]:
    re_label = str(re_cfg["Label"])
    re_value = float(re_cfg["Value"])
    method = str(method_cfg["Name"])
    out_dir = ROOT / re_label / method
    row: dict[str, object] = {
        "re": re_value,
        "re_label": re_label,
        "method": method,
        "output_dir": str(out_dir),
    }
    history_path = find_one(out_dir, "*_history.csv")
    if history_path is None:
        row["status"] = "missing_history"
        return row

    history = load_history(history_path)
    pde = history["L_cont"] + history["L_mom"]
    finite_mask = np.isfinite(pde) & np.isfinite(history["loss"])
    row["history_file"] = str(history_path)
    row["status"] = "ok" if bool(np.all(finite_mask)) else "nonfinite_history"
    if np.any(finite_mask):
        finite_idx = np.where(finite_mask)[0]
        best_local = finite_idx[int(np.argmin(pde[finite_mask]))]
        best_loss_local = finite_idx[int(np.argmin(history["loss"][finite_mask]))]
    else:
        best_local = len(history) - 1
        best_loss_local = len(history) - 1

    final = history[-1]
    row.update(
        {
            "final_iter": int(final["iter"]),
            "final_L_cont": float(final["L_cont"]),
            "final_L_mom": float(final["L_mom"]),
            "final_pde": float(final["L_cont"] + final["L_mom"]),
            "final_loss": float(final["loss"]),
            "final_L_gate": float(final["L_gate"]),
            "final_gate_mean": float(final["gate_mean"]),
            "final_spatial_gate_mean": float(final["spatial_gate_mean"]),
            "best_pde_iter": int(history["iter"][best_local]),
            "best_pde": float(pde[best_local]),
            "best_pde_loss": float(history["loss"][best_local]),
            "best_loss_iter": int(history["iter"][best_loss_local]),
            "best_loss": float(history["loss"][best_loss_local]),
            "best_loss_pde": float(pde[best_loss_local]),
        }
    )
    row.update(field_metrics(out_dir, method))
    row["ghia_re100_rmse"] = ghia_rmse(find_one(out_dir, "*_ghia_re100_centerline_error.csv"))
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


def fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4e}"
    return str(value)


def pivot(rows: list[dict[str, object]], metric: str) -> tuple[list[float], list[str], dict[str, list[float]]]:
    ok_rows = [row for row in rows if row.get("status") in {"ok", "nonfinite_history"}]
    re_values = sorted({float(row["re"]) for row in ok_rows})
    methods = ["fno", "cfno", "hf_cfno", "subgrid_hf_cfno"]
    data = {method: [] for method in methods}
    for re_value in re_values:
        for method in methods:
            match = next((row for row in ok_rows if float(row["re"]) == re_value and row["method"] == method), None)
            data[method].append(float(match.get(metric, math.nan)) if match else math.nan)
    return re_values, methods, data


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    columns = [
        "re_label",
        "method",
        "status",
        "best_pde",
        "best_pde_iter",
        "final_pde",
        "final_L_cont",
        "final_L_mom",
        "interior_max_speed",
        "interior_max_abs_vorticity",
        "final_gate_mean",
        "ghia_re100_rmse",
    ]
    lines = [
        "# Reynolds operator search summary",
        "",
        "Methods: FNO, CFNO, legacy HF-CFNO with replicate cavity padding, and subgrid vorticity-gated HF-CFNO without explicit L_gate alignment.",
        "",
        "|" + "|".join(columns) + "|",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in sorted(rows, key=lambda r: (float(r["re"]), str(r["method"]))):
        lines.append("|" + "|".join(fmt(row.get(col, "")) for col in columns) + "|")
    lines.extend(
        [
            "",
            "Metric notes:",
            "",
            "- `best_pde` is `min(L_cont + L_mom)` over the run.",
            "- `final_pde` is the residual at the final iteration.",
            "- `ghia_re100_rmse` is only valid and only emitted for Re=100.",
            "- Plain FNO/CFNO use the same Fourier coordinate features as the HF variants.",
            "",
            "Aggregate figures:",
            "",
            "- `BestPDE_vs_Re.png`",
            "- `FinalPDE_vs_Re.png`",
            "- `GateMean_vs_Re.png`",
            "- `MaxVorticity_vs_Re.png`",
            "- `MetricHeatmap_best_pde.png`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_metric(rows: list[dict[str, object]], metric: str, ylabel: str, filename: str, logy: bool = True) -> None:
    re_values, methods, data = pivot(rows, metric)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method in methods:
        values = np.array(data[method], dtype=float)
        ax.plot(re_values, values, marker="o", label=method)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Reynolds number")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(ROOT / filename, dpi=200)
    plt.close(fig)


def plot_heatmap(rows: list[dict[str, object]], metric: str, filename: str) -> None:
    re_values, methods, data = pivot(rows, metric)
    matrix = np.array([data[method] for method in methods], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    im = ax.imshow(np.log10(matrix), aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xticks(np.arange(len(re_values)))
    ax.set_xticklabels([f"1e{int(round(np.log10(v)))}" for v in re_values])
    ax.set_title(f"log10({metric})")
    fig.colorbar(im, ax=ax)
    fig.savefig(ROOT / filename, dpi=200)
    plt.close(fig)


def write_plots(rows: list[dict[str, object]]) -> None:
    plot_metric(rows, "best_pde", "best PDE residual", "BestPDE_vs_Re.png")
    plot_metric(rows, "final_pde", "final PDE residual", "FinalPDE_vs_Re.png")
    plot_metric(rows, "final_gate_mean", "final effective gate mean", "GateMean_vs_Re.png", logy=False)
    plot_metric(rows, "interior_max_abs_vorticity", "interior max abs vorticity", "MaxVorticity_vs_Re.png")
    plot_heatmap(rows, "best_pde", "MetricHeatmap_best_pde.png")


def main() -> None:
    reynolds, methods = load_config()
    rows = [summarize_case(re_cfg, method_cfg) for re_cfg in reynolds for method_cfg in methods]
    write_csv(rows, ROOT / "summary.csv")
    write_markdown(rows, ROOT / "summary.md")
    write_plots(rows)
    print(f"Wrote {ROOT / 'summary.csv'}")
    print(f"Wrote {ROOT / 'summary.md'}")


if __name__ == "__main__":
    main()
