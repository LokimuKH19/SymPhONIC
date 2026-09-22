from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

import run_legacy_hf_pde_suite as suite


VARIANT_ORDER = {"FNO": 0, "CFNO": 1, "HF_FNO": 2, "HF_CFNO": 3}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def namespace_from_config(config_path: Path, profile: str) -> argparse.Namespace:
    args = argparse.Namespace()
    defaults = {
        "output_root": "LegacyHF_PDE_Benchmark_PINO_20260705",
        "epochs": 1200,
        "grid": 64,
        "train_samples": 32,
        "val_samples": 8,
        "test_samples": 4,
        "batch_size": 16,
        "lr": 1e-3,
        "modes": 8,
        "high_modes": 16,
        "depth": 4,
        "target_params": 500000,
        "seed": 20260705,
        "variants": "FNO,CFNO,HF_FNO,HF_CFNO",
        "attention_rank": 8,
        "attention_gate_init": -2.0,
        "pdes": "",
        "cases": "",
        "profile": profile,
        "training_mode": "pde",
        "physics_weight": 0.1,
        "append_existing_global": False,
        "resume": False,
        "cpu": False,
    }
    if config_path.exists():
        config = load_json(config_path).get("args", {})
        defaults.update({k.replace("-", "_"): v for k, v in config.items()})
    defaults["profile"] = profile
    for key, value in defaults.items():
        setattr(args, key, value)
    return args


def build_spec_index(root: Path) -> dict[tuple[str, str], tuple[str, suite.CaseSpec]]:
    config_by_profile = {
        "baseline": root / "benchmark_config.json",
        "high_nonlinear": root / "benchmark_config_high_nonlinear.json",
    }
    specs: dict[tuple[str, str], tuple[str, suite.CaseSpec]] = {}
    for profile, config_path in config_by_profile.items():
        args = namespace_from_config(config_path, profile)
        for spec in suite.build_case_specs(args):
            specs[(spec.pde, spec.case)] = (profile, spec)
    return specs


def spec_for_npz(root: Path, npz_path: Path, spec_index: dict[tuple[str, str], tuple[str, suite.CaseSpec]]) -> tuple[str, suite.CaseSpec, str]:
    rel = npz_path.relative_to(root)
    parts = rel.parts
    if len(parts) < 4:
        raise ValueError(f"Unexpected sample_fields.npz path: {npz_path}")
    pde, case, variant = parts[0], parts[1], parts[2]
    key = (pde, case)
    if key not in spec_index:
        raise KeyError(f"No CaseSpec found for {pde}/{case}")
    profile, spec = spec_index[key]
    with np.load(npz_path) as data:
        shape = tuple(int(v) for v in data["prediction"].shape)
    if spec.is_transient:
        spec = replace(spec, nx=shape[0], ny=shape[0], nt=shape[1])
    elif spec.dimension == 1:
        spec = replace(spec, nx=shape[0], ny=shape[1])
    else:
        spec = replace(spec, nx=shape[0], ny=shape[1])
    return profile, spec, variant


def summarize_array(arr: np.ndarray, prefix: str) -> dict[str, float]:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    abs_flat = np.abs(flat)
    return {
        f"{prefix}_mean": float(np.mean(flat)),
        f"{prefix}_mean_abs": float(np.mean(abs_flat)),
        f"{prefix}_mse": float(np.mean(flat**2)),
        f"{prefix}_rms": float(np.sqrt(np.mean(flat**2))),
        f"{prefix}_p95_abs": float(np.percentile(abs_flat, 95)),
        f"{prefix}_max_abs": float(np.max(abs_flat)),
    }


def source_summary(source: np.ndarray) -> dict[str, float]:
    flat = np.asarray(source, dtype=np.float64).reshape(-1)
    abs_flat = np.abs(flat)
    rms = float(np.sqrt(np.mean(flat**2)))
    max_abs = float(np.max(abs_flat))
    return {
        "source_mean_abs": float(np.mean(abs_flat)),
        "source_rms": rms,
        "source_max_abs": max_abs,
    }


def residual_stats_for_file(root: Path, npz_path: Path, spec_index: dict[tuple[str, str], tuple[str, suite.CaseSpec]]) -> dict[str, Any]:
    profile, spec, variant = spec_for_npz(root, npz_path, spec_index)
    with np.load(npz_path) as data:
        prediction = np.asarray(data["prediction"], dtype=np.float64)
        exact = np.asarray(data["exact"], dtype=np.float64)
        source = np.asarray(data["source"], dtype=np.float64)
        stored_pred = np.asarray(data["residual_prediction"], dtype=np.float64) if "residual_prediction" in data.files else None
        stored_exact = np.asarray(data["residual_exact"], dtype=np.float64) if "residual_exact" in data.files else None

    recomputed_pred = np.asarray(suite.residual_field(spec, prediction, source), dtype=np.float64)
    recomputed_exact = np.asarray(suite.residual_field(spec, exact, source), dtype=np.float64)
    field_error = prediction - exact

    row: dict[str, Any] = {
        "pde": spec.pde,
        "case": spec.case,
        "variant": variant,
        "profile": profile,
        "kind": spec.kind,
        "dimension": spec.dimension,
        "shape": "x".join(str(v) for v in prediction.shape),
        "point_count": int(prediction.size),
        "source_npz": str(npz_path.relative_to(root)),
        "all_grid_points": True,
        "boundary_points_included": True,
        "interior_crop_applied": False,
        "residual_definition": "L(prediction)-source, recomputed from sample_fields.npz",
    }
    row.update({"spec_params_json": json.dumps(spec.params, sort_keys=True)})
    row.update(source_summary(source))
    row.update(summarize_array(field_error, "field_error"))
    row.update(summarize_array(recomputed_pred, "pred_residual"))
    row.update(summarize_array(recomputed_exact, "exact_residual"))
    exact_rms = float(np.sqrt(np.mean(exact.reshape(-1) ** 2)))
    exact_max_abs = float(np.max(np.abs(exact)))
    row["exact_field_rms"] = exact_rms
    row["exact_field_max_abs"] = exact_max_abs
    row["field_error_rms_over_exact_rms"] = float(row["field_error_rms"] / max(exact_rms, 1e-30))
    row["field_error_max_abs_over_exact_max_abs"] = float(row["field_error_max_abs"] / max(exact_max_abs, 1e-30))
    row["pred_residual_mean_abs_over_source_rms"] = float(row["pred_residual_mean_abs"] / max(row["source_rms"], 1e-30))
    row["pred_residual_rms_over_source_rms"] = float(row["pred_residual_rms"] / max(row["source_rms"], 1e-30))
    row["pred_residual_max_abs_over_source_max_abs"] = float(row["pred_residual_max_abs"] / max(row["source_max_abs"], 1e-30))
    row["pred_over_exact_residual_rms"] = float(row["pred_residual_rms"] / max(row["exact_residual_rms"], 1e-30))
    row["pred_minus_exact_residual_mean_abs"] = float(np.mean(np.abs(recomputed_pred - recomputed_exact)))
    row["stored_pred_residual_max_abs_diff"] = (
        float(np.max(np.abs(stored_pred - recomputed_pred))) if stored_pred is not None else np.nan
    )
    row["stored_exact_residual_max_abs_diff"] = (
        float(np.max(np.abs(stored_exact - recomputed_exact))) if stored_exact is not None else np.nan
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rank_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["pde"], row["case"])].append(row)

    ranked: list[dict[str, Any]] = []
    for (pde, case), case_rows in sorted(grouped.items()):
        by_mean = sorted(case_rows, key=lambda r: (r["pred_residual_mean_abs"], VARIANT_ORDER.get(r["variant"], 99)))
        by_max = sorted(case_rows, key=lambda r: (r["pred_residual_max_abs"], VARIANT_ORDER.get(r["variant"], 99)))
        by_rms = sorted(case_rows, key=lambda r: (r["pred_residual_rms"], VARIANT_ORDER.get(r["variant"], 99)))
        by_field_mean = sorted(case_rows, key=lambda r: (r["field_error_mean_abs"], VARIANT_ORDER.get(r["variant"], 99)))
        by_field_rms = sorted(case_rows, key=lambda r: (r["field_error_rms"], VARIANT_ORDER.get(r["variant"], 99)))
        by_field_max = sorted(case_rows, key=lambda r: (r["field_error_max_abs"], VARIANT_ORDER.get(r["variant"], 99)))
        rank_mean = {id(r): i + 1 for i, r in enumerate(by_mean)}
        rank_max = {id(r): i + 1 for i, r in enumerate(by_max)}
        rank_rms = {id(r): i + 1 for i, r in enumerate(by_rms)}
        rank_field_mean = {id(r): i + 1 for i, r in enumerate(by_field_mean)}
        rank_field_rms = {id(r): i + 1 for i, r in enumerate(by_field_rms)}
        rank_field_max = {id(r): i + 1 for i, r in enumerate(by_field_max)}
        for row in sorted(case_rows, key=lambda r: VARIANT_ORDER.get(r["variant"], 99)):
            out = {
                "pde": pde,
                "case": case,
                "variant": row["variant"],
                "profile": row["profile"],
                "point_count": row["point_count"],
                "rank_mean_abs": rank_mean[id(row)],
                "rank_rms": rank_rms[id(row)],
                "rank_max_abs": rank_max[id(row)],
                "rank_field_mean_abs": rank_field_mean[id(row)],
                "rank_field_rms": rank_field_rms[id(row)],
                "rank_field_max_abs": rank_field_max[id(row)],
                "field_error_mean_abs": row["field_error_mean_abs"],
                "field_error_rms": row["field_error_rms"],
                "field_error_max_abs": row["field_error_max_abs"],
                "field_error_rms_over_exact_rms": row["field_error_rms_over_exact_rms"],
                "field_error_max_abs_over_exact_max_abs": row["field_error_max_abs_over_exact_max_abs"],
                "pred_residual_mean_abs": row["pred_residual_mean_abs"],
                "pred_residual_rms": row["pred_residual_rms"],
                "pred_residual_max_abs": row["pred_residual_max_abs"],
                "pred_residual_rms_over_source_rms": row["pred_residual_rms_over_source_rms"],
                "pred_residual_max_abs_over_source_max_abs": row["pred_residual_max_abs_over_source_max_abs"],
            }
            ranked.append(out)
    return ranked


def aggregate_by_variant(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)

    out_rows: list[dict[str, Any]] = []
    for variant, variant_rows in sorted(grouped.items(), key=lambda kv: VARIANT_ORDER.get(kv[0], 99)):
        mean_abs = np.array([r["pred_residual_mean_abs"] for r in variant_rows], dtype=np.float64)
        rms = np.array([r["pred_residual_rms"] for r in variant_rows], dtype=np.float64)
        max_abs = np.array([r["pred_residual_max_abs"] for r in variant_rows], dtype=np.float64)
        norm_rms = np.array([r["pred_residual_rms_over_source_rms"] for r in variant_rows], dtype=np.float64)
        norm_max = np.array([r["pred_residual_max_abs_over_source_max_abs"] for r in variant_rows], dtype=np.float64)
        field_mean_abs = np.array([r["field_error_mean_abs"] for r in variant_rows], dtype=np.float64)
        field_rms = np.array([r["field_error_rms"] for r in variant_rows], dtype=np.float64)
        field_max_abs = np.array([r["field_error_max_abs"] for r in variant_rows], dtype=np.float64)
        field_rel_rms = np.array([r["field_error_rms_over_exact_rms"] for r in variant_rows], dtype=np.float64)
        out_rows.append({
            "variant": variant,
            "case_count": len(variant_rows),
            "mean_field_error_mean_abs": float(np.mean(field_mean_abs)),
            "median_field_error_mean_abs": float(np.median(field_mean_abs)),
            "mean_field_error_rms": float(np.mean(field_rms)),
            "median_field_error_rms": float(np.median(field_rms)),
            "mean_field_error_max_abs": float(np.mean(field_max_abs)),
            "median_field_error_max_abs": float(np.median(field_max_abs)),
            "mean_field_error_relative_rms": float(np.mean(field_rel_rms)),
            "median_field_error_relative_rms": float(np.median(field_rel_rms)),
            "mean_of_mean_abs": float(np.mean(mean_abs)),
            "median_of_mean_abs": float(np.median(mean_abs)),
            "mean_of_rms": float(np.mean(rms)),
            "median_of_rms": float(np.median(rms)),
            "mean_of_max_abs": float(np.mean(max_abs)),
            "median_of_max_abs": float(np.median(max_abs)),
            "mean_rms_over_source_rms": float(np.mean(norm_rms)),
            "median_rms_over_source_rms": float(np.median(norm_rms)),
            "mean_max_abs_over_source_max_abs": float(np.mean(norm_max)),
            "median_max_abs_over_source_max_abs": float(np.median(norm_max)),
        })
    return out_rows


def winners_by_case(ranked_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ranked_rows:
        grouped[(row["pde"], row["case"])].append(row)

    winners: list[dict[str, Any]] = []
    for (pde, case), case_rows in sorted(grouped.items()):
        best_mean = min(case_rows, key=lambda r: r["rank_mean_abs"])
        best_rms = min(case_rows, key=lambda r: r["rank_rms"])
        best_max = min(case_rows, key=lambda r: r["rank_max_abs"])
        best_field_mean = min(case_rows, key=lambda r: r["rank_field_mean_abs"])
        best_field_rms = min(case_rows, key=lambda r: r["rank_field_rms"])
        best_field_max = min(case_rows, key=lambda r: r["rank_field_max_abs"])
        winners.append({
            "pde": pde,
            "case": case,
            "best_field_mean_abs_variant": best_field_mean["variant"],
            "best_field_mean_abs": best_field_mean["field_error_mean_abs"],
            "best_field_rms_variant": best_field_rms["variant"],
            "best_field_rms": best_field_rms["field_error_rms"],
            "best_field_max_abs_variant": best_field_max["variant"],
            "best_field_max_abs": best_field_max["field_error_max_abs"],
            "best_mean_abs_variant": best_mean["variant"],
            "best_mean_abs": best_mean["pred_residual_mean_abs"],
            "best_rms_variant": best_rms["variant"],
            "best_rms": best_rms["pred_residual_rms"],
            "best_max_abs_variant": best_max["variant"],
            "best_max_abs": best_max["pred_residual_max_abs"],
        })
    return winners


def write_markdown_summary(
    root: Path,
    rows: list[dict[str, Any]],
    ranked_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    winner_rows: list[dict[str, Any]],
    stem: str,
) -> None:
    worst_max = sorted(rows, key=lambda r: r["pred_residual_max_abs"], reverse=True)[:10]
    worst_mean = sorted(rows, key=lambda r: r["pred_residual_mean_abs"], reverse=True)[:10]
    worst_field = sorted(rows, key=lambda r: r["field_error_mean_abs"], reverse=True)[:10]
    lines = [
        "# PINO Full-Field PDE Residual Statistics",
        "",
        f"- Root: `{root}`",
        f"- `sample_fields.npz` files analyzed: {len(rows)}",
        f"- Cases: {len({(r['pde'], r['case']) for r in rows})}",
        "- Field error definition: `prediction - exact`, computed over the same saved full field.",
        "- Residual definition: `L(prediction) - source`, recomputed from the saved field arrays.",
        "- Scope: every saved grid point is included; no PDE interior crop is applied.",
        "- Boundary derivatives use the same `np.gradient(..., edge_order=2)` full-field finite-difference evaluator as the plotting/evaluation path.",
        "",
        "## Aggregate By Model",
        "",
        "| variant | cases | mean field mean(abs) | median field RMS | mean residual mean(abs) | median residual RMS | mean residual max(abs) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['variant']} | {row['case_count']} | {row['mean_field_error_mean_abs']:.6e} | "
            f"{row['median_field_error_rms']:.6e} | {row['mean_of_mean_abs']:.6e} | "
            f"{row['median_of_rms']:.6e} | {row['mean_of_max_abs']:.6e} |"
        )
    lines.extend([
        "",
        "## Best Variant Per Case",
        "",
        "| PDE | case | best field mean(abs) | best field RMS | best residual mean(abs) | best residual RMS | best residual max(abs) |",
        "|---|---|---|---|---|---|---|",
    ])
    for row in winner_rows:
        lines.append(
            f"| {row['pde']} | {row['case']} | {row['best_field_mean_abs_variant']} ({row['best_field_mean_abs']:.6e}) | "
            f"{row['best_field_rms_variant']} ({row['best_field_rms']:.6e}) | "
            f"{row['best_mean_abs_variant']} ({row['best_mean_abs']:.6e}) | "
            f"{row['best_rms_variant']} ({row['best_rms']:.6e}) | {row['best_max_abs_variant']} ({row['best_max_abs']:.6e}) |"
        )
    lines.extend([
        "",
        "## Largest Field Mean-Abs Errors",
        "",
        "| PDE | case | variant | field mean(abs) | field RMS | field max(abs) | residual mean(abs) |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in worst_field:
        lines.append(
            f"| {row['pde']} | {row['case']} | {row['variant']} | {row['field_error_mean_abs']:.6e} | "
            f"{row['field_error_rms']:.6e} | {row['field_error_max_abs']:.6e} | {row['pred_residual_mean_abs']:.6e} |"
        )
    lines.extend([
        "",
        "## Largest Max-Abs Residuals",
        "",
        "| PDE | case | variant | mean(abs) | RMS | max(abs) |",
        "|---|---|---|---:|---:|---:|",
    ])
    for row in worst_max:
        lines.append(
            f"| {row['pde']} | {row['case']} | {row['variant']} | {row['pred_residual_mean_abs']:.6e} | "
            f"{row['pred_residual_rms']:.6e} | {row['pred_residual_max_abs']:.6e} |"
        )
    lines.extend([
        "",
        "## Largest Mean-Abs Residuals",
        "",
        "| PDE | case | variant | mean(abs) | RMS | max(abs) |",
        "|---|---|---|---:|---:|---:|",
    ])
    for row in worst_mean:
        lines.append(
            f"| {row['pde']} | {row['case']} | {row['variant']} | {row['pred_residual_mean_abs']:.6e} | "
            f"{row['pred_residual_rms']:.6e} | {row['pred_residual_max_abs']:.6e} |"
        )
    lines.extend([
        "",
        "## Output Files",
        "",
        f"- `{stem}_statistics.csv`: one row per `PDE/case/model`.",
        f"- `{stem}_case_ranking.csv`: per-case model ranking by field error and PDE residual metrics.",
        f"- `{stem}_model_summary.csv`: aggregate field-error and residual statistics by model family.",
        f"- `{stem}_case_winners.csv`: best model per case under field-error and residual metrics.",
        f"- `{stem}_statistics.json`: machine-readable copy of the same analysis.",
    ])
    (root / f"{stem}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize full-field PDE residuals saved in sample_fields.npz files.")
    parser.add_argument(
        "--root",
        default=r"D:\Ansys\SymPhONIC\OtherPDEs\LegacyHF_PDE_Benchmark_PINO_20260705",
        help="Benchmark root containing PDE/case/model/sample_fields.npz.",
    )
    parser.add_argument(
        "--output-stem",
        default="full_field_pde_residual",
        help="Output filename stem. Use another stem when existing CSV files are locked by another program.",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    spec_index = build_spec_index(root)
    sample_paths = sorted(root.rglob("sample_fields.npz"))
    if not sample_paths:
        raise FileNotFoundError(f"No sample_fields.npz under {root}")

    rows = [residual_stats_for_file(root, path, spec_index) for path in sample_paths]
    rows.sort(key=lambda r: (r["pde"], r["case"], VARIANT_ORDER.get(r["variant"], 99), r["variant"]))
    ranked_rows = rank_case_rows(rows)
    aggregate_rows = aggregate_by_variant(rows)
    winner_rows = winners_by_case(ranked_rows)

    primary_fields = [
        "pde",
        "case",
        "variant",
        "profile",
        "kind",
        "dimension",
        "shape",
        "point_count",
        "all_grid_points",
        "boundary_points_included",
        "interior_crop_applied",
        "source_mean_abs",
        "source_rms",
        "source_max_abs",
        "exact_field_rms",
        "exact_field_max_abs",
        "field_error_mean",
        "field_error_mean_abs",
        "field_error_mse",
        "field_error_rms",
        "field_error_p95_abs",
        "field_error_max_abs",
        "field_error_rms_over_exact_rms",
        "field_error_max_abs_over_exact_max_abs",
        "pred_residual_mean",
        "pred_residual_mean_abs",
        "pred_residual_mse",
        "pred_residual_rms",
        "pred_residual_p95_abs",
        "pred_residual_max_abs",
        "pred_residual_mean_abs_over_source_rms",
        "pred_residual_rms_over_source_rms",
        "pred_residual_max_abs_over_source_max_abs",
        "exact_residual_mean",
        "exact_residual_mean_abs",
        "exact_residual_mse",
        "exact_residual_rms",
        "exact_residual_p95_abs",
        "exact_residual_max_abs",
        "pred_over_exact_residual_rms",
        "pred_minus_exact_residual_mean_abs",
        "stored_pred_residual_max_abs_diff",
        "stored_exact_residual_max_abs_diff",
        "spec_params_json",
        "residual_definition",
        "source_npz",
    ]
    stem = args.output_stem
    write_csv(root / f"{stem}_statistics.csv", rows, primary_fields)
    write_csv(root / f"{stem}_case_ranking.csv", ranked_rows)
    write_csv(root / f"{stem}_model_summary.csv", aggregate_rows)
    write_csv(root / f"{stem}_case_winners.csv", winner_rows)

    used_spec_keys = {(r["pde"], r["case"]) for r in rows}
    payload = {
        "root": str(root),
        "sample_fields_analyzed": len(rows),
        "case_count": len({(r["pde"], r["case"]) for r in rows}),
        "all_grid_points": True,
        "boundary_points_included": True,
        "interior_crop_applied": False,
        "residual_definition": "L(prediction)-source, recomputed from sample_fields.npz arrays",
        "specs": {
            f"{pde}/{case}": asdict(spec)
            for (pde, case), (_, spec) in spec_index.items()
            if (pde, case) in used_spec_keys
        },
        "rows": rows,
        "case_ranking": ranked_rows,
        "model_summary": aggregate_rows,
        "case_winners": winner_rows,
    }
    with (root / f"{stem}_statistics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    write_markdown_summary(root, rows, ranked_rows, aggregate_rows, winner_rows, stem)
    print(f"Analyzed {len(rows)} sample_fields.npz files across {payload['case_count']} cases.")
    print(root / f"{stem}_statistics.csv")
    print(root / (f"{stem}_summary.md" if stem != "full_field_pde_residual" else "full_field_pde_residual_summary.md"))


if __name__ == "__main__":
    main()
