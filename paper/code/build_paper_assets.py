#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_summary(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df["metric"], df["value"]))


def read_window_means(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("variable")


def pct(value: float) -> str:
    return f"{value:.2f}\\%"


def num(value: float) -> str:
    return f"{value:.4f}"


def write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_macros(delta_selection: dict, immediate_summary: dict[str, str], anticipated_summary: dict[str, str]) -> str:
    selected_delta = float(delta_selection["selected_delta"])
    immediate_iters = int(float(immediate_summary["iterations"]))
    anticipated_iters = int(float(anticipated_summary["iterations"]))
    return f"""
\\newcommand{{\\SelectedDelta}}{{{selected_delta:.4f}}}
\\newcommand{{\\ImmediateIterations}}{{{immediate_iters}}}
\\newcommand{{\\AnticipatedIterations}}{{{anticipated_iters}}}
"""


def build_baseline_table(df: pd.DataFrame) -> str:
    keep = [
        "ynew_max_abs_error",
        "ynew_max_rel_error",
        "deterministic_rerun_max_abs_delta",
    ]
    rows = df[df["check"].isin(keep)].copy()
    labels = {
        "ynew_max_abs_error": "Max abs. error vs. MATLAB anchor",
        "ynew_max_rel_error": "Max rel. error vs. MATLAB anchor",
        "deterministic_rerun_max_abs_delta": "Deterministic rerun max abs. delta",
    }
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Reference baseline validation checks.}",
        "\\label{tab:baseline-validation}",
        "\\begin{tabular}{lrrl}",
        "\\toprule",
        "Check & Value & Threshold & Status \\\\",
        "\\midrule",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"{labels[row.check]} & {num(float(row.value))} & {num(float(row.threshold))} & {row.status} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def build_ai_results_table(delta_selection: dict, immediate_summary: dict[str, str], anticipated_summary: dict[str, str],
                           immediate_windows: pd.DataFrame, anticipated_windows: pd.DataFrame) -> str:
    rows = [
        (
            "Immediate",
            "t=1",
            int(float(immediate_summary["converged"])) == 1,
            int(float(immediate_summary["iterations"])),
            float(immediate_summary["final_ymax"]),
            float(immediate_windows.loc["rw_cell_pct", "t2_t20"]),
            float(immediate_windows.loc["rw_cell_pct", "t21_t100"]),
            float(immediate_windows.loc["rw_cell_pct", "t101_tT"]),
        ),
        (
            "Anticipated",
            "t=21",
            int(float(anticipated_summary["converged"])) == 1,
            int(float(anticipated_summary["iterations"])),
            float(anticipated_summary["final_ymax"]),
            float(anticipated_windows.loc["rw_cell_pct", "t2_t20"]),
            float(anticipated_windows.loc["rw_cell_pct", "t21_t100"]),
            float(anticipated_windows.loc["rw_cell_pct", "t101_tT"]),
        ),
    ]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Reference AI counterfactual results at the selected common shock size $\\Delta = {float(delta_selection['selected_delta']):.4f}$. Real-wage effects are reported for the focus labor market used in the task outputs (New York services).}}",
        "\\label{tab:ai-results}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Scenario & Activation & Conv. & Iter. & Final $Y_{\\max}$ & Early RW & Mid RW & Late RW \\\\",
        "\\midrule",
    ]
    for label, activation, converged, iterations, final_ymax, early_rw, mid_rw, late_rw in rows:
        lines.append(
            f"{label} & {activation} & {'Yes' if converged else 'No'} & {iterations} & {num(final_ymax)} & {pct(early_rw)} & {pct(mid_rw)} & {pct(late_rw)} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build generated paper assets from canonical cognitive outputs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_validation = pd.read_csv(input_dir / "baseline_validation_reference.csv")
    delta_selection = json.loads((input_dir / "selected_reference_delta.json").read_text())
    immediate_summary = read_summary(input_dir / "summary_counterfactual_4sector_reference_cognitive_immediate.csv")
    anticipated_summary = read_summary(input_dir / "summary_counterfactual_4sector_reference_cognitive_anticipated.csv")
    immediate_windows = read_window_means(input_dir / "key_econ_window_means_reference_cognitive_immediate.csv")
    anticipated_windows = read_window_means(input_dir / "key_econ_window_means_reference_cognitive_anticipated.csv")

    write(output_dir / "macros.tex", build_macros(delta_selection, immediate_summary, anticipated_summary))
    write(output_dir / "table_baseline_validation.tex", build_baseline_table(baseline_validation))
    write(
        output_dir / "table_ai_results.tex",
        build_ai_results_table(
            delta_selection,
            immediate_summary,
            anticipated_summary,
            immediate_windows,
            anticipated_windows,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
