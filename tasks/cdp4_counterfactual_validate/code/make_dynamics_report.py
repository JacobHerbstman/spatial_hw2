#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

# Ensure matplotlib cache dir is writable in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


def slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_")
    return out or "default"


def resolve_file(base_dir: Path, stem: str, profile: str, shock: str) -> Path:
    candidates = [
        base_dir / f"{stem}_{profile}_{shock}.csv",
        base_dir / f"{stem}_{profile}.csv",
        base_dir / f"{stem}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not resolve CSV for stem={stem} in {base_dir}")


def format_float(v: float) -> str:
    if pd.isna(v):
        return ""
    av = abs(v)
    if av != 0 and (av < 1e-4 or av >= 1e4):
        return f"{v:.4e}"
    return f"{v:.6f}"


def latex_escape(s: str) -> str:
    rep = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    out = str(s)
    for k, v in rep.items():
        out = out.replace(k, v)
    return out


def cell_to_latex(cell) -> str:
    if isinstance(cell, str):
        return latex_escape(cell)
    if pd.isna(cell):
        return ""
    if isinstance(cell, (int, float)):
        return format_float(float(cell))
    return latex_escape(str(cell))


def write_tex_table(df: pd.DataFrame, out_path: Path, caption: str, label: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(df.columns)
    align = "l" * len(cols)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\hline"]
    lines.append(" & ".join(latex_escape(str(c)) for c in cols) + " \\\\")
    lines.append("\\hline")
    for _, row in df.iterrows():
        lines.append(" & ".join(cell_to_latex(row[c]) for c in cols) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    tex = "\n".join(lines) + "\n"
    wrapper = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{tex}"
        "\\end{table}\n"
    )
    out_path.write_text(wrapper)


def maybe_set_log_scale(ax, *series: pd.Series) -> None:
    positive = False
    for series_item in series:
        if pd.Series(series_item).gt(0).any():
            positive = True
            break
    if positive:
        ax.set_yscale("log")


def make_plots_and_tables(profile: str, shock: str, dynamics_dir: Path, validate_dir: Path, out_dir: Path) -> None:
    shock_slug = slug(shock)

    outer_trace = pd.read_csv(resolve_file(dynamics_dir, "outer_trace_counterfactual_4sector", profile, shock_slug))
    parity = pd.read_csv(resolve_file(validate_dir, "parity_by_time_counterfactual_4sector", profile, shock_slug))
    summary_raw = pd.read_csv(resolve_file(dynamics_dir, "summary_counterfactual_4sector", profile, shock_slug))
    validation = pd.read_csv(resolve_file(validate_dir, "validation_counterfactual_4sector", profile, shock_slug))
    bench_dyn = pd.read_csv(resolve_file(dynamics_dir, "benchmark_counterfactual_4sector", profile, shock_slug))
    bench_val = pd.read_csv(resolve_file(validate_dir, "benchmark_validate_counterfactual_4sector", profile, shock_slug))

    summary_map = dict(zip(summary_raw["metric"], summary_raw["value"]))
    summary = pd.DataFrame([summary_map])

    # Figure 1: combined diagnostics
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(outer_trace["outer_iter"], outer_trace["Ymax"], marker="o", linewidth=1.8)
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_title("Outer Iteration Convergence (Ymax)")
    axs[0, 0].set_xlabel("Outer Iteration")
    axs[0, 0].set_ylabel("Ymax (log scale)")
    axs[0, 0].grid(alpha=0.25)

    axs[0, 1].plot(outer_trace["outer_iter"], outer_trace["mean_static_iterations"], marker="o", linewidth=1.6, label="Mean Static Iters")
    axs[0, 1].plot(outer_trace["outer_iter"], outer_trace["max_static_iterations"], marker="s", linewidth=1.6, label="Max Static Iters")
    axs[0, 1].set_title("Static Solver Iterations by Outer Loop")
    axs[0, 1].set_xlabel("Outer Iteration")
    axs[0, 1].set_ylabel("Iterations")
    axs[0, 1].grid(alpha=0.25)
    axs[0, 1].legend(frameon=False)

    axs[1, 0].plot(parity["t"], parity["max_abs_error_t"], linewidth=1.5, label="Max Abs Error")
    axs[1, 0].plot(parity["t"], parity["mean_abs_error_t"], linewidth=1.5, label="Mean Abs Error")
    maybe_set_log_scale(axs[1, 0], parity["max_abs_error_t"], parity["mean_abs_error_t"])
    axs[1, 0].set_title("Parity Errors Over Time")
    axs[1, 0].set_xlabel("t")
    axs[1, 0].set_ylabel("Absolute Error (log scale)")
    axs[1, 0].grid(alpha=0.25)
    axs[1, 0].legend(frameon=False)

    axs[1, 1].plot(parity["t"], parity["max_abs_error_t"], linewidth=1.5)
    axs[1, 1].set_title("Parity Max Abs Error (Linear)")
    axs[1, 1].set_xlabel("t")
    axs[1, 1].set_ylabel("Max Abs Error")
    axs[1, 1].grid(alpha=0.25)

    fig.suptitle(f"Counterfactual Dynamics Diagnostics: profile={profile}, shock={shock_slug}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = out_dir / f"dynamics_counterfactual_{profile}_{shock_slug}.pdf"
    fig.savefig(fig_path)
    plt.close(fig)

    # Figure 2: convergence-only panel
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(outer_trace["outer_iter"], outer_trace["Ymax"], marker="o", linewidth=1.9)
    ax2.set_yscale("log")
    ax2.set_title("Counterfactual Outer Convergence")
    ax2.set_xlabel("Outer Iteration")
    ax2.set_ylabel("Ymax (log scale)")
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    fig2_path = out_dir / f"dynamics_outer_counterfactual_{profile}_{shock_slug}.pdf"
    fig2.savefig(fig2_path)
    plt.close(fig2)

    # Figure 3: parity-only panel
    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    ax3.plot(parity["t"], parity["max_abs_error_t"], linewidth=1.8, label="Max Abs Error")
    ax3.plot(parity["t"], parity["mean_abs_error_t"], linewidth=1.8, label="Mean Abs Error")
    maybe_set_log_scale(ax3, parity["max_abs_error_t"], parity["mean_abs_error_t"])
    ax3.set_title("Counterfactual Parity by Time")
    ax3.set_xlabel("t")
    ax3.set_ylabel("Absolute Error (log scale)")
    ax3.grid(alpha=0.25)
    ax3.legend(frameon=False)
    fig3.tight_layout()
    fig3_path = out_dir / f"dynamics_parity_counterfactual_{profile}_{shock_slug}.pdf"
    fig3.savefig(fig3_path)
    plt.close(fig3)

    # Table 1: summary + benchmarks
    summary_tab = summary.copy()
    summary_tab["benchmark_dyn_wall_seconds"] = bench_dyn.loc[0, "wall_seconds"]
    summary_tab["benchmark_val_wall_seconds"] = bench_val.loc[0, "wall_seconds"]
    summary_tab["benchmark_dyn_alloc_bytes"] = bench_dyn.loc[0, "alloc_bytes"]
    summary_tab["benchmark_val_alloc_bytes"] = bench_val.loc[0, "alloc_bytes"]
    write_tex_table(
        summary_tab,
        out_dir / f"table_counterfactual_summary_{profile}_{shock_slug}.tex",
        caption=f"Counterfactual summary metrics ({profile}, {shock_slug}).",
        label=f"tab:counterfactual_summary_{profile}_{shock_slug}",
    )

    # Table 2: validation checks
    val_tab = validation[["check", "value", "threshold", "status"]].copy()
    write_tex_table(
        val_tab,
        out_dir / f"table_counterfactual_validation_{profile}_{shock_slug}.tex",
        caption=f"Counterfactual validation checks ({profile}, {shock_slug}).",
        label=f"tab:counterfactual_validation_{profile}_{shock_slug}",
    )

    # Table 3: selected time diagnostics
    T = len(parity)
    selected = sorted(set([1, 2, 5, 10, 20, 50, 100, T]))
    selected = [t for t in selected if 1 <= t <= T]
    parity_tab = parity[parity["t"].isin(selected)][["t", "max_abs_error_t", "mean_abs_error_t"]].copy()
    write_tex_table(
        parity_tab,
        out_dir / f"table_counterfactual_parity_selected_t_{profile}_{shock_slug}.tex",
        caption=f"Selected-time parity diagnostics ({profile}, {shock_slug}).",
        label=f"tab:counterfactual_parity_selected_{profile}_{shock_slug}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create PDF dynamics plots and LaTeX tables from counterfactual outputs.")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--shock", default="identity")
    parser.add_argument("--dynamics-dir", default="../../cdp4_counterfactual_dynamics/output")
    parser.add_argument("--validate-dir", default="../output")
    parser.add_argument("--output-dir", default="../output")
    args = parser.parse_args()

    dynamics_dir = Path(args.dynamics_dir).resolve()
    validate_dir = Path(args.validate_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    make_plots_and_tables(args.profile, args.shock, dynamics_dir, validate_dir, out_dir)
    print(f"Wrote report artifacts to {out_dir}")


if __name__ == "__main__":
    main()
