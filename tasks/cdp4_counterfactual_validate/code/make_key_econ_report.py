#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_")
    return out or "default"


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


def fmt(v):
    if pd.isna(v):
        return ""
    v = float(v)
    av = abs(v)
    if av != 0 and (av < 1e-4 or av >= 1e4):
        return f"{v:.4e}"
    return f"{v:.4f}"


def write_table(df: pd.DataFrame, out_path: Path, caption: str, label: str) -> None:
    cols = list(df.columns)
    align = "l" * len(cols)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\hline"]
    lines.append(" & ".join(latex_escape(c) for c in cols) + " \\\\")
    lines.append("\\hline")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            x = row[c]
            if isinstance(x, str):
                vals.append(latex_escape(x))
            elif pd.isna(x):
                vals.append("")
            else:
                vals.append(fmt(x))
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    body = "\n".join(lines)
    tex = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}\n"
        "\\end{table}\n"
    )
    out_path.write_text(tex)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="fast")
    p.add_argument("--shock", default="toy_smoke")
    p.add_argument("--input-dir", default="../output")
    p.add_argument("--output-dir", default="../output")
    args = p.parse_args()

    shock = slug(args.shock)
    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = pd.read_csv(in_dir / f"key_econ_timeseries_{args.profile}_{shock}.csv")
    selected = pd.read_csv(in_dir / f"key_econ_selected_t_{args.profile}_{shock}.csv")
    windows = pd.read_csv(in_dir / f"key_econ_window_means_{args.profile}_{shock}.csv")

    metrics = [
        ("rw_cell_pct", "Real Wage (shock cell)", "pct"),
        ("emp_cell_pct", "Employment (shock cell)", "pct"),
        ("inflow_cell_pct", "Migration Inflow (shock cell)", "pct"),
        ("outflow_cell_pct", "Migration Outflow (shock cell)", "pct"),
        ("stay_prob_cell_pp", "Stay Probability (shock cell)", "pp"),
        ("y_cell_pct", "Value Function Proxy Y (shock cell)", "pct"),
        ("exports_cell_pct", "Exports (shock cell)", "pct"),
        ("exports_external_cell_pct", "External Exports (shock cell)", "pct"),
        ("imports_region_pct", "Imports (shock region)", "pct"),
        ("imports_external_region_pct", "External Imports (shock region)", "pct"),
        ("cell_import_share_region_pp", "Cell Import Share in Shock Region", "pp"),
        ("domestic_import_share_region_pp", "Domestic Import Share in Shock Region", "pp"),
    ]
    available_metrics = [m for m in metrics if m[0] in ts.columns]
    if not available_metrics:
        raise ValueError("No expected key-econ metric columns were found in the input timeseries CSV.")

    ncols = 3
    nrows = max(1, (len(available_metrics) + ncols - 1) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows))
    axs = axs.flatten() if hasattr(axs, "flatten") else [axs]
    for i, (col, title, unit) in enumerate(available_metrics):
        axs[i].plot(ts["t"], ts[col], linewidth=1.8)
        axs[i].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axs[i].set_title(title)
        axs[i].set_xlabel("t")
        axs[i].set_ylabel("pp vs identity" if unit == "pp" else "% change vs identity")
        axs[i].grid(alpha=0.25)
    for j in range(len(available_metrics), len(axs)):
        axs[j].axis("off")
    fig.suptitle(f"Key Economic Variable Responses: profile={args.profile}, shock={shock}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / f"key_econ_impacts_{args.profile}_{shock}.pdf")
    plt.close(fig)

    early = ts[ts["t"] <= 30].copy()
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    early_priority = [
        "rw_cell_pct",
        "emp_cell_pct",
        "inflow_cell_pct",
        "outflow_cell_pct",
        "exports_cell_pct",
        "imports_region_pct",
    ]
    early_metrics = [m for m in available_metrics if m[0] in early_priority][:6]
    for col, title, unit in early_metrics:
        ax2.plot(early["t"], early[col], linewidth=1.8, label=title)
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax2.set_title("Early-Period Key Responses (t<=30)")
    ax2.set_xlabel("t")
    ax2.set_ylabel("Change vs identity")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False)
    fig2.tight_layout()
    fig2.savefig(out_dir / f"key_econ_impacts_early_{args.profile}_{shock}.pdf")
    plt.close(fig2)

    write_table(
        selected,
        out_dir / f"table_key_econ_selected_t_{args.profile}_{shock}.tex",
        caption=f"Key economic variable percent changes at selected times ({args.profile}, {shock}).",
        label=f"tab:key_econ_selected_{args.profile}_{shock}",
    )
    write_table(
        windows,
        out_dir / f"table_key_econ_window_means_{args.profile}_{shock}.tex",
        caption=f"Window means of key economic variable percent changes ({args.profile}, {shock}).",
        label=f"tab:key_econ_window_{args.profile}_{shock}",
    )

    print(f"Wrote key economic impact artifacts to {out_dir}")


if __name__ == "__main__":
    main()
