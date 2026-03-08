#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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


def format_table(df: pd.DataFrame, columns: list[str], max_rows: int = 10) -> str:
    if df.empty:
        return "(no rows)"
    return df.loc[:, columns].head(max_rows).to_string(index=False, float_format=lambda x: f"{x:,.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a PDF summary for broad state-sector shock impacts.")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--shock", default="generic")
    parser.add_argument("--input-dir", default="../output")
    parser.add_argument("--output-dir", default="../output")
    args = parser.parse_args()

    profile = args.profile
    shock = slug(args.shock)
    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = pd.read_csv(resolve_file(in_dir, "broad_shock_timeseries", profile, shock))
    sector = pd.read_csv(resolve_file(in_dir, "broad_shock_sector_timeseries", profile, shock))
    selected = pd.read_csv(resolve_file(in_dir, "broad_shock_state_sector_selected_t", profile, shock))
    rankings = pd.read_csv(resolve_file(in_dir, "broad_shock_state_rankings", profile, shock))

    ranking_t = int(rankings["t"].iloc[0])
    top_emp = rankings.nlargest(10, "agg_emp_pct")[["state_abbr", "agg_emp_pct"]]
    bottom_emp = rankings.nsmallest(10, "agg_emp_pct")[["state_abbr", "agg_emp_pct"]]
    top_rw = rankings.nlargest(10, "agg_rw_pct")[["state_abbr", "agg_rw_pct"]]
    bottom_rw = rankings.nsmallest(10, "agg_rw_pct")[["state_abbr", "agg_rw_pct"]]

    final_t = ranking_t if ranking_t in selected["t"].unique() else int(selected["t"].max())
    final_slice = selected[selected["t"] == final_t].copy()
    largest_emp_cells = final_slice.reindex(final_slice["emp_pct"].abs().sort_values(ascending=False).index).head(10)
    largest_rw_cells = final_slice.reindex(final_slice["rw_pct"].abs().sort_values(ascending=False).index).head(10)

    pdf_path = out_dir / f"broad_shock_summary_{profile}_{shock}.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes[0, 0].plot(ts["t"], ts["agg_emp_pct"], linewidth=1.8, color="#8c510a")
        axes[0, 0].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axes[0, 0].set_title("Aggregate Employment Effect")
        axes[0, 0].set_xlabel("t")
        axes[0, 0].set_ylabel("Percent change")
        axes[0, 0].grid(alpha=0.25)

        axes[0, 1].plot(ts["t"], ts["agg_rw_pct"], linewidth=1.8, color="#01665e")
        axes[0, 1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axes[0, 1].set_title("Aggregate Real Wage Effect")
        axes[0, 1].set_xlabel("t")
        axes[0, 1].set_ylabel("Percent change")
        axes[0, 1].grid(alpha=0.25)

        for sector_name, group in sector.groupby("sector_name", sort=False):
            axes[1, 0].plot(group["t"], group["emp_pct"], linewidth=1.6, label=sector_name)
            axes[1, 1].plot(group["t"], group["rw_pct"], linewidth=1.6, label=sector_name)
        axes[1, 0].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axes[1, 0].set_title("Sector Employment Effects")
        axes[1, 0].set_xlabel("t")
        axes[1, 0].set_ylabel("Percent change")
        axes[1, 0].grid(alpha=0.25)
        axes[1, 0].legend(frameon=False, fontsize=8)

        axes[1, 1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axes[1, 1].set_title("Sector Real Wage Effects")
        axes[1, 1].set_xlabel("t")
        axes[1, 1].set_ylabel("Percent change")
        axes[1, 1].grid(alpha=0.25)
        axes[1, 1].legend(frameon=False, fontsize=8)

        fig.suptitle(f"Broad Shock Summary: profile={profile}, shock={shock}", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes2[0, 0].barh(top_emp["state_abbr"], top_emp["agg_emp_pct"], color="#5ab4ac")
        axes2[0, 0].set_title(f"Top 10 States by Employment Effect (t={ranking_t})")
        axes2[0, 0].set_xlabel("Percent change")

        axes2[0, 1].barh(bottom_emp["state_abbr"], bottom_emp["agg_emp_pct"], color="#d8b365")
        axes2[0, 1].set_title(f"Bottom 10 States by Employment Effect (t={ranking_t})")
        axes2[0, 1].set_xlabel("Percent change")

        axes2[1, 0].barh(top_rw["state_abbr"], top_rw["agg_rw_pct"], color="#5ab4ac")
        axes2[1, 0].set_title(f"Top 10 States by Real Wage Effect (t={ranking_t})")
        axes2[1, 0].set_xlabel("Percent change")

        axes2[1, 1].barh(bottom_rw["state_abbr"], bottom_rw["agg_rw_pct"], color="#d8b365")
        axes2[1, 1].set_title(f"Bottom 10 States by Real Wage Effect (t={ranking_t})")
        axes2[1, 1].set_xlabel("Percent change")

        pdf.savefig(fig2)
        plt.close(fig2)

        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
        axes3[0].axis("off")
        axes3[0].text(
            0.0,
            1.0,
            "Largest absolute state-sector employment effects\n"
            f"(selected time t={final_t})\n\n"
            + format_table(largest_emp_cells, ["state_abbr", "sector_name", "emp_pct"]),
            va="top",
            ha="left",
            family="monospace",
            fontsize=8,
        )
        axes3[1].axis("off")
        axes3[1].text(
            0.0,
            1.0,
            "Largest absolute state-sector real wage effects\n"
            f"(selected time t={final_t})\n\n"
            + format_table(largest_rw_cells, ["state_abbr", "sector_name", "rw_pct"]),
            va="top",
            ha="left",
            family="monospace",
            fontsize=8,
        )
        pdf.savefig(fig3)
        plt.close(fig3)

    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
