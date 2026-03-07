#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[1] / "temp" / "mpl").resolve()))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = ROOT / "intermediate"
OUTPUT_DIR = ROOT / "output"
MAT_PATH = Path(__file__).resolve().parents[3] / "CDP_codes_four_sectors" / "Base_year_four_sectors.mat"

SECTOR_ORDER = {
    1: "Manufacturing",
    2: "Construction",
    3: "Wholesale/Retail",
    4: "Services",
}


def log(message: str) -> None:
    print(message, flush=True)


def format_table(df: pd.DataFrame, columns: list[str]) -> str:
    display = df[columns].copy()
    for column in display.columns:
        if display[column].dtype.kind in {"f", "i"}:
            if column.endswith("idx"):
                display[column] = display[column].map(lambda x: f"{int(x)}")
            else:
                display[column] = display[column].map(lambda x: f"{float(x):.3f}")
    return display.to_string(index=False)


def make_panel_page(matrix: pd.DataFrame, national: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.22)

    ax1 = fig.add_subplot(grid[0, 0])
    national = national.sort_values("sector_idx")
    ax1.bar(national["sector_name"], national["cognitive_intensity_raw"], color=["#8c510a", "#d8b365", "#5ab4ac", "#01665e"])
    ax1.set_title("Panel 1. National Sector Cognitive Intensity")
    ax1.set_ylabel("Raw cognitive intensity")
    ax1.tick_params(axis="x", rotation=20)

    ax2 = fig.add_subplot(grid[0, 1])
    services = matrix[matrix["sector_idx"] == 4].sort_values("cognitive_intensity_raw")
    ax2.barh(services["state_abbr"], services["cognitive_intensity_raw"], color="#01665e")
    ax2.set_title("Panel 2. Services Intensity by State")
    ax2.set_xlabel("Raw cognitive intensity")

    ax3 = fig.add_subplot(grid[1, 0])
    manufacturing = matrix[matrix["sector_idx"] == 1].sort_values("cognitive_intensity_raw")
    ax3.barh(manufacturing["state_abbr"], manufacturing["cognitive_intensity_raw"], color="#8c510a")
    ax3.set_title("Panel 3. Manufacturing Intensity by State")
    ax3.set_xlabel("Raw cognitive intensity")

    ax4 = fig.add_subplot(grid[1, 1])
    state_order = (
        matrix.groupby(["state_idx", "state_abbr"], as_index=False)["cognitive_intensity_raw"]
        .mean()
        .sort_values("cognitive_intensity_raw", ascending=False)
    )
    heatmap = (
        matrix.merge(state_order[["state_idx"]], on="state_idx")
        .pivot(index="sector_name", columns="state_abbr", values="cognitive_intensity_raw")
        .reindex(index=[SECTOR_ORDER[idx] for idx in sorted(SECTOR_ORDER)])
    )
    heatmap = heatmap[state_order["state_abbr"].tolist()]
    image = ax4.imshow(heatmap.values, aspect="auto", cmap="YlGnBu")
    ax4.set_title("Panel 4. State-Sector Heatmap")
    ax4.set_yticks(range(len(heatmap.index)))
    ax4.set_yticklabels(heatmap.index)
    ax4.set_xticks(range(len(heatmap.columns)))
    ax4.set_xticklabels(heatmap.columns, rotation=90, fontsize=7)
    fig.colorbar(image, ax=ax4, fraction=0.046, pad=0.04)

    return fig


def make_table_and_hist_page(matrix: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.4, 1.0], hspace=0.25)
    ax_text = fig.add_subplot(grid[0, 0])
    ax_hist = fig.add_subplot(grid[1, 0])

    top10 = matrix.nlargest(10, "cognitive_intensity_raw")[
        ["state_abbr", "sector_name", "cognitive_intensity_raw", "cognitive_intensity"]
    ]
    bottom10 = matrix.nsmallest(10, "cognitive_intensity_raw")[
        ["state_abbr", "sector_name", "cognitive_intensity_raw", "cognitive_intensity"]
    ]
    services = matrix[matrix["sector_idx"] == 4].sort_values("cognitive_intensity_raw", ascending=False)
    text = "\n".join(
        [
            "Panel 5. Top and Bottom State-Sector Pairs",
            "",
            "Top 10 state-sector pairs",
            format_table(top10, ["state_abbr", "sector_name", "cognitive_intensity_raw", "cognitive_intensity"]),
            "",
            "Bottom 10 state-sector pairs",
            format_table(bottom10, ["state_abbr", "sector_name", "cognitive_intensity_raw", "cognitive_intensity"]),
            "",
            "Services top 5 states",
            format_table(services.head(5), ["state_abbr", "cognitive_intensity_raw", "cognitive_intensity"]),
            "",
            "Services bottom 5 states",
            format_table(services.tail(5), ["state_abbr", "cognitive_intensity_raw", "cognitive_intensity"]),
        ]
    )
    ax_text.axis("off")
    ax_text.text(0.0, 1.0, text, family="monospace", fontsize=8, va="top")

    lambda_hat = 1.0 + 0.05 * matrix["cognitive_intensity"]
    ax_hist.hist(lambda_hat, bins=20, color="#4c78a8", edgecolor="white")
    ax_hist.set_title("Panel 6. Implied Lambda-Hat Distribution (delta = 5%)")
    ax_hist.set_xlabel("lambda_hat")
    ax_hist.set_ylabel("Count")
    ax_hist.axvline(lambda_hat.mean(), color="black", linestyle="--", linewidth=1.0)

    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "temp" / "mpl").mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(OUTPUT_DIR / "cognitive_intensity_matrix.csv")
    national = pd.read_csv(INTERMEDIATE_DIR / "national_sector_cognitive.csv")
    rankings_path = OUTPUT_DIR / "cognitive_intensity_rankings.txt"

    services = matrix[matrix["sector_idx"] == 4].sort_values("cognitive_intensity_raw", ascending=False)
    manufacturing = matrix[matrix["sector_idx"] == 1].sort_values("cognitive_intensity_raw", ascending=False)
    mfg_by_state = manufacturing.set_index("state_idx")["cognitive_intensity_raw"].sort_index()
    svc_by_state = services.set_index("state_idx")["cognitive_intensity_raw"].sort_index()
    corr_mfg_svc = float(np.corrcoef(mfg_by_state.values, svc_by_state.values)[0, 1])

    mat = loadmat(MAT_PATH)
    l0 = np.asarray(mat["L0"], dtype=float)
    services_share = l0[3, :] / l0.sum(axis=0)
    corr_services_share = float(np.corrcoef(svc_by_state.values, services_share)[0, 1])

    national_map = national.set_index("sector_name")["cognitive_intensity_raw"].to_dict()
    services_top5 = services.head(5)["state_abbr"].tolist()
    services_bottom5 = services.tail(5)["state_abbr"].tolist()
    services_high = services.iloc[0]
    services_low = services.iloc[-1]
    mfg_high = manufacturing.iloc[0]
    mfg_low = manufacturing.iloc[-1]

    services_check = "PASS" if any(state in services_top5 for state in ["NY", "MA", "CT"]) else "FAIL"
    services_bottom_check = "PASS" if any(state in services_bottom5 for state in ["NV", "HI"]) else "FAIL"
    national_check = "PASS" if national_map["Services"] > national_map["Manufacturing"] else "FAIL"
    mfg_variation_check = "PASS" if manufacturing["cognitive_intensity_raw"].std() > 0.01 else "FAIL"
    services_ratio = float(services_high["cognitive_intensity_raw"] / services_low["cognitive_intensity_raw"])

    lines = [
        "=== SANITY CHECKS ===",
        "1. National cognitive intensity by sector:",
        f"   Manufacturing: {national_map['Manufacturing']:.2f}",
        f"   Construction: {national_map['Construction']:.2f}",
        f"   Wholesale/Retail: {national_map['Wholesale/Retail']:.2f}",
        f"   Services: {national_map['Services']:.2f}",
        f"   CHECK: Services > Manufacturing? [{national_check}]",
        "",
        "2. State variation in services:",
        f"   Highest: {services_high['state_abbr']} (state) = {services_high['cognitive_intensity_raw']:.2f}",
        f"   Lowest:  {services_low['state_abbr']} (state) = {services_low['cognitive_intensity_raw']:.2f}",
        f"   Ratio: {services_ratio:.1f}x",
        f"   CHECK: NY or MA or CT in top 5? [{services_check}]",
        f"   CHECK: NV or HI in bottom 5? [{services_bottom_check}]",
        "",
        "3. State variation in manufacturing:",
        f"   Highest: {mfg_high['state_abbr']} (state) = {mfg_high['cognitive_intensity_raw']:.2f}",
        f"   Lowest:  {mfg_low['state_abbr']} (state) = {mfg_low['cognitive_intensity_raw']:.2f}",
        f"   CHECK: Variation exists (std > 0.01)? [{mfg_variation_check}]",
        "",
        "4. Cross-sector correlation:",
        f"   Corr(mfg_intensity, svc_intensity) = {corr_mfg_svc:.2f}",
        "   (Expected: moderate positive - states with more cognitive mfg also tend to have more cognitive services)",
        "",
        "5. Correlation with known economic indicators:",
        f"   Corr(svc_intensity, services_employment_share) = {corr_services_share:.2f}",
        "   (Use L0 from Base_year_four_sectors.mat for employment shares)",
        "",
        "Top 10 state-sector pairs by cognitive intensity:",
        format_table(matrix.nlargest(10, "cognitive_intensity_raw"), ["state_abbr", "sector_name", "cognitive_intensity_raw", "cognitive_intensity"]),
        "",
        "Bottom 10 state-sector pairs by cognitive intensity:",
        format_table(matrix.nsmallest(10, "cognitive_intensity_raw"), ["state_abbr", "sector_name", "cognitive_intensity_raw", "cognitive_intensity"]),
        "",
        "Services top 5 states:",
        format_table(services.head(5), ["state_abbr", "cognitive_intensity_raw", "cognitive_intensity"]),
        "",
        "Services bottom 5 states:",
        format_table(services.tail(5), ["state_abbr", "cognitive_intensity_raw", "cognitive_intensity"]),
    ]
    summary_text = "\n".join(lines) + "\n"
    print(summary_text, end="")
    rankings_path.write_text(summary_text)

    pdf_path = OUTPUT_DIR / "cognitive_intensity_summary.pdf"
    with PdfPages(pdf_path) as pdf:
        fig1 = make_panel_page(matrix, national)
        pdf.savefig(fig1)
        plt.close(fig1)

        fig2 = make_table_and_hist_page(matrix)
        pdf.savefig(fig2)
        plt.close(fig2)

    log(f"Wrote summary PDF to {pdf_path}")


if __name__ == "__main__":
    main()
