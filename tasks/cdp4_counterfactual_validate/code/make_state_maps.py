#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
import pandas as pd

DEFAULT_GEOMETRY_FILE = (
    Path(__file__).resolve().parents[1]
    / "input"
    / "us_states_census_2024"
    / "cb_2024_us_state_500k.shp"
)


STATE_GRID_POSITIONS = {
    "WA": (0, 0), "MT": (0, 2), "ND": (0, 4), "MN": (0, 5), "WI": (0, 6), "MI": (0, 7), "VT": (0, 9), "ME": (0, 10),
    "OR": (1, 0), "ID": (1, 1), "WY": (1, 2), "SD": (1, 4), "IA": (1, 5), "IL": (1, 6), "IN": (1, 7), "OH": (1, 8), "PA": (1, 9), "NY": (1, 10),
    "CA": (2, 0), "NV": (2, 1), "UT": (2, 2), "CO": (2, 3), "NE": (2, 4), "MO": (2, 5), "KY": (2, 7), "WV": (2, 8), "VA": (2, 9), "MD": (2, 10),
    "AZ": (3, 1), "NM": (3, 3), "KS": (3, 4), "AR": (3, 5), "TN": (3, 7), "NC": (3, 9), "SC": (3, 10), "DE": (3, 11),
    "AK": (4, 0), "HI": (4, 1), "OK": (4, 4), "LA": (4, 5), "MS": (4, 6), "AL": (4, 7), "GA": (4, 8), "FL": (4, 10), "NJ": (4, 11),
    "TX": (5, 3), "CT": (5, 10), "RI": (5, 11), "MA": (6, 10), "NH": (6, 11),
}

STATE_NAME_TO_ABBR = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA",
    "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT",
    "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM",
    "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
}

TIME_SNAPSHOTS = [2, 10, 50, 199]


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


def choose_renderer(renderer: str, geometry_file: str) -> str:
    requested = renderer.lower()
    geometry_ok = bool(geometry_file) and Path(geometry_file).exists()
    if requested == "grid":
        return "grid"
    if requested == "geopandas":
        if not geometry_ok:
            raise FileNotFoundError("Renderer `geopandas` requires --geometry-file pointing to a readable local geometry file.")
        try:
            import geopandas  # noqa: F401
        except Exception as exc:
            raise RuntimeError("Renderer `geopandas` requested but geopandas is unavailable.") from exc
        return "geopandas"
    if requested == "auto":
        if geometry_ok:
            try:
                import geopandas  # noqa: F401
                return "geopandas"
            except Exception:
                pass
        return "grid"
    raise ValueError(f"Unsupported renderer={renderer}. Use auto, grid, or geopandas.")


def normalize_abs_max(series: pd.Series) -> float:
    finite = series[pd.notna(series)]
    if finite.empty:
        return 1e-6
    vmax = float(finite.abs().max())
    return max(vmax, 1e-6)


def draw_grid_panel(ax, panel_df: pd.DataFrame, value_col: str, title: str, cmap_name: str = "RdBu_r") -> None:
    vmax = normalize_abs_max(panel_df[value_col])
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    max_row = max(pos[0] for pos in STATE_GRID_POSITIONS.values())
    max_col = max(pos[1] for pos in STATE_GRID_POSITIONS.values())
    values = {getattr(row, "state_abbr"): getattr(row, value_col) for row in panel_df.itertuples(index=False)}

    for abbr, (row, col) in STATE_GRID_POSITIONS.items():
        val = values.get(abbr, float("nan"))
        face = "#f0f0f0" if pd.isna(val) else cmap(norm(val))
        rect = Rectangle((col, row), 1.0, 1.0, facecolor=face, edgecolor="black", linewidth=0.8)
        ax.add_patch(rect)
        label = abbr if pd.isna(val) else f"{abbr}\n{val:.2f}"
        ax.text(col + 0.5, row + 0.5, label, ha="center", va="center", fontsize=6)

    ax.set_xlim(0, max_col + 1)
    ax.set_ylim(max_row + 1, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)


def _canonicalize_abbr(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip().upper()
    if s in STATE_GRID_POSITIONS:
        return s
    return STATE_NAME_TO_ABBR.get(s)


def _load_geometry(geometry_file: str):
    import geopandas as gpd

    gdf = gpd.read_file(geometry_file)
    abbr_cols = ["STUSPS", "state_abbr", "STATE_ABBR", "postal", "POSTAL", "USPS", "abbr"]
    for col in abbr_cols:
        if col in gdf.columns:
            gdf["state_abbr"] = gdf[col].map(_canonicalize_abbr)
            break
    else:
        name_cols = ["NAME", "STATE_NAME", "name", "state"]
        for col in name_cols:
            if col in gdf.columns:
                gdf["state_abbr"] = gdf[col].map(_canonicalize_abbr)
                break
        else:
            raise ValueError("Could not find a state abbreviation/name column in the geometry file.")

    gdf = gdf[gdf["state_abbr"].isin(STATE_GRID_POSITIONS.keys())].copy()
    if gdf.empty:
        raise ValueError("Geometry file did not resolve to any of the 50 US state abbreviations.")
    gdf = gdf.dissolve(by="state_abbr", as_index=False)
    return gdf


def _plot_geo_layer(ax, gdf, value_col: str, norm, cmap_name: str) -> None:
    gdf.plot(
        ax=ax,
        column=value_col,
        cmap=cmap_name,
        norm=norm,
        edgecolor="black",
        linewidth=0.4,
        missing_kwds={"color": "#f0f0f0", "edgecolor": "black", "linewidth": 0.4},
    )
    ax.set_axis_off()


def draw_geopandas_panel(ax, panel_df: pd.DataFrame, value_col: str, title: str, geometry_file: str,
                         cmap_name: str = "RdBu_r") -> None:
    vmax = normalize_abs_max(panel_df[value_col])
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    gdf = _load_geometry(geometry_file).merge(panel_df[["state_abbr", value_col]], on="state_abbr", how="left")
    contiguous = gdf[~gdf["state_abbr"].isin(["AK", "HI"])].copy()
    alaska = gdf[gdf["state_abbr"] == "AK"].copy()
    hawaii = gdf[gdf["state_abbr"] == "HI"].copy()

    _plot_geo_layer(ax, contiguous, value_col, norm, cmap_name)
    ax.set_title(title, fontsize=10)

    if not alaska.empty:
        ax_ak = ax.inset_axes([0.02, 0.02, 0.22, 0.22])
        _plot_geo_layer(ax_ak, alaska, value_col, norm, cmap_name)
        ax_ak.set_title("AK", fontsize=7, pad=1)
    if not hawaii.empty:
        ax_hi = ax.inset_axes([0.25, 0.02, 0.12, 0.12])
        _plot_geo_layer(ax_hi, hawaii, value_col, norm, cmap_name)
        ax_hi.set_title("HI", fontsize=7, pad=1)

    sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)


def render_map_figure(df: pd.DataFrame, times: list[int], rows: list[tuple[str, str]], title: str,
                      out_path: Path, renderer: str, geometry_file: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for row_idx, (value_col, label) in enumerate(rows):
        for col_idx, t in enumerate(times):
            ax = axes[row_idx, col_idx]
            panel_df = df[df["t"] == t].copy()
            panel_title = f"{label}, t={t}"
            if renderer == "geopandas":
                draw_geopandas_panel(ax, panel_df, value_col, panel_title, geometry_file)
            else:
                draw_grid_panel(ax, panel_df, value_col, panel_title)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create state-level choropleth PDF reports from counterfactual CSV outputs.")
    parser.add_argument("--profile", default="reference")
    parser.add_argument("--shock", default="toy")
    parser.add_argument("--input-dir", default="../output")
    parser.add_argument("--output-dir", default="../output")
    parser.add_argument("--geometry-file", default="")
    parser.add_argument("--renderer", default="auto")
    args = parser.parse_args()

    profile = args.profile
    shock = slug(args.shock)
    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_file(in_dir, "state_maps_data", profile, shock)
    df = pd.read_csv(csv_path)
    required_cols = {
        "t", "state_idx", "state_abbr",
        "rw_pct_mfg", "rw_pct_con", "rw_pct_whr", "rw_pct_svc", "rw_pct_agg",
        "emp_pct_mfg", "emp_pct_con", "emp_pct_whr", "emp_pct_svc", "emp_pct_agg",
        "mfg_share_pp", "svc_share_pp", "net_migration_diff",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    available_times = sorted(int(x) for x in df["t"].dropna().unique())
    times = [t for t in TIME_SNAPSHOTS if t in available_times]
    if len(times) != len(TIME_SNAPSHOTS):
        missing_times = [t for t in TIME_SNAPSHOTS if t not in times]
        raise ValueError(f"Missing expected map time snapshots in CSV: {missing_times}")

    geometry_file = args.geometry_file
    if not geometry_file and DEFAULT_GEOMETRY_FILE.exists():
        geometry_file = str(DEFAULT_GEOMETRY_FILE)

    renderer = choose_renderer(args.renderer, geometry_file)

    render_map_figure(
        df,
        times,
        [("rw_pct_agg", "Aggregate"), ("rw_pct_svc", "Services")],
        "Real Wage Effects by State",
        out_dir / f"state_map_realwages_{profile}_{shock}.pdf",
        renderer,
        geometry_file,
    )
    render_map_figure(
        df,
        times,
        [("emp_pct_agg", "Aggregate"), ("emp_pct_mfg", "Manufacturing")],
        "Employment Effects by State",
        out_dir / f"state_map_employment_{profile}_{shock}.pdf",
        renderer,
        geometry_file,
    )
    render_map_figure(
        df,
        times,
        [("mfg_share_pp", "Manufacturing Share (pp)"), ("svc_share_pp", "Services Share (pp)")],
        "Sectoral Composition Shift by State",
        out_dir / f"state_map_sectoral_shift_{profile}_{shock}.pdf",
        renderer,
        geometry_file,
    )

    print(f"Wrote state map PDFs to {out_dir} using renderer={renderer}")


if __name__ == "__main__":
    main()
