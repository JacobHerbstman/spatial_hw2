#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "state_idx",
    "state_abbr",
    "sector_idx",
    "sector_name",
    "cognitive_intensity",
    "cognitive_intensity_raw",
]

SCENARIOS = {
    "immediate": {
        "description": "Permanent AI productivity shock active from t=1 through the final shock period.",
        "start_t": 1,
        "filename": "lambda_immediate.csv",
        "shock_name": "cognitive_immediate",
    },
    "anticipated": {
        "description": "Permanent AI productivity shock announced at t=0 and activated at t=21.",
        "start_t": 21,
        "filename": "lambda_anticipated.csv",
        "shock_name": "cognitive_anticipated",
    },
}


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_matrix(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise RuntimeError(f"Matrix is missing required columns: {missing}")
    if len(df) != 200:
        raise RuntimeError(f"Expected 200 rows in matrix, found {len(df)}")
    if df[["state_idx", "sector_idx"]].duplicated().any():
        raise RuntimeError("Matrix has duplicate (state_idx, sector_idx) rows")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise RuntimeError("Matrix has missing values in required columns")

    out = df.copy()
    out["state_idx"] = pd.to_numeric(out["state_idx"], errors="raise").astype(int)
    out["sector_idx"] = pd.to_numeric(out["sector_idx"], errors="raise").astype(int)
    out["cognitive_intensity"] = pd.to_numeric(out["cognitive_intensity"], errors="raise")
    out["cognitive_intensity_raw"] = pd.to_numeric(out["cognitive_intensity_raw"], errors="raise")

    if sorted(out["state_idx"].unique().tolist()) != list(range(1, 51)):
        raise RuntimeError("Matrix state_idx values must be exactly 1..50")
    if sorted(out["sector_idx"].unique().tolist()) != [1, 2, 3, 4]:
        raise RuntimeError("Matrix sector_idx values must be exactly 1..4")
    if ((out["cognitive_intensity"] < 0.0) | (out["cognitive_intensity"] > 1.0)).any():
        raise RuntimeError("Matrix cognitive_intensity values must lie in [0, 1]")

    return out.sort_values(["state_idx", "sector_idx"]).reset_index(drop=True)


def default_metadata_path(matrix_path: Path) -> Path:
    return matrix_path.with_name("cognitive_intensity_metadata.json")


def default_national_sector_path(matrix_path: Path) -> Path:
    return matrix_path.parents[1] / "intermediate" / "national_sector_cognitive.csv"


def load_required_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Required upstream metadata file is missing: {path}")
    return json.loads(path.read_text())


def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Required upstream CSV file is missing: {path}")
    return pd.read_csv(path)


def format_table_rows(df: pd.DataFrame, value_col: str, limit: int, label: str) -> list[str]:
    rows = []
    for row in df.head(limit).itertuples(index=False):
        rows.append(f"  {row.state_abbr}-{row.sector_name}: {label}={getattr(row, value_col):.4f}")
    return rows


def build_lambda_rows(snapshot: pd.DataFrame, start_t: int, end_t: int) -> pd.DataFrame:
    active = snapshot[snapshot["lambda_value"] > 1.0].copy()
    rows: list[dict[str, float | int]] = []
    for row in active.itertuples(index=False):
        for t in range(start_t, end_t + 1):
            rows.append(
                {
                    "t": t,
                    "j": int(row.sector_idx),
                    "n": int(row.state_idx),
                    "value": float(row.lambda_value),
                }
            )
    return pd.DataFrame(rows, columns=["t", "j", "n", "value"])


def build_data_notes(
    matrix: pd.DataFrame,
    metadata: dict,
    national_sector: pd.DataFrame,
) -> str:
    coverage = metadata.get("qcew_coverage", {})
    overall_cdp = coverage.get("overall_cdp_sectors", {})
    by_sector = pd.DataFrame(coverage.get("by_sector", []))
    if by_sector.empty:
        raise RuntimeError("Upstream metadata is missing qcew_coverage.by_sector")
    if national_sector.empty:
        raise RuntimeError("National sector cognitive CSV is empty")

    national = national_sector.copy()
    national["sector_name"] = national["sector_name"].astype(str)
    national["cognitive_intensity_raw"] = pd.to_numeric(national["cognitive_intensity_raw"], errors="raise")
    national_values = national.set_index("sector_name")["cognitive_intensity_raw"].to_dict()
    sector_std = (
        matrix.groupby("sector_name")["cognitive_intensity"]
        .std()
        .rename("std")
        .to_dict()
    )
    wholesale_cov = by_sector.loc[by_sector["sector_name"] == "Wholesale/Retail", "prefix_match_share"]
    wholesale_cov_value = float(wholesale_cov.iloc[0]) if not wholesale_cov.empty else float("nan")
    prefix_share = float(overall_cdp.get("prefix_match_share", float("nan")))
    construction_raw = float(national_values["Construction"])
    services_raw = float(national_values["Services"])
    construction_std = float(sector_std["Construction"])
    services_std = float(sector_std["Services"])

    return "\n".join(
        [
            "# Cognitive Non-Routine Intensity Data Notes",
            "",
            "## Task Classification",
            "Occupations are classified by cognitive non-routine (CNR) task intensity following",
            "Acemoglu and Autor (2011, Handbook of Labor Economics, Table 2), using O*NET Work",
            "Activities importance scores for five items:",
            "- Analyzing Data or Information (4.A.2.a.4)",
            "- Thinking Creatively (4.A.2.b.2)",
            "- Interpreting the Meaning of Information for Others (4.A.4.a.1)",
            "- Establishing and Maintaining Interpersonal Relationships (4.A.4.a.4)",
            "- Guiding, Directing, and Motivating Subordinates (4.A.4.b.4)",
            "",
            "This is the classification used by Rossi-Hansberg, Sarte, and Schwartzman (2023)",
            "and Autor and Dorn (2013).",
            "",
            "## State-Sector Variation",
            "State-sector cognitive intensity is computed by weighting detailed NAICS industry",
            "employment (BLS QCEW 2023) by national industry-level cognitive scores (BLS OES x",
            f"O*NET 2023). Coverage of ~{100.0 * prefix_share:.1f}% of CDP-sector employment reflects",
            "NAICS granularity mismatch between QCEW and OES files. State rankings within sectors",
            "are robust to this coverage limitation.",
            "",
            "## Known Limitations",
            (
                f"1. National sector ordering shows Construction ~= Services "
                f"({construction_raw:.3f} vs {services_raw:.3f}). This reflects genuine CNR task "
                "content in skilled trades (project management, planning). Construction has minimal "
                f"geographic variation (std={construction_std:.3f} vs Services std={services_std:.3f}) "
                "so it does not drive cross-state heterogeneity in results."
            ),
            (
                f"2. Wholesale/Retail has the lowest coverage (~{100.0 * wholesale_cov_value:.1f}% "
                "prefix match) and the narrowest cross-state variation."
            ),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate public cognitive counterfactual lambda CSVs.")
    script_dir = Path(__file__).resolve().parent
    default_matrix = script_dir.parents[1] / "cognitive_intensity_data" / "output" / "cognitive_intensity_matrix.csv"
    parser.add_argument("--matrix", default=str(default_matrix))
    parser.add_argument("--output-dir", default=str(script_dir.parent / "output"))
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--time-horizon", type=int, default=200)
    parser.add_argument("--metadata-path", default="")
    parser.add_argument("--national-sector-path", default="")
    args = parser.parse_args()

    matrix_path = Path(args.matrix).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.time_horizon < 23:
        raise RuntimeError("time_horizon must be at least 23 so the anticipated scenario can begin at t=21")

    matrix = validate_matrix(pd.read_csv(matrix_path))
    matrix["lambda_value"] = 1.0 + args.delta * matrix["cognitive_intensity"]
    metadata_path = Path(args.metadata_path).expanduser().resolve() if args.metadata_path else default_metadata_path(matrix_path)
    national_sector_path = (
        Path(args.national_sector_path).expanduser().resolve()
        if args.national_sector_path
        else default_national_sector_path(matrix_path)
    )
    metadata = load_required_json(metadata_path)
    national_sector = load_required_csv(national_sector_path)

    shock_periods = args.time_horizon - 2
    positive = matrix[matrix["lambda_value"] > 1.0].copy()
    zero_cells = int((matrix["lambda_value"] == 1.0).sum())
    positive_lambda_min = float(positive["lambda_value"].min()) if not positive.empty else 1.0
    positive_lambda_max = float(positive["lambda_value"].max()) if not positive.empty else 1.0
    strongest = matrix.sort_values(["lambda_value", "state_abbr", "sector_idx"], ascending=[False, True, True]).head(10)
    weakest_nontrivial = positive.sort_values(["lambda_value", "state_abbr", "sector_idx"], ascending=[True, True, True]).head(10)

    calibration: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_path": str(matrix_path),
        "matrix_checksum_sha256": sha256_file(matrix_path),
        "metadata_path": str(metadata_path),
        "national_sector_path": str(national_sector_path),
        "delta": args.delta,
        "time_horizon": args.time_horizon,
        "shock_periods": shock_periods,
        "matrix_summary": {
            "row_count": int(len(matrix)),
            "positive_cell_count": int(len(positive)),
            "zero_shock_cell_count": zero_cells,
            "cognitive_intensity_min": float(matrix["cognitive_intensity"].min()),
            "cognitive_intensity_max": float(matrix["cognitive_intensity"].max()),
            "lambda_min": float(matrix["lambda_value"].min()),
            "lambda_max": float(matrix["lambda_value"].max()),
            "lambda_nontrivial_min": positive_lambda_min if not positive.empty else None,
            "lambda_nontrivial_max": positive_lambda_max if not positive.empty else None,
        },
        "scenario_descriptions": {},
        "strongest_shocks": strongest[
            ["state_idx", "state_abbr", "sector_idx", "sector_name", "cognitive_intensity", "lambda_value"]
        ].to_dict(orient="records"),
        "weakest_nontrivial_shocks": weakest_nontrivial[
            ["state_idx", "state_abbr", "sector_idx", "sector_name", "cognitive_intensity", "lambda_value"]
        ].to_dict(orient="records"),
        "upstream_metadata": metadata,
    }

    for scenario_key, config in SCENARIOS.items():
        start_t = int(config["start_t"])
        end_t = shock_periods
        lambda_df = build_lambda_rows(matrix, start_t, end_t)
        lambda_path = output_dir / str(config["filename"])
        lambda_df.to_csv(lambda_path, index=False)
        calibration["scenario_descriptions"][scenario_key] = {
            "shock_name": config["shock_name"],
            "description": config["description"],
            "active_period_start": start_t,
            "active_period_end": end_t,
            "row_count": int(len(lambda_df)),
            "lambda_csv_path": str(lambda_path),
        }

    notes_path = output_dir / "data_notes.md"
    notes_path.write_text(build_data_notes(matrix, metadata, national_sector))
    calibration_path = output_dir / "shock_calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2))

    immediate_rows = calibration["scenario_descriptions"]["immediate"]["row_count"]
    anticipated_rows = calibration["scenario_descriptions"]["anticipated"]["row_count"]
    strongest_lines = format_table_rows(strongest, "lambda_value", 5, "lambda")
    weakest_lines = format_table_rows(weakest_nontrivial, "lambda_value", 5, "lambda")

    log("=== SHOCK CALIBRATION ===")
    log(f"delta = {args.delta:.2f}")
    log(
        "Immediate scenario: "
        f"{immediate_rows} rows, t=1 to {shock_periods}, "
        f"lambda range [{positive_lambda_min:.4f}, {positive_lambda_max:.4f}]"
    )
    log(
        "Anticipated scenario: "
        f"{anticipated_rows} rows, t=21 to {shock_periods}, "
        f"lambda range [{positive_lambda_min:.4f}, {positive_lambda_max:.4f}]"
    )
    log("")
    log("Strongest shocks (by lambda):")
    for line in strongest_lines:
        log(line)
    log("")
    log("Weakest non-trivial shocks:")
    for line in weakest_lines:
        log(line)
    log("")
    log(f"Cells with lambda=1.0 (no shock): {zero_cells} out of {len(matrix)}")
    log(f"Wrote {output_dir / 'lambda_immediate.csv'}")
    log(f"Wrote {output_dir / 'lambda_anticipated.csv'}")
    log(f"Wrote {calibration_path}")
    log(f"Wrote {notes_path}")


if __name__ == "__main__":
    main()
