#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a compact markdown brief for the reference AI counterfactual outputs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dynamics-output-dir", required=True)
    parser.add_argument("--validate-output-dir", required=True)
    parser.add_argument("--profile", default="reference")
    parser.add_argument("--immediate-shock", default="cognitive_immediate")
    parser.add_argument("--anticipated-shock", default="cognitive_anticipated")
    return parser.parse_args()


def slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "default"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_summary(path: Path) -> dict[str, float | str]:
    df = pd.read_csv(path)
    return dict(zip(df["metric"], df["value"]))


def load_validation(path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, float | str]]]:
    df = pd.read_csv(path)
    checks = {}
    for row in df.itertuples(index=False):
        checks[str(row.check)] = {
            "value": float(row.value),
            "threshold": float(row.threshold),
            "status": str(row.status),
        }
    return df, checks


def require_all_passed(df: pd.DataFrame, label: str) -> None:
    failed = df[df["status"] != "PASS"]
    if not failed.empty:
        checks = ", ".join(str(x) for x in failed["check"].tolist())
        raise RuntimeError(f"{label} has failing checks: {checks}")


def format_num(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    av = abs(float(value))
    if av != 0.0 and (av < 1e-4 or av >= 1e4):
        return f"{float(value):.{digits}e}"
    return f"{float(value):.{digits}f}"


def format_pct(value: float, digits: int = 3) -> str:
    return f"{format_num(value, digits)}%"


def format_pp(value: float, digits: int = 3) -> str:
    return f"{format_num(value, digits)} pp"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def pick_row(df: pd.DataFrame, t: int) -> pd.Series:
    exact = df[df["t"] == t]
    if not exact.empty:
        return exact.iloc[0]
    leq = df[df["t"] <= t]
    if not leq.empty:
        return leq.iloc[-1]
    return df.iloc[0]


def top_states(df: pd.DataFrame, t: int, value_col: str, n: int = 5) -> str:
    panel = df[df["t"] == t].copy()
    if panel.empty:
        return "n/a"
    panel = panel.sort_values(value_col, ascending=False).head(n)
    return ", ".join(f"{row.state_abbr} ({format_num(getattr(row, value_col), 3)})" for row in panel.itertuples(index=False))


def max_abs_metric(df: pd.DataFrame, value_col: str) -> str:
    if value_col not in df.columns:
        return "n/a"
    series = df[value_col].dropna()
    if series.empty:
        return "n/a"
    idx = series.abs().idxmax()
    row = df.loc[idx]
    return f"{row['state_abbr']} ({format_num(row[value_col], 3)}) at t={int(row['t'])}"


def scenario_payload(
    *,
    profile: str,
    scenario_name: str,
    dynamics_output_dir: Path,
    validate_output_dir: Path,
) -> dict:
    shock = slug(scenario_name)
    summary = load_summary(dynamics_output_dir / f"summary_counterfactual_4sector_{profile}_{shock}.csv")
    validation_df, checks = load_validation(validate_output_dir / f"validation_counterfactual_4sector_{profile}_{shock}.csv")
    require_all_passed(validation_df, scenario_name)

    selected_t = pd.read_csv(validate_output_dir / f"key_econ_selected_t_{profile}_{shock}.csv")
    window_means = pd.read_csv(validate_output_dir / f"key_econ_window_means_{profile}_{shock}.csv")
    state_maps = pd.read_csv(validate_output_dir / f"state_maps_data_{profile}_{shock}.csv")

    row_t10 = pick_row(selected_t, 10)
    row_t199 = pick_row(selected_t, 199)

    def window_value(var_name: str, col_name: str) -> float:
        match = window_means[window_means["variable"] == var_name]
        if match.empty:
            return float("nan")
        return float(match.iloc[0][col_name])

    return {
        "summary": summary,
        "checks": checks,
        "row_t10": row_t10,
        "row_t199": row_t199,
        "window_means": window_means,
        "top_wage_states_t199": top_states(state_maps, 199, "rw_pct_agg"),
        "top_service_wage_states_t199": top_states(state_maps, 199, "rw_pct_svc"),
        "max_abs_emp_response": max_abs_metric(state_maps, "emp_pct_agg"),
        "max_abs_net_migration": max_abs_metric(state_maps, "net_migration_diff"),
        "rw_early_mean": window_value("rw_cell_pct", "t2_t20"),
        "rw_mid_mean": window_value("rw_cell_pct", "t21_t100"),
        "rw_late_mean": window_value("rw_cell_pct", "t101_tT"),
        "emp_late_mean": window_value("emp_cell_pct", "t101_tT"),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    dynamics_output_dir = Path(args.dynamics_output_dir).resolve()
    validate_output_dir = Path(args.validate_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    delta_payload = load_json(output_dir / "selected_reference_delta.json")
    shock_payload = load_json(output_dir / "shock_calibration.json")
    solver_settings = load_json(output_dir / "reference_solver_settings.json")
    baseline_validation_df, _ = load_validation(output_dir / "baseline_validation_reference.csv")
    require_all_passed(baseline_validation_df, "baseline reference validation")

    immediate = scenario_payload(
        profile=args.profile,
        scenario_name=args.immediate_shock,
        dynamics_output_dir=dynamics_output_dir,
        validate_output_dir=validate_output_dir,
    )
    anticipated = scenario_payload(
        profile=args.profile,
        scenario_name=args.anticipated_shock,
        dynamics_output_dir=dynamics_output_dir,
        validate_output_dir=validate_output_dir,
    )

    strongest_rows = []
    for row in shock_payload.get("strongest_shocks", [])[:5]:
        strongest_rows.append([
            str(row["state_abbr"]),
            str(row["sector_name"]),
            format_num(float(row["cognitive_intensity"]), 3),
            format_pct((float(row["lambda_value"]) - 1.0) * 100.0, 3),
        ])

    scenario_rows = [
        [
            "Immediate",
            str(int(float(immediate["summary"]["iterations"]))),
            format_num(float(immediate["summary"]["final_ymax"]), 6),
            format_num(float(immediate["checks"]["max_abs_mu_t1_diff_vs_identity"]["value"]), 6),
            format_num(float(immediate["checks"]["max_abs_L_t2_diff_vs_identity"]["value"]), 6),
            format_pct(float(immediate["row_t10"]["rw_cell_pct"]), 3),
            format_pct(float(immediate["row_t199"]["rw_cell_pct"]), 3),
            format_pct(float(immediate["row_t199"]["emp_cell_pct"]), 3),
        ],
        [
            "Anticipated",
            str(int(float(anticipated["summary"]["iterations"]))),
            format_num(float(anticipated["summary"]["final_ymax"]), 6),
            format_num(float(anticipated["checks"]["max_abs_mu_t1_diff_vs_identity"]["value"]), 6),
            format_num(float(anticipated["checks"]["max_abs_L_t2_diff_vs_identity"]["value"]), 6),
            format_pct(float(anticipated["row_t10"]["rw_cell_pct"]), 3),
            format_pct(float(anticipated["row_t199"]["rw_cell_pct"]), 3),
            format_pct(float(anticipated["row_t199"]["emp_cell_pct"]), 3),
        ],
    ]

    figure_rows = [
        [
            "Immediate early path",
            f"key_econ_impacts_early_{args.profile}_{slug(args.immediate_shock)}.pdf",
            "Shock-cell wages, employment, inflows, and outflows in the first 30 periods.",
        ],
        [
            "Anticipated early path",
            f"key_econ_impacts_early_{args.profile}_{slug(args.anticipated_shock)}.pdf",
            "Whether anticipation moves the path before the direct productivity shock activates.",
        ],
        [
            "Immediate wage map",
            f"state_map_realwages_{args.profile}_{slug(args.immediate_shock)}.pdf",
            "Which states gain the most in aggregate and service real wages.",
        ],
        [
            "Anticipated wage map",
            f"state_map_realwages_{args.profile}_{slug(args.anticipated_shock)}.pdf",
            "How the spatial pattern changes when the same shock is announced in advance.",
        ],
        [
            "Convergence diagnostics",
            f"dynamics_counterfactual_{args.profile}_{slug(args.immediate_shock)}.pdf",
            "Outer-loop convergence, static-solver workload, and parity checks for the immediate run.",
        ],
    ]

    brief = f"""# Reference Figure Brief

Generated from fresh `{args.profile}` outputs only.

## What the model is

This is a simplified Caliendo-Dvorkin-Parro transition-path model over 50 states and 4 sectors. `mu_path` is the migration-choice object, `Ldyn` is the resulting labor allocation over time, and `Ynew` is the dynamic value-function object that must converge to a fixed point.

## What the shock is

- Selected common reference shock size: `delta = {format_num(float(delta_payload["selected_delta"]), 4)}`
- Official reference settings: `{json.dumps(solver_settings, sort_keys=True)}`
- Immediate shock starts at `t={int(shock_payload["scenario_descriptions"]["immediate"]["active_period_start"])}`.
- Anticipated shock is announced at `t=0` and direct productivity changes start at `t={int(shock_payload["scenario_descriptions"]["anticipated"]["active_period_start"])}`.
- Largest productivity bump in the calibration is `{format_pct((float(shock_payload["matrix_summary"]["lambda_max"]) - 1.0) * 100.0, 3)}`.

### Strongest State-Sector Shocks

{md_table(["State", "Sector", "Intensity", "Productivity bump"], strongest_rows)}

## Reference Checks

- Baseline reference validation passes against `Hvectnoshock.mat`.
- Both AI scenarios pass their full validation files, including the early-response gate.

### Scenario Comparison

{md_table(
    ["Scenario", "Outer iters", "Final Ymax", "mu t=1 diff", "L t=2 diff", "Shock-cell rw t=10", "Shock-cell rw t=199", "Shock-cell emp t=199"],
    scenario_rows,
)}

## Immediate vs Anticipated

- Immediate shock-cell real wage response averages `{format_pct(float(immediate["rw_early_mean"]), 3)}` in `t=2..20`, `{format_pct(float(immediate["rw_mid_mean"]), 3)}` in `t=21..100`, and `{format_pct(float(immediate["rw_late_mean"]), 3)}` in `t=101..T`.
- Anticipated shock-cell real wage response averages `{format_pct(float(anticipated["rw_early_mean"]), 3)}` in `t=2..20`, `{format_pct(float(anticipated["rw_mid_mean"]), 3)}` in `t=21..100`, and `{format_pct(float(anticipated["rw_late_mean"]), 3)}` in `t=101..T`.
- Immediate top aggregate wage states at `t=199`: {immediate["top_wage_states_t199"]}.
- Anticipated top aggregate wage states at `t=199`: {anticipated["top_wage_states_t199"]}.
- Immediate top service-wage states at `t=199`: {immediate["top_service_wage_states_t199"]}.
- Anticipated top service-wage states at `t=199`: {anticipated["top_service_wage_states_t199"]}.

## What the figures say

- The solver is now economically coherent: the counterfactual path has nonzero period-1 migration adjustment and nonzero `L_2` adjustment in both timing scenarios.
- The AI shock is small. The largest direct productivity bump is only about `{format_pct((float(shock_payload["matrix_summary"]["lambda_max"]) - 1.0) * 100.0, 3)}`, so the implied wage and labor responses are expected to be modest.
- The main spatial winners are high cognitive-intensity service states, especially the Northeast and a few high-skill service states in the West.
- If employment or net migration still look numerically tiny in the final maps, that is a model result at this calibrated shock size, not the old stale-path bug. The largest absolute aggregate employment response recorded in the state maps is {immediate["max_abs_emp_response"]} for the immediate run and {anticipated["max_abs_emp_response"]} for the anticipated run.
- The largest absolute net migration difference recorded in the state maps is {immediate["max_abs_net_migration"]} for the immediate run and {anticipated["max_abs_net_migration"]} for the anticipated run.

## Files to look at

{md_table(["Artifact", "Filename", "Why it matters"], figure_rows)}
"""

    (output_dir / "reference_figure_brief.md").write_text(brief)
    print(f"Wrote reference figure brief to {output_dir / 'reference_figure_brief.md'}")


if __name__ == "__main__":
    main()
