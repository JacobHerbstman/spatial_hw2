#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


def _summary_dict(path: Path) -> dict[str, float | str]:
    df = pd.read_csv(path)
    return dict(zip(df["metric"], df["value"]))


def _read_ynew_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    value_cols = [c for c in df.columns if c != "row_idx"]
    return df[value_cols].to_numpy(dtype=float)


def _write_note(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def identity_check(args: argparse.Namespace) -> int:
    baseline_df = pd.read_csv(args.baseline_validation)
    baseline_pass = bool((baseline_df["status"] == "PASS").all())

    summary = _summary_dict(Path(args.summary))
    converged = int(float(summary["converged"])) == 1
    final_ymax = float(summary["final_ymax"])
    min_ldyn = float(summary["min_Ldyn"])

    ynew = _read_ynew_csv(Path(args.ynew))
    hno = loadmat(args.hno_shock)["Ynew"]
    if hno.shape != ynew.shape:
        raise ValueError(f"Hvectnoshock shape {hno.shape} does not match runner output {ynew.shape}.")

    max_abs_diff = float(np.max(np.abs(ynew - hno)))
    max_rel_diff = float(max_abs_diff / max(1.0, np.max(np.abs(hno))))

    selected = pd.read_csv(args.selected_sector)
    finite_selected = bool(np.isfinite(selected[["employment", "employment_hat_vs_t0", "real_wage"]]).all().all())

    checks = [
        ("baseline_reference_validation", 1.0 if baseline_pass else 0.0, 1.0, "PASS" if baseline_pass else "FAIL"),
        ("identity_converged", 1.0 if converged else 0.0, 1.0, "PASS" if converged else "FAIL"),
        ("identity_final_ymax", final_ymax, 1e-3, "PASS" if final_ymax <= 1e-3 else "FAIL"),
        ("identity_ynew_max_abs_diff_vs_hvectnoshock", max_abs_diff, 1e-3, "PASS" if max_abs_diff <= 1e-3 else "FAIL"),
        ("identity_ynew_max_rel_diff_vs_hvectnoshock", max_rel_diff, 1e-3, "PASS" if max_rel_diff <= 1e-3 else "FAIL"),
        ("identity_ldyn_nonnegative", min_ldyn, -1e-10, "PASS" if min_ldyn >= -1e-10 else "FAIL"),
        ("identity_selected_outputs_finite", 1.0 if finite_selected else 0.0, 1.0, "PASS" if finite_selected else "FAIL"),
    ]
    report = pd.DataFrame(checks, columns=["check", "value", "threshold", "status"])
    report.to_csv(args.output_report, index=False)

    note = f"""
# Identity Check

- Baseline validation reference all PASS: {baseline_pass}
- Minimal identity converged: {converged}
- Final Ymax: {final_ymax:.6g}
- Max abs diff vs Hvectnoshock: {max_abs_diff:.6g}
- Max rel diff vs Hvectnoshock: {max_rel_diff:.6g}
- Minimum Ldyn: {min_ldyn:.6g}
"""
    _write_note(Path(args.output_note), note)

    return 0 if (report["status"] == "PASS").all() else 1


def tiny_smoke(args: argparse.Namespace) -> int:
    summary = _summary_dict(Path(args.summary))
    converged = int(float(summary["converged"])) == 1
    final_ymax = float(summary["final_ymax"])

    identity = pd.read_csv(args.identity_sector)
    shock = pd.read_csv(args.shock_sector)
    merged = shock.merge(
        identity,
        on=["t", "state_idx", "state_abbr", "sector_idx", "sector_name"],
        suffixes=("_shock", "_id"),
        how="inner",
    )
    focus = merged[
        (merged["t"] == args.focus_time)
        & (merged["state_idx"] == args.focus_state)
        & (merged["sector_idx"] == args.focus_sector)
    ].copy()
    if focus.empty:
        raise ValueError("Tiny smoke focus cell was not found in selected-time outputs.")
    focus["real_wage_effect"] = focus["real_wage_shock"] / focus["real_wage_id"] - 1.0
    focus["employment_effect"] = focus["employment_shock"] / focus["employment_id"] - 1.0
    impact = float(focus.iloc[0]["real_wage_effect"])
    positive = impact > 0.0

    checks = [
        ("tiny_converged", 1.0 if converged else 0.0, 1.0, "PASS" if converged else "FAIL"),
        ("tiny_final_ymax", final_ymax, 1e-3, "PASS" if final_ymax <= 1e-3 else "FAIL"),
        ("tiny_focus_real_wage_impact_positive", impact, 0.0, "PASS" if positive else "FAIL"),
    ]
    report = pd.DataFrame(checks, columns=["check", "value", "threshold", "status"])
    report.to_csv(args.output_report, index=False)

    note = f"""
# Tiny Smoke Check

- Converged: {converged}
- Final Ymax: {final_ymax:.6g}
- Focus state/sector/time: state={args.focus_state}, sector={args.focus_sector}, t={args.focus_time}
- Focus real wage effect vs identity: {impact:.6%}
"""
    _write_note(Path(args.output_note), note)

    return 0 if (report["status"] == "PASS").all() else 1


def immediate_summary(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _summary_dict(Path(args.summary))
    converged = int(float(summary["converged"])) == 1
    final_ymax = float(summary["final_ymax"])
    iterations = int(float(summary["iterations"]))
    max_abs_ynew = float(summary["max_abs_Ynew"])

    status_path = output_dir / "immediate_status.md"
    if (not converged) or final_ymax > 1e-3:
        note = f"""
# Immediate Cognitive Run Status

The exact MATLAB-faithful immediate cognitive run did not converge.

- Iterations: {iterations}
- Converged: {converged}
- Final Ymax: {final_ymax:.6g}
- Max abs Ynew: {max_abs_ynew:.6g}

Stop rule triggered: do not interpret this run as an assignment-ready counterfactual.
"""
        _write_note(status_path, note)
        return 1

    identity_sector = pd.read_csv(args.identity_sector)
    shock_sector = pd.read_csv(args.shock_sector)
    sector = shock_sector.merge(
        identity_sector,
        on=["t", "state_idx", "state_abbr", "sector_idx", "sector_name"],
        suffixes=("_shock", "_id"),
        how="inner",
    )
    sector["real_wage_effect"] = sector["real_wage_shock"] / sector["real_wage_id"] - 1.0
    sector["employment_effect"] = sector["employment_shock"] / sector["employment_id"] - 1.0
    sector.to_csv(output_dir / "immediate_sector_effects_selected_times.csv", index=False)

    identity_agg = pd.read_csv(args.identity_aggregate)
    shock_agg = pd.read_csv(args.shock_aggregate)
    agg = shock_agg.merge(
        identity_agg,
        on=["t", "state_idx", "state_abbr"],
        suffixes=("_shock", "_id"),
        how="inner",
    )
    agg["employment_total_effect"] = agg["employment_total_shock"] / agg["employment_total_id"] - 1.0
    agg.to_csv(output_dir / "immediate_state_aggregate_effects_selected_times.csv", index=False)

    final_t = int(sector["t"].max())
    service_final = sector[(sector["t"] == final_t) & (sector["sector_idx"] == 4)].copy()
    service_top = service_final.nlargest(10, "real_wage_effect")
    service_bottom = service_final.nsmallest(10, "real_wage_effect")
    service_rankings = pd.concat(
        [
            service_top.assign(rank_group="top"),
            service_bottom.assign(rank_group="bottom"),
        ],
        ignore_index=True,
    )
    service_rankings.to_csv(output_dir / "immediate_services_real_wage_rankings_t199.csv", index=False)

    agg_final = agg[agg["t"] == final_t].copy()
    agg_top = agg_final.nlargest(10, "employment_total_effect")
    agg_bottom = agg_final.nsmallest(10, "employment_total_effect")
    agg_rankings = pd.concat(
        [
            agg_top.assign(rank_group="top"),
            agg_bottom.assign(rank_group="bottom"),
        ],
        ignore_index=True,
    )
    agg_rankings.to_csv(output_dir / "immediate_aggregate_employment_rankings_t199.csv", index=False)

    focus = sector[(sector["sector_idx"] == 4) & (sector["state_abbr"].isin(["NY", "MA"]))].copy()
    focus = focus.sort_values(["state_abbr", "t"]).reset_index(drop=True)
    focus.to_csv(output_dir / "immediate_focus_cases_services.csv", index=False)

    note = f"""
# Immediate Cognitive Run Status

The exact MATLAB-faithful immediate cognitive run converged.

- Iterations: {iterations}
- Final Ymax: {final_ymax:.6g}
- Selected-time sector effects: `immediate_sector_effects_selected_times.csv`
- Selected-time aggregate effects: `immediate_state_aggregate_effects_selected_times.csv`
- Final-period services rankings: `immediate_services_real_wage_rankings_t199.csv`
- Final-period aggregate employment rankings: `immediate_aggregate_employment_rankings_t199.csv`
- NY/MA services focus cases: `immediate_focus_cases_services.csv`
"""
    _write_note(status_path, note)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summaries and stop rules for the minimal cognitive task.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    identity = subparsers.add_parser("identity-check")
    identity.add_argument("--baseline-validation", required=True)
    identity.add_argument("--summary", required=True)
    identity.add_argument("--ynew", required=True)
    identity.add_argument("--hno-shock", required=True)
    identity.add_argument("--selected-sector", required=True)
    identity.add_argument("--output-report", required=True)
    identity.add_argument("--output-note", required=True)
    identity.set_defaults(func=identity_check)

    tiny = subparsers.add_parser("tiny-smoke")
    tiny.add_argument("--summary", required=True)
    tiny.add_argument("--identity-sector", required=True)
    tiny.add_argument("--shock-sector", required=True)
    tiny.add_argument("--output-report", required=True)
    tiny.add_argument("--output-note", required=True)
    tiny.add_argument("--focus-state", type=int, default=32)
    tiny.add_argument("--focus-sector", type=int, default=4)
    tiny.add_argument("--focus-time", type=int, default=2)
    tiny.set_defaults(func=tiny_smoke)

    immediate = subparsers.add_parser("immediate-summary")
    immediate.add_argument("--summary", required=True)
    immediate.add_argument("--identity-sector", required=True)
    immediate.add_argument("--identity-aggregate", required=True)
    immediate.add_argument("--shock-sector", required=True)
    immediate.add_argument("--shock-aggregate", required=True)
    immediate.add_argument("--output-dir", required=True)
    immediate.set_defaults(func=immediate_summary)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
