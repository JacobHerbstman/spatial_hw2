#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

IMMEDIATE_SHOCK_NAME = "cognitive_immediate"
ANTICIPATED_SHOCK_NAME = "cognitive_anticipated"
SCENARIOS = (
    (IMMEDIATE_SHOCK_NAME, "lambda_immediate.csv"),
    (ANTICIPATED_SHOCK_NAME, "lambda_anticipated.csv"),
)
REFERENCE_SOLVER_SETTINGS = {
    "WARM_START_STATIC": "0",
    "USE_ANDERSON": "0",
    "HVECT_RELAX": "0.5",
    "USE_THREADS": "0",
    "THREADS_DYNAMIC": "0",
    "THREADS_STATIC": "0",
}
REFERENCE_TOL_DYNAMIC = "1e-4"
REFERENCE_TOL_DYNAMIC_FLOAT = float(REFERENCE_TOL_DYNAMIC)
DEFAULT_FULL_MAX_ITER = 1000
DEFAULT_SCREEN_MAX_ITER = 10
DEFAULT_SCREEN_DIVERGENCE_RATIO = 3.0
DEFAULT_SCREEN_MIN_YMAX = 5e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the largest AI shock delta that converges for both reference scenarios.")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temp-dir", required=True)
    parser.add_argument("--dynamics-dir", required=True)
    parser.add_argument("--dynamics-output-dir", required=True)
    parser.add_argument("--validate-dir", required=True)
    parser.add_argument("--validate-output-dir", required=True)
    parser.add_argument("--baseline-anchor-file", required=True)
    parser.add_argument("--identity-output-file", required=True)
    parser.add_argument("--candidates", default="0.01,0.005,0.0025,0.001,0.0005,0.00025")
    parser.add_argument("--time-horizon", type=int, default=200)
    parser.add_argument("--full-max-iter", type=int, default=DEFAULT_FULL_MAX_ITER)
    parser.add_argument("--screen-max-iter", type=int, default=DEFAULT_SCREEN_MAX_ITER)
    parser.add_argument("--screen-divergence-ratio", type=float, default=DEFAULT_SCREEN_DIVERGENCE_RATIO)
    parser.add_argument("--screen-min-ymax", type=float, default=DEFAULT_SCREEN_MIN_YMAX)
    return parser.parse_args()


def parse_candidates(raw: str) -> list[float]:
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    if not out:
        raise RuntimeError("No delta candidates were provided.")
    return out


def delta_slug(delta: float) -> str:
    token = f"{delta:.6f}".rstrip("0").rstrip(".")
    return token.replace(".", "p")


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, check=True, env=env)


def load_summary(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    summary = dict(zip(df["metric"], df["value"]))
    converged = int(float(summary["converged"])) == 1
    final_ymax = float(summary["final_ymax"])
    max_abs_ynew = float(summary["max_abs_Ynew"])
    return {
        "path": str(path),
        "converged": converged,
        "final_ymax": final_ymax,
        "max_abs_ynew": max_abs_ynew,
    }


def load_validation(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    checks = {}
    for row in df.itertuples(index=False):
        checks[str(row.check)] = {
            "value": float(row.value),
            "threshold": float(row.threshold),
            "status": str(row.status),
        }
    return {
        "path": str(path),
        "all_passed": bool((df["status"] == "PASS").all()),
        "checks": checks,
    }


def load_trace_tail(path: Path, rows: int = 10) -> list[dict[str, float]] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return []
    return df.tail(rows).to_dict(orient="records")


def is_clearly_diverging(
    trace_tail: list[dict[str, float]] | None,
    *,
    divergence_ratio: float,
    min_ymax: float,
) -> bool:
    if not trace_tail or len(trace_tail) < 8:
        return False
    ymax = [float(row["Ymax"]) for row in trace_tail if "Ymax" in row]
    if len(ymax) < 8:
        return False
    recent = ymax[-5:]
    monotone_recent = all(recent[idx] >= recent[idx - 1] for idx in range(1, len(recent)))
    if not monotone_recent:
        return False
    first_ymax = ymax[0]
    last_ymax = ymax[-1]
    min_seen = min(ymax)
    return last_ymax > min_ymax and last_ymax > first_ymax * divergence_ratio and last_ymax > min_seen * 1.25


def failure_record(path: Path, exc: subprocess.CalledProcessError | None) -> dict[str, object]:
    return {
        "path": str(path),
        "converged": False,
        "run_failed": True,
        "returncode": exc.returncode if exc is not None else None,
        "summary_exists": path.exists(),
    }


def clear_scenario_outputs(dynamics_output_dir: Path, scenario_name: str) -> None:
    candidates = [
        dynamics_output_dir / f"counterfactual_4sector_path_reference_{scenario_name}.jld2",
        dynamics_output_dir / f"summary_counterfactual_4sector_reference_{scenario_name}.csv",
        dynamics_output_dir / f"outer_trace_counterfactual_4sector_reference_{scenario_name}.csv",
        dynamics_output_dir / f"benchmark_counterfactual_4sector_reference_{scenario_name}.csv",
    ]
    for path in candidates:
        if path.exists() or path.is_symlink():
            path.unlink()


def clear_validation_outputs(validate_output_dir: Path, scenario_name: str) -> None:
    candidates = [
        validate_output_dir / f"validation_counterfactual_4sector_reference_{scenario_name}.csv",
        validate_output_dir / f"parity_by_time_counterfactual_4sector_reference_{scenario_name}.csv",
        validate_output_dir / f"benchmark_validate_counterfactual_4sector_reference_{scenario_name}.csv",
    ]
    for path in candidates:
        if path.exists() or path.is_symlink():
            path.unlink()


def run_validation_for_scenario(
    *,
    scenario_name: str,
    lambda_path: Path,
    validate_dir: Path,
    validate_output_dir: Path,
    counterfactual_output_file: Path,
    identity_output_file: Path,
    baseline_anchor_file: Path,
    time_horizon: int,
    env: dict[str, str],
) -> dict[str, object]:
    validation_path = validate_output_dir / f"validation_counterfactual_4sector_reference_{scenario_name}.csv"
    clear_validation_outputs(validate_output_dir, scenario_name)
    cmd = [
        "make",
        "-C",
        str(validate_dir),
        "generic_validate",
        "PROFILE=reference",
        "FAIL_ON_CHECKS=1",
        "REQUIRE_T1_RESPONSE=1",
        "SHOCK_INPUT_MODE=csv",
        f"SHOCK_NAME={scenario_name}",
        f"LAMBDA_CSV={lambda_path}",
        f"TIME_HORIZON={time_horizon}",
        f"COUNTERFACTUAL_OUTPUT_FILE={counterfactual_output_file}",
        f"IDENTITY_OUTPUT_FILE={identity_output_file}",
        f"BASELINE_ANCHOR_FILE={baseline_anchor_file}",
        f"TOL_DYNAMIC={REFERENCE_TOL_DYNAMIC}",
        f"CONFIG_TAG={scenario_name}_delta_validation",
    ]
    run_cmd(cmd, env=env)
    return load_validation(validation_path)


def run_candidate(delta: float, script_dir: Path, matrix_path: Path, temp_dir: Path,
                  dynamics_dir: Path, dynamics_output_dir: Path,
                  validate_dir: Path, validate_output_dir: Path,
                  identity_output_file: Path,
                  baseline_anchor_file: Path, time_horizon: int,
                  full_max_iter: int, screen_max_iter: int,
                  screen_divergence_ratio: float, screen_min_ymax: float) -> tuple[dict[str, object], Path]:
    candidate_dir = temp_dir / f"delta_{delta_slug(delta)}"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    generate_cmd = [
        sys.executable,
        str(script_dir / "generate_lambda_csvs.py"),
        "--matrix",
        str(matrix_path),
        "--output-dir",
        str(candidate_dir),
        "--delta",
        str(delta),
        "--time-horizon",
        str(time_horizon),
    ]
    run_cmd(generate_cmd, env=os.environ.copy())

    candidate_result: dict[str, object] = {
        "delta": delta,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_dir": str(candidate_dir),
        "scenarios": {},
    }

    env = os.environ.copy()
    env.update(REFERENCE_SOLVER_SETTINGS)

    for scenario_name, lambda_filename in SCENARIOS:
        lambda_path = candidate_dir / lambda_filename
        config_tag = f"{scenario_name}_delta_{delta_slug(delta)}"
        cmd = [
            "make",
            "-C",
            str(dynamics_dir),
            "csv_reference",
            f"LAMBDA_CSV={lambda_path}",
            f"SHOCK_NAME={scenario_name}",
            f"TIME_HORIZON={time_horizon}",
            f"BASELINE_ANCHOR_FILE={baseline_anchor_file}",
            f"CONFIG_TAG={config_tag}",
        ]
        summary_path = dynamics_output_dir / f"summary_counterfactual_4sector_reference_{scenario_name}.csv"
        trace_path = dynamics_output_dir / f"outer_trace_counterfactual_4sector_reference_{scenario_name}.csv"
        counterfactual_output_file = dynamics_output_dir / f"counterfactual_4sector_path_reference_{scenario_name}.jld2"

        screen_cmd = cmd + [f"MAX_ITER_DYNAMIC={screen_max_iter}"]
        clear_scenario_outputs(dynamics_output_dir, scenario_name)
        try:
            run_cmd(screen_cmd, env=env)
        except subprocess.CalledProcessError as exc:
            scenario_result = failure_record(summary_path, exc)
            scenario_result["trace_tail"] = load_trace_tail(trace_path)
            candidate_result["scenarios"][scenario_name] = scenario_result
            candidate_result["all_converged"] = False
            candidate_result["failed_scenario"] = scenario_name
            candidate_result["failure_reason"] = "screen_solver_command_failed"
            return candidate_result, candidate_dir

        screen_result = load_summary(summary_path)
        screen_trace_tail = load_trace_tail(trace_path, rows=screen_max_iter)
        if is_clearly_diverging(
            screen_trace_tail,
            divergence_ratio=screen_divergence_ratio,
            min_ymax=screen_min_ymax,
        ):
            screen_result["trace_tail"] = screen_trace_tail
            screen_result["screen_rejected"] = True
            screen_result["screen_reason"] = "diverging_trace"
            candidate_result["scenarios"][scenario_name] = screen_result
            candidate_result["all_converged"] = False
            candidate_result["failed_scenario"] = scenario_name
            candidate_result["failure_reason"] = "screen_rejected_divergence"
            return candidate_result, candidate_dir

        if screen_result["converged"] and screen_result["final_ymax"] <= REFERENCE_TOL_DYNAMIC_FLOAT:
            scenario_result = screen_result
            scenario_result["trace_tail"] = screen_trace_tail
            scenario_result["screened_only"] = True
        else:
            full_cmd = cmd + [f"MAX_ITER_DYNAMIC={full_max_iter}"]
            clear_scenario_outputs(dynamics_output_dir, scenario_name)
            try:
                run_cmd(full_cmd, env=env)
            except subprocess.CalledProcessError as exc:
                scenario_result = failure_record(summary_path, exc)
                scenario_result["trace_tail"] = load_trace_tail(trace_path)
                scenario_result["screen_summary"] = screen_result
                scenario_result["screen_trace_tail"] = screen_trace_tail
                candidate_result["scenarios"][scenario_name] = scenario_result
                candidate_result["all_converged"] = False
                candidate_result["failed_scenario"] = scenario_name
                candidate_result["failure_reason"] = "full_solver_command_failed"
                return candidate_result, candidate_dir

            scenario_result = load_summary(summary_path)
            scenario_result["trace_tail"] = load_trace_tail(trace_path)
            scenario_result["screen_summary"] = screen_result
            scenario_result["screen_trace_tail"] = screen_trace_tail

        try:
            validation_result = run_validation_for_scenario(
                scenario_name=scenario_name,
                lambda_path=lambda_path,
                validate_dir=validate_dir,
                validate_output_dir=validate_output_dir,
                counterfactual_output_file=counterfactual_output_file,
                identity_output_file=identity_output_file,
                baseline_anchor_file=baseline_anchor_file,
                time_horizon=time_horizon,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            scenario_result["validation"] = load_validation(
                validate_output_dir / f"validation_counterfactual_4sector_reference_{scenario_name}.csv"
            ) if (validate_output_dir / f"validation_counterfactual_4sector_reference_{scenario_name}.csv").exists() else None
            scenario_result["validation_failed"] = True
            scenario_result["validation_returncode"] = exc.returncode
            candidate_result["scenarios"][scenario_name] = scenario_result
            candidate_result["all_converged"] = False
            candidate_result["failed_scenario"] = scenario_name
            candidate_result["failure_reason"] = "validation_failed"
            return candidate_result, candidate_dir

        scenario_result["validation"] = validation_result
        candidate_result["scenarios"][scenario_name] = scenario_result

    candidate_result["all_converged"] = all(
        scenario["converged"] and
        scenario["final_ymax"] <= REFERENCE_TOL_DYNAMIC_FLOAT and
        scenario.get("validation", {}).get("all_passed", False)
        for scenario in candidate_result["scenarios"].values()
    )
    return candidate_result, candidate_dir


def copy_selected_outputs(candidate_dir: Path, output_dir: Path) -> None:
    for filename in ("lambda_immediate.csv", "lambda_anticipated.csv", "shock_calibration.json", "data_notes.md"):
        shutil.copy2(candidate_dir / filename, output_dir / filename)


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    matrix_path = Path(args.matrix).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    temp_dir = Path(args.temp_dir).expanduser().resolve()
    dynamics_dir = Path(args.dynamics_dir).expanduser().resolve()
    dynamics_output_dir = Path(args.dynamics_output_dir).expanduser().resolve()
    validate_dir = Path(args.validate_dir).expanduser().resolve()
    validate_output_dir = Path(args.validate_output_dir).expanduser().resolve()
    baseline_anchor_file = Path(args.baseline_anchor_file).expanduser().resolve()
    identity_output_file = Path(args.identity_output_file).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    candidates = parse_candidates(args.candidates)
    attempts: list[dict[str, object]] = []
    selected_result: dict[str, object] | None = None
    selected_candidate_dir: Path | None = None

    for delta in candidates:
        result, candidate_dir = run_candidate(
            delta=delta,
            script_dir=script_dir,
            matrix_path=matrix_path,
            temp_dir=temp_dir,
            dynamics_dir=dynamics_dir,
            dynamics_output_dir=dynamics_output_dir,
            validate_dir=validate_dir,
            validate_output_dir=validate_output_dir,
            identity_output_file=identity_output_file,
            baseline_anchor_file=baseline_anchor_file,
            time_horizon=args.time_horizon,
            full_max_iter=args.full_max_iter,
            screen_max_iter=args.screen_max_iter,
            screen_divergence_ratio=args.screen_divergence_ratio,
            screen_min_ymax=args.screen_min_ymax,
        )
        attempts.append(result)
        if result["all_converged"]:
            selected_result = result
            selected_candidate_dir = candidate_dir
            break

    if selected_result is None or selected_candidate_dir is None:
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_delta": None,
            "time_horizon": args.time_horizon,
            "matrix_path": str(matrix_path),
            "candidate_results": attempts,
            "reference_solver_settings": REFERENCE_SOLVER_SETTINGS,
        }
        (output_dir / "selected_reference_delta.json").write_text(json.dumps(payload, indent=2))
        (output_dir / "reference_solver_settings.json").write_text(json.dumps(REFERENCE_SOLVER_SETTINGS, indent=2))
        raise RuntimeError("No reference delta candidate converged for both immediate and anticipated AI scenarios.")

    copy_selected_outputs(selected_candidate_dir, output_dir)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_delta": selected_result["delta"],
        "selected_candidate_dir": str(selected_candidate_dir),
        "time_horizon": args.time_horizon,
        "matrix_path": str(matrix_path),
        "candidate_results": attempts,
        "reference_solver_settings": REFERENCE_SOLVER_SETTINGS,
    }
    (output_dir / "selected_reference_delta.json").write_text(json.dumps(payload, indent=2))
    (output_dir / "reference_solver_settings.json").write_text(json.dumps(REFERENCE_SOLVER_SETTINGS, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
