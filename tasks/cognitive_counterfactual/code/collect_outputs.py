#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def collect_files(
    scenario_name: str,
    profile: str,
    dynamics_output_dir: Path,
    validate_output_dir: Path,
    output_dir: Path,
    lambda_file: Path | None = None,
    delta_selection_file: Path | None = None,
    solver_settings_file: Path | None = None,
    baseline_validation_file: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    expected = [
        dynamics_output_dir / f"counterfactual_4sector_path_{profile}_{scenario_name}.jld2",
        dynamics_output_dir / f"summary_counterfactual_4sector_{profile}_{scenario_name}.csv",
        dynamics_output_dir / f"outer_trace_counterfactual_4sector_{profile}_{scenario_name}.csv",
        dynamics_output_dir / f"benchmark_counterfactual_4sector_{profile}_{scenario_name}.csv",
        validate_output_dir / f"validation_counterfactual_4sector_{profile}_{scenario_name}.csv",
        validate_output_dir / f"parity_by_time_counterfactual_4sector_{profile}_{scenario_name}.csv",
        validate_output_dir / f"benchmark_validate_counterfactual_4sector_{profile}_{scenario_name}.csv",
        validate_output_dir / f"dynamics_counterfactual_{profile}_{scenario_name}.pdf",
        validate_output_dir / f"dynamics_outer_counterfactual_{profile}_{scenario_name}.pdf",
        validate_output_dir / f"dynamics_parity_counterfactual_{profile}_{scenario_name}.pdf",
        validate_output_dir / f"table_counterfactual_summary_{profile}_{scenario_name}.tex",
        validate_output_dir / f"table_counterfactual_validation_{profile}_{scenario_name}.tex",
        validate_output_dir / f"table_counterfactual_parity_selected_t_{profile}_{scenario_name}.tex",
        validate_output_dir / f"key_econ_timeseries_{profile}_{scenario_name}.csv",
        validate_output_dir / f"key_econ_selected_t_{profile}_{scenario_name}.csv",
        validate_output_dir / f"key_econ_window_means_{profile}_{scenario_name}.csv",
        validate_output_dir / f"key_econ_impacts_{profile}_{scenario_name}.pdf",
        validate_output_dir / f"key_econ_impacts_early_{profile}_{scenario_name}.pdf",
        validate_output_dir / f"table_key_econ_selected_t_{profile}_{scenario_name}.tex",
        validate_output_dir / f"table_key_econ_window_means_{profile}_{scenario_name}.tex",
        validate_output_dir / f"state_maps_data_{profile}_{scenario_name}.csv",
        validate_output_dir / f"state_map_realwages_{profile}_{scenario_name}.pdf",
        validate_output_dir / f"state_map_employment_{profile}_{scenario_name}.pdf",
        validate_output_dir / f"state_map_sectoral_shift_{profile}_{scenario_name}.pdf",
    ]

    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing expected scenario outputs: {missing}")

    linked: list[str] = []
    for src in expected:
        dst = output_dir / src.name
        ensure_symlink(src.resolve(), dst)
        linked.append(str(dst.resolve(strict=False)))

    local_files = [
        output_dir / "lambda_immediate.csv",
        output_dir / "lambda_anticipated.csv",
        output_dir / "shock_calibration.json",
        output_dir / "data_notes.md",
        output_dir / "selected_reference_delta.json",
        output_dir / "reference_solver_settings.json",
        output_dir / "baseline_validation_reference.csv",
    ]
    selected_delta = None
    delta_selection_payload = None
    if delta_selection_file is not None and delta_selection_file.exists():
        delta_selection_payload = json.loads(delta_selection_file.read_text())
        selected_delta = delta_selection_payload.get("selected_delta")

    solver_settings_payload = None
    if solver_settings_file is not None and solver_settings_file.exists():
        solver_settings_payload = json.loads(solver_settings_file.read_text())

    manifest = {
        "scenario_name": scenario_name,
        "profile": profile,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dynamics_output_dir": str(dynamics_output_dir.resolve()),
        "validate_output_dir": str(validate_output_dir.resolve()),
        "linked_outputs": linked,
        "local_outputs": [str(path.resolve()) for path in local_files if path.exists()],
        "selected_delta": selected_delta,
        "delta_selection_file": str(delta_selection_file.resolve()) if delta_selection_file and delta_selection_file.exists() else None,
        "solver_settings_file": str(solver_settings_file.resolve()) if solver_settings_file and solver_settings_file.exists() else None,
        "baseline_validation_file": str(baseline_validation_file.resolve()) if baseline_validation_file and baseline_validation_file.exists() else None,
        "delta_selection": delta_selection_payload,
        "solver_settings": solver_settings_payload,
    }
    if lambda_file is not None and lambda_file.exists():
        manifest["lambda_file"] = str(lambda_file.resolve())
        manifest["lambda_file_sha256"] = sha256_file(lambda_file)

    (output_dir / f"scenario_outputs_{profile}_{scenario_name}.json").write_text(json.dumps(manifest, indent=2))
    (output_dir / f"scenario_manifest_{profile}_{scenario_name}.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect scenario outputs into the public cognitive counterfactual task output directory.")
    parser.add_argument("--scenario-name", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dynamics-output-dir", required=True)
    parser.add_argument("--validate-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lambda-file", default="")
    parser.add_argument("--delta-selection-file", default="")
    parser.add_argument("--solver-settings-file", default="")
    parser.add_argument("--baseline-validation-file", default="")
    args = parser.parse_args()

    collect_files(
        scenario_name=args.scenario_name,
        profile=args.profile,
        dynamics_output_dir=Path(args.dynamics_output_dir).resolve(),
        validate_output_dir=Path(args.validate_output_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        lambda_file=Path(args.lambda_file).resolve() if args.lambda_file else None,
        delta_selection_file=Path(args.delta_selection_file).resolve() if args.delta_selection_file else None,
        solver_settings_file=Path(args.solver_settings_file).resolve() if args.solver_settings_file else None,
        baseline_validation_file=Path(args.baseline_validation_file).resolve() if args.baseline_validation_file else None,
    )


if __name__ == "__main__":
    main()
