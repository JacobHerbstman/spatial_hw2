#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def collect_files(
    scenario_name: str,
    profile: str,
    shock_output_dir: Path,
    dynamics_output_dir: Path,
    validate_output_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    expected = [
        shock_output_dir / f"lambda_hat_{scenario_name}.csv",
        shock_output_dir / f"lambda_snapshot_{scenario_name}.csv",
        shock_output_dir / f"shock_manifest_{scenario_name}.json",
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
        validate_output_dir / f"broad_shock_timeseries_{profile}_{scenario_name}.csv",
        validate_output_dir / f"broad_shock_sector_timeseries_{profile}_{scenario_name}.csv",
        validate_output_dir / f"broad_shock_state_sector_selected_t_{profile}_{scenario_name}.csv",
        validate_output_dir / f"broad_shock_state_rankings_{profile}_{scenario_name}.csv",
        validate_output_dir / f"broad_shock_summary_{profile}_{scenario_name}.pdf",
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

    manifest = {
        "scenario_name": scenario_name,
        "profile": profile,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "shock_output_dir": str(shock_output_dir.resolve()),
        "dynamics_output_dir": str(dynamics_output_dir.resolve()),
        "validate_output_dir": str(validate_output_dir.resolve()),
        "linked_outputs": linked,
    }
    (output_dir / f"scenario_outputs_{profile}_{scenario_name}.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect scenario-specific outputs into the cognitive counterfactual task directory.")
    parser.add_argument("--scenario-name", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--shock-output-dir", required=True)
    parser.add_argument("--dynamics-output-dir", required=True)
    parser.add_argument("--validate-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    collect_files(
        scenario_name=args.scenario_name,
        profile=args.profile,
        shock_output_dir=Path(args.shock_output_dir).resolve(),
        dynamics_output_dir=Path(args.dynamics_output_dir).resolve(),
        validate_output_dir=Path(args.validate_output_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
    )


if __name__ == "__main__":
    main()
