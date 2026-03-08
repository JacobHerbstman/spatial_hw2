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


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_upstream_metadata(matrix_path: Path) -> dict | None:
    candidates = [
        matrix_path.with_name("cognitive_intensity_metadata.json"),
        matrix_path.with_name(f"{matrix_path.stem}_metadata.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text())
    return None


def validate_matrix(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise RuntimeError(f"Matrix is missing required columns: {missing}")

    if len(df) != 200:
        raise RuntimeError(f"Expected 200 rows in matrix, found {len(df)}")

    keyed = df.copy()
    if keyed[["state_idx", "sector_idx"]].duplicated().any():
        raise RuntimeError("Matrix has duplicate (state_idx, sector_idx) rows")
    if keyed[REQUIRED_COLUMNS].isna().any().any():
        raise RuntimeError("Matrix has missing values in required columns")

    state_idx = keyed["state_idx"].astype(int)
    sector_idx = keyed["sector_idx"].astype(int)
    if sorted(state_idx.unique().tolist()) != list(range(1, 51)):
        raise RuntimeError("Matrix state_idx values must be exactly 1..50")
    if sorted(sector_idx.unique().tolist()) != [1, 2, 3, 4]:
        raise RuntimeError("Matrix sector_idx values must be exactly 1..4")

    keyed["cognitive_intensity"] = pd.to_numeric(keyed["cognitive_intensity"], errors="raise")
    keyed["cognitive_intensity_raw"] = pd.to_numeric(keyed["cognitive_intensity_raw"], errors="raise")
    if (keyed["cognitive_intensity"] < 0).any() or (keyed["cognitive_intensity"] > 1).any():
        raise RuntimeError("Matrix cognitive_intensity values must lie in [0, 1]")

    keyed["state_idx"] = state_idx
    keyed["sector_idx"] = sector_idx
    return keyed.sort_values(["state_idx", "sector_idx"]).reset_index(drop=True)


def build_outputs(
    matrix_path: Path,
    output_dir: Path,
    scenario_name: str,
    delta: float,
    time_horizon: int,
    schedule: str,
) -> None:
    if schedule != "permanent":
        raise RuntimeError(f"Unsupported schedule={schedule}. Only `permanent` is implemented.")
    if time_horizon < 3:
        raise RuntimeError(f"time_horizon must be at least 3, found {time_horizon}")

    matrix = validate_matrix(pd.read_csv(matrix_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_checksum = sha256_file(matrix_path)
    upstream_metadata = load_upstream_metadata(matrix_path)

    snapshot = matrix.copy()
    snapshot["lambda_hat"] = 1.0 + delta * snapshot["cognitive_intensity"]

    shock_periods = time_horizon - 2
    lambda_rows: list[dict] = []
    for row in snapshot.itertuples(index=False):
        for t in range(1, shock_periods + 1):
            lambda_rows.append(
                {
                    "t": t,
                    "j": int(row.sector_idx),
                    "n": int(row.state_idx),
                    "value": float(row.lambda_hat),
                }
            )
    lambda_df = pd.DataFrame(lambda_rows)

    lambda_path = output_dir / f"lambda_hat_{scenario_name}.csv"
    snapshot_path = output_dir / f"lambda_snapshot_{scenario_name}.csv"
    manifest_path = output_dir / f"shock_manifest_{scenario_name}.json"

    lambda_df.to_csv(lambda_path, index=False)
    snapshot[
        [
            "state_idx",
            "state_abbr",
            "sector_idx",
            "sector_name",
            "cognitive_intensity",
            "cognitive_intensity_raw",
            "lambda_hat",
        ]
    ].to_csv(snapshot_path, index=False)

    manifest = {
        "scenario_name": scenario_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_path": str(matrix_path.resolve()),
        "matrix_checksum_sha256": matrix_checksum,
        "delta": delta,
        "time_horizon": time_horizon,
        "shock_periods": shock_periods,
        "schedule": schedule,
        "lambda_csv_path": str(lambda_path.resolve()),
        "lambda_snapshot_path": str(snapshot_path.resolve()),
        "kappa_mode": "identity",
        "matrix_summary": {
            "row_count": int(len(snapshot)),
            "state_count": int(snapshot["state_idx"].nunique()),
            "sector_count": int(snapshot["sector_idx"].nunique()),
            "cognitive_intensity_min": float(snapshot["cognitive_intensity"].min()),
            "cognitive_intensity_max": float(snapshot["cognitive_intensity"].max()),
            "lambda_hat_min": float(snapshot["lambda_hat"].min()),
            "lambda_hat_max": float(snapshot["lambda_hat"].max()),
        },
        "upstream_metadata": upstream_metadata,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    log(f"Wrote {lambda_path}")
    log(f"Wrote {snapshot_path}")
    log(f"Wrote {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build solver-ready lambda_hat CSVs from a 50x4 cognitive intensity matrix.")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenario-name", required=True)
    parser.add_argument("--delta", required=True, type=float)
    parser.add_argument("--time-horizon", required=True, type=int)
    parser.add_argument("--schedule", default="permanent")
    args = parser.parse_args()

    build_outputs(
        matrix_path=Path(args.matrix).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        scenario_name=args.scenario_name,
        delta=args.delta,
        time_horizon=args.time_horizon,
        schedule=args.schedule,
    )


if __name__ == "__main__":
    main()
