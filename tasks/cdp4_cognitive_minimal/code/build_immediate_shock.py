#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


SECTOR_NAMES = {
    1: "Manufacturing",
    2: "Construction",
    3: "Wholesale/Retail",
    4: "Services",
}

STATE_ORDER = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build MATLAB-style lambda shocks for the minimal cognitive task.")
    parser.add_argument("--matrix", required=True, help="Path to cognitive_intensity_matrix.csv")
    parser.add_argument("--delta", type=float, default=0.05, help="Maximum TFP boost")
    parser.add_argument("--time-horizon", type=int, default=200, help="Model time horizon")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--tiny-state", type=int, default=32, help="State index for tiny smoke shock")
    parser.add_argument("--tiny-sector", type=int, default=4, help="Sector index for tiny smoke shock")
    parser.add_argument("--tiny-delta", type=float, default=0.01, help="Tiny smoke shock size")
    return parser.parse_args()


def validate_matrix(df: pd.DataFrame) -> pd.DataFrame:
    expected_cols = [
        "state_idx",
        "state_abbr",
        "sector_idx",
        "sector_name",
        "cognitive_intensity",
        "cognitive_intensity_raw",
    ]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Unexpected columns {list(df.columns)}; expected {expected_cols}.")
    if len(df) != 200:
        raise ValueError(f"Expected 200 rows, found {len(df)}.")
    if df[["state_idx", "sector_idx"]].duplicated().any():
        raise ValueError("Duplicate (state_idx, sector_idx) rows found.")
    if set(df["state_idx"]) != set(range(1, 51)):
        raise ValueError("state_idx must cover 1..50.")
    if set(df["sector_idx"]) != set(range(1, 5)):
        raise ValueError("sector_idx must cover 1..4.")
    if df.isna().any().any():
        raise ValueError("Matrix contains missing values.")

    df = df.sort_values(["state_idx", "sector_idx"]).reset_index(drop=True)
    expected_states = {idx + 1: abbr for idx, abbr in enumerate(STATE_ORDER)}
    for state_idx, state_abbr in expected_states.items():
        actual = df.loc[df["state_idx"] == state_idx, "state_abbr"].unique()
        if len(actual) != 1 or actual[0] != state_abbr:
            raise ValueError(f"State mismatch at index {state_idx}: found {actual}, expected {state_abbr}.")
    for sector_idx, sector_name in SECTOR_NAMES.items():
        actual = df.loc[df["sector_idx"] == sector_idx, "sector_name"].unique()
        if len(actual) != 1 or actual[0] != sector_name:
            raise ValueError(f"Sector mismatch at index {sector_idx}: found {actual}, expected {sector_name}.")
    if (df["cognitive_intensity"] < 0).any() or (df["cognitive_intensity"] > 1).any():
        raise ValueError("cognitive_intensity must lie in [0, 1].")
    return df


def build_lambdas(df: pd.DataFrame, delta: float, time_horizon: int) -> tuple[np.ndarray, pd.DataFrame]:
    lambdas = np.ones((4, 87, time_horizon), dtype=float)
    snapshot = df.copy()
    snapshot["lambda_value"] = 1.0 + delta * snapshot["cognitive_intensity"]
    active_periods = max(time_horizon - 2, 0)

    if active_periods > 0:
        for row in snapshot.itertuples(index=False):
            j = int(row.sector_idx) - 1
            n = int(row.state_idx) - 1
            lambdas[j, n, :active_periods] = row.lambda_value

    return lambdas, snapshot


def build_identity(time_horizon: int) -> np.ndarray:
    return np.ones((4, 87, time_horizon), dtype=float)


def build_tiny(time_horizon: int, tiny_state: int, tiny_sector: int, tiny_delta: float) -> np.ndarray:
    lambdas = np.ones((4, 87, time_horizon), dtype=float)
    active_periods = max(time_horizon - 2, 0)
    if active_periods > 0:
        lambdas[tiny_sector - 1, tiny_state - 1, :active_periods] = 1.0 + tiny_delta
    return lambdas


def write_mat(path: Path, array: np.ndarray, description: str) -> None:
    savemat(path, {"lambdas": array, "description": description})


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = Path(args.matrix)
    df = validate_matrix(pd.read_csv(matrix_path))

    lambdas_immediate, snapshot = build_lambdas(df, args.delta, args.time_horizon)
    lambdas_identity = build_identity(args.time_horizon)
    lambdas_tiny = build_tiny(args.time_horizon, args.tiny_state, args.tiny_sector, args.tiny_delta)

    write_mat(
        output_dir / "lambda_cognitive_immediate.mat",
        lambdas_immediate,
        f"Immediate permanent cognitive shock with delta={args.delta}",
    )
    write_mat(
        output_dir / "lambda_identity.mat",
        lambdas_identity,
        "Identity shock for minimal cognitive counterfactual parity checks",
    )
    write_mat(
        output_dir / "lambda_tiny_ny_services.mat",
        lambdas_tiny,
        f"Tiny permanent one-cell shock at state={args.tiny_state}, sector={args.tiny_sector}, delta={args.tiny_delta}",
    )

    snapshot.to_csv(output_dir / "lambda_snapshot_cognitive_immediate.csv", index=False)

    strongest = (
        snapshot.sort_values("lambda_value", ascending=False)
        .head(10)[["state_abbr", "sector_name", "lambda_value"]]
        .to_dict(orient="records")
    )
    metadata = {
        "matrix_path": str(matrix_path.resolve()),
        "delta": args.delta,
        "time_horizon": args.time_horizon,
        "active_periods": max(args.time_horizon - 2, 0),
        "positive_cells": int((snapshot["lambda_value"] > 1.0).sum()),
        "min_lambda": float(snapshot["lambda_value"].min()),
        "max_lambda": float(snapshot["lambda_value"].max()),
        "tiny_shock": {
            "state_idx": args.tiny_state,
            "state_abbr": STATE_ORDER[args.tiny_state - 1],
            "sector_idx": args.tiny_sector,
            "sector_name": SECTOR_NAMES[args.tiny_sector],
            "delta": args.tiny_delta,
        },
        "strongest_shocks": strongest,
    }
    with (output_dir / "shock_builder_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print("=== MINIMAL SHOCK BUILDER ===")
    print(f"matrix = {matrix_path}")
    print(f"delta = {args.delta}")
    print(f"time_horizon = {args.time_horizon}")
    print(f"positive cells = {metadata['positive_cells']}")
    print(f"lambda range = [{metadata['min_lambda']:.4f}, {metadata['max_lambda']:.4f}]")
    print(f"wrote {output_dir / 'lambda_cognitive_immediate.mat'}")
    print(f"wrote {output_dir / 'lambda_identity.mat'}")
    print(f"wrote {output_dir / 'lambda_tiny_ny_services.mat'}")


if __name__ == "__main__":
    main()
