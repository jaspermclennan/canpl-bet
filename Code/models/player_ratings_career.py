from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IN_SEASONAL = REPO_ROOT / "data" / "players" / "derived" / "player_ratings_seasonal.csv"
OUT_DIR = REPO_ROOT / "data" / "players" / "derived"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CAREER = OUT_DIR / "player_ratings_career.csv"

DECAY = 0.85  # Standardized space fixed here

def main() -> None:
    if not IN_SEASONAL.exists():
        raise FileNotFoundError(f"Seasonal ratings not found: {IN_SEASONAL}")

    df = pd.read_csv(IN_SEASONAL)

    required = {"playerName", "season", "Minutes", "AttackShrunk", "DefenseShrunk", "TotalShrunk"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in seasonal ratings: {sorted(missing)}")

    # Ensure numeric types
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["Minutes"] = pd.to_numeric(df["Minutes"], errors="coerce").fillna(0.0)

    # Drop rows with NaN scores (ineligible seasons)
    df = df.dropna(subset=["season", "AttackShrunk", "DefenseShrunk", "TotalShrunk"])

    if df.empty:
        raise ValueError("No eligible rows to compute career ratings.")

    # Calculate Weights Vectorized
    max_year = int(df["season"].max())
    df["RecencyWeight"] = (DECAY ** (max_year - df["season"])).astype(float)
    df["W"] = df["Minutes"] * df["RecencyWeight"]

    # Weighted Average Calculation
    # We pre-calculate (Score * Weight) to make aggregation faster
    for col in ["AttackShrunk", "DefenseShrunk", "TotalShrunk"]:
        df[f"weighted_{col}"] = df[col] * df["W"]

    career = df.groupby("playerName").agg(
        TotalMinutes=("Minutes", "sum"),
        SeasonsPlayed=("season", "nunique"),
        sum_w=("W", "sum"),
        sum_weighted_attack=("weighted_AttackShrunk", "sum"),
        sum_weighted_defense=("weighted_DefenseShrunk", "sum"),
        sum_weighted_total=("weighted_TotalShrunk", "sum")
    )

    # Avoid division by zero
    career = career[career["sum_w"] > 0].copy()
    
    career["CareerAttack"] = career["sum_weighted_attack"] / career["sum_w"]
    career["CareerDefense"] = career["sum_weighted_defense"] / career["sum_w"]
    career["CareerTotal"] = career["sum_weighted_total"] / career["sum_w"]

    # Cleanup and Sort
    out = career.reset_index()
    keep_cols = ["playerName", "TotalMinutes", "SeasonsPlayed", "CareerAttack", "CareerDefense", "CareerTotal"]
    out = out[keep_cols].sort_values("CareerTotal", ascending=False)

    out.to_csv(OUT_CAREER, index=False)
    print(f"Saved career ratings: {OUT_CAREER} ({len(out)} players)")

if __name__ == "__main__":
    main()