from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IN_SEASONAL = REPO_ROOT / "data" / "players" / "derived" / "player_ratings_seasonal.csv"
OUT_DIR = REPO_ROOT / "data" / "players" / "derived"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CAREER = OUT_DIR / "player_ratings_career.csv"

DECAY = 0.85  # Recency decay (2025 counts for 100%, 2024 for 85%, etc.)

def main() -> None:
    if not IN_SEASONAL.exists():
        raise FileNotFoundError(f"Seasonal ratings not found: {IN_SEASONAL}")

    df = pd.read_csv(IN_SEASONAL)

    # Added PercentileRank to required columns
    required = {"playerName", "season", "Minutes", "AttackShrunk", "DefenseShrunk", "TotalShrunk", "PercentileRank"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in seasonal ratings: {sorted(missing)}")

    # Ensure numeric types
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["Minutes"] = pd.to_numeric(df["Minutes"], errors="coerce").fillna(0.0)
    df["PercentileRank"] = pd.to_numeric(df["PercentileRank"], errors="coerce")

    # Drop ineligible seasons
    df = df.dropna(subset=["season", "AttackShrunk", "DefenseShrunk", "TotalShrunk"])

    if df.empty:
        raise ValueError("No eligible rows to compute career ratings.")

    # Calculate Weights
    max_year = int(df["season"].max())
    df["RecencyWeight"] = (DECAY ** (max_year - df["season"])).astype(float)
    df["W"] = df["Minutes"] * df["RecencyWeight"]

    # Pre-calculate weighted scores
    score_cols = ["AttackShrunk", "DefenseShrunk", "TotalShrunk", "PercentileRank"]
    for col in score_cols:
        df[f"weighted_{col}"] = df[col] * df["W"]

    # Aggregate
    # Added 'PeakPercentile' and 'PeakTotal' to capture their best season ever
    career = df.groupby("playerName").agg(
        TotalMinutes=("Minutes", "sum"),
        SeasonsPlayed=("season", "nunique"),
        PeakTotal=("TotalShrunk", "max"),
        PeakPercentile=("PercentileRank", "max"),
        sum_w=("W", "sum"),
        sum_weighted_attack=("weighted_AttackShrunk", "sum"),
        sum_weighted_defense=("weighted_DefenseShrunk", "sum"),
        sum_weighted_total=("weighted_TotalShrunk", "sum"),
        sum_weighted_pct=("weighted_PercentileRank", "sum")
    )

    # Final Divisions (Avoid division by zero)
    career = career[career["sum_w"] > 0].copy()
    
    career["CareerAttack"] = (career["sum_weighted_attack"] / career["sum_w"]).round(3)
    career["CareerDefense"] = (career["sum_weighted_defense"] / career["sum_w"]).round(3)
    career["CareerTotal"] = (career["sum_weighted_total"] / career["sum_w"]).round(3)
    career["CareerPercentile"] = (career["sum_weighted_pct"] / career["sum_w"]).round(1)

    # Cleanup and Sort
    out = career.reset_index()
    
    # Organize columns logically
    keep_cols = [
        "playerName", "TotalMinutes", "SeasonsPlayed", 
        "CareerTotal", "CareerPercentile", 
        "PeakTotal", "PeakPercentile",
        "CareerAttack", "CareerDefense"
    ]
    
    out = out[keep_cols].sort_values("CareerTotal", ascending=False)

    out.to_csv(OUT_CAREER, index=False)
    print(f"--- Career Calculation Complete ---")
    print(f"Saved: {OUT_CAREER}")
    print(f"Top Player: {out.iloc[0]['playerName']} with {out.iloc[0]['CareerTotal']} Career Rating")

if __name__ == "__main__":
    main()