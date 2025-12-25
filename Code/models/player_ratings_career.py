from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IN_SEASONAL = REPO_ROOT / "data" / "players" / "derived" / "player_ratings_seasonal.csv"
OUT_DIR = REPO_ROOT / "data" / "players" / "derived"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CAREER = OUT_DIR / "player_ratings_career.csv"

# Recency decay: 2025 = 1.0, 2024 = 0.85, 2023 = 0.72, etc.
DECAY = 0.85  

def main() -> None:
    if not IN_SEASONAL.exists():
        raise FileNotFoundError(f"Seasonal ratings not found: {IN_SEASONAL}")

    df = pd.read_csv(IN_SEASONAL)

    # CRITICAL: Added playerId to required columns
    required = {"playerId", "playerName", "season", "Minutes", "AttackShrunk", "DefenseShrunk", "TotalShrunk", "PercentileRank"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in seasonal ratings: {sorted(missing)}")

    # Ensure numeric types for math
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["Minutes"] = pd.to_numeric(df["Minutes"], errors="coerce").fillna(0.0)
    df["PercentileRank"] = pd.to_numeric(df["PercentileRank"], errors="coerce")

    # Drop rows missing core ratings
    df = df.dropna(subset=["season", "AttackShrunk", "DefenseShrunk", "TotalShrunk"])

    if df.empty:
        raise ValueError("No eligible rows to compute career ratings.")

    # We weight by Minutes * (Decay ^ Years_Ago)
    max_year = int(df["season"].max())
    df["RecencyWeight"] = (DECAY ** (max_year - df["season"])).astype(float)
    df["W"] = df["Minutes"] * df["RecencyWeight"]

    score_cols = ["AttackShrunk", "DefenseShrunk", "TotalShrunk", "PercentileRank"]
    for col in score_cols:
        df[f"weighted_{col}"] = df[col] * df["W"]

    career = df.groupby(["playerId", "playerName"]).agg(
        TotalMinutes=("Minutes", "sum"),
        SeasonsPlayed=("season", "nunique"),
        PeakTotal=("TotalShrunk", "max"),
        PeakPercentile=("PercentileRank", "max"),
        sum_w=("W", "sum"),
        sum_weighted_attack=("weighted_AttackShrunk", "sum"),
        sum_weighted_defense=("weighted_DefenseShrunk", "sum"),
        sum_weighted_total=("weighted_TotalShrunk", "sum"),
        sum_weighted_pct=("weighted_PercentileRank", "sum")
    ).reset_index()

    # Avoid division by zero
    career = career[career["sum_w"] > 0].copy()
    
    # Calculate Final Career Averages
    career["AttackShrunk"] = (career["sum_weighted_attack"] / career["sum_w"]).round(3)
    career["DefenseShrunk"] = (career["sum_weighted_defense"] / career["sum_w"]).round(3)
    career["TotalShrunk"] = (career["sum_weighted_total"] / career["sum_w"]).round(3)
    career["PercentileRank"] = (career["sum_weighted_pct"] / career["sum_w"]).round(1)

    # Organize columns logically for CSV
    keep_cols = [
        "playerId", "playerName", "TotalMinutes", "SeasonsPlayed", 
        "TotalShrunk", "PercentileRank", 
        "PeakTotal", "PeakPercentile",
        "AttackShrunk", "DefenseShrunk"
    ]
    
    out = career[keep_cols].sort_values("TotalShrunk", ascending=False)

    out.to_csv(OUT_CAREER, index=False)
    
    print(f"\n" + "="*40)
    print(f"   CAREER CALCULATION COMPLETE")
    print(f"="*40)
    print(f"Saved: {OUT_CAREER}")
    print(f"Top Rated Career: {out.iloc[0]['playerName']} ({out.iloc[0]['TotalShrunk']})")
    print(f"Total Players Processed: {len(out)}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()