from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import os

# Detects REPO_ROOT 
cwd = Path(os.getcwd())
if cwd.name == "canpl-bet-3":
    REPO_ROOT = cwd
else:
    # Assumes script is in Code/models/
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent

LINEUP_FILE = REPO_ROOT / "data" / "lineups" / "assumed_lineup.csv"
SEASONAL_RATINGS = REPO_ROOT / "data" / "players" / "derived" / "player_ratings_seasonal.csv"
CAREER_RATINGS = REPO_ROOT / "data" / "players" / "derived" / "player_ratings_career.csv"

OUT_DIR = REPO_ROOT / "data" / "matches" / "derived"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "match_team_strength.csv"

COVERAGE_THRESHOLD = 0.80

def main():
    print("--- Starting Match Team Strength Calculation ---")
    
    # Validation & Loading
    if not LINEUP_FILE.exists():
        print(f"Lineup file missing at {LINEUP_FILE}")
        return
    if not SEASONAL_RATINGS.exists():
        print(f"Seasonal ratings missing at {SEASONAL_RATINGS}")
        return

    lineups = pd.read_csv(LINEUP_FILE)
    seasonal = pd.read_csv(SEASONAL_RATINGS)
    
    career = pd.read_csv(CAREER_RATINGS) if CAREER_RATINGS.exists() else pd.DataFrame()

    # Preparation: Clean Seasonal Columns for Join
    seasonal_cols = ["playerId", "season", "AttackShrunk", "DefenseShrunk", "TotalShrunk"]
    seasonal = seasonal[seasonal_cols].copy()

    # Join Lineups with Seasonal Ratings
    # Merge on both ID and Season to get the correct historical rating
    df = lineups.merge(
        seasonal, 
        on=['playerId', 'season'], 
        how='left'
    )

    # Fallback Logic: Join with Career Ratings
    if not career.empty:
        career_cols = ["playerId", "AttackShrunk", "DefenseShrunk", "TotalShrunk"]
        df = df.merge(
            career[career_cols], 
            on='playerId', 
            how='left', 
            suffixes=('', '_career') # This creates 'TotalShrunk_career'
        )

    out_rows = []

    # This loop calculates the minutes-weighted strength
    for match_id, match_df in df.groupby("match_id"):
        teams = match_df["team"].unique()
        if len(teams) != 2:
            continue
            
        # Parse match_id for side detection
        # Format: 2022_2022-04-07_York_United_vs_HFX_Wanderers
        parts = str(match_id).split("_vs_")
        
        for team in teams:
            t_df = match_df[match_df["team"] == team].copy()
            opp = [t for t in teams if t != team][0]
            
            side = "home" if team.replace(" ", "_") in parts[0] else "away"
            
            # Weighted Math: (Expected Mins / 90)
            t_df["w"] = t_df["expected_minutes"] / 90.0
            
            # Fill Gaps: Use Career if Seasonal is NaN, else 0.0 will need to come back to this 
            t_df["final_attack"] = t_df["AttackShrunk"].fillna(t_df.get("AttackShrunk_career", 0.0)).fillna(0.0)
            t_df["final_defense"] = t_df["DefenseShrunk"].fillna(t_df.get("DefenseShrunk_career", 0.0)).fillna(0.0)
            t_df["final_total"] = t_df["TotalShrunk"].fillna(t_df.get("TotalShrunk_career", 0.0)).fillna(0.0)

            # Coverage Metrics
            mins_sum = t_df["expected_minutes"].sum()
            missing_seasonal_mask = t_df["TotalShrunk"].isna()
            rated_mins_sum = t_df.loc[~missing_seasonal_mask, "expected_minutes"].sum()
            
            fallback_mins = 0.0
            if "TotalShrunk_career" in t_df.columns:
                fallback_mins = t_df.loc[missing_seasonal_mask & t_df["TotalShrunk_career"].notna(), "expected_minutes"].sum()
            
            coverage_rate = rated_mins_sum / mins_sum if mins_sum > 0 else 0

            # Compile the Summary Row
            out_rows.append({
                "match_id": match_id,
                "season": t_df["season"].iloc[0],
                "date": t_df["date"].iloc[0],
                "team": team,
                "side": side,
                "opponent": opp,
                "team_attack": round((t_df["final_attack"] * t_df["w"]).sum(), 4),
                "team_defense": round((t_df["final_defense"] * t_df["w"]).sum(), 4),
                "team_total": round((t_df["final_total"] * t_df["w"]).sum(), 4),
                "roster_count": len(t_df),
                "minutes_sum": round(mins_sum, 1),
                "rated_count": (~missing_seasonal_mask).sum(),
                "rated_minutes_sum": round(rated_mins_sum, 1),
                "coverage_rate": round(coverage_rate, 3),
                "coverage_ok": coverage_rate >= COVERAGE_THRESHOLD,
                "fallback_used_minutes": round(fallback_mins, 1),
                "missing_player_ids": "|".join(t_df.loc[missing_seasonal_mask, "playerId"].astype(str)),
                "missing_player_names": "|".join(t_df.loc[missing_seasonal_mask, "playerName"].astype(str))
            })

    # Save Final Dataset
    out_df = pd.DataFrame(out_rows)
    if not out_df.empty:
        # Standardize column order for readability
        cols = [
            "match_id", "season", "date", "team", "side", "opponent",
            "team_attack", "team_defense", "team_total",
            "roster_count", "minutes_sum", "rated_count", "rated_minutes_sum",
            "coverage_rate", "coverage_ok", "fallback_used_minutes",
            "missing_player_ids", "missing_player_names"
        ]
        out_df = out_df[cols].sort_values(["season", "date", "match_id"])
    
    out_df.to_csv(OUT_FILE, index=False)
    
    print("-" * 30)
    print(f"Success! Team strengths saved to: {OUT_FILE}")
    print(f"Avg Coverage Rate: {out_df['coverage_rate'].mean():.1%}")
    print(f"Matches needing data repair: {len(out_df[~out_df['coverage_ok']])}")

if __name__ == "__main__":
    main()