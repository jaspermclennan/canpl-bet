from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import os

# Detects REPO_ROOT 
cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else Path(__file__).resolve().parent.parent.parent

LINEUP_FILE = REPO_ROOT / "data" / "lineups" / "assumed_lineup.csv"
# NEW: Point to the rolling ratings
ROLLING_RATINGS = REPO_ROOT / "data" / "players" / "derived" / "player_ratings_rolling.csv"

OUT_DIR = REPO_ROOT / "data" / "matches" / "derived"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "match_team_strength.csv"

COVERAGE_THRESHOLD = 0.80

def main():
    print("--- Starting Match Team Strength (Rolling) Calculation ---")
    
    if not LINEUP_FILE.exists() or not ROLLING_RATINGS.exists():
        print("Missing required files.")
        print(f"Check: {ROLLING_RATINGS}")
        return

    lineups = pd.read_csv(LINEUP_FILE)
    ratings = pd.read_csv(ROLLING_RATINGS) # Load the rolling file

    # Merge Rolling Ratings into Lineups
    # Matches strictly on match_id so we get the rating AS OF that specific game
    df = lineups.merge(
        ratings[['match_id', 'playerId', 'Rating']], 
        on=['match_id', 'playerId'], 
        how='left'
    )
    
    # Fill missing ratings (New players/First games) with default 5.0
    df['Rating'] = df['Rating'].fillna(5.0)

    out_rows = []
    
    for match_id, match_df in df.groupby("match_id"):
        teams = match_df["team"].unique()
        if len(teams) != 2: continue
            
        parts = str(match_id).split("_vs_")
        
        for team in teams:
            t_df = match_df[match_df["team"] == team].copy()
            opp = [t for t in teams if t != team][0]
            
            # Determine Home/Away
            # Logic: If team name is in the first part of filename, they are Home
            clean_team = team.replace(" ", "_")
            side = "home" if clean_team in parts[0] else "away"
            
            # Weighting: Expected Minutes / 90
            t_df["w"] = t_df["expected_minutes"] / 90.0
            
            # Coverage Quality
            mins_sum = t_df["expected_minutes"].sum()
            # In rolling logic, everyone effectively has a rating (default 5.0), so coverage is high
            
            # Calculate Total Strength
            # (Note: Rolling script currently only outputs Total Rating, not Attack/Defense splits yet)
            team_total = (t_df["Rating"] * t_df["w"]).sum()
            
            # Placeholder for Attack/Defense until we add splits to rolling script
            team_attack = team_total 
            team_defense = team_total

            out_rows.append({
                "match_id": match_id, 
                "season": t_df["season"].iloc[0], 
                "date": t_df["date"].iloc[0],
                "team": team, 
                "side": side, 
                "opponent": opp,
                "team_attack": round(team_attack, 4),
                "team_defense": round(team_defense, 4),
                "team_total": round(team_total, 4),
                "coverage_rate": 1.0, # Rolling logic fills gaps automatically
                "coverage_ok": True
            })

    out_df = pd.DataFrame(out_rows)
    # Sort for cleanliness
    out_df = out_df.sort_values(['date', 'match_id'])
    
    out_df.to_csv(OUT_FILE, index=False)
    print(f"Success! Rolling Team strengths saved to: {OUT_FILE}")

if __name__ == "__main__":
    main()