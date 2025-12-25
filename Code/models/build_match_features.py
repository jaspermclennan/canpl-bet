from __future__ import annotations
from pathlib import Path
import pandas as pd 
import os 

cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else Path(__file__).resolve().parent.parent.parent

STRENGTH_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_team_strength.csv"
OUT_DIR = REPO_ROOT / "data" / "matches" / "derived"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "match_features.csv"

def main():
    if not STRENGTH_FILE.exists():
        print(f"cannot find {STRENGTH_FILE}")
        return 
    
    # load the team strength data 
    df = pd.read_csv(STRENGTH_FILE)
    
    # split into Home and Away dataframes
    home_df = df[df["side"] == "home"].copy()
    away_df = df[df["side"] == "away"].copy()
    
    # rename columns to avoid collisions after merging
    # match_id, season, and date as join keys
    base_keys = ["match_id", "season", "date"]
    
    home_cols = {col: f"home_{col}" for col in df.columns if col not in base_keys + ["side"]}
    away_cols = {col: f"away_{col}" for col in df.columns if col not in base_keys + ["side"]}
    
    home_df = home_df.rename(columns=home_cols)
    away_df = away_df.rename(columns=away_cols)
    
    # merge side-by-side
    features = pd.merge(home_df, away_df, on=base_keys)
    
    # calculate the differentials 
    # positive means home is stronger negative means away
    features["diff_total"] = features["home_team_total"] - features["away_team_total"]
    features["diff_attack"] = features["home_team_attack"] - features["away_team_attack"]
    features["diff_defense"] = features["home_team_defense"] - features["away_team_defense"]
    
    # coverage quality control
    features["both_coverage_ok"] = features["home_coverage_ok"] & features["away_coverage_ok"]
    
    # organization 
    cols_order = base_keys + [
        "home_team", "away_team",
        "diff_total", "diff_attack", "diff_defense",
        "home_team_total", "away_team_total",
        "both_coverage_ok"
    ]
    remaining_cols = [c for c in features.columns if c not in cols_order]
    features = features[cols_order + remaining_cols]
    
    # Save
    features.to_csv(OUT_FILE, index=False)
    
    print("-" * 30)
    print(f"Success match features created")
    print(f"Total matches processed: {len(features)}")
    print(f"High quality matches (both teams OK): {features['both_coverage_ok'].sum()}")
    print(f"File saved: {OUT_FILE}")
    print("-" * 30)
    
if __name__ == "__main__":
    main()