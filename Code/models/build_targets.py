import pandas as pd
from pathlib import Path
import os

cwd = Path(os.getcwd())
REPO_ROOT = cwd if (cwd / "data").exists() else Path(__file__).resolve().parents[2]

FEATURES_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_with_form.csv"
BASELINE_FILE = REPO_ROOT / "data" / "matches" / "processed" / "all_matches_with_baseline.csv" 
OUT_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"


TEAM_NAME_MAP = {
    "HFX Wanderers": "Wanderers",
    "Halifax Wanderers": "Wanderers",
    "HFX Wanderers FC": "Wanderers",
    "York United": "York",
    "York United FC": "York",
    "Atlético Ottawa": "Atlético",
    "Atletico Ottawa": "Atlético",
    "Pacific": "Pacific",
    "Pacific FC": "Pacific",
    "Valour": "Valour",
    "Valour FC": "Valour",
    "Forge": "Forge",
    "Forge FC": "Forge",
    "Cavalry": "Cavalry",
    "Cavalry FC": "Cavalry",
    "Edmonton": "Edmonton",
    "FC Edmonton": "Edmonton"
}

def norm_team(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace(TEAM_NAME_MAP)

def norm_date(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str[:10]

def main():
    features = pd.read_csv(FEATURES_FILE)
    baseline = pd.read_csv(BASELINE_FILE)

    for df, s_col, d_col, h_col, a_col in [
        (features, 'season', 'date', 'home_team', 'away_team'),
        (baseline, 'Season', 'Date', 'HomeTeam', 'AwayTeam')
    ]:
        df["j_s"] = pd.to_numeric(df[s_col]).astype(int)
        df["j_d"] = norm_date(df[d_col])
        df["j_h"] = norm_team(df[h_col])
        df["j_a"] = norm_team(df[a_col])
        
    res_map = {"H": 2, "D": 1, "A": 0}
    baseline["label"] = baseline["Result"].str.upper().map(res_map)
    
    final_df = pd.merge(
        features,
        baseline[["j_s", "j_d", "j_h", "j_a", "HomeScore", "AwayScore", "label"]],
        on=["j_s", "j_d", "j_h", "j_a"],
        how="inner"
    ).drop(columns=["j_s", "j_d", "j_h", "j_a"])
    
    final_df.to_csv(OUT_FILE, index=False)
    print(f"Joined {len(final_df)} matches into {OUT_FILE}")
    
if __name__ == "__main__":
    main()
