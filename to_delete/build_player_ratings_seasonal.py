from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("player_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BASE_PLAYERS = BASE_DIR / "data" / "players" / "cleaned" / "cpl_players_all_seasons_cleaned.csv"
OUT_DIR = BASE_DIR / "data" / "players" / "derived"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SEASONAL = OUT_DIR / "player_ratings_seasonal.csv"

# --- CALIBRATION CONSTANTS ---
MIN_MINUTES = 450
SHRINK_M = 450  

# --- REFACTORED POSITIONAL WEIGHTS ---
ROLE_WEIGHTS = {
    "forward": {
        "attack_scale": 0.80, "defense_scale": 0.20, "neg_scale": 0.5,
        "feature_overrides": {"G_per_game": 2.5, "A_per_game": 1.5, "SOT_per_game": 1.2},
    },
    "midfielder": {
        "attack_scale": 0.50, "defense_scale": 0.50, "neg_scale": 0.7,
        "feature_overrides": {"A_per_game": 1.8, "KP_per_game": 1.8, "PassPct": 1.2},
    },
    "defender": {
        "attack_scale": 0.20, "defense_scale": 0.80, "neg_scale": 1.0,
        "feature_overrides": {"Tackles_pg": 2.0, "PassPct": 0.8, "FoulsCommitted_pg": -0.5},
    },
    "goalkeeper": {
        "attack_scale": 0.05, "defense_scale": 0.95, "neg_scale": 1.0,
        "feature_overrides": {"SavePct": 3.0, "GAA": -2.0},
    },
}

ATTACK_FEATURES = {"G_per_game", "A_per_game", "S_per_game", "SOT_per_game", "KP_per_game", "PassPct"}
DEFENSE_FEATURES = {"Tackles_pg", "FoulsSuffered_pg", "SavePct", "GAA"}
NEGATIVE_FEATURES = {"YellowCards_pg", "RedCards_pg", "Offsides_pg", "FoulsCommitted_pg"}

def safe_zscore(s: pd.Series) -> pd.Series:
    sd = s.std(skipna=True)
    return (s - s.mean(skipna=True)) / sd if sd > 0 else pd.Series(0.0, index=s.index)

def normalize_role(role_val: object) -> str:
    r = str(role_val).strip().lower()
    mapping = {"gk": "goalkeeper", "df": "defender", "mf": "midfielder", "fw": "forward"}
    return next((v for k, v in mapping.items() if k in r), r)

def main() -> None:
    logger.info("Starting refined player rating processing...")
    if not BASE_PLAYERS.exists(): return
    
    df = pd.read_csv(BASE_PLAYERS)
    df["role"] = df["role"].apply(normalize_role)
    df["Minutes"] = pd.to_numeric(df["Minutes"], errors="coerce").fillna(0.0)
    df["Eligible"] = df["Minutes"] >= MIN_MINUTES

    gp = df["GamesPlayed"].replace(0, np.nan)
    for col in ["Tackles", "FoulsCommitted", "FoulsSuffered", "Offsides", "YellowCards", "RedCards"]:
        df[f"{col}_pg"] = (pd.to_numeric(df[col], errors="coerce") / gp).fillna(0.0)

    out_groups = []
    for (season, role), g in df.groupby(["season", "role"]):
        g = g.copy()
        cfg = ROLE_WEIGHTS.get(role, {})
        
        all_feats = ATTACK_FEATURES | DEFENSE_FEATURES | NEGATIVE_FEATURES
        for f in all_feats:
            if f in g.columns:
                g[f"z_{f}"] = safe_zscore(pd.to_numeric(g[f], errors="coerce"))

        g["AttackRaw"] = sum(g[f"z_{f}"].fillna(0) for f in ATTACK_FEATURES if f"z_{f}" in g.columns)
        g["DefenseRaw"] = sum(g[f"z_{f}"].fillna(0) for f in DEFENSE_FEATURES if f"z_{f}" in g.columns)
        g["NegRaw"] = sum(g[f"z_{f}"].fillna(0) for f in NEGATIVE_FEATURES if f"z_{f}" in g.columns)
        
        # Applying positional weights
        g["AttackRaw"] *= cfg.get("attack_scale", 1.0)
        g["DefenseRaw"] *= cfg.get("defense_scale", 1.0)
        g["NegRaw"] *= cfg.get("neg_scale", 1.0)
        
        g["TotalRaw"] = g["AttackRaw"] + g["DefenseRaw"] - g["NegRaw"]
        
        # Shrinkage calculation
        shrink = g["Minutes"] / (g["Minutes"] + SHRINK_M)
        g["TotalShrunk"] = g["TotalRaw"] * shrink
        
        # Calculate and save the missing columns
        g["AttackShrunk"] = g["AttackRaw"] * shrink
        g["DefenseShrunk"] = g["DefenseRaw"] * shrink
        
        g.loc[g["Eligible"], "PercentileRank"] = (
            g.loc[g["Eligible"], "TotalShrunk"].rank(pct=True) * 100
        ).round(1)
        
        out_groups.append(g)

    final = pd.concat(out_groups, ignore_index=True)
    final = final.sort_values(["playerName", "season"])
    final["ScoreDelta"] = final.groupby("playerName")["TotalShrunk"].diff().round(3)

    cols = [
        "playerName", "playerId", "season", "team", "role", 
        "TotalShrunk", "PercentileRank", "ScoreDelta", "Minutes",
        "AttackShrunk", "DefenseShrunk"
    ]
    final[cols].to_csv(OUT_SEASONAL, index=False)
    logger.info(f"Refined ratings saved to {OUT_SEASONAL}")

if __name__ == "__main__":
    main()