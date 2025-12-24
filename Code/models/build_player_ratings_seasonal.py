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
        logging.FileHandler("player_processing.log"), # Saves to a file
        logging.StreamHandler()                       # Also prints to console
    ]
)
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # back to repo root
BASE_PLAYERS = BASE_DIR / "data" / "players" / "cleaned" / "cpl_players_all_seasons_cleaned.csv"
OUT_DIR = BASE_DIR / "data" / "players" / "derived"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SEASONAL = OUT_DIR / "player_ratings_seasonal.csv"

MIN_MINUTES = 450  # come back to this later
SHRINK_M = 450     # shrink towards 0 for low minutes

def safe_zscore(s: pd.Series) -> pd.Series:
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd

def normalize_role(role_val: object) -> str:
    r = str(role_val).strip().lower()
    if r in {"gk", "goalkeeper"}:
        return "goalkeeper"
    if r in {"df", "defender"}:
        return "defender"
    if r in {"mf", "midfielder"}:
        return "midfielder"
    if r in {"fw", "forward"}:
        return "forward"
    return r

ATTACK_FEATURES = {
    "G_per_game", "A_per_game", "S_per_game", "SOT_per_game", "KP_per_game",
    "Goals", "Assists", "Shots", "ShotsOnTarget", "KeyPasses", "GoalInvolvements",
    "PassPct", "FoulsSuffered_pg",
}

DEFENSE_FEATURES = {
    "Tackles_pg", "Tackles", "SavePct", "GAA",
    "FoulsCommitted_pg", "FoulsCommitted",
}

NEGATIVE_FEATURES = {
    "YellowCards_pg", "RedCards_pg", "Offsides_pg",
    "YellowCards", "RedCards", "Offsides",
}

ROLE_WEIGHTS = {
    "forward": {
        "attack_scale": 1.0,
        "defense_scale": 0.4,
        "neg_scale": 0.6,
        "feature_overrides": {
            "G_per_game": 2.5, "A_per_game": 1.5, "SOT_per_game": 0.9, "KP_per_game": 0.7,
            "Offsides_pg": -0.3, "YellowCards_pg": -0.4, "RedCards_pg": -1.8,
        },
    },
    "midfielder": {
        "attack_scale": 0.9,
        "defense_scale": 0.8,
        "neg_scale": 0.8,
        "feature_overrides": {
            "A_per_game": 1.6, "KP_per_game": 1.5, "PassPct": 0.6,
            "Tackles_pg": 0.8, "FoulsCommitted_pg": -0.6,
            "YellowCards_pg": -0.5, "RedCards_pg": -1.8,
        },
    },
    "defender": {
        "attack_scale": 0.6,
        "defense_scale": 1.2,
        "neg_scale": 1.0,
        "feature_overrides": {
            "Tackles_pg": 1.4, "PassPct": 0.6,
            "FoulsCommitted_pg": -0.7, "YellowCards_pg": -0.6, "RedCards_pg": -2.0,
        },
    },
    "goalkeeper": {
        "attack_scale": 0.4,
        "defense_scale": 1.6,
        "neg_scale": 1.0,
        "feature_overrides": {
            "SavePct": 2.5, "GAA": -2.0, "PassPct": 0.6,
            "YellowCards_pg": -0.4, "RedCards_pg": -2.0,
        },
    },
}

def main() -> None:
    logger.info("Starting player rating processing...")
    
    # Validation & Loading
    if not BASE_PLAYERS.exists():
        logger.error(f"Base player file not found: {BASE_PLAYERS}")
        return
    
    df = pd.read_csv(BASE_PLAYERS)
    
    required = {"playerName", "season", "role", "Minutes", "GamesPlayed"}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"Missing required columns in base file: {sorted(missing)}")
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    
    # Pre-processing & Eligibility
    df["role"] = df["role"].apply(normalize_role)
    df["Minutes"] = pd.to_numeric(df["Minutes"], errors="coerce").fillna(0.0)
    df["GamesPlayed"] = pd.to_numeric(df["GamesPlayed"], errors="coerce").fillna(0.0)
    df["Eligible"] = df["Minutes"] >= MIN_MINUTES
    
    logger.info(f"Total rows: {len(df)} | Eligible: {df['Eligible'].sum()}")

    gp = df["GamesPlayed"].replace(0, np.nan)
    
    # Rate Stats Calculation
    per_game_sources = {
        "Tackles_pg": "Tackles", "FoulsCommitted_pg": "FoulsCommitted",
        "FoulsSuffered_pg": "FoulsSuffered", "Offsides_pg": "Offsides",
        "YellowCards_pg": "YellowCards", "RedCards_pg": "RedCards",
    }
    for out_col, base_col in per_game_sources.items():
        if base_col in df.columns:
            df[out_col] = (pd.to_numeric(df[base_col], errors="coerce") / gp).fillna(0.0)
        else:
            df[out_col] = 0.0
            
    standard_numeric = [
        "G_per_game", "A_per_game", "S_per_game", "SOT_per_game", "KP_per_game",
        "Goals", "Assists", "Shots", "ShotsOnTarget", "KeyPasses", 
        "GoalInvolvements", "PassPct", "SavePct", "GAA"
    ]
    for col in standard_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0 if "_per_game" in col else np.nan)
        else:
            df[col] = 0.0 if "_per_game" in col else np.nan

    # Main Processing Loop
    numeric_candidates = df.select_dtypes(include=["number"]).columns.tolist()
    z_features = [c for c in numeric_candidates if c not in {"Minutes", "GamesPlayed"}]
    
    out_groups = []
    for (season, role), g in df.groupby(["season", "role"], dropna=False):
        g = g.copy()
        role_key = str(role)
        role_cfg = ROLE_WEIGHTS.get(role_key, {}) 
        
        for feat in z_features:
            g[f"z_{feat}"] = safe_zscore(g[feat])
            
        g["AttackRaw"] = 0.0
        g["DefenseRaw"] = 0.0
        g["NegRaw"] = 0.0
        
        for feat in ATTACK_FEATURES:
            if f"z_{feat}" in g.columns:
                g["AttackRaw"] += g[f"z_{feat}"].fillna(0)
        for feat in DEFENSE_FEATURES:
            if f"z_{feat}" in g.columns:
                g["DefenseRaw"] += g[f"z_{feat}"].fillna(0)
        for feat in NEGATIVE_FEATURES:
            if f"z_{feat}" in g.columns:
                g["NegRaw"] += g[f"z_{feat}"].fillna(0)
        
        if role_cfg:
            g["AttackRaw"] *= role_cfg.get("attack_scale", 1.0)
            g["DefenseRaw"] *= role_cfg.get("defense_scale", 1.0)
            g["NegRaw"] *= role_cfg.get("neg_scale", 1.0)
        
            overrides = role_cfg.get("feature_overrides", {})
            for feat, wt in overrides.items():
                zcol = f"z_{feat}"
                if zcol in g.columns:
                    if feat in ATTACK_FEATURES:
                        g["AttackRaw"] += (wt - 1.0) * g[zcol].fillna(0)
                    elif feat in DEFENSE_FEATURES:
                        g["DefenseRaw"] += (wt - 1.0) * g[zcol].fillna(0)
                    elif feat in NEGATIVE_FEATURES:
                        g["NegRaw"] += (wt - 1.0) * g[zcol].fillna(0)
                
        g["TotalRaw"] = g["AttackRaw"] + g["DefenseRaw"] - g["NegRaw"]
        shrink = g["Minutes"] / (g["Minutes"] + SHRINK_M)
        g["AttackShrunk"] = g["AttackRaw"] * shrink
        g["DefenseShrunk"] = g["DefenseRaw"] * shrink
        g["TotalShrunk"] = g["TotalRaw"] * shrink
        
        # --- Percentile Rank Logic ---
        # Only rank eligible players so the percentiles aren't skewed by 1-minute cameos
        eligible_mask = g["Eligible"]
        if eligible_mask.any():
            g.loc[eligible_mask, "PercentileRank"] = (
                g.loc[eligible_mask, "TotalShrunk"].rank(pct=True) * 100
            ).round(1)
        
        g.loc[~eligible_mask, ["AttackShrunk", "DefenseShrunk", "TotalShrunk"]] = np.nan
        out_groups.append(g)
    
    # Consolidation & SOS Calculation
    scored = pd.concat(out_groups, ignore_index=True)
    
    logger.info("Calculating Season-over-Season improvements...")
    scored = scored.sort_values(["playerName", "season"])
    scored["PrevSeasonScore"] = scored.groupby("playerName")["TotalShrunk"].shift(1)
    scored["ScoreDelta"] = (scored["TotalShrunk"] - scored["PrevSeasonScore"]).round(3)

    # Final Save
    keep_cols = [
        "playerName", "playerId", "season", "team", "role",
        "TotalShrunk", "PercentileRank", "ScoreDelta", "PrevSeasonScore",
        "AttackShrunk", "DefenseShrunk", "Minutes", "Eligible",
        "G_per_game", "A_per_game", "SOT_per_game", "KP_per_game",
        "Tackles_pg", "PassPct", "SavePct", "GAA"
    ]
    keep_cols = [c for c in keep_cols if c in scored.columns]
    
    scored = scored[keep_cols].sort_values(
        ["season", "PercentileRank"], 
        ascending=[True, False]
    )
    
    scored.to_csv(OUT_SEASONAL, index=False)
    logger.info(f"Successfully saved {len(scored)} ratings to {OUT_SEASONAL}")

if __name__ == "__main__":
    main()