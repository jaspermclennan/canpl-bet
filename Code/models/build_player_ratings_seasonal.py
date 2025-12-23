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
        logging.StreamHandler()                      # Also prints to console
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
    
    if not BASE_PLAYERS.exists():
        logger.error(f"File not found: {BASE_PLAYERS}")
        return # Graceful exit instead of crash

    df = pd.read_csv(BASE_PLAYERS)
    
    # Track Eligibility
    eligible_count = (df["Minutes"] >= MIN_MINUTES).sum()
    ineligible_count = len(df) - eligible_count
    logger.info(f"Total players: {len(df)} | Eligible: {eligible_count} | Ineligible: {ineligible_count}")

    # Inside your loop, you can log specific skips
    for (season, role), g in df.groupby(["season", "role"], dropna=False):
        if str(role) not in ROLE_WEIGHTS:
            logger.warning(f"Unknown role '{role}' found in season {season}. Using default weights.")
            
    if not BASE_PLAYERS.exists():
        raise FileNotFoundError(f"Base player file not found: {BASE_PLAYERS}")
    
    df = pd.read_csv(BASE_PLAYERS)
    
    required = {"playerName", "season", "role", "Minutes", "GamesPlayed"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in base file: {sorted(missing)}")
    
    # Corrected the function name from normalize_role
    df["role"] = df["role"].apply(normalize_role)
    
    df["Minutes"] = pd.to_numeric(df["Minutes"], errors="coerce").fillna(0.0)
    df["GamesPlayed"] = pd.to_numeric(df["GamesPlayed"], errors="coerce").fillna(0.0)
    gp = df["GamesPlayed"].replace(0, np.nan)
    
    per_game_sources = {
        "Tackles_pg": "Tackles",
        "FoulsCommitted_pg": "FoulsCommitted",
        "FoulsSuffered_pg": "FoulsSuffered",
        "Offsides_pg": "Offsides",
        "YellowCards_pg": "YellowCards",
        "RedCards_pg": "RedCards",
    }
    for out_col, base_col in per_game_sources.items():
        if base_col in df.columns:
            df[out_col] = (pd.to_numeric(df[base_col], errors="coerce") / gp).fillna(0.0)
        else:
            df[out_col] = 0.0
            
    for col in ["G_per_game", "A_per_game", "S_per_game", "SOT_per_game", "KP_per_game"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
            
    for col in ["Goals", "Assists", "Shots", "ShotsOnTarget", "KeyPasses", "GoalInvolvements", "PassPct", "SavePct", "GAA"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["Eligible"] = df["Minutes"] >= MIN_MINUTES
    
    # Fixed syntax: include=["number"] and corrected variable name
    numeric_candidates = df.select_dtypes(include=["number"]).columns.tolist()
    z_features = [c for c in numeric_candidates if c not in {"Minutes", "GamesPlayed"}]
    
    out_groups = []
    for (season, role), g in df.groupby(["season", "role"], dropna=False):
        g = g.copy()
        role_key = str(role)
        role_cfg = ROLE_WEIGHTS.get(role_key, {}) # Default to empty dict
        
        for feat in z_features:
            g[f"z_{feat}"] = safe_zscore(g[feat])
            
        g["AttackRaw"] = 0.0
        g["DefenseRaw"] = 0.0
        g["NegRaw"] = 0.0
        
        for feat in ATTACK_FEATURES:
            if f"z_{feat}" in g.columns:
                g["AttackRaw"] += g[f"z_{feat}"].fillna(0) # Use 'g' not 'f'
                
        for feat in DEFENSE_FEATURES:
            if f"z_{feat}" in g.columns:
                g["DefenseRaw"] += g[f"z_{feat}"].fillna(0)

        for feat in NEGATIVE_FEATURES:
            if f"z_{feat}" in g.columns:
                g["NegRaw"] += g[f"z_{feat}"].fillna(0)
        
        if role_cfg:
            g["AttackRaw"] *= role_cfg["attack_scale"]
            g["DefenseRaw"] *= role_cfg["defense_scale"]
            g["NegRaw"] *= role_cfg["neg_scale"]
        
            overrides = role_cfg.get("feature_overrides", {})
            for feat, wt in overrides.items():
                zcol = f"z_{feat}"
                if zcol not in g.columns:
                    continue
            
                if feat in ATTACK_FEATURES:
                    g["AttackRaw"] += (wt - 1.0) * g[zcol].fillna(0)
                elif feat in DEFENSE_FEATURES:
                    g["DefenseRaw"] += (wt - 1.0) * g[zcol].fillna(0)
                elif feat in NEGATIVE_FEATURES:
                    g["NegRaw"] += (wt - 1.0) * g[zcol].fillna(0)
                else:
                    g["AttackRaw"] += wt * g[zcol].fillna(0)
                
        # Total Raw: Subtract NegRaw
        g["TotalRaw"] = g["AttackRaw"] + g["DefenseRaw"] - g["NegRaw"]
        
        # Minutes shrinkage
        shrink = g["Minutes"] / (g["Minutes"] + SHRINK_M)
        g["AttackShrunk"] = g["AttackRaw"] * shrink
        g["DefenseShrunk"] = g["DefenseRaw"] * shrink
        g["TotalShrunk"] = g["TotalRaw"] * shrink
        
        # Filter Eligibility
        g.loc[~g["Eligible"], ["AttackShrunk", "DefenseShrunk", "TotalShrunk"]] = np.nan
        
        out_groups.append(g)
    
    if not out_groups:
        print("No data processed.")
        return

    scored = pd.concat(out_groups, ignore_index=True)

    keep_cols = [
        "playerName", "playerId", "season", "team", "position", "role",
        "GamesPlayed", "Minutes", "Eligible",
        "AttackRaw", "DefenseRaw", "NegRaw", "TotalRaw",
        "AttackShrunk", "DefenseShrunk", "TotalShrunk",
        "G_per_game", "A_per_game", "SOT_per_game", "KP_per_game",
        "Tackles_pg", "FoulsCommitted_pg", "YellowCards_pg", "RedCards_pg", "Offsides_pg",
        "PassPct", "SavePct", "GAA",
    ]
    keep_cols = [c for c in keep_cols if c in scored.columns]
    
    scored = scored[keep_cols].sort_values(["season", "role", "TotalShrunk"], ascending=[True, True, False])
    scored.to_csv(OUT_SEASONAL, index=False)
    
    logger.info(f"Successfully saved ratings to {OUT_SEASONAL}")
    print(f"Saved seasonal player ratings: {OUT_SEASONAL} ({len(scored)} rows)")

if __name__ == "__main__":
    main()