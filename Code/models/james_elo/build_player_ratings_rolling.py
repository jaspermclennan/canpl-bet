import pandas as pd
import numpy as np
from pathlib import Path
import re

# --- PATH SETUP ---
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

MATCHES_FILE = REPO_ROOT / "data" / "matches" / "processed" / "all_matches_with_baseline.csv"
LINEUPS_FILE = REPO_ROOT / "data" / "lineups" / "assumed_lineup.csv"
OUT_FILE = REPO_ROOT / "data" / "players" / "derived" / "player_ratings_rolling.csv"

# SETTINGS
K_FACTOR = 20.0
HOME_ADVANTAGE = 50.0

# If True, scale Elo updates by goal margin
USE_MARGIN_SCALING = True
MARGIN_CAP = 3
MARGIN_SCALE = 0.10  # +10% per goal up to cap

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
    "FC Edmonton": "Edmonton",
}

def clean_team_name(name) -> str:
    name = str(name).strip()
    name = name.replace(" FC", "").replace("FC ", "FC ").replace("FC", "").strip()
    return TEAM_NAME_MAP.get(name, name)

def norm_key(s: str) -> str:
    """Aggressive normalize for comparisons (lowercase, no spaces)."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def main():
    print("--- BUILDING PLAYER ELO RATINGS (IMPROVED) ---")

    if not MATCHES_FILE.exists():
        raise FileNotFoundError(f"Match file not found: {MATCHES_FILE}")
    if not LINEUPS_FILE.exists():
        raise FileNotFoundError(f"Lineups file not found: {LINEUPS_FILE}")

    matches = pd.read_csv(MATCHES_FILE)

    # Scores
    if "HomeScore" not in matches.columns and "home_score" in matches.columns:
        matches = matches.rename(columns={"home_score": "HomeScore", "away_score": "AwayScore"})

    # Date
    date_col = "Date" if "Date" in matches.columns else ("date" if "date" in matches.columns else None)
    if not date_col:
        raise ValueError("Match file missing date column (expected Date or date).")

    matches["date"] = pd.to_datetime(matches[date_col], errors="coerce")
    matches = matches.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Generate match_id (must match other parts of your pipeline)
    matches["season_id"] = matches["date"].dt.year.astype(str)
    matches["date_id"] = matches["date"].dt.strftime("%Y-%m-%d")
    matches["h_clean"] = matches["HomeTeam"].apply(clean_team_name)
    matches["a_clean"] = matches["AwayTeam"].apply(clean_team_name)

    matches["match_id"] = (
        matches["season_id"] + "_" +
        matches["date_id"] + "_" +
        matches["h_clean"] + "_vs_" +
        matches["a_clean"]
    )

    # Index lineups by match_id
    lineups = pd.read_csv(LINEUPS_FILE)

    # Optional: if lineups have a team column with FC suffixes etc, clean it.
    lineups["team_clean"] = lineups["team"].apply(clean_team_name)

    lineup_map = {}
    for mid, group in lineups.groupby("match_id"):
        lineup_map[mid] = group[["playerId", "team", "team_clean"]].to_dict("records")

    player_ratings = {}  # playerId -> rating
    history_records = []

    print(f"   Processing {len(matches)} matches chronologically...")

    for _, row in matches.iterrows():
        mid = row["match_id"]
        players = lineup_map.get(mid, [])
        if not players:
            continue

        h_team = row["h_clean"]
        a_team = row["a_clean"]

        h_norm = norm_key(h_team)
        a_norm = norm_key(a_team)

        h_ratings, a_ratings = [], []
        active_home, active_away = [], []

        for p in players:
            pid = p["playerId"]
            p_team = p.get("team_clean", p.get("team", ""))

            p_norm = norm_key(p_team)
            rating = float(player_ratings.get(pid, 1500.0))

            # Save pre-match rating (compatible output)
            history_records.append({
                "match_id": mid,
                "playerId": pid,
                "team": p.get("team", ""),
                "date": row["date"],
                "Rating": rating,
            })

            # Strict assignment to home/away
            if p_norm == h_norm:
                h_ratings.append(rating)
                active_home.append(pid)
            elif p_norm == a_norm:
                a_ratings.append(rating)
                active_away.append(pid)
            else:
                # unknown team mapping -> ignore this player for strength + updates
                continue

        if not h_ratings or not a_ratings:
            continue

        h_elo = float(np.mean(h_ratings) + HOME_ADVANTAGE)
        a_elo = float(np.mean(a_ratings))

        # Elo expected score for home (win=1, draw=0.5, loss=0)
        ea_h = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo) / 400.0))
        ea_a = 1.0 - ea_h

        # Actual score
        hs = float(row["HomeScore"])
        as_ = float(row["AwayScore"])

        if hs > as_:
            sa_h, sa_a = 1.0, 0.0
        elif hs == as_:
            sa_h, sa_a = 0.5, 0.5
        else:
            sa_h, sa_a = 0.0, 1.0

        # Margin-of-victory scaling 
        k_eff = K_FACTOR
        if USE_MARGIN_SCALING:
            margin = min(abs(hs - as_), MARGIN_CAP)
            k_eff = K_FACTOR * (1.0 + (margin * MARGIN_SCALE))

        for pid in active_home:
            player_ratings[pid] = float(player_ratings.get(pid, 1500.0) + k_eff * (sa_h - ea_h))
        for pid in active_away:
            player_ratings[pid] = float(player_ratings.get(pid, 1500.0) + k_eff * (sa_a - ea_a))

    out_df = pd.DataFrame(history_records)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_FILE, index=False)

    print(f"✅ Saved ELO ratings to: {OUT_FILE}")
    print(f"   Total Player-Match Records: {len(out_df)}")

if __name__ == "__main__":
    main()
