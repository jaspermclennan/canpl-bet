import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import json


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "data").exists():
            return p
    return start.parents[3]


REPO_ROOT = find_repo_root(Path(__file__).resolve())

# Inputs
ROLLING_FILE = REPO_ROOT / "data" / "players" / "derived" / "player_ratings_rolling.csv"
LINEUPS_FILE = REPO_ROOT / "data" / "lineups" / "assumed_lineup.csv"
MODEL_DIR = REPO_ROOT / "models"

# Output
OUT_FILE = REPO_ROOT / "data" / "matches" / "derived" / "2026_season_predictions.csv"


def get_latest_rosters():
    """Determines who is on which team based on late-2025 data."""
    print("   Building 2026 rosters from 2025 data...")

    if not ROLLING_FILE.exists():
        print("Missing rolling ratings file.")
        print(f"Looked for: {ROLLING_FILE}")
        return None

    ratings = pd.read_csv(ROLLING_FILE)
    ratings["date"] = pd.to_datetime(ratings["date"], errors="coerce", utc=True)
    ratings = ratings.dropna(subset=["date"])

    # Required columns in ratings
    required = {"playerId", "team", "Rating"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"player_ratings_rolling.csv missing columns: {sorted(missing)}")

    # Latest row per player = last known team + last rating
    latest_ratings = (
        ratings.sort_values("date")
               .groupby("playerId", as_index=False)
               .tail(1)
               [["playerId", "team", "Rating", "date"]]
               .rename(columns={"team": "team_latest"})
    )

    # Optional: enrich with playerName from lineups if available
    if LINEUPS_FILE.exists():
        lineups = pd.read_csv(LINEUPS_FILE)
        if "date" in lineups.columns:
            lineups["date"] = pd.to_datetime(lineups["date"], errors="coerce", utc=True)

        # Try to pull the most recent playerName we can find
        if {"playerId", "playerName"}.issubset(lineups.columns):
            name_map = (
                lineups.dropna(subset=["playerId", "playerName"])
                      .sort_values("date" if "date" in lineups.columns else "playerId")
                      .groupby("playerId", as_index=False)
                      .tail(1)[["playerId", "playerName"]]
            )
            latest_ratings = latest_ratings.merge(name_map, on="playerId", how="left")

    # For downstream code, keep column name 'team'
    roster_df = latest_ratings.rename(columns={"team_latest": "team"})
    return roster_df


def calculate_team_strength(roster_df):
    """Calculates the 'Best XI' strength for each team."""
    team_strengths = {}

    print("\n--- 2026 POWER RANKINGS (Start of Season) ---")
    print(f"{'Rank':<5} | {'Team':<20} | {'ELO Strength (Top 11)':<10}")
    print("-" * 55)

    for team, group in roster_df.groupby("team"):
        top_players = group.sort_values("Rating", ascending=False).head(14)

        starters = top_players.iloc[:11]["Rating"].sum()
        subs = top_players.iloc[11:]["Rating"].sum() * 0.3
        total_strength = starters + subs

        team_strengths[team] = float(total_strength)

    sorted_teams = sorted(team_strengths.items(), key=lambda x: x[1], reverse=True)
    for rank, (team, strength) in enumerate(sorted_teams, 1):
        print(f"{rank:<5} | {team:<20} | {strength:.1f}")

    return team_strengths


def predict_opening_matchups(team_strengths):
    """Uses your trained probability model to produce H/D/A probabilities for hypothetical matchups."""
    model_path = MODEL_DIR / "logistic_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    meta_path = MODEL_DIR / "probability_model_meta.json"

    if not model_path.exists() or not scaler_path.exists():
        print("\nModel files not found. Run build_probability_model.py first.")
        return
    if not meta_path.exists():
        print("\nMissing probability_model_meta.json. Re-run build_probability_model.py.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feat_order = meta.get("features", [])
    if not feat_order:
        raise ValueError("Meta file has no 'features' list; cannot construct feature vectors.")

    print("\n--- 2026 OPENING MATCHUP PROJECTIONS ---")
    print(f"{'Home':<15} vs {'Away':<15} | {'Home Win':<8} | {'Draw':<8} | {'Away Win':<8} | {'Fair Odds (H/D/A)'}")
    print("-" * 95)

    teams = list(team_strengths.keys())
    eps = 1e-9

    # We don't have 2026 form yet. Use neutral defaults (0) for form diffs.
    # If later you simulate early-season form, this is where you'd plug it in.
    for home in teams:
        for away in teams:
            if home == away:
                continue

            feature_map = {
                "diff_total": team_strengths[home] - team_strengths[away],
                "diff_form_pts": 0.0,
                "diff_form_gd": 0.0,
            }

            # Build vector in trained feature order
            x = np.array([[feature_map.get(f, 0.0) for f in feat_order]], dtype=float)

            features_scaled = scaler.transform(x)
            probs = model.predict_proba(features_scaled)[0]
            class_to_prob = dict(zip(model.classes_, probs))

            # Your labels are [0, 1, 2] where 2=Home win, 1=Draw, 0=Away win
            p_away = float(class_to_prob.get(0, 0.0))
            p_draw = float(class_to_prob.get(1, 0.0))
            p_home = float(class_to_prob.get(2, 0.0))

            odds_h = 1.0 / max(eps, p_home)
            odds_d = 1.0 / max(eps, p_draw)
            odds_a = 1.0 / max(eps, p_away)

            print(
                f"{home:<15} vs {away:<15} | "
                f"{p_home:>7.1%} | {p_draw:>7.1%} | {p_away:>7.1%} | "
                f"{odds_h:>5.2f}/{odds_d:>5.2f}/{odds_a:>5.2f}"
            )



def main():
    roster = get_latest_rosters()
    if roster is None or roster.empty:
        return

    team_strengths = calculate_team_strength(roster)
    predict_opening_matchups(team_strengths)


if __name__ == "__main__":
    main()
