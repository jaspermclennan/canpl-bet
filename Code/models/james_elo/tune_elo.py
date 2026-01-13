import pandas as pd
import numpy as np
from pathlib import Path
import itertools
import os


# --- PATH SETUP ---
cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet" else Path(__file__).resolve().parent.parent.parent.parent

MATCHES_FILE = REPO_ROOT / "data" / "matches" / "derived" / "match_model_ready.csv"
LINEUPS_FILE = REPO_ROOT / "data" / "lineups" / "assumed_lineup.csv"

def _norm_key(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def run_elo_simulation(matches, lineup_map, k_factor, home_adv):
    player_ratings = {}
    losses = []

    for _, row in matches.iterrows():
        mid = row["match_id"]

        players = lineup_map.get(mid, [])
        if not players:
            continue

        h_key = _norm_key(row.get("home_team", ""))
        a_key = _norm_key(row.get("away_team", ""))

        home_ratings, away_ratings = [], []
        home_ids, away_ids = [], []

        for p in players:
            pid = p["playerId"]
            p_key = _norm_key(p.get("team", ""))
            r = float(player_ratings.get(pid, 1500.0))

            if p_key == h_key:
                home_ratings.append(r)
                home_ids.append(pid)
            elif p_key == a_key:
                away_ratings.append(r)
                away_ids.append(pid)

        if not home_ratings or not away_ratings:
            continue

        h_elo = float(np.mean(home_ratings) + home_adv)
        a_elo = float(np.mean(away_ratings))

        # expected score for home (win=1, draw=0.5, loss=0)
        exp_home = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo) / 400.0))
        exp_away = 1.0 - exp_home

        # actual score from label (0 away, 1 draw, 2 home)
        label = int(row["label"])
        if label == 2:
            act_home, act_away = 1.0, 0.0
        elif label == 1:
            act_home, act_away = 0.5, 0.5
        else:
            act_home, act_away = 0.0, 1.0

        # Brier-style loss on expected score
        losses.append((act_home - exp_home) ** 2)

        # rating updates
        for pid in home_ids:
            player_ratings[pid] = float(player_ratings.get(pid, 1500.0) + k_factor * (act_home - exp_home))
        for pid in away_ids:
            player_ratings[pid] = float(player_ratings.get(pid, 1500.0) + k_factor * (act_away - exp_away))

    if not losses:
        return float("inf")
    return float(np.mean(losses))

def main():
    print("--- STARTING HYPERPARAMETER TUNING (IMPROVED) ---")

    matches = pd.read_csv(MATCHES_FILE)

    # Ensure required columns exist
    required = ["match_id", "home_team", "away_team", "label"]
    missing = [c for c in required if c not in matches.columns]
    if missing:
        raise ValueError(f"match_model_ready.csv missing columns: {missing}")

    matches["date"] = pd.to_datetime(matches.get("date"), errors="coerce")
    if "date" in matches.columns and matches["date"].notna().any():
        matches = matches.sort_values("date").reset_index(drop=True)

    lineups = pd.read_csv(LINEUPS_FILE)

    lineup_map = {}
    for mid, group in lineups.groupby("match_id"):
        lineup_map[mid] = group[["playerId", "team"]].to_dict("records")

    k_values = [10, 15, 20, 25, 30, 40]
    hfa_values = [20, 35, 50, 65, 80]

    best_score = float("inf")
    best_params = None

    print(f"Testing {len(k_values) * len(hfa_values)} combinations.")

    for k, hfa in itertools.product(k_values, hfa_values):
        loss = run_elo_simulation(matches, lineup_map, k, hfa)
        print(f"K={k:<2} | HFA={hfa:<2} | BrierLoss={loss:.5f}")

        if loss < best_score:
            best_score = loss
            best_params = {"k": k, "hfa": hfa}

    print("-" * 45)
    print("BEST PARAMETERS FOUND:")
    print(f"   K_FACTOR: {best_params['k']}")
    print(f"   HOME_ADVANTAGE: {best_params['hfa']}")
    print(f"   Lowest Brier Loss: {best_score:.5f}")
    print("-" * 45)

if __name__ == "__main__":
    main()
