from __future__ import annotations

import os
from math import radians, cos, sin, asin, sqrt
from pathlib import Path

import pandas as pd

from cpl_stadiums import STADIUMS, TEAM_MAP
from fetch_weather import get_weather_estimate


cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else Path(__file__).resolve().parent.parent.parent.parent

MATCH_FILE = REPO_ROOT / "data" / "matches" / "processed" / "all_matches_with_baseline.csv"
OUT_FILE = REPO_ROOT / "data" / "matches" / "derived" / "external_factors.csv"


_TZ_ORDER = {
    "Pacific": 0,
    "Mountain": 1,
    "Central": 2,
    "Eastern": 3,
    "Atlantic": 4,
}


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometers between two (lon, lat) points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371.0  # Earth radius (km)
    return c * r


def tz_index(tz_name: str) -> int:
    return _TZ_ORDER.get(str(tz_name), _TZ_ORDER["Eastern"])


def get_fatigue_score(days_rest: int, distance_traveled_km: float, timezone_change: int) -> float:
    """Simple heuristic fatigue score (higher = more fatigue)."""
    fatigue = 0.0

    # 1) Rest
    if days_rest <= 3:
        fatigue += 3.0
    elif days_rest == 4:
        fatigue += 1.5
    elif days_rest == 5:
        fatigue += 0.5

    # 2) Travel: 1000 km => +0.5
    fatigue += (distance_traveled_km / 1000.0) * 0.5

    # 3) Time-zone change: 1 zone => +0.5
    fatigue += abs(timezone_change) * 0.5

    return float(fatigue)


def main():
    print("--- CALCULATING EXTERNAL FACTORS (TRAVEL, WEATHER, SCORING) ---")

    if not MATCH_FILE.exists():
        print(f"Missing {MATCH_FILE}")
        return

    df = pd.read_csv(MATCH_FILE)
    df["date"] = pd.to_datetime(df["Date"])
    df["home_team"] = df["HomeTeam"].map(TEAM_MAP).fillna(df["HomeTeam"])
    df["away_team"] = df["AwayTeam"].map(TEAM_MAP).fillna(df["AwayTeam"])
    df = df.sort_values("date").reset_index(drop=True)

    # Track history per team
    last_played: dict[str, pd.Timestamp] = {}
    last_location: dict[str, str] = {}  # last venue key 

    # Track goal history {team: [g1, g2, ...]}
    team_goals_history: dict[str, list[float]] = {}

    features = []
    print(f"   Processing {len(df)} matches...")

    for _, row in df.iterrows():
        mid = row.get("match_id")
        if pd.isna(mid):
            mid = f"{row['date'].strftime('%Y-%m-%d')}_{row['home_team']}_vs_{row['away_team']}"

        h_team = row["home_team"]
        a_team = row["away_team"]
        match_date = row["date"]

        # Venue is home team's stadium
        h_stad_info = STADIUMS.get(h_team, STADIUMS.get("York"))
        current_loc_info = h_stad_info

        # --- HOME TEAM ---
        h_days_rest = 7
        h_dist = 0.0
        h_tz_change = 0

        if h_team in last_played:
            h_days_rest = int((match_date - last_played[h_team]).days)
            prev_loc_name = last_location.get(h_team, h_team)
            prev_loc = STADIUMS.get(prev_loc_name, h_stad_info)

            h_dist = haversine(prev_loc["lon"], prev_loc["lat"], h_stad_info["lon"], h_stad_info["lat"])
            h_tz_change = tz_index(h_stad_info.get("tz")) - tz_index(prev_loc.get("tz"))

        h_fatigue = get_fatigue_score(h_days_rest, h_dist, h_tz_change)

        # --- AWAY TEAM ---
        a_days_rest = 7
        a_dist = 0.0
        a_tz_change = 0

        if a_team in last_played:
            a_days_rest = int((match_date - last_played[a_team]).days)
            prev_loc_name = last_location.get(a_team, a_team)
            prev_loc = STADIUMS.get(prev_loc_name, STADIUMS.get("York"))

            a_dist = haversine(prev_loc["lon"], prev_loc["lat"], current_loc_info["lon"], current_loc_info["lat"])
            a_tz_change = tz_index(current_loc_info.get("tz")) - tz_index(prev_loc.get("tz"))
        else:
            # If no prior match, approximate travel as home-base -> venue
            a_home_base = STADIUMS.get(a_team, STADIUMS.get("York"))
            a_dist = haversine(a_home_base["lon"], a_home_base["lat"], current_loc_info["lon"], current_loc_info["lat"])
            a_tz_change = tz_index(current_loc_info.get("tz")) - tz_index(a_home_base.get("tz"))

        a_fatigue = get_fatigue_score(a_days_rest, a_dist, a_tz_change)

        # --- WEATHER ---
        month = int(match_date.month)
        avg_temp, rain_prob = get_weather_estimate(h_stad_info["name"], month)

        # --- SCORING FORM (rolling avg goals for last 5 matches, before today) ---
        def get_avg_goals(team: str) -> float:
            hist = team_goals_history.get(team, [])
            if not hist:
                return 1.2  # league-ish default
            recent = hist[-5:]
            return float(sum(recent) / len(recent))

        h_avg_goals = get_avg_goals(h_team)
        a_avg_goals = get_avg_goals(a_team)

        # --- UPDATE HISTORY ---
        last_played[h_team] = match_date
        last_played[a_team] = match_date
        last_location[h_team] = h_team  # venue key for this match
        last_location[a_team] = h_team  # away team also played at home team's venue

        # Add actual goals from today to history (if present)
        if pd.notna(row.get("HomeScore")):
            team_goals_history.setdefault(h_team, []).append(float(row["HomeScore"]))
            team_goals_history.setdefault(a_team, []).append(float(row["AwayScore"]))

        features.append(
            {
                "match_id": mid,
                "fatigue_home": round(h_fatigue, 4),
                "fatigue_away": round(a_fatigue, 4),
                "travel_km_away": round(a_dist, 1),
                "tz_change_away": int(a_tz_change),
                "weather_temp": float(avg_temp),
                "weather_rain_prob": float(rain_prob),
                "avg_goals_home": round(h_avg_goals, 4),
                "avg_goals_away": round(a_avg_goals, 4),
                "rain_impact_home": float(rain_prob) * float(h_avg_goals),
                "rain_impact_away": float(rain_prob) * float(a_avg_goals),
            }
        )

    out_df = pd.DataFrame(features)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_FILE, index=False)

    print(f"Saved extended features to: {OUT_FILE}")
    cols = ["match_id", "fatigue_away", "travel_km_away", "tz_change_away", "weather_rain_prob", "avg_goals_away", "rain_impact_away"]
    print(out_df[cols].tail())


if __name__ == "__main__":
    main()
