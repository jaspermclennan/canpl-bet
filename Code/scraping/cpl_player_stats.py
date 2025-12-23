import requests
import pandas as pd

from pathlib import Path
from config import SEASON_ID_TO_YEAR

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "https://api-sdp.canpl.ca/v1/cpl/football/seasons/{season_id}/stats/players"

HEADERS = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Content-Type": "application/json; charset=UTF-8",
    "Host": "api-sdp.canpl.ca",
    "Origin": "https://www.canpl.ca",
    "Referer": "https://www.canpl.ca/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/142.0.0.0 Safari/537.36",
}

# Season IDs
season_ids = [
    "cpl::Football_Season::fd43e1d61dfe4396a7356bc432de0007",  # 2025
    "cpl::Football_Season::6fb9e6fae4f24ce9bf4fa3172616a762",  # 2024
    "cpl::Football_Season::fc0855108c9044218a84fc5d2bee0000",  # 2023
    "cpl::Football_Season::046f0ab31ba641c7b7bf27eb0dda4b9d",  # 2022
    "cpl::Football_Season::2f07c39671b84933ad7bb1e1958a7427",  # 2021
    "cpl::Football_Season::11aa5cc094d0481fa8e73d326763584f",  # 2020
    "cpl::Football_Season::c8c9bdc288f34aa89073a8bd89d2da3e",  # 2019
]

roles = ["goalkeeper", "defender", "midfielder", "forward"]

def fetch_players_page(season_id, role, page=1):
    params = {
        "locale": "en-US",
        "category": "general",
        "role": role,
        "direction": "desc",
        "page": page,
        "pageNumElement": 250,
    }
    url = BASE_URL.format(season_id=season_id)
    res = requests.get(url, params=params, headers=HEADERS)
    res.raise_for_status()
    return res.json()

def fetch_all_players(season_id, role):
    players = []
    page = 1
    while True:
        data = fetch_players_page(season_id, role, page)
        if not data["players"]:
            break
        players.extend(data["players"])
        page += 1
    return players

def flatten_player(p):
    row = {
        "playerId": p["playerId"],
        "playerName": p.get("displayName") or p.get("shortName"),
        "team": p["team"]["shortName"],
        "position": p["roleLabel"],
    }
    for stat in p["stats"]:
        abbr = stat["statsLabelAbbreviation"]
        if abbr:
            row[abbr] = stat["statsValue"]
    return row

all_rows = []

for season_id in season_ids:
    season_year = SEASON_ID_TO_YEAR.get(season_id)
    
    if season_year is None:
        raise KeyError(f"Missing season_id in config: {season_id}")
        
    for role in roles:
        players = fetch_all_players(season_id, role)
        for p in players:
            row = flatten_player(p)
            row["season"] = season_year   # or use season_id
            row["role"] = role            # e.g. goalkeeper/defender
            all_rows.append(row)

df = pd.DataFrame(all_rows)

out_path = DATA_DIR / "cpl_players_combined.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")