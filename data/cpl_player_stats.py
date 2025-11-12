import requests
import pandas as pd

# Base URL for CPL Player Stats API
BASE_URL = "https://api-sdp.canpl.ca/v1/cpl/football/seasons/{season_id}/stats/players"

HEADERS = { 
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Content-Type": "application/json; charset=UTF-8",
    "Host": "adi-sdp.canpl.ca",
    "Origin": "https://canpl.ca",
    "Referer": "https://canpl.ca/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
}

# Season IDs
season_ids = {
    "cpl::Football_Season::fd43e1d61dfe4396a7356bc432de0007", #2025
    "cpl::Football_Season::6fb9e6fae4f24ce9bf4fa3172616a762", #2024
    "cpl::Football_Season::fc0855108c9044218a84fc5d2bee0000", #2023
    "cpl::Football_Season::046f0ab31ba641c7b7bf27eb0dda4b9d", #2022
    "cpl::Football_Season::2f07c39671b84933ad7bb1e1958a7427", #2021
    "cpl::Football_Season::11aa5cc094d0481fa8e73d326763584f", #2020
    "cpl::Football_Season::c8c9bdc288f34aa89073a8bd89d2da3e", #2019
}

roles = ["all", "goalkeeper", "defender", "midfielder", "forward"]

def fetch_player(season_id, role, page=1):
    """Fetch a single page of stats for a season/role"""
    params = {
        "locale": "en-US",
        "category": "general",
        "role": role,
        "direction": "desc",
        "page": page,
        "pageNumElements": 250,
    }
    url = BASE_URL.format(season_id=season_id)
    r = requests.get(url, params=params, headers=HEADERS)
    r.raise_for_status()
    return r.json()

def fetch_all_players(season_id, role):
    """Loop through all pages until no players are returned"""
    players =[]
    page = 1
    while True:
        data = fetch_player(season_id, role, page)
        if not data["players"]:
            break
        players.extend(data["players"])
        page += 1
    return players

def flatten_player():
    """Flatten a player dict into a flat row of stats"""
    row = {
        "playerId": p["playerId"],
        "playerName": p.get("displayName") or p[("shortName")],
        "team": p["team"]["shortName"],
        "position": p["roleLabel"],
    }
    # Convert stats into columns keyed by abbreviation
    for stat in p["stats"]:
        abbr = stat["statsLabelAbbreviation"]
        if abbr:
            row[abbr] = stat["statsValue"]
    return row

for season_id in season_ids:
    for role in roles:
        players = fetch_all_players(season_id, role)
        flat = [flatten_player(p) for p in players]
        df = pd.DataFrame(flat)

        season_year = season_id.split("::")[-1][:4] # Extract year from season_id
        outfile = f"players_{season_year}_{role}.csv"
        df.to_csv(outfile, index=False)
        print(f"Saved {len(df)} rows to {outfile}")