import requests
import pandas as pd

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
season_ids = {
    "cpl::Football_Season::fd43e1d61dfe4396a7356bc432de0007", #2025
    "cpl::Football_Season::6fb9e6fae4f24ce9bf4fa3172616a762", #2024
    "cpl::Football_Season::fc0855108c9044218a84fc5d2bee0000", #2023
    "cpl::Football_Season::046f0ab31ba641c7b7bf27eb0dda4b9d", #2022
    "cpl::Football_Season::2f07c39671b84933ad7bb1e1958a7427", #2021
    "cpl::Football_Season::11aa5cc094d0481fa8e73d326763584f", #2020
    "cpl::Football_Season::c8c9bdc288f34aa89073a8bd89d2da3e", #2019
}

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

for season_id in season_ids:
    for role in roles:
        try:
            players = fetch_all_players(season_id, role)
        except requests.exceptions.HTTPError as e:
            print(f"Error fetching {role} for season {season_id}: {e}")
            continue
        flat = [flatten_player(p) for p in players]
        df = pd.DataFrame(flat)
        # Extract a year from the season ID if you need it
        season_year = season_id.split("::")[-1][:4]
        outfile = f"players_{season_year}_{role}.csv"
        df.to_csv(outfile, index=False)
        print(f"Saved {len(df)} rows to {outfile}")