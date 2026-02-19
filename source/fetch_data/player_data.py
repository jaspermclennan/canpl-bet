# @AUTHOR: James Rankin

import requests
import pandas as pd
from pathlib import Path
import time  # for optional sleep if needed



BASE_URL = "https://api-sdp.canpl.ca/v1/cpl/football/seasons/{season_id}/stats/players"

HEADERS = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Origin": "https://www.canpl.ca",
    "Referer": "https://www.canpl.ca/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
}

# Year → season_id mapping (like your team script)
seasons = {
    2019: "cpl::Football_Season::c8c9bdc288f34aa89073a8bd89d2da3e",
    2020: "cpl::Football_Season::11aa5cc094d0481fa8e73d326763584f",
    2021: "cpl::Football_Season::2f07c39671b84933ad7bb1e1958a7427",
    2022: "cpl::Football_Season::046f0ab31ba641c7b7bf27eb0dda4b9d",
    2023: "cpl::Football_Season::fc0855108c9044218a84fc5d2bee0000",
    2024: "cpl::Football_Season::6fb9e6fae4f24ce9bf4fa3172616a762",
    2025: "cpl::Football_Season::fd43e1d61dfe4396a7356bc432de0007",
}

roles = ["goalkeeper", "defender", "midfielder", "forward"]

def fetch_players_page(season_id, role, page=1, timeout=15):
    params = {
        "locale": "en-US",
        "category": "general",
        "role": role,
        "direction": "desc",
        "page": page,
        "pageNumElement": 100,  # lowered from 250 – API seems to ignore large values anyway
    }
    url = BASE_URL.format(season_id=season_id)
    try:
        res = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"  Request failed (page {page}, role {role}): {e}")
        return {"players": []}  # treat as end

def fetch_all_players(season_id, role):
    players = []
    page = 1
    max_pages = 50  # safety limit to prevent true infinite loop
    while page <= max_pages:
        print(f"    Fetching {role} - page {page}...", end="", flush=True)
        data = fetch_players_page(season_id, role, page)
        current_players = data.get("players", [])
        if not current_players:
            print(" done (no more players)")
            break
        players.extend(current_players)
        print(f" got {len(current_players)}")
        page += 1
        time.sleep(0.5)  # polite delay to avoid hammering API
    if page > max_pages:
        print(f"  WARNING: Hit safety limit ({max_pages} pages) for {role} – possible API issue")
    return players

# Main loop – like your team script
all_rows = []  # optional: keep combined if you want

for year, season_id in seasons.items():
    print(f"\nProcessing season {year} ({season_id})...")
    
    season_rows = []
    
    for role in roles:
        print(f"  Role: {role}")
        players = fetch_all_players(season_id, role)
        print(f"    Found {len(players)} {role}s")
        
        for p in players:
            row = {
                "playerId": p.get("playerId"),
                "playerName": p.get("displayName") or p.get("shortName", "Unknown"),
                "team": p.get("team", {}).get("shortName", "Unknown"),
                "position": p.get("roleLabel", "Unknown"),
                "season": year,
                "role": role,
            }
            for stat in p.get("stats", []):
                abbr = stat.get("statsLabelAbbreviation")
                if abbr and abbr != "%":
                    row[abbr] = stat.get("statsValue")
                else:
                    label = stat.get("statsLabel")
                    if label:
                        col_name = ''.join(word.capitalize() for word in label.lower().split())
                        col_name = col_name.replace("%", "Pct").replace(" ", "")
                        row[col_name] = stat.get("statsValue")
            season_rows.append(row)
            all_rows.append(row)  
    
    if season_rows:
        df_season = pd.DataFrame(season_rows)
        output_path = f"data/raw/player/players_{year}.csv"
        df_season.to_csv(output_path, index=False)
        print(f"Saved {len(df_season)} players for {year} → {output_path}")
    else:
        print(f"No players found for {year}")

# Optional: one combined file
if all_rows:
    df_all = pd.DataFrame(all_rows)
    combined_output_path = f"data/raw/player/cpl_players_combined.csv"
    df_all.to_csv(combined_output_path, index=False)
    print(f"\nSaved combined file with {len(df_all)} rows → {combined_output_path}")