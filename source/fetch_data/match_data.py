# @AUTHOUR: Jasper McLennan

import requests
import pandas as pd
import os

os.makedirs("data/raw/players", exist_ok=True)

# Season IDs mapped for extraction
seasons = {
    "2021": "cpl::Football_Season::2f07c39671b84933ad7bb1e1958a7427",
    "2022": "cpl::Football_Season::046f0ab31ba641c7b7bf27eb0dda4b9d",
    "2023": "cpl::Football_Season::fc0855108c9044218a84fc5d2bee0000",
    "2024": "cpl::Football_Season::6fb9e6fae4f24ce9bf4fa3172616a762",
    "2025": "cpl::Football_Season::fd43e1d61dfe4396a7356bc432de0007"
}

roles = ["goalkeeper", "defender", "midfielder", "forward"]
headers = {"User-Agent": "Mozilla/5.0"}

for year, season_id in seasons.items():
    print(f"Processing Player Stats for {year}...")
    all_players = []

    for role in roles:
        # We use a lower pageNumElement (50) to ensure the API doesn't block the request
        url = f"https://api-sdp.canpl.ca/v1/cpl/football/seasons/{season_id}/stats/players"
        params = {"role": role, "pageNumElement": 50, "page": 1}
        
        r = requests.get(url, params=params, headers=headers)
        r.raise_for_status()
        data = r.json()

        for p in data.get("players", []):
            row = {
                "Year": year,
                "Name": p.get("displayName"),
                "Team": p["team"].get("officialName"),
                "Position": p.get("roleLabel")
            }
            # Add all individual stats from the list
            for stat in p.get("stats", []):
                abbr = stat.get("statsLabelAbbreviation")
                if abbr:
                    row[abbr] = stat.get("statsValue")
            
            all_players.append(row)

    df = pd.DataFrame(all_players)
    output_path = f"data/raw/player/players_{year}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} players → {output_path}")