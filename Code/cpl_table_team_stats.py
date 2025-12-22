import requests
import pandas as pd
import os

BASE_URL = "https://api-sdp.canpl.ca/v1/cpl/football/seasons/{season_id}/stats/teams"

params = {
    "locale": "en-US",
    "category": "general",
    "orderBy": "goals",
    "direction": "desc",
    "pageNumElement": "30"
}

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

# Add more seasons
seasons = {
    "2022": "cpl::Football_Season::046f0ab31ba641c7b7bf27eb0dda4b9d",
    "2023": "cpl::Football_Season::fc0855108c9044218a84fc5d2bee0000",
    "2024": "cpl::Football_Season::6fb9e6fae4f24ce9bf4fa3172616a762",
    "2025": "cpl::Football_Season::fd43e1d61dfe4396a7356bc432de0007"
}

os.makedirs("data/tables", exist_ok=True)


for year, season_id in seasons.items():
    url = BASE_URL.format(season_id=season_id)
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()

    print(f"Processing {year}...")

    rows = []
    for team_entry in data["teams"]:
        team_name = team_entry.get("officialName", "Unknown Team")
        team_dict = {
            "Year": year,
            "Team": team_name
        }

        for s in team_entry["stats"]:
            team_dict[s["statsLabel"]] = s["statsValue"]

        rows.append(team_dict)

    df = pd.DataFrame(rows)
    output_path = f"data/table_team_stats/{year}_table_team_stats.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} teams for {year} → {output_path}")
