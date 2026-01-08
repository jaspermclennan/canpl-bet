import requests
import pandas as pd
import os

os.makedirs("data/matches", exist_ok=True)

seasons = {
    
    "2021": "cpl::Football_Season::2f07c39671b84933ad7bb1e1958a7427",
    "2022": "cpl::Football_Season::046f0ab31ba641c7b7bf27eb0dda4b9d",
    "2023": "cpl::Football_Season::fc0855108c9044218a84fc5d2bee0000",
    "2024": "cpl::Football_Season::6fb9e6fae4f24ce9bf4fa3172616a762",
    "2025": "cpl::Football_Season::fd43e1d61dfe4396a7356bc432de0007"
}

headers = {"User-Agent": "Mozilla/5.0"}

for year, season_id in seasons.items():
    print(f"Processing {year}...")

    url = f"https://api-sdp.canpl.ca/v1/cpl/football/seasons/{season_id}/matches"
    r = requests.get(url, headers=headers)
    r.raise_for_status()

    rows = []
    for m in r.json().get("matches", []):
        rows.append({
            "Season": year,
            "Date": m.get("matchDateUtc"),
            "HomeTeam": m.get("home", {}).get("officialName"),
            "AwayTeam": m.get("away", {}).get("officialName"),
            "HomeScore": m.get("providerHomeScore"),
            "AwayScore": m.get("providerAwayScore"),
            "Status": m.get("status"),
            "Venue": m.get("stadiumName")
        })

    df = pd.DataFrame(rows)

    output_path = f"data/matches/matches_{year}.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} matches → {output_path}")
