import requests
import pandas as pd
import csv


url = "https://api-sdp.canpl.ca/v1/cpl/football/seasons/cpl::Football_Season::fd43e1d61dfe4396a7356bc432de0007/stats/teams"

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

response = requests.get(url, params=params, headers=headers)
data = response.json()

# --- Flatten team stats ---
rows = []
for team_entry in data["teams"]:
    team_info = team_entry.get("team", {})
    team_name = team_entry.get("officialName", "Unknown Team")

    team_dict = {"Team": team_name}
    for s in team_entry["stats"]:
        team_dict[s["statsLabel"]] = s["statsValue"]

    rows.append(team_dict)

df = pd.DataFrame(rows)
df.to_csv("data/2025_canpl_team_stats.csv", index=False)

print(f"Saved {len(df)} teams to canpl_team_stats.csv")
print(df.head())