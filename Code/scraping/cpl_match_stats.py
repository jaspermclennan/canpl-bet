import requests
import csv
import os

# 1. Setup folders and Season IDs
if not os.path.exists('data'):
    os.makedirs('data')

SEASONS = {
    "2022": "cpl::Football_Season::046f0ab31ba641c7b7bf27eb0dda4b9d",
    "2023": "cpl::Football_Season::fc0855108c9044218a84fc5d2bee0000",
    "2024": "cpl::Football_Season::6fb9e6fae4f24ce9bf4fa3172616a762",
    "2025": "cpl::Football_Season::fd43e1d61dfe4396a7356bc432de0007"
}

# 2. Define headers in CSV
fields = ["Season", "Date", "HomeTeam", "AwayTeam", "HomeScore", "AwayScore", "Status", "Venue"]

for year, s_id in SEASONS.items():
    print(f"Processing {year}...")
    url = f"https://api-sdp.canpl.ca/v1/cpl/football/seasons/{s_id}/matches"
    
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    
    if response.status_code == 200:
        data = response.json()
        matches = data.get("matches", [])
        
        filename = f"data/matches/matches_{year}.csv"
        
        # 3. Use csv.DictWriter to handle the data row-by-row
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            
            for m in matches:
                # Manually 'flatten' the nested JSON structure
                row = {
                    "Season": year,
                    "Date": m.get("matchDateUtc"),
                    "HomeTeam": m.get("home", {}).get("officialName"),
                    "AwayTeam": m.get("away", {}).get("officialName"),
                    "HomeScore": m.get("providerHomeScore"),
                    "AwayScore": m.get("providerAwayScore"),
                    "Status": m.get("status"),
                    "Venue": m.get("stadiumName")
                }
                writer.writerow(row)
                
        print(f"Saved {len(matches)} matches to {filename}")
    else:
        print(f"Error {response.status_code} for season {year}")