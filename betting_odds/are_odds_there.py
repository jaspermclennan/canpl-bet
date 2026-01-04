import os
import time
import requests
import pandas as pd

BASE = "https://api.oddspapi.io/v4"
API_KEY = os.environ.get("ODDSPAPI_KEY")

FIXTURES_CSV = "betting_odds/canpl_fixtures_2025.csv"

def get_json(path, params=None):
    params = dict(params or {})
    params["apiKey"] = API_KEY
    url = f"{BASE}{path}"
    r = requests.get(url, params=params, timeout=30)
    print("GET", r.url)
    print("Status:", r.status_code)
    if r.status_code != 200:
        print("Body:", (r.text or "")[:500])
    r.raise_for_status()
    return r.json()

def main():
    print("Starting odds smoke test...")
    print("API key present:", bool(API_KEY))
    print("Reading CSV:", FIXTURES_CSV)

    df = pd.read_csv(FIXTURES_CSV)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if "fixtureId" not in df.columns:
        raise KeyError("fixtureId column not found in fixtures CSV")

    sample = df["fixtureId"].dropna().astype(str).head(3).tolist()
    print("Sample fixtureIds:", sample)

    if not sample:
        print("No fixtureIds found to test. Exiting.")
        return

    for fid in sample:
        print("\nFixtureId:", fid)
        data = get_json("/fixtures", {
            "tournamentId": 28432,
            "hasOdds": "true",
            "language": "en",
        })
        print("fixtures with odds:", len(data))
        if data:
            print("sample:", data[0]["fixtureId"], data[0].get("startTime")) 
            bms = data.get("bookmakerOdds") or {}
            print("hasOdds:", data.get("hasOdds"))
            bms = data.get("bookmakerOdds") or {}
            print("bookmakerOdds keys:", list(bms.keys()))

        time.sleep(2.1)

if __name__ == "__main__":
    main()
