# scripts/ingest_odds.py
import os, json, datetime as dt
from pathlib import Path

import requests
from dotenv import load_dotenv
load_dotenv()

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("ODDS_API_KEY", "")

def save_snapshot(name: str, data):
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = RAW_DIR / f"{name}_{stamp}.json"
    out.write_text(json.dumps(data, indent=2))
    print("Saved", out)

def mock_odds():
    # Minimal mock map keyed by event id (must match match ids for joining)
    return [
        {"idEvent": "M1", "home_team": "Pacific FC", "away_team": "Cavalry FC",
         "home_moneyline": -120, "away_moneyline": +110},
        {"idEvent": "M2", "home_team": "York United", "away_team": "Forge FC",
         "home_moneyline": +180, "away_moneyline": -200},
    ]

def main():
    if not API_KEY:
        print("No ODDS_API_KEY set — writing MOCK odds so you can proceed.")
        save_snapshot("odds", mock_odds())
        return

    sport_key = "soccer_canada_cpl"
    url = f"https://api.the-odds-api.com/v4/sports/{bff3e48471ad905cf0ec9edcea7a121f}/odds"
    params = dict(regions="us,eu", markets="h2h", oddsFormat="american", apiKey=API_KEY)
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    save_snapshot("odds", resp.json())

if __name__ == "__main__":
    main()
