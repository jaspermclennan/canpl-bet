# scripts/ingest_matches.py
import os, json, datetime as dt
from pathlib import Path

import requests
from dotenv import load_dotenv
load_dotenv()

# ---- CONFIG ----
# Canadian Premier League season;
LEAGUE_ID = "4820"   # using TheSportsDB later
SEASON = "2025/26"

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

KEY = os.getenv("THESPORTSDB_KEY", "")

def save_snapshot(name: str, data):
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = RAW_DIR / f"{name}_{stamp}.json"
    out.write_text(json.dumps(data, indent=2))
    print("Saved", out)

def mock_matches():
    # Minimal mock of two matches with scores
    return {
        "events": [
            {
                "idEvent": "M1",
                "dateEvent": "2025-05-10",
                "strHomeTeam": "Pacific FC",
                "strAwayTeam": "Cavalry FC",
                "intHomeScore": "2",
                "intAwayScore": "1",
            },
            {
                "idEvent": "M2",
                "dateEvent": "2025-05-17",
                "strHomeTeam": "York United",
                "strAwayTeam": "Forge FC",
                "intHomeScore": "0",
                "intAwayScore": "1",
            },
        ]
    }

def main():
    if not KEY:
        print("No THESPORTSDB_KEY set — writing MOCK matches so you can proceed.")
        save_snapshot(f"matches_{SEASON}", mock_matches())
        return

    url = f"https://www.thesportsdb.com/api/v1/json/{bff3e48471ad905cf0ec9edcea7a121f}/eventsseason.php"
    resp = requests.get(url, params={"id": LEAGUE_ID, "s": SEASON}, timeout=30)
    resp.raise_for_status()
    save_snapshot(f"matches_{SEASON}", resp.json())
    save_snapshot(f"matches_{SEASON}", mock_matches())

if __name__ == "__main__":
    main()
