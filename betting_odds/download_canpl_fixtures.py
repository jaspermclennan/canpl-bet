import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

BASE = "https://api.oddspapi.io/v4"
API_KEY = os.environ["ODDSPAPI_KEY"]

SOCCER_SPORT_ID = 10
CANPL_TOURNAMENT_ID = 28432  # Canadian Premier League

FIXTURES_COOLDOWN_S = 2.05  # docs say 2000ms cooldown :contentReference[oaicite:2]{index=2}

def get_json(path, params=None):
    params = dict(params or {})
    params["apiKey"] = API_KEY

    while True:
        r = requests.get(f"{BASE}{path}", params=params, timeout=30)

        # Rate limit handling
        if r.status_code == 429:
            try:
                payload = r.json()
                retry_ms = int(payload["error"].get("retryMs", 1000))
            except Exception:
                retry_ms = 1000
            time.sleep((retry_ms / 1000) + 0.10)
            continue

        # IMPORTANT: print body on errors so we can see what's going on
        if r.status_code != 200:
            print("GET", r.url)
            print("Status:", r.status_code)
            print("Response:", (r.text or "")[:500])

            # Some APIs use 404 to mean "no data"
            if r.status_code == 404:
                return []

        r.raise_for_status()
        return r.json()

def fetch_fixtures_window(from_iso: str, to_iso: str, status_id: int | None):
    params = {
        "tournamentId": CANPL_TOURNAMENT_ID,
        "sportId": SOCCER_SPORT_ID,
        "from": from_iso,
        "to": to_iso,
        "language": "en",
    }
    if status_id is not None:
        params["statusId"] = status_id  # 0 not started, 1 live, 2 finished, 3 cancelled :contentReference[oaicite:3]{index=3}
    data = get_json("/fixtures", params)
    time.sleep(FIXTURES_COOLDOWN_S)  # endpoint cooldown :contentReference[oaicite:4]{index=4}
    return data

def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch_season_year_finished(year: int):
    # CanPL typically runs spring–fall; querying only season months cuts requests.
    start = datetime(year, 3, 1, tzinfo=timezone.utc)
    end   = datetime(year, 12, 1, tzinfo=timezone.utc)

    out = []
    cur = start
    step = timedelta(days=7)  # weekly chunks
    while cur < end:
        nxt = min(cur + step, end)
        out.extend(fetch_fixtures_window(iso(cur), iso(nxt), status_id=2))
        cur = nxt
    return out

def fetch_upcoming(days_ahead: int = 180):
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days_ahead)
    return fetch_fixtures_window(iso(now), iso(end), status_id=0)

def main():
    all_fx = []

    # Finished fixtures 2022–2025
    for year in [2022, 2023, 2024, 2025]:
        print(f"\nFetching finished fixtures for {year}...")
        year_fx = fetch_season_year_finished(year)
        print(f"  got {len(year_fx)} rows")
        all_fx.extend(year_fx)

    # Upcoming fixtures
    print("\nFetching upcoming fixtures...")
    up = fetch_upcoming(180)
    print(f"  got {len(up)} rows")
    all_fx.extend(up)

    # Normalize
    rows = []
    for f in all_fx:
        rows.append({
            "fixtureId": f.get("fixtureId"),
            "startTime": f.get("startTime"),
            "participant1Id": f.get("participant1Id"),
            "participant2Id": f.get("participant2Id"),
            "participant1Name": f.get("participant1Name"),
            "participant2Name": f.get("participant2Name"),
            "statusId": f.get("statusId"),
            "tournamentId": f.get("tournamentId"),
            "seasonId": f.get("seasonId"),
            "hasOdds": f.get("hasOdds"),
        })

    df = pd.DataFrame(rows).dropna(subset=["fixtureId"]).drop_duplicates(subset=["fixtureId"])
    df = df.sort_values("startTime")

    out_csv = "betting_odds/canpl_fixtures_2022_2025_plus_upcoming.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} fixtures to {out_csv}")

if __name__ == "__main__":
    main()
