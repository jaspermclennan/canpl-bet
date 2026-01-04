import os
import time
import requests
import pandas as pd

BASE = "https://api.oddspapi.io/v4"
API_KEY = os.environ["ODDSPAPI_KEY"]

CANPL_TOURNAMENT_ID = 28432
FIXTURES_COOLDOWN_S = 2.05  # fixtures endpoint cooldown

def get_json(path, params=None):
    params = dict(params or {})
    params["apiKey"] = API_KEY

    while True:
        r = requests.get(f"{BASE}{path}", params=params, timeout=30)

        if r.status_code == 429:
            try:
                payload = r.json()
                retry_ms = int(payload["error"].get("retryMs", 1000))
            except Exception:
                retry_ms = 1000
            time.sleep((retry_ms / 1000) + 0.10)
            continue

        # OddsPapi sometimes uses 404 for "no fixtures found"
        if r.status_code == 404:
            return []

        r.raise_for_status()
        return r.json()

def fetch_fixtures_2025(status_id: int):
    # Use season window to keep results tight:
    # regular season: Apr 5 to mid-Oct; playoffs start Oct 22
    # We'll query Apr 1 to Nov 15 to capture everything.
    params = {
        "tournamentId": CANPL_TOURNAMENT_ID,
        "from": "2025-04-01T00:00:00Z",
        "to":   "2025-11-15T00:00:00Z",
        "statusId": status_id,  # 0 scheduled, 2 finished
        "language": "en",
    }
    data = get_json("/fixtures", params)
    time.sleep(FIXTURES_COOLDOWN_S)
    return data

def normalize(fixtures, label=""):
    if not fixtures:
        print(f"normalize(): no fixtures for {label}")
        return pd.DataFrame(columns=[
            "fixtureId","startTime","statusId","participant1Name","participant2Name","hasOdds","seasonId"
        ])

    # Inspect first record keys once (helps if field names differ)
    print(f"normalize(): {label} sample keys:", list(fixtures[0].keys())[:20])

    rows = []
    for f in fixtures:
        rows.append({
            # try both common keys defensively
            "fixtureId": f.get("fixtureId") or f.get("id"),
            "startTime": f.get("startTime") or f.get("start_time") or f.get("date"),
            "statusId": f.get("statusId") or f.get("status_id"),
            "participant1Name": f.get("participant1Name") or (f.get("homeTeam") or {}).get("name"),
            "participant2Name": f.get("participant2Name") or (f.get("awayTeam") or {}).get("name"),
            "hasOdds": f.get("hasOdds"),
            "seasonId": f.get("seasonId"),
        })

    df = pd.DataFrame(rows)

    # If fixtureId is still missing, show columns and stop early
    if "fixtureId" not in df.columns:
        print("normalize(): columns:", df.columns.tolist())
        raise KeyError("fixtureId column not present after normalization")

    df = df.dropna(subset=["fixtureId"])
    df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce", utc=True)
    return df


def main():
    print("Fetching 2025 finished fixtures...")
    finished = fetch_fixtures_2025(status_id=2)
    print("finished count:", len(finished))

    print("Fetching 2025 upcoming fixtures...")
    upcoming = fetch_fixtures_2025(status_id=0)
    print("upcoming count:", len(upcoming))

    df_finished = normalize(finished, "finished")
    df_upcoming = normalize(upcoming, "upcoming")

    frames = [df_finished]
    if not df_upcoming.empty:
        frames.append(df_upcoming)

    df = pd.concat(frames, ignore_index=True)


    df = df.drop_duplicates(subset=["fixtureId"]).sort_values("startTime")

    out_csv = "betting_odds/canpl_fixtures_2025.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} fixtures -> {out_csv}")

    print(df[["startTime","participant1Name","participant2Name","statusId","hasOdds"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
