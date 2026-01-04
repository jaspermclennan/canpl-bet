import os
import time
import json
import requests
import pandas as pd

BASE = "https://api.oddspapi.io/v4"
API_KEY = os.environ["ODDSPAPI_KEY"]

FIXTURES_CSV = "betting_odds/canpl_fixtures_2022_2025_plus_upcoming.csv"
OUT_JSONL = "betting_odds/bet365_hist_raw.jsonl"
OUT_FLAT = "betting_odds/bet365_hist_flat.csv"

def get_json(path, params=None, min_delay_s=0.0):
    if min_delay_s:
        time.sleep(min_delay_s)

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
            time.sleep((retry_ms / 1000) + 0.05)
            continue

        r.raise_for_status()
        return r.json()

def flatten_hist(hist_json: dict):
    fixture_id = hist_json.get("fixtureId")
    rows = []

    bookmakers = hist_json.get("bookmakers") or {}
    for bm_slug, bm_data in bookmakers.items():
        markets = (bm_data or {}).get("markets") or {}
        for market_id, market_data in markets.items():
            outcomes = (market_data or {}).get("outcomes") or {}
            for outcome_id, outcome_data in outcomes.items():
                players = (outcome_data or {}).get("players") or {}
                for player_key, entries in players.items():
                    if not isinstance(entries, list):
                        continue
                    for e in entries:
                        rows.append({
                            "fixtureId": fixture_id,
                            "bookmaker": bm_slug,
                            "marketId": market_id,
                            "outcomeId": outcome_id,
                            "playerKey": str(player_key),
                            "createdAt": e.get("createdAt"),
                            "price": e.get("price"),
                            "limit": e.get("limit"),
                            "active": e.get("active"),
                        })
    return rows

def main():
    fx = pd.read_csv(FIXTURES_CSV)
    fixture_ids = fx["fixtureId"].dropna().astype(str).tolist()

    all_rows = []
    with open(OUT_JSONL, "w", encoding="utf-8") as fraw:
        for i, fixture_id in enumerate(fixture_ids, 1):
            try:
                # Historical endpoint has ~5000ms cooldown; be conservative
                hist = get_json("/historical-odds", {
                    "fixtureId": fixture_id,
                    "bookmakers": "bet365",
                    "language": "en",
                }, min_delay_s=5.2)

                fraw.write(json.dumps(hist) + "\n")
                all_rows.extend(flatten_hist(hist))

                print(f"[{i}/{len(fixture_ids)}] OK fixtureId={fixture_id} rows={len(all_rows)}")

            except requests.HTTPError as e:
                print(f"[{i}/{len(fixture_ids)}] FAIL fixtureId={fixture_id} err={e}")
                continue

    pd.DataFrame(all_rows).to_csv(OUT_FLAT, index=False)
    print(f"Wrote flat odds to {OUT_FLAT}")

if __name__ == "__main__":
    main()
