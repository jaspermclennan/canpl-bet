import os
import time
import json
import requests
import pandas as pd

BASE = "https://api.oddspapi.io/v4"
API_KEY = os.environ["ODDSPAPI_KEY"]

FIXTURES_CSV = "betting_odds/canpl_fixtures_2025.csv"
OUT_JSONL = "betting_odds/canpl_2025_bet365_hist_raw.jsonl"
OUT_FLAT = "betting_odds/canpl_2025_bet365_hist_flat.csv"

HIST_COOLDOWN_S = 5.2  # historical odds endpoint cooldown (be conservative)

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
            time.sleep((retry_ms / 1000) + 0.10)
            continue

        if r.status_code == 404:
            # no odds found for that fixture/bookmaker
            return None

        r.raise_for_status()
        return r.json()

def already_downloaded_fixture_ids(jsonl_path: str) -> set[str]:
    done = set()
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    fid = obj.get("fixtureId")
                    if fid:
                        done.add(str(fid))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return done

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
    df_fx = pd.read_csv(FIXTURES_CSV)
    fixture_ids = df_fx["fixtureId"].dropna().astype(str).tolist()

    done = already_downloaded_fixture_ids(OUT_JSONL)
    remaining = [fid for fid in fixture_ids if fid not in done]

    # QUOTA GUARD: adjust this number to stay within 250 calls
    # Example: cap at 180 to keep headroom for retries and other requests
    MAX_FIXTURES_TO_FETCH = 180
    remaining = remaining[:MAX_FIXTURES_TO_FETCH]

    print(f"Total fixtures in file: {len(fixture_ids)}")
    print(f"Already downloaded:     {len(done)}")
    print(f"Fetching now:           {len(remaining)} (capped)")

    all_rows = []
    with open(OUT_JSONL, "a", encoding="utf-8") as fraw:
        for i, fixture_id in enumerate(remaining, 1):
            hist = get_json(
                "/historical-odds",
                {"fixtureId": fixture_id, "bookmakers": "bet365", "language": "en"},
                min_delay_s=HIST_COOLDOWN_S,
            )

            if hist is None:
                print(f"[{i}/{len(remaining)}] No odds for fixtureId={fixture_id}")
                continue

            fraw.write(json.dumps(hist) + "\n")
            all_rows.extend(flatten_hist(hist))
            print(f"[{i}/{len(remaining)}] OK fixtureId={fixture_id} new_rows={len(all_rows)}")

    if all_rows:
        out_df = pd.DataFrame(all_rows)
        # append-friendly: if file exists, merge after (simple approach: overwrite from combined raw later)
        out_df.to_csv(OUT_FLAT, mode="a", header=not os.path.exists(OUT_FLAT), index=False)
        print(f"Wrote/updated {OUT_FLAT}")

if __name__ == "__main__":
    main()
