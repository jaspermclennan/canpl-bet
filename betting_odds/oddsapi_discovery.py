import os
import requests

BASE = "https://api.oddspapi.io/v4"
API_KEY = os.environ.get("ODDSPAPI_KEY")

import time
import json

def get_json(path, params=None):
    params = dict(params or {})
    params["apiKey"] = API_KEY

    while True:
        r = requests.get(f"{BASE}{path}", params=params, timeout=30)
        print("GET", r.url)

        if r.status_code == 429:
            try:
                payload = r.json()
                retry_ms = int(payload["error"].get("retryMs", 1000))
            except Exception:
                retry_ms = 1000
            time.sleep((retry_ms / 1000) + 0.05)
            continue

        if r.status_code != 200:
            print("Status:", r.status_code)
            print("Response:", r.text[:500])

        r.raise_for_status()
        return r.json()



def main():
    # 1) Find soccer sportId
    sports = get_json("/sports", {"language": "en"})
    soccer = [s for s in sports if s.get("slug") == "soccer"]
    print("\nSOCCER matches:", soccer[:3])
    if not soccer:
        raise RuntimeError("Could not find soccer in /sports response.")
    soccer_id = soccer[0]["sportId"]
    print("soccer sportId =", soccer_id)

    tournaments = get_json("/tournaments", {"sportId": soccer_id, "language": "en"})

    # 1) All Canada tournaments
    canada = [t for t in tournaments if (t.get("categoryName") or "").strip().lower() == "canada"]
    print("\nALL TOURNAMENTS IN CATEGORY: Canada")
    for t in canada:
        print(
            "tournamentId=", t.get("tournamentId"),
            "| tournamentName=", t.get("tournamentName"),
            "| categoryName=", t.get("categoryName"),
            "| slug=", t.get("tournamentSlug"),
        )

    tournaments = get_json("/tournaments", {"sportId": soccer_id, "language": "en"})

    # 1) All Canada tournaments
    canada = [t for t in tournaments if (t.get("categoryName") or "").strip().lower() == "canada"]
    print("\nALL TOURNAMENTS IN CATEGORY: Canada")
    for t in canada:
        print(
            "tournamentId=", t.get("tournamentId"),
            "| tournamentName=", t.get("tournamentName"),
            "| categoryName=", t.get("categoryName"),
            "| slug=", t.get("tournamentSlug"),
        )

    # 2) Search for CanPL / CPL patterns across everything
    terms = ["canpl", "cpl", "canadian premier", "canadian-premier", "premier league canada", "canada premier"]
    print("\nSEARCH RESULTS FOR CanPL TERMS:")
    for term in terms:
        hits = []
        for t in tournaments:
            blob = " ".join([
                str(t.get("tournamentName") or ""),
                str(t.get("tournamentSlug") or ""),
                str(t.get("categoryName") or ""),
            ]).lower()
            if term in blob:
                hits.append(t)
        print(f"\nTerm: {term}  | hits: {len(hits)}")
        for t in hits[:30]:
            print(
                "tournamentId=", t.get("tournamentId"),
                "| tournamentName=", t.get("tournamentName"),
                "| categoryName=", t.get("categoryName"),
                "| slug=", t.get("tournamentSlug"),
            )



    # Print anything that looks like Canada / Premier League
    keywords = ["canada", "canadian", "premier"]
    hits = []
    for t in tournaments:
        name = (t.get("tournamentName") or "").lower()
        cat  = (t.get("categoryName") or "").lower()
        slug = (t.get("tournamentSlug") or "").lower()
        if any(k in name or k in cat or k in slug for k in keywords):
            hits.append(t)

    print("\nTOURNAMENT HITS (Canada/Premier):")
    for t in hits[:50]:
        print(
            "tournamentId=", t.get("tournamentId"),
            "| tournamentName=", t.get("tournamentName"),
            "| categoryName=", t.get("categoryName"),
            "| slug=", t.get("tournamentSlug"),
        )

    # 3) Confirm bookmaker slug for bet365 exists
    bookmakers = get_json("/bookmakers", {"language": "en"})
    bet365 = [b for b in bookmakers if b.get("slug") == "bet365"]
    print("\nBET365 bookmaker entry:", bet365[:1])

if __name__ == "__main__":
    main()
