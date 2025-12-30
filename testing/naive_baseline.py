from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "matches" / "raw"
OUT_DIR = BASE_DIR / "data" / "matches" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "all_matches_with_baseline.csv"


def result_hda(home: int, away: int) -> str:
    if home > away:
        return "H"
    if home < away:
        return "A"
    return "D"


def main() -> None:
    files = sorted(RAW_DIR.glob("matches_*.csv"))
    if not files:
        raise FileNotFoundError(f"No files found in {RAW_DIR} matching matches_*.csv")

    dfs = []
    required = {"Season", "Date", "HomeTeam", "AwayTeam", "HomeScore", "AwayScore"}
    for f in files:
        df = pd.read_csv(f)

        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"{f.name} missing columns: {sorted(missing)}")

        dfs.append(df)

    matches = pd.concat(dfs, ignore_index=True)

    # filter finished matches
    if "Status" in matches.columns:
        matches = matches[matches["Status"] == "FINISHED"].copy()

    # parse/sort date
    matches["Date"] = pd.to_datetime(matches["Date"], utc=True, errors="coerce")
    matches = matches.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # create result column
    matches["Result"] = matches.apply(
        lambda r: result_hda(int(r["HomeScore"]), int(r["AwayScore"])),
        axis=1,
    )

    # naive probabilities 
    freqs = matches["Result"].value_counts(normalize=True)
    pH = float(freqs.get("H", 0.0))
    pD = float(freqs.get("D", 0.0))
    pA = float(freqs.get("A", 0.0))

    matches["NaiveProbHome"] = pH
    matches["NaiveProbDraw"] = pD
    matches["NaiveProbAway"] = pA

    matches.to_csv(OUT_FILE, index=False)

    print(f"Loaded {len(files)} files, {len(matches)} finished matches")
    print(f"Naive probs: H={pH:.3f}, D={pD:.3f}, A={pA:.3f}")
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
