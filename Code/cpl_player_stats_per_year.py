import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

PLAYERS_RAW_DIR = DATA_DIR / "players" / "raw"
PLAYERS_CLEAN_DIR = DATA_DIR / "players" / "cleaned"
PLAYERS_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = PLAYERS_RAW_DIR / "cpl_players_combined.csv"

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Make sure your scraper writes to data/players/raw/ "
            f"or update INPUT_CSV."
        )

    df = pd.read_csv(INPUT_CSV)

    numeric_cols = [
        "GP", "Mins", "G", "A", "S", "GI", "SOT", "KP",
        "Tackles", "FC", "FS", "OFF", "YC", "RC",
        "Pass%", "GAA", "%"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    gp_nonzero = df["GP"].replace(0, pd.NA)

    df["G_per_game"] = (df["G"] / gp_nonzero).fillna(0)
    df["A_per_game"] = (df["A"] / gp_nonzero).fillna(0)
    df["S_per_game"] = (df["S"] / gp_nonzero).fillna(0)
    df["SOT_per_game"] = (df["SOT"] / gp_nonzero).fillna(0)
    df["KP_per_game"] = (df["KP"] / gp_nonzero).fillna(0)
    df["Minutes_per_game"] = (df["Mins"] / gp_nonzero).fillna(0)

    if "playerName" in df.columns and "season" in df.columns:
        df = df.sort_values(by=["playerName", "season"], ascending=[True, True])
    elif "playerName" in df.columns:
        df = df.sort_values(by="playerName")

    out_all = PLAYERS_CLEAN_DIR / "cpl_players_all_seasons_cleaned.csv"
    df.to_csv(out_all, index=False)
    print(f"Saved cleaned player file: {out_all} ({len(df)} rows)")

    if "season" not in df.columns:
        raise KeyError("Column 'season' not found in the dataframe.")

    for season, group in df.groupby("season"):
        out_name = PLAYERS_CLEAN_DIR / f"cpl_players_{season}.csv"
        group.to_csv(out_name, index=False)
        print(f"Saved {out_name} with {len(group)} rows")

if __name__ == "__main__":
    main()
