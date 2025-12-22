import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

INPUT_CSV = DATA_DIR / "cpl_players_combined.csv"

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Make sure to run cpl_player_stats.py first."
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

    df["G_per_game"] = df["G"] / gp_nonzero
    df["A_per_game"] = df["A"] / gp_nonzero
    df["S_per_game"] = df["S"] / gp_nonzero
    df["SOT_per_game"] = df["SOT"] / gp_nonzero
    df["KP_per_game"] = df["KP"] / gp_nonzero
    df["Minutes_per_game"] = df["Mins"] / gp_nonzero

    per_game_cols = [
        "G_per_game", "A_per_game", "S_per_game",
        "SOT_per_game", "KP_per_game", "Minutes_per_game"
    ]
    df[per_game_cols] = df[per_game_cols].fillna(0)

    out_all = DATA_DIR / "cpl_players_all_seasons_cleaned.csv"
    df.to_csv(out_all, index=False)
    print(f"Saved cleaned player file: {out_all} ({len(df)} rows)")

    if "season" not in df.columns:
        raise KeyError("Column 'season' not found in the dataframe.")

    for season, group in df.groupby("season"):
        season_str = str(season)
        out_name = DATA_DIR / f"cpl_players_{season_str}.csv"
        group.to_csv(out_name, index=False)
        print(f"Saved {out_name} with {len(group)} rows")

if __name__ == "__main__":
    main()
