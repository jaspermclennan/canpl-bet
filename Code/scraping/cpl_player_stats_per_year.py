import pandas as pd
from pathlib import Path

# Directory setup
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

PLAYERS_RAW_DIR = DATA_DIR / "players" / "raw"
PLAYERS_CLEAN_DIR = DATA_DIR / "players" / "cleaned"
PLAYERS_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = PLAYERS_RAW_DIR / "cpl_players_combined.csv"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Make sure your scraper writes to data/players/raw/"
        )

    df = pd.read_csv(INPUT_CSV)

    # 1. Dynamic Rename Logic for Percentages
    # This prevents the generic "%" key from accidentally renaming the wrong column.
    dynamic_renames = {}
    for col in df.columns:
        col_clean = col.lower().strip()
        if "pass" in col_clean and "%" in col_clean:
            dynamic_renames[col] = "PassPct"
        elif "save" in col_clean and "%" in col_clean:
            dynamic_renames[col] = "SavePct"
        elif col == "%": # Fallback for exactly "%"
            dynamic_renames[col] = "SavePct"

    # 2. Standard Column Mapping
    standard_renames = {
        "YC": "YellowCards",
        "RC": "RedCards",
        "FC": "FoulsCommitted",
        "FS": "FoulsSuffered",
        "OFF": "Offsides",
        "KP": "KeyPasses",
        "SOT": "ShotsOnTarget",
        "GI": "GoalInvolvements",
        "Mins": "Minutes",
        "S": "Shots",
        "G": "Goals",
        "A": "Assists",
        "GP": "GamesPlayed",
    }
    
    # Combine both maps
    final_rename_map = {**standard_renames, **dynamic_renames}
    df = df.rename(columns={k: v for k, v in final_rename_map.items() if k in df.columns})

    # List of all numeric columns we expect to have
    numeric_cols = [
        "GamesPlayed", "Minutes", "Goals", "Assists", "Shots", "GoalInvolvements",
        "ShotsOnTarget", "KeyPasses", "Tackles", "FoulsCommitted", "FoulsSuffered", 
        "Offsides", "YellowCards", "RedCards", "PassPct", "GAA", "SavePct",
        "G_per_game", "A_per_game", "S_per_game", "SOT_per_game", "KP_per_game", 
        "Minutes_per_game",
    ]

    # Ensure numeric types
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Calculate per-game stats
    if "GamesPlayed" in df.columns:
        # Replace 0 with NA to avoid division by zero errors
        gp_nonzero = df["GamesPlayed"].replace(0, pd.NA)
    else:
        raise KeyError("Expected 'GP'/'GamesPlayed' column not found.")

    def per_game(out_col, base_col):
        if base_col in df.columns:
            df[out_col] = (df[base_col] / gp_nonzero).fillna(0)

    per_game("G_per_game", "Goals")
    per_game("A_per_game", "Assists")
    per_game("S_per_game", "Shots")
    per_game("SOT_per_game", "ShotsOnTarget")
    per_game("KP_per_game", "KeyPasses")
    per_game("Minutes_per_game", "Minutes")

    # Sorting
    sort_cols = [c for c in ["playerName", "season", "team", "position"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=True)

    # Reorder Columns
    id_cols_front = [c for c in ["playerName", "season", "team", "position", "role"] if c in df.columns]
    player_id_last = ["playerId"] if "playerId" in df.columns else []
    numeric_cols_present = [c for c in numeric_cols if c in df.columns]
    
    covered = set(id_cols_front + numeric_cols_present + player_id_last)
    other_cols = [c for c in df.columns if c not in covered]

    df = df[id_cols_front + numeric_cols_present + other_cols + player_id_last]

    # Save full dataset
    out_all = PLAYERS_CLEAN_DIR / "cpl_players_all_seasons.csv"
    df.to_csv(out_all, index=False)
    print(f"Saved cleaned player file: {out_all} ({len(df)} rows)")

    # Save individual seasons
    if "season" not in df.columns:
        raise KeyError("Column 'season' not found in the dataframe.")

    for season, group in df.groupby("season"):
        out_name = PLAYERS_CLEAN_DIR / f"cpl_players_{season}.csv"
        group.to_csv(out_name, index=False)
        print(f"Saved {out_name} with {len(group)} rows")


if __name__ == "__main__":
    main()