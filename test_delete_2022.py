import pandas as pd
from pathlib import Path
import os

# Path Setup - using the Current Working Directory (cwd)
PROJECT_ROOT = Path(os.getcwd())
LINEUP_FILE = PROJECT_ROOT / "data" / "lineups" / "assumed_lineup.csv"
PLAYER_BASE = PROJECT_ROOT / "data" / "players" / "cleaned" / "cpl_players_all_seasons_cleaned.csv"

# The missing 2022 Edmonton Roster 
# (You can expand this list if you find more names)
EDMONTON_2022_PATCH = [
    "T. Warschewski", "M. Hernandez", "A. Koch", "N. Akio", 
    "S. Shome", "T. Timoteo", "K. Porter", "G. Darling",
    "C. Fayia", "L. Singh", "M. Krutzen", "K. Mansaray",
    "S. Triantafillou", "O. Ada", "C. Smith", "A. González"
]

def main():
    print(f"--- Data Gap Repair Started ---")
    print(f"Checking for: {LINEUP_FILE}")

    if not LINEUP_FILE.exists():
        print(f"❌ ERROR: Still can't find {LINEUP_FILE}")
        print(f"I am currently looking in: {os.getcwd()}")
        return

    # 1. Load the Player Base
    if not PLAYER_BASE.exists():
        print(f"❌ ERROR: Cannot find Player Base at {PLAYER_BASE}")
        return

    players = pd.read_csv(PLAYER_BASE)
    
    # 2. Check current Edmonton 2022 count
    existing_edmonton = players[(players["season"] == 2022) & (players["team"] == "Edmonton")]["playerName"].tolist()
    print(f"Current players for Edmonton 2022 in DB: {len(existing_edmonton)}")
    
    # 3. Create the patch rows
    new_rows = []
    for name in EDMONTON_2022_PATCH:
        if name not in existing_edmonton:
            new_rows.append({
                "playerName": name,
                "team": "Edmonton",
                "season": 2022,
                "Minutes": 1000, # Baseline to ensure they are picked for lineups
                "role": "midfielder" # Generic placeholder
            })
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Ensure 'season' is int
        new_df["season"] = new_df["season"].astype(int)
        
        updated_players = pd.concat([players, new_df], ignore_index=True)
        updated_players.to_csv(PLAYER_BASE, index=False)
        
        print(f"✅ SUCCESS: Added {len(new_rows)} missing players to {PLAYER_BASE.name}")
        print("\n--- NEXT STEP ---")
        print("Rerun: python3 Code/models/build_assumed_lineups.py")
    else:
        print("No new players needed to be added.")

if __name__ == "__main__":
    main()