import pandas as pd
import glob
import os

# Ensure the combined folders exist
os.makedirs('data/matches/combined', exist_ok=True)
os.makedirs('data/teams/combined', exist_ok=True)

###################### FOR MATCHES ##########################
match_files = glob.glob('data/matches/matches_*.csv') 
if match_files:
    dfs_matches = [pd.read_csv(f) for f in match_files]
    df_matches_all = pd.concat(dfs_matches, ignore_index=True)
    
    # Optional: Clean match columns as well if needed
    df_matches_all.columns = df_matches_all.columns.str.strip().str.title()
    
    df_matches_all.to_csv('data/matches/combined/matches_combined.csv', index=False)
    print(f"Successfully combined {len(match_files)} match files.")

###################### FOR TEAM STATS ##########################
team_files = glob.glob('data/teams/teams_*.csv') 
if team_files:
    dfs_teams = [pd.read_csv(f) for f in team_files]
    df_teams_all = pd.concat(dfs_teams, ignore_index=True)

    # --- THE CLEANING FIX ---
    # 1. Standardize Case: Merges 'Yellow cards' and 'Yellow Cards'
    df_teams_all.columns = df_teams_all.columns.str.strip().str.title()

    # 2. Merge Duplicates: Groups same-named columns and picks the first valid stat
    # This prevents 'Yellow Cards' from appearing twice in your correlations
# .T flips the table so we can group without using the deprecated axis=1
    df_teams_all = df_teams_all.T.groupby(level=0).first().T
    # 3. Sort & Reset: Keep it alphabetical by Team and chronological by Year
    df_teams_all = df_teams_all.sort_values(by=['Team', 'Year'])
    df_teams_all = df_teams_all.reset_index(drop=True)

    # Save the sorted, cleaned master table
    df_teams_all.to_csv('data/teams/combined/teams_combined.csv', index=False)
    print(f"Successfully cleaned and combined {len(team_files)} team files.")