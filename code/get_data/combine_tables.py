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

    # 1. Standardize Case for all columns immediately
    df_teams_all.columns = df_teams_all.columns.str.strip().str.title()

    # 2. FRANCHISE MERGER: Do this BEFORE deduplicating columns
    df_teams_all['Team'] = df_teams_all['Team'].replace({
        'York9 FC': 'Inter Toronto',
        'York United': 'Inter Toronto'
    })

# 3. DEDUPLICATE COLUMNS SAFELY
    # Standardize column names to Title Case first
    df_teams_all.columns = df_teams_all.columns.str.strip().str.title()
    
    # Identify duplicate column names and keep only the first occurrence
    # This avoids the "Unalignable boolean Series" error
    df_teams_all = df_teams_all.loc[:, ~df_teams_all.columns.duplicated()]

    # 4. SORT & RESET
    # Double check that essential columns exist before sorting
    if 'Team' in df_teams_all.columns and 'Year' in df_teams_all.columns:
        df_teams_all = df_teams_all.sort_values(by=['Year', 'Team'], ascending=[True, True])
    else:
        print("Warning: 'Team' or 'Year' column missing after deduplication!")
        
    df_teams_all = df_teams_all.reset_index(drop=True)

    # Save the master table
    df_teams_all.to_csv('data/teams/combined/teams_combined.csv', index=False)
    print(f"Successfully cleaned {len(team_files)} files. York/Inter Toronto merged.")
    print(f"Final Columns Check: {list(df_teams_all.columns[:5])}...")