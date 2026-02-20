#Author: Jasper McLennan

import pandas as pd
import glob
import os



###################### FOR MATCHES ##########################
match_files = glob.glob('data/raw/match/matches_*.csv') 
if match_files:
    dfs_matches = [pd.read_csv(f) for f in match_files]
    df_matches_all = pd.concat(dfs_matches, ignore_index=True)
    
    # Optional: Clean match columns as well if needed
    df_matches_all.columns = df_matches_all.columns.str.strip().str.title()
    
    df_matches_all.to_csv('data/raw/match/combined/matches_combined.csv', index=False)
    print(f"Successfully combined {len(match_files)} match files.")