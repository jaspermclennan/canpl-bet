import pandas as pd
import glob
import csv

# 1. Find all  match files (e.g., matches_2022.csv, matches_2023.csv, etc.)
# If they are in a folder called 'data', use 'data/matches_*.csv'
match_files = glob.glob('data/matches/matches_*.csv') 

# 2. Load them into a list of DataFrames
# This line creates a list where each item is one year's worth of matches
dfs = [pd.read_csv(f) for f in match_files]

# 3. Stack them into one big table
# ignore_index=True gives you a fresh set of row numbers from 0 to the total
df_matches_all = pd.concat(dfs, ignore_index=True)

# 4. Save your new Master file so you don't have to do this again
df_matches_all.to_csv('data/matches/combined/matches_combined.csv', index=False)