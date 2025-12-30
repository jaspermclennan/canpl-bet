import pandas as pd
import sys

# ----------------------------
# 1. Load Data
# ----------------------------
df_teams = pd.read_csv('data/teams/combined/teams_combined.csv')

# Use the weights we just generated in correlations.py 
# (which were calculated using data BEFORE the target year)
df_corr = pd.read_csv('data/analysis/correlations.csv')

# ----------------------------
# 2. Selection: Find Significant Stats
# ----------------------------
threshold = 0.25
significant_df = df_corr[df_corr['Total Points'].abs() > threshold]
significant_stats = significant_df['Stat_Name'].tolist()

# Clean list of stats to process
exclude_cols = ['Total Points', 'Games played', 'Year', 'Team']
significant_stats = [s for s in significant_stats if s not in exclude_cols]

# ----------------------------
# 3. Calculate Z-Scores & Weighted Strength for ALL Years
# ----------------------------
zscore_frames = []

# Loop through every year in the combined table
for year, season_df in df_teams.groupby("Year"):
    season_z = season_df[['Team', 'Year', 'Total Points']].copy()
    weighted_columns = []
    
    for stat in significant_stats:
        if stat in season_df.columns:
            mean = season_df[stat].mean()
            std = season_df[stat].std()
            
            # Standardize (Z-Score)
            z_val = (season_df[stat] - mean) / std if std > 0 else 0
            
            # Apply Weight (Lookup from our correlations file)
            # This uses the weights that were generated historically
            try:
                weight = df_corr.loc[df_corr['Stat_Name'] == stat, 'Total Points'].values[0]
                col_name = f"{stat}_w"
                season_z[col_name] = z_val * weight
                weighted_columns.append(col_name)
            except IndexError:
                continue # Skip if stat wasn't in our correlation file

    # Sum all weighted stats into the Strength Score
    season_z['Strength_Score'] = season_z[weighted_columns].mean(axis=1) * 100
    
    # Keep only the essential columns to keep the file clean
    season_z = season_z[['Team', 'Year', 'Total Points', 'Strength_Score']]
    zscore_frames.append(season_z)

# ----------------------------
# 4. Save Master Team Profiles
# ----------------------------
df_strength = pd.concat(zscore_frames, ignore_index=True)

# Sort by Year and then Strength
df_strength = df_strength.sort_values(by=['Year', 'Strength_Score'], ascending=[False, False])

# Save to the original master path so your next script finds it
output_path = 'data/analysis/team_strengths.csv'
df_strength.to_csv(output_path, index=False)

print("\n--- FINAL TEAM STRENGTH RANKINGS (ALL YEARS) ---")
print(df_strength.head(50))
print(f"\nSaved master table to: {output_path}")