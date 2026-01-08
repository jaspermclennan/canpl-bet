import pandas as pd
import sys

# ----------------------------
# 1. Pipeline Integration
# ----------------------------
# Check if the pipeline passed a TARGET_YEAR, otherwise default to 2026
target_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026

# ----------------------------
# Paths
# ----------------------------
input_csv = 'data/teams/combined/teams_combined.csv'
output_csv = f'data/analysis/correlations_{target_year}.csv'
# ----------------------------
# Load Data
# ----------------------------
df_teams_all = pd.read_csv(input_csv)

# ----------------------------
# THE LEAKAGE FIX: Filter out the Target Year
# ----------------------------
# We only calculate correlations based on data BEFORE the target year.
# This ensures our weights are purely historical and not "peeking" at the future.
train_df = df_teams_all[df_teams_all['Year'] < target_year].copy()

print(f"--- Calculating Correlations using data from years: {sorted(train_df['Year'].unique())} ---")

# ----------------------------
# Define "Leakage" stats to exclude
# ----------------------------
exclude_stats = [
    'Total Points', 'Total Wins', 'Total Losses', 'Total Draws', 
    'Points', 'Wins', 'Losses', 'Draws', 
    'Games Played', 'Games Played', 'Year', "Goals Against", "Goals Conceded"
]

# ----------------------------
# Analyze Predictive Stats
# ----------------------------
# 1. Calculate correlations based ONLY on the training data
all_corrs = train_df.corr(numeric_only=True)['Total Points']

# 2. Drop the NaNs (removes stats with missing year data)
predictive_corrs = all_corrs.dropna()

# 3. Drop the "leakage" labels
predictive_corrs = predictive_corrs.drop(labels=exclude_stats, errors='ignore').sort_values(ascending=False)

# ----------------------------
# Formatting for Output
# ----------------------------
# Convert to DataFrame and reset index so 'Stat_Name' is a column
df_output = predictive_corrs.reset_index()
df_output.columns = ['Stat_Name', 'Total Points']

print("\n--- TOP 10 POSITIVE PREDICTIVE PERFORMANCE STATS ---")
print(df_output.head(10))


print("\n--- TOP 10 NEGATIVE PREDICTIVE PERFORMANCE STATS ---")
print(df_output.tail(10))

# ----------------------------
# Save the Clean List
# ----------------------------
# Saved as a standard CSV (comma separated) for better compatibility
df_output.to_csv(output_csv, index=False)
print(f"\nCleaned correlation list saved to: {output_csv}")