import pandas as pd

# ----------------------------
# Paths
# ----------------------------
input_csv = 'data/teams/combined/teams_combined.csv'
output_csv = 'data/teams/combined/teams_zscores.csv'

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(input_csv)

# ----------------------------
# Stats selected for z-scores
# ----------------------------
selected_stats = [



    # ATTACK & FINISHING
    "Goals",
    "Goals Openplay",
    "Goals from Inside Box",
    "Home Goals",
    "Away Goals",
    "Goal Assists",
    "Key Passes (Attempt Assists)",
    "Total Shots",
    "Shots On Target ( inc goals )",
    "Goal Conversion",

    # POSSESSION & BUILD-UP
    "Possession Percentage",
    "Total Passes",
    "Open Play Passes",
    "Successful Open Play Passes",
    "Total Successful Passes ( Excl Crosses & Corners )",
    "Successful Passes Opposition Half",
    "Passing Accuracy",
    "Passing % Opp Half",

    # CROSSING / PRESSURE
    "Corners Won",
    "Successful Crosses open play",

    # DEFENSIVE PERFORMANCE
    "Clean Sheets",
    "Blocked Shots",
    "Number of Defensive Actions",
    "Duels won",
    "Shots On Conceded",
    "Shots On Conceded Inside Box",
    "Goals Conceded Inside Box",
    "Goals Conceded",
    "Total losses",

    # DISCIPLINE / ERRORS
    "Red cards",
    "Total Red Cards",
    "Straight Red Cards",
    "Penalty conceded",
    "Penalty Goals Conceded",
    "Own Goals Conceded",

    # GOALKEEPER
    "GK Successful Distribution",
    "GK Unsuccessful Distribution",
]

# Keep only stats that exist in dataframe
selected_stats = [col for col in selected_stats if col in df.columns]

# ----------------------------
# Compute z-scores BY SEASON
# ----------------------------
zscore_frames = []

for year, season_df in df.groupby("Year"):
    season_z = season_df.copy()

    for col in selected_stats:
        mean = season_df[col].mean()
        std = season_df[col].std()

        season_z[col + " z score"] = (
            (season_df[col] - mean) / std if std != 0 else 0
        )

    zscore_frames.append(season_z)

# Combine seasons
df_zscores = pd.concat(zscore_frames, ignore_index=True)

# ----------------------------
# Keep ONLY identifiers + points + z-scores
# ----------------------------
id_cols = [
    "Team",
    "Year",
    "Games played",
    "Total points"
]

zscore_cols = [col for col in df_zscores.columns if col.endswith(" z score")]

final_cols = [col for col in id_cols if col in df_zscores.columns] + zscore_cols

df_final = df_zscores[final_cols]

# ----------------------------
# Sort by Team, then Year
# ----------------------------
df_final = df_final.sort_values(
    by=["Team", "Year"]
).reset_index(drop=True)

# ----------------------------
# Save output
# ----------------------------
df_final.to_csv(output_csv, index=False)
print(f"Clean z-score table saved to: {output_csv}")
