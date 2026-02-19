import pandas as pd

# 1. Load Data
matches = pd.read_csv('data/matches/combined/matches_combined.csv')
teams = pd.read_csv('data/analysis/team_strengths.csv') 

# 2. Cleanup & Standardize
matches.columns = matches.columns.str.strip().str.title()
teams.columns = teams.columns.str.strip().str.title()

# Define Result (0: Side A Win, 1: Draw, 2: Side B Win)
def determine_result(row):
    if row['Homescore'] > row['Awayscore']: return 0
    if row['Homescore'] == row['Awayscore']: return 1
    return 2

matches['Result'] = matches.apply(determine_result, axis=1)

# 3. MERGE PRIORS
# Merge Home Priors
df_merged = pd.merge(matches, teams[['Year', 'Team', 'Strength_Score']], 
                     left_on=['Season', 'Hometeam'], right_on=['Year', 'Team'], how='left')
df_merged = df_merged.rename(columns={'Strength_Score': 'Home_Prior'}).drop(['Team', 'Year'], axis=1)

# Merge Away Priors
df_merged = pd.merge(df_merged, teams[['Year', 'Team', 'Strength_Score']], 
                     left_on=['Season', 'Awayteam'], right_on=['Year', 'Team'], how='left')
df_merged = df_merged.rename(columns={'Strength_Score': 'Away_Prior'}).drop(['Team', 'Year'], axis=1)

# ---------------------------------------------------------
# 4. NEUTRALIZATION (THE MIRROR STEP)
# ---------------------------------------------------------
# Drop any rows with missing strengths before mirroring
df_merged = df_merged.dropna(subset=['Home_Prior', 'Away_Prior'])

# Side A is Home, Side B is Away
side_a_original = df_merged.copy()

# Side A is Away, Side B is Home (The Mirror)
side_b_mirrored = df_merged.copy()
side_b_mirrored['Home_Prior'] = df_merged['Away_Prior']
side_b_mirrored['Away_Prior'] = df_merged['Home_Prior']

# Flip the results for the mirror: 
# If Original was Home Win (0), Mirror is now an Away Win (2)
# If Original was Away Win (2), Mirror is now a Home Win (0)
# Draw (1) stays a Draw (1)
side_b_mirrored['Result'] = side_b_mirrored['Result'].replace({0: 2, 2: 0})

# Combine them into one giant neutral dataset
neutral_training_set = pd.concat([side_a_original, side_b_mirrored], ignore_index=True)
# ---------------------------------------------------------

# 5. Save the final training set
neutral_training_set.to_csv('data/analysis/ml_training_priors.csv', index=False)

print(f"Original Matches: {len(df_merged)}")
print(f"Neutralized Training Rows: {len(neutral_training_set)}")
print("ML Training Set saved to data/analysis/ml_training_priors.csv")