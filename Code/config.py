"""
Centralized configuration for the CPL prediction pipeline.
This module contains all shared constants, mappings, and configuration values.
"""

# ============================================================================
# SEASON MAPPINGS
# ============================================================================

SEASON_ID_TO_YEAR = {
    "cpl::Football_Season::c8c9bdc288f34aa89073a8bd89d2da3e": 2019,
    "cpl::Football_Season::11aa5cc094d0481fa8e73d326763584f": 2020,
    "cpl::Football_Season::2f07c39671b84933ad7bb1e1958a7427": 2021,
    "cpl::Football_Season::046f0ab31ba641c7b7bf27eb0dda4b9d": 2022,
    "cpl::Football_Season::fc0855108c9044218a84fc5d2bee0000": 2023,
    "cpl::Football_Season::6fb9e6fae4f24ce9bf4fa3172616a762": 2024,
    "cpl::Football_Season::fd43e1d61dfe4396a7356bc432de0007": 2025,
}

# ============================================================================
# TEAM NAME MAPPINGS
# ============================================================================
# Maps various team name formats to their canonical short form
TEAM_NAME_MAP = {
    "HFX Wanderers": "Wanderers",
    "Halifax Wanderers": "Wanderers",
    "HFX Wanderers FC": "Wanderers",
    "York United": "York",
    "York United FC": "York",
    "Atlético Ottawa": "Atlético",
    "Atletico Ottawa": "Atlético",
    "Pacific": "Pacific",
    "Pacific FC": "Pacific",
    "Valour": "Valour",
    "Valour FC": "Valour",
    "Forge": "Forge",
    "Forge FC": "Forge",
    "Cavalry": "Cavalry",
    "Cavalry FC": "Cavalry",
    "Edmonton": "Edmonton",
    "FC Edmonton": "Edmonton",
    "Inter Toronto": "Inter Toronto",
    "Inter Toronto FC": "Inter Toronto",
    "Vancouver FC": "Vancouver",
    "Vancouver": "Vancouver",
    "FC Supra du Québec": "FC Supra du Québec",
    "Supra du Québec": "FC Supra du Québec",
}

# Lowercase normalized mapping for fuzzy matching
TEAM_NAME_MAP_LOWERCASE = {
    "hfx wanderers": "wanderers",
    "halifax wanderers": "wanderers",
    "york united": "york",
    "atletico ottawa": "atletico",
    "atlético ottawa": "atletico",
    "forge fc": "forge",
    "cavalry fc": "cavalry",
    "pacific fc": "pacific",
    "valour fc": "valour",
    "vancouver fc": "vancouver",
    "fc edmonton": "edmonton",
    "edmonton fc": "edmonton",
    "inter toronto": "inter toronto",
    "inter toronto fc": "inter toronto",
    "fc supra du québec": "fc supra du quebec",
    "supra du québec": "fc supra du quebec",
}

# Numeric team IDs (used in APIs and some analyses)
TEAM_ID_MAP = {
    1: 'Cavalry',
    2: 'Forge',
    3: 'Atlético Ottawa',
    4: 'HFX Wanderers',
    5: 'Inter Toronto',
    6: 'Pacific',
    7: 'Vancouver FC',
    8: 'FC Supra du Québec'
}

# Reverse mapping: canonical name to ID
TEAM_NAME_TO_ID = {v: k for k, v in TEAM_ID_MAP.items()}

# ============================================================================
# LEAGUE ROSTER BY YEAR
# ============================================================================
# Which teams were active in each season
LEAGUE_HISTORY = {
    2021: ['Forge', 'Cavalry', 'Atlético', 'Wanderers', 'Inter Toronto', 'Pacific', 'Valour', 'Edmonton'],
    2022: ['Forge', 'Cavalry', 'Atlético', 'Wanderers', 'Inter Toronto', 'Pacific', 'Valour', 'Edmonton'],
    2023: ['Forge', 'Cavalry', 'Atlético', 'Wanderers', 'Inter Toronto', 'Pacific', 'Valour', 'Vancouver'],
    2024: ['Forge', 'Cavalry', 'Atlético', 'Wanderers', 'Inter Toronto', 'Pacific', 'Valour', 'Vancouver'],
    2025: ['Forge', 'Cavalry', 'Atlético', 'Wanderers', 'Inter Toronto', 'Pacific', 'Valour', 'Vancouver'],
    2026: ['Forge', 'Cavalry', 'Atlético', 'Wanderers', 'Inter Toronto', 'Pacific', 'Vancouver', 'FC Supra du Québec']
}

# ============================================================================
# ELO RATING CONFIGURATION
# ============================================================================
ELO_K_FACTOR = 20.0  # Sensitivity of rating adjustments
ELO_HOME_ADVANTAGE = 50.0  # Home team rating bonus
ELO_DEFAULT_RATING = 5.0  # Default rating for new players/teams

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================
ROLLING_WINDOW = 5  # Number of recent matches for form features
COVERAGE_THRESHOLD = 0.80  # Minimum team coverage threshold (%)
MIN_LINEUP_COVERAGE = 0.70  # Minimum player minutes coverage for matches