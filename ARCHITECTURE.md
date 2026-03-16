## CPL Prediction Project - Architecture & Setup Guide

### Project Purpose
This project uses machine learning and statistical models to predict Canadian Premier League (CPL) match outcomes and identify value bets against sports markets. It combines two predictive models:

1. **Poisson Model** - Statistical goal distribution model
2. **Machine Learning Model** - XGBoost trained on historical match features

---

## ✅ Recent Fixes Applied

### 1. **Created Missing Validation Script** 
   - **File**: `validate_team_strength.py`
   - **Purpose**: Validates the output from the match team strength calculation pipeline
   - **Checks**: Data integrity, required columns, score ranges, home/away balance

### 2. **Fixed Import Issues**
   - Updated relative imports to use proper path detection
   - All modules now consistently find the `REPO_ROOT` directory
   - Fixed imports in:
     - `Code/models/external_factors/build_fatigue_features.py`
     - `Code/models/external_factors/fetch_weather.py`
     - `Code/analysis/ensemble.py`

### 3. **Centralized Configuration**
   - **File**: `Code/config.py` - Now contains all shared constants
   - Moved team name mappings from 10 different files to one central location
   - Includes:
     - Team name normalization maps
     - Team ID mappings
     - ELO rating parameters
     - Feature engineering thresholds
   - All modules now import from this centralized config

---

## 📊 Data Pipeline Architecture

```
┌─ RAW DATA ──────────────────────────────────────────────────────────┐
│ • Match stats (scores, dates, teams)                                │
│ • Player stats (minutes played, performance metrics)                 │
│ • Team stats (goals, discipline, etc.)                              │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Data Collection Layer  │
                    │ (Code/get_data/)        │
                    │  • Web scraping         │
                    │  • CSV consolidation    │
                    │  • Data cleaning        │
                    └────────────┬────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
    ┌────▼─────┐         ┌──────▼──────┐       ┌───────▼───────┐
    │  Players │         │   Matches   │       │     Teams     │
    │ processed│         │  processed  │       │   processed   │
    └────┬─────┘         └──────┬──────┘       └───────┬───────┘
         │                      │                      │
         │        ┌─────────────┼──────────────┐      │
         │        │             │              │      │
    ┌────▼───┐   │      ┌──────▼──────────┐   │  ┌───▼────────┐
    │ Player │   │      │ Assumed Lineups │   │  │ Not Used   │
    │ Ratings│   │      │ (best 11/team)  │   │  └────────────┘
    │ (ELO)  │   │      └───────┬────────┘   │
    └────┬───┘   │              │            │
         │       │      ┌───────▼────────────▼──────┐
         │       │      │ Match Team Strength       │
         │       │      │ (aggregate player rating) │
         │       │      └───────┬──────────────────┘
         │       │              │
         │       │      ┌───────▼────────────┐
         │       │      │ Match Features     │
         │       │      │ (strength deltas)  │
         │       │      └───────┬────────────┘
         │       │              │
         │       │      ┌───────▼────────────────┐
         │       │      │ Rolling Features      │
         │       │      │ (5-match form trends) │
         │       │      └───────┬────────────────┘
         │       │              │
         │       └──────┬───────┘
         │              │
         │      ┌───────▼─────────────┐
         │      │ Training Targets    │
         │      │ (actual results)    │
         │      └───────┬─────────────┘
         │              │
         │      ┌───────▼────────────────────┐
         └─────►│ Model Training             │
                │ • ELO Probability Model    │
                │ • XGBoost ML Model         │
                │ • External Factors Model   │
                └───────┬────────────────────┘
                        │
                ┌───────▼──────────────┐
                │ Ensemble Predictions │
                │ (weighted average)   │
                └───────┬──────────────┘
                        │
                ┌───────▼──────────────┐
                │ Profitability Test   │
                │ (Kelly Criterion)    │
                └──────────────────────┘
```

---

## 🔄 Execution Pipeline (run_james_pipeline.py)

**Order matters!** Run in this sequence:

```
1. build_player_ratings_rolling.py
   └─ Calculates ELO ratings for each player based on match history
   
2. build_assumed_lineups.py
   └─ Determines which 11 players likely played in each match
   
3. build_match_team_strength.py
   └─ Aggregates player ratings to get team strength scores
   
4. build_match_features.py
   └─ Calculates differentials (home team strength - away team strength)
   
5. build_rolling_features.py
   └─ Calculates 5-match rolling averages for form tracking
   
6. build_targets.py
   └─ Extracts actual match outcomes (Win/Loss/Draw labels)
   
7. build_probability_model.py
   └─ Trains logistic regression on ELO features
   
8. validate_team_strength.py ✅ NEW
   └─ Validates data integrity before proceeding
```

---

## 📁 Directory Structure

```
canpl-bet/
├── Code/
│   ├── config.py __________________ ✅ Centralized configuration (NEW!)
│   ├── get_data/                   Data collection & cleaning
│   │   ├── cpl_match_stats.py
│   │   ├── cpl_player_stats.py
│   │   ├── cpl_team_stats.py
│   │   └── combine_tables.py
│   ├── models/                     Core feature engineering
│   │   ├── build_assumed_lineups.py
│   │   ├── build_match_features.py
│   │   ├── build_match_team_strength.py
│   │   ├── build_targets.py
│   │   ├── james_elo/             ELO rating system
│   │   │   ├── build_player_ratings_rolling.py
│   │   │   ├── build_probability_model.py
│   │   │   ├── build_rolling_features.py
│   │   │   └── tune_elo.py
│   │   ├── james_ml/              XGBoost ensemble
│   │   │   ├── train_xgboost.py
│   │   │   ├── predict_ml.py
│   │   │   └── train_final_model.py
│   │   └── external_factors/      Weather, travel, fatigue
│   │       ├── fetch_weather.py
│   │       ├── build_fatigue_features.py
│   │       ├── cpl_stadiums.py
│   │       └── train_external_model.py
│   ├── analysis/                  Model comparison & backtesting
│   │   ├── ensemble.py
│   │   ├── pre_match_odds_poisson.py
│   │   ├── pre_match_odds_ml.py
│   │   └── comparison/
│   └── pipeline/
│       └── pipeline.py
├── data/
│   ├── matches/
│   │   ├── raw/                   Scraped match data
│   │   ├── processed/             Cleaned match data
│   │   └── derived/               Feature-engineered data
│   ├── players/
│   │   ├── raw/
│   │   ├── cleaned/
│   │   └── derived/               Player ratings, statistics
│   └── teams/                     Team statistics
├── testing/
│   ├── naive_baseline.py           Data validation
│   └── evaluate_profitability.py   Backtest Kelly Criterion strategy
└── run_james_pipeline.py _________ Main pipeline orchestrator
```

---

## 🧠 How the Models Work

### **ELO Rating System**
- Rates individual players based on match outcomes
- After each match, winner's rating increases, loser's decreases
- Parameters: `K_FACTOR=20.0` (sensitivity), `HOME_ADVANTAGE=50.0` (rating boost)
- Aggregated into team strength scores

### **XGBoost Model**
- Trained on features including:
  - Team strength differential
  - Rolling form (last 5 matches)
  - Goals for/against trends
  - Home team advantage
- Outputs probability of Win/Loss/Draw

### **Ensemble**
- Combines Poisson statistical odds with ML probabilities
- Weighted average finds optimal balance
- `ensemble.py` optimizes weights using historical validation data

---

## ⚙️ Key Configuration Values

All defined in `Code/config.py`:

```python
# ELO Settings
ELO_K_FACTOR = 20.0              # How much ratings change per match
ELO_HOME_ADVANTAGE = 50.0        # Home team rating bonus
ELO_DEFAULT_RATING = 5.0         # Starting rating for new players

# Feature Engineering
ROLLING_WINDOW = 5               # Recent matches for form features
COVERAGE_THRESHOLD = 0.80        # Min team coverage %
```

---

## 🚀 Quick Start

### Run the full pipeline:
```bash
python3 run_james_pipeline.py
```

### Test a specific module:
```bash
cd Code/models
python3 build_match_features.py
```

### Check data integrity:
```bash
python3 validate_team_strength.py
```

---

## 🔍 Data Files to Monitor

- **Player Ratings**: `data/players/derived/player_ratings_rolling.csv`
  - Tracks ELO ratings chronologically
  - Used to compute team strength per match

- **Match Features**: `data/matches/derived/match_model_with_form.csv`
  - Input to ML models
  - Contains strength differential + form features

- **Predictions**: `data/matches/derived/2026_season_predictions.csv`
  - Final ensemble predictions
  - Used for betting strategy

---

## ❓ Common Issues & Solutions

### Import Errors
- **Problem**: `ModuleNotFoundError: cannot import ...`
- **Solution**: All imports now use centralized `config.py`. Make sure `Code/` is in Python path
- **Fixed**: `Code/models/external_factors/` modules now properly detect `REPO_ROOT`

### Missing Files
- **Problem**: Pipeline fails at `validate_team_strength.py` step
- **Solution**: ✅ File now exists and validates output
- **Fixed**: Created `/validate_team_strength.py`

### Team Name Inconsistencies
- **Problem**: Same team referenced as "HFX Wanderers", "Wanderers", "Halifax", etc.
- **Solution**: ✅ Centralized `TEAM_NAME_MAP` in `config.py` normalizes all variants
- **Fixed**: 10 identical mappings consolidated to single source

---

## 📈 Next Steps for Optimization

1. **Add logging** - Track execution flow and catch silent errors
2. **Error handling** - Add try/except in scrapers to retry failed requests
3. **Configuration** - Make K-factors and thresholds adjustable per season
4. **Testing** - Add unit tests for feature engineering logic
5. **Documentation** - Add docstrings explaining what each pipeline step does

---

## Need Help?

- **Understanding a module?** Check the docstrings at the top of each Python file
- **Debugging?** Run `validate_team_strength.py` to check data integrity
- **Changing parameters?** Update values in `Code/config.py`, not in individual files
- **Adding a feature?** Follow the pipeline order and add data to `data/matches/derived/`

---

*Last Updated: March 16, 2026*
*Architecture Fixes: Import paths standardized, centralized config deployed, validation added*
