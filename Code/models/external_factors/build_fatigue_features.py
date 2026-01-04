import pandas as pd
import numpy as np 
from math import radians, cos, sin, asin, sqrt
from pathlib import Path 
import os 
from cpl_stadiums import STADIUMS, TEAM_MAP
from fetch_weather import get_weather_estimate

cwd = Path(os.getcwd())
REPO_ROOT = cwd if cwd.name == "canpl-bet-3" else Path(__file__).resolve().parent.parent.parent.parent

# Schedule who played and when
MATCH_FILE = REPO_ROOT / "data" / "matches" / "processed" / "all_matches_with_baseline.csv"
OUT_FILE = REPO_ROOT / "data" / "matches" / "derived" / "external_factors.csv"

def haversine(lon1, lat1, lon2, lat2):
    # Calculate the circle distance in kilometers between two points on earth trig
    
    # convert degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 = lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # radius of the earth
    return c * r

def get_fatigue_score(days_rest, distance_traveled, timezone_change):
    # base fatigue start at 0
    # rest time whats short?
    # travel 1 point per x km's
    # jet lag
    fatigue = 0.0
    
    # 1 . Rest
    if days_rest <= 3:
        fatigue += 3.0
    elif days_rest == 4:
        fatigue += 1.5
    elif days_rest == 5:
        fatigue += 0.5
        
    # 2. Travel Penalty
    # 1000km = 0.5 fatigue points
    fatigue += (distance_traveled / 1000.0) * 0.5
    
    # 3. Jet Lag Penalty 
    fatigue += abs(timezone_change) * 0.5
    
    # 4. will need to add weather penalty
    
    return fatigue

def main():
print("--- CALCULATING EXTERNAL FACTORS (TRAVEL, WEATHER, SCORING) ---")
    
    if not MATCH_FILE.exists():
        print(f"Missing {MATCH_FILE}")
        return

    df = pd.read_csv(MATCH_FILE)
    df['date'] = pd.to_datetime(df['Date'])
    df['home_team'] = df['HomeTeam'].map(TEAM_MAP).fillna(df['HomeTeam'])
    df['away_team'] = df['AwayTeam'].map(TEAM_MAP).fillna(df['AwayTeam'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Track History
    last_played = {} 
    last_location = {} 
    
    # Track Goal History {team: [g1, g2, g3...]}
    team_goals_history = {}
    
    features = []
    
    print(f"   Processing {len(df)} matches...")

    for _, row in df.iterrows():
        mid = row['match_id']
        if pd.isna(mid):
            mid = f"{row['date'].strftime('%Y-%m-%d')}_{row['home_team']}_vs_{row['away_team']}"
            
        h_team = row['home_team']
        a_team = row['away_team']
        match_date = row['date']
        
        # 1. TRAVEL & FATIGUE
        h_stad_info = STADIUMS.get(h_team, STADIUMS.get("York"))
        current_loc_info = h_stad_info 
        
        # Home Calculation
        h_days_rest = 7
        h_dist = 0.0
        if h_team in last_played:
            h_days_rest = (match_date - last_played[h_team]).days
            prev_loc_name = last_location.get(h_team, h_team)
            prev_loc = STADIUMS.get(prev_loc_name, h_stad_info)
            h_dist = haversine(prev_loc['lon'], prev_loc['lat'], h_stad_info['lon'], h_stad_info['lat'])
        h_fatigue = get_fatigue_score(h_days_rest, h_dist)
        
        # Away Calculation
        a_days_rest = 7
        a_dist = 0.0
        if a_team in last_played:
            a_days_rest = (match_date - last_played[a_team]).days
            prev_loc_name = last_location.get(a_team, a_team)
            prev_loc = STADIUMS.get(prev_loc_name, STADIUMS.get("York"))
            a_dist = haversine(prev_loc['lon'], prev_loc['lat'], current_loc_info['lon'], current_loc_info['lat'])
        else:
            a_home_base = STADIUMS.get(a_team, STADIUMS.get("York"))
            a_dist = haversine(a_home_base['lon'], a_home_base['lat'], current_loc_info['lon'], current_loc_info['lat'])
        a_fatigue = get_fatigue_score(a_days_rest, a_dist)
        
        # 2. WEATHER (The Estimator)
        month = match_date.month
        avg_temp, rain_prob = get_weather_estimate(h_stad_info['name'], month)
        
        # 3. SCORING OFFENSE (Rolling Average of last 5 games)
        # We check history BEFORE today's game
        def get_avg_goals(team):
            hist = team_goals_history.get(team, [])
            if not hist: return 1.2 # League average default
            recent = hist[-5:] # Last 5 games
            return sum(recent) / len(recent)
            
        h_avg_goals = get_avg_goals(h_team)
        a_avg_goals = get_avg_goals(a_team)
        
        # 4. UPDATE HISTORY (For next loop)
        last_played[h_team] = match_date
        last_played[a_team] = match_date
        last_location[h_team] = h_team 
        last_location[a_team] = h_team
        
        # Add actual goals from TODAY to history (if played)
        if pd.notna(row['HomeScore']):
            if h_team not in team_goals_history: team_goals_history[h_team] = []
            if a_team not in team_goals_history: team_goals_history[a_team] = []
            team_goals_history[h_team].append(row['HomeScore'])
            team_goals_history[a_team].append(row['AwayScore'])

        # 5. SAVE
        features.append({
            'match_id': mid,
            'fatigue_home': h_fatigue,
            'fatigue_away': a_fatigue,
            'travel_km_away': round(a_dist, 1),
            'weather_temp': avg_temp,
            'weather_rain_prob': rain_prob,
            'avg_goals_home': round(h_avg_goals, 2),
            'avg_goals_away': round(a_avg_goals, 2),
            # THE INTERACTION FEATURE
            # High Rain * High Offense = High Impact
            'rain_impact_home': rain_prob * h_avg_goals,
            'rain_impact_away': rain_prob * a_avg_goals
        })
        
    out_df = pd.DataFrame(features)
    out_df.to_csv(OUT_FILE, index=False)
    print(f"Saved extended features to: {OUT_FILE}")
    print(out_df[['match_id', 'fatigue_away', 'weather_rain_prob', 'avg_goals_away', 'rain_impact_away']].tail())

if __name__ == "__main__":
    main()