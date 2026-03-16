import sys
from pathlib import Path

# Add Code directory to path for config import
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "Code"))
from config import TEAM_NAME_MAP

STADIUMS = {
    "Pacific":   {"name": "Starlight Stadium",        "lat": 48.4429, "lon": -123.5042, "tz": "Pacific"},
    "Vancouver": {"name": "Willoughby Park",          "lat": 49.1440, "lon": -122.6656, "tz": "Pacific"},
    "Cavalry":   {"name": "ATCO Field",               "lat": 50.8850, "lon": -114.1006, "tz": "Mountain"},
    "Valour":    {"name": "Princess Auto Stadium",    "lat": 49.8078, "lon":  -97.1431, "tz": "Central"},
    "Forge":     {"name": "Tim Hortons Field",        "lat": 43.2524, "lon":  -79.8302, "tz": "Eastern"},
    "York":      {"name": "York Lions Stadium",       "lat": 43.7747, "lon":  -79.5068, "tz": "Eastern"},
    "Atlético":  {"name": "TD Place",                 "lat": 45.3982, "lon":  -75.6836, "tz": "Eastern"},
    "Wanderers": {"name": "Wanderers Grounds",        "lat": 44.6444, "lon":  -63.5836, "tz": "Atlantic"},
    "Edmonton":  {"name": "Clarke Stadium",           "lat": 53.5574, "lon": -113.4764, "tz": "Mountain"},
}

# Use the centralized team name map from config
TEAM_MAP = TEAM_NAME_MAP
