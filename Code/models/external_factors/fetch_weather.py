# Not using a paid API
import pandas as pd
import numpy as np
from pathlib import Path
import os 
from cpl_stadiums import STADIUMS

CLIMATE_DATA = {
"Pacific": {  # Victoria (Rainy winters, mild summers)
        4: (10, 0.40), 5: (14, 0.25), 6: (17, 0.15), 7: (20, 0.05), 8: (20, 0.05), 9: (17, 0.20), 10: (12, 0.45)
    },
    "Mountain": { # Calgary/Edmonton (Cold start/end, mild dry summer)
        4: (8, 0.30), 5: (14, 0.35), 6: (18, 0.40), 7: (23, 0.30), 8: (22, 0.25), 9: (16, 0.20), 10: (9, 0.15)
    },
    "Central": {  # Winnipeg (Hot summer)
        4: (6, 0.25), 5: (15, 0.30), 6: (21, 0.35), 7: (26, 0.30), 8: (25, 0.25), 9: (18, 0.25), 10: (8, 0.20)
    },
    "Eastern": {  # Toronto/Hamilton/Ottawa (Humid, standard)
        4: (9, 0.40), 5: (16, 0.35), 6: (22, 0.30), 7: (26, 0.25), 8: (25, 0.25), 9: (20, 0.30), 10: (13, 0.35)
    },
    "Atlantic": { # Halifax (Wet, windy)
        4: (7, 0.50), 5: (12, 0.45), 6: (17, 0.40), 7: (22, 0.30), 8: (22, 0.30), 9: (18, 0.40), 10: (12, 0.50)
    }
}

def get_weather_estimate(stadium_name, month):
    # return estimated temp and rain/snow probability
    region = "Eastern"
    for team, data in STATIUMS.items():
        if data['name'] == stadium_name:
            region = data['tz']
            break
        
    # look up monthly average
    month = max(4, min(10, month))
    
    if region in CLIMATE_DATA:
        return CLIMATE_DATA[region].get(month, (15, 0.25))
    return (15, 0.25)

if __name__ == "__main__":
    main()
