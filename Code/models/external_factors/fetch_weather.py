from __future__ import annotations
from cpl_stadiums import STADIUMS

# (avg_temp_celsius, rain_probability)
CLIMATE_DATA = {
    # Victoria 
    "Pacific": {
        4: (10, 0.40), 5: (14, 0.25), 6: (17, 0.15), 7: (20, 0.05), 8: (20, 0.05), 9: (17, 0.20), 10: (12, 0.45)
    },
    # Calgary / Edmonton
    "Mountain": {
        4: (8, 0.30), 5: (14, 0.35), 6: (18, 0.40), 7: (23, 0.30), 8: (22, 0.25), 9: (16, 0.20), 10: (9, 0.15)
    },
    # Winnipeg
    "Central": {
        4: (6, 0.25), 5: (15, 0.30), 6: (21, 0.35), 7: (26, 0.30), 8: (25, 0.25), 9: (18, 0.25), 10: (8, 0.20)
    },
    # Ontario / Ottawa
    "Eastern": {
        4: (9, 0.25), 5: (16, 0.30), 6: (22, 0.35), 7: (26, 0.30), 8: (25, 0.25), 9: (20, 0.25), 10: (13, 0.25)
    },
    # Halifax
    "Atlantic": {
        4: (7, 0.35), 5: (12, 0.35), 6: (17, 0.35), 7: (20, 0.30), 8: (20, 0.30), 9: (16, 0.35), 10: (10, 0.40)
    },
}

DEFAULT_ESTIMATE = (15, 0.25)


def get_weather_estimate(stadium_name: str, month: int):
    """Return (avg_temp_c, rain_prob) for a given stadium and month.

    Stadium lookup is by exact stadium_name match against STADIUMS[*]['name'].
    If the stadium is unknown, defaults to Eastern.
    """
    region = "Eastern"
    for _, data in STADIUMS.items():
        if data.get("name") == stadium_name:
            region = data.get("tz", "Eastern")
            break

    # Apr-Oct
    month = int(month)
    month = max(4, min(10, month))

    return CLIMATE_DATA.get(region, CLIMATE_DATA["Eastern"]).get(month, DEFAULT_ESTIMATE)


def main():
    for team, data in STADIUMS.items():
        temp, rain = get_weather_estimate(data["name"], 7)
        print(f"{team:10s} | {data['name']:<24s} | Jul est: {temp}C, rain={rain:.0%}")


if __name__ == "__main__":
    main()
