"""Project-wide configuration: file paths, city coordinates and default costs."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

FLIGHTS_FILE = DATA_DIR / "flights.json"
HOTELS_FILE = DATA_DIR / "hotels.json"
PLACES_FILE = DATA_DIR / "places.json"

# Latitude / longitude for every city present in the datasets
CITY_COORDINATES = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Goa": (15.2993, 74.1240),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Jaipur": (26.9124, 75.7873),
}

# Alternative names users might type
CITY_ALIASES = {
    "bengaluru": "Bangalore",
    "bombay": "Mumbai",
    "calcutta": "Kolkata",
    "madras": "Chennai",
    "new delhi": "Delhi",
    "panaji": "Goa",
}

# Open-Meteo (free, no API key)
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_MAX_DAYS = 15  # live forecast window
ARCHIVE_DELAY_DAYS = 7  # archive data lags a few days behind today
MAX_TRIP_DAYS = 7

# Budget defaults (INR)
DEFAULT_DAILY_LOCAL_EXPENSE = 800  # food + local travel per day
