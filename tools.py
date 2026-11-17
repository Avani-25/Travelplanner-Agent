"""LangChain tools used by the travel planning agent.

Tools: flight search, hotel recommendation, places discovery,
weather lookup (Open-Meteo) and budget estimation.
"""

import logging
from datetime import date, datetime, timedelta

import requests
from langchain_core.tools import tool

from src.config import (
    ARCHIVE_DELAY_DAYS,
    CITY_COORDINATES,
    DEFAULT_DAILY_LOCAL_EXPENSE,
    FLIGHTS_FILE,
    FORECAST_MAX_DAYS,
    HOTELS_FILE,
    MAX_TRIP_DAYS,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    PLACES_FILE,
)
from src.data_loader import load_json, normalize_city

logger = logging.getLogger(__name__)

# WMO weather codes used by Open-Meteo -> readable text
WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Severe thunderstorm with hail",
}


# --------------------------------------------------------------------------
# 1. FLIGHT SEARCH TOOL
# --------------------------------------------------------------------------
def _duration_hours(departure: str, arrival: str) -> float:
    """Return flight duration in hours from two ISO timestamps."""
    delta = datetime.fromisoformat(arrival) - datetime.fromisoformat(departure)
    return round(delta.total_seconds() / 3600, 1)


@tool
def search_flights(source: str, destination: str, sort_by: str = "cheapest") -> dict:
    """Search flights between two cities in the flights dataset.

    Args:
        source: Departure city, e.g. 'Delhi'.
        destination: Arrival city, e.g. 'Goa'.
        sort_by: 'cheapest' (lowest price first) or 'fastest' (shortest duration first).

    Returns a dict with the sorted flight options and a recommended flight.
    If no flight exists on the route, it lists destinations reachable from the source.
    To find a return flight, call this tool again with source and destination swapped.
    """
    try:
        source, destination = normalize_city(source), normalize_city(destination)
        flights = load_json(FLIGHTS_FILE)
        matches = [f for f in flights if f["from"] == source and f["to"] == destination]

        if not matches:
            reachable = sorted({f["to"] for f in flights if f["from"] == source})
            return {
                "found": False,
                "message": f"No flights from {source} to {destination} in the dataset.",
                "destinations_available_from_source": reachable,
            }

        options = [
            {
                "flight_id": f["flight_id"],
                "airline": f["airline"],
                "from": f["from"],
                "to": f["to"],
                "departure_time": datetime.fromisoformat(f["departure_time"]).strftime(
                    "%H:%M"
                ),
                "arrival_time": datetime.fromisoformat(f["arrival_time"]).strftime(
                    "%H:%M"
                ),
                "duration_hours": _duration_hours(
                    f["departure_time"], f["arrival_time"]
                ),
                "price": f["price"],
            }
            for f in matches
        ]

        if sort_by.lower() == "fastest":
            options.sort(key=lambda o: (o["duration_hours"], o["price"]))
        else:
            options.sort(key=lambda o: (o["price"], o["duration_hours"]))

        return {
            "found": True,
            "sort_by": sort_by,
            "options": options,
            "recommended": options[0],
        }
    except Exception as exc:  # tool must never crash the agent
        logger.exception("search_flights failed")
        return {"found": False, "error": f"Flight search failed: {exc}"}


# --------------------------------------------------------------------------
# 2. HOTEL RECOMMENDATION TOOL
# --------------------------------------------------------------------------
@tool
def recommend_hotels(
    city: str,
    min_stars: int = 0,
    max_price_per_night: float = 0,
    sort_by: str = "best_value",
) -> dict:
    """Recommend hotels in a city, filtered by star rating and price.

    Args:
        city: City name, e.g. 'Goa'.
        min_stars: Minimum star rating (1-5). Use 0 for no filter.
        max_price_per_night: Maximum price per night in INR. Use 0 for no limit.
        sort_by: 'best_value' (stars per rupee), 'rating' (highest stars first)
                 or 'cheapest' (lowest price first).

    Returns up to 5 hotels and a recommended one.
    """
    try:
        city = normalize_city(city)
        hotels = load_json(HOTELS_FILE)
        in_city = [h for h in hotels if h["city"] == city]

        if not in_city:
            return {
                "found": False,
                "message": f"No hotels found in {city}.",
                "cities_available": sorted({h["city"] for h in hotels}),
            }

        filtered = [
            h
            for h in in_city
            if h["stars"] >= min_stars
            and (
                max_price_per_night <= 0 or h["price_per_night"] <= max_price_per_night
            )
        ]
        if not filtered:
            return {
                "found": False,
                "message": (
                    f"No hotels in {city} match min_stars={min_stars} and "
                    f"max_price_per_night={max_price_per_night}. Try relaxing the filters."
                ),
            }

        sort_by = sort_by.lower()
        if sort_by == "rating":
            filtered.sort(key=lambda h: (-h["stars"], h["price_per_night"]))
        elif sort_by == "cheapest":
            filtered.sort(key=lambda h: h["price_per_night"])
        else:  # best_value
            filtered.sort(key=lambda h: -(h["stars"] / h["price_per_night"]))

        top = filtered[:5]
        return {
            "found": True,
            "sort_by": sort_by,
            "options": top,
            "recommended": top[0],
        }
    except Exception as exc:
        logger.exception("recommend_hotels failed")
        return {"found": False, "error": f"Hotel search failed: {exc}"}


# --------------------------------------------------------------------------
# 3. PLACES DISCOVERY TOOL
# --------------------------------------------------------------------------
@tool
def discover_places(
    city: str,
    place_type: str = "",
    min_rating: float = 0.0,
    limit: int = 5,
) -> dict:
    """Discover tourist attractions in a city, ranked by rating.

    Args:
        city: City name, e.g. 'Kolkata'.
        place_type: Optional attraction type such as 'temple', 'beach', 'fort',
                    'museum', 'park', 'lake', 'market', 'monument'. Empty means any type.
        min_rating: Minimum rating (0-5).
        limit: Maximum number of places to return.

    Returns places sorted by highest rating first, plus the types available in that city.
    """
    try:
        city = normalize_city(city)
        places = load_json(PLACES_FILE)
        in_city = [p for p in places if p["city"] == city]

        if not in_city:
            return {
                "found": False,
                "message": f"No places found for {city}.",
                "cities_available": sorted({p["city"] for p in places}),
            }

        filtered = [
            p
            for p in in_city
            if p["rating"] >= min_rating
            and (not place_type or p["type"].lower() == place_type.strip().lower())
        ]
        filtered.sort(key=lambda p: -p["rating"])

        return {
            "found": bool(filtered),
            "city": city,
            "types_available_in_city": sorted({p["type"] for p in in_city}),
            "places": filtered[:limit],
            "message": (
                "" if filtered else "No places match the filters. Try relaxing them."
            ),
        }
    except Exception as exc:
        logger.exception("discover_places failed")
        return {"found": False, "error": f"Places search failed: {exc}"}


# --------------------------------------------------------------------------
# 4. WEATHER LOOKUP TOOL (Open-Meteo, no API key)
# --------------------------------------------------------------------------
def _years_to_shift_back(end: date) -> int:
    """How many years back we must go so the dates fall inside archive data."""
    cutoff = date.today() - timedelta(days=ARCHIVE_DELAY_DAYS)
    years = 0
    while end - timedelta(days=365 * years) > cutoff:
        years += 1
    return years


@tool
def get_weather(city: str, start_date: str, end_date: str) -> dict:
    """Get day-wise weather for a city between two dates using the Open-Meteo API.

    Args:
        city: City name, e.g. 'Goa'.
        start_date: First travel day in YYYY-MM-DD format.
        end_date: Last travel day in YYYY-MM-DD format.

    Uses the live forecast when the dates are within the next ~15 days.
    Otherwise it uses historical data for the same dates in a past year and
    labels the result as an estimate (check the 'source' and 'note' fields).
    """
    try:
        city = normalize_city(city)
        if city not in CITY_COORDINATES:
            return {
                "error": f"No coordinates for '{city}'. Available: {sorted(CITY_COORDINATES)}"
            }

        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        if end < start:
            return {"error": "end_date must be on or after start_date."}
        if (end - start).days + 1 > MAX_TRIP_DAYS:
            return {
                "error": f"Trips longer than {MAX_TRIP_DAYS} days are not supported."
            }

        today = date.today()
        latitude, longitude = CITY_COORDINATES[city]

        if start >= today and end <= today + timedelta(days=FORECAST_MAX_DAYS):
            url, source, shift_years = OPEN_METEO_FORECAST_URL, "live_forecast", 0
            note = "Live forecast from Open-Meteo."
        else:
            shift_years = _years_to_shift_back(end)
            url = OPEN_METEO_ARCHIVE_URL
            if shift_years == 0:
                source, note = (
                    "historical_actual",
                    "Observed weather for these past dates.",
                )
            else:
                source = "historical_estimate"
                note = (
                    f"Dates are beyond the forecast window, so this is the weather for the "
                    f"same dates {shift_years} year(s) earlier, used as an estimate."
                )

        shift = timedelta(days=365 * shift_years)
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": (start - shift).isoformat(),
            "end_date": (end - shift).isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "timezone": "auto",
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        daily = response.json()["daily"]

        days = []
        for i, day_str in enumerate(daily["time"]):
            code = daily["weather_code"][i]
            days.append(
                {
                    "date": (date.fromisoformat(day_str) + shift).isoformat(),
                    "condition": WMO_WEATHER_CODES.get(code, "Unknown"),
                    "temp_max_c": daily["temperature_2m_max"][i],
                    "temp_min_c": daily["temperature_2m_min"][i],
                    "precipitation_mm": daily["precipitation_sum"][i],
                }
            )
        return {"city": city, "source": source, "note": note, "days": days}

    except requests.RequestException as exc:
        logger.exception("Open-Meteo request failed")
        return {"error": f"Weather API request failed: {exc}"}
    except (ValueError, KeyError) as exc:
        logger.exception("get_weather failed")
        return {"error": f"Could not process weather request: {exc}"}


# --------------------------------------------------------------------------
# 5. BUDGET ESTIMATION TOOL
# --------------------------------------------------------------------------
@tool
def estimate_budget(
    flight_cost: float,
    hotel_price_per_night: float,
    num_days: int,
    daily_local_expense: float = DEFAULT_DAILY_LOCAL_EXPENSE,
) -> dict:
    """Estimate the total trip cost in INR.

    Args:
        flight_cost: TOTAL flight cost for the whole trip (add the return flight if one is booked).
        hotel_price_per_night: Hotel price per night.
        num_days: Number of trip days. Nights are num_days - 1.
        daily_local_expense: Food and local travel per day (default 800).

    Returns a breakdown of flight, hotel, food & travel, and the total.
    """
    try:
        if num_days < 1:
            return {"error": "num_days must be at least 1."}

        nights = num_days - 1
        hotel_total = hotel_price_per_night * nights
        local_total = daily_local_expense * num_days
        total = flight_cost + hotel_total + local_total

        return {
            "currency": "INR",
            "num_days": num_days,
            "nights": nights,
            "flight": flight_cost,
            "hotel": hotel_total,
            "food_and_local_travel": local_total,
            "total": total,
        }
    except Exception as exc:
        logger.exception("estimate_budget failed")
        return {"error": f"Budget estimation failed: {exc}"}


# All tools in one list, handed to the agent in Phase 3
ALL_TOOLS = [
    search_flights,
    recommend_hotels,
    discover_places,
    get_weather,
    estimate_budget,
]
