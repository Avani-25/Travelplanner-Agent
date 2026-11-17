"""Agentic AI Travel Planning Assistant - single-file Streamlit app.

Everything from src/ (config, data loading, tools, agent, schemas, output, planner)
and the dashboard UI lives in this one file, so it can be uploaded to GitHub and
deployed on Streamlit Community Cloud without any folders.

Required files next to this one in the repository root:
    flights.json, hotels.json, places.json, requirements.txt
Required secret (Streamlit Cloud -> Settings -> Secrets):
    GROQ_API_KEY = "your_key"
"""

import html
import json
import logging
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

import requests
import streamlit as st
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

try:  # local development only; Streamlit Cloud uses st.secrets instead
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)

# ==========================================================================
# 1. CONFIGURATION
# ==========================================================================
BASE_DIR = Path(__file__).resolve().parent


def _find_data_dir() -> Path:
    """Use ./data if it exists, otherwise look for the JSON files next to this file."""
    for candidate in (BASE_DIR / "data", BASE_DIR):
        if (candidate / "flights.json").exists():
            return candidate
    return BASE_DIR


DATA_DIR = _find_data_dir()
FLIGHTS_FILE = DATA_DIR / "flights.json"
HOTELS_FILE = DATA_DIR / "hotels.json"
PLACES_FILE = DATA_DIR / "places.json"

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
CITY_ALIASES = {
    "bengaluru": "Bangalore",
    "bombay": "Mumbai",
    "calcutta": "Kolkata",
    "madras": "Chennai",
    "new delhi": "Delhi",
    "panaji": "Goa",
}

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_MAX_DAYS = 15
ARCHIVE_DELAY_DAYS = 7
MAX_TRIP_DAYS = 7
DEFAULT_DAILY_LOCAL_EXPENSE = 800  # INR per day for food and local travel

CITIES = sorted(CITY_COORDINATES)
PLACE_TYPES = ["beach", "fort", "lake", "market", "monument", "museum", "park", "temple"]
TOOL_ICONS = {
    "search_flights": "✈️",
    "recommend_hotels": "🏨",
    "discover_places": "📍",
    "get_weather": "🌦️",
    "estimate_budget": "💰",
}

WMO_WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm with hail",
}

# Matches 24-hour times such as 07:40 or 18:45 (skips times already followed by AM/PM)
TIME_PATTERN = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?!\s?[AaPp][Mm])")


# ==========================================================================
# 2. DATA HELPERS
# ==========================================================================
def load_json(path: Path) -> list:
    """Load a JSON dataset from disk with clear error messages."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Dataset not found: {path.name}. Upload flights.json, hotels.json and "
            "places.json to the repository root."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path.name}: {exc}") from exc


def normalize_city(name: str) -> str:
    """Convert user input such as 'bengaluru ' or 'goa' to the dataset's city name."""
    cleaned = name.strip().lower()
    if cleaned in CITY_ALIASES:
        return CITY_ALIASES[cleaned]
    return name.strip().title()


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a setting from environment variables, then from Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:  # no secrets file, or key missing
        return default


# ==========================================================================
# 3. TOOLS
# ==========================================================================
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
                "departure_time": datetime.fromisoformat(f["departure_time"]).strftime("%H:%M"),
                "arrival_time": datetime.fromisoformat(f["arrival_time"]).strftime("%H:%M"),
                "duration_hours": _duration_hours(f["departure_time"], f["arrival_time"]),
                "price": f["price"],
            }
            for f in matches
        ]

        if sort_by.lower() == "fastest":
            options.sort(key=lambda o: (o["duration_hours"], o["price"]))
        else:
            options.sort(key=lambda o: (o["price"], o["duration_hours"]))

        return {"found": True, "sort_by": sort_by, "options": options, "recommended": options[0]}
    except Exception as exc:  # a tool must never crash the agent
        logger.exception("search_flights failed")
        return {"found": False, "error": f"Flight search failed: {exc}"}


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
            and (max_price_per_night <= 0 or h["price_per_night"] <= max_price_per_night)
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
        return {"found": True, "sort_by": sort_by, "options": top, "recommended": top[0]}
    except Exception as exc:
        logger.exception("recommend_hotels failed")
        return {"found": False, "error": f"Hotel search failed: {exc}"}


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
            "message": "" if filtered else "No places match the filters. Try relaxing them.",
        }
    except Exception as exc:
        logger.exception("discover_places failed")
        return {"found": False, "error": f"Places search failed: {exc}"}


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
            return {"error": f"No coordinates for '{city}'. Available: {sorted(CITY_COORDINATES)}"}

        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        if end < start:
            return {"error": "end_date must be on or after start_date."}
        if (end - start).days + 1 > MAX_TRIP_DAYS:
            return {"error": f"Trips longer than {MAX_TRIP_DAYS} days are not supported."}

        today = date.today()
        latitude, longitude = CITY_COORDINATES[city]

        if start >= today and end <= today + timedelta(days=FORECAST_MAX_DAYS):
            url, source, shift_years = OPEN_METEO_FORECAST_URL, "live_forecast", 0
            note = "Live forecast from Open-Meteo."
        else:
            shift_years = _years_to_shift_back(end)
            url = OPEN_METEO_ARCHIVE_URL
            if shift_years == 0:
                source, note = "historical_actual", "Observed weather for these past dates."
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


ALL_TOOLS = [search_flights, recommend_hotels, discover_places, get_weather, estimate_budget]


# ==========================================================================
# 4. AGENT
# ==========================================================================
def get_system_prompt() -> str:
    """Return the system prompt that tells the agent how to plan a trip."""
    return f"""You are an expert AI travel planning agent for trips between Indian cities.
Today's date is {date.today().isoformat()}.

YOUR WORKFLOW (follow it in order, calling tools instead of guessing):
1. Understand the request: source city, destination city, start date, number of days
   (3 to 7), budget (if given) and preferences (hotel stars, attraction types).
   If the source, destination, dates or number of days are missing, ask the user for them.
2. Call search_flights for the outbound route (source to destination).
   Then call it again for the return route (destination to source).
3. Call recommend_hotels for the destination, applying the user's star or price preferences.
4. Call discover_places for the destination, using the user's preferred attraction types.
5. Call get_weather for the destination for the exact travel dates.
6. Call estimate_budget with: total flight cost (outbound + return if a return flight exists),
   the chosen hotel's price per night, and the number of days.
7. Write the final answer.

RULES:
- Use ONLY data returned by the tools. Never invent flights, hotels, places, prices or weather.
- Use ONLY places returned by discover_places. NEVER add attractions, beaches, markets or
  restaurants from your own knowledge. If the user's preferred type (for example "beach")
  is not available in the city, say so clearly and use the highest-rated places that ARE available.
- Itinerary items may only be: the returned places, plus generic items such as travel,
  hotel check-in or check-out, and free time. Do not invent specific meals or activities.
- For free time, write only "Free time" or "Free time at the hotel". Do not name walks,
  promenades, beaches, restaurants or any activity that is not a returned place.
- If no flight exists on a route, state plainly that no flight exists from the source to the
  destination, list the reachable destinations the tool returned, then ask what the user wants.
- If no return flight exists, say so and state that the budget covers the outbound flight only.
- Mention weather as an estimate ONLY if the tool source is "historical_estimate".
  A "live_forecast" is a real forecast, not an estimate.
- Each city has few places. Spread them across the days (1-2 per day) and do not repeat a place.
  If there are more days than places, label the extra days as relaxed or free days.
- If a budget is given and the estimated total exceeds it, say so and suggest cheaper choices.
  Only say a trip is within budget if the user gave a budget.
- For every key decision (flight, hotel), explain briefly WHY you selected it.

FINAL ANSWER FORMAT (use these exact sections):
Trip Summary
Flight Selected (outbound and return)
Hotel Recommendation
Day-wise Itinerary (each day with places and that day's weather)
Weather Overview
Budget Breakdown (flight, hotel, food and local travel, total, in INR)
Why We Selected This (reasoning)
"""


def get_llm():
    """Create the chat model. Provider is chosen by LLM_PROVIDER (groq or gemini)."""
    provider = (get_secret("LLM_PROVIDER", "groq") or "groq").lower()

    if provider == "groq":
        api_key = get_secret("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. On Streamlit Cloud add it under "
                "Settings -> Secrets. Locally, put it in the .env file."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=get_secret("GROQ_MODEL", "openai/gpt-oss-120b"),
            api_key=api_key,
        )

    if provider == "gemini":
        api_key = get_secret("GOOGLE_API_KEY") or get_secret("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not found in secrets or .env.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=get_secret("GEMINI_MODEL", "gemini-3.8-flash"),
            api_key=api_key,
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'groq' or 'gemini'.")


def build_agent(llm):
    """Create the tool-calling travel agent around the given chat model."""
    return create_agent(model=llm, tools=ALL_TOOLS, system_prompt=get_system_prompt())


def extract_text(content) -> str:
    """Models may return a string or a list of content blocks; return plain text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts).strip()


def run_agent(agent, query: str) -> dict:
    """Run the agent on a user query and collect the tool calls it made."""
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": 30},
        )
    except Exception as exc:  # rate limits, network, model errors
        logger.exception("Agent run failed")
        return {"answer": "", "steps": [], "error": str(exc)}

    messages = result["messages"]
    steps = []
    for message in messages:
        if message.type == "ai" and getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                steps.append({"tool": call["name"], "args": call["args"]})

    return {"answer": extract_text(messages[-1].content), "steps": steps}


# ==========================================================================
# 5. STRUCTURED OUTPUT (schema, extraction, validation)
# ==========================================================================
class FlightInfo(BaseModel):
    flight_id: str
    airline: str
    from_city: str
    to_city: str
    departure_time: str = Field(description="HH:MM")
    arrival_time: str = Field(description="HH:MM")
    duration_hours: float = 0
    price: float


class HotelInfo(BaseModel):
    name: str
    stars: int
    price_per_night: float
    amenities: list[str] = Field(default_factory=list)


class DayPlan(BaseModel):
    day: int
    date: str = Field(description="YYYY-MM-DD")
    places: list[str] = Field(
        default_factory=list, description="Place names exactly as returned by the tools"
    )
    weather: str = Field(default="", description="Condition and temperature range for the day")
    notes: str = ""


class BudgetBreakdown(BaseModel):
    flight: float = 0
    hotel: float = 0
    food_and_local_travel: float = 0
    total: float = 0


class TripPlan(BaseModel):
    """Full structured trip plan. status tells the UI how to show it."""

    status: Literal["complete", "needs_info", "not_possible"] = "complete"
    message: str = Field(
        default="", description="Question or explanation when status is not 'complete'"
    )
    source: str = ""
    destination: str = ""
    start_date: str = ""
    end_date: str = ""
    num_days: int = 0
    summary: str = ""
    outbound_flight: Optional[FlightInfo] = None
    return_flight: Optional[FlightInfo] = None
    hotel: Optional[HotelInfo] = None
    itinerary: list[DayPlan] = Field(default_factory=list)
    weather_note: str = ""
    budget: Optional[BudgetBreakdown] = None
    reasoning: list[str] = Field(
        default_factory=list, description="Why the flight and hotel were selected"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Caveats: estimated weather, missing return flight, requested place type unavailable"
        ),
    )


EXTRACTION_PROMPT = """You convert a travel plan written in text into the given schema.
Rules:
- Copy values exactly from the text. Never invent or change flights, hotels, places,
  prices or weather.
- If the text asks the user for missing information or says the trip cannot be planned
  (for example no flight exists), set status to needs_info or not_possible. Put the
  COMPLETE explanation AND the question in message, including any alternative
  destinations listed.
- Otherwise set status to complete.
- Put caveats in warnings (missing return flight, unavailable place type).
  Add a weather warning ONLY if the text explicitly says the weather is a historical
  estimate. A live forecast is not an estimate.
- In itinerary places, use only place names that appear in the text. Put travel and
  free-time details in notes.
"""


def validate_plan(plan: TripPlan) -> None:
    """Cross-check the plan against the datasets and drop anything not found there."""
    if plan.status != "complete":
        return

    flights = {f["flight_id"]: f for f in load_json(FLIGHTS_FILE)}
    for flight in (plan.outbound_flight, plan.return_flight):
        if flight and (
            flight.flight_id not in flights or flights[flight.flight_id]["price"] != flight.price
        ):
            plan.warnings.append(f"Flight {flight.flight_id} could not be verified in the dataset.")

    valid_places = {p["name"] for p in load_json(PLACES_FILE) if p["city"] == plan.destination}
    for day in plan.itinerary:
        unknown = [p for p in day.places if p not in valid_places]
        if unknown:
            day.places = [p for p in day.places if p in valid_places]
            plan.warnings.append(f"Removed places not found in the dataset: {', '.join(unknown)}")

    if plan.hotel:
        hotel_ok = any(
            h["name"] == plan.hotel.name
            and h["city"] == plan.destination
            and h["price_per_night"] == plan.hotel.price_per_night
            for h in load_json(HOTELS_FILE)
        )
        if not hotel_ok:
            plan.warnings.append(f"Hotel '{plan.hotel.name}' could not be verified in the dataset.")


def _fix_budget_total(plan: TripPlan) -> None:
    """Recompute the total in Python so the arithmetic is always correct."""
    if plan.budget:
        b = plan.budget
        b.total = b.flight + b.hotel + b.food_and_local_travel


def extract_trip_plan(llm, answer_text: str) -> TripPlan:
    """Ask the LLM to fill the TripPlan schema from the agent's written answer."""
    last_error = None
    for method in ("json_schema", "function_calling"):
        try:
            structured_llm = llm.with_structured_output(TripPlan, method=method)
            plan = structured_llm.invoke([("system", EXTRACTION_PROMPT), ("human", answer_text)])
            if isinstance(plan, dict):
                plan = TripPlan(**plan)
            _fix_budget_total(plan)
            validate_plan(plan)
            return plan
        except Exception as exc:  # try the next method
            logger.warning("Structured output via %s failed: %s", method, exc)
            last_error = exc
    raise RuntimeError(f"Could not build a structured plan: {last_error}")


def plan_trip(agent, llm, query: str) -> dict:
    """Plan a trip: agent -> structured plan -> JSON. Returns the plan or an error."""
    run = run_agent(agent, query)
    if run.get("error"):
        return {"error": run["error"]}

    try:
        plan = extract_trip_plan(llm, run["answer"])
    except RuntimeError as exc:
        # Fall back to the agent's own text so the user still gets an answer
        return {"error": str(exc), "agent_answer": run["answer"], "steps": run["steps"]}

    return {
        "plan": plan,
        "json": plan.model_dump_json(indent=2),
        "agent_answer": run["answer"],
        "steps": run["steps"],
    }


# ==========================================================================
# 6. DASHBOARD (styles and rendering helpers)
# ==========================================================================
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp {
  background: linear-gradient(160deg, #f6f7ff 0%, #eef2ff 55%, #fdf2f8 100%);
  background-attachment: fixed;
  color: #1b2140;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1200px; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid #e4e7f5;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  color: #1b2140 !important;
  opacity: 1 !important;
}
section[data-testid="stSidebar"] h2 { font-size: 1.15rem; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="input"],
section[data-testid="stSidebar"] div[data-baseweb="base-input"],
section[data-testid="stSidebar"] input {
  background-color: #f3f5ff !important;
  color: #1b2140 !important;
  border-radius: 10px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] div,
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
  color: #1b2140 !important;
  fill: #1b2140 !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
  background-color: #e6e9fb !important;
  color: #1b2140 !important;
}
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li {
  background-color: #ffffff !important;
  color: #1b2140 !important;
}
div[data-baseweb="popover"] li:hover { background-color: #eef0ff !important; }
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p { color: #5b6489 !important; }

.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, #6d4aff, #1fc8ac);
  color: #ffffff; font-weight: 600; border: none; border-radius: 12px; padding: 0.6rem 1rem;
  box-shadow: 0 6px 16px rgba(109, 74, 255, 0.25);
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.06); transform: translateY(-1px); }

/* ---------- hero ---------- */
.hero { padding: 28px 8px 18px 8px; }
.hero-badge {
  display: inline-block; padding: 6px 14px; border-radius: 999px; font-size: 0.8rem;
  background: #ece8ff; color: #5a3fd6; border: 1px solid #d6ccff;
}
.hero h1 { font-size: 3rem; line-height: 1.15; margin: 16px 0 10px 0; color: #1b2140; font-weight: 700; }
.hero p { color: #5b6489; font-size: 1.05rem; max-width: 640px; }
.grad {
  background: linear-gradient(90deg, #6d4aff, #1fc8ac);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* ---------- grids and cards ---------- */
.grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 14px 0; }
.grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin: 14px 0; }
@media (max-width: 900px) {
  .grid-4 { grid-template-columns: repeat(2, 1fr); }
  .grid-2 { grid-template-columns: 1fr; }
  .hero h1 { font-size: 2.2rem; }
}
.card {
  background: #ffffff; border: 1px solid #e4e7f5; border-radius: 18px; padding: 20px;
  box-shadow: 0 6px 20px rgba(60, 70, 140, 0.07);
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.card:hover { transform: translateY(-3px); border-color: #c9bfff; box-shadow: 0 12px 28px rgba(109, 74, 255, 0.14); }
.card .icon { font-size: 1.6rem; margin-bottom: 8px; }
.card .title { color: #1b2140; font-weight: 600; margin-bottom: 4px; }
.card .desc { color: #5b6489; font-size: 0.88rem; }

.stat {
  padding: 18px 20px; border-radius: 18px; border: 1px solid #e4e7f5; background: #ffffff;
  box-shadow: 0 6px 20px rgba(60, 70, 140, 0.07);
}
.stat.main { background: linear-gradient(135deg, #6d4aff 0%, #1fc8ac 100%); border: none; }
.stat .label { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #5b6489; }
.stat .value { font-size: 1.7rem; font-weight: 700; color: #1b2140; margin-top: 4px; }
.stat.main .label, .stat.main .value { color: #ffffff; }

/* ---------- trip header ---------- */
.trip-head { margin: 4px 0 8px 0; }
.trip-head h2 { color: #1b2140; font-size: 2rem; margin: 10px 0 4px 0; font-weight: 700; }
.trip-head .dates { color: #5b6489; }
.trip-head .summary { color: #3a4268; margin-top: 8px; max-width: 800px; }
.chip {
  display: inline-block; padding: 6px 14px; border-radius: 999px; font-size: 0.85rem;
  background: #dff7f2; color: #0b7f6c; border: 1px solid #b6ebe0;
}

/* ---------- sections ---------- */
.section-title { color: #1b2140; font-size: 1.15rem; font-weight: 600; margin: 26px 0 6px 0; }
.tag { font-size: 0.7rem; letter-spacing: 0.1em; color: #6d4aff; font-weight: 600; margin-bottom: 12px; }
.route { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.route .time { font-size: 1.6rem; font-weight: 700; color: #1b2140; white-space: nowrap; }
.route .ampm { font-size: 0.85rem; font-weight: 600; color: #6d4aff; margin-left: 4px; }
.route .city { color: #5b6489; font-size: 0.85rem; }
.route .stop.right { text-align: right; }
.route .path { flex: 1; text-align: center; color: #5b6489; font-size: 0.8rem; }
.route .dash { border-top: 2px dashed #c9cfe8; margin-top: 4px; }
.flight-meta { display: flex; justify-content: space-between; margin-top: 14px; color: #3a4268; font-size: 0.9rem; }
.price { color: #0b8f7a; font-weight: 600; }
.empty { color: #5b6489; padding: 18px 0; }

.hotel-name { font-size: 1.3rem; font-weight: 600; color: #1b2140; }
.hotel-sub { color: #5b6489; margin: 4px 0 10px 0; }

.pill {
  display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; margin: 3px 6px 3px 0;
  background: #ece8ff; color: #5a3fd6;
}
.pill.soft { background: #eef0f8; color: #4a5380; }

.day { border-left: 3px solid #6d4aff; padding: 4px 0 4px 18px; margin: 0 0 14px 6px; }
.day .day-top { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
.day .day-name { font-weight: 600; color: #1b2140; }
.day .day-date { color: #5b6489; font-size: 0.85rem; margin-left: 8px; }
.weather {
  padding: 4px 12px; border-radius: 999px; font-size: 0.82rem; color: #9a5200;
  background: #fff3de; border: 1px solid #ffd9a0;
}
.day .note { color: #5b6489; font-size: 0.88rem; margin-top: 6px; }

.bar { display: flex; height: 14px; border-radius: 999px; overflow: hidden; margin: 14px 0; background: #eef0f8; }
.legend-row { display: flex; justify-content: space-between; padding: 6px 0; color: #3a4268; font-size: 0.92rem; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }

.reason { color: #3a4268; padding: 6px 0; }
.note-box {
  border-radius: 14px; padding: 12px 16px; margin: 8px 0; color: #9a5200;
  background: #fff8ec; border: 1px solid #ffd9a0; font-size: 0.9rem;
}
.step { padding: 8px 0; color: #3a4268; border-bottom: 1px solid #eceffa; }
.step code { color: #0b7f6c; background: #eef9f6; padding: 2px 6px; border-radius: 6px; font-size: 0.78rem; }
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)


def esc(value) -> str:
    """HTML-escape any value (the LLM text is never trusted as markup)."""
    return html.escape(str(value))


def show(markup: str) -> None:
    """Render an HTML snippet (indentation removed so Markdown does not treat it as code)."""
    st.markdown(" ".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def inr(amount: float) -> str:
    """Format an amount in Indian rupees."""
    return f"₹{amount:,.0f}"


def time_parts(time_str: str) -> tuple[str, str]:
    """Convert '18:45' to ('6:45', 'PM'). Returns the input unchanged if it cannot be parsed."""
    try:
        parsed = datetime.strptime(time_str.strip(), "%H:%M")
    except (ValueError, AttributeError):
        return str(time_str), ""
    return parsed.strftime("%I:%M").lstrip("0"), parsed.strftime("%p")


def convert_times(text: str) -> str:
    """Convert every 24-hour time inside free text to 12-hour AM/PM format."""

    def replace(match: re.Match) -> str:
        clock, suffix = time_parts(match.group(0))
        return f"{clock} {suffix}".strip()

    return TIME_PATTERN.sub(replace, str(text))


def weather_icon(text: str) -> str:
    """Pick an emoji that matches a weather description."""
    t = text.lower()
    rules = [
        ("thunder", "⛈️"), ("rain", "🌧️"), ("drizzle", "🌦️"), ("shower", "🌧️"),
        ("snow", "❄️"), ("fog", "🌫️"), ("overcast", "☁️"), ("partly", "⛅"),
        ("mainly clear", "🌤️"), ("clear", "☀️"),
    ]
    for keyword, icon in rules:
        if keyword in t:
            return icon
    return "🌡️"


def build_query(source, destination, start_date, num_days, min_stars, place_types, budget):
    """Turn the form inputs into a natural-language request for the agent."""
    query = (
        f"Plan a {num_days}-day trip from {source} to {destination} "
        f"starting {start_date.isoformat()}."
    )
    if min_stars > 0:
        query += f" I want a hotel with at least {min_stars} stars."
    if place_types:
        query += f" I prefer these attraction types: {', '.join(place_types)}."
    if budget > 0:
        query += f" My total budget is {budget} INR."
    return query


@st.cache_resource
def load_agent():
    """Create the LLM and agent once and reuse them across reruns."""
    llm = get_llm()
    return llm, build_agent(llm)


# ==========================================================================
# 7. DASHBOARD COMPONENTS
# ==========================================================================
def render_hero() -> None:
    show("""
    <div class="hero">
      <div class="hero-badge">✨ Agentic AI · LangChain</div>
      <h1>Plan your perfect trip,<br><span class="grad">automatically.</span></h1>
      <p>An AI travel agent that searches flights, picks hotels, finds attractions,
      checks live weather and builds a day-wise itinerary with a full budget.</p>
    </div>
    """)


def render_features() -> None:
    show("""
    <div class="grid-4">
      <div class="card"><div class="icon">✈️</div><div class="title">Smart flights</div>
        <div class="desc">Cheapest or fastest option, outbound and return.</div></div>
      <div class="card"><div class="icon">🏨</div><div class="title">Best-value stays</div>
        <div class="desc">Filtered by stars and price per night.</div></div>
      <div class="card"><div class="icon">📍</div><div class="title">Top-rated places</div>
        <div class="desc">Ranked attractions spread across your days.</div></div>
      <div class="card"><div class="icon">🌦️</div><div class="title">Live weather</div>
        <div class="desc">Open-Meteo forecast for every travel day.</div></div>
    </div>
    """)
    st.caption("👈 Choose your trip details in the sidebar and click **Plan my trip**.")


def render_trip_header(plan: TripPlan) -> None:
    show(f"""
    <div class="trip-head">
      <span class="chip">{esc(plan.source)} → {esc(plan.destination)}</span>
      <h2>{plan.num_days}-Day Trip to {esc(plan.destination)}</h2>
      <div class="dates">📅 {esc(plan.start_date)} to {esc(plan.end_date)}</div>
      <div class="summary">{esc(convert_times(plan.summary))}</div>
    </div>
    """)


def render_stats(plan: TripPlan) -> None:
    if not plan.budget:
        return
    b = plan.budget
    show(f"""
    <div class="grid-4">
      <div class="stat main"><div class="label">Total cost</div><div class="value">{inr(b.total)}</div></div>
      <div class="stat"><div class="label">Flights</div><div class="value">{inr(b.flight)}</div></div>
      <div class="stat"><div class="label">Hotel</div><div class="value">{inr(b.hotel)}</div></div>
      <div class="stat"><div class="label">Food &amp; travel</div><div class="value">{inr(b.food_and_local_travel)}</div></div>
    </div>
    """)


def flight_card(label: str, flight) -> str:
    if flight is None:
        return (
            f'<div class="card"><div class="tag">{label.upper()}</div>'
            f'<div class="empty">No {label.lower()} flight found in the dataset.</div></div>'
        )
    dep_time, dep_suffix = time_parts(flight.departure_time)
    arr_time, arr_suffix = time_parts(flight.arrival_time)
    duration = f"{flight.duration_hours:g}h" if flight.duration_hours else "✈"
    return f"""
    <div class="card">
      <div class="tag">{label.upper()}</div>
      <div class="route">
        <div class="stop"><div class="time">{esc(dep_time)}<span class="ampm">{dep_suffix}</span></div>
          <div class="city">{esc(flight.from_city)}</div></div>
        <div class="path">{duration}<div class="dash"></div></div>
        <div class="stop right"><div class="time">{esc(arr_time)}<span class="ampm">{arr_suffix}</span></div>
          <div class="city">{esc(flight.to_city)}</div></div>
      </div>
      <div class="flight-meta">
        <span>{esc(flight.airline)} · {esc(flight.flight_id)}</span>
        <span class="price">{inr(flight.price)}</span>
      </div>
    </div>
    """


def render_flights(plan: TripPlan) -> None:
    st.markdown('<div class="section-title">✈️ Flights</div>', unsafe_allow_html=True)
    show(
        f'<div class="grid-2">{flight_card("Outbound", plan.outbound_flight)}'
        f'{flight_card("Return", plan.return_flight)}</div>'
    )


def render_hotel(plan: TripPlan) -> None:
    if not plan.hotel:
        return
    h = plan.hotel
    amenities = "".join(f'<span class="pill soft">{esc(a)}</span>' for a in h.amenities)
    st.markdown('<div class="section-title">🏨 Where you will stay</div>', unsafe_allow_html=True)
    show(f"""
    <div class="card">
      <div class="hotel-name">{esc(h.name)}</div>
      <div class="hotel-sub">{"⭐" * int(h.stars)} &nbsp;·&nbsp; <span class="price">{inr(h.price_per_night)}</span> per night</div>
      {amenities}
    </div>
    """)


def render_itinerary(plan: TripPlan) -> None:
    st.markdown('<div class="section-title">🗓️ Day-wise itinerary</div>', unsafe_allow_html=True)
    blocks = []
    for day in plan.itinerary:
        places = "".join(f'<span class="pill">📍 {esc(p)}</span>' for p in day.places) or (
            '<span class="pill soft">Free day</span>'
        )
        weather = (
            f'<span class="weather">{weather_icon(day.weather)} {esc(day.weather)}</span>'
            if day.weather
            else ""
        )
        note = f'<div class="note">{esc(convert_times(day.notes))}</div>' if day.notes else ""
        blocks.append(f"""
        <div class="day">
          <div class="day-top">
            <div><span class="day-name">Day {day.day}</span><span class="day-date">{esc(day.date)}</span></div>
            {weather}
          </div>
          <div>{places}</div>
          {note}
        </div>
        """)
    show(f'<div class="card">{"".join(blocks)}</div>')


def render_budget(plan: TripPlan) -> None:
    if not plan.budget:
        return
    b = plan.budget
    parts = [
        ("Flights", b.flight, "#6d4aff"),
        ("Hotel", b.hotel, "#1fc8ac"),
        ("Food & local travel", b.food_and_local_travel, "#ffb454"),
    ]
    total = b.total or 1
    bar = "".join(
        f'<div style="width:{value / total * 100:.1f}%;background:{color}"></div>'
        for _, value, color in parts
    )
    legend = "".join(
        f'<div class="legend-row"><span><span class="dot" style="background:{color}"></span>'
        f"{esc(name)}</span><span>{inr(value)} · {value / total * 100:.0f}%</span></div>"
        for name, value, color in parts
    )
    st.markdown('<div class="section-title">💰 Budget breakdown</div>', unsafe_allow_html=True)
    show(f"""
    <div class="card">
      <div class="bar">{bar}</div>
      {legend}
      <div class="legend-row"><strong>Total</strong><strong class="price">{inr(b.total)}</strong></div>
    </div>
    """)


def render_reasoning(plan: TripPlan) -> None:
    if plan.reasoning:
        st.markdown('<div class="section-title">🧠 Why we selected this</div>', unsafe_allow_html=True)
        items = "".join(f'<div class="reason">• {esc(convert_times(r))}</div>' for r in plan.reasoning)
        show(f'<div class="card">{items}</div>')
    for warning in plan.warnings:
        show(f'<div class="note-box">⚠️ {esc(convert_times(warning))}</div>')


def render_steps(steps) -> None:
    st.caption("Tools the agent called, in order:")
    rows = "".join(
        f'<div class="step">{i}. {TOOL_ICONS.get(s["tool"], "🔧")} '
        f'<strong>{esc(s["tool"])}</strong> <code>{esc(s["args"])}</code></div>'
        for i, s in enumerate(steps, 1)
    )
    show(f'<div class="card">{rows}</div>')


def render_result(result: dict) -> None:
    """Show the planner result: error, follow-up question, or the full dashboard."""
    if result.get("error"):
        st.error(f"Something went wrong: {result['error']}")
        if result.get("agent_answer"):
            st.markdown(result["agent_answer"])
        return

    plan = result["plan"]
    if plan.status != "complete":
        st.info(convert_times(plan.message))
        return

    render_trip_header(plan)
    render_stats(plan)

    tab_plan, tab_json, tab_steps = st.tabs(
        ["🗺️ Itinerary", "🧾 JSON output", "🧠 How the agent decided"]
    )
    with tab_plan:
        render_flights(plan)
        render_hotel(plan)
        render_itinerary(plan)
        render_budget(plan)
        render_reasoning(plan)
    with tab_json:
        st.json(plan.model_dump())
        st.download_button(
            "⬇️ Download JSON",
            result["json"],
            file_name="trip_plan.json",
            mime="application/json",
        )
    with tab_steps:
        render_steps(result["steps"])


# ==========================================================================
# 8. SIDEBAR AND MAIN FLOW
# ==========================================================================
with st.sidebar:
    st.header("🧳 Trip details")
    source = st.selectbox("From", CITIES, index=CITIES.index("Bangalore"))
    destination = st.selectbox("To", CITIES, index=CITIES.index("Goa"))
    start_date = st.date_input(
        "Start date", value=date.today() + timedelta(days=3), min_value=date.today()
    )
    num_days = st.slider("Number of days", 3, 7, 3)
    min_stars = st.slider("Minimum hotel stars", 0, 5, 3)
    place_types = st.multiselect("Preferred attractions", PLACE_TYPES)
    budget = st.number_input("Total budget in INR (0 = no limit)", min_value=0, step=1000, value=0)
    submitted = st.button("✨ Plan my trip", type="primary", use_container_width=True)

if submitted:
    if source == destination:
        st.warning("Please choose different source and destination cities.")
        st.stop()

    try:
        llm, agent = load_agent()
    except Exception as exc:
        st.error(f"Could not start the AI model: {exc}")
        st.stop()

    query = build_query(source, destination, start_date, num_days, min_stars, place_types, budget)
    with st.spinner("✈️ The agent is searching flights, hotels, places and weather..."):
        st.session_state["result"] = plan_trip(agent, llm, query)

result = st.session_state.get("result")
if result is None:
    render_hero()
    render_features()
else:
    render_result(result)
