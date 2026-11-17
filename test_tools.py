"""Unit tests for the travel planner tools."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from src.tools import (
    _years_to_shift_back,
    discover_places,
    estimate_budget,
    get_weather,
    recommend_hotels,
    search_flights,
)


# ---------------------------- flights ----------------------------
def test_cheapest_flight_bangalore_to_goa():
    result = search_flights.invoke(
        {"source": "Bangalore", "destination": "Goa", "sort_by": "cheapest"}
    )
    assert result["found"] is True
    assert result["recommended"]["flight_id"] == "FL0021"
    assert result["recommended"]["price"] == 5356


def test_fastest_flight_bangalore_to_goa():
    result = search_flights.invoke(
        {"source": "Bangalore", "destination": "Goa", "sort_by": "fastest"}
    )
    assert result["recommended"]["flight_id"] == "FL0020"
    assert result["recommended"]["duration_hours"] == 1.0


def test_flight_city_alias_is_normalised():
    result = search_flights.invoke({"source": "Bengaluru", "destination": "goa"})
    assert result["found"] is True


def test_flight_no_route_lists_alternatives():
    result = search_flights.invoke({"source": "Delhi", "destination": "Goa"})
    assert result["found"] is False
    assert "Kolkata" in result["destinations_available_from_source"]


# ---------------------------- hotels ----------------------------
def test_hotels_best_value_with_min_stars():
    result = recommend_hotels.invoke({"city": "Goa", "min_stars": 4})
    assert result["found"] is True
    assert all(h["stars"] >= 4 for h in result["options"])
    assert result["recommended"]["price_per_night"] == 1232


def test_hotels_impossible_filter_returns_message():
    result = recommend_hotels.invoke({"city": "Goa", "max_price_per_night": 100})
    assert result["found"] is False
    assert "message" in result


def test_hotels_unknown_city():
    result = recommend_hotels.invoke({"city": "Paris"})
    assert result["found"] is False


# ---------------------------- places ----------------------------
def test_places_sorted_by_rating():
    result = discover_places.invoke({"city": "Kolkata"})
    ratings = [p["rating"] for p in result["places"]]
    assert ratings == sorted(ratings, reverse=True)
    assert ratings[0] == 4.9


def test_places_type_and_rating_filter():
    result = discover_places.invoke(
        {"city": "Kolkata", "place_type": "temple", "min_rating": 4.5}
    )
    assert len(result["places"]) == 2
    assert all(p["type"] == "temple" for p in result["places"])


def test_places_missing_type_is_reported():
    result = discover_places.invoke({"city": "Goa", "place_type": "beach"})
    assert result["found"] is False
    assert "beach" not in result["types_available_in_city"]


# ---------------------------- budget ----------------------------
def test_budget_calculation():
    result = estimate_budget.invoke(
        {"flight_cost": 4800, "hotel_price_per_night": 3200, "num_days": 3}
    )
    assert result["nights"] == 2
    assert result["hotel"] == 6400
    assert result["food_and_local_travel"] == 2400
    assert result["total"] == 13600


def test_budget_rejects_zero_days():
    result = estimate_budget.invoke(
        {"flight_cost": 1000, "hotel_price_per_night": 1000, "num_days": 0}
    )
    assert "error" in result


# ---------------------------- weather ----------------------------
def test_weather_unknown_city():
    result = get_weather.invoke(
        {"city": "Paris", "start_date": "2026-11-20", "end_date": "2026-11-22"}
    )
    assert "error" in result


def test_weather_end_before_start():
    result = get_weather.invoke(
        {"city": "Goa", "start_date": "2026-11-22", "end_date": "2026-11-20"}
    )
    assert "error" in result


def test_weather_trip_too_long():
    start = date.today() + timedelta(days=1)
    result = get_weather.invoke(
        {
            "city": "Goa",
            "start_date": start.isoformat(),
            "end_date": (start + timedelta(days=10)).isoformat(),
        }
    )
    assert "error" in result


def test_weather_live_forecast_is_parsed():
    start = date.today() + timedelta(days=1)
    days = [(start + timedelta(days=i)).isoformat() for i in range(3)]
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "daily": {
            "time": days,
            "temperature_2m_max": [34.0, 33.5, 35.1],
            "temperature_2m_min": [20.0, 19.5, 21.0],
            "precipitation_sum": [0.0, 0.0, 1.2],
            "weather_code": [0, 1, 61],
        }
    }
    with patch("src.tools.requests.get", return_value=fake_response):
        result = get_weather.invoke(
            {"city": "Goa", "start_date": days[0], "end_date": days[-1]}
        )
    assert result["source"] == "live_forecast"
    assert len(result["days"]) == 3
    assert result["days"][0]["condition"] == "Clear sky"
    assert result["days"][2]["condition"] == "Slight rain"


def test_weather_far_future_uses_historical_shift():
    assert _years_to_shift_back(date.today() - timedelta(days=30)) == 0
    assert _years_to_shift_back(date.today() + timedelta(days=400)) == 2
