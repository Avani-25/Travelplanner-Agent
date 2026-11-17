"""Convert the agent's answer into a TripPlan, validate it and render it as readable text."""

import logging
import re
from datetime import datetime

from src.config import FLIGHTS_FILE, HOTELS_FILE, PLACES_FILE
from src.data_loader import load_json
from src.schemas import TripPlan

logger = logging.getLogger(__name__)

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

# Matches 24-hour times such as 07:40 or 18:45 (skips times already followed by AM/PM)
TIME_PATTERN = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?!\s?[AaPp][Mm])")


# --------------------------------------------------------------------------
# Time formatting helpers
# --------------------------------------------------------------------------
def to_12h(time_str: str) -> str:
    """Convert '18:45' to '6:45 PM'. Returns the input unchanged if it cannot be parsed."""
    try:
        parsed = datetime.strptime(time_str.strip(), "%H:%M")
    except (ValueError, AttributeError):
        return str(time_str)
    return parsed.strftime("%I:%M %p").lstrip("0")


def convert_times(text: str) -> str:
    """Convert every 24-hour time inside free text to 12-hour AM/PM format."""
    return TIME_PATTERN.sub(lambda match: to_12h(match.group(0)), str(text))


def _inr(amount: float) -> str:
    """Format an amount in Indian rupees."""
    return f"₹{amount:,.0f}"


# --------------------------------------------------------------------------
# Validation against the datasets
# --------------------------------------------------------------------------
def validate_plan(plan: TripPlan) -> None:
    """Cross-check the plan against the datasets and drop anything not found there."""
    if plan.status != "complete":
        return

    flights = {f["flight_id"]: f for f in load_json(FLIGHTS_FILE)}
    for flight in (plan.outbound_flight, plan.return_flight):
        if flight and (
            flight.flight_id not in flights
            or flights[flight.flight_id]["price"] != flight.price
        ):
            plan.warnings.append(
                f"Flight {flight.flight_id} could not be verified in the dataset."
            )

    valid_places = {
        p["name"] for p in load_json(PLACES_FILE) if p["city"] == plan.destination
    }
    for day in plan.itinerary:
        unknown = [p for p in day.places if p not in valid_places]
        if unknown:
            day.places = [p for p in day.places if p in valid_places]
            plan.warnings.append(
                f"Removed places not found in the dataset: {', '.join(unknown)}"
            )

    if plan.hotel:
        hotel_ok = any(
            h["name"] == plan.hotel.name
            and h["city"] == plan.destination
            and h["price_per_night"] == plan.hotel.price_per_night
            for h in load_json(HOTELS_FILE)
        )
        if not hotel_ok:
            plan.warnings.append(
                f"Hotel '{plan.hotel.name}' could not be verified in the dataset."
            )


def _fix_budget_total(plan: TripPlan) -> None:
    """Recompute the total in Python so the arithmetic is always correct."""
    if plan.budget:
        b = plan.budget
        b.total = b.flight + b.hotel + b.food_and_local_travel


# --------------------------------------------------------------------------
# Structured extraction
# --------------------------------------------------------------------------
def extract_trip_plan(llm, answer_text: str) -> TripPlan:
    """Ask the LLM to fill the TripPlan schema from the agent's written answer."""
    last_error = None
    for method in ("json_schema", "function_calling"):
        try:
            structured_llm = llm.with_structured_output(TripPlan, method=method)
            plan = structured_llm.invoke(
                [("system", EXTRACTION_PROMPT), ("human", answer_text)]
            )
            if isinstance(plan, dict):
                plan = TripPlan(**plan)
            _fix_budget_total(plan)
            validate_plan(plan)
            return plan
        except Exception as exc:  # try the next method
            logger.warning("Structured output via %s failed: %s", method, exc)
            last_error = exc
    raise RuntimeError(f"Could not build a structured plan: {last_error}")


# --------------------------------------------------------------------------
# Human-readable rendering
# --------------------------------------------------------------------------
def render_text(plan: TripPlan) -> str:
    """Render a TripPlan as clean human-readable text (times in 12-hour AM/PM format)."""
    if plan.status != "complete":
        return convert_times(plan.message or "The trip could not be planned.")

    lines = [
        f"Your {plan.num_days}-Day Trip to {plan.destination} "
        f"({plan.start_date} to {plan.end_date})",
        "",
    ]
    if plan.summary:
        lines += [convert_times(plan.summary), ""]

    lines.append("Flights:")
    for label, flight in (
        ("Outbound", plan.outbound_flight),
        ("Return", plan.return_flight),
    ):
        if flight:
            lines.append(
                f"- {label}: {flight.airline} {flight.flight_id} ({_inr(flight.price)}) - "
                f"{flight.from_city} {to_12h(flight.departure_time)} -> "
                f"{flight.to_city} {to_12h(flight.arrival_time)}"
            )
        else:
            lines.append(f"- {label}: not available in the dataset")

    if plan.hotel:
        h = plan.hotel
        amenities = ", ".join(h.amenities) if h.amenities else "n/a"
        lines += [
            "",
            "Hotel:",
            f"- {h.name} ({h.stars}-star, {_inr(h.price_per_night)}/night)",
            f"  Amenities: {amenities}",
        ]

    lines += ["", "Itinerary and Weather:"]
    for day in plan.itinerary:
        places = ", ".join(day.places) if day.places else "Free day"
        lines.append(f"Day {day.day} ({day.date}): {places}")
        if day.weather:
            lines.append(f"  Weather: {day.weather}")
        if day.notes:
            lines.append(f"  Note: {convert_times(day.notes)}")

    if plan.weather_note:
        lines += ["", f"Weather note: {convert_times(plan.weather_note)}"]

    if plan.budget:
        b = plan.budget
        lines += [
            "",
            "Estimated Total Budget:",
            f"- Flights: {_inr(b.flight)}",
            f"- Hotel: {_inr(b.hotel)}",
            f"- Food & Local Travel: {_inr(b.food_and_local_travel)}",
            "-------------------------------------",
            f"Total Cost: {_inr(b.total)}",
        ]

    if plan.reasoning:
        lines += ["", "Why we selected this:"]
        lines += [f"- {convert_times(r)}" for r in plan.reasoning]
    if plan.warnings:
        lines += ["", "Please note:"]
        lines += [f"- {convert_times(w)}" for w in plan.warnings]

    return "\n".join(lines)
