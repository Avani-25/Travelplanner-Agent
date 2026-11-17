"""Pydantic schemas describing the structured trip plan."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


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
    weather: str = Field(
        default="", description="Condition and temperature range for the day"
    )
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
            "Caveats: estimated weather, missing return flight, "
            "requested place type unavailable"
        ),
    )
