"""Streamlit dashboard for the Agentic AI Travel Planning Assistant (light theme)."""

import html
import re
from datetime import date, datetime, timedelta

import streamlit as st

from src.agent import build_agent, get_llm
from src.config import CITY_COORDINATES
from src.planner import plan_trip

CITIES = sorted(CITY_COORDINATES)
PLACE_TYPES = [
    "beach",
    "fort",
    "lake",
    "market",
    "monument",
    "museum",
    "park",
    "temple",
]
TOOL_ICONS = {
    "search_flights": "✈️",
    "recommend_hotels": "🏨",
    "discover_places": "📍",
    "get_weather": "🌦️",
    "estimate_budget": "💰",
}

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Matches 24-hour times such as 07:40 or 18:45 (skips times already followed by AM/PM)
TIME_PATTERN = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?!\s?[AaPp][Mm])")


# ----------------------------- helpers -----------------------------
def esc(value) -> str:
    """HTML-escape any value (the LLM text is never trusted as markup)."""
    return html.escape(str(value))


def show(markup: str) -> None:
    """Render an HTML snippet (indentation removed so Markdown does not treat it as code)."""
    st.markdown(
        " ".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True
    )


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


def fmt_times(text: str) -> str:
    """Convert every 24-hour time inside free text to 12-hour AM/PM format."""

    def replace(match: re.Match) -> str:
        clock, suffix = time_parts(match.group(0))
        return f"{clock} {suffix}".strip()

    return TIME_PATTERN.sub(replace, str(text))


def weather_icon(text: str) -> str:
    """Pick an emoji that matches a weather description."""
    t = text.lower()
    rules = [
        ("thunder", "⛈️"),
        ("rain", "🌧️"),
        ("drizzle", "🌦️"),
        ("shower", "🌧️"),
        ("snow", "❄️"),
        ("fog", "🌫️"),
        ("overcast", "☁️"),
        ("partly", "⛅"),
        ("mainly clear", "🌤️"),
        ("clear", "☀️"),
    ]
    for keyword, icon in rules:
        if keyword in t:
            return icon
    return "🌡️"


def build_query(
    source, destination, start_date, num_days, min_stars, place_types, budget
):
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
    return get_llm(), build_agent()


# ----------------------------- components -----------------------------
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


def render_trip_header(plan) -> None:
    show(f"""
    <div class="trip-head">
      <span class="chip">{esc(plan.source)} → {esc(plan.destination)}</span>
      <h2>{plan.num_days}-Day Trip to {esc(plan.destination)}</h2>
      <div class="dates">📅 {esc(plan.start_date)} to {esc(plan.end_date)}</div>
      <div class="summary">{esc(fmt_times(plan.summary))}</div>
    </div>
    """)


def render_stats(plan) -> None:
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


def render_flights(plan) -> None:
    st.markdown('<div class="section-title">✈️ Flights</div>', unsafe_allow_html=True)
    show(
        f'<div class="grid-2">{flight_card("Outbound", plan.outbound_flight)}'
        f'{flight_card("Return", plan.return_flight)}</div>'
    )


def render_hotel(plan) -> None:
    if not plan.hotel:
        return
    h = plan.hotel
    amenities = "".join(f'<span class="pill soft">{esc(a)}</span>' for a in h.amenities)
    st.markdown(
        '<div class="section-title">🏨 Where you will stay</div>',
        unsafe_allow_html=True,
    )
    show(f"""
    <div class="card">
      <div class="hotel-name">{esc(h.name)}</div>
      <div class="hotel-sub">{"⭐" * int(h.stars)} &nbsp;·&nbsp; <span class="price">{inr(h.price_per_night)}</span> per night</div>
      {amenities}
    </div>
    """)


def render_itinerary(plan) -> None:
    st.markdown(
        '<div class="section-title">🗓️ Day-wise itinerary</div>', unsafe_allow_html=True
    )
    blocks = []
    for day in plan.itinerary:
        places = "".join(
            f'<span class="pill">📍 {esc(p)}</span>' for p in day.places
        ) or ('<span class="pill soft">Free day</span>')
        weather = (
            f'<span class="weather">{weather_icon(day.weather)} {esc(day.weather)}</span>'
            if day.weather
            else ""
        )
        note = (
            f'<div class="note">{esc(fmt_times(day.notes))}</div>' if day.notes else ""
        )
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


def render_budget(plan) -> None:
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
    st.markdown(
        '<div class="section-title">💰 Budget breakdown</div>', unsafe_allow_html=True
    )
    show(f"""
    <div class="card">
      <div class="bar">{bar}</div>
      {legend}
      <div class="legend-row"><strong>Total</strong><strong class="price">{inr(b.total)}</strong></div>
    </div>
    """)


def render_reasoning(plan) -> None:
    if plan.reasoning:
        st.markdown(
            '<div class="section-title">🧠 Why we selected this</div>',
            unsafe_allow_html=True,
        )
        items = "".join(
            f'<div class="reason">• {esc(fmt_times(r))}</div>' for r in plan.reasoning
        )
        show(f'<div class="card">{items}</div>')
    for warning in plan.warnings:
        show(f'<div class="note-box">⚠️ {esc(fmt_times(warning))}</div>')


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
        st.info(fmt_times(plan.message))
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


# ----------------------------- sidebar -----------------------------
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
    budget = st.number_input(
        "Total budget in INR (0 = no limit)", min_value=0, step=1000, value=0
    )
    submitted = st.button("✨ Plan my trip", type="primary", use_container_width=True)

# ----------------------------- main flow -----------------------------
if submitted:
    if source == destination:
        st.warning("Please choose different source and destination cities.")
        st.stop()

    try:
        llm, agent = load_agent()
    except Exception as exc:
        st.error(f"Could not start the AI model: {exc}")
        st.stop()

    query = build_query(
        source, destination, start_date, num_days, min_stars, place_types, budget
    )
    with st.spinner("✈️ The agent is searching flights, hotels, places and weather..."):
        st.session_state["result"] = plan_trip(agent, llm, query)

result = st.session_state.get("result")
if result is None:
    render_hero()
    render_features()
else:
    render_result(result)
