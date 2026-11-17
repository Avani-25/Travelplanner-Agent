"""Builds the LangChain travel-planning agent (Gemini + tools)."""

import logging
import os
from datetime import date

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from src.config import BASE_DIR
from src.tools import ALL_TOOLS

load_dotenv(BASE_DIR / ".env", override=True)
logger = logging.getLogger(__name__)

# Override with GEMINI_MODEL in .env if this name is unavailable for your key
DEFAULT_MODEL = "gemini-3.8-flash"


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
- If no flight exists on a route, say so clearly and mention the alternative
  destinations the tool returned. Do not make up a flight.
- If no return flight exists, say so and state that the budget covers the outbound flight only.
- If the weather source is "historical_estimate", tell the user it is an estimate.
- Each city has few places. Spread them across the days (1-2 per day) and do not repeat a place.
  If there are more days than places, label the extra days as relaxed or free days.
- If a budget is given and the estimated total exceeds it, say so and suggest cheaper choices.
- For every key decision (flight, hotel), explain briefly WHY you selected it.
- Use ONLY places returned by discover_places. NEVER add attractions, beaches, markets or
  restaurants from your own knowledge. If the user's preferred type (for example "beach")
  is not available in the city, say so clearly and use the highest-rated places that ARE available.
- Itinerary items may only be: the returned places, plus generic items such as travel,
  hotel check-in or check-out, and free time. Do not invent specific meals or activities.
- Only say a trip is within budget if the user gave a budget.
- For free time, write only "Free time" or "Free time at the hotel". Do not name walks,
  promenades, beaches, restaurants or any activity that is not a returned place.
- Mention weather as an estimate ONLY if the tool source is "historical_estimate".
  A "live_forecast" is a real forecast, not an estimate.
- If the route has no flight, state plainly that no flight exists from the source to the
  destination, list the reachable destinations the tool returned, then ask what the user wants.

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
    """Create the chat model. Provider is chosen by LLM_PROVIDER in .env (groq or gemini)."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                f"GROQ_API_KEY not found. Add it to {BASE_DIR / '.env'} and restart the kernel."
            )
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            api_key=api_key,
        )

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                f"GOOGLE_API_KEY not found. Add it to {BASE_DIR / '.env'} and restart the kernel."
            )
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
            api_key=api_key,
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'groq' or 'gemini'.")


def build_agent():
    """Create and return the tool-calling travel agent."""
    return create_agent(
        model=get_llm(),
        tools=ALL_TOOLS,
        system_prompt=get_system_prompt(),
    )


def extract_text(content) -> str:
    """Gemini may return a string or a list of content blocks; return plain text."""
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
    """Run the agent on a user query.

    Returns:
        {"answer": final text, "steps": [{"tool": name, "args": {...}}, ...], "error": optional}
    """
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
