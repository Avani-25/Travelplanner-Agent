# **Travelplanner-Agent**
# ✈**Agentic AI-Based Travel Planning Assistant (LangChain)**

An AI travel agent that plans multi-day trips between Indian cities. It reasons over
flight, hotel and attraction datasets plus live weather, then produces a day-wise
itinerary with a budget and an explanation of its choices.

## ☂**Problem Statement**
Planning a trip means comparing flights, hotels, attractions, weather and cost across
many websites. This project automates that with a tool-using LLM agent that works like
a travel expert and justifies its decisions.

## 💻**Business Use Case**
Travel agencies, hotel platforms and flight aggregators can use agentic assistants to
reduce support workload, personalise recommendations, automate itinerary design and
save customers time and money.

## 📌**Features**
- Autonomous LangChain agent with multi-step reasoning (tool calling)
- Five tools: flight search, hotel recommendation, places discovery, weather, budget
- Filtering and ranking: cheapest/fastest flight, best-value/highest-rated hotel,
  top-rated places by type
- Live weather from the free Open-Meteo API, with a historical-data fallback for
  dates beyond the forecast window (clearly labelled as an estimate)
- Structured output (validated Pydantic schema) as JSON and as readable text
- Post-validation against the datasets so the plan contains no invented data
- Streamlit interface with interactive filters

## 🧱**Architecture**
```
User request (Streamlit UI)
        |
        v
LangChain agent (LLM + system prompt)
        |  decides which tools to call, in order
        v
search_flights -> recommend_hotels -> discover_places -> get_weather -> estimate_budget
        |
        v
Agent's written plan --> structured extraction (TripPlan schema)
        |
        v
Validation against datasets --> JSON + readable itinerary
```

## 📐**Project Structure**
```
travel-planner-agent/
├── app.py                 # Streamlit interface
├── data/                  # flights.json, hotels.json, places.json
├── notebooks/             # 01 data exploration, 02 tools, 03 agent, 04 final output
├── src/
│   ├── config.py          # paths, city coordinates, constants
│   ├── data_loader.py     # JSON loading, city normalisation
│   ├── tools.py           # the five LangChain tools
│   ├── agent.py           # LLM setup, system prompt, agent runner
│   ├── schemas.py         # Pydantic TripPlan schema
│   ├── output.py          # structured extraction, validation, text rendering
│   └── planner.py         # end-to-end pipeline
├── tests/test_tools.py    # pytest unit tests
├── requirements.txt
├── .env.example
└── README.md
```

## ⚙**Setup**
1. Clone the repository and open the folder in VS Code.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your own API key (free key from
   https://console.groq.com or https://aistudio.google.com/apikey).
5. Place `flights.json`, `hotels.json` and `places.json` in `data/`.

## **Run**
```bash
streamlit run app.py
```
Open http://localhost:8501, choose your trip details in the sidebar and click
"Plan my trip". Run the tests with `pytest -v`.

## 👀**Example Output**
```
Your 3-Day Trip to Goa (2026-11-20 to 2026-11-22)

Flights:
- Outbound: Go First FL0021 (₹5,356) - Bangalore 07:27 -> Goa 09:27
- Return: SpiceJet FL0023 (₹4,537) - Goa 17:10 -> Bangalore 20:10
Hotel:
- Royal Heritage (5-star, ₹1,232/night)
Estimated Total Budget:
- Flights: ₹9,893 | Hotel: ₹2,464 | Food & Local Travel: ₹2,400
Total Cost: ₹14,757
```

## 🖨**Design Decisions**
- **Datasets:** the flight data has about one flight per route and fixed 2025 dates, so
  flights are matched by route and the time of day is shown. Many city pairs have no
  flight, and the agent says so instead of inventing one.
- **Places:** each city has only five places, so attractions are spread across days and
  extra days are marked as free time. Place names are synthetic, so filtering uses the
  `type` field.
- **Weather:** the live forecast is used within about 15 days; otherwise the same
  dates from a past year are used and labelled as an estimate.
- **Budget:** total = flights + hotel (nights = days - 1) + daily food and local travel
  (default ₹800/day), calculated in Python rather than by the LLM.
- **No SQL database:** the project uses the provided JSON datasets only.

## 📎**Limitations and Future Work**
- Datasets are small and synthetic; swap in real flight and hotel APIs.
- Add multi-traveller pricing, train options and map visualisation.
- Add user-specific preference memory across sessions.

## 🛠**Tech Stack**
Python, LangChain, Groq / Gemini LLMs, Open-Meteo API, Pydantic, Streamlit, pytest.
