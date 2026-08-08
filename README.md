# Waypoint — AI Travel Planner

An AI-powered trip itinerary generator that turns a destination, trip length, budget, and interests into a full day-by-day plan — grounded in real local data, not just LLM guesswork.

**Live demo:**
- Frontend: https://travel-planner-frontend.vercel.app
- Backend API docs: https://travel-planner-backend-nlwx.onrender.com/docs

---

## Problem Statement

Planning a trip means digging through dozens of sources — blogs, review sites, forums — to piece together where to go, what to eat, and how to budget it all. Waypoint collapses that into one step: tell it a destination, a number of days, a budget, and a few interests, and it generates a complete itinerary with real local recommendations and a realistic cost breakdown.

## How It Works (RAG Architecture)

This isn't a single prompt to an LLM — it's a **Retrieval-Augmented Generation (RAG)** pipeline, meaning the AI's output is grounded in a curated knowledge base rather than relying purely on the model's own (sometimes inaccurate) general knowledge.

1. **Retrieval** — When a request comes in, the system searches a **ChromaDB** vector database of 98 hand-seeded entries covering real attractions, food spots, and activities across 9 Indian cities (Bangalore, Jaipur, Mumbai, Delhi, Goa, Kodaikanal, Ooty, Mysore, Shimla), each with realistic ₹ price ranges.
2. **Generation** — The retrieved facts are passed as context to **OpenAI's gpt-4o-mini**, which writes the actual activity suggestions, budget breakdown, and day-by-day itinerary — reasoning over the real data rather than inventing places or prices from scratch.
3. **Orchestration** — The whole process is modeled as a **LangGraph** state machine with six sequential nodes:

```
destination → preferences → activities → budget → itinerary → final_plan
```

Each node reads and updates a shared state object, making the pipeline easy to trace, debug, and extend (e.g. adding a "weather" or "flights" node later).

4. **Observability** — Every run is traced end-to-end in **LangSmith**, showing each node's input/output, latency, and token cost — useful both for debugging and for demonstrating exactly how the AI reasoned through a request.
5. **Persistence** — Every generated trip is saved to a **Supabase** Postgres table, so there's a permanent record of every plan the system has produced.

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| Observability | LangSmith |
| LLM | OpenAI (gpt-4o-mini) |
| Vector store | ChromaDB |
| Backend API | FastAPI (Python) |
| Database | Supabase (Postgres) |
| Frontend | Static HTML/CSS/JS |
| Backend hosting | Render |
| Frontend hosting | Vercel |

## Project Structure

```
travel-planner-backend/
├── main.py            # FastAPI app, /plan-trip endpoint, Supabase integration
├── graph.py            # LangGraph pipeline: nodes, state, compiled graph
├── chroma_setup.py     # Seeds ChromaDB with destination knowledge base
├── requirements.txt
└── render.yaml

travel-planner-frontend/
└── index.html           # Single-file frontend, calls the backend API
```

## Running Locally

**Backend:**
```bash
cd travel-planner-backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
# create a .env file with OPENAI_API_KEY, LANGCHAIN_API_KEY,
# LANGCHAIN_TRACING_V2=true, LANGCHAIN_PROJECT, SUPABASE_URL, SUPABASE_KEY
python chroma_setup.py           # seed the vector store
uvicorn main:app --reload
```

**Frontend:**
Open `index.html` directly in a browser, or serve it with any static file server. Update the `API_URL` constant near the bottom of the file to point at your backend.

## API

**POST** `/plan-trip`
```json
{
  "destination": "Jaipur",
  "days": 3,
  "budget": 8000,
  "interests": ["food", "photography"]
}
```
Returns a markdown-formatted itinerary as `{ "final_plan": "..." }`.

## Notes for Evaluators

- The knowledge base currently covers 9 Indian cities. Cities outside this set will fall back to the LLM's general knowledge, with less grounding and a higher chance of minor factual inaccuracy.
- Row Level Security is disabled on the Supabase `trips` table for demo simplicity. In a production deployment, this would be replaced with scoped policies (e.g. per-user access via Supabase Auth).
- The backend is hosted on Render's free tier, which spins down after inactivity — the first request after idle time may take 30-60 seconds to respond while the server wakes up.
