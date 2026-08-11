from typing import TypedDict, List
import logging
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DIR")

from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import chromadb
import httpx
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class TripState(TypedDict):
    destination: str
    days: int
    budget: int
    interests: List[str]
    retrieved_context: str
    activities: str
    budget_breakdown: str
    itinerary: str
    final_plan: str


# Setup LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# Setup ChromaDB persistent client and collection
client = chromadb.PersistentClient(path=CHROMA_DIR)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-3-small")
collection = client.get_or_create_collection(name="travel_guides", embedding_function=openai_ef)


# Node functions
def destination_node(state: TripState) -> TripState:
    return state


def preferences_node(state: TripState) -> TripState:
    query = f"{state['destination']} {' '.join(state['interests'])}"
    results = collection.query(query_texts=[query], n_results=5,where={"city": state["destination"]})
    texts = []
    for docs in results.get("documents", []):
        if isinstance(docs, list):
            texts.extend(docs)
    state["retrieved_context"] = "\n\n".join(texts)
    return state


def activities_node(state: TripState) -> TripState:
    prompt = (
        "Using the following context:\n{context}\n\n" 
        "Destination: {destination}\nDays: {days}\nInterests: {interests}\n\n"
        "Suggest 5-8 concrete activities (short bullet list)."
    )
    tpl = PromptTemplate(template=prompt, input_variables=["context", "destination", "days", "interests"])
    try:
        out = llm.invoke(tpl.format(context=state["retrieved_context"], destination=state["destination"], days=state["days"], interests=", ".join(state["interests"]))).content
    except Exception:
        # fallback to direct call
        out = llm.invoke(tpl.format(context=state["retrieved_context"], destination=state["destination"], days=state["days"], interests=", ".join(state["interests"]))).content
    state["activities"] = out
    return state


def budget_node(state: TripState) -> TripState:
    prompt = (
        "You are given a total budget of ₹{budget} (Indian Rupees). Split it across stay, food, transport, and activities with approximate amounts and short justification."
    )
    tpl = PromptTemplate(template=prompt, input_variables=["budget"])
    try:
        out = llm.invoke(tpl.format(budget=state["budget"])).content
    except Exception:
        out = llm.invoke(tpl.format(budget=state["budget"])).content
    state["budget_breakdown"] = out
    return state


def itinerary_node(state: TripState) -> TripState:
    prompt = (
        "Using the activities:\n{activities}\n\nAnd the budget breakdown:\n{budget_breakdown}\n\n"
        "Create a day-by-day itinerary for {days} days with Morning/Afternoon/Evening sections."
    )
    tpl = PromptTemplate(template=prompt, input_variables=["activities", "budget_breakdown", "days"])
    try:
        out = llm.invoke(tpl.format(activities=state["activities"], budget_breakdown=state["budget_breakdown"], days=state["days"])).content
    except Exception:
        out = llm.invoke(tpl.format(activities=state["activities"], budget_breakdown=state["budget_breakdown"], days=state["days"])).content
    state["itinerary"] = out
    return state


def final_plan_node(state: TripState) -> TripState:
    md = f"# Trip Plan for {state['destination']}\n\n"
    md += f"## Context\n{state['retrieved_context']}\n\n"
    md += f"## Activities\n{state['activities']}\n\n"
    md += f"## Budget Breakdown\n{state['budget_breakdown']}\n\n"
    md += f"## Itinerary\n{state['itinerary']}\n"
    state["final_plan"] = md
    return state


from langgraph.graph import StateGraph, END

def build_graph():
    graph = StateGraph(TripState)

    graph.add_node("destination", destination_node)
    graph.add_node("preferences", preferences_node)
    graph.add_node("activities", activities_node)
    graph.add_node("budget", budget_node)
    graph.add_node("itinerary", itinerary_node)
    graph.add_node("final_plan", final_plan_node)

    graph.set_entry_point("destination")
    graph.add_edge("destination", "preferences")
    graph.add_edge("preferences", "activities")
    graph.add_edge("activities", "budget")
    graph.add_edge("budget", "itinerary")
    graph.add_edge("itinerary", "final_plan")
    graph.add_edge("final_plan", END)

    return graph.compile()

trip_graph = build_graph()
@tool
def search_local_guide(query: str, destination: str = "") -> str:
    """Search the local travel knowledge base for facts about attractions, food, prices, and activities."""
    query_args = {"query_texts": [query], "n_results": 5}
    if destination:
        query_args["where"] = {"city": destination}
    results = collection.query(**query_args)
    texts = []
    for docs in results.get("documents", []):
        if isinstance(docs, list):
            texts.extend(docs)
    return "\n\n".join(texts) if texts else "No matching local data found."

local_expert_agent = create_react_agent(llm, tools=[search_local_guide])

def ask_local_expert(
    question: str,
    destination: str | None = None,
    itinerary: str | None = None,
    history: List[Dict[str, str]] | None = None,
) -> str:
    instructions = [
        "Answer the user's travel question accurately and concisely.",
        "Use the generated itinerary when it is relevant.",
    ]
    if destination:
        instructions.append(
            f"The destination is {destination}. When using search_local_guide, "
            f"pass destination={destination!r} and do not recommend places from other cities."
        )

    current_message = []
    if itinerary:
        current_message.append(f"Generated itinerary:\n{itinerary}")
    current_message.append(f"User question:\n{question}")

    messages = [{"role": "system", "content": "\n".join(instructions)}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": "\n\n".join(current_message)})

    result = local_expert_agent.invoke({"messages": messages})
    return result["messages"][-1].content


def get_destination_weather(destination: str, days: int) -> str:
    """Return a compact current forecast summary for packing guidance."""
    weather_location_aliases = {
        "Bangalore": "Bengaluru",
        "Mysore": "Mysuru",
        "Ooty": "Udhagamandalam",
    }
    location_query = weather_location_aliases.get(destination, destination)
    try:
        with httpx.Client(timeout=10.0) as weather_client:
            location_response = weather_client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={
                    "name": location_query,
                    "count": 1,
                    "language": "en",
                    "format": "json",
                    "countryCode": "IN",
                },
            )
            location_response.raise_for_status()
            locations = location_response.json().get("results", [])
            if not locations:
                return "Current forecast unavailable for this destination."

            location = locations[0]
            forecast_response = weather_client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "daily": (
                        "temperature_2m_max,temperature_2m_min,"
                        "precipitation_probability_max,weather_code"
                    ),
                    "forecast_days": min(max(days, 1), 16),
                    "timezone": "auto",
                },
            )
            forecast_response.raise_for_status()
            daily = forecast_response.json().get("daily", {})

        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        rain = daily.get("precipitation_probability_max", [])
        if not dates or not highs or not lows:
            return "Current forecast unavailable for this destination."

        return (
            f"Current {len(dates)}-day forecast for {location['name']}: "
            f"highs {min(highs):.0f}-{max(highs):.0f}°C, "
            f"lows {min(lows):.0f}-{max(lows):.0f}°C, "
            f"maximum rain probability {max(rain or [0]):.0f}%."
        )
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Weather lookup failed for %s: %s", destination, exc)
        return "Current forecast unavailable; pack for typical destination conditions."


def generate_packing_list(
    destination: str,
    days: int,
    people: int,
    travel_style: str,
    itinerary: str,
) -> tuple[str, str]:
    weather_summary = get_destination_weather(destination, days)
    prompt = f"""
You are an expert travel packing assistant. Create a practical packing checklist.

Destination: {destination}
Trip duration: {days} days
Number of travellers: {people}
Travel style: {travel_style}
Current weather signal: {weather_summary}
Confirmed itinerary:
{itinerary}

Use Markdown checkboxes in every list. Organize the answer under these headings:
Weather snapshot, Essentials, Clothing, Activity-specific gear, Toiletries and health,
Documents and money, Electronics, Shared items, and Before you leave.
Give realistic quantities, clearly distinguish per-person items from shared group items,
adapt recommendations to the activities and travel style, and avoid unnecessary items.
Mention that the weather is a current forecast and should be rechecked before departure.
"""
    packing_list = llm.invoke(prompt).content
    return packing_list, weather_summary


if __name__ == "__main__":
    sample: TripState = {
        "destination": "Bangalore",
        "days": 4,
        "budget": 1200,
        "interests": ["food", "culture", "nature"],
        "retrieved_context": "",
        "activities": "",
        "budget_breakdown": "",
        "itinerary": "",
        "final_plan": "",
    }
    out = trip_graph.invoke(sample)
    print(out["final_plan"]) 
