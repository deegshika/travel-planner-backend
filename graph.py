from typing import TypedDict, List
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
from chromadb.config import Settings
from chromadb.utils import embedding_functions


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
def search_local_guide(query: str) -> str:
    """Search the local travel knowledge base for facts about attractions, food, prices, and activities."""
    results = collection.query(query_texts=[query], n_results=5)
    texts = []
    for docs in results.get("documents", []):
        if isinstance(docs, list):
            texts.extend(docs)
    return "\n\n".join(texts) if texts else "No matching local data found."

local_expert_agent = create_react_agent(llm, tools=[search_local_guide])

def ask_local_expert(question: str) -> str:
    result = local_expert_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


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
