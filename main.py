import logging
import os
from typing import Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field
from supabase import create_client

from graph import (
    trip_graph,
    ask_local_expert,
    generate_packing_list,
    replace_itinerary_activity,
)
from chroma_setup import build_chroma
import chromadb
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None


class TripRequest(BaseModel):
    destination: str
    days: int
    budget: int
    interests: list[str]

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class QuestionRequest(BaseModel):
    question: str
    destination: str | None = None
    itinerary: str | None = None
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class PackingRequest(BaseModel):
    destination: str = Field(min_length=2, max_length=100)
    days: int = Field(ge=1, le=30)
    people: int = Field(ge=1, le=20)
    travel_style: Literal["light", "balanced", "prepared"]
    itinerary: str = Field(min_length=20, max_length=30000)


class ReplaceActivityRequest(BaseModel):
    destination: str = Field(min_length=2, max_length=100)
    itinerary: str = Field(min_length=20, max_length=30000)
    activity: str = Field(min_length=2, max_length=500)
    replacement_preferences: str = Field(default="", max_length=1000)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    try:
        CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name="travel_guides")
        if collection.count() == 0:
            logger.info("ChromaDB empty — seeding now...")
            build_chroma()
        else:
            logger.info(f"ChromaDB already has {collection.count()} documents.")
    except Exception as e:
        logger.error("Failed to seed ChromaDB on startup: %s", e)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/plan-trip")
def plan_trip(request: TripRequest) -> dict[str, object]:
    result = trip_graph.invoke({
        "destination": request.destination,
        "days": request.days,
        "budget": request.budget,
        "interests": request.interests,
    })
    if supabase:
        try:
            supabase.table("trips").insert({
                "destination": request.destination,
                "days": request.days,
                "budget": request.budget,
                "interests": ", ".join(request.interests),
                "final_plan": result["final_plan"],
            }).execute()
        except Exception as e:
            print(f"Supabase insert failed: {e}")
    return {"final_plan": result["final_plan"]}

@app.post("/ask")
def ask(request: QuestionRequest) -> dict[str, str]:
    answer = ask_local_expert(
        question=request.question,
        destination=request.destination,
        itinerary=request.itinerary,
        history=[message.model_dump() for message in request.history],
    )
    return {"answer": answer}


@app.post("/packing-list")
def packing_list(request: PackingRequest) -> dict[str, str]:
    packing_result, weather_summary = generate_packing_list(
        destination=request.destination,
        days=request.days,
        people=request.people,
        travel_style=request.travel_style,
        itinerary=request.itinerary,
    )
    return {
        "packing_list": packing_result,
        "weather_summary": weather_summary,
    }


@app.post("/replace-activity")
def replace_activity(request: ReplaceActivityRequest) -> dict[str, str]:
    replacement = replace_itinerary_activity(
        destination=request.destination,
        itinerary=request.itinerary,
        activity=request.activity,
        replacement_preferences=request.replacement_preferences,
    )
    return replacement
