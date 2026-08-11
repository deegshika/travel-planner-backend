import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from supabase import create_client

from graph import trip_graph, ask_local_expert
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

class QuestionRequest(BaseModel):
    question: str


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
    answer = ask_local_expert(request.question)
    return {"answer": answer}
