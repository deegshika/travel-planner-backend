import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client

from graph import trip_graph
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

logger = logging.getLogger(__name__)

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None


class TripRequest(BaseModel):
    destination: str
    days: int
    budget: int
    interests: list[str]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
