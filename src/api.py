from contextlib import asynccontextmanager
from typing import Union

import pandas as pd
import uvicorn
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import search_engine
from Utils import search, create_search_index
from fastapi import FastAPI


# --- Pydantic Models for API ---
# These models define the structure of your API's
# inputs (Request) and outputs (Response) and
# provide automatic data validation.

class SearchQuery(BaseModel):
    query: str = Field(..., description="The natural language query to search for.")
    top_k: int = Field(10, description="The number of results to return.", gt=0, le=20)


class SearchResult(BaseModel):
    filename: str
    content_snippet: str


class SearchResponse(BaseModel):
    results: list[SearchResult]


# --- Global Variables ---
# These variables will be populated on startup and
# used by the /search endpoint.
model = None
index = None
all_documents = None
index_to_filename = None


# --- FastAPI Startup Event ---
# We use the 'lifespan' context manager to load the
# model and index when the server starts.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ONCE on server startup
    global model, index, all_documents, index_to_filename

    print("Server starting up...")
    print("Loading fine-tuned model...")
    # Load the model you trained (from your TEST mode)
    model = SentenceTransformer('../cosqa-finetuned-model')

    print("Loading validation data for indexing...")
    val_raw = pd.read_json("hf://datasets/gonglinyuan/CoSQA/cosqa-dev.json")

    print("Creating search index...")
    # This calls your helper function
    index, all_documents, index_to_filename = create_search_index(val_raw, model)

    print("--- Search Engine Ready ---")

    yield

    # This code runs on server shutdown (optional)
    print("Server shutting down...")
    model = None
    index = None


# --- Initialize FastAPI App ---
app = FastAPI(
    title="Code Search API",
    description="An API for semantic search over the CoSQA dataset.",
    version="1.0",
    lifespan=lifespan  # This links the startup/shutdown logic
)

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(query: SearchQuery):
    """
    Accepts a text query and returns the top_k most
    relevant code snippets from the indexed collection.
    """
    search_results = search(query.query, index, model, index_to_filename, all_documents)
    return {"results": search_results}

# --- Run the Server ---
# This block allows you to run the server by
# executing `python main_api.py` in your terminal.
if __name__ == "__main__":
    print("Starting Uvicorn server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)