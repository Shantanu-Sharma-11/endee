"""
FastAPI backend for the Campus AI Assistant.
Serves the chat API and frontend static files.
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# make sure imports work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.rag_pipeline import RAGPipeline, QueryResult

# config
DATA_DIR = os.path.join(ROOT, "data")
FRONTEND_DIR = os.path.join(ROOT, "frontend")
ENDEE_URL = os.environ.get("ENDEE_URL", "http://localhost:8080/api/v1")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("campus_ai")

pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    log.info("Starting Campus AI Assistant...")
    log.info("  Data dir: %s", DATA_DIR)
    log.info("  Endee:    %s", ENDEE_URL)

    pipeline = RAGPipeline(data_dir=DATA_DIR, endee_url=ENDEE_URL)
    # run in thread so we don't block the event loop
    await asyncio.to_thread(pipeline.initialize)

    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Campus AI Assistant",
    description="RAG-powered assistant for Galgotias University using Endee vector DB",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    if request.url.path.startswith("/api/"):
        log.info("%s %s -> %d (%.0fms)", request.method, request.url.path,
                 response.status_code, ms)
    return response


# --- request/response models ---

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    num_chunks_retrieved: int
    similarities: List[float] = []


# --- routes ---

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None and pipeline.is_ready,
    }


@app.get("/api/stats")
async def stats():
    if not pipeline:
        raise HTTPException(503, "Pipeline not ready")
    return pipeline.stats()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not pipeline or not pipeline.is_ready:
        raise HTTPException(503, "Pipeline not ready yet")

    result = await asyncio.to_thread(pipeline.query, req.question.strip())

    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        num_chunks_retrieved=result.num_chunks_retrieved,
        similarities=result.similarities,
    )


# --- frontend ---

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def index():
    path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "Frontend not found"}


if __name__ == "__main__":
    import uvicorn
    log.info("Server: http://localhost:%d", PORT)
    uvicorn.run("backend.main:app", host=HOST, port=PORT,
                reload=False, log_level=LOG_LEVEL.lower())
