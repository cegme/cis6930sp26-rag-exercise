"""FastAPI server for the collaborative RAG + RAGAS demo.

Manages an in-memory ChromaDB collection, handles document ingestion,
retrieval, answer generation, and RAGAS evaluation. The server holds
the NavigatorAI API key so students do not need their own.
"""

import os
from contextlib import asynccontextmanager

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from loguru import logger
from pydantic import BaseModel

from evaluation import run_ragas_evaluation
from rag import add_document, generate_answer, get_llm, get_stats, init_collection, retrieve

load_dotenv()

RESET_TOKEN = os.getenv("RESET_TOKEN", "instructor-reset")


class DocumentInput(BaseModel):
    content: str
    contributor: str = "anonymous"
    source: str = "manual"


class QueryInput(BaseModel):
    question: str
    k: int = 5


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ChromaDB and LLM on startup."""
    api_key = os.getenv("NAVIGATOR_API_KEY")
    if not api_key:
        logger.error("NAVIGATOR_API_KEY not set")
        raise RuntimeError("NAVIGATOR_API_KEY environment variable is required")

    client = chromadb.Client()
    collection = init_collection(client)
    llm = get_llm(api_key)

    app.state.collection = collection
    app.state.llm = llm
    app.state.chroma_client = client

    logger.info("Server ready")
    yield
    logger.info("Server shutting down")


app = FastAPI(title="Collaborative RAG Demo", lifespan=lifespan)


@app.post("/documents")
async def add_doc(doc: DocumentInput):
    """Add a text document to the knowledge base."""
    if not doc.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    chunks_added = add_document(
        app.state.collection, doc.content, doc.contributor, doc.source
    )
    return {
        "status": "ok",
        "chunks_added": chunks_added,
        "contributor": doc.contributor,
        "source": doc.source,
    }


@app.get("/documents")
async def list_docs(contributor: str | None = Query(default=None)):
    """List documents with metadata, optionally filtered by contributor."""
    collection = app.state.collection
    if collection.count() == 0:
        return {"documents": [], "total": 0}

    all_data = collection.get(include=["documents", "metadatas"])

    docs = []
    for doc_text, meta in zip(all_data["documents"], all_data["metadatas"]):
        if contributor and meta.get("contributor") != contributor:
            continue
        docs.append({"text": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text, "metadata": meta})

    return {"documents": docs, "total": len(docs)}


@app.post("/query")
async def query_docs(q: QueryInput):
    """Retrieve relevant chunks and generate an answer."""
    results = retrieve(app.state.collection, q.question, k=q.k)
    retrieved_docs = results["documents"][0] if results["documents"][0] else []
    metadatas = results["metadatas"][0] if results["metadatas"][0] else []
    distances = results["distances"][0] if results["distances"][0] else []

    answer = generate_answer(q.question, retrieved_docs, app.state.llm)

    context_list = []
    for doc_text, meta, dist in zip(retrieved_docs, metadatas, distances):
        context_list.append({
            "text": doc_text,
            "contributor": meta.get("contributor", "unknown"),
            "source": meta.get("source", "unknown"),
            "distance": round(dist, 4),
        })

    return {
        "question": q.question,
        "answer": answer,
        "contexts": context_list,
        "num_contexts": len(context_list),
    }


@app.get("/stats")
async def stats():
    """Get collection statistics."""
    return get_stats(app.state.collection)


@app.post("/evaluate")
async def evaluate_rag():
    """Run RAGAS evaluation on pre-defined test questions."""
    logger.info("Starting RAGAS evaluation...")
    result = run_ragas_evaluation(app.state.collection, app.state.llm)
    return result


@app.post("/reset")
async def reset_collection(token: str = Query(...)):
    """Clear the collection. Requires instructor token."""
    if token != RESET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid reset token")

    client = app.state.chroma_client
    client.delete_collection("collaborative_rag")
    app.state.collection = init_collection(client)
    logger.warning("Collection reset by instructor")
    return {"status": "ok", "message": "Collection cleared"}
