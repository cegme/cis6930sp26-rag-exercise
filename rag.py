"""Core RAG engine for the collaborative demo.

Handles document chunking, ChromaDB storage, retrieval, and LLM generation
using NavigatorAI (gpt-4o-mini).
"""

import re
import uuid
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions
from langchain_openai import ChatOpenAI
from loguru import logger

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def init_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Create or get the ChromaDB collection with sentence-transformer embeddings.

    Args:
        client: ChromaDB client instance.

    Returns:
        ChromaDB collection configured with the embedding function.
    """
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name="collaborative_rag",
        embedding_function=ef,
    )
    logger.info(f"Collection initialized with {collection.count()} existing chunks")
    return collection


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into chunks at sentence boundaries.

    Args:
        text: The input text to chunk.
        chunk_size: Target size for each chunk in characters.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if not sentence.strip():
            continue
        if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Grab overlap from end of current chunk
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def add_document(
    collection: chromadb.Collection,
    content: str,
    contributor: str,
    source: str = "manual",
) -> int:
    """Chunk and add a document to the collection.

    Args:
        collection: ChromaDB collection to add to.
        content: The document text to add.
        contributor: Name of the person contributing the document.
        source: Source label for the document.

    Returns:
        Number of chunks added.
    """
    chunks = chunk_text(content)
    if not chunks:
        return 0

    ids = [str(uuid.uuid4()) for _ in chunks]
    timestamp = datetime.now().isoformat()
    metadatas = [
        {
            "contributor": contributor,
            "source": source,
            "timestamp": timestamp,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    logger.info(f"Added {len(chunks)} chunks from {contributor} (source: {source})")
    return len(chunks)


def retrieve(
    collection: chromadb.Collection, query: str, k: int = 5
) -> dict:
    """Retrieve relevant chunks for a query.

    Args:
        collection: ChromaDB collection to search.
        query: The search query.
        k: Number of results to return.

    Returns:
        Dict with 'documents', 'metadatas', and 'distances' keys.
    """
    if collection.count() == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    results = collection.query(query_texts=[query], n_results=min(k, collection.count()))
    return results


def generate_answer(question: str, contexts: list[str], llm: ChatOpenAI) -> str:
    """Generate an answer using retrieved contexts.

    Args:
        question: The user's question.
        contexts: List of retrieved text chunks.
        llm: The LLM instance to use for generation.

    Returns:
        Generated answer string.
    """
    if not contexts:
        context_str = "No relevant documents found in the knowledge base."
    else:
        parts = []
        for i, ctx in enumerate(contexts, 1):
            parts.append(f"[{i}] {ctx}")
        context_str = "\n\n".join(parts)

    prompt = (
        "You are a helpful teaching assistant for a Data Engineering course. "
        "Answer the question based ONLY on the provided context. "
        "If the context does not contain enough information, say so clearly.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    return response.content


def get_llm(api_key: str, model: str = "gpt-4o-mini") -> ChatOpenAI:
    """Create a ChatOpenAI instance pointing to NavigatorAI.

    Args:
        api_key: The NavigatorAI API key.
        model: Model name to use.

    Returns:
        Configured ChatOpenAI instance.
    """
    return ChatOpenAI(
        model=model,
        temperature=0.1,
        api_key=api_key,
        base_url="https://api.ai.it.ufl.edu/v1",
    )


def get_stats(collection: chromadb.Collection) -> dict:
    """Get collection statistics.

    Args:
        collection: ChromaDB collection.

    Returns:
        Dict with total_chunks, contributor counts, and sources.
    """
    total = collection.count()
    if total == 0:
        return {
            "total_chunks": 0,
            "contributors": {},
            "sources": {},
        }

    all_data = collection.get(include=["metadatas"])
    contributor_counts = {}
    source_counts = {}
    for meta in all_data["metadatas"]:
        name = meta.get("contributor", "unknown")
        contributor_counts[name] = contributor_counts.get(name, 0) + 1
        src = meta.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    return {
        "total_chunks": total,
        "contributors": contributor_counts,
        "sources": source_counts,
    }
