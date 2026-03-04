"""Pre-defined evaluation questions with ground truth answers.

These questions are designed so that an empty collection scores poorly,
and adding course-relevant content about RAG, embeddings, and data engineering
progressively improves RAGAS metrics.
"""

TEST_QUESTIONS = [
    {
        "question": "What is Retrieval-Augmented Generation (RAG) and why is it useful?",
        "ground_truth": (
            "RAG is a technique that combines information retrieval with language "
            "model generation. It retrieves relevant documents from a knowledge base "
            "and provides them as context to an LLM, reducing hallucinations and "
            "grounding answers in actual data."
        ),
    },
    {
        "question": "What are the main components of a RAG pipeline?",
        "ground_truth": (
            "A RAG pipeline consists of document ingestion (loading and chunking text), "
            "embedding generation (converting text to vectors), vector storage "
            "(indexing embeddings in a database like ChromaDB), retrieval (finding "
            "relevant chunks via similarity search), and generation (using an LLM to "
            "produce answers from retrieved context)."
        ),
    },
    {
        "question": "How do vector embeddings represent text for similarity search?",
        "ground_truth": (
            "Vector embeddings map text into dense numerical vectors in a "
            "high-dimensional space where semantically similar texts are close "
            "together. Models like all-MiniLM-L6-v2 produce these embeddings, "
            "and similarity is measured using cosine similarity or Euclidean distance."
        ),
    },
    {
        "question": "Why is document chunking important and what strategies exist?",
        "ground_truth": (
            "Document chunking splits large documents into smaller pieces that fit "
            "within embedding model context limits and improve retrieval precision. "
            "Strategies include fixed-size chunking, sentence-boundary chunking, "
            "and recursive character splitting. Overlap between chunks preserves "
            "context at boundaries."
        ),
    },
    {
        "question": "What is prompt engineering and how does it affect LLM outputs?",
        "ground_truth": (
            "Prompt engineering is the practice of designing input prompts to guide "
            "LLM behavior and output quality. Techniques include providing clear "
            "instructions, few-shot examples, system messages, and structured output "
            "formats. Good prompts reduce hallucinations and improve answer relevance."
        ),
    },
]
