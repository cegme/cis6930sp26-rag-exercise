"""RAGAS evaluation integration.

Runs RAGAS metrics (faithfulness, answer_relevancy, context_precision) against
pre-defined test questions to measure RAG quality in real time.
"""

from datasets import Dataset
from langchain_openai import ChatOpenAI
from loguru import logger
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    Faithfulness,
)

from rag import generate_answer, retrieve
from test_questions import TEST_QUESTIONS


def run_ragas_evaluation(
    collection,
    llm: ChatOpenAI,
    k: int = 5,
) -> dict:
    """Run RAGAS evaluation on pre-defined test questions.

    Retrieves contexts and generates answers for each test question,
    then evaluates using RAGAS metrics.

    Args:
        collection: ChromaDB collection to query.
        llm: LLM instance for answer generation.
        k: Number of chunks to retrieve per question.

    Returns:
        Dict with 'aggregate' scores and 'per_question' breakdown.
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for tq in TEST_QUESTIONS:
        question = tq["question"]
        ground_truth = tq["ground_truth"]

        results = retrieve(collection, question, k=k)
        retrieved_docs = results["documents"][0] if results["documents"][0] else []
        answer = generate_answer(question, retrieved_docs, llm)

        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved_docs if retrieved_docs else ["No context available."])
        ground_truths.append(ground_truth)

        logger.debug(f"Q: {question[:60]}... -> {len(retrieved_docs)} contexts")

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    evaluator_llm = LangchainLLMWrapper(llm)

    logger.info("Running RAGAS evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()],
        llm=evaluator_llm,
    )
    logger.info(f"RAGAS evaluation complete: {result}")

    per_question = []
    result_df = result.to_pandas()
    for _, row in result_df.iterrows():
        per_question.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "faithfulness": _safe_score(row.get("faithfulness")),
                "answer_relevancy": _safe_score(row.get("answer_relevancy")),
                "context_precision": _safe_score(row.get("context_precision")),
            }
        )

    aggregate = {
        "faithfulness": _safe_score(result.get("faithfulness")),
        "answer_relevancy": _safe_score(result.get("answer_relevancy")),
        "context_precision": _safe_score(result.get("context_precision")),
    }

    return {
        "aggregate": aggregate,
        "per_question": per_question,
        "num_questions": len(TEST_QUESTIONS),
        "total_chunks_in_collection": collection.count(),
    }


def _safe_score(value) -> float | None:
    """Convert a score to float, handling None and NaN."""
    if value is None:
        return None
    try:
        score = float(value)
        if score != score:  # NaN check
            return None
        return round(score, 4)
    except (TypeError, ValueError):
        return None
