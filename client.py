"""CLI client for the collaborative RAG demo.

Provides subcommands to interact with the RAG server: add documents,
query the knowledge base, view stats, and run RAGAS evaluation.
"""

import argparse
import os
import sys

import httpx
from loguru import logger

# ANSI color codes
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def get_server_url(args) -> str:
    """Get server URL from args or environment variable."""
    url = getattr(args, "server", None) or os.getenv("RAG_SERVER")
    if not url:
        logger.error("Server URL not set. Use --server or export RAG_SERVER=http://host:8000")
        sys.exit(1)
    return url.rstrip("/")


def cmd_add(args):
    """Add a text document to the knowledge base."""
    url = get_server_url(args)
    payload = {
        "content": args.text,
        "contributor": args.name,
        "source": args.source,
    }
    resp = httpx.post(f"{url}/documents", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print(f"{GREEN}Added {data['chunks_added']} chunks{RESET} as {BOLD}{data['contributor']}{RESET}")


def cmd_add_url(args):
    """Fetch a URL and add its text content to the knowledge base."""
    from bs4 import BeautifulSoup

    url = get_server_url(args)

    logger.info(f"Fetching {args.url}...")
    page_resp = httpx.get(args.url, timeout=30, follow_redirects=True)
    page_resp.raise_for_status()

    soup = BeautifulSoup(page_resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)

    if not text.strip():
        logger.error("No text content found at URL")
        sys.exit(1)

    payload = {
        "content": text,
        "contributor": args.name,
        "source": args.url,
    }
    resp = httpx.post(f"{url}/documents", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print(f"{GREEN}Added {data['chunks_added']} chunks{RESET} from {BLUE}{args.url}{RESET}")


def cmd_query(args):
    """Query the knowledge base and get an answer."""
    url = get_server_url(args)
    payload = {"question": args.question, "k": args.k}
    resp = httpx.post(f"{url}/query", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    print(f"\n{BOLD}Question:{RESET} {data['question']}")
    print(f"\n{BOLD}Answer:{RESET}\n{data['answer']}")
    print(f"\n{DIM}--- Retrieved {data['num_contexts']} contexts ---{RESET}")
    for i, ctx in enumerate(data["contexts"], 1):
        print(f"{DIM}[{i}] {ctx['contributor']} ({ctx['source']}) dist={ctx['distance']}{RESET}")
        print(f"    {ctx['text'][:120]}...")


def cmd_stats(args):
    """Show collection statistics."""
    url = get_server_url(args)
    resp = httpx.get(f"{url}/stats", timeout=10)
    resp.raise_for_status()
    data = resp.json()

    print(f"\n{BOLD}Collection Stats{RESET}")
    print(f"  Total chunks: {GREEN}{data['total_chunks']}{RESET}")
    if data.get("contributors"):
        print(f"\n  {BOLD}Contributors:{RESET}")
        for name, count in sorted(data["contributors"].items()):
            print(f"    {name}: {count} chunks")
    if data.get("sources"):
        print(f"\n  {BOLD}Sources:{RESET}")
        for src, count in sorted(data["sources"].items()):
            print(f"    {src}: {count} chunks")


def cmd_docs(args):
    """List documents in the collection."""
    url = get_server_url(args)
    params = {}
    if args.contributor:
        params["contributor"] = args.contributor
    resp = httpx.get(f"{url}/documents", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    print(f"\n{BOLD}Documents ({data['total']} total){RESET}")
    for doc in data["documents"]:
        meta = doc["metadata"]
        print(f"  {BLUE}{meta.get('contributor', '?')}{RESET} ({meta.get('source', '?')}) "
              f"{DIM}{meta.get('timestamp', '')}{RESET}")
        print(f"    {doc['text'][:100]}...")


def cmd_evaluate(args):
    """Run RAGAS evaluation on test questions."""
    url = get_server_url(args)
    print(f"{YELLOW}Running RAGAS evaluation (this may take a minute)...{RESET}")
    resp = httpx.post(f"{url}/evaluate", timeout=300)
    resp.raise_for_status()
    data = resp.json()

    agg = data["aggregate"]
    print(f"\n{BOLD}=== RAGAS Evaluation Results ==={RESET}")
    print(f"  Collection size: {data['total_chunks_in_collection']} chunks")
    print(f"  Questions evaluated: {data['num_questions']}")

    print(f"\n  {BOLD}Aggregate Scores:{RESET}")
    for metric, score in agg.items():
        color = GREEN if score and score > 0.7 else YELLOW if score and score > 0.4 else RED
        display = f"{score:.4f}" if score is not None else "N/A"
        print(f"    {metric}: {color}{display}{RESET}")

    print(f"\n  {BOLD}Per-Question Breakdown:{RESET}")
    for pq in data["per_question"]:
        print(f"\n  {BLUE}Q: {pq['question'][:70]}...{RESET}")
        print(f"    A: {pq['answer'][:100]}...")
        for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
            score = pq.get(metric)
            color = GREEN if score and score > 0.7 else YELLOW if score and score > 0.4 else RED
            display = f"{score:.4f}" if score is not None else "N/A"
            print(f"    {metric}: {color}{display}{RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="CLI client for the Collaborative RAG Demo"
    )
    parser.add_argument(
        "--server",
        default=os.getenv("RAG_SERVER"),
        help="Server URL (default: $RAG_SERVER env var)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # add
    p_add = subparsers.add_parser("add", help="Add text to the knowledge base")
    p_add.add_argument("text", help="Text content to add")
    p_add.add_argument("--name", default="anonymous", help="Your name")
    p_add.add_argument("--source", default="manual", help="Source label")
    p_add.set_defaults(func=cmd_add)

    # add-url
    p_url = subparsers.add_parser("add-url", help="Add content from a URL")
    p_url.add_argument("url", help="URL to fetch and add")
    p_url.add_argument("--name", default="anonymous", help="Your name")
    p_url.set_defaults(func=cmd_add_url)

    # query
    p_query = subparsers.add_parser("query", help="Query the knowledge base")
    p_query.add_argument("question", help="Your question")
    p_query.add_argument("--k", type=int, default=5, help="Number of contexts to retrieve")
    p_query.set_defaults(func=cmd_query)

    # stats
    p_stats = subparsers.add_parser("stats", help="Show collection stats")
    p_stats.set_defaults(func=cmd_stats)

    # docs
    p_docs = subparsers.add_parser("docs", help="List documents")
    p_docs.add_argument("--contributor", help="Filter by contributor name")
    p_docs.set_defaults(func=cmd_docs)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run RAGAS evaluation")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
