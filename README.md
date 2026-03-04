# Day 21: Collaborative RAG + RAGAS Demo

A live, collaborative demo where students build a shared RAG knowledge base and evaluate it with RAGAS metrics in real time.

## Quick Start (Students)

### 1. SSH to HiPerGator

```bash
ssh <gatorlink>@hpg.rc.ufl.edu
```

### 2. Set the server URL

The instructor will provide the hostname. Replace `HOSTNAME` below:

```bash
export RAG_SERVER=http://HOSTNAME:8000
```

### 3. Verify connection

```bash
uv run python client.py stats
```

## Commands

### Add text

```bash
uv run python client.py add "RAG combines retrieval with generation to ground LLM answers in actual documents." --name "YourName"
```

### Add content from a URL

```bash
uv run python client.py add-url "https://en.wikipedia.org/wiki/Retrieval-augmented_generation" --name "YourName"
```

### Query the knowledge base

```bash
uv run python client.py query "What is RAG?"
```

### View collection stats

```bash
uv run python client.py stats
```

### List documents

```bash
uv run python client.py docs
uv run python client.py docs --contributor "YourName"
```

### Run RAGAS evaluation

```bash
uv run python client.py evaluate
```

## Instructor Setup

1. Copy `.env.example` to `.env` and set your `NAVIGATOR_API_KEY`
2. Pre-download the embedding model:
   ```bash
   uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```
3. Install dependencies: `uv sync`
4. Submit the server job: `sbatch run_server.sh`
5. Check the log for the server URL: `cat rag-demo-*.log`
6. Test: `curl http://HOSTNAME:8000/stats`
