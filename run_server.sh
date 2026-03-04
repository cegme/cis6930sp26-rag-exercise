#!/bin/bash
#SBATCH --job-name=rag-demo
#SBATCH --account=cis6930
#SBATCH --qos=cis6930
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=01:30:00
#SBATCH --output=rag-demo-%j.log

echo "============================================"
echo "RAG Demo Server Starting"
echo "SERVER URL: http://$(hostname):8000"
echo "============================================"

cd "$SLURM_SUBMIT_DIR"

# Ensure venv is created with a Python available on the compute node
uv sync

uv run uvicorn server:app --host 0.0.0.0 --port 8000
