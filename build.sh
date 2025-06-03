#!/bin/bash
uv run scripts/01_extract_chunks.py --input papers/ --output data/chunks.jsonl
uv run scripts/02_embed_and_index.py --chunks data/chunks.jsonl --index-path data/faiss_index/
