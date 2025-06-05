# Makefile for biorag - Biological Literature RAG System

.PHONY: help install clean test lint format build-db query investigate hypothesize analyze dev-install

# Default target
help:
	@echo "BioRAG - Biological Literature RAG System"
	@echo ""
	@echo "Development Commands:"
	@echo "  install      Install dependencies with uv"
	@echo "  dev-install  Install with dev dependencies"
	@echo "  clean        Clean up cache files and build artifacts"
	@echo "  test         Run tests"
	@echo "  lint         Run linting with ruff"
	@echo "  format       Format code with ruff"
	@echo ""
	@echo "Database Pipeline:"
	@echo "  build-db     Extract chunks and build vector index from PDFs"
	@echo "  extract      Extract chunks from PDFs only"
	@echo "  index        Build vector index from existing chunks"
	@echo ""
	@echo "Query Commands:"
	@echo "  query        Interactive query mode"
	@echo "  investigate  Gene investigation (set GENE=<symbol>)"
	@echo "  hypothesize  Generate hypothesis (set QUERY=<text>)"
	@echo "  analyze      Analyze pathway gaps (set GENE=<symbol>)"
	@echo ""
	@echo "Examples:"
	@echo "  make investigate GENE=TP53"
	@echo "  make hypothesize QUERY='ROS in Alzheimer disease'"
	@echo "  make analyze GENE=NRF2"

# Installation
install:
	uv sync

dev-install:
	uv sync --dev

# Cleanup
clean:
	@echo "Cleaning cache files and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage 2>/dev/null || true

# Testing and Quality
test:
	uv run python test_structured.py
	uv run pytest tests/test_bio_intelligence.py -v

test-slow:
	uv run pytest tests/test_bio_intelligence.py -v -m slow

lint:
	uv run ruff check src/ scripts/ biorag_cli.py

format:
	uv run ruff format src/ scripts/ biorag_cli.py

# Database Pipeline
build-db: extract index
	@echo "Database build complete!"

extract:
	@echo "Extracting chunks from PDFs..."
	uv run biorag_cli.py extract papers/ --output data/chunks.jsonl

index:
	@echo "Building vector index..."
	uv run biorag_cli.py index data/chunks.jsonl --index-path data/faiss_index/

# Query Commands
query:
	uv run biorag_cli.py query

query-structured:
	uv run biorag_cli.py query --structured

# Gene Investigation (usage: make investigate GENE=TP53)
investigate:
	@if [ -z "$(GENE)" ]; then \
		echo "Usage: make investigate GENE=<gene_symbol>"; \
		echo "Example: make investigate GENE=TP53"; \
		exit 1; \
	fi
	uv run biorag_cli.py investigate --gene $(GENE) --include-gaps --output results_$(GENE).json

# Hypothesis Generation (usage: make hypothesize QUERY="your query")
hypothesize:
	@if [ -z "$(QUERY)" ]; then \
		echo "Usage: make hypothesize QUERY='your biological query'"; \
		echo "Example: make hypothesize QUERY='ROS in Alzheimer disease'"; \
		exit 1; \
	fi
	uv run biorag_cli.py hypothesize "$(QUERY)" --grounded --mechanism

# Pathway Analysis (usage: make analyze GENE=NRF2)
analyze:
	@if [ -z "$(GENE)" ]; then \
		echo "Usage: make analyze GENE=<gene_symbol>"; \
		echo "Example: make analyze GENE=NRF2"; \
		exit 1; \
	fi
	uv run biorag_cli.py analyze --gene $(GENE) --output gaps_$(GENE).json

# Quick development targets
quick-test:
	uv run pytest tests/test_bio_intelligence.py::TestGeneResolver -v

rebuild-db: clean-db build-db

clean-db:
	@echo "Cleaning database files..."
	rm -rf data/chunks.jsonl data/faiss_index/ 2>/dev/null || true

# Show system status
status:
	@echo "BioRAG System Status:"
	@echo "UV installed: $$(command -v uv >/dev/null 2>&1 && echo 'Yes' || echo 'No')"
	@echo "Papers directory: $$(ls papers/ 2>/dev/null | wc -l | tr -d ' ') PDFs"
	@echo "Chunks file: $$(test -f data/chunks.jsonl && echo 'Exists' || echo 'Missing')"
	@echo "Vector index: $$(test -d data/faiss_index && echo 'Exists' || echo 'Missing')"
	@echo "Environment file: $$(test -f .env && echo 'Exists' || echo 'Missing')"

# Advanced targets for development
profile:
	uv run python -m cProfile -o profile.stats biorag_cli.py query --query "SOD2 mitochondrial function"

check-deps:
	uv tree

update-deps:
	uv lock --upgrade

# Legacy script support (for backwards compatibility)
legacy-extract:
	uv run scripts/01_extract_chunks.py --input papers/ --output data/chunks.jsonl

legacy-index:
	uv run scripts/02_embed_and_index.py --input data/chunks.jsonl --index-path data/faiss_index/

legacy-query:
	uv run scripts/03_rag_query.py

legacy-investigate:
	@if [ -z "$(GENE)" ]; then \
		echo "Usage: make legacy-investigate GENE=<gene_symbol>"; \
		exit 1; \
	fi
	uv run scripts/05_gene_investigation.py --gene $(GENE)