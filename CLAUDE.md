# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Build/Setup Pipeline:**
```bash
# Full build pipeline (extract, embed, index)
./build.sh

# Or run individually:
uv run scripts/01_extract_chunks.py --input papers/ --output data/chunks.jsonl
uv run scripts/02_embed_and_index.py --chunks data/chunks.jsonl --index-path data/faiss_index/
```

**Query System:**
```bash
# Interactive mode
./run.sh
# Or: uv run scripts/03_rag_query.py --index-path data/faiss_index/

# Single query
uv run scripts/03_rag_query.py --index-path data/faiss_index/ --query "your question"

# Structured JSON output
uv run scripts/03_rag_query.py --index-path data/faiss_index/ --structured --query "What pathways are involved in mitochondrial ROS detoxification?"
```

**Bio Intelligence System:**
```bash
# Comprehensive gene investigation
uv run scripts/05_gene_investigation.py --query "SOD2 role in mitochondrial stress"
uv run scripts/05_gene_investigation.py --gene "NRF2" --include-gaps --output results.json

# Hypothesis generation
uv run scripts/06_hypothesis_generator.py --topic "ROS in Alzheimer's disease"
uv run scripts/06_hypothesis_generator.py --topic "NADPH oxidase diabetes" --grounded --mechanism

# Pathway gap analysis
uv run scripts/07_pathway_analysis.py --gene "NRF2" --compare-databases
uv run scripts/07_pathway_analysis.py --pathway "oxidative stress response" --find-gaps
```

**Linting:**
```bash
uv run ruff check src/ scripts/
uv run ruff format src/ scripts/
```

**Testing:**
```bash
# Test structured output functionality
python test_structured.py

# Test bio intelligence modules
uv run pytest tests/test_bio_intelligence.py -v
```

## Architecture Overview

This is a biological literature RAG system with integrated bio-intelligence capabilities:

### Core RAG Pipeline (3-stage):

**Stage 1: PDF Processing (`src/pdf_processor.py`)**
- Extracts text from scientific PDFs with metadata preservation
- Chunks text with configurable overlap for optimal retrieval
- Outputs to JSONL format with source tracking

**Stage 2: Embedding & Indexing (`src/embedder.py`, `src/vector_store.py`)**
- Generates OpenAI embeddings with caching for efficiency
- Builds FAISS vector index for similarity search
- Maintains metadata mapping for citation tracking

**Stage 3: RAG Pipeline (`src/rag_pipeline.py`)**
- Retrieves relevant chunks using vector similarity
- Generates answers via OpenAI API with scientific context
- Supports both standard text and structured JSON output modes
- Includes confidence scoring and citation tracking

**Key Design Patterns:**
- Scripts use UV for Python execution (`uv run`)
- Environment variables loaded from `.env` file for API keys
- Modular components allow independent testing and swapping
- Structured output mode returns JSON with pathways, genes, and citations for biological queries

### Bio Intelligence Modules (`src/bio_intelligence/`):

**Gene Resolver (`gene_resolver.py`)**
- Extracts gene symbols from biological queries using LLM + regex
- Resolves genes to Ensembl IDs and pathway databases (Reactome, KEGG)
- Provides confidence scoring and pathway mapping

**Hypothesis Generator (`hypothesis_generator.py`)**
- Generates literature-grounded biological hypotheses using DSPy
- Creates testable predictions and mechanistic explanations
- Integrates RAG context with pathway knowledge

**Gap Detector (`gap_detector.py`)**
- Identifies discrepancies between literature and curated databases
- Detects missing upstream regulators and downstream targets
- Supports batch analysis across multiple genes

**API Clients (`bio_apis.py`)**
- Unified interface to MyGene.info, Reactome, KEGG, UniProt
- Robust error handling, retry logic, and caching
- Async operations for performance

**Composed Modules (`composed_modules.py`)**
- ComprehensiveGeneInvestigation: Chains all modules for complete analysis
- BatchInvestigator: Processes multiple queries efficiently
- Investigation reporting and export functionality

**Data Flow:**
```
Core: PDFs → Text Chunks → Embeddings → FAISS Index → Query → Context Retrieval → LLM Generation → Cited Answer

Bio Intelligence: Query → Gene Resolution → Database APIs → Literature Context → Hypothesis Generation → Gap Detection → Investigation Report
```

**Structured Output Schema:**
```json
{
  "pathways": [{"name": "...", "genes": ["..."], "description": "..."}],
  "key_findings": [{"finding": "...", "evidence": "..."}],
  "citations": [{"source": "...", "page": 1, "relevance": "high"}]
}
```

## Environment Setup

Requires `.env` file with:
```
OPENAI_API_KEY=your_key_here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

Place source PDFs in `papers/` directory before running build pipeline.