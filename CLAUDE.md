# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Unified CLI System:**
```bash
# Show all available commands
uv run biorag_cli.py --help

# Extract chunks from PDFs
uv run biorag_cli.py extract papers/ --output data/chunks.jsonl

# Create embeddings and vector index  
uv run biorag_cli.py index data/chunks.jsonl --index-path data/faiss_index/

# Query the RAG system
uv run biorag_cli.py query --query "How does SOD2 work?"
uv run biorag_cli.py query --structured --query "What pathways are involved in mitochondrial ROS detoxification?"
uv run biorag_cli.py query  # Interactive mode

# Comprehensive gene investigation
uv run biorag_cli.py investigate --gene TP53 --include-gaps --output results.json
uv run biorag_cli.py investigate --query "SOD2 role in mitochondrial stress" --mechanism

# Generate biological hypotheses
uv run biorag_cli.py hypothesize "ROS in Alzheimer's disease" --grounded --mechanism
uv run biorag_cli.py hypothesize "NADPH oxidase diabetes" --simple

# Analyze pathway gaps
uv run biorag_cli.py analyze --gene NRF2 --output gaps.json
uv run biorag_cli.py analyze --genes-file genes.txt --output batch_analysis.json
```

**Convenient Wrapper:**
```bash
# Use the wrapper script for shorter commands
./biorag --help
./biorag investigate --gene TP53
```

**Legacy Scripts (still available):**
```bash
# Individual scripts still work for backwards compatibility
uv run scripts/01_extract_chunks.py --input papers/ --output data/chunks.jsonl
uv run scripts/05_gene_investigation.py --gene TP53
```

**Linting and Testing:**
```bash
# Linting
uv run ruff check src/ scripts/
uv run ruff format src/ scripts/

# Testing
python test_structured.py
uv run pytest tests/test_bio_intelligence.py -v

# Single test file
uv run pytest tests/test_bio_intelligence.py::TestGeneResolver -v
```

## Architecture Overview

This is a biological literature RAG system with integrated bio-intelligence capabilities for advanced gene research and hypothesis generation.

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
- Provides `query()` and `retrieve_context()` methods for bio-intelligence integration

### Bio Intelligence Modules (`src/bio_intelligence/`):

**Gene Resolver (`gene_resolver.py`)**
- Extracts gene symbols from biological queries using DSPy LLM + regex fallback
- Resolves genes to Ensembl IDs and pathway databases (Reactome, KEGG)
- Provides confidence scoring and pathway mapping
- Handles both relative and absolute imports for script compatibility

**Hypothesis Generator (`hypothesis_generator.py`)**
- Generates literature-grounded biological hypotheses using DSPy
- Creates testable predictions and mechanistic explanations
- Integrates RAG context with pathway knowledge from databases
- Supports both simple and grounded hypothesis generation modes

**Gap Detector (`gap_detector.py`)**
- Identifies discrepancies between literature mentions and curated databases
- Detects missing upstream regulators and downstream targets
- Supports batch analysis across multiple genes
- Generates knowledge gap reports with confidence scoring

**API Clients (`bio_apis.py`)**
- Unified interface to MyGene.info, Reactome, KEGG, UniProt
- Uses direct REST API calls for Reactome (bypasses reactome2py.analysis issues)
- Robust error handling, retry logic, and caching
- Async operations for performance

**Composed Modules (`composed_modules.py`)**
- ComprehensiveGeneInvestigation: Chains all modules for complete analysis
- BatchInvestigator: Processes multiple queries efficiently
- Investigation reporting and JSON export functionality
- Confidence scoring and summarization via DSPy

### Critical Integration Points:

**DSPy Configuration**: All bio-intelligence scripts must configure DSPy with OpenAI:
```python
import dspy
llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
lm = dspy.LM(model=f"openai/{llm_model}", max_tokens=1000)
dspy.configure(lm=lm)
```

**Import Handling**: Bio-intelligence modules use try/except blocks for imports to support both package imports and script execution:
```python
try:
    from .bio_apis import BioAPIClient
except ImportError:
    from bio_apis import BioAPIClient
```

**RAG Integration**: Bio-intelligence modules expect RAG pipeline to have `query()` or `retrieve_context()` methods for literature context retrieval.

**Data Flow:**
```
Core: PDFs → Text Chunks → Embeddings → FAISS Index → Query → Context Retrieval → LLM Generation → Cited Answer

Bio Intelligence: Query → Gene Resolution → Database APIs → Literature Context → Hypothesis Generation → Gap Detection → Investigation Report
```

## Environment Setup

Requires `.env` file with:
```
OPENAI_API_KEY=your_key_here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

Place source PDFs in `papers/` directory before running build pipeline. The system is pre-populated with mitochondrial research papers for biological queries.

## Development Notes

**CLI Architecture**: The main interface is `biorag_cli.py` with a `./biorag` wrapper script. Each command is implemented as an async function with consistent argument parsing and error handling patterns.

**Package Management**: Project uses UV for dependency management. The CLI uses `uv run` internally via the wrapper script.

**API Rate Limits**: Bio-intelligence modules include retry logic and respect API rate limits for external biological databases.

**Testing Strategy**: Bio-intelligence tests use mocks for external APIs to avoid rate limits during development. Mark integration tests with `@pytest.mark.slow` decorator.

**Error Handling**: All modules include comprehensive error handling and logging. Bio-intelligence modules gracefully degrade when external APIs are unavailable.

**Adding New Commands**: To add a new command to the CLI:
1. Add a subparser in `create_parser()`
2. Implement a `cmd_<name>(args)` function 
3. Add the command routing in `main()`
4. Update help examples and CLAUDE.md