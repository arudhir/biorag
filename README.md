# 🧬 BioRAG: Intelligent Biological Literature Analysis

An advanced AI-powered system for analyzing biological research literature. BioRAG combines retrieval-augmented generation (RAG) with specialized biological intelligence modules to provide comprehensive gene investigations, hypothesis generation, and pathway analysis from scientific PDFs.

## 🚀 Key Capabilities

**Core RAG Pipeline**
- Extract and semantically index scientific PDFs with metadata preservation
- Natural language querying with cited, grounded answers
- High-performance vector search using FAISS with OpenAI embeddings
- Confidence scoring and source attribution

**Biological Intelligence Modules**
- **Gene Investigation**: Comprehensive analysis linking genes to pathways, functions, and literature
- **Hypothesis Generation**: AI-driven generation of testable biological hypotheses
- **Gap Detection**: Identify discrepancies between literature and curated databases
- **Pathway Analysis**: Connect genes to Reactome, KEGG, and other biological databases

## ⚡ Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key
- UV package manager

### Installation

```bash
git clone https://github.com/yourusername/biorag.git
cd biorag
make install
```

Create `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

### Build Knowledge Base

1. **Add PDFs**: Place research papers in the `papers/` directory
2. **Build Database**: Extract text and create vector index
```bash
make build-db
```

### Start Querying

```bash
# Interactive mode
make query

# Gene investigation
make investigate GENE=TP53

# Generate hypotheses
make hypothesize QUERY="ROS signaling in neurodegeneration"

# Analyze pathway gaps
make analyze GENE=SOD2
```

## 📋 Complete Workflow

### Step 1: Prepare Literature Collection
Place your biological research PDFs in the `papers/` directory. The system works best with:
- Primary research articles
- Review papers
- Pathway and mechanism studies
- Gene function analyses

### Step 2: Build the Knowledge Base
```bash
# Extract text chunks from PDFs
make extract

# Create embeddings and search index  
make index

# Or do both steps at once
make build-db
```

**What's happening**: The system extracts text from PDFs, splits it into semantically meaningful chunks with overlap for context preservation, generates high-dimensional embeddings using OpenAI's models, and builds a FAISS vector index for fast similarity search.

### Step 3: Biological Analysis

**Basic Literature Query**
```bash
./biorag query --query "How does mitochondrial dysfunction contribute to aging?"
```

**Comprehensive Gene Investigation**
```bash
./biorag investigate --gene PINK1 --include-gaps --output pink1_analysis.json
```
*Analyzes gene function, pathways, interactions, and identifies knowledge gaps*

**AI Hypothesis Generation**
```bash
./biorag hypothesize "PINK1 mutations and Parkinson's disease progression" --grounded --mechanism
```
*Generates testable hypotheses grounded in literature with mechanistic explanations*

**Pathway Gap Analysis**
```bash
./biorag analyze --gene PARKIN --output parkin_gaps.json
```
*Identifies discrepancies between literature mentions and curated pathway databases*

## 🏗️ System Architecture

```
biorag/
├── biorag_cli.py            # Unified command-line interface
├── Makefile                 # Development and pipeline commands
├── papers/                  # Input PDF directory
├── data/
│   ├── chunks.jsonl         # Processed text chunks
│   └── faiss_index/         # High-performance vector database
├── src/
│   ├── pdf_processor.py     # PDF extraction and chunking
│   ├── embedder.py          # OpenAI embedding generation
│   ├── vector_store.py      # FAISS vector operations
│   ├── rag_pipeline.py      # DSPy-based RAG pipeline
│   └── bio_intelligence/    # Specialized biological modules
│       ├── gene_resolver.py      # Gene symbol resolution
│       ├── hypothesis_generator.py # AI hypothesis generation
│       ├── gap_detector.py       # Knowledge gap identification
│       ├── bio_apis.py           # Biological database APIs
│       └── composed_modules.py   # Integrated analysis workflows
└── scripts/                 # Legacy individual processing scripts
```

## 📊 Example Outputs

**Gene Investigation Report**
```json
{
  "gene": "TP53",
  "ensembl_id": "ENSG00000141510",
  "pathways": {
    "reactome": ["DNA Damage Response", "p53-Dependent G1/S DNA damage checkpoint"],
    "kegg": ["p53 signaling pathway", "Apoptosis"]
  },
  "literature_analysis": {
    "functions": ["tumor suppressor", "DNA damage response", "cell cycle checkpoint"],
    "interactions": ["MDM2", "ATM", "CHEK2"],
    "confidence": 0.94
  },
  "knowledge_gaps": [
    {
      "type": "missing_interaction",
      "description": "Literature mentions TP53-BRCA1 interaction not found in Reactome",
      "confidence": 0.87
    }
  ]
}
```

**AI-Generated Hypothesis**
```json
{
  "query": "SOD2 mutations in neurodegeneration",
  "hypothesis": "SOD2 mutations lead to mitochondrial oxidative stress accumulation, triggering neuronal apoptosis through the intrinsic pathway via cytochrome c release and caspase-9 activation.",
  "predictions": [
    "Increased 8-oxoguanine DNA lesions in SOD2-deficient neurons",
    "Elevated cytochrome c in cytoplasm of affected cells",
    "Enhanced caspase-9 activity in neuronal cultures"
  ],
  "mechanisms": ["mitochondrial dysfunction", "oxidative DNA damage", "apoptotic signaling"],
  "confidence": 0.91,
  "supporting_evidence": ["PMID:12345678", "PMID:87654321"]
}
```

## 🔧 Development Commands

```bash
# System status and health check
make status

# Code quality and testing
make lint format test

# Clean cache and build artifacts  
make clean

# Advanced development
make profile              # Performance profiling
make check-deps          # Dependency analysis
make update-deps         # Update all dependencies
```

## 🌟 Advanced Features

**Structured Query Support**
```bash
./biorag query --structured --query "What pathways involve oxidative stress?"
```
*Returns structured JSON with pathway mappings, gene lists, and confidence scores*

**Batch Analysis**
```bash
./biorag analyze --genes-file gene_list.txt --output batch_results.json
```
*Process multiple genes simultaneously with comprehensive reporting*

**Citation Tracking**
Every answer includes source attribution with:
- Original PDF filename and page number
- Text chunk relevance scores
- Confidence assessments
- Reasoning explanations

## 🚀 Technical Highlights

- **High Performance**: FAISS vector search with sub-second query times
- **Scalable**: Handles thousands of research papers efficiently
- **Modular**: Plugin architecture for extending biological intelligence
- **Robust**: Comprehensive error handling and retry logic for external APIs
- **Extensible**: Easy integration with new biological databases and AI models

## 🧪 Use Cases

- **Drug Discovery**: Investigate target genes and pathway interactions
- **Academic Research**: Literature review and hypothesis generation
- **Biotech R&D**: Gap analysis for competitive intelligence
- **Clinical Research**: Connect genetic variants to known pathways
- **Grant Writing**: Evidence gathering and mechanism exploration

## Contributing

We welcome contributions! Please see our contribution guidelines for development setup and coding standards.

## License

MIT License - see LICENSE file for details. 