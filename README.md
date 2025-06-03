# 🧬 Biological PDF RAG System

A Retrieval-Augmented Generation (RAG) system for querying biological research PDFs using natural language. The system provides cited, grounded answers by leveraging state-of-the-art language models and vector search.

## Features

- PDF text extraction with metadata preservation
- Efficient chunking with configurable overlap
- OpenAI embeddings with caching
- FAISS vector similarity search
- DSPy-based RAG pipeline
- Interactive query interface
- Citation tracking and relevance scoring

## Prerequisites

- Python 3.9+
- OpenAI API key
- UV package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/biorag.git
cd biorag
```

2. Install dependencies using UV:
```bash
uv add dspy-ai faiss-cpu openai pypdf pymupdf tiktoken python-dotenv jsonlines tqdm numpy pandas
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

## Usage

### 1. Extract Chunks from PDFs

Place your PDF files in the `papers/` directory, then run:

```bash
python scripts/01_extract_chunks.py --input papers/ --output data/chunks.jsonl
```

Optional arguments:
- `--chunk-size`: Size of text chunks in tokens (default: 800)
- `--overlap`: Overlap between chunks in tokens (default: 200)

### 2. Create Embeddings and Index

```bash
python scripts/02_embed_and_index.py --chunks data/chunks.jsonl --index-path data/faiss_index/
```

Optional arguments:
- `--model`: Embedding model to use (default: text-embedding-3-small)
- `--batch-size`: Batch size for embedding generation (default: 100)
- `--cache-path`: Path to save/load embedding cache

### 3. Query the System

Interactive mode:
```bash
python scripts/03_rag_query.py --index-path data/faiss_index/
```

Single query:
```bash
python scripts/03_rag_query.py --index-path data/faiss_index/ --query "How does NADPH oxidase work?"
```

Optional arguments:
- `--top-k`: Number of chunks to retrieve (default: 5)
- `--output`: Output JSON file for results
- `--model`: Embedding model to use

## Project Structure

```
biorag/
├── .env                      # Environment variables
├── pyproject.toml           # Project configuration
├── papers/                   # Input PDF directory
├── data/
│   ├── chunks.jsonl         # Processed text chunks
│   └── faiss_index/         # Vector database
├── src/
│   ├── pdf_processor.py     # PDF extraction logic
│   ├── embedder.py          # Embedding utilities
│   ├── vector_store.py      # FAISS operations
│   └── rag_pipeline.py      # DSPy RAG module
└── scripts/
    ├── 01_extract_chunks.py
    ├── 02_embed_and_index.py
    └── 03_rag_query.py
```

## Example Output

```json
{
  "question": "How is NADPH generated in cells?",
  "answer": "NADPH is primarily generated through two pathways: the pentose phosphate pathway (PPP) and the malate-pyruvate cycle. The PPP, particularly through glucose-6-phosphate dehydrogenase (G6PD), is the main source of cytosolic NADPH.",
  "citations": [
    {
      "source": "glucose_metabolism_2023.pdf",
      "page": 3,
      "chunk_id": "glucose_metabolism_2023_p3_chunk1",
      "relevance_score": 0.89
    }
  ],
  "confidence": 0.87,
  "reasoning": "Found relevant information about NADPH generation in multiple sources discussing cellular metabolism pathways."
}
```

## Error Handling

The system includes robust error handling for:
- Corrupted PDF files
- API rate limits
- Memory management
- Index corruption
- No results found

## Future Enhancements

- Web interface using Streamlit or FastAPI
- Advanced semantic chunking
- Multi-modal support for figures/tables
- Real-time PDF updates
- Hybrid search (dense + sparse)
- Query expansion for scientific terms

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 