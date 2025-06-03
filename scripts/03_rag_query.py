#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedder import EmbeddingManager
from src.rag_pipeline import BiologicalRAG
from src.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Query the biological RAG system")
    parser.add_argument("--index-path", required=True, help="Path to FAISS index directory")
    parser.add_argument("--query", help="Query to process (if not provided, enters interactive mode)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model to use")
    parser.add_argument("--structured", action="store_true", help="Return structured JSON output")
    
    args = parser.parse_args()
    
    # Validate index path
    index_path = Path(args.index_path)
    if not index_path.exists() or not index_path.is_dir():
        print(f"Error: Index directory '{args.index_path}' does not exist")
        sys.exit(1)
    
    # Initialize components
    embedder = EmbeddingManager(model=args.model)
    vector_store = VectorStore()
    vector_store.load_index(str(index_path))
    
    rag = BiologicalRAG(
        vector_store=vector_store,
        embedder=embedder,
        top_k=args.top_k
    )
    
    def process_query(query: str) -> None:
        """Process a single query and display results."""
        print("\nProcessing query...")
        response = rag.forward(query, structured_output=args.structured)
        
        # Display formatted response
        print("\n" + rag.format_response(response))
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(response, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    if args.query:
        # Process single query
        process_query(args.query)
    else:
        # Interactive mode
        print("Entering interactive mode. Type 'exit' to quit.")
        while True:
            try:
                query = input("\nEnter your question: ").strip()
                if query.lower() in ('exit', 'quit'):
                    break
                if query:
                    process_query(query)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 