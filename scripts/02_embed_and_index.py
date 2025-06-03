#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedder import EmbeddingManager
from src.vector_store import VectorStore

def main():
    parser = argparse.ArgumentParser(description="Create embeddings and FAISS index from chunks")
    parser.add_argument("--chunks", required=True, help="Input JSONL file containing chunks")
    parser.add_argument("--index-path", required=True, help="Output directory for FAISS index")
    parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model to use")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for embedding generation")
    parser.add_argument("--cache-path", help="Path to save/load embedding cache")
    
    args = parser.parse_args()
    
    # Validate input file
    chunks_file = Path(args.chunks)
    if not chunks_file.exists():
        print(f"Error: Input file '{args.chunks}' does not exist")
        sys.exit(1)
    
    # Create output directory
    index_path = Path(args.index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize embedder
    embedder = EmbeddingManager(
        model=args.model,
        batch_size=args.batch_size
    )
    
    # Load cache if provided
    if args.cache_path:
        cache_path = Path(args.cache_path)
        if cache_path.exists():
            print(f"Loading embedding cache from {cache_path}...")
            embedder.load_cache(str(cache_path))
    
    # Process chunks and create embeddings
    print("Generating embeddings...")
    embeddings_file = index_path / "embeddings.jsonl"
    embedder.process_chunks_file(str(chunks_file), str(embeddings_file))
    
    # Create and save FAISS index
    print("Creating FAISS index...")
    vector_store = VectorStore()
    vector_store.process_embeddings_file(str(embeddings_file))
    vector_store.save_index(str(index_path))
    
    # Save cache if provided
    if args.cache_path:
        print(f"Saving embedding cache to {args.cache_path}...")
        embedder.save_cache(args.cache_path)
    
    print(f"Index saved to {index_path}")

if __name__ == "__main__":
    main() 