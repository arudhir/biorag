import json
from pathlib import Path
from typing import Dict, List

import faiss
import jsonlines
import numpy as np


class VectorStore:
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.metadata: List[Dict] = []
        self.chunks_map: Dict[str, str] = {}  # chunk_id -> text mapping
    
    def add_documents(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        """Add documents to the index with their metadata."""
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        query_embedding = np.ascontiguousarray(query_embedding)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                metadata = self.metadata[idx]
                chunk_id = metadata["chunk_id"]
                result = {
                    **metadata,
                    "text": self.chunks_map.get(chunk_id, "Text not found"),
                    "relevance_score": float(score)
                }
                results.append(result)
        
        return results
    
    def save_index(self, path: str) -> None:
        """Save the FAISS index and metadata to disk."""
        index_path = Path(path)
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path / "index.faiss"))
        
        # Save metadata
        with open(index_path / "metadata.json", 'w') as f:
            json.dump(self.metadata, f)
    
    def load_index(self, path: str) -> None:
        """Load the FAISS index and metadata from disk."""
        index_path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path / "index.faiss"))
        
        # Load metadata
        with open(index_path / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Load text chunks from chunks.jsonl
        chunks_file = index_path.parent / "chunks.jsonl"
        if chunks_file.exists():
            with jsonlines.open(chunks_file) as reader:
                for item in reader:
                    chunk_id = item['metadata']['chunk_id']
                    self.chunks_map[chunk_id] = item['text']
    
    def update_documents(self, new_embeddings: np.ndarray, new_metadata: List[Dict]) -> None:
        """Update existing documents in the index."""
        # Remove old entries
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        
        # Add new entries
        self.add_documents(new_embeddings, new_metadata)
    
    def process_embeddings_file(self, embeddings_file: str) -> None:
        """Process a JSONL file of embeddings and add to index."""
        embeddings = []
        metadata_list = []
        
        # Read embeddings
        with jsonlines.open(embeddings_file) as reader:
            for item in reader:
                embeddings.append(item['embedding'])
                metadata_list.append(item['metadata'])
        
        # Convert to numpy array with float32 dtype for FAISS
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Add to index
        self.add_documents(embeddings_array, metadata_list) 