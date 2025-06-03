import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
import numpy as np
import openai
from tqdm import tqdm


class EmbeddingManager:
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 100):
        self.model = model
        self.batch_size = batch_size
        self.cache: Dict[str, np.ndarray] = {}
        
        # Initialize OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
    
    def _get_embedding(self, text: str, retries: int = 3) -> Optional[np.ndarray]:
        """Get embedding for a single text with retry logic."""
        for attempt in range(retries):
            try:
                response = openai.embeddings.create(
                    model=self.model,
                    input=text
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                if attempt == retries - 1:
                    print(f"Failed to get embedding after {retries} attempts: {str(e)}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def embed_texts(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Get embeddings for a list of texts with batching."""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = []
            
            for text in batch:
                # Check cache first
                if text in self.cache:
                    batch_embeddings.append(self.cache[text])
                else:
                    embedding = self._get_embedding(text)
                    if embedding is not None:
                        self.cache[text] = embedding
                    batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            time.sleep(0.1)  # Rate limiting
        
        return embeddings
    
    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """Get embedding for a single query string."""
        return self._get_embedding(query)
    
    def save_cache(self, path: str) -> None:
        """Save embedding cache to disk."""
        cache_dict = {k: v.tolist() for k, v in self.cache.items()}
        with open(path, 'w') as f:
            json.dump(cache_dict, f)
    
    def load_cache(self, path: str) -> None:
        """Load embedding cache from disk."""
        if not Path(path).exists():
            return
            
        with open(path, 'r') as f:
            cache_dict = json.load(f)
            self.cache = {k: np.array(v) for k, v in cache_dict.items()}
    
    def process_chunks_file(self, chunks_file: str, output_file: str) -> None:
        """Process a JSONL file of chunks and add embeddings."""
        texts = []
        metadata_list = []
        
        # Read chunks
        with jsonlines.open(chunks_file) as reader:
            for item in reader:
                texts.append(item['text'])
                metadata_list.append(item['metadata'])
        
        # Get embeddings
        embeddings = self.embed_texts(texts)
        
        # Write results
        with jsonlines.open(output_file, mode='w') as writer:
            for text, embedding, metadata in zip(texts, embeddings, metadata_list):
                if embedding is not None:
                    writer.write({
                        'text': text,
                        'embedding': embedding.tolist(),
                        'metadata': metadata
                    }) 