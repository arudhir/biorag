#!/usr/bin/env python3
"""
Test script for structured JSON output functionality.
"""
import sys
import os
import json

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.embedder import EmbeddingManager
from src.vector_store import VectorStore
from src.rag_pipeline import BiologicalRAG

def test_structured_output():
    """Test the structured JSON output feature."""
    
    # Initialize components
    embedder = EmbeddingManager(model="text-embedding-3-small")
    vector_store = VectorStore()
    
    # Load existing index
    index_path = "data/faiss_index"
    if not os.path.exists(index_path):
        print(f"Error: Index directory '{index_path}' does not exist")
        print("Please run the indexing scripts first:")
        print("1. python scripts/01_extract_chunks.py")
        print("2. python scripts/02_embed_and_index.py")
        return
    
    vector_store.load_index(index_path)
    
    rag = BiologicalRAG(
        vector_store=vector_store,
        embedder=embedder,
        top_k=5
    )
    
    # Test queries
    test_queries = [
        "What pathways are involved in mitochondrial ROS detoxification?",
        "What genes regulate mitochondrial biogenesis?",
        "How do mitochondria contribute to cellular energy production?"
    ]
    
    print("Testing Structured JSON Output")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 40)
        
        # Test structured output
        response = rag.forward(query, structured_output=True)
        
        if response.get("output_type") == "structured":
            print("✅ Structured output generated successfully")
            print("\nStructured Answer:")
            print(json.dumps(response["structured_answer"], indent=2))
            
            # Validate structure
            structured = response["structured_answer"]
            if "pathways" in structured and isinstance(structured["pathways"], list):
                print(f"✅ Found {len(structured['pathways'])} pathways")
            if "key_findings" in structured and isinstance(structured["key_findings"], list):
                print(f"✅ Found {len(structured['key_findings'])} key findings")
            if "citations" in structured and isinstance(structured["citations"], list):
                print(f"✅ Found {len(structured['citations'])} citations")
        else:
            print("❌ Failed to generate structured output")
            if "error" in response:
                print(f"Error: {response['error']}")
        
        print(f"\nConfidence: {response.get('confidence', 0):.3f}")
        print("\n" + "="*50)

if __name__ == "__main__":
    test_structured_output()