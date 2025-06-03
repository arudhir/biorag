import json
import os
from typing import Dict

import openai

try:
    from .embedder import EmbeddingManager
    from .vector_store import VectorStore
except ImportError:
    from embedder import EmbeddingManager
    from vector_store import VectorStore


class BiologicalRAG:
    def __init__(self, vector_store: VectorStore, embedder: EmbeddingManager, top_k: int = 5):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=api_key)
    
    def forward(self, question: str, structured_output: bool = False) -> Dict:
        """Process a question through the RAG pipeline."""
        # Get query embedding
        query_embedding = self.embedder.embed_query(question)
        if query_embedding is None:
            return {
                "error": "Failed to generate query embedding",
                "answer": None,
                "citations": [],
                "confidence": 0.0
            }
        
        # Retrieve relevant chunks
        results = self.vector_store.search(query_embedding, k=self.top_k)
        
        if not results:
            return {
                "error": "No relevant documents found",
                "answer": None,
                "citations": [],
                "confidence": 0.0
            }
        
        # Format context for generation
        context = "\n\n".join([
            f"Source: {r['source']}, Page {r['page']}\n{r['text']}"
            for r in results
        ])
        
        # Generate answer
        try:
            if structured_output:
                # Create prompt for structured JSON output
                prompt = f"""Based on the following context from scientific papers about mitochondria, answer the question and format your response as valid JSON.

Context:
{context}

Question: {question}

Please return your answer as a JSON object with the following structure:
{{
  "pathways": [
    {{
      "name": "pathway name",
      "genes": ["GENE1", "GENE2", "GENE3"],
      "description": "detailed description of the pathway and its role"
    }}
  ],
  "key_findings": [
    {{
      "finding": "important finding or mechanism",
      "evidence": "supporting evidence from the context"
    }}
  ],
  "citations": [
    {{
      "source": "paper_name.pdf",
      "page": 1,
      "relevance": "high"
    }}
  ]
}}

Ensure the response is valid JSON. Extract pathways, genes, and key findings from the provided context."""

                system_message = "You are a scientific assistant that returns structured JSON responses. Always return valid JSON formatted according to the requested schema. Focus on biological pathways, genes, and mechanisms."
            else:
                # Standard prompt for text response
                prompt = f"""Based on the following context from scientific papers about mitochondria, answer the question.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the scientific evidence provided in the context. Include specific details and mechanisms when available."""

                system_message = "You are a helpful scientific assistant specializing in biology and mitochondrial research. Provide accurate, detailed answers based on the provided scientific context."

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800 if structured_output else 500
            )
            
            answer = response.choices[0].message.content
            
            # Format response
            if structured_output:
                # Try to parse JSON response
                try:
                    structured_answer = json.loads(answer)
                    # Enhance citations with our metadata
                    if "citations" in structured_answer:
                        for i, citation in enumerate(structured_answer["citations"]):
                            if i < len(results):
                                citation.update({
                                    "chunk_id": results[i]["chunk_id"],
                                    "relevance_score": results[i]["relevance_score"]
                                })
                    
                    response_dict = {
                        "question": question,
                        "structured_answer": structured_answer,
                        "raw_answer": answer,
                        "citations": [
                            {
                                "source": r["source"],
                                "page": r["page"],
                                "chunk_id": r["chunk_id"],
                                "relevance_score": r["relevance_score"]
                            }
                            for r in results
                        ],
                        "confidence": float(max(r["relevance_score"] for r in results)),
                        "output_type": "structured"
                    }
                except json.JSONDecodeError as e:
                    # Fallback to standard response if JSON parsing fails
                    response_dict = {
                        "question": question,
                        "answer": answer,
                        "error": f"Failed to parse structured output: {str(e)}",
                        "citations": [
                            {
                                "source": r["source"],
                                "page": r["page"],
                                "chunk_id": r["chunk_id"],
                                "relevance_score": r["relevance_score"]
                            }
                            for r in results
                        ],
                        "confidence": float(max(r["relevance_score"] for r in results)),
                        "output_type": "fallback"
                    }
            else:
                response_dict = {
                    "question": question,
                    "answer": answer,
                    "citations": [
                        {
                            "source": r["source"],
                            "page": r["page"],
                            "chunk_id": r["chunk_id"],
                            "relevance_score": r["relevance_score"]
                        }
                        for r in results
                    ],
                    "confidence": float(max(r["relevance_score"] for r in results)),
                    "output_type": "standard"
                }
            
            return response_dict
            
        except Exception as e:
            return {
                "error": f"Failed to generate answer: {str(e)}",
                "answer": None,
                "citations": [],
                "confidence": 0.0
            }
    
    async def query(self, question: str) -> Dict:
        """Alias for forward method to maintain compatibility with bio-intelligence modules"""
        return self.forward(question, structured_output=False)
        
    def retrieve_context(self, question: str) -> Dict:
        """Retrieve context without full response generation"""
        chunks = self.vector_store.search(question, k=self.top_k)
        return {
            "retrieved_chunks": chunks,
            "citations": [{"source": chunk.get("source", ""), "page": chunk.get("page", 1)} for chunk in chunks]
        }

    def format_response(self, response: Dict) -> str:
        """Format the response for display."""
        if response.get("error"):
            return f"Error: {response['error']}"
        
        output = []
        output.append(f"Question: {response['question']}\n")
        
        if response.get("output_type") == "structured" and "structured_answer" in response:
            output.append("Structured Answer:")
            output.append(json.dumps(response["structured_answer"], indent=2))
            output.append("")
        else:
            output.append(f"Answer: {response['answer']}\n")
        
        output.append("\nCitations:")
        for citation in response["citations"]:
            output.append(
                f"- {citation['source']} (Page {citation['page']}, "
                f"Relevance: {citation['relevance_score']:.2f})"
            )
        
        return "\n".join(output) 