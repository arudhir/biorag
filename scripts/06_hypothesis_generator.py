#!/usr/bin/env python3
"""
Hypothesis Generator Script

Generate literature-grounded biological hypotheses using DSPy and RAG.

Usage:
    uv run scripts/06_hypothesis_generator.py --topic "ROS in Alzheimer's disease"
    uv run scripts/06_hypothesis_generator.py --topic "NADPH oxidase diabetes" --grounded --mechanism
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
import dspy

from bio_intelligence import BioAPIClient, HypothesisGenerator
from rag_pipeline import BiologicalRAG
from vector_store import VectorStore
from embedder import EmbeddingManager


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def load_rag_system(index_path: str) -> BiologicalRAG:
    """Load the existing RAG system"""
    print(f"Loading RAG system from {index_path}...")
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index path not found: {index_path}")
        
    vector_store = VectorStore()
    vector_store.load_index(index_path)
    
    embedder = EmbeddingManager()
    rag_pipeline = BiologicalRAG(vector_store, embedder)
    return rag_pipeline


async def generate_hypothesis(topic: str, rag_pipeline: BiologicalRAG, 
                            bio_client: BioAPIClient, grounded: bool = True,
                            generate_mechanism: bool = False) -> dict:
    """Generate biological hypothesis"""
    
    generator = HypothesisGenerator(rag_pipeline, bio_client)
    
    print(f"Generating hypothesis for: {topic}")
    print("=" * 60)
    
    # Generate hypothesis
    hypothesis = await generator.forward(
        topic=topic,
        use_context=grounded,
        generate_mechanism=generate_mechanism
    )
    
    # Display results
    print("\n💡 BIOLOGICAL HYPOTHESIS")
    print(f"Topic: {hypothesis.topic}")
    print(f"Confidence: {hypothesis.confidence}")
    
    print("\n📝 HYPOTHESIS:")
    print(f"  {hypothesis.hypothesis}")
    
    if hypothesis.testable_predictions:
        print("\n🔬 TESTABLE PREDICTIONS:")
        for i, prediction in enumerate(hypothesis.testable_predictions, 1):
            print(f"  {i}. {prediction}")
            
    if hypothesis.mechanistic_details:
        print("\n⚙️ MECHANISTIC DETAILS:")
        print(f"  {hypothesis.mechanistic_details}")
        
    if hypothesis.literature_sources:
        print(f"\n📚 LITERATURE SUPPORT ({len(hypothesis.literature_sources)} sources):")
        for i, source in enumerate(hypothesis.literature_sources[:3], 1):
            print(f"  {i}. {source.get('text', '')[:100]}...")
            
    if hypothesis.pathway_evidence:
        print("\n🧬 PATHWAY EVIDENCE:")
        for evidence in hypothesis.pathway_evidence[:3]:
            gene = evidence.get('gene', 'Unknown')
            pathway_count = sum(len(pathways) for pathways in evidence.get('pathways', {}).values())
            print(f"  • {gene}: {pathway_count} pathways")
            
    # Convert to exportable format
    return {
        "topic": hypothesis.topic,
        "hypothesis": hypothesis.hypothesis,
        "confidence": hypothesis.confidence,
        "testable_predictions": hypothesis.testable_predictions,
        "mechanistic_details": hypothesis.mechanistic_details,
        "literature_sources": [
            {
                "text": source.get("text", ""),
                "source": source.get("source", ""),
                "relevance": source.get("relevance", "")
            }
            for source in hypothesis.literature_sources
        ],
        "pathway_evidence": hypothesis.pathway_evidence,
        "generation_method": "grounded" if grounded else "simple",
        "timestamp": datetime.now().isoformat()
    }


async def compare_hypotheses(topics: list, rag_pipeline: BiologicalRAG, 
                           bio_client: BioAPIClient) -> dict:
    """Generate and compare multiple hypotheses"""
    
    generator = HypothesisGenerator(rag_pipeline, bio_client)
    
    print(f"Generating hypotheses for {len(topics)} topics...")
    print("=" * 60)
    
    # Generate hypotheses in parallel
    tasks = [
        generator.forward(topic, use_context=True, generate_mechanism=False)
        for topic in topics
    ]
    
    hypotheses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter successful results
    valid_hypotheses = [
        h for h in hypotheses 
        if not isinstance(h, Exception)
    ]
    
    print("\n📊 HYPOTHESIS COMPARISON")
    print(f"Successfully generated: {len(valid_hypotheses)}/{len(topics)}")
    
    # Compare by confidence and support
    for i, hypothesis in enumerate(valid_hypotheses, 1):
        lit_support = len(hypothesis.literature_sources)
        pathway_support = len(hypothesis.pathway_evidence)
        prediction_count = len(hypothesis.testable_predictions)
        
        print(f"\n{i}. {hypothesis.topic}")
        print(f"   Confidence: {hypothesis.confidence}")
        print(f"   Literature sources: {lit_support}")
        print(f"   Pathway evidence: {pathway_support}")
        print(f"   Predictions: {prediction_count}")
        print(f"   Hypothesis: {hypothesis.hypothesis[:100]}...")
        
    return {
        "topics": topics,
        "hypotheses_generated": len(valid_hypotheses),
        "hypotheses": [
            {
                "topic": h.topic,
                "hypothesis": h.hypothesis,
                "confidence": h.confidence,
                "support_score": len(h.literature_sources) + len(h.pathway_evidence)
            }
            for h in valid_hypotheses
        ],
        "comparison_timestamp": datetime.now().isoformat()
    }


async def main():
    parser = argparse.ArgumentParser(description="Biological hypothesis generation")
    parser.add_argument("--topic", type=str, help="Biological topic for hypothesis generation")
    parser.add_argument("--topics-file", type=str, help="File with multiple topics (one per line)")
    parser.add_argument("--index-path", type=str, default="data/faiss_index", 
                       help="Path to FAISS index")
    parser.add_argument("--grounded", action="store_true", default=True,
                       help="Use literature and pathway grounding (default)")
    parser.add_argument("--simple", action="store_true",
                       help="Generate simple hypothesis without grounding")
    parser.add_argument("--mechanism", action="store_true",
                       help="Generate detailed mechanistic explanations")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple hypotheses")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Determine topics
    topics = []
    if args.topic:
        topics = [args.topic]
    elif args.topics_file:
        if os.path.exists(args.topics_file):
            with open(args.topics_file, 'r') as f:
                topics = [line.strip() for line in f if line.strip()]
        else:
            print(f"Error: Topics file not found: {args.topics_file}")
            return 1
    else:
        print("Error: Must provide either --topic or --topics-file")
        return 1
        
    # Load environment variables
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment")
        return 1
        
    # Configure DSPy with OpenAI
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    lm = dspy.LM(model=f"openai/{llm_model}", max_tokens=1000)
    dspy.configure(lm=lm)
        
    try:
        # Initialize systems
        rag_pipeline = await load_rag_system(args.index_path)
        bio_client = BioAPIClient()
        
        # Determine grounding mode
        grounded = not args.simple  # Default to grounded unless --simple specified
        
        if len(topics) == 1 and not args.compare:
            # Single hypothesis generation
            result = await generate_hypothesis(
                topic=topics[0],
                rag_pipeline=rag_pipeline,
                bio_client=bio_client,
                grounded=grounded,
                generate_mechanism=args.mechanism
            )
        else:
            # Multiple hypothesis comparison
            result = await compare_hypotheses(
                topics=topics,
                rag_pipeline=rag_pipeline,
                bio_client=bio_client
            )
            
        # Save results if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
            
        return 0
        
    except Exception as e:
        logging.error(f"Hypothesis generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)