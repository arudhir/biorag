#!/usr/bin/env python3
"""
Gene Investigation Script

Perform comprehensive gene investigations using the bio-intelligence modules
including gene resolution, hypothesis generation, and gap detection.

Usage:
    uv run scripts/05_gene_investigation.py --query "SOD2 role in mitochondrial stress"
    uv run scripts/05_gene_investigation.py --gene "NRF2" --include-gaps --output results.json
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
import dspy

from bio_intelligence import BioAPIClient, ComprehensiveGeneInvestigation
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


async def perform_investigation(query: str, rag_pipeline: BiologicalRAG, 
                              bio_client: BioAPIClient, include_gaps: bool = True,
                              generate_mechanism: bool = False) -> dict:
    """Perform comprehensive gene investigation"""
    
    investigator = ComprehensiveGeneInvestigation(rag_pipeline, bio_client)
    
    print(f"Starting investigation for: {query}")
    print("=" * 60)
    
    # Perform investigation
    report = await investigator.forward(
        query=query,
        include_gaps=include_gaps,
        generate_mechanism=generate_mechanism
    )
    
    # Display results
    print("\n🔬 INVESTIGATION REPORT")
    print(f"Query: {report.query}")
    print(f"Primary Gene: {report.primary_gene or 'None identified'}")
    print(f"Confidence Score: {report.confidence_score:.2f}")
    print(f"Timestamp: {report.timestamp}")
    
    if report.gene_resolution.get("gene_symbols"):
        print("\n🧬 GENES IDENTIFIED:")
        for i, gene in enumerate(report.gene_resolution["gene_symbols"], 1):
            print(f"  {i}. {gene}")
            
    if report.hypothesis:
        print("\n💡 HYPOTHESIS:")
        print(f"  {report.hypothesis.hypothesis}")
        print(f"  Confidence: {report.hypothesis.confidence}")
        
        if report.hypothesis.testable_predictions:
            print("\n🔬 TESTABLE PREDICTIONS:")
            for i, prediction in enumerate(report.hypothesis.testable_predictions, 1):
                print(f"  {i}. {prediction}")
                
        if report.hypothesis.mechanistic_details:
            print("\n⚙️ MECHANISM:")
            print(f"  {report.hypothesis.mechanistic_details}")
            
    if report.pathway_gaps:
        gaps = report.pathway_gaps
        total_gaps = len(gaps.upstream_gaps) + len(gaps.downstream_gaps)
        print(f"\n🔍 PATHWAY GAPS DETECTED: {total_gaps}")
        
        if gaps.upstream_gaps:
            print("  Upstream regulators in literature but not databases:")
            for gap in gaps.upstream_gaps:
                print(f"    • {gap}")
                
        if gaps.downstream_gaps:
            print("  Downstream targets in literature but not databases:")
            for gap in gaps.downstream_gaps:
                print(f"    • {gap}")
                
    print("\n📋 SUMMARY:")
    print(f"  {report.investigation_summary}")
    
    if report.recommendations:
        print("\n💼 RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
            
    # Export to JSON format for return
    return investigator.export_report(report, format="json")


async def main():
    parser = argparse.ArgumentParser(description="Comprehensive gene investigation")
    parser.add_argument("--query", type=str, help="Biological query to investigate")
    parser.add_argument("--gene", type=str, help="Specific gene symbol to investigate")
    parser.add_argument("--index-path", type=str, default="data/faiss_index", 
                       help="Path to FAISS index")
    parser.add_argument("--include-gaps", action="store_true", default=True,
                       help="Include pathway gap detection")
    parser.add_argument("--generate-mechanism", action="store_true",
                       help="Generate detailed mechanistic explanations")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Determine query
    if args.query:
        query = args.query
    elif args.gene:
        query = f"biological function and regulation of {args.gene} gene"
    else:
        print("Error: Must provide either --query or --gene")
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
        
        # Perform investigation
        result = await perform_investigation(
            query=query,
            rag_pipeline=rag_pipeline,
            bio_client=bio_client,
            include_gaps=args.include_gaps,
            generate_mechanism=args.generate_mechanism
        )
        
        # Save results if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
            
        return 0
        
    except Exception as e:
        logging.error(f"Investigation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)