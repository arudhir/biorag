#!/usr/bin/env python3
"""
BioRAG CLI - Unified command-line interface for the Biological RAG System

A comprehensive tool for biological literature analysis, gene investigation,
and hypothesis generation using advanced RAG and bio-intelligence capabilities.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
import dspy

# Import core modules
from embedder import EmbeddingManager
from pdf_processor import PDFProcessor
from rag_pipeline import BiologicalRAG
from vector_store import VectorStore

# Import bio-intelligence modules  
from bio_intelligence import (
    BioAPIClient, 
    ComprehensiveGeneInvestigation,
    HypothesisGenerator,
    PathwayGapDetector,
    GeneResolver
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def setup_dspy():
    """Configure DSPy with OpenAI"""
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    lm = dspy.LM(model=f"openai/{llm_model}", max_tokens=1000)
    dspy.configure(lm=lm)


async def load_rag_system(index_path: str) -> BiologicalRAG:
    """Load the RAG system components"""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index path not found: {index_path}")
        
    vector_store = VectorStore()
    vector_store.load_index(index_path)
    
    embedder = EmbeddingManager()
    rag_pipeline = BiologicalRAG(vector_store, embedder)
    return rag_pipeline


# ==============================================================================
# EXTRACT COMMAND
# ==============================================================================

def cmd_extract(args):
    """Extract chunks from PDF files"""
    print(f"🔄 Extracting chunks from PDFs in {args.input}")
    
    processor = PDFProcessor(chunk_size=args.chunk_size, overlap=args.overlap)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process PDFs
    chunks = processor.extract_chunks_from_directory(args.input)
    
    print(f"📄 Processed {len(chunks)} chunks")
    print(f"💾 Saving to {args.output}")
    
    processor.save_chunks(chunks, args.output)
    print("✅ Extraction complete!")


# ==============================================================================
# INDEX COMMAND  
# ==============================================================================

def cmd_index(args):
    """Create embeddings and build vector index"""
    print(f"🔄 Creating embeddings and index from {args.chunks}")
    
    # Create output directory
    index_path = Path(args.index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    embedder = EmbeddingManager(
        model=args.model,
        cache_path=args.cache_path
    )
    vector_store = VectorStore()
    
    # Generate embeddings and build index
    print("🧮 Generating embeddings...")
    embedder.embed_chunks_from_file(
        args.chunks, 
        batch_size=args.batch_size
    )
    
    print("🏗️ Building vector index...")
    vector_store.build_index_from_embeddings(
        embedder.embeddings,
        embedder.chunks
    )
    
    print(f"💾 Saving index to {args.index_path}")
    vector_store.save_index(str(index_path))
    
    print("✅ Indexing complete!")


# ==============================================================================
# QUERY COMMAND
# ==============================================================================

async def cmd_query(args):
    """Query the RAG system"""
    print(f"🔍 Loading RAG system from {args.index_path}")
    
    rag_pipeline = await load_rag_system(args.index_path)
    
    if args.query:
        # Single query mode
        print(f"❓ Query: {args.query}")
        result = rag_pipeline.forward(args.query, structured_output=args.structured)
        
        if args.structured:
            print("\n📊 Structured Results:")
            print(json.dumps(result, indent=2))
        else:
            print(f"\n💡 Answer: {result.get('answer', 'No answer generated')}")
            
            if result.get('citations'):
                print(f"\n📚 Citations ({len(result['citations'])}):")
                for i, citation in enumerate(result['citations'][:3], 1):
                    print(f"  {i}. {citation.get('source', 'Unknown')} (page {citation.get('page', 'N/A')})")
                    
        # Save results if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")
            
    else:
        # Interactive mode
        print("🎯 Interactive mode - type 'quit' to exit")
        while True:
            try:
                query = input("\n❓ Enter your question: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not query:
                    continue
                    
                result = rag_pipeline.forward(query, structured_output=args.structured)
                
                if args.structured:
                    print("\n📊 Results:")
                    print(json.dumps(result, indent=2))
                else:
                    print(f"\n💡 {result.get('answer', 'No answer generated')}")
                    
            except KeyboardInterrupt:
                break
                
        print("\n👋 Goodbye!")


# ==============================================================================
# INVESTIGATE COMMAND
# ==============================================================================

async def cmd_investigate(args):
    """Perform comprehensive gene investigation"""
    print(f"🔬 Starting gene investigation")
    
    # Determine query
    if args.query:
        query = args.query
    elif args.gene:
        query = f"biological function and regulation of {args.gene} gene"
    else:
        print("❌ Error: Must provide either --query or --gene")
        return 1
        
    print(f"📋 Query: {query}")
    
    # Load systems
    rag_pipeline = await load_rag_system(args.index_path)
    bio_client = BioAPIClient()
    investigator = ComprehensiveGeneInvestigation(rag_pipeline, bio_client)
    
    # Perform investigation
    print("🔄 Running investigation...")
    report = await investigator.forward(
        query=query,
        include_gaps=args.include_gaps,
        generate_mechanism=args.mechanism
    )
    
    # Display results
    print(f"\n🔬 INVESTIGATION REPORT")
    print(f"Query: {report.query}")
    print(f"Primary Gene: {report.primary_gene or 'None identified'}")
    print(f"Confidence Score: {report.confidence_score:.2f}")
    
    if report.gene_resolution.get("gene_symbols"):
        print(f"\n🧬 GENES IDENTIFIED ({len(report.gene_resolution['gene_symbols'])}):")
        for i, gene in enumerate(report.gene_resolution["gene_symbols"], 1):
            print(f"  {i}. {gene}")
            
    if report.hypothesis:
        print(f"\n💡 HYPOTHESIS ({report.hypothesis.confidence} confidence):")
        print(f"  {report.hypothesis.hypothesis}")
        
        if report.hypothesis.testable_predictions:
            print(f"\n🔬 TESTABLE PREDICTIONS ({len(report.hypothesis.testable_predictions)}):")
            for i, prediction in enumerate(report.hypothesis.testable_predictions, 1):
                print(f"  {i}. {prediction}")
                
    if report.pathway_gaps and (report.pathway_gaps.upstream_gaps or report.pathway_gaps.downstream_gaps):
        total_gaps = len(report.pathway_gaps.upstream_gaps) + len(report.pathway_gaps.downstream_gaps)
        print(f"\n🔍 PATHWAY GAPS DETECTED: {total_gaps}")
        
        if report.pathway_gaps.upstream_gaps:
            print(f"  ⬆️ Upstream gaps: {', '.join(report.pathway_gaps.upstream_gaps[:5])}")
        if report.pathway_gaps.downstream_gaps:
            print(f"  ⬇️ Downstream gaps: {', '.join(report.pathway_gaps.downstream_gaps[:5])}")
            
    print(f"\n📋 SUMMARY:")
    print(f"  {report.investigation_summary}")
    
    # Export results
    if args.output:
        result = investigator.export_report(report, format="json")
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
        
    print("✅ Investigation complete!")


# ==============================================================================
# HYPOTHESIZE COMMAND
# ==============================================================================

async def cmd_hypothesize(args):
    """Generate biological hypotheses"""
    print(f"💡 Generating hypothesis for: {args.topic}")
    
    # Load systems
    rag_pipeline = await load_rag_system(args.index_path)
    bio_client = BioAPIClient()
    generator = HypothesisGenerator(rag_pipeline, bio_client)
    
    # Generate hypothesis
    print("🔄 Generating hypothesis...")
    hypothesis = await generator.forward(
        topic=args.topic,
        use_context=args.grounded,
        generate_mechanism=args.mechanism
    )
    
    # Display results
    print(f"\n💡 BIOLOGICAL HYPOTHESIS")
    print(f"Topic: {hypothesis.topic}")
    print(f"Confidence: {hypothesis.confidence}")
    
    print(f"\n📝 HYPOTHESIS:")
    print(f"  {hypothesis.hypothesis}")
    
    if hypothesis.testable_predictions:
        print(f"\n🔬 TESTABLE PREDICTIONS ({len(hypothesis.testable_predictions)}):")
        for i, prediction in enumerate(hypothesis.testable_predictions, 1):
            print(f"  {i}. {prediction}")
            
    if hypothesis.mechanistic_details:
        print(f"\n⚙️ MECHANISM:")
        print(f"  {hypothesis.mechanistic_details}")
        
    if hypothesis.literature_sources:
        print(f"\n📚 LITERATURE SUPPORT ({len(hypothesis.literature_sources)} sources)")
        
    # Export results
    if args.output:
        result = {
            "topic": hypothesis.topic,
            "hypothesis": hypothesis.hypothesis,
            "confidence": hypothesis.confidence,
            "testable_predictions": hypothesis.testable_predictions,
            "mechanistic_details": hypothesis.mechanistic_details,
            "literature_sources": len(hypothesis.literature_sources),
            "pathway_evidence": len(hypothesis.pathway_evidence)
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
        
    print("✅ Hypothesis generation complete!")


# ==============================================================================
# ANALYZE COMMAND
# ==============================================================================

async def cmd_analyze(args):
    """Analyze pathway gaps"""
    print(f"🔍 Analyzing pathway gaps")
    
    # Load systems
    rag_pipeline = await load_rag_system(args.index_path)
    bio_client = BioAPIClient()
    gap_detector = PathwayGapDetector(rag_pipeline, bio_client)
    
    if args.gene:
        # Single gene analysis
        print(f"🧬 Analyzing gene: {args.gene}")
        report = await gap_detector.forward(args.gene)
        
        total_gaps = len(report.upstream_gaps) + len(report.downstream_gaps)
        print(f"\n📊 PATHWAY GAP ANALYSIS")
        print(f"Gene: {report.gene}")
        print(f"Total gaps detected: {total_gaps}")
        
        if report.upstream_gaps:
            print(f"\n⬆️ UPSTREAM GAPS ({len(report.upstream_gaps)}):")
            for gap in report.upstream_gaps[:5]:
                print(f"  • {gap}")
                
        if report.downstream_gaps:
            print(f"\n⬇️ DOWNSTREAM GAPS ({len(report.downstream_gaps)}):")
            for gap in report.downstream_gaps[:5]:
                print(f"  • {gap}")
                
        if not report.upstream_gaps and not report.downstream_gaps:
            print("  ✅ No significant gaps detected")
            
    elif args.genes_file:
        # Batch analysis
        if not os.path.exists(args.genes_file):
            print(f"❌ Error: Genes file not found: {args.genes_file}")
            return 1
            
        with open(args.genes_file, 'r') as f:
            genes = [line.strip() for line in f if line.strip()]
            
        print(f"🧬 Analyzing {len(genes)} genes from {args.genes_file}")
        reports = await gap_detector.batch_gap_detection(genes)
        summary = gap_detector.summarize_gaps(reports)
        
        print(f"\n📊 BATCH ANALYSIS SUMMARY")
        print(f"Genes analyzed: {summary['total_genes_analyzed']}")
        print(f"Genes with upstream gaps: {summary['genes_with_upstream_gaps']}")
        print(f"Genes with downstream gaps: {summary['genes_with_downstream_gaps']}")
        
        if summary['most_common_missing_upstream']:
            print(f"\n⬆️ TOP MISSING UPSTREAM REGULATORS:")
            for reg, count in list(summary['most_common_missing_upstream'].items())[:5]:
                print(f"  • {reg}: {count} genes")
                
    # Export results
    if args.output:
        if args.gene:
            result = {
                "gene": report.gene,
                "upstream_gaps": report.upstream_gaps,
                "downstream_gaps": report.downstream_gaps,
                "total_gaps": total_gaps
            }
        else:
            result = summary
            
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
        
    print("✅ Analysis complete!")


# ==============================================================================
# MAIN CLI SETUP
# ==============================================================================

def create_parser():
    """Create the main argument parser with subcommands"""
    parser = argparse.ArgumentParser(
        prog='biorag',
        description='🧬 BioRAG - Biological Literature Analysis and Gene Investigation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  biorag extract papers/ --output data/chunks.jsonl
  biorag index data/chunks.jsonl --index-path data/faiss_index/
  biorag query --index-path data/faiss_index/ --query "How does SOD2 work?"
  biorag investigate --gene TP53 --include-gaps --output results.json
  biorag hypothesize "ROS in Alzheimer's disease" --grounded --mechanism
  biorag analyze --gene NRF2 --output gaps.json

For more help on a specific command: biorag <command> --help
        """)
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # EXTRACT command
    extract_parser = subparsers.add_parser('extract', help='Extract chunks from PDF files')
    extract_parser.add_argument('input', help='Input directory containing PDF files')
    extract_parser.add_argument('--output', '-o', default='data/chunks.jsonl', 
                               help='Output JSONL file path (default: data/chunks.jsonl)')
    extract_parser.add_argument('--chunk-size', type=int, default=800,
                               help='Size of text chunks in tokens (default: 800)')
    extract_parser.add_argument('--overlap', type=int, default=200,
                               help='Overlap between chunks in tokens (default: 200)')
    
    # INDEX command
    index_parser = subparsers.add_parser('index', help='Create embeddings and build vector index')
    index_parser.add_argument('chunks', help='Input JSONL file with text chunks')
    index_parser.add_argument('--index-path', '-i', default='data/faiss_index/',
                             help='Output directory for FAISS index (default: data/faiss_index/)')
    index_parser.add_argument('--model', default='text-embedding-3-small',
                             help='Embedding model to use (default: text-embedding-3-small)')
    index_parser.add_argument('--batch-size', type=int, default=100,
                             help='Batch size for embedding generation (default: 100)')
    index_parser.add_argument('--cache-path', help='Path to save/load embedding cache')
    
    # QUERY command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('--index-path', '-i', default='data/faiss_index/',
                             help='Path to FAISS index directory (default: data/faiss_index/)')
    query_parser.add_argument('--query', '-q', help='Query to process (if not provided, enters interactive mode)')
    query_parser.add_argument('--structured', '-s', action='store_true',
                             help='Return structured JSON output')
    query_parser.add_argument('--top-k', type=int, default=5,
                             help='Number of chunks to retrieve (default: 5)')
    query_parser.add_argument('--output', '-o', help='Output JSON file for results')
    
    # INVESTIGATE command
    investigate_parser = subparsers.add_parser('investigate', help='Perform comprehensive gene investigation')
    investigate_parser.add_argument('--query', '-q', help='Biological query to investigate')
    investigate_parser.add_argument('--gene', '-g', help='Specific gene symbol to investigate')
    investigate_parser.add_argument('--index-path', '-i', default='data/faiss_index/',
                                   help='Path to FAISS index (default: data/faiss_index/)')
    investigate_parser.add_argument('--include-gaps', action='store_true', default=True,
                                   help='Include pathway gap detection (default: True)')
    investigate_parser.add_argument('--mechanism', action='store_true',
                                   help='Generate detailed mechanistic explanations')
    investigate_parser.add_argument('--output', '-o', help='Output JSON file path')
    
    # HYPOTHESIZE command
    hypothesize_parser = subparsers.add_parser('hypothesize', help='Generate biological hypotheses')
    hypothesize_parser.add_argument('topic', help='Biological topic for hypothesis generation')
    hypothesize_parser.add_argument('--index-path', '-i', default='data/faiss_index/',
                                   help='Path to FAISS index (default: data/faiss_index/)')
    hypothesize_parser.add_argument('--grounded', action='store_true', default=True,
                                   help='Use literature and pathway grounding (default: True)')
    hypothesize_parser.add_argument('--simple', action='store_true',
                                   help='Generate simple hypothesis without grounding')
    hypothesize_parser.add_argument('--mechanism', action='store_true',
                                   help='Generate detailed mechanistic explanations')
    hypothesize_parser.add_argument('--output', '-o', help='Output JSON file path')
    
    # ANALYZE command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze pathway gaps')
    analyze_parser.add_argument('--gene', '-g', help='Single gene symbol to analyze')
    analyze_parser.add_argument('--genes-file', '-f', help='File with gene symbols (one per line)')
    analyze_parser.add_argument('--index-path', '-i', default='data/faiss_index/',
                               help='Path to FAISS index (default: data/faiss_index/)')
    analyze_parser.add_argument('--output', '-o', help='Output JSON file path')
    
    return parser


async def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Check if no command provided
    if not args.command:
        parser.print_help()
        return 0
        
    # Setup logging
    setup_logging(args.verbose)
    
    # Load environment variables
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("   Please create a .env file with your OpenAI API key")
        return 1
    
    # Configure DSPy for bio-intelligence commands
    if args.command in ['investigate', 'hypothesize', 'analyze']:
        setup_dspy()
    
    try:
        # Route to appropriate command handler
        if args.command == 'extract':
            cmd_extract(args)
        elif args.command == 'index':
            cmd_index(args)
        elif args.command == 'query':
            await cmd_query(args)
        elif args.command == 'investigate':
            await cmd_investigate(args)
        elif args.command == 'hypothesize':
            if args.simple:
                args.grounded = False
            await cmd_hypothesize(args)
        elif args.command == 'analyze':
            await cmd_analyze(args)
        else:
            parser.print_help()
            return 1
            
        return 0
        
    except Exception as e:
        logging.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)