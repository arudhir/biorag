#!/usr/bin/env python3
"""
Pathway Analysis Script

Analyze pathway gaps and completeness across biological databases.

Usage:
    uv run scripts/07_pathway_analysis.py --gene "NRF2" --compare-databases
    uv run scripts/07_pathway_analysis.py --pathway "oxidative stress response" --find-gaps
    uv run scripts/07_pathway_analysis.py --genes-file genes.txt --batch-analysis
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
import dspy

from bio_intelligence import BioAPIClient, GeneResolver, PathwayGapDetector
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


async def analyze_gene_gaps(gene: str, rag_pipeline: BiologicalRAG, 
                          bio_client: BioAPIClient) -> dict:
    """Analyze pathway gaps for a single gene"""
    
    gap_detector = PathwayGapDetector(rag_pipeline, bio_client)
    
    print(f"Analyzing pathway gaps for: {gene}")
    print("=" * 60)
    
    # Perform gap detection
    report = await gap_detector.forward(gene)
    
    # Display results
    print("\n🔍 PATHWAY GAP ANALYSIS")
    print(f"Gene: {report.gene}")
    
    print("\n📊 DATABASE COVERAGE:")
    print(f"  Curated upstream regulators: {len(report.curated_upstream)}")
    print(f"  Curated downstream targets: {len(report.curated_downstream)}")
    print(f"  Literature upstream mentions: {len(report.literature_upstream)}")
    print(f"  Literature downstream mentions: {len(report.literature_downstream)}")
    
    total_gaps = len(report.upstream_gaps) + len(report.downstream_gaps)
    print(f"\n🔍 GAPS DETECTED: {total_gaps}")
    
    if report.upstream_gaps:
        print("\n⬆️ UPSTREAM GAPS (in literature but not databases):")
        for i, gap in enumerate(report.upstream_gaps, 1):
            print(f"  {i}. {gap}")
            
    if report.downstream_gaps:
        print("\n⬇️ DOWNSTREAM GAPS (in literature but not databases):")
        for i, gap in enumerate(report.downstream_gaps, 1):
            print(f"  {i}. {gap}")
            
    if not report.upstream_gaps and not report.downstream_gaps:
        print("  ✅ No significant gaps detected between literature and databases")
        
    if report.literature_sources:
        print(f"\n📚 LITERATURE SOURCES: {len(report.literature_sources)}")
        
    # Convert to exportable format
    return {
        "gene": report.gene,
        "curated_data": {
            "upstream_regulators": report.curated_upstream,
            "downstream_targets": report.curated_downstream
        },
        "literature_data": {
            "upstream_regulators": report.literature_upstream,
            "downstream_targets": report.literature_downstream
        },
        "gaps": {
            "upstream_gaps": report.upstream_gaps,
            "downstream_gaps": report.downstream_gaps,
            "total_gaps": total_gaps
        },
        "literature_sources_count": len(report.literature_sources),
        "analysis_timestamp": datetime.now().isoformat()
    }


async def batch_gene_analysis(genes: list, rag_pipeline: BiologicalRAG, 
                            bio_client: BioAPIClient) -> dict:
    """Analyze pathway gaps for multiple genes"""
    
    gap_detector = PathwayGapDetector(rag_pipeline, bio_client)
    
    print(f"Performing batch analysis for {len(genes)} genes...")
    print("=" * 60)
    
    # Perform batch gap detection
    reports = await gap_detector.batch_gap_detection(genes)
    
    # Generate summary
    summary = gap_detector.summarize_gaps(reports)
    
    print("\n📊 BATCH ANALYSIS SUMMARY")
    print(f"Total genes analyzed: {summary['total_genes_analyzed']}")
    print(f"Genes with upstream gaps: {summary['genes_with_upstream_gaps']}")
    print(f"Genes with downstream gaps: {summary['genes_with_downstream_gaps']}")
    
    if summary['most_common_missing_upstream']:
        print("\n⬆️ MOST COMMON MISSING UPSTREAM REGULATORS:")
        for regulator, count in list(summary['most_common_missing_upstream'].items())[:5]:
            print(f"  • {regulator}: mentioned for {count} genes")
            
    if summary['most_common_missing_downstream']:
        print("\n⬇️ MOST COMMON MISSING DOWNSTREAM TARGETS:")
        for target, count in list(summary['most_common_missing_downstream'].items())[:5]:
            print(f"  • {target}: mentioned for {count} genes")
            
    if summary['high_confidence_gaps']:
        print(f"\n🎯 HIGH CONFIDENCE GAPS ({len(summary['high_confidence_gaps'])}):")
        for gap in summary['high_confidence_gaps'][:5]:
            print(f"  • {gap['gene']}: {gap['missing_element']} ({gap['gap_type']})")
            
    # Individual gene reports
    print("\n📋 INDIVIDUAL GENE REPORTS:")
    for gene, report in reports.items():
        gap_count = len(report.upstream_gaps) + len(report.downstream_gaps)
        print(f"  • {gene}: {gap_count} gaps detected")
        
    return {
        "summary": summary,
        "individual_reports": {
            gene: {
                "upstream_gaps": report.upstream_gaps,
                "downstream_gaps": report.downstream_gaps,
                "total_gaps": len(report.upstream_gaps) + len(report.downstream_gaps)
            }
            for gene, report in reports.items()
        },
        "batch_timestamp": datetime.now().isoformat()
    }


async def analyze_pathway_genes(pathway_query: str, rag_pipeline: BiologicalRAG, 
                              bio_client: BioAPIClient) -> dict:
    """Find genes in a pathway and analyze their gaps"""
    
    gene_resolver = GeneResolver(bio_client)
    gap_detector = PathwayGapDetector(rag_pipeline, bio_client)
    
    print(f"Analyzing pathway: {pathway_query}")
    print("=" * 60)
    
    # Extract genes from pathway query
    gene_data = await gene_resolver.forward(pathway_query)
    
    if not gene_data.get("gene_symbols"):
        print("❌ No genes identified in pathway query")
        return {
            "pathway_query": pathway_query,
            "genes_found": [],
            "analysis": "No genes identified",
            "timestamp": datetime.now().isoformat()
        }
        
    genes = gene_data["gene_symbols"]
    print(f"🧬 Identified {len(genes)} genes: {', '.join(genes)}")
    
    # Analyze gaps for pathway genes
    reports = await gap_detector.batch_gap_detection(genes)
    
    # Pathway-specific analysis
    pathway_gaps = {}
    total_pathway_gaps = 0
    
    for gene, report in reports.items():
        gene_gaps = len(report.upstream_gaps) + len(report.downstream_gaps)
        pathway_gaps[gene] = gene_gaps
        total_pathway_gaps += gene_gaps
        
    print("\n🔍 PATHWAY GAP ANALYSIS:")
    print(f"Total gaps across pathway: {total_pathway_gaps}")
    print(f"Average gaps per gene: {total_pathway_gaps / len(genes):.1f}")
    
    # Identify pathway-level patterns
    all_upstream = []
    all_downstream = []
    
    for report in reports.values():
        all_upstream.extend(report.upstream_gaps)
        all_downstream.extend(report.downstream_gaps)
        
    common_upstream = Counter(all_upstream)
    common_downstream = Counter(all_downstream)
    
    if common_upstream:
        print("\n⬆️ PATHWAY UPSTREAM REGULATORS (missing from databases):")
        for regulator, count in common_upstream.most_common(5):
            print(f"  • {regulator}: affects {count} genes in pathway")
            
    if common_downstream:
        print("\n⬇️ PATHWAY DOWNSTREAM TARGETS (missing from databases):")
        for target, count in common_downstream.most_common(5):
            print(f"  • {target}: regulated by {count} genes in pathway")
            
    return {
        "pathway_query": pathway_query,
        "genes_found": genes,
        "pathway_gap_summary": {
            "total_gaps": total_pathway_gaps,
            "average_gaps_per_gene": total_pathway_gaps / len(genes),
            "genes_with_gaps": sum(1 for gaps in pathway_gaps.values() if gaps > 0)
        },
        "common_missing_regulators": dict(common_upstream.most_common(10)),
        "common_missing_targets": dict(common_downstream.most_common(10)),
        "individual_gene_gaps": pathway_gaps,
        "timestamp": datetime.now().isoformat()
    }


async def main():
    parser = argparse.ArgumentParser(description="Pathway gap analysis")
    parser.add_argument("--gene", type=str, help="Single gene symbol to analyze")
    parser.add_argument("--genes-file", type=str, help="File with gene symbols (one per line)")
    parser.add_argument("--pathway", type=str, help="Pathway name or description to analyze")
    parser.add_argument("--index-path", type=str, default="data/faiss_index", 
                       help="Path to FAISS index")
    parser.add_argument("--compare-databases", action="store_true",
                       help="Compare coverage across databases")
    parser.add_argument("--find-gaps", action="store_true", default=True,
                       help="Find gaps between literature and databases")
    parser.add_argument("--batch-analysis", action="store_true",
                       help="Perform batch analysis with summary")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Determine analysis type and targets
    genes = []
    pathway_query = None
    
    if args.gene:
        genes = [args.gene]
    elif args.genes_file:
        if os.path.exists(args.genes_file):
            with open(args.genes_file, 'r') as f:
                genes = [line.strip() for line in f if line.strip()]
        else:
            print(f"Error: Genes file not found: {args.genes_file}")
            return 1
    elif args.pathway:
        pathway_query = args.pathway
    else:
        print("Error: Must provide --gene, --genes-file, or --pathway")
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
        
        # Perform analysis based on input type
        if pathway_query:
            result = await analyze_pathway_genes(
                pathway_query, rag_pipeline, bio_client
            )
        elif len(genes) == 1 and not args.batch_analysis:
            result = await analyze_gene_gaps(
                genes[0], rag_pipeline, bio_client
            )
        else:
            result = await batch_gene_analysis(
                genes, rag_pipeline, bio_client
            )
            
        # Save results if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
            
        return 0
        
    except Exception as e:
        logging.error(f"Pathway analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)