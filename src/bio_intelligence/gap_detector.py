"""
Gap Detector Module

Detect discrepancies between literature mentions and curated pathway databases
to identify potential knowledge gaps in biological understanding.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import dspy

try:
    from .bio_apis import BioAPIClient, PathwayInfo
    from .gene_resolver import GeneResolver
except ImportError:
    from bio_apis import BioAPIClient, PathwayInfo
    from gene_resolver import GeneResolver

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeGap:
    gap_type: str  # "upstream_regulator", "downstream_target", "pathway_missing", "interaction_missing"
    gene_of_interest: str
    missing_element: str
    evidence_source: str
    confidence: float
    literature_context: str


@dataclass
class PathwayGapReport:
    gene: str
    curated_upstream: List[str]
    curated_downstream: List[str]
    literature_upstream: List[str]
    literature_downstream: List[str]
    upstream_gaps: List[str]
    downstream_gaps: List[str]
    literature_sources: List[Dict[str, Any]]
    pathway_gaps: List[KnowledgeGap]


class RegulatorExtractor(dspy.Signature):
    """Extract upstream regulators and downstream targets from literature"""
    literature_passages = dspy.InputField(desc="Scientific literature passages about gene regulation")
    gene_of_interest = dspy.InputField(desc="The gene being investigated")
    upstream_regulators = dspy.OutputField(desc="Genes or proteins that regulate the gene of interest")
    downstream_targets = dspy.OutputField(desc="Genes or proteins regulated by the gene of interest")


class PathwayAnalyzer(dspy.Signature):
    """Analyze pathway completeness and identify potential gaps"""
    curated_pathways = dspy.InputField(desc="Curated pathway data from databases")
    literature_evidence = dspy.InputField(desc="Literature evidence about the gene")
    gene_name = dspy.InputField(desc="Gene being analyzed")
    missing_interactions = dspy.OutputField(desc="Potential missing interactions not in curated pathways")
    pathway_gaps = dspy.OutputField(desc="Gaps between literature and curated pathway data")


class PathwayGapDetector(dspy.Module):
    """Detect gaps between literature mentions and curated pathway databases"""
    
    def __init__(self, rag_pipeline, bio_client: BioAPIClient):
        super().__init__()
        self.rag_pipeline = rag_pipeline
        self.bio_client = bio_client
        self.gene_resolver = GeneResolver(bio_client)
        
        # DSPy modules for analysis
        self.extract_regulators = dspy.ChainOfThought(RegulatorExtractor)
        self.analyze_pathways = dspy.ChainOfThought(PathwayAnalyzer)
        
    def _extract_curated_regulators(self, pathways: Dict[str, List[PathwayInfo]], 
                                   direction: str = "upstream") -> List[str]:
        """Extract known regulators from curated pathway data"""
        regulators = set()
        
        for source, pathway_list in pathways.items():
            for pathway in pathway_list:
                if pathway.genes:
                    # For now, treat all genes in pathways as potential regulators
                    # In a real implementation, you'd parse pathway topology
                    regulators.update(pathway.genes)
                    
        return list(regulators)
        
    def _parse_literature_regulators(self, regulator_text: str) -> List[str]:
        """Parse regulator genes from LLM output"""
        if not regulator_text:
            return []
            
        # Split by common delimiters and clean
        regulators = []
        for item in re.split(r'[,;]|\band\b', regulator_text):
            item = item.strip()
            # Extract gene-like patterns
            gene_matches = re.findall(r'\b[A-Z][A-Z0-9-]{1,10}\b', item)
            regulators.extend(gene_matches)
            
        # Remove duplicates and common false positives
        false_positives = {'DNA', 'RNA', 'ATP', 'ROS', 'NO'}
        return list(set(reg for reg in regulators if reg not in false_positives))
        
    async def _get_literature_evidence(self, gene_symbol: str) -> Dict[str, Any]:
        """Get literature evidence about gene regulation"""
        queries = [
            f"regulation of {gene_symbol} gene expression",
            f"{gene_symbol} transcriptional regulation",
            f"{gene_symbol} signaling pathway",
            f"upstream regulators of {gene_symbol}",
            f"{gene_symbol} downstream targets"
        ]
        
        all_passages = []
        all_sources = []
        
        for query in queries[:2]:  # Limit to avoid too many queries
            try:
                if hasattr(self.rag_pipeline, 'query'):
                    result = await self.rag_pipeline.query(query)
                elif hasattr(self.rag_pipeline, 'retrieve_context'):
                    result = self.rag_pipeline.retrieve_context(query)
                else:
                    continue
                    
                if result and result.get("retrieved_chunks"):
                    passages = [chunk.get("text", "") for chunk in result["retrieved_chunks"][:3]]
                    all_passages.extend(passages)
                    all_sources.extend(result.get("citations", []))
                    
            except Exception as e:
                logger.warning(f"Literature query failed for {query}: {e}")
                
        return {
            "passages": all_passages,
            "sources": all_sources
        }
        
    def _identify_knowledge_gaps(self, gene: str, curated_data: Dict[str, Any], 
                               literature_data: Dict[str, Any]) -> List[KnowledgeGap]:
        """Identify specific knowledge gaps"""
        gaps = []
        
        # Upstream regulator gaps
        for regulator in literature_data.get("upstream", []):
            if regulator not in curated_data.get("upstream", []):
                gaps.append(KnowledgeGap(
                    gap_type="upstream_regulator",
                    gene_of_interest=gene,
                    missing_element=regulator,
                    evidence_source="literature",
                    confidence=0.7,  # Default confidence
                    literature_context=f"Literature mentions {regulator} as upstream regulator of {gene}"
                ))
                
        # Downstream target gaps  
        for target in literature_data.get("downstream", []):
            if target not in curated_data.get("downstream", []):
                gaps.append(KnowledgeGap(
                    gap_type="downstream_target",
                    gene_of_interest=gene,
                    missing_element=target,
                    evidence_source="literature",
                    confidence=0.7,
                    literature_context=f"Literature mentions {target} as downstream target of {gene}"
                ))
                
        return gaps
        
    async def forward(self, gene_symbol: str) -> PathwayGapReport:
        """Detect pathway gaps for a specific gene"""
        
        try:
            # 1. Get curated pathway data
            ensembl_id = await self.bio_client.symbol_to_ensembl(gene_symbol)
            pathways = await self.bio_client.get_all_pathways(gene_symbol)
            
            # Extract known regulators from curated data
            curated_upstream = self._extract_curated_regulators(pathways, "upstream")
            curated_downstream = self._extract_curated_regulators(pathways, "downstream")
            
            # 2. Get literature evidence
            lit_evidence = await self._get_literature_evidence(gene_symbol)
            
            if not lit_evidence.get("passages"):
                return PathwayGapReport(
                    gene=gene_symbol,
                    curated_upstream=curated_upstream,
                    curated_downstream=curated_downstream,
                    literature_upstream=[],
                    literature_downstream=[],
                    upstream_gaps=[],
                    downstream_gaps=[],
                    literature_sources=[],
                    pathway_gaps=[]
                )
                
            # 3. Extract regulators from literature
            try:
                lit_extraction = self.extract_regulators(
                    literature_passages="\n".join(lit_evidence["passages"][:3]),
                    gene_of_interest=gene_symbol
                )
                
                lit_upstream = self._parse_literature_regulators(lit_extraction.upstream_regulators)
                lit_downstream = self._parse_literature_regulators(lit_extraction.downstream_targets)
                
            except Exception as e:
                logger.error(f"Literature regulator extraction failed: {e}")
                lit_upstream = []
                lit_downstream = []
                
            # 4. Find gaps
            upstream_gaps = list(set(lit_upstream) - set(curated_upstream))
            downstream_gaps = list(set(lit_downstream) - set(curated_downstream))
            
            # 5. Identify detailed knowledge gaps
            curated_data = {"upstream": curated_upstream, "downstream": curated_downstream}
            literature_data = {"upstream": lit_upstream, "downstream": lit_downstream}
            knowledge_gaps = self._identify_knowledge_gaps(gene_symbol, curated_data, literature_data)
            
            # 6. Format literature sources
            literature_sources = [
                {"text": source[:200] + "...", "relevance": "medium"}
                for source in lit_evidence.get("sources", [])[:5]
            ]
            
            return PathwayGapReport(
                gene=gene_symbol,
                curated_upstream=curated_upstream,
                curated_downstream=curated_downstream,
                literature_upstream=lit_upstream,
                literature_downstream=lit_downstream,
                upstream_gaps=upstream_gaps,
                downstream_gaps=downstream_gaps,
                literature_sources=literature_sources,
                pathway_gaps=knowledge_gaps
            )
            
        except Exception as e:
            logger.error(f"Gap detection failed for {gene_symbol}: {e}")
            return PathwayGapReport(
                gene=gene_symbol,
                curated_upstream=[],
                curated_downstream=[],
                literature_upstream=[],
                literature_downstream=[],
                upstream_gaps=[],
                downstream_gaps=[],
                literature_sources=[],
                pathway_gaps=[]
            )
            
    async def batch_gap_detection(self, gene_symbols: List[str]) -> Dict[str, PathwayGapReport]:
        """Detect gaps for multiple genes"""
        import asyncio
        
        tasks = [self.forward(gene) for gene in gene_symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        gap_reports = {}
        for gene, result in zip(gene_symbols, results):
            if isinstance(result, PathwayGapReport):
                gap_reports[gene] = result
            else:
                logger.error(f"Gap detection failed for {gene}: {result}")
                
        return gap_reports
        
    def summarize_gaps(self, gap_reports: Dict[str, PathwayGapReport]) -> Dict[str, Any]:
        """Summarize gap findings across multiple genes"""
        summary = {
            "total_genes_analyzed": len(gap_reports),
            "genes_with_upstream_gaps": 0,
            "genes_with_downstream_gaps": 0,
            "most_common_missing_upstream": {},
            "most_common_missing_downstream": {},
            "high_confidence_gaps": []
        }
        
        all_upstream_gaps = []
        all_downstream_gaps = []
        
        for gene, report in gap_reports.items():
            if report.upstream_gaps:
                summary["genes_with_upstream_gaps"] += 1
                all_upstream_gaps.extend(report.upstream_gaps)
                
            if report.downstream_gaps:
                summary["genes_with_downstream_gaps"] += 1
                all_downstream_gaps.extend(report.downstream_gaps)
                
            # Collect high confidence gaps
            for gap in report.pathway_gaps:
                if gap.confidence > 0.8:
                    summary["high_confidence_gaps"].append({
                        "gene": gene,
                        "gap_type": gap.gap_type,
                        "missing_element": gap.missing_element,
                        "confidence": gap.confidence
                    })
                    
        # Count most common missing elements
        from collections import Counter
        summary["most_common_missing_upstream"] = dict(Counter(all_upstream_gaps).most_common(10))
        summary["most_common_missing_downstream"] = dict(Counter(all_downstream_gaps).most_common(10))
        
        return summary


class PathwayCompletenessAnalyzer(dspy.Module):
    """Analyze pathway completeness across multiple databases"""
    
    def __init__(self, bio_client: BioAPIClient):
        super().__init__()
        self.bio_client = bio_client
        
    async def analyze_pathway_coverage(self, pathway_name: str) -> Dict[str, Any]:
        """Analyze how well a pathway is covered across databases"""
        # This would require more sophisticated pathway analysis
        # For now, return a placeholder structure
        return {
            "pathway_name": pathway_name,
            "database_coverage": {
                "reactome": "partial",
                "kegg": "complete", 
                "uniprot": "minimal"
            },
            "missing_interactions": [],
            "confidence": 0.5
        }