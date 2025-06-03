"""
Composed Investigation Modules

Chain multiple bio-intelligence modules for comprehensive gene and pathway
investigations with integrated reporting.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import dspy

try:
    from .bio_apis import BioAPIClient
    from .gap_detector import PathwayGapDetector, PathwayGapReport
    from .gene_resolver import GeneResolver
    from .hypothesis_generator import BiologicalHypothesis, HypothesisGenerator
except ImportError:
    from bio_apis import BioAPIClient
    from gap_detector import PathwayGapDetector, PathwayGapReport
    from gene_resolver import GeneResolver
    from hypothesis_generator import BiologicalHypothesis, HypothesisGenerator

logger = logging.getLogger(__name__)


@dataclass
class InvestigationReport:
    """Complete investigation report combining all analysis modules"""
    query: str
    timestamp: datetime
    gene_resolution: Dict[str, Any]
    hypothesis: Optional[BiologicalHypothesis]
    pathway_gaps: Optional[PathwayGapReport]
    investigation_summary: str
    primary_gene: Optional[str] = None
    confidence_score: float = 0.0
    recommendations: List[str] = None


class InvestigationSummarizer(dspy.Signature):
    """Generate investigation summary from all analysis components"""
    gene_data = dspy.InputField(desc="Gene resolution and pathway information")
    hypothesis = dspy.InputField(desc="Generated biological hypothesis")
    gap_analysis = dspy.InputField(desc="Pathway gap detection results")
    query = dspy.InputField(desc="Original research query")
    investigation_summary = dspy.OutputField(desc="Comprehensive summary of the investigation findings")
    key_findings = dspy.OutputField(desc="Most important findings from the investigation")
    research_recommendations = dspy.OutputField(desc="Recommended next steps for research")


class ComprehensiveGeneInvestigation(dspy.Module):
    """Complete investigation pipeline combining all bio-intelligence modules"""
    
    def __init__(self, rag_pipeline, bio_client: BioAPIClient):
        super().__init__()
        self.rag_pipeline = rag_pipeline
        self.bio_client = bio_client
        
        # Initialize component modules
        self.gene_resolver = GeneResolver(bio_client)
        self.hypothesis_generator = HypothesisGenerator(rag_pipeline, bio_client)
        self.gap_detector = PathwayGapDetector(rag_pipeline, bio_client)
        
        # DSPy module for summarization
        self.investigation_summarizer = dspy.ChainOfThought(InvestigationSummarizer)
        
    def _calculate_confidence_score(self, gene_data: Dict[str, Any], 
                                   hypothesis: Optional[BiologicalHypothesis],
                                   gaps: Optional[PathwayGapReport]) -> float:
        """Calculate overall confidence score for the investigation"""
        score = 0.0
        
        # Gene resolution confidence
        resolved_genes = gene_data.get("resolved_genes", [])
        if resolved_genes:
            avg_gene_confidence = sum(g.confidence for g in resolved_genes) / len(resolved_genes)
            score += avg_gene_confidence * 0.3
            
        # Hypothesis confidence
        if hypothesis:
            if hypothesis.confidence == "high":
                score += 0.4
            elif hypothesis.confidence == "medium":
                score += 0.2
            elif hypothesis.confidence == "low":
                score += 0.1
                
        # Literature support
        if hypothesis and hypothesis.literature_sources:
            score += min(len(hypothesis.literature_sources) * 0.05, 0.2)
            
        # Gap analysis completeness
        if gaps and (gaps.upstream_gaps or gaps.downstream_gaps):
            score += 0.1  # Finding gaps is valuable
            
        return min(score, 1.0)
        
    def _format_investigation_data(self, gene_data: Dict[str, Any], 
                                 hypothesis: Optional[BiologicalHypothesis],
                                 gaps: Optional[PathwayGapReport]) -> Dict[str, str]:
        """Format data for summarization prompt"""
        
        # Format gene data
        gene_text = f"Identified {len(gene_data.get('gene_symbols', []))} genes: {', '.join(gene_data.get('gene_symbols', []))}"
        if gene_data.get("resolved_genes"):
            top_gene = gene_data["resolved_genes"][0]
            gene_text += f"\nPrimary gene: {top_gene.symbol} (confidence: {top_gene.confidence:.2f})"
            if top_gene.pathways:
                pathway_count = sum(len(pathways) for pathways in top_gene.pathways.values())
                gene_text += f"\nPathways found: {pathway_count}"
                
        # Format hypothesis
        hypothesis_text = "No hypothesis generated"
        if hypothesis:
            hypothesis_text = f"Hypothesis: {hypothesis.hypothesis}\nConfidence: {hypothesis.confidence}"
            if hypothesis.testable_predictions:
                hypothesis_text += f"\nPredictions: {len(hypothesis.testable_predictions)} testable predictions"
                
        # Format gap analysis
        gap_text = "No gap analysis performed"
        if gaps:
            gap_text = f"Gap analysis for {gaps.gene}:\n"
            gap_text += f"Upstream gaps: {len(gaps.upstream_gaps)} potential missing regulators\n"
            gap_text += f"Downstream gaps: {len(gaps.downstream_gaps)} potential missing targets"
            
        return {
            "gene_data": gene_text,
            "hypothesis": hypothesis_text,
            "gap_analysis": gap_text
        }
        
    async def forward(self, query: str, include_gaps: bool = True, 
                     generate_mechanism: bool = False) -> InvestigationReport:
        """Perform comprehensive gene investigation"""
        
        timestamp = datetime.now()
        
        try:
            # 1. Resolve genes and pathways
            logger.info(f"Starting gene resolution for query: {query}")
            gene_data = await self.gene_resolver.forward(query)
            
            if not gene_data.get("gene_symbols"):
                return InvestigationReport(
                    query=query,
                    timestamp=timestamp,
                    gene_resolution=gene_data,
                    hypothesis=None,
                    pathway_gaps=None,
                    investigation_summary="No genes identified in query. Unable to perform biological investigation.",
                    confidence_score=0.0,
                    recommendations=["Refine query to include specific gene names or biological processes"]
                )
                
            primary_gene = gene_data["gene_symbols"][0]
            logger.info(f"Primary gene identified: {primary_gene}")
            
            # 2. Generate hypothesis (run in parallel with gap detection if needed)
            hypothesis_task = self.hypothesis_generator.forward(
                query, 
                use_context=True, 
                generate_mechanism=generate_mechanism
            )
            
            # 3. Detect pathway gaps for primary gene
            gaps_task = None
            if include_gaps:
                gaps_task = self.gap_detector.forward(primary_gene)
                
            # Wait for both tasks to complete
            results = await asyncio.gather(hypothesis_task, gaps_task, return_exceptions=True)
            
            hypothesis = results[0] if not isinstance(results[0], Exception) else None
            gaps = results[1] if gaps_task and not isinstance(results[1], Exception) else None
            
            if isinstance(results[0], Exception):
                logger.error(f"Hypothesis generation failed: {results[0]}")
            if gaps_task and isinstance(results[1], Exception):
                logger.error(f"Gap detection failed: {results[1]}")
                
            # 4. Calculate confidence score
            confidence_score = self._calculate_confidence_score(gene_data, hypothesis, gaps)
            
            # 5. Generate investigation summary
            try:
                formatted_data = self._format_investigation_data(gene_data, hypothesis, gaps)
                
                summary_result = self.investigation_summarizer(
                    gene_data=formatted_data["gene_data"],
                    hypothesis=formatted_data["hypothesis"], 
                    gap_analysis=formatted_data["gap_analysis"],
                    query=query
                )
                
                investigation_summary = summary_result.investigation_summary
                recommendations = self._parse_recommendations(summary_result.research_recommendations)
                
            except Exception as e:
                logger.error(f"Investigation summarization failed: {e}")
                investigation_summary = self._generate_fallback_summary(gene_data, hypothesis, gaps)
                recommendations = ["Review individual analysis components for detailed insights"]
                
            return InvestigationReport(
                query=query,
                timestamp=timestamp,
                gene_resolution=gene_data,
                hypothesis=hypothesis,
                pathway_gaps=gaps,
                investigation_summary=investigation_summary,
                primary_gene=primary_gene,
                confidence_score=confidence_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Investigation failed for query '{query}': {e}")
            return InvestigationReport(
                query=query,
                timestamp=timestamp,
                gene_resolution={"gene_symbols": [], "resolved_genes": []},
                hypothesis=None,
                pathway_gaps=None,
                investigation_summary=f"Investigation failed due to error: {str(e)}",
                confidence_score=0.0,
                recommendations=["Check system logs and retry with a simpler query"]
            )
            
    def _generate_fallback_summary(self, gene_data: Dict[str, Any], 
                                 hypothesis: Optional[BiologicalHypothesis],
                                 gaps: Optional[PathwayGapReport]) -> str:
        """Generate basic summary when LLM summarization fails"""
        parts = []
        
        if gene_data.get("gene_symbols"):
            parts.append(f"Identified {len(gene_data['gene_symbols'])} genes: {', '.join(gene_data['gene_symbols'])}")
            
        if hypothesis:
            parts.append(f"Generated {hypothesis.confidence}-confidence hypothesis about the biological mechanism")
            
        if gaps:
            gap_count = len(gaps.upstream_gaps) + len(gaps.downstream_gaps)
            if gap_count > 0:
                parts.append(f"Detected {gap_count} potential knowledge gaps in pathway databases")
                
        return ". ".join(parts) if parts else "Limited investigation results available."
        
    def _parse_recommendations(self, recommendations_text: str) -> List[str]:
        """Parse research recommendations from LLM output"""
        if not recommendations_text:
            return []
            
        recommendations = []
        for line in recommendations_text.split('\n'):
            line = line.strip()
            line = line.lstrip('0123456789.-•* ')  # Remove numbering
            if line and len(line) > 10:
                recommendations.append(line)
                
        return recommendations[:5]  # Limit to 5 recommendations
        
    def export_report(self, report: InvestigationReport, format: str = "json") -> Dict[str, Any]:
        """Export investigation report in specified format"""
        if format == "json":
            return {
                "query": report.query,
                "timestamp": report.timestamp.isoformat(),
                "primary_gene": report.primary_gene,
                "confidence_score": report.confidence_score,
                "gene_resolution": {
                    "total_genes": len(report.gene_resolution.get("gene_symbols", [])),
                    "genes": report.gene_resolution.get("gene_symbols", []),
                    "extraction_methods": report.gene_resolution.get("extraction_methods", {})
                },
                "hypothesis": {
                    "hypothesis": report.hypothesis.hypothesis if report.hypothesis else None,
                    "confidence": report.hypothesis.confidence if report.hypothesis else None,
                    "testable_predictions": report.hypothesis.testable_predictions if report.hypothesis else [],
                    "mechanistic_details": report.hypothesis.mechanistic_details if report.hypothesis else None
                } if report.hypothesis else None,
                "pathway_gaps": {
                    "upstream_gaps": report.pathway_gaps.upstream_gaps if report.pathway_gaps else [],
                    "downstream_gaps": report.pathway_gaps.downstream_gaps if report.pathway_gaps else [],
                    "total_gaps": len(report.pathway_gaps.upstream_gaps) + len(report.pathway_gaps.downstream_gaps) if report.pathway_gaps else 0
                } if report.pathway_gaps else None,
                "investigation_summary": report.investigation_summary,
                "recommendations": report.recommendations or []
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")


class BatchInvestigator(dspy.Module):
    """Perform investigations on multiple queries in batch"""
    
    def __init__(self, rag_pipeline, bio_client: BioAPIClient, max_concurrent: int = 3):
        super().__init__()
        self.investigator = ComprehensiveGeneInvestigation(rag_pipeline, bio_client)
        self.max_concurrent = max_concurrent
        
    async def investigate_multiple(self, queries: List[str], 
                                 include_gaps: bool = True) -> List[InvestigationReport]:
        """Investigate multiple queries with controlled concurrency"""
        
        # Process in batches to avoid overwhelming APIs
        results = []
        for i in range(0, len(queries), self.max_concurrent):
            batch = queries[i:i + self.max_concurrent]
            
            tasks = [
                self.investigator.forward(query, include_gaps=include_gaps)
                for query in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, InvestigationReport):
                    results.append(result)
                else:
                    logger.error(f"Batch investigation failed: {result}")
                    
        return results
        
    def compare_investigations(self, reports: List[InvestigationReport]) -> Dict[str, Any]:
        """Compare multiple investigation reports"""
        comparison = {
            "total_investigations": len(reports),
            "high_confidence_count": sum(1 for r in reports if r.confidence_score > 0.7),
            "genes_investigated": list(set(r.primary_gene for r in reports if r.primary_gene)),
            "common_pathways": self._find_common_pathways(reports),
            "investigation_quality": {
                "avg_confidence": sum(r.confidence_score for r in reports) / len(reports) if reports else 0,
                "hypotheses_generated": sum(1 for r in reports if r.hypothesis),
                "gaps_detected": sum(1 for r in reports if r.pathway_gaps and (r.pathway_gaps.upstream_gaps or r.pathway_gaps.downstream_gaps))
            }
        }
        
        return comparison
        
    def _find_common_pathways(self, reports: List[InvestigationReport]) -> List[str]:
        """Find pathways mentioned across multiple investigations"""
        all_pathways = []
        
        for report in reports:
            if report.gene_resolution.get("resolved_genes"):
                for gene in report.gene_resolution["resolved_genes"]:
                    if gene.pathways:
                        for pathway_list in gene.pathways.values():
                            all_pathways.extend([p.name for p in pathway_list])
                            
        # Count pathway occurrences
        from collections import Counter
        pathway_counts = Counter(all_pathways)
        
        # Return pathways mentioned in multiple investigations
        return [pathway for pathway, count in pathway_counts.items() if count > 1]