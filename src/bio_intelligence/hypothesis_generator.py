"""
Hypothesis Generator Module

Generate literature-grounded biological hypotheses using DSPy, RAG retrieval,
and biological pathway knowledge.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import dspy

try:
    from .bio_apis import BioAPIClient
    from .gene_resolver import GeneResolver, ResolvedGene
except ImportError:
    from bio_apis import BioAPIClient
    from gene_resolver import GeneResolver, ResolvedGene

logger = logging.getLogger(__name__)


@dataclass
class BiologicalHypothesis:
    topic: str
    hypothesis: str
    confidence: str
    literature_sources: List[Dict[str, Any]]
    pathway_evidence: List[Dict[str, Any]]
    testable_predictions: List[str]
    mechanistic_details: Optional[str] = None


class SimpleHypothesisGenerator(dspy.Signature):
    """Generate a basic biological hypothesis from a topic"""
    topic = dspy.InputField(desc="Biological topic or research question")
    hypothesis = dspy.OutputField(desc="A testable biological hypothesis")


class GroundedHypothesisGenerator(dspy.Signature):
    """Generate detailed hypothesis grounded in literature and pathway data"""
    literature_context = dspy.InputField(desc="Relevant literature passages and citations")
    pathway_context = dspy.InputField(desc="Biological pathway information and gene interactions")
    topic = dspy.InputField(desc="Biological topic or research question")
    detailed_hypothesis = dspy.OutputField(desc="Detailed, mechanistic hypothesis with specific predictions")
    testable_predictions = dspy.OutputField(desc="List of specific, testable experimental predictions")


class MechanismGenerator(dspy.Signature):
    """Generate mechanistic details for a biological hypothesis"""
    hypothesis = dspy.InputField(desc="Biological hypothesis")
    pathway_data = dspy.InputField(desc="Relevant pathway and gene interaction data")
    mechanistic_details = dspy.OutputField(desc="Detailed molecular mechanism underlying the hypothesis")


class HypothesisGenerator(dspy.Module):
    """Generate hypotheses grounded in literature and biological knowledge"""
    
    def __init__(self, rag_pipeline, bio_client: BioAPIClient):
        super().__init__()
        self.rag_pipeline = rag_pipeline
        self.bio_client = bio_client
        self.gene_resolver = GeneResolver(bio_client)
        
        # DSPy modules for different types of hypothesis generation
        self.simple_hypothesis = dspy.ChainOfThought(SimpleHypothesisGenerator)
        self.grounded_hypothesis = dspy.ChainOfThought(GroundedHypothesisGenerator)
        self.mechanism_generator = dspy.ChainOfThought(MechanismGenerator)
        
    def _format_literature_context(self, rag_result: Dict[str, Any]) -> str:
        """Format RAG retrieval results for prompt context"""
        if not rag_result or not rag_result.get("retrieved_chunks"):
            return "No relevant literature found."
            
        context_parts = []
        for i, chunk in enumerate(rag_result["retrieved_chunks"][:5], 1):
            context_parts.append(
                f"Source {i}: {chunk.get('text', '')[:500]}..."
            )
            
        return "\n\n".join(context_parts)
        
    def _format_pathway_context(self, resolved_genes: List[ResolvedGene]) -> str:
        """Format pathway information for prompt context"""
        if not resolved_genes:
            return "No gene pathway information available."
            
        context_parts = []
        for gene in resolved_genes[:3]:  # Limit to top 3 genes
            gene_info = f"Gene: {gene.symbol}"
            if gene.ensembl_id:
                gene_info += f" (Ensembl: {gene.ensembl_id})"
                
            pathways = []
            if gene.pathways:
                for source, pathway_list in gene.pathways.items():
                    for pathway in pathway_list[:3]:  # Limit pathways per source
                        pathways.append(f"{source.upper()}: {pathway.name}")
                        
            if pathways:
                gene_info += f"\nPathways: {'; '.join(pathways)}"
                
            context_parts.append(gene_info)
            
        return "\n\n".join(context_parts)
        
    async def forward(self, topic: str, use_context: bool = True, generate_mechanism: bool = False) -> BiologicalHypothesis:
        """Generate a biological hypothesis with optional literature and pathway grounding"""
        
        if not use_context:
            # Simple hypothesis generation without context
            try:
                result = self.simple_hypothesis(topic=topic)
                return BiologicalHypothesis(
                    topic=topic,
                    hypothesis=result.hypothesis,
                    confidence="low",
                    literature_sources=[],
                    pathway_evidence=[],
                    testable_predictions=[]
                )
            except Exception as e:
                logger.error(f"Simple hypothesis generation failed: {e}")
                return BiologicalHypothesis(
                    topic=topic,
                    hypothesis="Unable to generate hypothesis",
                    confidence="none",
                    literature_sources=[],
                    pathway_evidence=[],
                    testable_predictions=[]
                )
                
        # Grounded hypothesis with literature and pathway context
        try:
            # 1. Get literature context from RAG
            lit_result = await self._get_literature_context(topic)
            lit_context = self._format_literature_context(lit_result)
            
            # 2. Extract genes and get pathway context
            gene_result = await self.gene_resolver.forward(topic)
            resolved_genes = gene_result.get("resolved_genes", [])
            pathway_context = self._format_pathway_context(resolved_genes)
            
            # 3. Generate grounded hypothesis
            grounded_result = self.grounded_hypothesis(
                literature_context=lit_context,
                pathway_context=pathway_context,
                topic=topic
            )
            
            # 4. Parse testable predictions
            predictions = self._parse_predictions(grounded_result.testable_predictions)
            
            # 5. Generate mechanistic details if requested
            mechanistic_details = None
            if generate_mechanism and resolved_genes:
                try:
                    mechanism_result = self.mechanism_generator(
                        hypothesis=grounded_result.detailed_hypothesis,
                        pathway_data=pathway_context
                    )
                    mechanistic_details = mechanism_result.mechanistic_details
                except Exception as e:
                    logger.warning(f"Mechanism generation failed: {e}")
                    
            # 6. Extract literature sources
            literature_sources = self._extract_literature_sources(lit_result)
            
            # 7. Format pathway evidence
            pathway_evidence = self._format_pathway_evidence(resolved_genes)
            
            return BiologicalHypothesis(
                topic=topic,
                hypothesis=grounded_result.detailed_hypothesis,
                confidence="high",
                literature_sources=literature_sources,
                pathway_evidence=pathway_evidence,
                testable_predictions=predictions,
                mechanistic_details=mechanistic_details
            )
            
        except Exception as e:
            logger.error(f"Grounded hypothesis generation failed: {e}")
            # Fallback to simple hypothesis
            return await self.forward(topic, use_context=False)
            
    async def _get_literature_context(self, topic: str) -> Dict[str, Any]:
        """Get relevant literature context using RAG pipeline"""
        try:
            # Use the existing RAG pipeline to retrieve relevant context
            if hasattr(self.rag_pipeline, 'query'):
                return await self.rag_pipeline.query(topic)
            elif hasattr(self.rag_pipeline, 'retrieve_context'):
                return self.rag_pipeline.retrieve_context(topic)
            else:
                # Fallback for different RAG interface
                return {"retrieved_chunks": [], "citations": []}
        except Exception as e:
            logger.error(f"Literature context retrieval failed: {e}")
            return {"retrieved_chunks": [], "citations": []}
            
    def _parse_predictions(self, predictions_text: str) -> List[str]:
        """Parse testable predictions from LLM output"""
        if not predictions_text:
            return []
            
        # Split by common delimiters and clean up
        predictions = []
        for line in predictions_text.split('\n'):
            line = line.strip()
            # Remove numbering and bullet points
            line = line.lstrip('0123456789.-•* ')
            if line and len(line) > 10:  # Filter out very short lines
                predictions.append(line)
                
        return predictions[:5]  # Limit to 5 predictions
        
    def _extract_literature_sources(self, lit_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and format literature sources"""
        sources = []
        
        if lit_result and lit_result.get("retrieved_chunks"):
            for chunk in lit_result["retrieved_chunks"][:5]:
                source = {
                    "text": chunk.get("text", "")[:200] + "...",
                    "source": chunk.get("source", "Unknown"),
                    "page": chunk.get("page", 1),
                    "relevance": "high" if chunk.get("score", 0) > 0.8 else "medium"
                }
                sources.append(source)
                
        return sources
        
    def _format_pathway_evidence(self, resolved_genes: List[ResolvedGene]) -> List[Dict[str, Any]]:
        """Format pathway evidence for the hypothesis"""
        evidence = []
        
        for gene in resolved_genes[:3]:  # Top 3 genes
            gene_evidence = {
                "gene": gene.symbol,
                "ensembl_id": gene.ensembl_id,
                "confidence": gene.confidence,
                "pathways": {}
            }
            
            if gene.pathways:
                for source, pathway_list in gene.pathways.items():
                    gene_evidence["pathways"][source] = [
                        {"id": p.id, "name": p.name, "description": p.description}
                        for p in pathway_list[:3]
                    ]
                    
            evidence.append(gene_evidence)
            
        return evidence


class HypothesisComparator(dspy.Module):
    """Compare and rank multiple hypotheses"""
    
    def __init__(self):
        super().__init__()
        
    def compare_hypotheses(self, hypotheses: List[BiologicalHypothesis]) -> List[BiologicalHypothesis]:
        """Rank hypotheses by quality metrics"""
        def score_hypothesis(h: BiologicalHypothesis) -> float:
            score = 0.0
            
            # Base score from confidence
            if h.confidence == "high":
                score += 0.4
            elif h.confidence == "medium":
                score += 0.2
                
            # Literature support
            score += min(len(h.literature_sources) * 0.1, 0.3)
            
            # Pathway evidence
            score += min(len(h.pathway_evidence) * 0.1, 0.2)
            
            # Testable predictions
            score += min(len(h.testable_predictions) * 0.05, 0.1)
            
            return score
            
        return sorted(hypotheses, key=score_hypothesis, reverse=True)