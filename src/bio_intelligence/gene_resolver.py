"""
Gene Resolver Module

Extract gene symbols from biological queries and resolve them to database IDs 
with pathway information using DSPy and biological APIs.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import dspy

try:
    from .bio_apis import BioAPIClient, PathwayInfo
except ImportError:
    from bio_apis import BioAPIClient, PathwayInfo

logger = logging.getLogger(__name__)


@dataclass
class ResolvedGene:
    symbol: str
    ensembl_id: Optional[str] = None
    confidence: float = 0.0
    pathways: Dict[str, List[PathwayInfo]] = None
    aliases: List[str] = None


class GeneExtractor(dspy.Signature):
    """Extract gene symbols from biological text queries"""
    biological_query = dspy.InputField(desc="A biological query or text passage")
    gene_symbols = dspy.OutputField(desc="Comma-separated list of gene symbols found in the query")


class GeneResolver(dspy.Module):
    """Extract gene symbols and resolve to biological database IDs with pathway info"""
    
    def __init__(self, bio_client: BioAPIClient, max_genes: int = 10):
        super().__init__()
        self.extract_genes = dspy.ChainOfThought(GeneExtractor)
        self.bio_client = bio_client
        self.max_genes = max_genes
        
        # Common gene name patterns for validation
        self.gene_patterns = [
            r'\b[A-Z][A-Z0-9]{1,8}\b',  # Standard gene symbols (e.g., TP53, SOD2)
            r'\b[A-Z]{2,}-\d+\b',       # Gene families (e.g., COX-1, NRF-2)
            r'\b[A-Z][a-z]{2,}\d*\b'    # Mixed case genes (e.g., Nrf2, p53)
        ]
        
    def _extract_gene_symbols_regex(self, text: str) -> List[str]:
        """Extract potential gene symbols using regex patterns"""
        candidates = set()
        
        for pattern in self.gene_patterns:
            matches = re.findall(pattern, text)
            candidates.update(matches)
            
        # Filter out common false positives
        false_positives = {
            'DNA', 'RNA', 'ATP', 'ADP', 'NAD', 'NADH', 'FAD', 'FADH',
            'CoA', 'GTP', 'GDP', 'CTP', 'UTP', 'cAMP', 'cGMP', 'Pi',
            'PPi', 'ROS', 'NO', 'CO', 'H2O', 'O2', 'CO2', 'NH3',
            'pH', 'PCR', 'SDS', 'PAGE', 'BLAST', 'FASTA'
        }
        
        filtered_candidates = [c for c in candidates if c not in false_positives]
        return filtered_candidates[:self.max_genes]
        
    def _parse_llm_gene_symbols(self, gene_symbols_text: str) -> List[str]:
        """Parse gene symbols from LLM output"""
        if not gene_symbols_text:
            return []
            
        # Split by commas and clean up
        symbols = [s.strip() for s in gene_symbols_text.split(',')]
        symbols = [s for s in symbols if s and len(s) > 1]
        
        # Remove any explanatory text in parentheses
        cleaned_symbols = []
        for symbol in symbols:
            cleaned = re.sub(r'\s*\([^)]*\)', '', symbol).strip()
            if cleaned:
                cleaned_symbols.append(cleaned)
                
        return cleaned_symbols[:self.max_genes]
        
    async def _resolve_gene_to_databases(self, gene_symbol: str) -> Optional[ResolvedGene]:
        """Resolve a single gene symbol to database information"""
        try:
            # Get basic gene info
            gene_info = await self.bio_client.get_gene_info(gene_symbol)
            if not gene_info:
                return None
                
            # Get pathways from all databases
            pathways = await self.bio_client.get_all_pathways(gene_symbol)
            
            # Calculate confidence based on available data
            confidence = 0.5  # Base confidence for finding the gene
            if gene_info.ensembl_id:
                confidence += 0.3
            if pathways.get("reactome"):
                confidence += 0.1
            if pathways.get("kegg"):
                confidence += 0.1
                
            return ResolvedGene(
                symbol=gene_symbol,
                ensembl_id=gene_info.ensembl_id,
                confidence=min(confidence, 1.0),
                pathways=pathways,
                aliases=gene_info.aliases or []
            )
        except Exception as e:
            logger.error(f"Error resolving gene {gene_symbol}: {e}")
            return None
            
    async def forward(self, query: str) -> Dict[str, Any]:
        """Extract and resolve genes from a biological query"""
        
        # 1. Extract gene symbols using LLM
        try:
            extraction = self.extract_genes(biological_query=query)
            llm_genes = self._parse_llm_gene_symbols(extraction.gene_symbols)
        except Exception as e:
            logger.warning(f"LLM gene extraction failed: {e}")
            llm_genes = []
            
        # 2. Extract gene symbols using regex as backup
        regex_genes = self._extract_gene_symbols_regex(query)
        
        # 3. Combine and deduplicate gene symbols
        all_genes = list(set(llm_genes + regex_genes))
        
        if not all_genes:
            return {
                "original_query": query,
                "gene_symbols": [],
                "resolved_genes": [],
                "extraction_methods": {"llm": llm_genes, "regex": regex_genes}
            }
            
        # 4. Resolve genes to database information in parallel
        resolution_tasks = [
            self._resolve_gene_to_databases(gene) 
            for gene in all_genes
        ]
        
        resolved_results = await asyncio.gather(*resolution_tasks, return_exceptions=True)
        
        # Filter out failed resolutions and exceptions
        resolved_genes = []
        for result in resolved_results:
            if isinstance(result, ResolvedGene):
                resolved_genes.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Gene resolution failed: {result}")
                
        # Sort by confidence score
        resolved_genes.sort(key=lambda x: x.confidence, reverse=True)
        
        return {
            "original_query": query,
            "gene_symbols": [g.symbol for g in resolved_genes],
            "resolved_genes": resolved_genes,
            "extraction_methods": {
                "llm": llm_genes, 
                "regex": regex_genes
            },
            "total_genes_found": len(resolved_genes)
        }
        
    def get_top_gene(self, resolution_result: Dict[str, Any]) -> Optional[ResolvedGene]:
        """Get the highest confidence resolved gene"""
        resolved_genes = resolution_result.get("resolved_genes", [])
        return resolved_genes[0] if resolved_genes else None
        
    def get_genes_with_pathways(self, resolution_result: Dict[str, Any]) -> List[ResolvedGene]:
        """Get only genes that have pathway information"""
        resolved_genes = resolution_result.get("resolved_genes", [])
        return [
            gene for gene in resolved_genes 
            if gene.pathways and (gene.pathways.get("reactome") or gene.pathways.get("kegg"))
        ]


class BatchGeneResolver(GeneResolver):
    """Batch version for processing multiple queries efficiently"""
    
    async def resolve_multiple_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Resolve genes from multiple queries in parallel"""
        tasks = [self.forward(query) for query in queries]
        return await asyncio.gather(*tasks, return_exceptions=True)
        
    async def resolve_gene_list(self, gene_symbols: List[str]) -> Dict[str, ResolvedGene]:
        """Resolve a list of known gene symbols"""
        tasks = [
            self._resolve_gene_to_databases(symbol) 
            for symbol in gene_symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        resolved_dict = {}
        for symbol, result in zip(gene_symbols, results):
            if isinstance(result, ResolvedGene):
                resolved_dict[symbol] = result
                
        return resolved_dict