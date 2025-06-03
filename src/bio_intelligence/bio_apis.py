"""
Biological API Client

Centralized API clients for biological databases including MyGene.info, 
Reactome, and UniProt with robust error handling and caching.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
import mygene
import reactome2py
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class GeneInfo:
    symbol: str
    ensembl_id: Optional[str] = None
    uniprot_id: Optional[str] = None
    description: Optional[str] = None
    aliases: List[str] = None


@dataclass 
class PathwayInfo:
    id: str
    name: str
    description: Optional[str] = None
    genes: List[str] = None
    source: str = "unknown"


class MyGeneClient:
    """Client for MyGene.info API for gene symbol resolution"""
    
    def __init__(self):
        self.mg = mygene.MyGeneInfo()
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def symbol_to_ensembl(self, gene_symbol: str) -> Optional[str]:
        """Convert gene symbol to Ensembl ID"""
        try:
            result = self.mg.query(gene_symbol, species="human", fields="ensembl.gene")
            if result.get("hits"):
                ensembl_data = result["hits"][0].get("ensembl")
                if ensembl_data:
                    return ensembl_data.get("gene")
        except Exception as e:
            logger.error(f"Error querying MyGene for {gene_symbol}: {e}")
        return None
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_gene_info(self, gene_symbol: str) -> Optional[GeneInfo]:
        """Get comprehensive gene information"""
        try:
            result = self.mg.query(
                gene_symbol, 
                species="human", 
                fields="ensembl.gene,uniprot.Swiss-Prot,summary,alias"
            )
            if result.get("hits"):
                hit = result["hits"][0]
                ensembl_id = None
                if hit.get("ensembl"):
                    ensembl_id = hit["ensembl"].get("gene")
                    
                uniprot_id = None
                if hit.get("uniprot", {}).get("Swiss-Prot"):
                    uniprot_id = hit["uniprot"]["Swiss-Prot"]
                    
                aliases = hit.get("alias", [])
                if isinstance(aliases, str):
                    aliases = [aliases]
                    
                return GeneInfo(
                    symbol=gene_symbol,
                    ensembl_id=ensembl_id,
                    uniprot_id=uniprot_id,
                    description=hit.get("summary"),
                    aliases=aliases
                )
        except Exception as e:
            logger.error(f"Error getting gene info for {gene_symbol}: {e}")
        return None


class ReactomeClient:
    """Client for Reactome pathway database"""
    
    def __init__(self):
        self.base_url = "https://reactome.org/ContentService"
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_pathways_for_gene(self, ensembl_id: str) -> List[PathwayInfo]:
        """Get pathways for a given Ensembl gene ID"""
        try:
            # Use direct REST API calls instead of reactome2py.analysis
            url = f"https://reactome.org/ContentService/data/pathways/low/entity/{ensembl_id}"
            response = requests.get(url, timeout=10)
            
            pathway_list = []
            if response.status_code == 200:
                pathways = response.json()
                for pathway in pathways:
                    pathway_list.append(PathwayInfo(
                        id=pathway.get("stId", ""),
                        name=pathway.get("displayName", ""),
                        description=pathway.get("displayName", ""),
                        source="reactome"
                    ))
            return pathway_list
        except Exception as e:
            logger.error(f"Error getting Reactome pathways for {ensembl_id}: {e}")
            return []


class KEGGClient:
    """Client for KEGG pathway database"""
    
    def __init__(self):
        self.base_url = "https://rest.kegg.jp"
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_pathways_for_gene(self, gene_symbol: str) -> List[PathwayInfo]:
        """Get KEGG pathways for a gene symbol"""
        try:
            # First, find the KEGG gene ID
            async with aiohttp.ClientSession() as session:
                # Search for human gene
                search_url = f"{self.base_url}/find/hsa/{gene_symbol}"
                async with session.get(search_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        lines = content.strip().split('\n')
                        
                        pathway_list = []
                        for line in lines:
                            if line.startswith('hsa:'):
                                gene_id = line.split('\t')[0]
                                # Get pathways for this gene
                                pathway_url = f"{self.base_url}/link/pathway/{gene_id}"
                                async with session.get(pathway_url) as pathway_response:
                                    if pathway_response.status == 200:
                                        pathway_content = await pathway_response.text()
                                        for pathway_line in pathway_content.strip().split('\n'):
                                            if pathway_line:
                                                parts = pathway_line.split('\t')
                                                if len(parts) >= 2:
                                                    pathway_id = parts[0].replace('path:', '')
                                                    # Get pathway name
                                                    name_url = f"{self.base_url}/get/{pathway_id}"
                                                    async with session.get(name_url) as name_response:
                                                        if name_response.status == 200:
                                                            name_content = await name_response.text()
                                                            name_lines = name_content.split('\n')
                                                            pathway_name = pathway_id  # fallback
                                                            for name_line in name_lines:
                                                                if name_line.startswith('NAME'):
                                                                    pathway_name = name_line.split('NAME')[1].strip()
                                                                    break
                                                            
                                                            pathway_list.append(PathwayInfo(
                                                                id=pathway_id,
                                                                name=pathway_name,
                                                                description=pathway_name,
                                                                source="kegg"
                                                            ))
                        return pathway_list
        except Exception as e:
            logger.error(f"Error getting KEGG pathways for {gene_symbol}: {e}")
        return []


class UniProtClient:
    """Client for UniProt protein database"""
    
    def __init__(self):
        self.base_url = "https://rest.uniprot.org"
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_protein_info(self, uniprot_id: str) -> Dict[str, Any]:
        """Get protein information from UniProt"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/uniprotkb/{uniprot_id}.json"
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
        except Exception as e:
            logger.error(f"Error getting UniProt info for {uniprot_id}: {e}")
        return {}


class BioAPIClient:
    """Unified client for all biological databases"""
    
    def __init__(self):
        self.mygene = MyGeneClient()
        self.reactome = ReactomeClient()
        self.kegg = KEGGClient()
        self.uniprot = UniProtClient()
        self._gene_cache = {}
        
    async def symbol_to_ensembl(self, gene_symbol: str) -> Optional[str]:
        """Convert gene symbol to Ensembl ID with caching"""
        if gene_symbol in self._gene_cache:
            return self._gene_cache[gene_symbol].get("ensembl_id")
            
        ensembl_id = self.mygene.symbol_to_ensembl(gene_symbol)
        if ensembl_id:
            self._gene_cache[gene_symbol] = {"ensembl_id": ensembl_id}
        return ensembl_id
        
    async def get_gene_info(self, gene_symbol: str) -> Optional[GeneInfo]:
        """Get comprehensive gene information"""
        return self.mygene.get_gene_info(gene_symbol)
        
    async def get_reactome_pathways(self, ensembl_id: str) -> List[PathwayInfo]:
        """Get Reactome pathways for an Ensembl ID"""
        return self.reactome.get_pathways_for_gene(ensembl_id)
        
    async def get_kegg_pathways(self, gene_symbol: str) -> List[PathwayInfo]:
        """Get KEGG pathways for a gene symbol"""
        return await self.kegg.get_pathways_for_gene(gene_symbol)
        
    async def get_protein_info(self, uniprot_id: str) -> Dict[str, Any]:
        """Get protein information from UniProt"""
        return await self.uniprot.get_protein_info(uniprot_id)
        
    async def get_all_pathways(self, gene_symbol: str) -> Dict[str, List[PathwayInfo]]:
        """Get pathways from all databases for a gene"""
        ensembl_id = await self.symbol_to_ensembl(gene_symbol)
        
        # Run pathway queries in parallel
        reactome_task = None
        if ensembl_id:
            reactome_task = asyncio.create_task(self.get_reactome_pathways(ensembl_id))
        kegg_task = asyncio.create_task(self.get_kegg_pathways(gene_symbol))
        
        pathways = {"reactome": [], "kegg": []}
        
        if reactome_task:
            pathways["reactome"] = await reactome_task
        pathways["kegg"] = await kegg_task
        
        return pathways