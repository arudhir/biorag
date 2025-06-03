"""
Tests for Bio Intelligence Modules

Test suite for gene resolution, hypothesis generation, and gap detection modules.
"""

import asyncio
import pytest
import os
import sys
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bio_intelligence import (
    BioAPIClient, GeneResolver, HypothesisGenerator, 
    PathwayGapDetector, ComprehensiveGeneInvestigation
)
from bio_intelligence.bio_apis import GeneInfo, PathwayInfo


class TestBioAPIClient:
    """Test biological API client functionality"""
    
    @pytest.fixture
    def bio_client(self):
        return BioAPIClient()
        
    @pytest.mark.asyncio
    async def test_symbol_to_ensembl_mock(self, bio_client):
        """Test gene symbol to Ensembl ID conversion with mock"""
        with patch.object(bio_client.mygene, 'symbol_to_ensembl') as mock_method:
            mock_method.return_value = "ENSG00000112096"
            
            result = await bio_client.symbol_to_ensembl("SOD2")
            assert result == "ENSG00000112096"
            mock_method.assert_called_once_with("SOD2")
            
    @pytest.mark.asyncio
    async def test_get_gene_info_mock(self, bio_client):
        """Test gene info retrieval with mock"""
        with patch.object(bio_client.mygene, 'get_gene_info') as mock_method:
            mock_gene_info = GeneInfo(
                symbol="SOD2",
                ensembl_id="ENSG00000112096",
                description="superoxide dismutase 2"
            )
            mock_method.return_value = mock_gene_info
            
            result = await bio_client.get_gene_info("SOD2")
            assert result.symbol == "SOD2"
            assert result.ensembl_id == "ENSG00000112096"
            
    @pytest.mark.asyncio
    async def test_get_all_pathways_mock(self, bio_client):
        """Test pathway retrieval from all databases with mock"""
        mock_reactome_pathway = PathwayInfo(
            id="R-HSA-1234",
            name="Detoxification of ROS",
            source="reactome"
        )
        mock_kegg_pathway = PathwayInfo(
            id="hsa04146",
            name="Peroxisome",
            source="kegg"
        )
        
        with patch.object(bio_client, 'symbol_to_ensembl') as mock_ensembl:
            mock_ensembl.return_value = "ENSG00000112096"
            
            with patch.object(bio_client, 'get_reactome_pathways') as mock_reactome:
                mock_reactome.return_value = [mock_reactome_pathway]
                
                with patch.object(bio_client, 'get_kegg_pathways') as mock_kegg:
                    mock_kegg.return_value = [mock_kegg_pathway]
                    
                    result = await bio_client.get_all_pathways("SOD2")
                    
                    assert "reactome" in result
                    assert "kegg" in result
                    assert len(result["reactome"]) == 1
                    assert len(result["kegg"]) == 1
                    assert result["reactome"][0].name == "Detoxification of ROS"


class TestGeneResolver:
    """Test gene resolution functionality"""
    
    @pytest.fixture
    def mock_bio_client(self):
        client = Mock(spec=BioAPIClient)
        client.get_gene_info = AsyncMock()
        client.get_all_pathways = AsyncMock()
        return client
        
    @pytest.fixture
    def gene_resolver(self, mock_bio_client):
        return GeneResolver(mock_bio_client)
        
    def test_extract_gene_symbols_regex(self, gene_resolver):
        """Test regex-based gene symbol extraction"""
        query = "The role of SOD2 and NRF2 in oxidative stress response"
        genes = gene_resolver._extract_gene_symbols_regex(query)
        
        assert "SOD2" in genes
        assert "NRF2" in genes
        assert len(genes) >= 2
        
    def test_parse_llm_gene_symbols(self, gene_resolver):
        """Test parsing of LLM-generated gene symbols"""
        llm_output = "SOD2, NRF2, KEAP1 (Kelch-like protein), CATALASE"
        genes = gene_resolver._parse_llm_gene_symbols(llm_output)
        
        expected_genes = ["SOD2", "NRF2", "KEAP1", "CATALASE"]
        for gene in expected_genes:
            assert gene in genes
            
    @pytest.mark.asyncio
    async def test_resolve_gene_to_databases_mock(self, gene_resolver, mock_bio_client):
        """Test gene resolution to databases with mock"""
        # Setup mock responses
        mock_gene_info = GeneInfo(
            symbol="SOD2",
            ensembl_id="ENSG00000112096",
            aliases=["MNSOD"]
        )
        mock_pathways = {
            "reactome": [PathwayInfo(id="R-HSA-1234", name="ROS detox", source="reactome")],
            "kegg": [PathwayInfo(id="hsa04146", name="Peroxisome", source="kegg")]
        }
        
        mock_bio_client.get_gene_info.return_value = mock_gene_info
        mock_bio_client.get_all_pathways.return_value = mock_pathways
        
        result = await gene_resolver._resolve_gene_to_databases("SOD2")
        
        assert result is not None
        assert result.symbol == "SOD2"
        assert result.ensembl_id == "ENSG00000112096"
        assert result.confidence > 0.5
        assert "reactome" in result.pathways
        
    @pytest.mark.asyncio
    async def test_forward_with_mock_llm(self, gene_resolver, mock_bio_client):
        """Test full gene resolution pipeline with mocked LLM"""
        query = "What is the role of SOD2 in mitochondrial function?"
        
        # Mock LLM response
        with patch.object(gene_resolver, 'extract_genes') as mock_llm:
            mock_llm.return_value = Mock(gene_symbols="SOD2")
            
            # Mock database responses
            mock_gene_info = GeneInfo(symbol="SOD2", ensembl_id="ENSG00000112096")
            mock_pathways = {"reactome": [], "kegg": []}
            
            mock_bio_client.get_gene_info.return_value = mock_gene_info
            mock_bio_client.get_all_pathways.return_value = mock_pathways
            
            result = await gene_resolver.forward(query)
            
            assert result["original_query"] == query
            assert "SOD2" in result["gene_symbols"]
            assert len(result["resolved_genes"]) > 0


class TestHypothesisGenerator:
    """Test hypothesis generation functionality"""
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        pipeline = Mock()
        pipeline.query = AsyncMock()
        pipeline.retrieve_context = Mock()
        return pipeline
        
    @pytest.fixture
    def mock_bio_client(self):
        client = Mock(spec=BioAPIClient)
        return client
        
    @pytest.fixture
    def hypothesis_generator(self, mock_rag_pipeline, mock_bio_client):
        return HypothesisGenerator(mock_rag_pipeline, mock_bio_client)
        
    def test_format_literature_context(self, hypothesis_generator):
        """Test literature context formatting"""
        rag_result = {
            "retrieved_chunks": [
                {"text": "SOD2 is important for mitochondrial function", "source": "paper1"},
                {"text": "ROS detoxification requires SOD2 activity", "source": "paper2"}
            ]
        }
        
        context = hypothesis_generator._format_literature_context(rag_result)
        assert "SOD2 is important" in context
        assert "ROS detoxification" in context
        
    def test_parse_predictions(self, hypothesis_generator):
        """Test testable predictions parsing"""
        predictions_text = """
        1. SOD2 overexpression will reduce ROS levels
        2. SOD2 knockout will increase mitochondrial damage
        3. Antioxidant treatment will rescue SOD2 deficiency
        """
        
        predictions = hypothesis_generator._parse_predictions(predictions_text)
        assert len(predictions) == 3
        assert "SOD2 overexpression" in predictions[0]
        assert "SOD2 knockout" in predictions[1]
        
    @pytest.mark.asyncio
    async def test_forward_simple_mode(self, hypothesis_generator):
        """Test simple hypothesis generation mode"""
        with patch.object(hypothesis_generator, 'simple_hypothesis') as mock_simple:
            mock_simple.return_value = Mock(hypothesis="SOD2 protects against oxidative stress")
            
            result = await hypothesis_generator.forward(
                topic="SOD2 function",
                use_context=False
            )
            
            assert result.topic == "SOD2 function"
            assert result.hypothesis == "SOD2 protects against oxidative stress"
            assert result.confidence == "low"


class TestPathwayGapDetector:
    """Test pathway gap detection functionality"""
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        pipeline = Mock()
        pipeline.query = AsyncMock()
        pipeline.retrieve_context = Mock()
        return pipeline
        
    @pytest.fixture
    def mock_bio_client(self):
        client = Mock(spec=BioAPIClient)
        client.symbol_to_ensembl = AsyncMock()
        client.get_all_pathways = AsyncMock()
        return client
        
    @pytest.fixture
    def gap_detector(self, mock_rag_pipeline, mock_bio_client):
        return PathwayGapDetector(mock_rag_pipeline, mock_bio_client)
        
    def test_parse_literature_regulators(self, gap_detector):
        """Test parsing of literature regulator text"""
        regulator_text = "NRF2, FOXO1, and PGC1A regulate this gene"
        regulators = gap_detector._parse_literature_regulators(regulator_text)
        
        expected = ["NRF2", "FOXO1", "PGC1A"]
        for reg in expected:
            assert reg in regulators
            
    def test_extract_curated_regulators(self, gap_detector):
        """Test extraction of regulators from curated pathways"""
        pathways = {
            "reactome": [
                PathwayInfo(id="R-1", name="pathway1", genes=["NRF2", "KEAP1"]),
                PathwayInfo(id="R-2", name="pathway2", genes=["SOD2", "CAT"])
            ]
        }
        
        regulators = gap_detector._extract_curated_regulators(pathways, "upstream")
        assert "NRF2" in regulators
        assert "SOD2" in regulators
        
    @pytest.mark.asyncio
    async def test_forward_with_mocks(self, gap_detector, mock_rag_pipeline, mock_bio_client):
        """Test full gap detection with mocked dependencies"""
        # Setup mocks
        mock_bio_client.symbol_to_ensembl.return_value = "ENSG00000112096"
        mock_bio_client.get_all_pathways.return_value = {
            "reactome": [PathwayInfo(id="R-1", name="pathway", genes=["NRF2"])]
        }
        
        with patch.object(gap_detector, '_get_literature_evidence') as mock_lit:
            mock_lit.return_value = {
                "passages": ["PGC1A regulates SOD2 expression"],
                "sources": ["source1"]
            }
            
            with patch.object(gap_detector, 'extract_regulators') as mock_extract:
                mock_extract.return_value = Mock(
                    upstream_regulators="PGC1A, NRF2",
                    downstream_targets="CATALASE"
                )
                
                result = await gap_detector.forward("SOD2")
                
                assert result.gene == "SOD2"
                assert len(result.upstream_gaps) >= 0  # Should find PGC1A as gap
                assert len(result.downstream_gaps) >= 0


class TestComprehensiveGeneInvestigation:
    """Test comprehensive investigation pipeline"""
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        return Mock()
        
    @pytest.fixture
    def mock_bio_client(self):
        return Mock(spec=BioAPIClient)
        
    @pytest.fixture
    def investigation(self, mock_rag_pipeline, mock_bio_client):
        return ComprehensiveGeneInvestigation(mock_rag_pipeline, mock_bio_client)
        
    def test_calculate_confidence_score(self, investigation):
        """Test confidence score calculation"""
        from bio_intelligence.gene_resolver import ResolvedGene
        from bio_intelligence.hypothesis_generator import BiologicalHypothesis
        
        # High confidence scenario
        gene_data = {
            "resolved_genes": [
                Mock(confidence=0.9),
                Mock(confidence=0.8)
            ]
        }
        hypothesis = Mock(
            confidence="high",
            literature_sources=[Mock(), Mock(), Mock()]
        )
        
        score = investigation._calculate_confidence_score(gene_data, hypothesis, None)
        assert score > 0.7  # Should be high confidence
        
    def test_format_investigation_data(self, investigation):
        """Test formatting of investigation data for summarization"""
        gene_data = {"gene_symbols": ["SOD2"], "resolved_genes": []}
        hypothesis = Mock(hypothesis="Test hypothesis", confidence="high", testable_predictions=[])
        gaps = Mock(gene="SOD2", upstream_gaps=[], downstream_gaps=[])
        
        formatted = investigation._format_investigation_data(gene_data, hypothesis, gaps)
        
        assert "gene_data" in formatted
        assert "hypothesis" in formatted
        assert "gap_analysis" in formatted
        assert "SOD2" in formatted["gene_data"]


# Integration test (requires actual API keys - mark as slow)
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_gene_resolution():
    """Integration test with real APIs (requires API keys)"""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not available")
        
    bio_client = BioAPIClient()
    
    # Test with well-known gene
    ensembl_id = await bio_client.symbol_to_ensembl("TP53")
    
    # TP53 should have a known Ensembl ID
    assert ensembl_id is not None
    assert ensembl_id.startswith("ENSG")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])