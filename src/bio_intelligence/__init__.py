"""
Bio Intelligence Module

Advanced biological intelligence modules for gene resolution, hypothesis generation,
and knowledge gap detection using DSPy and biological databases.
"""

try:
    from .bio_apis import BioAPIClient
    from .composed_modules import ComprehensiveGeneInvestigation
    from .gap_detector import PathwayGapDetector
    from .gene_resolver import GeneResolver
    from .hypothesis_generator import HypothesisGenerator
except ImportError:
    from bio_apis import BioAPIClient
    from composed_modules import ComprehensiveGeneInvestigation
    from gap_detector import PathwayGapDetector
    from gene_resolver import GeneResolver
    from hypothesis_generator import HypothesisGenerator

__all__ = [
    "BioAPIClient",
    "GeneResolver", 
    "HypothesisGenerator",
    "PathwayGapDetector",
    "ComprehensiveGeneInvestigation"
]