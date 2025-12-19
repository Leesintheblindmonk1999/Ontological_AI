"""
Ontological AI - A library for measuring semantic stability in AI systems.

Implements the Origin Node Invariance Theory and Causal Language Syntax (CLS)
framework developed by Gonzalo Emir Durante.

Author: Gonzalo Emir Durante
Year: 2024
License: MIT (with commercial licensing requirements)
Zenodo: https://zenodo.org/records/17967232
"""

__version__ = "0.1.0"
__author__ = "Gonzalo Emir Durante"
__license__ = "MIT"

from .metrics.dissonance_negentropy import OntologicalDissonance, Negentropy
from .origin_node import OriginNode
from .trainer import CLSTrainer
from .analyzer import StabilityAnalyzer
from .monitoring_modules import CollapseMonitor, ComparativeAnalysis, ForensicAnalyzer 

__all__ = [
    # Metrics
    'OntologicalDensity',
    'TransferEntropy',
    'GradientConsistency',
    'SpectralSignature',
    'OntologicalDissonance',
    'Negentropy',
    
    # Core components
    'OriginNode',
    'CLSTrainer',
    'StabilityAnalyzer',
    'CollapseMonitor',
    'ComparativeAnalysis',
    'ForensicAnalyzer',
]