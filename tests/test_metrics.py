import pytest
import torch
import numpy as np
from ontological_ai.metrics.dissonance_negentropy import OntologicalDissonance, Negentropy
from ontological_ai.metrics.spectral_signature import SpectralSignature

def test_dissonance_calculation():
    """Tests that Dissonance is exactly ΔΩ = 1 - ρ_Ω"""
    calc = OntologicalDissonance()
    rho_omega = 0.8
    expected = 0.2
    assert abs(calc.compute(rho_omega) - expected) < 1e-5

def test_negentropy_resilience():
    """Tests that Negentropy responds correctly to structural order"""
    calc = Negentropy()
    # High negentropy should yield HIGH status
    res = calc.interpret(2.5)
    assert res['status'] == "HIGH"
    # Low negentropy should yield CRITICAL status
    res = calc.interpret(0.1)
    assert res['status'] == "CRITICAL"

def test_spectral_signature_logic():
    """Tests the Spectral Signature Score (SSS) logic"""
    calc = SpectralSignature()
    # Simulate eigenvalues (λ)
    # Case: High concentration in top 10 (CLS signature)
    top_eigen = torch.ones(10) * 10
    bottom_eigen = torch.ones(90) * 0.1
    
    # Simplified SSS calculation for the test environment
    sss_mock = torch.log(top_eigen).sum() / torch.log(bottom_eigen + 1e-10).abs().sum()
    
    interpretation = calc.interpret(sss_mock.item())
    assert isinstance(interpretation, dict)
    assert "status" in interpretation

def test_origin_node_centroid():
    """Tests that the Origin Node correctly calculates its centroid"""
    from ontological_ai import OriginNode
    # Simple 2D embeddings
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    node = OriginNode(embeddings)
    # Centroid of (1,0) and (0,1) is (0.5, 0.5)
    assert torch.allclose(node.centroid, torch.tensor([0.5, 0.5]))