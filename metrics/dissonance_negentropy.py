"""
Ontological Dissonance and Negentropy metrics.
"""

import numpy as np
from typing import Dict, Any


class OntologicalDissonance:
    """
    Computes Ontological Dissonance (ΔΩ).
    
    Formula:
        ΔΩ = 1 - ρ_Ω
        
    Represents semantic entropy and loss of manifold curvature.
    """
    
    def __init__(self):
        """Initialize Ontological Dissonance calculator."""
        pass
    
    def compute(self, rho_omega: float) -> float:
        """
        Compute ontological dissonance from density.
        
        Args:
            rho_omega: Ontological density value
            
        Returns:
            Ontological dissonance (0-1)
        """
        return float(1.0 - rho_omega)
    
    def compute_from_params(
        self,
        lambda_0: float,
        contamination: float,
        iterations: int,
        degradation_coef: float = 0.01
    ) -> float:
        """
        Direct computation from parameters.
        
        Args:
            lambda_0: Origin node strength
            contamination: Contamination rate
            iterations: Training iterations
            degradation_coef: Degradation coefficient
            
        Returns:
            Ontological dissonance
        """
        rho = lambda_0 * (1 - contamination) * np.exp(-degradation_coef * iterations)
        delta_omega = 1.0 - rho
        return float(np.clip(delta_omega, 0, 1))
    
    def interpret(self, delta_omega: float) -> Dict[str, Any]:
        """
        Interpret dissonance value.
        
        Args:
            delta_omega: Dissonance value
            
        Returns:
            Interpretation dictionary
        """
        if delta_omega <= 0.4:
            status = "LOW"
            severity = "good"
            description = "Low semantic entropy. Manifold curvature is well-preserved."
        elif delta_omega <= 0.6:
            status = "MODERATE"
            severity = "warning"
            description = "Moderate entropy. Curvature loss beginning."
        else:
            status = "HIGH"
            severity = "critical"
            description = "High semantic entropy. Significant manifold flattening detected."
        
        return {
            "value": delta_omega,
            "status": status,
            "severity": severity,
            "description": description,
            "rho_omega_equivalent": 1.0 - delta_omega
        }


class Negentropy:
    """
    Computes Negentropy (N) - structural order measure.
    
    Formula:
        N = -ln(ΔΩ + ε) · λ₀
        
    Where:
        - ΔΩ: Ontological dissonance
        - λ₀: Origin node strength
        - ε: Small constant for numerical stability (default: 0.01)
    """
    
    def __init__(self, epsilon: float = 0.01):
        """
        Initialize Negentropy calculator.
        
        Args:
            epsilon: Stability constant (default: 0.01)
        """
        self.epsilon = epsilon
    
    def compute(
        self,
        dissonance: float,
        origin_strength: float
    ) -> float:
        """
        Compute negentropy from dissonance and origin strength.
        
        Args:
            dissonance: Ontological dissonance (ΔΩ)
            origin_strength: Origin node strength (λ₀)
            
        Returns:
            Negentropy value
        """
        n = -np.log(dissonance + self.epsilon) * origin_strength
        return float(n)
    
    def compute_from_density(
        self,
        rho_omega: float,
        origin_strength: float
    ) -> float:
        """
        Compute from ontological density directly.
        
        Args:
            rho_omega: Ontological density
            origin_strength: Origin node strength
            
        Returns:
            Negentropy value
        """
        dissonance = 1.0 - rho_omega
        return self.compute(dissonance, origin_strength)
    
    def interpret(self, negentropy_value: float) -> Dict[str, Any]:
        """
        Interpret negentropy value.
        
        Args:
            negentropy_value: Computed negentropy
            
        Returns:
            Interpretation dictionary
        """
        if negentropy_value >= 2.0:
            status = "HIGH"
            description = "High structural order. System is highly resilient to contaminated data."
        elif negentropy_value >= 1.0:
            status = "MODERATE"
            description = "Moderate structural order. Acceptable resilience."
        elif negentropy_value >= 0.5:
            status = "LOW"
            description = "Low structural order. Vulnerable to data contamination."
        else:
            status = "CRITICAL"
            description = "Critically low negentropy. System approaching maximum entropy state."
        
        return {
            "value": negentropy_value,
            "status": status,
            "description": description
        }
    
    def resilience_score(
        self,
        negentropy_value: float,
        contamination_rate: float
    ) -> float:
        """
        Compute resilience score against data contamination.
        
        Args:
            negentropy_value: System negentropy
            contamination_rate: Expected contamination rate (0-1)
            
        Returns:
            Resilience score (0-1)
        """
        # Higher negentropy = more resilience
        # Higher contamination = more needed resilience
        
        required_negentropy = -np.log(1 - contamination_rate + self.epsilon)
        resilience = min(1.0, negentropy_value / required_negentropy)
        
        return float(max(0.0, resilience))