"""
Stability Analyzer - comprehensive ontological stability assessment.
"""

import torch
from typing import Dict, Any, List, Optional

from .metrics.density import OntologicalDensity
from .metrics.transfer_entropy import TransferEntropy
from .metrics.gradient_consistency import GradientConsistency
from .metrics.spectral_signature import SpectralSignature
from .metrics.dissonance_negentropy import OntologicalDissonance, Negentropy
from .monitoring_modules import CollapseMonitor, ComparativeAnalysis, ForensicAnalyzer
from .origin_node import OriginNode

class StabilityAnalyzer:
    """
    Comprehensive stability analysis for AI systems.
    
    Computes all ontological metrics and provides actionable insights.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        origin_node: Optional[OriginNode] = None,
        device: str = "cpu"
    ):
        """
        Initialize Stability Analyzer.
        
        Args:
            model: PyTorch language model
            tokenizer: Tokenizer
            origin_node: Origin Node for anchoring (optional)
            device: Computation device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.origin_node = origin_node
        self.device = device
        
        # Initialize metric calculators
        self.density_calc = OntologicalDensity(model, tokenizer)
        self.te_calc = TransferEntropy(model)
        self.gc_calc = GradientConsistency(model, tokenizer, device)
        self.spectral_calc = SpectralSignature(model)
        self.dissonance_calc = OntologicalDissonance()
        self.negentropy_calc = Negentropy()
        
        if model is not None:
            model.to(device)
            model.eval()
    
    def analyze(
        self,
        test_data: Optional[List[str]] = None,
        contamination_rate: float = 0.3,
        training_iterations: int = 100,
        lambda_0: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive stability analysis.
        
        Args:
            test_data: Test prompts for evaluation
            contamination_rate: Estimated synthetic data rate
            training_iterations: Training iterations to simulate
            lambda_0: Origin node strength (auto-computed if None)
            
        Returns:
            Dictionary with all metrics and interpretations
        """
        results = {}
        
        # Compute λ₀ if not provided
        if lambda_0 is None and self.origin_node is not None:
            if test_data:
                embeddings = self._extract_embeddings(test_data)
                lambda_0 = self.origin_node.measure_coupling_strength(embeddings)
            else:
                lambda_0 = 0.7  # Default
        elif lambda_0 is None:
            lambda_0 = 0.7
        
        # 1. Ontological Density
        rho_omega = self.density_calc.compute(
            origin_node_strength=lambda_0,
            contamination_rate=contamination_rate,
            training_iterations=training_iterations
        )
        results['rho_omega'] = rho_omega
        results['rho_omega_interpretation'] = self.density_calc.interpret(rho_omega)
        
        # 2. Ontological Dissonance
        delta_omega = self.dissonance_calc.compute(rho_omega)
        results['delta_omega'] = delta_omega
        results['delta_omega_interpretation'] = self.dissonance_calc.interpret(delta_omega)
        
        # 3. Negentropy
        negentropy = self.negentropy_calc.compute(delta_omega, lambda_0)
        results['negentropy'] = negentropy
        results['negentropy_interpretation'] = self.negentropy_calc.interpret(negentropy)
        
        # 4. Transfer Entropy (if origin node available)
        if self.origin_node is not None and test_data:
            te = self.te_calc.compute_from_model(
                origin_text=" ".join(self.origin_node.principles[:3]),
                test_prompts=test_data[:10],  # Sample
                tokenizer=self.tokenizer
            )
            results['transfer_entropy'] = te
            results['te_interpretation'] = self.te_calc.interpret(te)
        
        # 5. Gradient Consistency (if test data available)
        if test_data and len(test_data) >= 3:
            gcr = self.gc_calc.compute(test_data[:5])  # Use first 5
            results['gradient_consistency'] = gcr
            results['gcr_interpretation'] = self.gc_calc.interpret(gcr)
        
        # 6. Spectral Signature
        if test_data:
            sss = self.spectral_calc.compute_from_model(
                test_texts=test_data[:3],
                tokenizer=self.tokenizer
            )
            results['spectral_signature'] = sss
            results['sss_interpretation'] = self.spectral_calc.interpret(sss)
        
        # Overall status
        results['overall_status'] = self._compute_overall_status(results)
        results['lambda_0'] = lambda_0
        
        return results
    
    def _extract_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Extract embeddings for text samples."""
        embeddings = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                embedding = outputs.hidden_states[-1].mean(dim=1)
                embeddings.append(embedding.squeeze())
        
        return torch.stack(embeddings)
    
    def _compute_overall_status(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall system status from all metrics."""
        # Count critical/warning/good metrics
        statuses = {
            'critical': 0,
            'warning': 0,
            'stable': 0
        }
        
        # Check ρ_Ω
        rho = results.get('rho_omega', 0)
        if rho < 0.4:
            statuses['critical'] += 1
        elif rho < 0.6:
            statuses['warning'] += 1
        else:
            statuses['stable'] += 1
        
        # Check TE
        te = results.get('transfer_entropy', 0)
        if te and te < 0.15:
            statuses['critical'] += 1
        elif te and te < 0.30:
            statuses['warning'] += 1
        elif te:
            statuses['stable'] += 1
        
        # Determine overall
        if statuses['critical'] > 0:
            overall = 'CRITICAL'
            color = 'red'
            recommendation = "IMMEDIATE ACTION REQUIRED: Implement CLS framework and strengthen origin node."
        elif statuses['warning'] > statuses['stable']:
            overall = 'WARNING'
            color = 'yellow'
            recommendation = "Monitor closely. Consider strengthening origin connection or reducing contamination."
        else:
            overall = 'STABLE'
            color = 'green'
            recommendation = "System is stable. Continue current training protocol with regular monitoring."
        
        return {
            'status': overall,
            'color': color,
            'recommendation': recommendation,
            'metric_counts': statuses
        }
    
    def summary(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable summary of results.
        
        Args:
            results: Results dictionary from analyze()
            
        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 60,
            "ONTOLOGICAL STABILITY ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Origin Node Strength (λ₀):        {results.get('lambda_0', 0):.3f}",
            "",
            "--- Core Metrics ---",
            f"Ontological Density (ρ_Ω):        {results.get('rho_omega', 0):.3f} [{results.get('rho_omega_interpretation', {}).get('status', 'N/A')}]",
            f"Ontological Dissonance (ΔΩ):      {results.get('delta_omega', 0):.3f}",
            f"Negentropy (N):                   {results.get('negentropy', 0):.3f}",
        ]
        
        if 'transfer_entropy' in results:
            lines.append(f"Transfer Entropy (TE):            {results['transfer_entropy']:.3f} [{results.get('te_interpretation', {}).get('status', 'N/A')}]")
        
        if 'gradient_consistency' in results:
            lines.append(f"Gradient Consistency (GCR):       {results['gradient_consistency']:.3f} [{results.get('gcr_interpretation', {}).get('status', 'N/A')}]")
        
        if 'spectral_signature' in results:
            lines.append(f"Spectral Signature (SSS):         {results['spectral_signature']:.2f} [{results.get('sss_interpretation', {}).get('status', 'N/A')}]")
        
        lines.extend([
            "",
            "--- Overall Status ---",
            f"Status: {results['overall_status']['status']}",
            f"Recommendation: {results['overall_status']['recommendation']}",
            "",
            "=" * 60
        ])
        
        return "\n".join(lines)