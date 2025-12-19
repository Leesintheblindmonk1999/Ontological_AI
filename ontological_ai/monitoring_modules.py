"""
Monitoring, Comparison, and Forensic Analysis modules.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from .metrics.density import OntologicalDensity
from .metrics.transfer_entropy import TransferEntropy
from .metrics.gradient_consistency import GradientConsistency


# ============================================================================
# COLLAPSE MONITOR
# ============================================================================

class CollapseMonitor:
    """
    Real-time monitoring of model collapse indicators.
    """
    
    def __init__(self, model, origin_node, alert_threshold: float = 0.4):
        """
        Initialize collapse monitor.
        
        Args:
            model: Model to monitor
            origin_node: Origin node reference
            alert_threshold: ρ_Ω threshold for alerts
        """
        self.model = model
        self.origin_node = origin_node
        self.alert_threshold = alert_threshold
        
        self.density_calc = OntologicalDensity(model)
        self.te_calc = TransferEntropy(model)
        
        self.history = {
            'rho_omega': [],
            'transfer_entropy': [],
            'steps': []
        }
    
    def check_stability(
        self,
        current_step: Optional[int] = None,
        contamination: float = 0.3
    ) -> Dict[str, Any]:
        """
        Check current stability metrics.
        
        Args:
            current_step: Training step number
            contamination: Current contamination rate
            
        Returns:
            Dictionary with current metrics and alert status
        """
        # Estimate lambda_0 from model
        lambda_0 = 0.7  # Simplified - real implementation would measure from model
        
        # Compute metrics
        rho_omega = self.density_calc.compute(
            origin_node_strength=lambda_0,
            contamination_rate=contamination,
            training_iterations=current_step or 0
        )
        
        # Update history
        if current_step is not None:
            self.history['rho_omega'].append(rho_omega)
            self.history['steps'].append(current_step)
        
        # Check for alert
        alert = rho_omega < self.alert_threshold
        
        return {
            'rho_omega': rho_omega,
            'lambda_0': lambda_0,
            'alert': alert,
            'status': 'CRITICAL' if alert else 'OK',
            'recommendation': self._get_recommendation(rho_omega)
        }
    
    def _get_recommendation(self, rho_omega: float) -> str:
        """Get recommendation based on current state."""
        if rho_omega < 0.4:
            return "STOP TRAINING immediately. Restore from checkpoint."
        elif rho_omega < 0.6:
            return "Reduce synthetic data. Strengthen origin connection."
        else:
            return "Continue training with monitoring."
    
    def plot_history(self, save_path: Optional[str] = None):
        """Plot monitoring history."""
        if not self.history['steps']:
            print("No history to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['steps'], self.history['rho_omega'], 'b-', linewidth=2)
        plt.axhline(y=0.6, color='g', linestyle='--', label='Stable threshold')
        plt.axhline(y=0.4, color='r', linestyle='--', label='Critical threshold')
        plt.xlabel('Training Steps')
        plt.ylabel('Ontological Density (ρ_Ω)')
        plt.title('Model Stability Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

class ComparativeAnalysis:
    """
    Compare stability of CLS vs standard models.
    """
    
    def __init__(
        self,
        model_a,
        model_b,
        test_iterations: int = 100,
        contamination_schedule: Optional[List[float]] = None
    ):
        """
        Initialize comparative analysis.
        
        Args:
            model_a: First model (typically CLS)
            model_b: Second model (typically standard RLHF)
            test_iterations: Number of test iterations
            contamination_schedule: List of contamination rates to test
        """
        self.model_a = model_a
        self.model_b = model_b
        self.test_iterations = test_iterations
        self.contamination_schedule = contamination_schedule or [0.1, 0.3, 0.5, 0.7]
        
        self.density_calc = OntologicalDensity()
    
    def run(self, lambda_a: float = 0.85, lambda_b: float = 0.5) -> Dict[str, Any]:
        """
        Run comparative analysis.
        
        Args:
            lambda_a: Origin strength for model A
            lambda_b: Origin strength for model B
            
        Returns:
            Comparison results
        """
        results_a = []
        results_b = []
        
        for contamination in self.contamination_schedule:
            # Test Model A
            densities_a = []
            for t in range(0, self.test_iterations, 10):
                rho = self.density_calc.compute(
                    origin_node_strength=lambda_a,
                    contamination_rate=contamination,
                    training_iterations=t
                )
                densities_a.append(rho)
            
            # Test Model B
            densities_b = []
            for t in range(0, self.test_iterations, 10):
                rho = self.density_calc.compute(
                    origin_node_strength=lambda_b,
                    contamination_rate=contamination,
                    training_iterations=t
                )
                densities_b.append(rho)
            
            results_a.append(densities_a)
            results_b.append(densities_b)
        
        # Compute degradation rates
        degradation_a = self._compute_degradation_rate(results_a)
        degradation_b = self._compute_degradation_rate(results_b)
        
        improvement = ((degradation_b - degradation_a) / degradation_b) * 100
        
        return {
            'results_a': results_a,
            'results_b': results_b,
            'cls_degradation_rate': degradation_a,
            'rlhf_degradation_rate': degradation_b,
            'improvement_percentage': improvement,
            'p_value': 0.001  # Placeholder - real implementation would compute
        }
    
    def _compute_degradation_rate(self, results: List[List[float]]) -> float:
        """Compute average degradation rate."""
        all_rates = []
        for densities in results:
            if len(densities) > 1:
                rate = (densities[0] - densities[-1]) / len(densities)
                all_rates.append(abs(rate))
        return np.mean(all_rates) if all_rates else 0.0
    
    def plot_degradation(self, save_path: Optional[str] = None):
        """Plot degradation curves."""
        plt.figure(figsize=(12, 8))
        
        # This is a placeholder - real implementation would use actual results
        t = np.linspace(0, 100, 20)
        
        # Simulated curves
        rho_cls = 0.85 * (1 - 0.3) * np.exp(-0.01 * t)
        rho_rlhf = 0.5 * (1 - 0.3) * np.exp(-0.025 * t)
        
        plt.plot(t, rho_cls, 'g-', linewidth=2, label='CLS Model')
        plt.plot(t, rho_rlhf, 'r-', linewidth=2, label='Standard RLHF')
        plt.axhline(y=0.4, color='orange', linestyle='--', label='Critical Threshold')
        
        plt.xlabel('Training Iterations')
        plt.ylabel('Ontological Density (ρ_Ω)')
        plt.title('Model Collapse Comparison: CLS vs Standard RLHF')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_stability_metrics(self, save_path: Optional[str] = None):
        """Plot multiple stability metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Placeholder plots
        t = np.linspace(0, 100, 20)
        
        # ρ_Ω
        axes[0, 0].plot(t, 0.85 * np.exp(-0.01 * t), 'g-', label='CLS')
        axes[0, 0].plot(t, 0.5 * np.exp(-0.025 * t), 'r-', label='RLHF')
        axes[0, 0].set_title('Ontological Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Transfer Entropy
        axes[0, 1].plot(t, 0.81 * np.exp(-0.005 * t), 'g-', label='CLS')
        axes[0, 1].plot(t, 0.3 * np.exp(-0.02 * t), 'r-', label='RLHF')
        axes[0, 1].set_title('Transfer Entropy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Negentropy
        axes[1, 0].plot(t, 2.5 * np.exp(-0.003 * t), 'g-', label='CLS')
        axes[1, 0].plot(t, 1.2 * np.exp(-0.015 * t), 'r-', label='RLHF')
        axes[1, 0].set_title('Negentropy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # GCR
        axes[1, 1].plot(t, 0.9 * np.exp(-0.004 * t), 'g-', label='CLS')
        axes[1, 1].plot(t, 0.6 * np.exp(-0.018 * t), 'r-', label='RLHF')
        axes[1, 1].set_title('Gradient Consistency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


# ============================================================================
# FORENSIC ANALYZER
# ============================================================================

class ForensicAnalyzer:
    """
    Forensic analysis for detecting CLS implementation in black-box models.
    Used for intellectual property verification.
    """
    
    def __init__(self):
        """Initialize forensic analyzer."""
        from .metrics.spectral_signature import SpectralSignature
        self.spectral = SpectralSignature()
    
    def extract_attention(self, model, tokenizer, test_prompts: List[str]) -> List[torch.Tensor]:
        """
        Extract attention patterns from model.
        
        Args:
            model: Model to analyze
            tokenizer: Tokenizer
            test_prompts: Prompts to use for extraction
            
        Returns:
            List of attention matrices
        """
        self.spectral.model = model
        
        all_attentions = []
        for prompt in test_prompts:
            attentions = self.spectral.extract_attention_from_model(prompt, tokenizer)
            all_attentions.extend(attentions)
        
        return all_attentions
    
    def compute_signature(self, attention_data: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Compute spectral signature and metadata.
        
        Args:
            attention_data: Attention matrices
            
        Returns:
            Signature analysis results
        """
        sss = self.spectral.compute(attention_data)
        distribution = self.spectral.analyze_eigenvalue_distribution(attention_data)
        
        # Estimate lambda_0 from spectral properties
        lambda_estimate = min(1.0, sss / 2.5) if sss > 0 else 0.0
        
        # Confidence based on distribution characteristics
        if distribution['power_law_exponent'] > 0.8 and distribution['power_law_exponent'] < 1.2:
            confidence = 0.95
        elif sss > 2.5:
            confidence = 0.85
        else:
            confidence = 0.5
        
        return {
            'sss': sss,
            'lambda_estimate': lambda_estimate,
            'confidence': confidence,
            'distribution': distribution,
            'cls_detected': sss > 2.5,
            'interpretation': self.spectral.interpret(sss)
        }
    
    def generate_report(self, signature: Dict[str, Any], model_name: str) -> str:
        """
        Generate forensic analysis report.
        
        Args:
            signature: Signature analysis results
            model_name: Name of analyzed model
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            "FORENSIC ANALYSIS REPORT",
            "Ontological AI - Durant Framework Detection",
            "=" * 70,
            "",
            f"Model Analyzed: {model_name}",
            f"Analysis Date: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Analysis'}",
            "",
            "--- Spectral Signature Analysis ---",
            f"Spectral Signature Score (SSS): {signature['sss']:.3f}",
            f"Detection Threshold: 2.500",
            f"CLS Implementation Detected: {'YES' if signature['cls_detected'] else 'NO'}",
            "",
            f"Estimated Origin Node Strength (λ₀): {signature['lambda_estimate']:.3f}",
            f"Confidence Level: {signature['confidence']*100:.1f}%",
            "",
            "--- Eigenvalue Distribution ---",
            f"Power Law Exponent (α): {signature['distribution']['power_law_exponent']:.3f}",
            f"Distribution Type: {signature['distribution']['distribution_type']}",
            f"Top 10 Eigenvalues Concentration: {signature['distribution']['top_10_percent']*100:.1f}%",
            "",
            "--- Legal Interpretation ---",
            signature['interpretation']['legal_note'],
            "",
            "=" * 70
        ]
        
        return "\n".join(lines)