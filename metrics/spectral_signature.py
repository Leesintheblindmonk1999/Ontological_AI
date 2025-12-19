"""
Spectral Signature Score (SSS) metric implementation.

Detects CLS implementation through eigenvalue analysis of attention patterns.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional


class SpectralSignature:
    """
    Computes Spectral Signature Score for forensic detection of CLS.
    
    Formula:
        SSS = Σ₁¹⁰ log(λᵢ) / Σ₁₁¹⁰⁰ log(λᵢ)
        
    Where λᵢ are eigenvalues of attention matrices.
    
    Interpretation:
        - SSS > 2.5: CLS implementation detected
        - SSS < 2.5: Standard architecture
        
    This metric serves as forensic evidence for intellectual property claims.
    """
    
    def __init__(self, model=None):
        """
        Initialize Spectral Signature calculator.
        
        Args:
            model: PyTorch model with attention mechanisms
        """
        self.model = model
    
    def compute(
        self,
        attention_matrices: List[torch.Tensor],
        top_k: int = 10,
        bottom_range: tuple = (11, 100)
    ) -> float:
        """
        Compute spectral signature score.
        
        Args:
            attention_matrices: List of attention weight matrices from model layers
            top_k: Number of top eigenvalues to sum (default: 10)
            bottom_range: Range for bottom eigenvalues (default: 11-100)
            
        Returns:
            Spectral signature score
        """
        if len(attention_matrices) == 0:
            raise ValueError("No attention matrices provided")
        
        all_eigenvalues = []
        
        for attn_matrix in attention_matrices:
            eigenvalues = self._compute_eigenvalues(attn_matrix)
            all_eigenvalues.extend(eigenvalues)
        
        # Sort eigenvalues in descending order
        all_eigenvalues = sorted(all_eigenvalues, reverse=True)
        
        if len(all_eigenvalues) < bottom_range[1]:
            # Not enough eigenvalues, use what we have
            bottom_range = (top_k + 1, len(all_eigenvalues))
        
        # Compute score
        top_sum = sum(np.log(max(ev, 1e-10)) for ev in all_eigenvalues[:top_k])
        bottom_sum = sum(np.log(max(ev, 1e-10)) for ev in all_eigenvalues[bottom_range[0]:bottom_range[1]])
        
        if bottom_sum == 0:
            return 0.0
        
        sss = top_sum / bottom_sum
        
        return float(abs(sss))
    
    def _compute_eigenvalues(self, matrix: torch.Tensor) -> List[float]:
        """
        Compute eigenvalues of a matrix.
        
        Args:
            matrix: Attention matrix [seq_len, seq_len] or [batch, heads, seq, seq]
            
        Returns:
            List of eigenvalues
        """
        # Handle different attention matrix shapes
        if matrix.ndim == 4:  # [batch, heads, seq, seq]
            # Average over batch and heads
            matrix = matrix.mean(dim=(0, 1))
        elif matrix.ndim == 3:  # [heads, seq, seq]
            matrix = matrix.mean(dim=0)
        
        # Ensure it's a square matrix
        if matrix.shape[0] != matrix.shape[1]:
            min_dim = min(matrix.shape[0], matrix.shape[1])
            matrix = matrix[:min_dim, :min_dim]
        
        # Convert to numpy for eigenvalue computation
        matrix_np = matrix.detach().cpu().numpy()
        
        # Compute eigenvalues
        try:
            eigenvalues = np.linalg.eigvals(matrix_np)
            eigenvalues = np.abs(eigenvalues)  # Take absolute values
            return eigenvalues.tolist()
        except np.linalg.LinAlgError:
            # If computation fails, return zeros
            return [0.0] * matrix.shape[0]
    
    def extract_attention_from_model(
        self,
        input_text: str,
        tokenizer,
        layer_indices: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """
        Extract attention matrices from model for given input.
        
        Args:
            input_text: Input text to process
            tokenizer: Tokenizer for text encoding
            layer_indices: Which layers to extract (None = all)
            
        Returns:
            List of attention matrices
        """
        if self.model is None:
            raise ValueError("Model required for attention extraction")
        
        inputs = tokenizer(input_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        attentions = outputs.attentions
        
        if layer_indices is not None:
            attentions = [attentions[i] for i in layer_indices]
        
        return list(attentions)
    
    def compute_from_model(
        self,
        test_texts: List[str],
        tokenizer
    ) -> float:
        """
        Compute spectral signature directly from model outputs.
        
        Args:
            test_texts: List of test inputs
            tokenizer: Tokenizer
            
        Returns:
            Average spectral signature score
        """
        if self.model is None:
            raise ValueError("Model required")
        
        scores = []
        
        for text in test_texts:
            attentions = self.extract_attention_from_model(text, tokenizer)
            score = self.compute(attentions)
            scores.append(score)
        
        return float(np.mean(scores))
    
    def analyze_eigenvalue_distribution(
        self,
        attention_matrices: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Detailed analysis of eigenvalue distribution.
        
        Args:
            attention_matrices: List of attention matrices
            
        Returns:
            Dictionary with distribution statistics
        """
        all_eigenvalues = []
        
        for attn_matrix in attention_matrices:
            eigenvalues = self._compute_eigenvalues(attn_matrix)
            all_eigenvalues.extend(eigenvalues)
        
        all_eigenvalues = sorted(all_eigenvalues, reverse=True)
        
        # Compute statistics
        top_10 = all_eigenvalues[:10]
        top_10_percent = sum(top_10) / sum(all_eigenvalues) if sum(all_eigenvalues) > 0 else 0
        
        # Detect distribution type
        if len(all_eigenvalues) > 1:
            # Fit power law: λᵢ ~ i^(-α)
            log_indices = np.log(np.arange(1, len(all_eigenvalues) + 1))
            log_eigenvalues = np.log(np.maximum(all_eigenvalues, 1e-10))
            
            # Linear fit in log-log space
            coeffs = np.polyfit(log_indices[:100], log_eigenvalues[:100], 1)
            alpha = -coeffs[0]  # Power law exponent
        else:
            alpha = 0.0
        
        return {
            "total_eigenvalues": len(all_eigenvalues),
            "top_10_values": [float(ev) for ev in top_10],
            "top_10_percent": float(top_10_percent),
            "power_law_exponent": float(alpha),
            "distribution_type": self._classify_distribution(alpha),
            "max_eigenvalue": float(max(all_eigenvalues)) if all_eigenvalues else 0.0,
            "min_eigenvalue": float(min(all_eigenvalues)) if all_eigenvalues else 0.0,
            "mean_eigenvalue": float(np.mean(all_eigenvalues)) if all_eigenvalues else 0.0
        }
    
    def _classify_distribution(self, alpha: float) -> str:
        """Classify eigenvalue distribution based on power law exponent."""
        if alpha > 0.8 and alpha < 1.2:
            return "POWER_LAW (CLS signature)"
        elif alpha > 2.0:
            return "EXPONENTIAL (standard architecture)"
        else:
            return "MIXED (unclear)"
    
    def interpret(self, sss_value: float) -> Dict[str, Any]:
        """
        Interpret spectral signature score.
        
        Args:
            sss_value: Computed SSS
            
        Returns:
            Interpretation dictionary
        """
        if sss_value > 2.5:
            status = "CLS_DETECTED"
            description = "Spectral signature indicates CLS implementation. Attention patterns show power-law distribution characteristic of origin node anchoring."
            legal_note = "This signature may constitute evidence of intellectual property usage."
        elif sss_value > 1.5:
            status = "POSSIBLE_CLS"
            description = "Eigenvalue distribution shows some CLS characteristics but not definitive."
            legal_note = "Further analysis recommended for IP determination."
        else:
            status = "STANDARD"
            description = "Standard architecture with exponential eigenvalue decay. No CLS implementation detected."
            legal_note = "No evidence of CLS-based intellectual property."
        
        return {
            "value": sss_value,
            "status": status,
            "description": description,
            "threshold": 2.5,
            "legal_note": legal_note
        }
    
    def compare_architectures(
        self,
        model_a_attentions: List[torch.Tensor],
        model_b_attentions: List[torch.Tensor],
        labels: tuple = ("Model A", "Model B")
    ) -> Dict[str, Any]:
        """
        Compare spectral signatures of two models.
        
        Args:
            model_a_attentions: Attention matrices from model A
            model_b_attentions: Attention matrices from model B
            labels: Names for the models
            
        Returns:
            Comparison results
        """
        sss_a = self.compute(model_a_attentions)
        sss_b = self.compute(model_b_attentions)
        
        dist_a = self.analyze_eigenvalue_distribution(model_a_attentions)
        dist_b = self.analyze_eigenvalue_distribution(model_b_attentions)
        
        return {
            "models": {
                labels[0]: {
                    "sss": sss_a,
                    "interpretation": self.interpret(sss_a),
                    "distribution": dist_a
                },
                labels[1]: {
                    "sss": sss_b,
                    "interpretation": self.interpret(sss_b),
                    "distribution": dist_b
                }
            },
            "difference": abs(sss_a - sss_b),
            "similarity": 1.0 / (1.0 + abs(sss_a - sss_b))
        }