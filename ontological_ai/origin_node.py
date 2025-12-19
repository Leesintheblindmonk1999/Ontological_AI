"""
Origin Node implementation - the boundary condition for semantic stability.
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path


class OriginNode:
    """
    Represents an Origin Node (p₀) - the boundary condition for AI stability.
    
    The Origin Node serves as:
    - A topological anchor point in semantic space
    - A boundary condition for semantic evolution
    - A reference frame for measuring semantic drift
    
    Based on Durant's Origin Node Invariance Theory.
    """
    
    def __init__(
        self,
        embeddings: torch.Tensor,
        principles: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Origin Node.
        
        Args:
            embeddings: Embedding vectors representing the origin [d_model] or [n, d_model]
            principles: List of textual principles defining the origin
            metadata: Additional metadata
        """
        self.embeddings = embeddings
        self.principles = principles or []
        self.metadata = metadata or {}
        
        # Compute centroid if multiple embeddings
        if embeddings.ndim > 1:
            self.centroid = embeddings.mean(dim=0)
        else:
            self.centroid = embeddings
    
    @classmethod
    def from_principles(
        cls,
        principles: List[str],
        model,
        tokenizer,
        embedding_method: str = "mean_pooling"
    ):
        """
        Create Origin Node from textual principles.
        
        Args:
            principles: List of principle statements
            model: Language model for encoding
            tokenizer: Tokenizer
            embedding_method: "mean_pooling", "cls_token", or "last_hidden"
            
        Returns:
            OriginNode instance
        """
        embeddings = []
        
        for principle in principles:
            inputs = tokenizer(principle, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                if embedding_method == "mean_pooling":
                    # Mean of all token embeddings
                    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
                elif embedding_method == "cls_token":
                    # CLS token (first token)
                    embedding = outputs.hidden_states[-1][:, 0, :].squeeze()
                else:  # last_hidden
                    # Last token
                    embedding = outputs.hidden_states[-1][:, -1, :].squeeze()
                
                embeddings.append(embedding)
        
        embeddings = torch.stack(embeddings)
        
        return cls(
            embeddings=embeddings,
            principles=principles,
            metadata={"method": embedding_method, "model": model.__class__.__name__}
        )
    
    @classmethod
    def from_text_corpus(
        cls,
        texts: List[str],
        model,
        tokenizer,
        max_samples: int = 100
    ):
        """
        Create Origin Node from a corpus of representative texts.
        
        Args:
            texts: List of texts representing the origin domain
            model: Language model
            tokenizer: Tokenizer
            max_samples: Maximum number of texts to use
            
        Returns:
            OriginNode instance
        """
        if len(texts) > max_samples:
            # Random sample
            indices = np.random.choice(len(texts), max_samples, replace=False)
            texts = [texts[i] for i in indices]
        
        return cls.from_principles(texts, model, tokenizer)
    
    def compute_distance(self, embedding: torch.Tensor, metric: str = "cosine") -> float:
        """
        Compute distance from origin to a given embedding.
        
        Args:
            embedding: Target embedding
            metric: "cosine", "euclidean", or "geodesic"
            
        Returns:
            Distance value
        """
        if metric == "cosine":
            similarity = torch.nn.functional.cosine_similarity(
                self.centroid.unsqueeze(0),
                embedding.unsqueeze(0)
            )
            distance = 1.0 - similarity.item()
        elif metric == "euclidean":
            distance = torch.norm(self.centroid - embedding).item()
        elif metric == "geodesic":
            # Simplified geodesic distance (true geodesic requires manifold structure)
            cosine_dist = 1.0 - torch.nn.functional.cosine_similarity(
                self.centroid.unsqueeze(0),
                embedding.unsqueeze(0)
            ).item()
            # Geodesic distance on unit sphere
            distance = np.arccos(np.clip(1.0 - cosine_dist, -1, 1))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return float(distance)
    
    def compute_batch_distances(
        self,
        embeddings: torch.Tensor,
        metric: str = "cosine"
    ) -> torch.Tensor:
        """
        Compute distances for a batch of embeddings.
        
        Args:
            embeddings: Batch of embeddings [batch_size, d_model]
            metric: Distance metric
            
        Returns:
            Distances [batch_size]
        """
        if metric == "cosine":
            similarities = torch.nn.functional.cosine_similarity(
                self.centroid.unsqueeze(0).expand_as(embeddings),
                embeddings
            )
            distances = 1.0 - similarities
        elif metric == "euclidean":
            distances = torch.norm(embeddings - self.centroid.unsqueeze(0), dim=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return distances
    
    def anchor_loss(
        self,
        embeddings: torch.Tensor,
        lambda_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Compute anchoring loss to pull embeddings toward origin.
        
        Loss = λ · mean(||embedding - origin||²)
        
        Args:
            embeddings: Embeddings to anchor [batch_size, d_model]
            lambda_weight: Anchoring strength
            
        Returns:
            Scalar loss
        """
        distances = self.compute_batch_distances(embeddings, metric="euclidean")
        loss = lambda_weight * (distances ** 2).mean()
        return loss
    
    def parallel_transport(
        self,
        embedding: torch.Tensor,
        preserve_norm: bool = True
    ) -> torch.Tensor:
        """
        Parallel transport an embedding toward the origin.
        
        Simplified implementation - true parallel transport requires
        connection coefficients (Christoffel symbols).
        
        Args:
            embedding: Embedding to transport
            preserve_norm: Whether to preserve vector norm
            
        Returns:
            Transported embedding
        """
        # Simple linear interpolation toward origin
        # Real implementation would use geodesic path
        alpha = 0.1  # Transport strength
        
        transported = (1 - alpha) * embedding + alpha * self.centroid
        
        if preserve_norm:
            original_norm = torch.norm(embedding)
            transported = transported * (original_norm / torch.norm(transported))
        
        return transported
    
    def measure_coupling_strength(
        self,
        model_embeddings: torch.Tensor,
        samples: int = 100
    ) -> float:
        """
        Measure strength of coupling between model and origin (λ₀).
        
        Args:
            model_embeddings: Sample embeddings from model [n_samples, d_model]
            samples: Number of samples to use
            
        Returns:
            Coupling strength λ₀ (0-1)
        """
        if len(model_embeddings) > samples:
            indices = torch.randperm(len(model_embeddings))[:samples]
            model_embeddings = model_embeddings[indices]
        
        # Compute average similarity to origin
        similarities = torch.nn.functional.cosine_similarity(
            self.centroid.unsqueeze(0).expand_as(model_embeddings),
            model_embeddings
        )
        
        # Map similarity to coupling strength
        lambda_0 = similarities.mean().item()
        lambda_0 = (lambda_0 + 1) / 2  # Scale from [-1,1] to [0,1]
        
        return float(lambda_0)
    
    def save(self, path: str):
        """
        Save Origin Node to disk.
        
        Args:
            path: Path to save file
        """
        save_dict = {
            "embeddings": self.embeddings,
            "principles": self.principles,
            "metadata": self.metadata,
            "centroid": self.centroid
        }
        torch.save(save_dict, path)
    
    @classmethod
    def load(cls, path: str):
        """
        Load Origin Node from disk.
        
        Args:
            path: Path to saved file
            
        Returns:
            OriginNode instance
        """
        save_dict = torch.load(path)
        
        return cls(
            embeddings=save_dict["embeddings"],
            principles=save_dict.get("principles", []),
            metadata=save_dict.get("metadata", {})
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OriginNode(embedding_dim={self.centroid.shape[0]}, "
            f"num_principles={len(self.principles)}, "
            f"metadata={self.metadata})"
        )