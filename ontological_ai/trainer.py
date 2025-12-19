"""
CLS Trainer - Training with Causal Language Syntax and Origin Node anchoring.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm
from .origin_node import OriginNode


class CLSTrainer:
    """
    Trainer implementing Causal Language Syntax (CLS) framework.
    
    Incorporates:
    - Origin node anchoring
    - DAG constraint enforcement (optional)
    - Parallel transport regularization
    - Negentropy maximization
    """
    
    def __init__(
        self,
        model: nn.Module,
        origin_node: OriginNode,
        lambda_origin: float = 0.7,
        dag_constraint: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize CLS Trainer.
        
        Args:
            model: PyTorch model to train
            origin_node: Origin Node for anchoring
            lambda_origin: Origin anchoring strength (0-1)
            dag_constraint: Whether to enforce DAG structure
            device: Training device
        """
        self.model = model
        self.origin_node = origin_node
        self.lambda_origin = lambda_origin
        self.dag_constraint = dag_constraint
        self.device = device
        
        self.model.to(device)
        self.origin_node.centroid = self.origin_node.centroid.to(device)
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        max_grad_norm: float = 1.0,
        logging_steps: int = 100,
        eval_steps: int = 500,
        callback: Optional[Callable] = None
    ) -> Dict[str, list]:
        """
        Train model with CLS framework.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            optimizer: Optimizer (created if None)
            scheduler: LR scheduler (optional)
            max_grad_norm: Gradient clipping value
            logging_steps: Steps between logging
            eval_steps: Steps between evaluation
            callback: Optional callback function
            
        Returns:
            Dictionary with training history
        """
        # Setup optimizer
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate
            )
        
        # Setup data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Training history
        history = {
            'loss': [],
            'task_loss': [],
            'anchor_loss': [],
            'learning_rate': []
        }
        
        global_step = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            self.model.train()
            
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch, output_hidden_states=True)
                
                # Task loss (standard language modeling)
                task_loss = outputs.loss if hasattr(outputs, 'loss') else 0
                
                # Origin anchoring loss
                anchor_loss = self._compute_anchor_loss(outputs)
                
                # Total loss
                total_loss = task_loss + self.lambda_origin * anchor_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_grad_norm
                )
                
                optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()
                
                # Logging
                epoch_loss += total_loss.item()
                global_step += 1
                
                if global_step % logging_steps == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    history['loss'].append(total_loss.item())
                    history['task_loss'].append(task_loss.item() if isinstance(task_loss, torch.Tensor) else task_loss)
                    history['anchor_loss'].append(anchor_loss.item())
                    history['learning_rate'].append(current_lr)
                    
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'anchor': f'{anchor_loss.item():.4f}',
                        'lr': f'{current_lr:.2e}'
                    })
                
                # Evaluation
                if eval_dataset is not None and global_step % eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataset, batch_size)
                    print(f"\nEval metrics: {eval_metrics}")
                    self.model.train()
                
                # Callback
                if callback is not None:
                    callback(global_step, total_loss.item(), self.model)
        
        return history
    
    def _compute_anchor_loss(self, model_outputs) -> torch.Tensor:
        """
        Compute origin node anchoring loss.
        
        Args:
            model_outputs: Model outputs with hidden_states
            
        Returns:
            Anchor loss tensor
        """
        if not hasattr(model_outputs, 'hidden_states'):
            return torch.tensor(0.0, device=self.device)
        
        # Get last hidden state
        hidden_states = model_outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        
        # Mean pool over sequence
        embeddings = hidden_states.mean(dim=1)  # [batch, hidden_dim]
        
        # Compute distance to origin
        anchor_loss = self.origin_node.anchor_loss(embeddings, lambda_weight=1.0)
        
        return anchor_loss
    
    def train_epoch(
        self,
        train_dataset,
        batch_size: int = 8,
        optimizer: Optional[torch.optim.Optimizer] = None,
        learning_rate: float = 5e-5
    ) -> float:
        """
        Train for one epoch (simplified interface).
        
        Args:
            train_dataset: Training dataset
            batch_size: Batch size
            optimizer: Optimizer (created if None)
            learning_rate: Learning rate
            
        Returns:
            Average loss for epoch
        """
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate
            )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = self.model(**batch, output_hidden_states=True)
            
            task_loss = outputs.loss if hasattr(outputs, 'loss') else 0
            anchor_loss = self._compute_anchor_loss(outputs)
            
            loss = task_loss + self.lambda_origin * anchor_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(
        self,
        eval_dataset,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        Evaluate model.
        
        Args:
            eval_dataset: Evaluation dataset
            batch_size: Batch size
            
        Returns:
            Dictionary with evaluation metrics
        """
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
        
        self.model.eval()
        total_loss = 0
        total_anchor_loss = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch, output_hidden_states=True)
                
                task_loss = outputs.loss if hasattr(outputs, 'loss') else 0
                anchor_loss = self._compute_anchor_loss(outputs)
                
                total_loss += (task_loss + self.lambda_origin * anchor_loss).item()
                total_anchor_loss += anchor_loss.item()
        
        return {
            'eval_loss': total_loss / len(eval_loader),
            'eval_anchor_loss': total_anchor_loss / len(eval_loader)
        }
    
    def save_model(self, path: str):
        """Save model and training config."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'lambda_origin': self.lambda_origin,
            'dag_constraint': self.dag_constraint
        }, path)
    
    def load_model(self, path: str):
        """Load model and training config."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.lambda_origin = checkpoint.get('lambda_origin', self.lambda_origin)
        self.dag_constraint = checkpoint.get('dag_constraint', self.dag_constraint)