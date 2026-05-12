"""Core incremental learning models and algorithms."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

logger.info(f"Random seed set to {RANDOM_SEED}")


class IncrementalLearner(ABC):
    """Abstract base class for incremental learning algorithms."""
    
    @abstractmethod
    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None) -> None:
        """Update the model with new data incrementally."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        pass


class SGDIncrementalLearner(IncrementalLearner):
    """Stochastic Gradient Descent based incremental learner."""
    
    def __init__(self, loss: str = 'log', max_iter: int = 1000, random_state: int = RANDOM_SEED):
        self.model = SGDClassifier(
            loss=loss, 
            max_iter=max_iter, 
            random_state=random_state,
            warm_start=True
        )
        self.is_fitted = False
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None) -> None:
        """Update the model with new data incrementally."""
        if not self.is_fitted:
            self.model.partial_fit(X, y, classes=classes)
            self.is_fitted = True
        else:
            self.model.partial_fit(X, y)
        logger.info(f"Updated SGD model with {len(X)} samples")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        return self.model.score(X, y)


class OnlineNaiveBayes(IncrementalLearner):
    """Online Naive Bayes classifier for incremental learning."""
    
    def __init__(self, alpha: float = 1.0):
        self.model = MultinomialNB(alpha=alpha)
        self.is_fitted = False
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None) -> None:
        """Update the model with new data incrementally."""
        if not self.is_fitted:
            self.model.partial_fit(X, y, classes=classes)
            self.is_fitted = True
        else:
            self.model.partial_fit(X, y)
        logger.info(f"Updated Naive Bayes model with {len(X)} samples")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        return self.model.score(X, y)


class ElasticWeightConsolidation(nn.Module):
    """Elastic Weight Consolidation (EWC) for continual learning."""
    
    def __init__(self, model: nn.Module, lambda_reg: float = 1000.0):
        super().__init__()
        self.model = model
        self.lambda_reg = lambda_reg
        self.fisher_info = {}
        self.optimal_params = {}
        
    def compute_fisher_information(self, dataloader: DataLoader) -> None:
        """Compute Fisher Information Matrix for EWC."""
        self.model.eval()
        fisher_info = {}
        
        for name, param in self.model.named_parameters():
            fisher_info[name] = torch.zeros_like(param)
        
        for batch_x, batch_y in dataloader:
            self.model.zero_grad()
            output = self.model(batch_x)
            loss = nn.CrossEntropyLoss()(output, batch_y)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
        
        # Average over number of samples
        for name in fisher_info:
            fisher_info[name] /= len(dataloader.dataset)
        
        self.fisher_info = fisher_info
        
    def ewc_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute EWC regularization loss."""
        ce_loss = nn.CrossEntropyLoss()(output, target)
        ewc_loss = 0
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_info and name in self.optimal_params:
                ewc_loss += (self.fisher_info[name] * 
                           (param - self.optimal_params[name]) ** 2).sum()
        
        return ce_loss + self.lambda_reg * ewc_loss
    
    def save_optimal_params(self) -> None:
        """Save current model parameters as optimal."""
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()


class ExperienceReplay:
    """Experience Replay buffer for continual learning."""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer = []
        
    def add(self, experience: Tuple[np.ndarray, np.ndarray]) -> None:
        """Add experience to the buffer."""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Remove oldest experience
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch_x = np.array([self.buffer[i][0] for i in indices])
        batch_y = np.array([self.buffer[i][1] for i in indices])
        
        return batch_x, batch_y
    
    def __len__(self) -> int:
        return len(self.buffer)


class IncrementalLearningSystem:
    """Main system for incremental learning experiments."""
    
    def __init__(self, algorithm: str = 'sgd', **kwargs):
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.model = self._create_model()
        self.metrics_history = []
        
    def _create_model(self) -> IncrementalLearner:
        """Create the specified incremental learning model."""
        if self.algorithm == 'sgd':
            return SGDIncrementalLearner(**self.kwargs)
        elif self.algorithm == 'naive_bayes':
            return OnlineNaiveBayes(**self.kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def train_incremental(self, X: np.ndarray, y: np.ndarray, 
                         batch_size: int = 15, classes: Optional[np.ndarray] = None) -> List[float]:
        """Train the model incrementally with mini-batches."""
        accuracies = []
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            # Update model with current batch
            self.model.partial_fit(batch_X, batch_y, classes=classes)
            
            # Evaluate on remaining data
            if i + batch_size < len(X):
                remaining_X = X[i+batch_size:]
                remaining_y = y[i+batch_size:]
                accuracy = self.model.score(remaining_X, remaining_y)
                accuracies.append(accuracy)
                logger.info(f"Batch {i//batch_size + 1}: Accuracy = {accuracy:.4f}")
        
        self.metrics_history.extend(accuracies)
        return accuracies
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data."""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics
    
    def adapt_to_new_data(self, X_new: np.ndarray, y_new: np.ndarray) -> float:
        """Adapt the model to new incoming data."""
        self.model.partial_fit(X_new, y_new)
        logger.info(f"Adapted model to {len(X_new)} new samples")
        
        # Return current accuracy (would need test set for proper evaluation)
        return self.model.score(X_new, y_new)


def create_device() -> torch.device:
    """Create appropriate device for PyTorch operations."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using CUDA device")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    return device
