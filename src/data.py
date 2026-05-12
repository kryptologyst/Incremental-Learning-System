"""Data loading and preprocessing utilities for incremental learning."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for incremental learning experiments."""
    
    def __init__(self, dataset_name: str = 'iris', test_size: float = 0.2, random_state: int = 42):
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess the specified dataset."""
        if self.dataset_name == 'iris':
            data = load_iris()
        elif self.dataset_name == 'wine':
            data = load_wine()
        elif self.dataset_name == 'breast_cancer':
            data = load_breast_cancer()
        elif self.dataset_name == 'synthetic':
            data = make_classification(
                n_samples=1000, 
                n_features=20, 
                n_classes=3, 
                n_clusters_per_class=1,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        if isinstance(data, tuple):
            X, y = data
        else:
            X, y = data.data, data.target
        
        # Convert to DataFrame for easier handling
        if isinstance(X, np.ndarray):
            feature_names = getattr(data, 'feature_names', [f'feature_{i}' for i in range(X.shape[1])])
            X = pd.DataFrame(X, columns=feature_names)
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Loaded {self.dataset_name} dataset: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test samples")
        logger.info(f"Features: {X_train_scaled.shape[1]}, Classes: {len(np.unique(y))}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def create_streaming_data(self, X: np.ndarray, y: np.ndarray, 
                            batch_size: int = 15) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create streaming data batches for incremental learning."""
        batches = []
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            batches.append((batch_X, batch_y))
        
        logger.info(f"Created {len(batches)} batches of size {batch_size}")
        return batches
    
    def simulate_concept_drift(self, X: np.ndarray, y: np.ndarray, 
                             drift_point: int = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate concept drift by modifying the data distribution."""
        drift_idx = int(len(X) * drift_point)
        
        # Create a copy of the data
        X_drifted = X.copy()
        y_drifted = y.copy()
        
        # Apply concept drift after drift_point
        if drift_idx < len(X):
            # Add noise to features
            noise_scale = 0.1
            X_drifted[drift_idx:] += np.random.normal(0, noise_scale, X_drifted[drift_idx:].shape)
            
            # Optionally modify labels (for demonstration)
            # This is a simple example - in practice, concept drift is more complex
            logger.info(f"Applied concept drift at index {drift_idx}")
        
        return X_drifted, y_drifted


class IncrementalDataGenerator:
    """Generate synthetic data for incremental learning experiments."""
    
    def __init__(self, n_features: int = 20, n_classes: int = 3, random_state: int = 42):
        self.n_features = n_features
        self.n_classes = n_classes
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_sequential_tasks(self, n_tasks: int = 3, samples_per_task: int = 200) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate sequential learning tasks."""
        tasks = []
        
        for task_id in range(n_tasks):
            # Generate data for each task with slightly different distributions
            X, y = make_classification(
                n_samples=samples_per_task,
                n_features=self.n_features,
                n_classes=self.n_classes,
                n_clusters_per_class=1,
                random_state=self.random_state + task_id,
                class_sep=1.0 + task_id * 0.2  # Increase separation for later tasks
            )
            
            # Add task-specific noise
            noise_scale = 0.05 * task_id
            X += np.random.normal(0, noise_scale, X.shape)
            
            tasks.append((X, y))
            logger.info(f"Generated task {task_id + 1}: {X.shape[0]} samples, {X.shape[1]} features")
        
        return tasks
    
    def generate_non_stationary_data(self, n_samples: int = 1000, 
                                   drift_frequency: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Generate non-stationary data with periodic concept drift."""
        X = []
        y = []
        
        for i in range(0, n_samples, drift_frequency):
            batch_size = min(drift_frequency, n_samples - i)
            
            # Generate batch with changing parameters
            batch_X, batch_y = make_classification(
                n_samples=batch_size,
                n_features=self.n_features,
                n_classes=self.n_classes,
                n_clusters_per_class=1,
                random_state=self.random_state + i,
                class_sep=1.0 + (i // drift_frequency) * 0.1
            )
            
            X.append(batch_X)
            y.append(batch_y)
        
        X = np.vstack(X)
        y = np.hstack(y)
        
        logger.info(f"Generated non-stationary data: {X.shape[0]} samples with drift every {drift_frequency} samples")
        return X, y


def load_real_world_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load real-world datasets for incremental learning."""
    if dataset_name == 'iris':
        data = load_iris()
    elif dataset_name == 'wine':
        data = load_wine()
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
    else:
        raise ValueError(f"Unsupported real-world dataset: {dataset_name}")
    
    return data.data, data.target


def create_incremental_splits(X: np.ndarray, y: np.ndarray, 
                            n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create incremental splits for continual learning evaluation."""
    splits = []
    samples_per_split = len(X) // n_splits
    
    for i in range(n_splits):
        start_idx = i * samples_per_split
        end_idx = (i + 1) * samples_per_split if i < n_splits - 1 else len(X)
        
        split_X = X[start_idx:end_idx]
        split_y = y[start_idx:end_idx]
        
        splits.append((split_X, split_y))
    
    logger.info(f"Created {len(splits)} incremental splits")
    return splits
