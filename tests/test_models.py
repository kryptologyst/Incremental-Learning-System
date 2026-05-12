"""Test suite for incremental learning system."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import (
    IncrementalLearningSystem, SGDIncrementalLearner, 
    OnlineNaiveBayes, ElasticWeightConsolidation, ExperienceReplay
)
from src.data import DataLoader, IncrementalDataGenerator
from src.metrics import IncrementalLearningMetrics, ContinualLearningEvaluator


class TestSGDIncrementalLearner:
    """Test SGD incremental learner."""
    
    def test_initialization(self):
        """Test SGD learner initialization."""
        learner = SGDIncrementalLearner()
        assert learner.model is not None
        assert not learner.is_fitted
    
    def test_partial_fit(self):
        """Test partial fit functionality."""
        learner = SGDIncrementalLearner()
        X = np.random.randn(10, 4)
        y = np.random.randint(0, 3, 10)
        
        learner.partial_fit(X, y, classes=np.unique(y))
        assert learner.is_fitted
    
    def test_predict(self):
        """Test prediction functionality."""
        learner = SGDIncrementalLearner()
        X_train = np.random.randn(20, 4)
        y_train = np.random.randint(0, 3, 20)
        X_test = np.random.randn(5, 4)
        
        learner.partial_fit(X_train, y_train, classes=np.unique(y_train))
        predictions = learner.predict(X_test)
        
        assert len(predictions) == 5
        assert all(pred in np.unique(y_train) for pred in predictions)
    
    def test_score(self):
        """Test scoring functionality."""
        learner = SGDIncrementalLearner()
        X_train = np.random.randn(20, 4)
        y_train = np.random.randint(0, 3, 20)
        X_test = np.random.randn(10, 4)
        y_test = np.random.randint(0, 3, 10)
        
        learner.partial_fit(X_train, y_train, classes=np.unique(y_train))
        score = learner.score(X_test, y_test)
        
        assert 0 <= score <= 1


class TestOnlineNaiveBayes:
    """Test Online Naive Bayes learner."""
    
    def test_initialization(self):
        """Test Naive Bayes learner initialization."""
        learner = OnlineNaiveBayes()
        assert learner.model is not None
        assert not learner.is_fitted
    
    def test_partial_fit(self):
        """Test partial fit functionality."""
        learner = OnlineNaiveBayes()
        X = np.random.randn(10, 4)
        y = np.random.randint(0, 3, 10)
        
        learner.partial_fit(X, y, classes=np.unique(y))
        assert learner.is_fitted


class TestIncrementalLearningSystem:
    """Test main incremental learning system."""
    
    def test_initialization(self):
        """Test system initialization."""
        system = IncrementalLearningSystem(algorithm='sgd')
        assert system.algorithm == 'sgd'
        assert system.model is not None
    
    def test_train_incremental(self):
        """Test incremental training."""
        system = IncrementalLearningSystem(algorithm='sgd')
        X = np.random.randn(30, 4)
        y = np.random.randint(0, 3, 30)
        
        accuracies = system.train_incremental(X, y, batch_size=10, classes=np.unique(y))
        
        assert len(accuracies) > 0
        assert all(0 <= acc <= 1 for acc in accuracies)
    
    def test_evaluate(self):
        """Test evaluation functionality."""
        system = IncrementalLearningSystem(algorithm='sgd')
        X_train = np.random.randn(30, 4)
        y_train = np.random.randint(0, 3, 30)
        X_test = np.random.randn(10, 4)
        y_test = np.random.randint(0, 3, 10)
        
        system.train_incremental(X_train, y_train, classes=np.unique(y_train))
        metrics = system.evaluate(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_load_iris_dataset(self):
        """Test loading iris dataset."""
        loader = DataLoader(dataset_name='iris')
        X_train, X_test, y_train, y_test = loader.load_dataset()
        
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
    
    def test_create_streaming_data(self):
        """Test creating streaming data batches."""
        loader = DataLoader()
        X = np.random.randn(30, 4)
        y = np.random.randint(0, 3, 30)
        
        batches = loader.create_streaming_data(X, y, batch_size=10)
        
        assert len(batches) == 3
        assert all(len(batch[0]) == 10 for batch in batches[:-1])
    
    def test_simulate_concept_drift(self):
        """Test concept drift simulation."""
        loader = DataLoader()
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 3, 100)
        
        X_drifted, y_drifted = loader.simulate_concept_drift(X, y, drift_point=0.5)
        
        assert X_drifted.shape == X.shape
        assert y_drifted.shape == y.shape


class TestIncrementalDataGenerator:
    """Test synthetic data generation."""
    
    def test_generate_sequential_tasks(self):
        """Test generating sequential tasks."""
        generator = IncrementalDataGenerator()
        tasks = generator.generate_sequential_tasks(n_tasks=3, samples_per_task=100)
        
        assert len(tasks) == 3
        assert all(task[0].shape[0] == 100 for task in tasks)
    
    def test_generate_non_stationary_data(self):
        """Test generating non-stationary data."""
        generator = IncrementalDataGenerator()
        X, y = generator.generate_non_stationary_data(n_samples=500, drift_frequency=100)
        
        assert X.shape[0] == 500
        assert len(y) == 500


class TestIncrementalLearningMetrics:
    """Test metrics computation."""
    
    def test_compute_basic_metrics(self):
        """Test basic metrics computation."""
        metrics = IncrementalLearningMetrics()
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 1])
        
        result = metrics.compute_basic_metrics(y_true, y_pred)
        
        assert 'accuracy' in result
        assert result['accuracy'] == 1.0
    
    def test_compute_forgetting_metrics(self):
        """Test forgetting metrics computation."""
        metrics = IncrementalLearningMetrics()
        task_accuracies = [
            [0.8, 0.0, 0.0],
            [0.7, 0.9, 0.0],
            [0.6, 0.8, 0.85]
        ]
        
        result = metrics.compute_forgetting_metrics(task_accuracies)
        
        assert 'backward_transfer' in result
        assert 'forward_transfer' in result
        assert 'average_accuracy' in result
    
    def test_compute_learning_efficiency(self):
        """Test learning efficiency metrics."""
        metrics = IncrementalLearningMetrics()
        accuracies = [0.5, 0.7, 0.8, 0.85]
        sample_counts = [10, 20, 30, 40]
        
        result = metrics.compute_learning_efficiency(accuracies, sample_counts)
        
        assert 'area_under_learning_curve' in result
        assert 'learning_speed' in result
        assert 'sample_efficiency' in result


class TestExperienceReplay:
    """Test experience replay buffer."""
    
    def test_add_experience(self):
        """Test adding experiences to buffer."""
        buffer = ExperienceReplay(capacity=5)
        experience = (np.array([1, 2, 3]), np.array([1]))
        
        buffer.add(experience)
        assert len(buffer) == 1
    
    def test_sample_experience(self):
        """Test sampling experiences from buffer."""
        buffer = ExperienceReplay(capacity=10)
        
        # Add some experiences
        for i in range(5):
            experience = (np.array([i]), np.array([i % 3]))
            buffer.add(experience)
        
        batch_x, batch_y = buffer.sample(3)
        
        assert len(batch_x) == 3
        assert len(batch_y) == 3


if __name__ == "__main__":
    pytest.main([__file__])
