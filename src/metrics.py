"""Evaluation metrics and utilities for incremental learning."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class IncrementalLearningMetrics:
    """Metrics for evaluating incremental learning performance."""
    
    def __init__(self):
        self.metrics_history = []
        self.task_metrics = {}
        
    def compute_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute basic classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def compute_forgetting_metrics(self, task_accuracies: List[List[float]]) -> Dict[str, float]:
        """Compute catastrophic forgetting metrics."""
        task_accuracies = np.array(task_accuracies)
        
        # Backward Transfer (BWT) - how much learning new tasks hurts old tasks
        bwt = 0.0
        for i in range(1, task_accuracies.shape[0]):
            for j in range(i):
                bwt += task_accuracies[i, j] - task_accuracies[j, j]
        
        if task_accuracies.shape[0] > 1:
            bwt /= (task_accuracies.shape[0] * (task_accuracies.shape[0] - 1) / 2)
        
        # Forward Transfer (FWT) - how much learning old tasks helps new tasks
        fwt = 0.0
        for i in range(1, task_accuracies.shape[0]):
            fwt += task_accuracies[i, i] - task_accuracies[0, i]
        
        if task_accuracies.shape[0] > 1:
            fwt /= (task_accuracies.shape[0] - 1)
        
        # Average Accuracy (ACC) - average accuracy across all tasks
        acc = np.mean(np.diag(task_accuracies))
        
        return {
            'backward_transfer': bwt,
            'forward_transfer': fwt,
            'average_accuracy': acc,
            'catastrophic_forgetting': -bwt  # Negative BWT indicates forgetting
        }
    
    def compute_learning_efficiency(self, accuracies: List[float], 
                                  sample_counts: List[int]) -> Dict[str, float]:
        """Compute learning efficiency metrics."""
        if len(accuracies) != len(sample_counts):
            raise ValueError("Accuracies and sample counts must have same length")
        
        # Area Under Learning Curve (AULC)
        aulc = np.trapz(accuracies, sample_counts)
        
        # Learning Speed (final accuracy / total samples)
        learning_speed = accuracies[-1] / sample_counts[-1] if sample_counts[-1] > 0 else 0
        
        # Sample Efficiency (samples needed to reach 90% of final accuracy)
        target_accuracy = 0.9 * accuracies[-1]
        sample_efficiency = sample_counts[-1]  # Default to total samples
        
        for i, acc in enumerate(accuracies):
            if acc >= target_accuracy:
                sample_efficiency = sample_counts[i]
                break
        
        return {
            'area_under_learning_curve': aulc,
            'learning_speed': learning_speed,
            'sample_efficiency': sample_efficiency
        }
    
    def compute_stability_plasticity(self, accuracies: List[float]) -> Dict[str, float]:
        """Compute stability-plasticity trade-off metrics."""
        if len(accuracies) < 2:
            return {'stability': 0.0, 'plasticity': 0.0, 'stability_plasticity_ratio': 0.0}
        
        # Stability: resistance to forgetting (low variance in accuracy)
        stability = 1.0 - np.var(accuracies) / np.mean(accuracies) if np.mean(accuracies) > 0 else 0.0
        
        # Plasticity: ability to learn new information (improvement over time)
        plasticity = (accuracies[-1] - accuracies[0]) / accuracies[0] if accuracies[0] > 0 else 0.0
        
        # Stability-Plasticity Ratio
        sp_ratio = stability / (plasticity + 1e-8)  # Add small epsilon to avoid division by zero
        
        return {
            'stability': stability,
            'plasticity': plasticity,
            'stability_plasticity_ratio': sp_ratio
        }
    
    def update_metrics(self, task_id: int, metrics: Dict[str, float]) -> None:
        """Update metrics for a specific task."""
        if task_id not in self.task_metrics:
            self.task_metrics[task_id] = []
        
        self.task_metrics[task_id].append(metrics)
        self.metrics_history.append(metrics)
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary of all computed metrics."""
        if not self.metrics_history:
            return {}
        
        # Aggregate metrics across all tasks
        summary = {}
        for metric_name in self.metrics_history[0].keys():
            values = [m[metric_name] for m in self.metrics_history if metric_name in m]
            if values:
                summary[f'{metric_name}_mean'] = np.mean(values)
                summary[f'{metric_name}_std'] = np.std(values)
                summary[f'{metric_name}_final'] = values[-1]
        
        return summary


class ContinualLearningEvaluator:
    """Evaluator for continual learning scenarios."""
    
    def __init__(self):
        self.metrics = IncrementalLearningMetrics()
        self.task_results = {}
        
    def evaluate_task_sequence(self, model, tasks: List[Tuple[np.ndarray, np.ndarray]], 
                             test_tasks: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Evaluate model on a sequence of tasks."""
        task_accuracies = []
        
        for task_id, (X_train, y_train) in enumerate(tasks):
            # Train on current task
            model.train_incremental(X_train, y_train)
            
            # Evaluate on all previous tasks (including current)
            task_accs = []
            for test_id, (X_test, y_test) in enumerate(test_tasks[:task_id + 1]):
                metrics = model.evaluate(X_test, y_test)
                task_accs.append(metrics['accuracy'])
                logger.info(f"Task {task_id + 1}, Test {test_id + 1}: Accuracy = {metrics['accuracy']:.4f}")
            
            task_accuracies.append(task_accs)
        
        # Compute continual learning metrics
        forgetting_metrics = self.metrics.compute_forgetting_metrics(task_accuracies)
        
        return {
            'task_accuracies': task_accuracies,
            'forgetting_metrics': forgetting_metrics,
            'final_accuracy': task_accuracies[-1][-1] if task_accuracies else 0.0
        }
    
    def evaluate_online_learning(self, model, X_stream: np.ndarray, y_stream: np.ndarray,
                               batch_size: int = 15, evaluation_frequency: int = 5) -> Dict[str, Any]:
        """Evaluate online learning performance."""
        accuracies = []
        sample_counts = []
        
        for i in range(0, len(X_stream), batch_size):
            batch_X = X_stream[i:i+batch_size]
            batch_y = y_stream[i:i+batch_size]
            
            # Update model
            model.model.partial_fit(batch_X, batch_y)
            
            # Evaluate periodically
            if i % (batch_size * evaluation_frequency) == 0 and i > 0:
                # Evaluate on remaining data
                remaining_X = X_stream[i:]
                remaining_y = y_stream[i:]
                
                if len(remaining_X) > 0:
                    accuracy = model.model.score(remaining_X, remaining_y)
                    accuracies.append(accuracy)
                    sample_counts.append(i)
                    logger.info(f"Sample {i}: Accuracy = {accuracy:.4f}")
        
        # Compute learning efficiency metrics
        efficiency_metrics = self.metrics.compute_learning_efficiency(accuracies, sample_counts)
        stability_metrics = self.metrics.compute_stability_plasticity(accuracies)
        
        return {
            'accuracies': accuracies,
            'sample_counts': sample_counts,
            'efficiency_metrics': efficiency_metrics,
            'stability_metrics': stability_metrics
        }


def create_metrics_dataframe(metrics_history: List[Dict[str, float]]) -> pd.DataFrame:
    """Create a pandas DataFrame from metrics history."""
    if not metrics_history:
        return pd.DataFrame()
    
    df = pd.DataFrame(metrics_history)
    df.index.name = 'iteration'
    return df


def plot_learning_curves(metrics_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot learning curves from metrics DataFrame."""
    if metrics_df.empty:
        logger.warning("No metrics data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Incremental Learning Performance', fontsize=16)
    
    # Accuracy over time
    if 'accuracy' in metrics_df.columns:
        axes[0, 0].plot(metrics_df.index, metrics_df['accuracy'])
        axes[0, 0].set_title('Accuracy Over Time')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True)
    
    # F1 Score over time
    if 'f1_macro' in metrics_df.columns:
        axes[0, 1].plot(metrics_df.index, metrics_df['f1_macro'])
        axes[0, 1].set_title('F1 Score Over Time')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('F1 Score (Macro)')
        axes[0, 1].grid(True)
    
    # Precision vs Recall
    if 'precision_macro' in metrics_df.columns and 'recall_macro' in metrics_df.columns:
        axes[1, 0].scatter(metrics_df['precision_macro'], metrics_df['recall_macro'])
        axes[1, 0].set_title('Precision vs Recall')
        axes[1, 0].set_xlabel('Precision (Macro)')
        axes[1, 0].set_ylabel('Recall (Macro)')
        axes[1, 0].grid(True)
    
    # Learning efficiency
    if 'accuracy' in metrics_df.columns:
        cumulative_samples = np.arange(1, len(metrics_df) + 1) * 15  # Assuming batch size 15
        axes[1, 1].plot(cumulative_samples, metrics_df['accuracy'])
        axes[1, 1].set_title('Learning Efficiency')
        axes[1, 1].set_xlabel('Total Samples Seen')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curves saved to {save_path}")
    
    plt.show()


def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, 
                               class_names: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> None:
    """Create and display confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()
