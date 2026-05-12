"""Utility functions for the incremental learning system."""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def create_directories(directories: List[str]) -> None:
    """Create directories if they don't exist."""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, dict):
            json_results[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    json_results[key][sub_key] = sub_value.tolist()
                else:
                    json_results[key][sub_key] = sub_value
        else:
            json_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Results loaded from {filepath}")
    return results


def save_model(model: Any, filepath: str) -> None:
    """Save model using pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """Load model using pickle."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded from {filepath}")
    return model


def create_summary_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create a summary table from experiment results."""
    summary_data = []
    
    for experiment_name, result in results.items():
        row = {'Experiment': experiment_name}
        
        if 'test_metrics' in result:
            # Baseline experiment
            row.update({
                'Algorithm': 'SGD',
                'Accuracy': result['test_metrics']['accuracy'],
                'F1 Score': result['test_metrics']['classification_report']['macro avg']['f1-score'],
                'Precision': result['test_metrics']['classification_report']['macro avg']['precision'],
                'Recall': result['test_metrics']['classification_report']['macro avg']['recall']
            })
        elif 'forgetting_metrics' in result:
            # Continual learning experiment
            row.update({
                'Algorithm': 'SGD',
                'Final Accuracy': result['final_accuracy'],
                'Backward Transfer': result['forgetting_metrics']['backward_transfer'],
                'Forward Transfer': result['forgetting_metrics']['forward_transfer'],
                'Average Accuracy': result['forgetting_metrics']['average_accuracy']
            })
        elif 'efficiency_metrics' in result:
            # Online learning experiment
            row.update({
                'Algorithm': 'SGD',
                'Final Accuracy': result['accuracies'][-1] if result['accuracies'] else 0.0,
                'Learning Speed': result['efficiency_metrics']['learning_speed'],
                'Sample Efficiency': result['efficiency_metrics']['sample_efficiency'],
                'AULC': result['efficiency_metrics']['area_under_learning_curve']
            })
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def plot_comparison(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None) -> None:
    """Create comparison plots for different experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Incremental Learning System - Experiment Comparison', fontsize=16)
    
    # Extract data for plotting
    experiments = list(results.keys())
    
    # Plot 1: Accuracy comparison
    if any('test_metrics' in result for result in results.values()):
        baseline_results = [result for result in results.values() if 'test_metrics' in result]
        if baseline_results:
            accuracies = [result['test_metrics']['accuracy'] for result in baseline_results]
            axes[0, 0].bar(range(len(accuracies)), accuracies)
            axes[0, 0].set_title('Baseline Accuracy Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_xticks(range(len(accuracies)))
            axes[0, 0].set_xticklabels([f'Exp {i+1}' for i in range(len(accuracies))])
    
    # Plot 2: Learning curves
    if any('accuracies' in result for result in results.values()):
        for exp_name, result in results.items():
            if 'accuracies' in result and result['accuracies']:
                axes[0, 1].plot(result['accuracies'], label=exp_name, marker='o')
        axes[0, 1].set_title('Learning Curves')
        axes[0, 1].set_xlabel('Batch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot 3: Forgetting metrics
    if any('forgetting_metrics' in result for result in results.values()):
        forgetting_data = []
        exp_names = []
        for exp_name, result in results.items():
            if 'forgetting_metrics' in result:
                forgetting_data.append([
                    result['forgetting_metrics']['backward_transfer'],
                    result['forgetting_metrics']['forward_transfer'],
                    result['forgetting_metrics']['average_accuracy']
                ])
                exp_names.append(exp_name)
        
        if forgetting_data:
            forgetting_df = pd.DataFrame(forgetting_data, 
                                        columns=['Backward Transfer', 'Forward Transfer', 'Average Accuracy'],
                                        index=exp_names)
            forgetting_df.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Continual Learning Metrics')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Efficiency metrics
    if any('efficiency_metrics' in result for result in results.values()):
        efficiency_data = []
        exp_names = []
        for exp_name, result in results.items():
            if 'efficiency_metrics' in result:
                efficiency_data.append([
                    result['efficiency_metrics']['learning_speed'],
                    result['efficiency_metrics']['sample_efficiency'],
                    result['efficiency_metrics']['area_under_learning_curve']
                ])
                exp_names.append(exp_name)
        
        if efficiency_data:
            efficiency_df = pd.DataFrame(efficiency_data,
                                       columns=['Learning Speed', 'Sample Efficiency', 'AULC'],
                                       index=exp_names)
            efficiency_df.plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Learning Efficiency Metrics')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    
    plt.show()


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters."""
    required_keys = ['dataset', 'model', 'training']
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return False
    
    # Validate dataset config
    dataset_config = config['dataset']
    if 'name' not in dataset_config:
        logger.error("Dataset name not specified")
        return False
    
    # Validate model config
    model_config = config['model']
    if 'algorithm' not in model_config:
        logger.error("Model algorithm not specified")
        return False
    
    logger.info("Configuration validation passed")
    return True


def get_device_info() -> Dict[str, Any]:
    """Get information about available computing devices."""
    import torch
    
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'device_count': 0,
        'device_name': 'CPU'
    }
    
    if device_info['cuda_available']:
        device_info['device_count'] = torch.cuda.device_count()
        device_info['device_name'] = torch.cuda.get_device_name(0)
    elif device_info['mps_available']:
        device_info['device_name'] = 'Apple Silicon (MPS)'
    
    logger.info(f"Device info: {device_info}")
    return device_info


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_banner(title: str, width: int = 80) -> None:
    """Print a formatted banner."""
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def print_results_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a formatted summary of results."""
    print_banner("EXPERIMENT RESULTS SUMMARY")
    
    for exp_name, result in results.items():
        print(f"\n{exp_name.upper()}:")
        print("-" * 40)
        
        if 'test_metrics' in result:
            metrics = result['test_metrics']
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            if 'classification_report' in metrics:
                report = metrics['classification_report']
                print(f"F1 Score: {report['macro avg']['f1-score']:.4f}")
                print(f"Precision: {report['macro avg']['precision']:.4f}")
                print(f"Recall: {report['macro avg']['recall']:.4f}")
        
        elif 'forgetting_metrics' in result:
            metrics = result['forgetting_metrics']
            print(f"Final Accuracy: {result['final_accuracy']:.4f}")
            print(f"Backward Transfer: {metrics['backward_transfer']:.4f}")
            print(f"Forward Transfer: {metrics['forward_transfer']:.4f}")
            print(f"Average Accuracy: {metrics['average_accuracy']:.4f}")
        
        elif 'efficiency_metrics' in result:
            metrics = result['efficiency_metrics']
            print(f"Final Accuracy: {result['accuracies'][-1] if result['accuracies'] else 0.0:.4f}")
            print(f"Learning Speed: {metrics['learning_speed']:.6f}")
            print(f"Sample Efficiency: {metrics['sample_efficiency']:.0f}")
            print(f"AULC: {metrics['area_under_learning_curve']:.2f}")
    
    print("\n" + "=" * 80)
