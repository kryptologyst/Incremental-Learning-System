"""Main training script for incremental learning experiments."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models import IncrementalLearningSystem, SGDIncrementalLearner, OnlineNaiveBayes
from src.data import DataLoader, IncrementalDataGenerator
from src.metrics import IncrementalLearningMetrics, ContinualLearningEvaluator, plot_learning_curves

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """Create necessary directories."""
    directories = ['logs', 'checkpoints', 'assets', 'assets/plots', 'data']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def run_baseline_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run baseline incremental learning experiment."""
    logger.info("Starting baseline incremental learning experiment")
    
    # Load data
    data_loader = DataLoader(
        dataset_name=config['dataset']['name'],
        test_size=config['dataset']['test_size'],
        random_state=config['dataset']['random_state']
    )
    
    X_train, X_test, y_train, y_test = data_loader.load_dataset()
    
    # Create streaming batches
    batches = data_loader.create_streaming_data(
        X_train, y_train, 
        batch_size=config['dataset']['batch_size']
    )
    
    # Initialize model
    model = IncrementalLearningSystem(
        algorithm=config['model']['algorithm'],
        loss=config['model']['loss'],
        max_iter=config['model']['max_iter'],
        random_state=config['model']['random_state']
    )
    
    # Train incrementally
    accuracies = model.train_incremental(
        X_train, y_train,
        batch_size=config['dataset']['batch_size'],
        classes=np.unique(y_train)
    )
    
    # Evaluate on test set
    test_metrics = model.evaluate(X_test, y_test)
    
    logger.info(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
    
    return {
        'accuracies': accuracies,
        'test_metrics': test_metrics,
        'model': model
    }


def run_continual_learning_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run continual learning experiment with multiple tasks."""
    logger.info("Starting continual learning experiment")
    
    # Generate sequential tasks
    data_generator = IncrementalDataGenerator(
        n_features=20,
        n_classes=3,
        random_state=config['dataset']['random_state']
    )
    
    tasks = data_generator.generate_sequential_tasks(
        n_tasks=config['advanced']['continual_learning']['n_tasks'],
        samples_per_task=config['advanced']['continual_learning']['samples_per_task']
    )
    
    # Create test tasks (same distribution as training)
    test_tasks = []
    for X_train, y_train in tasks:
        # Use last 20% of each task as test data
        test_size = int(0.2 * len(X_train))
        X_test = X_train[-test_size:]
        y_test = y_train[-test_size:]
        test_tasks.append((X_test, y_test))
    
    # Initialize model and evaluator
    model = IncrementalLearningSystem(algorithm=config['model']['algorithm'])
    evaluator = ContinualLearningEvaluator()
    
    # Evaluate on task sequence
    results = evaluator.evaluate_task_sequence(model, tasks, test_tasks)
    
    logger.info(f"Continual learning results: {results['forgetting_metrics']}")
    
    return results


def run_online_learning_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run online learning experiment with concept drift."""
    logger.info("Starting online learning experiment")
    
    # Generate non-stationary data
    data_generator = IncrementalDataGenerator(random_state=config['dataset']['random_state'])
    X_stream, y_stream = data_generator.generate_non_stationary_data(
        n_samples=1000,
        drift_frequency=200
    )
    
    # Initialize model and evaluator
    model = IncrementalLearningSystem(algorithm=config['model']['algorithm'])
    evaluator = ContinualLearningEvaluator()
    
    # Evaluate online learning
    results = evaluator.evaluate_online_learning(
        model, X_stream, y_stream,
        batch_size=config['dataset']['batch_size'],
        evaluation_frequency=config['training']['evaluation_frequency']
    )
    
    logger.info(f"Online learning results: {results['efficiency_metrics']}")
    
    return results


def create_leaderboard(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create a leaderboard from experiment results."""
    leaderboard_data = []
    
    for experiment_name, result in results.items():
        if 'test_metrics' in result:
            # Baseline experiment
            leaderboard_data.append({
                'Experiment': experiment_name,
                'Algorithm': 'SGD',
                'Final Accuracy': result['test_metrics']['accuracy'],
                'F1 Score': result['test_metrics']['classification_report']['macro avg']['f1-score'],
                'Precision': result['test_metrics']['classification_report']['macro avg']['precision'],
                'Recall': result['test_metrics']['classification_report']['macro avg']['recall']
            })
        elif 'forgetting_metrics' in result:
            # Continual learning experiment
            leaderboard_data.append({
                'Experiment': experiment_name,
                'Algorithm': 'SGD',
                'Final Accuracy': result['final_accuracy'],
                'Backward Transfer': result['forgetting_metrics']['backward_transfer'],
                'Forward Transfer': result['forgetting_metrics']['forward_transfer'],
                'Average Accuracy': result['forgetting_metrics']['average_accuracy']
            })
        elif 'efficiency_metrics' in result:
            # Online learning experiment
            leaderboard_data.append({
                'Experiment': experiment_name,
                'Algorithm': 'SGD',
                'Final Accuracy': result['accuracies'][-1] if result['accuracies'] else 0.0,
                'Learning Speed': result['efficiency_metrics']['learning_speed'],
                'Sample Efficiency': result['efficiency_metrics']['sample_efficiency'],
                'AULC': result['efficiency_metrics']['area_under_learning_curve']
            })
    
    return pd.DataFrame(leaderboard_data)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Incremental Learning System Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['baseline', 'continual', 'online', 'all'],
                       help='Type of experiment to run')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    config = load_config(args.config)
    
    # Set random seeds
    np.random.seed(config['dataset']['random_state'])
    
    logger.info(f"Starting experiments with config: {args.config}")
    logger.info(f"Experiment type: {args.experiment}")
    
    results = {}
    
    # Run experiments based on selection
    if args.experiment in ['baseline', 'all']:
        results['baseline'] = run_baseline_experiment(config)
    
    if args.experiment in ['continual', 'all']:
        results['continual_learning'] = run_continual_learning_experiment(config)
    
    if args.experiment in ['online', 'all']:
        results['online_learning'] = run_online_learning_experiment(config)
    
    # Create leaderboard
    leaderboard = create_leaderboard(results)
    leaderboard.to_csv(f'{args.output_dir}/leaderboard.csv', index=False)
    logger.info("Leaderboard saved to results/leaderboard.csv")
    
    # Save results
    import json
    with open(f'{args.output_dir}/results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for exp_name, exp_result in results.items():
            json_results[exp_name] = {}
            for key, value in exp_result.items():
                if isinstance(value, np.ndarray):
                    json_results[exp_name][key] = value.tolist()
                elif isinstance(value, dict) and 'classification_report' in key:
                    # Handle sklearn classification report
                    json_results[exp_name][key] = value
                else:
                    json_results[exp_name][key] = value
        
        json.dump(json_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {args.output_dir}/results.json")
    
    # Print leaderboard
    print("\n" + "="*80)
    print("INCREMENTAL LEARNING SYSTEM - EXPERIMENT RESULTS")
    print("="*80)
    print(leaderboard.to_string(index=False))
    print("="*80)
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
