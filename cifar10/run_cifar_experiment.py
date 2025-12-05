#!/usr/bin/env python3
"""
run_cifar_experiment.py

Experiment runner for CIFAR-10 speedrun experiments.
Runs multiple trials with different seeds and collects statistics.

Usage:
    # Run baseline
    python run_cifar_experiment.py --config configs/baseline.json --n_runs 100
    
    # Run modification with different module
    python run_cifar_experiment.py --config configs/deterministic_translate.json --module airbench94_deterministic_translate --n_runs 100
"""

import sys
import argparse
import json
import importlib
import torch
import time
import numpy as np
from pathlib import Path
from experiment_logger import ExperimentLogger

# Add experiments directory to Python path so modules can be imported
sys.path.insert(0, str(Path(__file__).parent / 'experiments'))


def load_config(config_path):
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_airbench_module(module_name):
    """
    Dynamically import an airbench module by name.
    
    Args:
        module_name: Name of the module (e.g., 'airbench94' or 'airbench94_deterministic_translate')
    
    Returns:
        The imported module
    """
    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError as e:
        print(f"ERROR: Could not import {module_name}: {e}")
        print(f"Make sure {module_name}.py is in the experiments/ directory.")
        sys.exit(1)


def modify_airbench_hyperparameters(airbench_module, config):
    """
    Modify the hyperparameters in airbench module based on config.
    """
    # Update hyperparameters if specified in config
    if 'batch_size' in config:
        airbench_module.hyp['opt']['batch_size'] = config['batch_size']
    if 'learning_rate' in config:
        airbench_module.hyp['opt']['lr'] = config['learning_rate']
    if 'epochs' in config:
        airbench_module.hyp['opt']['train_epochs'] = config['epochs']
    if 'momentum' in config:
        airbench_module.hyp['opt']['momentum'] = config['momentum']
    if 'weight_decay' in config:
        airbench_module.hyp['opt']['weight_decay'] = config['weight_decay']
    
    # NEW: Support for augmentation config
    if 'aug' in config:
        for key, value in config['aug'].items():
            airbench_module.hyp['aug'][key] = value
    
    # NEW: Support for deterministic_translate flag
    if 'deterministic_translate' in config:
        airbench_module.hyp['aug']['deterministic_translate'] = config['deterministic_translate']
    
    return airbench_module


def run_single_trial(airbench_module, run_id, seed, logger):
    """
    Run a single training trial.
    
    Args:
        airbench_module: The imported airbench module
        run_id: Identifier for this run
        seed: Random seed to use
        logger: ExperimentLogger instance
    
    Returns:
        dict: Results containing accuracy and time
    """
    logger.log(f"Starting run {run_id} with seed {seed}")
    
    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Time the training
    start_time = time.time()
    
    try:
        # Run the main training function
        accuracy = airbench_module.main(run=run_id)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        result = {
            'run_id': run_id,
            'seed': seed,
            'accuracy': float(accuracy),
            'time_seconds': elapsed_time,
            'success': True
        }
        
        logger.log_run_result(**result)
        
        return result
        
    except Exception as e:
        logger.log(f"ERROR in run {run_id}: {str(e)}", level="ERROR")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return {
            'run_id': run_id,
            'seed': seed,
            'accuracy': 0.0,
            'time_seconds': elapsed_time,
            'success': False,
            'error': str(e)
        }


def run_experiment(config_path, n_runs, module_name='airbench94', output_dir=None):
    """
    Run a complete experiment with multiple trials.
    
    Args:
        config_path: Path to JSON configuration file
        n_runs: Number of runs to perform
        module_name: Name of the airbench module to use
        output_dir: Optional output directory (defaults to experiment_logs/)
    
    Returns:
        tuple: (experiment_dir, summary_dict)
    """
    # Load configuration
    config = load_config(config_path)
    
    experiment_name = config.get('experiment_name', 'unnamed_experiment')
    base_seed = config.get('base_seed', 42)
    
    # Initialize logger
    log_dir = output_dir if output_dir else "experiment_logs"
    logger = ExperimentLogger(experiment_name, log_dir=log_dir)
    
    # Log experiment details
    logger.log(f"Starting experiment: {experiment_name}")
    logger.log(f"Configuration file: {config_path}")
    logger.log(f"Module: {module_name}")
    logger.log(f"Number of runs: {n_runs}")
    logger.log(f"Base seed: {base_seed}")
    
    if 'hypothesis' in config:
        logger.log(f"Hypothesis: {config['hypothesis']}")
    if 'modification' in config:
        logger.log(f"Modification: {config['modification']}")
    
    # Log all hyperparameters
    logger.log_hyperparameters(config)
    
    # Load and configure airbench module
    logger.log(f"Loading and configuring {module_name}...")
    airbench_module = load_airbench_module(module_name)
    airbench_module = modify_airbench_hyperparameters(airbench_module, config)
    
    # Log the actual hyp dict being used
    logger.log(f"Effective augmentation config: {airbench_module.hyp['aug']}")
    
    # Run warmup if this is the first experiment
    logger.log("Running warmup...")
    try:
        airbench_module.main('warmup')
        logger.log("Warmup complete")
    except Exception as e:
        logger.log(f"Warning: Warmup failed: {e}", level="WARNING")
    
    # Run multiple trials
    logger.log("="*80)
    logger.log(f"Starting {n_runs} training runs...")
    logger.log("="*80)
    
    results = []
    successful_runs = 0
    
    for i in range(n_runs):
        seed = base_seed + i
        
        result = run_single_trial(airbench_module, i, seed, logger)
        results.append(result)
        
        if result['success']:
            successful_runs += 1
        
        # Log progress
        if (i + 1) % 5 == 0:
            logger.log(f"Progress: {i + 1}/{n_runs} runs completed ({successful_runs} successful)")
    
    logger.log("="*80)
    logger.log(f"All runs completed: {successful_runs}/{n_runs} successful")
    logger.log("="*80)
    
    # Compute and log statistics
    logger.log("Computing statistics...")
    
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        accuracies = [r['accuracy'] for r in successful_results]
        times = [r['time_seconds'] for r in successful_results]
        
        # Compute statistics
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)
        median_acc = np.median(accuracies)
        
        mean_time = np.mean(times)
        std_time = np.std(times, ddof=1)
        total_time = np.sum(times)
        
        # Confidence interval
        from scipy import stats
        ci = stats.t.interval(
            0.95,
            len(accuracies) - 1,
            loc=mean_acc,
            scale=stats.sem(accuracies)
        )
        
        # Log statistics
        logger.log(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f} (std)")
        logger.log(f"Accuracy 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        logger.log(f"Accuracy range: [{min_acc:.4f}, {max_acc:.4f}]")
        logger.log(f"Median accuracy: {median_acc:.4f}")
        logger.log(f"Mean time per run: {mean_time:.2f}s ± {std_time:.2f}s")
        logger.log(f"Total training time: {total_time:.2f}s ({total_time/3600:.2f} hours)")
        
        # Check if hypothesis is supported (if target accuracy specified)
        if 'target_accuracy' in config:
            target = config['target_accuracy']
            logger.log(f"Target accuracy: {target:.4f}")
            
            if mean_acc >= target:
                logger.log(f"✓ Target accuracy ACHIEVED: {mean_acc:.4f} >= {target:.4f}", level="INFO")
            else:
                logger.log(f"✗ Target accuracy NOT achieved: {mean_acc:.4f} < {target:.4f}", level="WARNING")
        
        # Compare to baseline if specified
        if 'baseline_accuracy' in config:
            baseline = config['baseline_accuracy']
            improvement = ((mean_acc - baseline) / baseline) * 100
            logger.log(f"Baseline accuracy: {baseline:.4f}")
            logger.log(f"Improvement: {improvement:+.2f}%")
            
            # Statistical significance test
            if 'baseline_std' in config:
                baseline_std = config['baseline_std']
                baseline_n = config.get('baseline_n', n_runs)
                
                # Two-sample t-test (Welch's t-test)
                t_stat, p_value = stats.ttest_ind_from_stats(
                    mean_acc, std_acc, len(accuracies),
                    baseline, baseline_std, baseline_n,
                    equal_var=False
                )
                logger.log(f"Statistical significance: t={t_stat:.3f}, p={p_value:.4f}")
                
                if p_value < 0.05:
                    logger.log("Result is statistically significant (p < 0.05)")
                else:
                    logger.log("Result is NOT statistically significant (p >= 0.05)")
    
    else:
        logger.log("ERROR: No successful runs to compute statistics from", level="ERROR")
    
    # Finalize experiment
    additional_info = {
        'config_path': str(config_path),
        'module_name': module_name,
        'n_runs': n_runs,
        'successful_runs': successful_runs,
        'failed_runs': n_runs - successful_runs,
    }
    
    exp_dir, summary = logger.finalize(additional_info)
    
    return exp_dir, summary


def main():
    parser = argparse.ArgumentParser(
        description='Run CIFAR-10 speedrun experiments with logging and statistics'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration JSON file'
    )
    parser.add_argument(
        '--n_runs',
        type=int,
        default=20,
        help='Number of training runs to perform (default: 20)'
    )
    parser.add_argument(
        '--module',
        type=str,
        default='airbench94',
        help='Name of airbench module to use (default: airbench94)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for logs (default: experiment_logs/)'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not Path(args.config).exists():
        print(f"ERROR: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Run experiment
    print(f"\n{'='*80}")
    print(f"CIFAR-10 Speedrun Experiment Runner")
    print(f"{'='*80}\n")
    
    try:
        exp_dir, summary = run_experiment(
            args.config, 
            args.n_runs, 
            module_name=args.module,
            output_dir=args.output_dir
        )
        
        print(f"\n{'='*80}")
        print(f"Experiment completed successfully!")
        print(f"Results directory: {exp_dir}")
        print(f"{'='*80}\n")
        
        # Print summary statistics
        if 'statistics' in summary and 'accuracy' in summary['statistics']:
            stats = summary['statistics']
            print("Summary Statistics:")
            print(f"  Runs: {stats['n_runs']}")
            print(f"  Mean Accuracy: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")
            if 'ci_95' in stats['accuracy']:
                ci = stats['accuracy']['ci_95']
                print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
            print()
        
    except Exception as e:
        print(f"\nERROR: Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()