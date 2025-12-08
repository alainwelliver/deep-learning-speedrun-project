#!/usr/bin/env python3
"""
run_nanogpt_experiment.py

Experiment runner for NanoGPT speedrun experiments.
Wraps torchrun, parses stdout, runs multiple trials with different seeds, and collects statistics.

Usage:
    # Run baseline with default settings (8 GPUs, 5 runs)
    python scripts/run_nanogpt_experiment.py --config configs/baseline.json --n_runs 5

    # Debug with single GPU
    python scripts/run_nanogpt_experiment.py --config configs/baseline.json --n_runs 1 --n_gpus 1

    # Dry run (print command without executing)
    python scripts/run_nanogpt_experiment.py --config configs/baseline.json --dry_run
"""

import sys
import os
import argparse
import json
import subprocess
import re
import time
from pathlib import Path

# Add parent directory to path to import experiment_logger
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiment_logger import GPTExperimentLogger


def load_config(config_path):
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def validate_script_exists(script_name, nanogpt_dir):
    """Validate that the training script exists."""
    script_path = nanogpt_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")
    return script_path


def parse_training_line(line):
    """
    Parse training loss output line.
    Format: step:X/5100 train_loss:Y.YYYY train_time:Zms step_avg:W.WWms
    """
    pattern = r'step:(\d+)/\d+ train_loss:([\d.]+) train_time:([\d.]+)ms step_avg:([\d.]+)ms'
    match = re.search(pattern, line)
    if match:
        return {
            'type': 'train',
            'step': int(match.group(1)),
            'train_loss': float(match.group(2)),
            'train_time_ms': float(match.group(3)),
            'step_avg_ms': float(match.group(4))
        }
    return None


def parse_validation_line(line):
    """
    Parse validation loss output line.
    Format: step:X/5100 val_loss:Y.YYYY train_time:Zms step_avg:W.WWms
    """
    pattern = r'step:(\d+)/\d+ val_loss:([\d.]+) train_time:([\d.]+)ms step_avg:([\d.]+)ms'
    match = re.search(pattern, line)
    if match:
        return {
            'type': 'val',
            'step': int(match.group(1)),
            'val_loss': float(match.group(2)),
            'train_time_ms': float(match.group(3)),
            'step_avg_ms': float(match.group(4))
        }
    return None


def parse_memory_line(line):
    """
    Parse peak memory output line.
    Format: peak memory consumption: X MiB
    """
    pattern = r'peak memory consumption: (\d+) MiB'
    match = re.search(pattern, line)
    if match:
        return int(match.group(1))
    return None


def run_single_trial(config, run_id, seed, n_gpus, logger, nanogpt_dir, dry_run=False):
    """
    Run a single training trial with torchrun.

    Args:
        config: Experiment configuration dict
        run_id: Identifier for this run
        seed: Random seed to use
        n_gpus: Number of GPUs to use
        logger: GPTExperimentLogger instance
        nanogpt_dir: Path to nanogpt directory
        dry_run: If True, just print command without running

    Returns:
        dict: Results containing final_val_loss, final_train_loss, time, etc.
    """
    script_name = config['script']

    # Validate script exists
    try:
        script_path = validate_script_exists(script_name, nanogpt_dir)
    except FileNotFoundError as e:
        logger.log(f"ERROR: {e}", level="ERROR")
        return {
            'run_id': run_id,
            'seed': seed,
            'success': False,
            'error': str(e)
        }

    logger.log(f"Starting run {run_id} with seed {seed}")

    # Build torchrun command
    cmd = [
        'torchrun',
        '--standalone',
        f'--nproc_per_node={n_gpus}',
        script_name
    ]

    # Set up environment with seed
    env = os.environ.copy()
    env['RUN_SEED'] = str(seed)
    env['PYTHONHASHSEED'] = str(seed)

    if dry_run:
        logger.log(f"DRY RUN - Would execute: {' '.join(cmd)}")
        logger.log(f"DRY RUN - With environment: RUN_SEED={seed}")
        return {
            'run_id': run_id,
            'seed': seed,
            'success': True,
            'dry_run': True
        }

    # Create run-specific output directory
    run_dir = logger.exp_dir / "runs" / f"run_{run_id}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_log_path = run_dir / "stdout.log"

    logger.log(f"Running: {' '.join(cmd)}")
    logger.log(f"Working directory: {nanogpt_dir}")
    logger.log(f"Stdout log: {stdout_log_path}")

    # Track metrics
    last_train_loss = None
    last_val_loss = None
    last_step = 0
    peak_memory = None

    start_time = time.time()

    try:
        # Run subprocess and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(nanogpt_dir),
            env=env
        )

        # Open log file for writing
        with open(stdout_log_path, 'w') as log_file:
            # Read output line by line in real-time
            for line in process.stdout:
                # Write to log file
                log_file.write(line)
                log_file.flush()

                # Also print to console (optional - can be noisy)
                # print(line, end='')

                # Parse for metrics
                train_metrics = parse_training_line(line)
                if train_metrics:
                    last_train_loss = train_metrics['train_loss']
                    last_step = train_metrics['step']
                    logger.log_step_metrics(
                        run_id=run_id,
                        step=train_metrics['step'],
                        train_loss=train_metrics['train_loss'],
                        train_time_ms=train_metrics['train_time_ms'],
                        step_avg_ms=train_metrics['step_avg_ms']
                    )

                val_metrics = parse_validation_line(line)
                if val_metrics:
                    last_val_loss = val_metrics['val_loss']
                    last_step = val_metrics['step']
                    logger.log_step_metrics(
                        run_id=run_id,
                        step=val_metrics['step'],
                        val_loss=val_metrics['val_loss'],
                        train_time_ms=val_metrics['train_time_ms'],
                        step_avg_ms=val_metrics['step_avg_ms']
                    )
                    # Log validation losses to console
                    logger.log(f"Run {run_id} - Step {val_metrics['step']}: val_loss={val_metrics['val_loss']:.4f}")

                memory = parse_memory_line(line)
                if memory:
                    peak_memory = memory

        # Wait for process to complete
        returncode = process.wait()

        end_time = time.time()
        elapsed_time = end_time - start_time

        if returncode != 0:
            logger.log(f"ERROR in run {run_id}: Process exited with code {returncode}", level="ERROR")
            return {
                'run_id': run_id,
                'seed': seed,
                'success': False,
                'error': f"Process exited with code {returncode}",
                'time_seconds': elapsed_time
            }

        # Check that we got metrics
        if last_val_loss is None:
            logger.log(f"WARNING: No validation loss captured for run {run_id}", level="WARNING")
            return {
                'run_id': run_id,
                'seed': seed,
                'success': False,
                'error': "No validation loss captured",
                'time_seconds': elapsed_time
            }

        result = {
            'run_id': run_id,
            'seed': seed,
            'final_val_loss': last_val_loss,
            'final_train_loss': last_train_loss if last_train_loss else 0.0,
            'time_seconds': elapsed_time,
            'final_step': last_step,
            'success': True
        }

        if peak_memory:
            result['peak_memory_mib'] = peak_memory

        logger.log_run_result(**result)

        return result

    except KeyboardInterrupt:
        logger.log(f"Run {run_id} interrupted by user", level="WARNING")
        if process:
            process.terminate()
            process.wait(timeout=10)
        return {
            'run_id': run_id,
            'seed': seed,
            'success': False,
            'error': 'Interrupted by user',
            'time_seconds': time.time() - start_time
        }

    except Exception as e:
        logger.log(f"ERROR in run {run_id}: {str(e)}", level="ERROR")
        import traceback
        logger.log(traceback.format_exc(), level="ERROR")

        return {
            'run_id': run_id,
            'seed': seed,
            'success': False,
            'error': str(e),
            'time_seconds': time.time() - start_time
        }


def run_experiment(config_path, n_runs, n_gpus_override=None, output_dir=None, dry_run=False):
    """
    Run a complete experiment with multiple trials.

    Args:
        config_path: Path to JSON configuration file
        n_runs: Number of runs to perform
        n_gpus_override: Optional override for number of GPUs (for debugging)
        output_dir: Optional output directory (defaults to experiment_logs/)
        dry_run: If True, print commands without executing

    Returns:
        tuple: (experiment_dir, summary_dict)
    """
    # Load configuration
    config = load_config(config_path)

    experiment_name = config.get('experiment_name', 'unnamed_experiment')
    base_seed = config.get('base_seed', 42)
    n_gpus = n_gpus_override if n_gpus_override is not None else config.get('n_gpus', 8)

    # Get path to nanogpt directory
    script_dir = Path(__file__).parent
    nanogpt_dir = script_dir.parent  # Go up from scripts/ to nanogpt/

    # Initialize logger
    log_dir = output_dir if output_dir else "experiment_logs"
    logger = GPTExperimentLogger(experiment_name, log_dir=log_dir)

    # Log experiment details
    logger.log(f"Starting experiment: {experiment_name}")
    logger.log(f"Configuration file: {config_path}")
    logger.log(f"Training script: {config.get('script', 'N/A')}")
    logger.log(f"Number of GPUs: {n_gpus}")
    if n_gpus_override:
        logger.log(f"WARNING: GPU count overridden from {config.get('n_gpus')} to {n_gpus}", level="WARNING")
    logger.log(f"Number of runs: {n_runs}")
    logger.log(f"Base seed: {base_seed}")

    if 'description' in config:
        logger.log(f"Description: {config['description']}")
    if 'target_val_loss' in config:
        logger.log(f"Target validation loss: {config['target_val_loss']}")

    # Log full configuration
    logger.log_config(config)

    if dry_run:
        logger.log("="*80)
        logger.log("DRY RUN MODE - No actual training will be performed")
        logger.log("="*80)

    # Run multiple trials
    logger.log("="*80)
    logger.log(f"Starting {n_runs} training runs...")
    logger.log("="*80)

    results = []
    successful_runs = 0

    try:
        for i in range(n_runs):
            seed = base_seed + i

            result = run_single_trial(config, i, seed, n_gpus, logger, nanogpt_dir, dry_run)
            results.append(result)

            if result.get('success', False):
                successful_runs += 1

            # Log progress
            logger.log(f"Progress: {i + 1}/{n_runs} runs completed ({successful_runs} successful)")
            logger.log("-"*80)

    except KeyboardInterrupt:
        logger.log("="*80, level="WARNING")
        logger.log("Experiment interrupted by user", level="WARNING")
        logger.log("Saving partial results...", level="WARNING")
        logger.log("="*80, level="WARNING")

    logger.log("="*80)
    logger.log(f"All runs completed: {successful_runs}/{n_runs} successful")
    logger.log("="*80)

    # Compute and log statistics
    logger.log("Computing statistics...")

    # Finalize experiment
    additional_info = {
        'config_path': str(config_path),
        'script_name': config.get('script', 'N/A'),
        'n_gpus': n_gpus,
        'n_runs': n_runs,
        'successful_runs': successful_runs,
        'failed_runs': n_runs - successful_runs,
    }

    if 'target_val_loss' in config:
        additional_info['target_val_loss'] = config['target_val_loss']

    exp_dir, summary = logger.finalize(additional_info)

    # Check if target was achieved
    if 'target_val_loss' in config and 'val_loss' in summary['statistics']:
        target = config['target_val_loss']
        mean_val_loss = summary['statistics']['val_loss']['mean']

        if mean_val_loss <= target:
            logger.log(f"✓ Target validation loss ACHIEVED: {mean_val_loss:.4f} <= {target:.4f}", level="INFO")
        else:
            logger.log(f"✗ Target validation loss NOT achieved: {mean_val_loss:.4f} > {target:.4f}", level="WARNING")

    return exp_dir, summary


def main():
    parser = argparse.ArgumentParser(
        description='Run NanoGPT speedrun experiments with logging and statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline with 5 runs on 8 GPUs
  python scripts/run_nanogpt_experiment.py --config configs/baseline.json --n_runs 5

  # Debug with 1 run on 1 GPU
  python scripts/run_nanogpt_experiment.py --config configs/baseline.json --n_runs 1 --n_gpus 1

  # Dry run (print commands without executing)
  python scripts/run_nanogpt_experiment.py --config configs/baseline.json --dry_run
        """
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
        default=5,
        help='Number of training runs to perform (default: 5)'
    )
    parser.add_argument(
        '--n_gpus',
        type=int,
        default=None,
        help='Override number of GPUs (for debugging, default: use value from config)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for logs (default: experiment_logs/)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print commands without executing (for testing)'
    )

    args = parser.parse_args()

    # Validate config file exists
    if not Path(args.config).exists():
        print(f"ERROR: Configuration file not found: {args.config}")
        sys.exit(1)

    # Run experiment
    print(f"\n{'='*80}")
    print(f"NanoGPT Speedrun Experiment Runner")
    print(f"{'='*80}\n")

    try:
        exp_dir, summary = run_experiment(
            args.config,
            args.n_runs,
            n_gpus_override=args.n_gpus,
            output_dir=args.output_dir,
            dry_run=args.dry_run
        )

        print(f"\n{'='*80}")
        print(f"Experiment completed successfully!")
        print(f"Results directory: {exp_dir}")
        print(f"{'='*80}\n")

        # Print summary statistics
        if 'statistics' in summary and 'val_loss' in summary['statistics']:
            stats = summary['statistics']
            print("Summary Statistics:")
            print(f"  Runs: {stats['successful_runs']}/{stats['n_runs']}")
            print(f"  Mean Validation Loss: {stats['val_loss']['mean']:.4f} ± {stats['val_loss']['std']:.4f}")
            if 'ci_95' in stats['val_loss']:
                ci = stats['val_loss']['ci_95']
                print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
            print(f"  Best Validation Loss: {stats['val_loss']['min']:.4f}")
            print()

    except Exception as e:
        print(f"\nERROR: Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
