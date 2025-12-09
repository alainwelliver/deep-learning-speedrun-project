#!/usr/bin/env python3
"""
merge_experiment_runs.py

Utility script to merge experiment runs from multiple experiment folders.
This is useful when an experiment was interrupted and restarted in a new folder.

Usage:
    python merge_experiment_runs.py --target <target_dir> --source <source_dir>

Example:
    python merge_experiment_runs.py \
        --target experiment_logs/stage_c_nanogpt_palm_parallel_20251209_172659 \
        --source experiment_logs/stage_c_nanogpt_palm_parallel_20251209_183645
"""

import argparse
import json
import shutil
from pathlib import Path
import sys
import numpy as np
from scipy import stats as scipy_stats


def copy_run_directory(source_exp_dir, target_exp_dir, source_run_name, target_run_name):
    """
    Copy a run directory from source to target experiment folder.

    Args:
        source_exp_dir: Path to source experiment directory
        target_exp_dir: Path to target experiment directory
        source_run_name: Name of run directory in source (e.g., "run_0_seed_42")
        target_run_name: Name of run directory in target (e.g., "run_2_seed_42")
    """
    source_run_path = source_exp_dir / "runs" / source_run_name
    target_run_path = target_exp_dir / "runs" / target_run_name

    if not source_run_path.exists():
        raise FileNotFoundError(f"Source run directory not found: {source_run_path}")

    if target_run_path.exists():
        print(f"Warning: Target run directory already exists: {target_run_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            sys.exit(1)
        shutil.rmtree(target_run_path)

    print(f"Copying {source_run_path} -> {target_run_path}")
    shutil.copytree(source_run_path, target_run_path)
    print("✓ Run directory copied")


def merge_results(source_exp_dir, target_exp_dir, new_run_id, keep_seed):
    """
    Merge results.jsonl from source into target.

    Args:
        source_exp_dir: Path to source experiment directory
        target_exp_dir: Path to target experiment directory
        new_run_id: New run_id to assign to the source result
        keep_seed: Whether to keep the original seed or not
    """
    source_results_file = source_exp_dir / "results.jsonl"
    target_results_file = target_exp_dir / "results.jsonl"

    if not source_results_file.exists():
        raise FileNotFoundError(f"Source results.jsonl not found: {source_results_file}")

    if not target_results_file.exists():
        raise FileNotFoundError(f"Target results.jsonl not found: {target_results_file}")

    # Read source results (expecting only one line)
    with open(source_results_file, 'r') as f:
        source_lines = [line.strip() for line in f if line.strip()]

    if len(source_lines) != 1:
        print(f"Warning: Expected 1 result in source, found {len(source_lines)}")

    # Parse and modify the result
    source_result = json.loads(source_lines[0])
    source_result['run_id'] = new_run_id

    print(f"Merging result: run_id={new_run_id}, seed={source_result['seed']}, "
          f"val_loss={source_result['final_val_loss']:.4f}")

    # Append to target results.jsonl
    with open(target_results_file, 'a') as f:
        f.write(json.dumps(source_result) + "\n")

    print("✓ Results merged")


def load_results(results_file):
    """Load all results from results.jsonl"""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_statistics(results):
    """
    Compute statistics from results (same logic as GPTExperimentLogger).

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with statistics
    """
    if not results:
        return {"error": "No results to compute statistics from"}

    # Filter successful runs
    successful_runs = [r for r in results if r.get('success', True)]

    if not successful_runs:
        return {
            "error": "No successful runs to compute statistics from",
            "n_runs": len(results),
            "successful_runs": 0,
            "failed_runs": len(results)
        }

    val_losses = [r['final_val_loss'] for r in successful_runs]
    train_losses = [r['final_train_loss'] for r in successful_runs]
    times = [r['time_seconds'] for r in successful_runs]

    stats = {
        "n_runs": len(results),
        "successful_runs": len(successful_runs),
        "failed_runs": len(results) - len(successful_runs),
        "val_loss": {
            "mean": float(np.mean(val_losses)),
            "std": float(np.std(val_losses, ddof=1)) if len(val_losses) > 1 else 0.0,
            "min": float(np.min(val_losses)),
            "max": float(np.max(val_losses)),
            "median": float(np.median(val_losses)),
        },
        "train_loss": {
            "mean": float(np.mean(train_losses)),
            "std": float(np.std(train_losses, ddof=1)) if len(train_losses) > 1 else 0.0,
            "min": float(np.min(train_losses)),
            "max": float(np.max(train_losses)),
            "median": float(np.median(train_losses)),
        },
        "time": {
            "mean": float(np.mean(times)),
            "std": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "total": float(np.sum(times)),
        }
    }

    # 95% confidence interval for val_loss
    if len(val_losses) > 1:
        ci = scipy_stats.t.interval(
            0.95,
            len(val_losses) - 1,
            loc=stats["val_loss"]["mean"],
            scale=scipy_stats.sem(val_losses)
        )
        stats["val_loss"]["ci_95"] = [float(ci[0]), float(ci[1])]

    return stats


def regenerate_summary(target_exp_dir, source_exp_dir):
    """
    Regenerate summary.json for the target experiment directory.

    Args:
        target_exp_dir: Path to target experiment directory
        source_exp_dir: Path to source experiment directory (for metadata)
    """
    results_file = target_exp_dir / "results.jsonl"
    config_file = target_exp_dir / "config.json"
    source_summary_file = source_exp_dir / "summary.json"
    target_summary_file = target_exp_dir / "summary.json"

    # Load results and compute statistics
    results = load_results(results_file)
    stats = compute_statistics(results)

    # Load config for additional_info
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Load source summary for metadata (git_info, gpu_info, system_info)
    with open(source_summary_file, 'r') as f:
        source_summary = json.load(f)

    # Create new summary
    from datetime import datetime
    summary = {
        "experiment_name": config["experiment_name"],
        "end_time": datetime.now().isoformat(),
        "git_info": source_summary["git_info"],
        "gpu_info": source_summary["gpu_info"],
        "system_info": source_summary["system_info"],
        "statistics": stats,
        "additional_info": {
            "config_path": config_file.relative_to(target_exp_dir.parent.parent).as_posix(),
            "script_name": config["script"],
            "n_gpus": config["n_gpus"],
            "n_runs": stats["n_runs"],
            "successful_runs": stats["successful_runs"],
            "failed_runs": stats["failed_runs"],
            "target_val_loss": config["target_val_loss"]
        }
    }

    # Save summary
    with open(target_summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary saved to {target_summary_file}")
    return summary


def validate_summary(summary, target_exp_dir):
    """Print summary statistics for validation"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    stats = summary["statistics"]
    print(f"Experiment: {summary['experiment_name']}")
    print(f"Directory: {target_exp_dir.name}")
    print(f"\nRuns: {stats['n_runs']} total, {stats['successful_runs']} successful, {stats['failed_runs']} failed")

    if 'val_loss' in stats:
        val_loss = stats['val_loss']
        print(f"\nValidation Loss:")
        print(f"  Mean:   {val_loss['mean']:.4f} ± {val_loss['std']:.4f}")
        print(f"  Median: {val_loss['median']:.4f}")
        print(f"  Range:  [{val_loss['min']:.4f}, {val_loss['max']:.4f}]")
        if 'ci_95' in val_loss:
            print(f"  95% CI: [{val_loss['ci_95'][0]:.4f}, {val_loss['ci_95'][1]:.4f}]")

    if 'train_loss' in stats:
        train_loss = stats['train_loss']
        print(f"\nTrain Loss:")
        print(f"  Mean:   {train_loss['mean']:.4f} ± {train_loss['std']:.4f}")
        print(f"  Median: {train_loss['median']:.4f}")
        print(f"  Range:  [{train_loss['min']:.4f}, {train_loss['max']:.4f}]")

    if 'time' in stats:
        time_stats = stats['time']
        print(f"\nTime:")
        print(f"  Mean:  {time_stats['mean']:.2f}s ({time_stats['mean']/60:.2f} min)")
        print(f"  Total: {time_stats['total']:.2f}s ({time_stats['total']/60:.2f} min)")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Merge experiment runs from multiple folders")
    parser.add_argument("--target", required=True, help="Target experiment directory path")
    parser.add_argument("--source", required=True, help="Source experiment directory path")
    parser.add_argument("--new-run-id", type=int, default=2, help="New run_id for the merged run (default: 2)")
    parser.add_argument("--source-run-name", default="run_0_seed_42", help="Source run directory name (default: run_0_seed_42)")
    parser.add_argument("--target-run-name", default="run_2_seed_42", help="Target run directory name (default: run_2_seed_42)")

    args = parser.parse_args()

    # Convert to Path objects
    target_exp_dir = Path(args.target)
    source_exp_dir = Path(args.source)

    # Validate directories exist
    if not target_exp_dir.exists():
        print(f"Error: Target directory not found: {target_exp_dir}")
        sys.exit(1)

    if not source_exp_dir.exists():
        print(f"Error: Source directory not found: {source_exp_dir}")
        sys.exit(1)

    print("\n" + "="*80)
    print("MERGING EXPERIMENT RUNS")
    print("="*80)
    print(f"Target: {target_exp_dir}")
    print(f"Source: {source_exp_dir}")
    print("="*80 + "\n")

    # Step 1: Copy run directory
    print("Step 1: Copying run directory...")
    copy_run_directory(source_exp_dir, target_exp_dir, args.source_run_name, args.target_run_name)

    # Step 2: Merge results
    print("\nStep 2: Merging results.jsonl...")
    merge_results(source_exp_dir, target_exp_dir, args.new_run_id, keep_seed=True)

    # Step 3: Regenerate summary
    print("\nStep 3: Regenerating summary.json...")
    summary = regenerate_summary(target_exp_dir, source_exp_dir)

    # Step 4: Validate
    print("\nStep 4: Validating results...")
    validate_summary(summary, target_exp_dir)

    print("✓ Merge complete!")


if __name__ == "__main__":
    main()
