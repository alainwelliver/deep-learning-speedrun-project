#!/usr/bin/env python3
"""
Stage C Experiment Analysis: Baseline vs PaLM-Parallel Architecture

This script performs comprehensive statistical analysis comparing the
PaLM-parallel architecture modification to baseline for NanoGPT Stage C
experiments.

Outputs:
- Descriptive statistics for both experiments
- Statistical comparisons with tests appropriate for small samples (n=3)
- Visualizations (distributions, training curves, comparisons)
- Export files (CSV, JSON)
- Comprehensive markdown report
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# ============================================================================
# SECTION 1: Setup and Configuration
# ============================================================================

# Experiment paths
BASELINE_PATH = Path("nanogpt/experiment_logs/stage_c_nanogpt_baseline_20251209_160731")
PALM_PARALLEL_PATH = Path("nanogpt/experiment_logs/stage_c_nanogpt_palm_parallel_20251209_172659")

# Output directories
FIGURES_DIR = Path("nanogpt/analysis/figures")
RESULTS_DIR = Path("nanogpt/analysis/results")

# Create output directories
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STAGE C EXPERIMENT ANALYSIS")
print("Baseline vs PaLM-Parallel Architecture")
print("=" * 80)
print()

# ============================================================================
# SECTION 2: Data Loading Functions
# ============================================================================

def load_experiment_data(exp_path: Path) -> Dict:
    """Load all data from an experiment directory."""
    exp_path = Path(exp_path)

    # Load summary
    with open(exp_path / "summary.json", 'r') as f:
        summary = json.load(f)

    # Load config
    with open(exp_path / "config.json", 'r') as f:
        config = json.load(f)

    # Load results
    results = []
    with open(exp_path / "results.jsonl", 'r') as f:
        for line in f:
            results.append(json.loads(line))

    return {
        'summary': summary,
        'config': config,
        'results': results,
        'path': exp_path
    }

def load_training_curves(exp_path: Path) -> Dict:
    """
    Stream parse metrics.jsonl to extract training curves.

    Returns a dict with structure:
    {
        run_id: {
            'train_steps': np.array,
            'train_losses': np.array,
            'val_steps': np.array,
            'val_losses': np.array,
            'step_times_ms': np.array,
        }
    }
    """
    exp_path = Path(exp_path)
    curves = defaultdict(lambda: {
        'train_steps': [],
        'train_losses': [],
        'val_steps': [],
        'val_losses': [],
        'step_times_ms': [],
    })

    metrics_file = exp_path / "metrics.jsonl"
    print(f"  Loading training curves from {metrics_file.name}...")

    with open(metrics_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            run_id = data['run_id']

            # Training loss (recorded every step)
            if 'train_loss' in data:
                curves[run_id]['train_steps'].append(data['step'])
                curves[run_id]['train_losses'].append(data['train_loss'])
                if 'step_avg_ms' in data:
                    curves[run_id]['step_times_ms'].append(data['step_avg_ms'])

            # Validation loss (recorded every 125 steps)
            if 'val_loss' in data:
                curves[run_id]['val_steps'].append(data['step'])
                curves[run_id]['val_losses'].append(data['val_loss'])

    # Convert lists to numpy arrays
    result = {}
    for run_id, data in curves.items():
        result[run_id] = {k: np.array(v) for k, v in data.items()}

    print(f"  ✓ Loaded curves for {len(result)} runs")
    return result

# ============================================================================
# SECTION 3: Statistical Functions
# ============================================================================

def compute_descriptive_stats(values: np.ndarray, confidence_levels=[0.95, 0.99]) -> Dict:
    """Compute comprehensive descriptive statistics."""
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))

    stats_dict = {
        "n": n,
        "mean": mean,
        "std": std,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
    }

    # Coefficient of variation
    if mean != 0:
        stats_dict["cv"] = abs(std / mean)
    else:
        stats_dict["cv"] = 0.0

    # Confidence intervals using t-distribution
    for conf_level in confidence_levels:
        if n > 1:
            sem = std / np.sqrt(n)
            h = sem * scipy_stats.t.ppf((1 + conf_level) / 2, n - 1)
            ci_key = f"ci_{int(conf_level * 100)}"
            stats_dict[ci_key] = [mean - h, mean + h]
        else:
            ci_key = f"ci_{int(conf_level * 100)}"
            stats_dict[ci_key] = [mean, mean]

    # Percentiles
    stats_dict["percentiles"] = {
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }

    return stats_dict

def bootstrap_confidence_interval(values: np.ndarray, n_bootstrap=10000, confidence=0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    np.random.seed(42)  # For reproducibility
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return lower, upper

def permutation_test(group1: np.ndarray, group2: np.ndarray, n_permutations=10000) -> float:
    """
    Perform permutation test for difference in means.
    Returns p-value.
    """
    np.random.seed(42)  # For reproducibility

    observed_diff = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = count / n_permutations
    return p_value

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d

def compare_experiments(baseline_values: np.ndarray, modified_values: np.ndarray,
                       metric_name: str, lower_is_better: bool = False) -> Dict:
    """
    Perform comprehensive statistical comparison between two experiments.

    Args:
        baseline_values: Array of baseline measurements
        modified_values: Array of modified measurements
        metric_name: Name of the metric being compared
        lower_is_better: If True, lower values are better (e.g., loss)

    Returns:
        Dict with comparison results
    """
    # Basic statistics
    baseline_mean = np.mean(baseline_values)
    modified_mean = np.mean(modified_values)

    absolute_diff = modified_mean - baseline_mean
    if baseline_mean != 0:
        percent_diff = (absolute_diff / baseline_mean) * 100
    else:
        percent_diff = 0.0

    # Determine improvement direction
    if lower_is_better:
        improvement = -percent_diff  # Negative change is improvement
    else:
        improvement = percent_diff  # Positive change is improvement

    # Welch's t-test (doesn't assume equal variances)
    t_stat, p_value_ttest = scipy_stats.ttest_ind(baseline_values, modified_values, equal_var=False)

    # Permutation test (better for small samples)
    p_value_perm = permutation_test(baseline_values, modified_values)

    # Effect size
    effect_size = cohens_d(baseline_values, modified_values)

    # Bootstrap CI for difference
    combined = np.concatenate([baseline_values, modified_values])
    labels = np.array([0]*len(baseline_values) + [1]*len(modified_values))

    np.random.seed(42)
    bootstrap_diffs = []
    for _ in range(10000):
        idx = np.random.choice(len(combined), size=len(combined), replace=True)
        boot_sample = combined[idx]
        boot_labels = labels[idx]
        boot_diff = np.mean(boot_sample[boot_labels == 1]) - np.mean(boot_sample[boot_labels == 0])
        bootstrap_diffs.append(boot_diff)

    boot_ci_lower = np.percentile(bootstrap_diffs, 2.5)
    boot_ci_upper = np.percentile(bootstrap_diffs, 97.5)

    # Normality test
    _, p_norm_baseline = scipy_stats.shapiro(baseline_values) if len(baseline_values) >= 3 else (0, 1.0)
    _, p_norm_modified = scipy_stats.shapiro(modified_values) if len(modified_values) >= 3 else (0, 1.0)

    # Levene's test for equal variances
    _, p_levene = scipy_stats.levene(baseline_values, modified_values)

    return {
        'metric': metric_name,
        'baseline_mean': baseline_mean,
        'modified_mean': modified_mean,
        'absolute_diff': absolute_diff,
        'percent_diff': percent_diff,
        'improvement': improvement,
        'ttest_statistic': t_stat,
        'ttest_p_value': p_value_ttest,
        'permutation_p_value': p_value_perm,
        'cohens_d': effect_size,
        'bootstrap_ci_diff': [boot_ci_lower, boot_ci_upper],
        'normality_baseline_p': p_norm_baseline,
        'normality_modified_p': p_norm_modified,
        'levene_p_value': p_levene,
        'lower_is_better': lower_is_better,
    }

# ============================================================================
# SECTION 4: Visualization Functions
# ============================================================================

def plot_distributions(baseline_data: Dict, palm_data: Dict, output_dir: Path):
    """Create distribution plots for final metrics."""

    # Extract final metrics
    baseline_results = baseline_data['results']
    palm_results = palm_data['results']

    baseline_val_loss = [r['final_val_loss'] for r in baseline_results]
    palm_val_loss = [r['final_val_loss'] for r in palm_results]

    baseline_train_loss = [r['final_train_loss'] for r in baseline_results]
    palm_train_loss = [r['final_train_loss'] for r in palm_results]

    baseline_time = [r['time_seconds'] for r in baseline_results]
    palm_time = [r['time_seconds'] for r in palm_results]

    baseline_memory = [r['peak_memory_mib'] for r in baseline_results]
    palm_memory = [r['peak_memory_mib'] for r in palm_results]

    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Distribution Comparisons: Baseline vs PaLM-Parallel', fontsize=14, fontweight='bold')

    # Plot 1: Validation Loss
    ax = axes[0, 0]
    positions = [1, 2]
    data_to_plot = [baseline_val_loss, palm_val_loss]
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True)

    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)

    # Overlay individual points
    for i, (pos, data) in enumerate(zip(positions, data_to_plot)):
        x = np.random.normal(pos, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.6, s=100, edgecolor='black', linewidth=1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(['Baseline', 'PaLM-Parallel'])
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title('Final Validation Loss')
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Training Loss
    ax = axes[0, 1]
    data_to_plot = [baseline_train_loss, palm_train_loss]
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True)

    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)

    for i, (pos, data) in enumerate(zip(positions, data_to_plot)):
        x = np.random.normal(pos, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.6, s=100, edgecolor='black', linewidth=1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(['Baseline', 'PaLM-Parallel'])
    ax.set_ylabel('Training Loss', fontweight='bold')
    ax.set_title('Final Training Loss')
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Training Time
    ax = axes[1, 0]
    data_to_plot = [baseline_time, palm_time]
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True)

    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)

    for i, (pos, data) in enumerate(zip(positions, data_to_plot)):
        x = np.random.normal(pos, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.6, s=100, edgecolor='black', linewidth=1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(['Baseline', 'PaLM-Parallel'])
    ax.set_ylabel('Time (seconds)', fontweight='bold')
    ax.set_title('Training Time')
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Peak Memory
    ax = axes[1, 1]
    data_to_plot = [baseline_memory, palm_memory]
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True)

    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)

    for i, (pos, data) in enumerate(zip(positions, data_to_plot)):
        x = np.random.normal(pos, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.6, s=100, edgecolor='black', linewidth=1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(['Baseline', 'PaLM-Parallel'])
    ax.set_ylabel('Peak Memory (MiB)', fontweight='bold')
    ax.set_title('Peak Memory Usage')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "metric_distributions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_file}")

def plot_training_curves(baseline_curves: Dict, palm_curves: Dict, output_dir: Path):
    """Plot training and validation loss curves over training steps."""

    # Create 2x2 grid: baseline train, palm train, baseline val, palm val
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Dynamics: Baseline vs PaLM-Parallel', fontsize=14, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Plot 1: Baseline Training Loss
    ax = axes[0, 0]
    all_train_losses = []
    min_len = min([len(curve['train_losses']) for curve in baseline_curves.values()])

    for run_id, curve in baseline_curves.items():
        steps = curve['train_steps'][:min_len]
        losses = curve['train_losses'][:min_len]
        ax.plot(steps, losses, alpha=0.3, color=colors[run_id], linewidth=1)
        all_train_losses.append(losses)

    # Plot mean using truncated arrays
    all_train_losses = np.array(all_train_losses)
    mean_losses = np.mean(all_train_losses, axis=0)
    std_losses = np.std(all_train_losses, axis=0)
    steps_truncated = baseline_curves[0]['train_steps'][:min_len]

    ax.plot(steps_truncated, mean_losses,
            color='black', linewidth=2, label='Mean')
    ax.fill_between(steps_truncated,
                     mean_losses - std_losses, mean_losses + std_losses,
                     alpha=0.2, color='gray', label='±1 SD')

    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('Training Loss', fontweight='bold')
    ax.set_title('Baseline: Training Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: PaLM-Parallel Training Loss
    ax = axes[0, 1]
    all_train_losses = []
    min_len = min([len(curve['train_losses']) for curve in palm_curves.values()])

    for run_id, curve in palm_curves.items():
        steps = curve['train_steps'][:min_len]
        losses = curve['train_losses'][:min_len]
        ax.plot(steps, losses, alpha=0.3, color=colors[run_id], linewidth=1)
        all_train_losses.append(losses)

    all_train_losses = np.array(all_train_losses)
    mean_losses = np.mean(all_train_losses, axis=0)
    std_losses = np.std(all_train_losses, axis=0)
    steps_truncated = palm_curves[0]['train_steps'][:min_len]

    ax.plot(steps_truncated, mean_losses,
            color='black', linewidth=2, label='Mean')
    ax.fill_between(steps_truncated,
                     mean_losses - std_losses, mean_losses + std_losses,
                     alpha=0.2, color='gray', label='±1 SD')

    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('Training Loss', fontweight='bold')
    ax.set_title('PaLM-Parallel: Training Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Baseline Validation Loss
    ax = axes[1, 0]
    all_val_losses = []
    min_len = min([len(curve['val_losses']) for curve in baseline_curves.values()])

    for run_id, curve in baseline_curves.items():
        steps = curve['val_steps'][:min_len]
        losses = curve['val_losses'][:min_len]
        ax.plot(steps, losses, alpha=0.3, color=colors[run_id],
                linewidth=1, marker='o', markersize=3)
        all_val_losses.append(losses)

    all_val_losses = np.array(all_val_losses)
    mean_losses = np.mean(all_val_losses, axis=0)
    std_losses = np.std(all_val_losses, axis=0)
    steps_truncated = baseline_curves[0]['val_steps'][:min_len]

    ax.plot(steps_truncated, mean_losses,
            color='black', linewidth=2, marker='o', markersize=4, label='Mean')
    ax.fill_between(steps_truncated,
                     mean_losses - std_losses, mean_losses + std_losses,
                     alpha=0.2, color='gray', label='±1 SD')

    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title('Baseline: Validation Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: PaLM-Parallel Validation Loss
    ax = axes[1, 1]
    all_val_losses = []
    min_len = min([len(curve['val_losses']) for curve in palm_curves.values()])

    for run_id, curve in palm_curves.items():
        steps = curve['val_steps'][:min_len]
        losses = curve['val_losses'][:min_len]
        ax.plot(steps, losses, alpha=0.3, color=colors[run_id],
                linewidth=1, marker='o', markersize=3)
        all_val_losses.append(losses)

    all_val_losses = np.array(all_val_losses)
    mean_losses = np.mean(all_val_losses, axis=0)
    std_losses = np.std(all_val_losses, axis=0)
    steps_truncated = palm_curves[0]['val_steps'][:min_len]

    ax.plot(steps_truncated, mean_losses,
            color='black', linewidth=2, marker='o', markersize=4, label='Mean')
    ax.fill_between(steps_truncated,
                     mean_losses - std_losses, mean_losses + std_losses,
                     alpha=0.2, color='gray', label='±1 SD')

    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title('PaLM-Parallel: Validation Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "training_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_file}")

def plot_loss_comparison(baseline_curves: Dict, palm_curves: Dict, output_dir: Path):
    """Create combined comparison plot of loss curves."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Loss Convergence Comparison', fontsize=14, fontweight='bold')

    # Training Loss Comparison - truncate to minimum length
    baseline_train_losses = [curve['train_losses'] for curve in baseline_curves.values()]
    palm_train_losses = [curve['train_losses'] for curve in palm_curves.values()]

    baseline_min_len = min([len(losses) for losses in baseline_train_losses])
    palm_min_len = min([len(losses) for losses in palm_train_losses])
    min_len = min(baseline_min_len, palm_min_len)

    baseline_train_losses = np.array([losses[:min_len] for losses in baseline_train_losses])
    palm_train_losses = np.array([losses[:min_len] for losses in palm_train_losses])

    baseline_mean = np.mean(baseline_train_losses, axis=0)
    baseline_std = np.std(baseline_train_losses, axis=0)
    palm_mean = np.mean(palm_train_losses, axis=0)
    palm_std = np.std(palm_train_losses, axis=0)

    steps = baseline_curves[0]['train_steps'][:min_len]

    ax1.plot(steps, baseline_mean, color='blue', linewidth=2, label='Baseline')
    ax1.fill_between(steps, baseline_mean - baseline_std, baseline_mean + baseline_std,
                      alpha=0.2, color='blue')

    ax1.plot(steps, palm_mean, color='red', linewidth=2, label='PaLM-Parallel')
    ax1.fill_between(steps, palm_mean - palm_std, palm_mean + palm_std,
                      alpha=0.2, color='red')

    ax1.set_xlabel('Training Step', fontweight='bold')
    ax1.set_ylabel('Training Loss', fontweight='bold')
    ax1.set_title('Training Loss Convergence')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Validation Loss Comparison - truncate to minimum length
    baseline_val_losses = [curve['val_losses'] for curve in baseline_curves.values()]
    palm_val_losses = [curve['val_losses'] for curve in palm_curves.values()]

    baseline_min_len = min([len(losses) for losses in baseline_val_losses])
    palm_min_len = min([len(losses) for losses in palm_val_losses])
    min_len = min(baseline_min_len, palm_min_len)

    baseline_val_losses = np.array([losses[:min_len] for losses in baseline_val_losses])
    palm_val_losses = np.array([losses[:min_len] for losses in palm_val_losses])

    baseline_mean = np.mean(baseline_val_losses, axis=0)
    baseline_std = np.std(baseline_val_losses, axis=0)
    palm_mean = np.mean(palm_val_losses, axis=0)
    palm_std = np.std(palm_val_losses, axis=0)

    val_steps = baseline_curves[0]['val_steps'][:min_len]

    ax2.plot(val_steps, baseline_mean, color='blue', linewidth=2,
             marker='o', markersize=4, label='Baseline')
    ax2.fill_between(val_steps, baseline_mean - baseline_std, baseline_mean + baseline_std,
                      alpha=0.2, color='blue')

    ax2.plot(val_steps, palm_mean, color='red', linewidth=2,
             marker='s', markersize=4, label='PaLM-Parallel')
    ax2.fill_between(val_steps, palm_mean - palm_std, palm_mean + palm_std,
                      alpha=0.2, color='red')

    ax2.set_xlabel('Training Step', fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontweight='bold')
    ax2.set_title('Validation Loss Convergence')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "loss_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_file}")

def plot_violin_distributions(baseline_data: Dict, palm_data: Dict, output_dir: Path):
    """Create violin plots for individual experiments."""

    # Extract metrics
    baseline_results = baseline_data['results']
    palm_results = palm_data['results']

    baseline_val_loss = [r['final_val_loss'] for r in baseline_results]
    palm_val_loss = [r['final_val_loss'] for r in palm_results]

    baseline_time = [r['time_seconds'] for r in baseline_results]
    palm_time = [r['time_seconds'] for r in palm_results]

    # Individual violin plots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Distribution Shapes: Baseline vs PaLM-Parallel',
                 fontsize=14, fontweight='bold')

    # Plot 1: Baseline validation loss
    ax = axes[0, 0]
    sns.violinplot(y=baseline_val_loss, ax=ax, color='steelblue')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title(f'Baseline Val Loss (n={len(baseline_val_loss)})')
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: PaLM validation loss
    ax = axes[0, 1]
    sns.violinplot(y=palm_val_loss, ax=ax, color='coral')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title(f'PaLM-Parallel Val Loss (n={len(palm_val_loss)})')
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Baseline time
    ax = axes[1, 0]
    sns.violinplot(y=baseline_time, ax=ax, color='steelblue')
    ax.set_ylabel('Training Time (s)', fontweight='bold')
    ax.set_title(f'Baseline Time (n={len(baseline_time)})')
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: PaLM time
    ax = axes[1, 1]
    sns.violinplot(y=palm_time, ax=ax, color='coral')
    ax.set_ylabel('Training Time (s)', fontweight='bold')
    ax.set_title(f'PaLM-Parallel Time (n={len(palm_time)})')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "violin_distributions_individual.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_file}")

def plot_violin_comparison(baseline_data: Dict, palm_data: Dict, output_dir: Path):
    """Create comparison violin plot with overlaid box plots (CIFAR-10 style)."""

    # Prepare data in long format for seaborn
    baseline_results = baseline_data['results']
    palm_results = palm_data['results']

    data = []
    for r in baseline_results:
        data.append({
            'Experiment': 'Baseline',
            'Validation Loss': r['final_val_loss'],
            'Training Time': r['time_seconds']
        })
    for r in palm_results:
        data.append({
            'Experiment': 'PaLM-Parallel',
            'Validation Loss': r['final_val_loss'],
            'Training Time': r['time_seconds']
        })

    df = pd.DataFrame(data)

    # Create 1x2 plot (val loss and time)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Violin Plot Comparison: Baseline vs PaLM-Parallel',
                 fontsize=14, fontweight='bold')

    # Validation loss comparison
    ax = axes[0]
    sns.violinplot(data=df, x='Experiment', y='Validation Loss',
                   ax=ax, palette='Set2')
    # Overlay box plot
    sns.boxplot(data=df, x='Experiment', y='Validation Loss',
                ax=ax, width=0.3, palette='dark:gray',
                linewidth=1.5, fliersize=0)
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_xlabel('')
    ax.set_title('Validation Loss Distribution')
    ax.grid(axis='y', alpha=0.3)

    # Training time comparison
    ax = axes[1]
    sns.violinplot(data=df, x='Experiment', y='Training Time',
                   ax=ax, palette='Set2')
    # Overlay box plot
    sns.boxplot(data=df, x='Experiment', y='Training Time',
                ax=ax, width=0.3, palette='dark:gray',
                linewidth=1.5, fliersize=0)
    ax.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax.set_xlabel('')
    ax.set_title('Training Time Distribution')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "violin_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_file}")

# ============================================================================
# SECTION 5: Main Analysis
# ============================================================================

print("Loading experiment data...")
print()

# Load data
baseline_data = load_experiment_data(BASELINE_PATH)
palm_data = load_experiment_data(PALM_PARALLEL_PATH)

print(f"✓ Loaded baseline: {len(baseline_data['results'])} runs")
print(f"✓ Loaded PaLM-parallel: {len(palm_data['results'])} runs")
print()

# Load training curves
print("Loading training curves...")
baseline_curves = load_training_curves(BASELINE_PATH)
palm_curves = load_training_curves(PALM_PARALLEL_PATH)
print()

# Extract final metrics
baseline_val_loss = np.array([r['final_val_loss'] for r in baseline_data['results']])
palm_val_loss = np.array([r['final_val_loss'] for r in palm_data['results']])

baseline_train_loss = np.array([r['final_train_loss'] for r in baseline_data['results']])
palm_train_loss = np.array([r['final_train_loss'] for r in palm_data['results']])

baseline_time = np.array([r['time_seconds'] for r in baseline_data['results']])
palm_time = np.array([r['time_seconds'] for r in palm_data['results']])

baseline_memory = np.array([r['peak_memory_mib'] for r in baseline_data['results']])
palm_memory = np.array([r['peak_memory_mib'] for r in palm_data['results']])

# Compute descriptive statistics
print("Computing descriptive statistics...")
baseline_stats = {
    'val_loss': compute_descriptive_stats(baseline_val_loss),
    'train_loss': compute_descriptive_stats(baseline_train_loss),
    'time': compute_descriptive_stats(baseline_time),
    'memory': compute_descriptive_stats(baseline_memory),
}

palm_stats = {
    'val_loss': compute_descriptive_stats(palm_val_loss),
    'train_loss': compute_descriptive_stats(palm_train_loss),
    'time': compute_descriptive_stats(palm_time),
    'memory': compute_descriptive_stats(palm_memory),
}
print("  ✓ Computed descriptive statistics")
print()

# Perform comparisons
print("Performing statistical comparisons...")
comparison_val_loss = compare_experiments(baseline_val_loss, palm_val_loss,
                                         "Validation Loss", lower_is_better=True)
comparison_train_loss = compare_experiments(baseline_train_loss, palm_train_loss,
                                           "Training Loss", lower_is_better=True)
comparison_time = compare_experiments(baseline_time, palm_time,
                                     "Training Time", lower_is_better=True)
comparison_memory = compare_experiments(baseline_memory, palm_memory,
                                       "Peak Memory", lower_is_better=True)
print("  ✓ Completed statistical comparisons")
print()

# Create visualizations
print("Generating visualizations...")
plot_distributions(baseline_data, palm_data, FIGURES_DIR)
plot_training_curves(baseline_curves, palm_curves, FIGURES_DIR)
plot_loss_comparison(baseline_curves, palm_curves, FIGURES_DIR)
plot_violin_distributions(baseline_data, palm_data, FIGURES_DIR)
plot_violin_comparison(baseline_data, palm_data, FIGURES_DIR)
print()

# Export results
print("Exporting results...")

# Summary statistics CSV
import csv
with open(RESULTS_DIR / "stage_c_summary_stats.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Experiment', 'Metric', 'Mean', 'Std', 'Min', 'Max', 'CI_95_Lower', 'CI_95_Upper'])

    for exp_name, stats in [('Baseline', baseline_stats), ('PaLM-Parallel', palm_stats)]:
        for metric_name, metric_stats in stats.items():
            writer.writerow([
                exp_name,
                metric_name,
                f"{metric_stats['mean']:.4f}",
                f"{metric_stats['std']:.4f}",
                f"{metric_stats['min']:.4f}",
                f"{metric_stats['max']:.4f}",
                f"{metric_stats['ci_95'][0]:.4f}",
                f"{metric_stats['ci_95'][1]:.4f}",
            ])

print(f"  ✓ Saved {RESULTS_DIR / 'stage_c_summary_stats.csv'}")

# Comparison results JSON
comparison_results = {
    'val_loss': comparison_val_loss,
    'train_loss': comparison_train_loss,
    'time': comparison_time,
    'memory': comparison_memory,
}

with open(RESULTS_DIR / "stage_c_comparison_results.json", 'w') as f:
    json.dump(comparison_results, f, indent=2)

print(f"  ✓ Saved {RESULTS_DIR / 'stage_c_comparison_results.json'}")

# Full results export
full_results = {
    'baseline': {
        'summary': baseline_data['summary'],
        'config': baseline_data['config'],
        'statistics': baseline_stats,
    },
    'palm_parallel': {
        'summary': palm_data['summary'],
        'config': palm_data['config'],
        'statistics': palm_stats,
    },
    'comparisons': comparison_results,
}

with open(RESULTS_DIR / "stage_c_full_results.json", 'w') as f:
    json.dump(full_results, f, indent=2)

print(f"  ✓ Saved {RESULTS_DIR / 'stage_c_full_results.json'}")
print()

# ============================================================================
# SECTION 6: Print Summary
# ============================================================================

print("=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)
print()

print("Validation Loss:")
print(f"  Baseline:      {baseline_stats['val_loss']['mean']:.4f} ± {baseline_stats['val_loss']['std']:.4f}")
print(f"  PaLM-Parallel: {palm_stats['val_loss']['mean']:.4f} ± {palm_stats['val_loss']['std']:.4f}")
print(f"  Difference:    {comparison_val_loss['absolute_diff']:.4f} ({comparison_val_loss['percent_diff']:+.2f}%)")
print(f"  t-test p:      {comparison_val_loss['ttest_p_value']:.6f}")
print(f"  Perm test p:   {comparison_val_loss['permutation_p_value']:.4f}")
print(f"  Cohen's d:     {comparison_val_loss['cohens_d']:.4f}")
print()

print("Training Time:")
print(f"  Baseline:      {baseline_stats['time']['mean']:.2f} ± {baseline_stats['time']['std']:.2f} seconds")
print(f"  PaLM-Parallel: {palm_stats['time']['mean']:.2f} ± {palm_stats['time']['std']:.2f} seconds")
print(f"  Difference:    {comparison_time['absolute_diff']:.2f} seconds ({comparison_time['percent_diff']:+.2f}%)")
print(f"  Improvement:   {comparison_time['improvement']:+.2f}%")
print(f"  t-test p:      {comparison_time['ttest_p_value']:.6f}")
print(f"  Cohen's d:     {comparison_time['cohens_d']:.4f}")
print()

print("Peak Memory:")
print(f"  Baseline:      {baseline_stats['memory']['mean']:.0f} ± {baseline_stats['memory']['std']:.0f} MiB")
print(f"  PaLM-Parallel: {palm_stats['memory']['mean']:.0f} ± {palm_stats['memory']['std']:.0f} MiB")
print(f"  Difference:    {comparison_memory['absolute_diff']:.0f} MiB ({comparison_memory['percent_diff']:+.2f}%)")
print(f"  Improvement:   {comparison_memory['improvement']:+.2f}%")
print()

print("=" * 80)
print("Analysis complete! Check the following outputs:")
print(f"  - Figures: {FIGURES_DIR}/")
print(f"  - Results: {RESULTS_DIR}/")
print("=" * 80)
