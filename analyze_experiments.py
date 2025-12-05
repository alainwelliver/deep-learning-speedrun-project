#!/usr/bin/env python3
"""
analyze_experiments.py

Comprehensive statistics analysis tool for CIFAR-10 speedrun experiments.
Computes extended statistics, performs comparisons, generates visualizations,
and exports results in formats suitable for academic reports.

Usage:
    # Analyze single experiment
    python analyze_experiments.py single experiment_logs/cifar10_baseline_*/

    # Compare experiments
    python analyze_experiments.py compare \\
        --baseline experiment_logs/baseline_*/ \\
        --modified experiment_logs/modified_*/

    # Or use as module
    from analyze_experiments import ExperimentAnalyzer, quick_summary
    analyzer = ExperimentAnalyzer("experiment_logs/baseline_*/")
    stats = analyzer.compute_extended_statistics()
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from scipy import stats as scipy_stats
import glob

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class ExperimentAnalyzer:
    """
    Loads and analyzes results from a single CIFAR-10 experiment.

    Supports loading from:
    - Experiment directory (reads results.jsonl and summary.json)
    - Direct path to summary.json

    Computes extended statistics beyond what ExperimentLogger provides:
    - Multiple confidence intervals (95%, 99%)
    - Percentiles (25th, 50th, 75th, 90th, 95th, 99th)
    - Coefficient of variation
    - Success rate
    """

    def __init__(self, experiment_path: Union[str, Path]):
        """
        Initialize analyzer with experiment directory or summary.json path.

        Args:
            experiment_path: Path to experiment directory or summary.json file.
                           Supports glob patterns (e.g., "experiment_logs/baseline_*/")
        """
        self.experiment_path = self._resolve_path(experiment_path)

        # Determine if path is directory or summary.json file
        if self.experiment_path.is_dir():
            self.exp_dir = self.experiment_path
            self.results_file = self.exp_dir / "results.jsonl"
            self.summary_file = self.exp_dir / "summary.json"
            self.hyperparams_file = self.exp_dir / "hyperparameters.json"
        elif self.experiment_path.name == "summary.json":
            self.exp_dir = self.experiment_path.parent
            self.results_file = self.exp_dir / "results.jsonl"
            self.summary_file = self.experiment_path
            self.hyperparams_file = self.exp_dir / "hyperparameters.json"
        else:
            raise ValueError(
                f"Invalid path: {experiment_path}. "
                "Must be experiment directory or summary.json file."
            )

        # Validate required files exist
        if not self.results_file.exists():
            raise FileNotFoundError(
                f"results.jsonl not found in {self.exp_dir}. "
                "Make sure you're pointing to a valid experiment directory."
            )

        # Load data
        self._results = None
        self._summary = None
        self._hyperparams = None

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """
        Resolve path, handling glob patterns.

        Args:
            path: Path or glob pattern

        Returns:
            Resolved Path object
        """
        path_str = str(path)

        # Check if it's a glob pattern
        if '*' in path_str or '?' in path_str:
            matches = sorted(glob.glob(path_str))
            if not matches:
                raise FileNotFoundError(f"No matches found for pattern: {path}")
            if len(matches) > 1:
                print(f"Warning: Multiple matches for pattern '{path}'. Using: {matches[0]}")
            path_str = matches[0]

        resolved = Path(path_str).resolve()

        if not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        return resolved

    def load_results(self) -> List[Dict]:
        """
        Load raw per-run results from results.jsonl.

        Returns:
            List of result dictionaries, one per run
        """
        if self._results is not None:
            return self._results

        results = []
        with open(self.results_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    results.append(json.loads(line))

        self._results = results
        return results

    def load_summary(self) -> Optional[Dict]:
        """
        Load pre-computed summary.json if it exists.

        Returns:
            Summary dictionary or None if file doesn't exist
        """
        if self._summary is not None:
            return self._summary

        if not self.summary_file.exists():
            return None

        with open(self.summary_file, 'r') as f:
            self._summary = json.load(f)

        return self._summary

    def load_hyperparameters(self) -> Optional[Dict]:
        """
        Load hyperparameters.json if it exists.

        Returns:
            Hyperparameters dictionary or None if file doesn't exist
        """
        if self._hyperparams is not None:
            return self._hyperparams

        if not self.hyperparams_file.exists():
            return None

        with open(self.hyperparams_file, 'r') as f:
            self._hyperparams = json.load(f)

        return self._hyperparams

    def get_metadata(self) -> Dict:
        """
        Extract metadata from summary.json.

        Returns:
            Dictionary with git_info, gpu_info, system_info, etc.
        """
        summary = self.load_summary()

        if summary is None:
            return {}

        metadata = {}

        for key in ['git_info', 'gpu_info', 'system_info', 'experiment_name', 'additional_info']:
            if key in summary:
                metadata[key] = summary[key]

        return metadata

    def compute_extended_statistics(
        self,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict:
        """
        Compute comprehensive statistics from experiment results.

        Args:
            confidence_levels: List of confidence levels for CIs (default: [0.95, 0.99])

        Returns:
            Dictionary with extended statistics for accuracy and time:
            - Basic: mean, std, median, min, max, range
            - Variability: coefficient of variation (CV)
            - Percentiles: 25th, 50th, 75th, 90th, 95th, 99th
            - Confidence intervals: for each level in confidence_levels
            - Success rate: percentage of successful runs
        """
        results = self.load_results()

        if not results:
            return {"error": "No results to analyze"}

        # Separate successful and failed runs
        successful = [r for r in results if r.get('success', True)]
        failed = [r for r in results if not r.get('success', True)]

        n_total = len(results)
        n_success = len(successful)
        n_failed = len(failed)
        success_rate = n_success / n_total if n_total > 0 else 0.0

        if not successful:
            return {
                "error": "No successful runs to analyze",
                "n_runs": n_total,
                "successful_runs": n_success,
                "failed_runs": n_failed,
                "success_rate": success_rate
            }

        # Extract metrics from successful runs
        accuracies = np.array([r['accuracy'] for r in successful])
        times = np.array([r['time_seconds'] for r in successful])

        # Compute statistics
        stats = {
            "n_runs": n_total,
            "successful_runs": n_success,
            "failed_runs": n_failed,
            "success_rate": float(success_rate),
            "accuracy": self._compute_metric_stats(accuracies, confidence_levels),
            "time": self._compute_metric_stats(times, confidence_levels),
        }

        return stats

    def _compute_metric_stats(
        self,
        values: np.ndarray,
        confidence_levels: List[float]
    ) -> Dict:
        """
        Compute statistics for a single metric (accuracy or time).

        Args:
            values: Array of metric values
            confidence_levels: List of confidence levels for CIs

        Returns:
            Dictionary with comprehensive statistics
        """
        n = len(values)

        # Basic statistics
        metric_stats = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
        }

        # Coefficient of variation (CV)
        if metric_stats["mean"] > 0:
            metric_stats["cv"] = metric_stats["std"] / metric_stats["mean"]
        else:
            metric_stats["cv"] = float('inf')

        # Percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        metric_stats["percentiles"] = {}
        for p in percentiles:
            metric_stats["percentiles"][f"p{p}"] = float(np.percentile(values, p))

        # Confidence intervals using t-distribution
        metric_stats["confidence_intervals"] = {}
        for conf_level in confidence_levels:
            ci = scipy_stats.t.interval(
                conf_level,
                n - 1,
                loc=metric_stats["mean"],
                scale=scipy_stats.sem(values)
            )
            ci_key = f"ci_{int(conf_level * 100)}"
            metric_stats["confidence_intervals"][ci_key] = [float(ci[0]), float(ci[1])]

        return metric_stats

    def get_experiment_name(self) -> str:
        """Get experiment name from summary or directory name."""
        summary = self.load_summary()
        if summary and 'experiment_name' in summary:
            return summary['experiment_name']
        return self.exp_dir.name


class ExperimentComparator:
    """
    Compares multiple experiments with statistical tests.

    Performs:
    - Pairwise statistical significance tests (t-test, Welch's t-test, Mann-Whitney U)
    - Effect size calculations (Cohen's d)
    - Assumption checking (normality, equal variances)
    - Experiment ranking
    """

    def __init__(self, experiments: Dict[str, ExperimentAnalyzer]):
        """
        Initialize comparator with multiple experiments.

        Args:
            experiments: Dictionary mapping experiment names to ExperimentAnalyzer instances
        """
        self.experiments = experiments
        self.names = list(experiments.keys())

        # Compute statistics for all experiments
        self.stats = {}
        for name, analyzer in experiments.items():
            self.stats[name] = analyzer.compute_extended_statistics()

    def get_raw_values(self, metric: str = 'accuracy') -> Dict[str, np.ndarray]:
        """
        Get raw per-run values for all experiments.

        Args:
            metric: Metric to extract ('accuracy' or 'time')

        Returns:
            Dictionary mapping experiment name to array of values
        """
        values = {}
        for name, analyzer in self.experiments.items():
            results = analyzer.load_results()
            successful = [r for r in results if r.get('success', True)]

            if metric == 'accuracy':
                values[name] = np.array([r['accuracy'] for r in successful])
            elif metric == 'time':
                values[name] = np.array([r['time_seconds'] for r in successful])
            else:
                raise ValueError(f"Invalid metric: {metric}. Must be 'accuracy' or 'time'")

        return values

    def check_assumptions(self, metric: str = 'accuracy') -> Dict:
        """
        Check statistical test assumptions for all experiments.

        Tests:
        - Normality (Shapiro-Wilk test)
        - Equal variances (Levene's test)

        Args:
            metric: Metric to check ('accuracy' or 'time')

        Returns:
            Dictionary with assumption test results
        """
        values = self.get_raw_values(metric)

        assumptions = {}

        # Test normality for each experiment
        assumptions['normality'] = {}
        for name, vals in values.items():
            if len(vals) >= 3:  # Shapiro-Wilk requires n >= 3
                stat, p_value = scipy_stats.shapiro(vals)
                assumptions['normality'][name] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'is_normal': p_value >= 0.05  # Null hypothesis: data is normal
                }
            else:
                assumptions['normality'][name] = {
                    'error': 'Too few samples for normality test'
                }

        # Test equal variances (Levene's test) across all experiments
        if len(values) >= 2:
            all_vals = list(values.values())
            stat, p_value = scipy_stats.levene(*all_vals)
            assumptions['equal_variances'] = {
                'statistic': float(stat),
                'p_value': float(p_value),
                'has_equal_variances': p_value >= 0.05  # Null hypothesis: equal variances
            }
        else:
            assumptions['equal_variances'] = {'error': 'Need at least 2 experiments'}

        return assumptions

    def compute_pairwise_tests(
        self,
        metric: str = 'accuracy',
        test: str = 'auto'
    ) -> Dict:
        """
        Perform pairwise statistical tests between all experiments.

        Args:
            metric: Metric to compare ('accuracy' or 'time')
            test: Test type ('auto', 't-test', 'welch', 'mann-whitney')
                  'auto' selects based on assumptions

        Returns:
            Dictionary with pairwise test results
        """
        values = self.get_raw_values(metric)
        results = {}

        # Get all pairs
        from itertools import combinations
        pairs = list(combinations(self.names, 2))

        for name1, name2 in pairs:
            vals1 = values[name1]
            vals2 = values[name2]

            pair_key = f"{name1}_vs_{name2}"

            # Select test type
            if test == 'auto':
                test_type = self._select_test(vals1, vals2)
            else:
                test_type = test

            # Perform test
            if test_type == 'mann-whitney':
                stat, p_value = scipy_stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
                test_name = "Mann-Whitney U"
            elif test_type == 'welch':
                stat, p_value = scipy_stats.ttest_ind(vals1, vals2, equal_var=False)
                test_name = "Welch's t-test"
            else:  # t-test
                stat, p_value = scipy_stats.ttest_ind(vals1, vals2, equal_var=True)
                test_name = "Independent t-test"

            results[pair_key] = {
                'test_used': test_name,
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'mean_diff': float(np.mean(vals1) - np.mean(vals2))
            }

        return results

    def _select_test(self, vals1: np.ndarray, vals2: np.ndarray) -> str:
        """
        Automatically select appropriate statistical test.

        Selection logic:
        - If n < 30: use Mann-Whitney U (non-parametric)
        - If not normal: use Mann-Whitney U
        - If unequal variances: use Welch's t-test
        - Otherwise: use independent t-test

        Args:
            vals1, vals2: Arrays of values to compare

        Returns:
            Test name: 't-test', 'welch', or 'mann-whitney'
        """
        n1, n2 = len(vals1), len(vals2)

        # Small sample size
        if n1 < 30 or n2 < 30:
            return 'mann-whitney'

        # Check normality
        _, p1 = scipy_stats.shapiro(vals1)
        _, p2 = scipy_stats.shapiro(vals2)

        if p1 < 0.05 or p2 < 0.05:  # Not normal
            return 'mann-whitney'

        # Check equal variances
        _, p_levene = scipy_stats.levene(vals1, vals2)

        if p_levene < 0.05:  # Unequal variances
            return 'welch'

        return 't-test'

    def compute_effect_sizes(self, metric: str = 'accuracy') -> Dict:
        """
        Compute Cohen's d effect size for all pairwise comparisons.

        Effect size interpretation:
        - Small: |d| = 0.2
        - Medium: |d| = 0.5
        - Large: |d| = 0.8

        Args:
            metric: Metric to compare ('accuracy' or 'time')

        Returns:
            Dictionary with effect sizes for all pairs
        """
        values = self.get_raw_values(metric)
        effect_sizes = {}

        from itertools import combinations
        pairs = list(combinations(self.names, 2))

        for name1, name2 in pairs:
            vals1 = values[name1]
            vals2 = values[name2]

            # Cohen's d
            mean_diff = np.mean(vals1) - np.mean(vals2)
            n1, n2 = len(vals1), len(vals2)
            var1, var2 = np.var(vals1, ddof=1), np.var(vals2, ddof=1)

            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

            # Interpretation
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                interpretation = "negligible"
            elif abs_d < 0.5:
                interpretation = "small"
            elif abs_d < 0.8:
                interpretation = "medium"
            else:
                interpretation = "large"

            pair_key = f"{name1}_vs_{name2}"
            effect_sizes[pair_key] = {
                'cohens_d': float(cohens_d),
                'interpretation': interpretation,
                'mean_diff': float(mean_diff),
                'pooled_std': float(pooled_std)
            }

        return effect_sizes

    def rank_experiments(self, metric: str = 'accuracy') -> List[Tuple[str, float, float]]:
        """
        Rank experiments by mean metric value.

        Args:
            metric: Metric to rank by ('accuracy' or 'time')

        Returns:
            List of (name, mean, std) tuples, sorted by mean (descending for accuracy, ascending for time)
        """
        rankings = []

        for name in self.names:
            stat = self.stats[name][metric]
            rankings.append((name, stat['mean'], stat['std']))

        # Sort descending for accuracy, ascending for time
        reverse = (metric == 'accuracy')
        rankings.sort(key=lambda x: x[1], reverse=reverse)

        return rankings


def compare_to_baseline(
    baseline_path: Union[str, Path],
    modified_path: Union[str, Path],
    metric: str = 'accuracy'
) -> Dict:
    """
    Quick comparison of a modified experiment to baseline.

    Args:
        baseline_path: Path to baseline experiment
        modified_path: Path to modified experiment
        metric: Metric to compare ('accuracy' or 'time')

    Returns:
        Dictionary with comparison results:
        - improvement_pct: Percentage improvement
        - p_value: Statistical significance p-value
        - cohens_d: Effect size
        - test_used: Which statistical test was used
    """
    baseline = ExperimentAnalyzer(baseline_path)
    modified = ExperimentAnalyzer(modified_path)

    comparator = ExperimentComparator({
        'Baseline': baseline,
        'Modified': modified
    })

    # Compute tests and effect sizes
    tests = comparator.compute_pairwise_tests(metric=metric, test='auto')
    effects = comparator.compute_effect_sizes(metric=metric)

    # Get baseline and modified means
    baseline_stats = baseline.compute_extended_statistics()
    modified_stats = modified.compute_extended_statistics()

    baseline_mean = baseline_stats[metric]['mean']
    modified_mean = modified_stats[metric]['mean']

    # Calculate improvement
    if metric == 'accuracy':
        improvement_pct = ((modified_mean - baseline_mean) / baseline_mean) * 100
    else:  # time - lower is better
        improvement_pct = ((baseline_mean - modified_mean) / baseline_mean) * 100

    # Extract comparison results
    pair_key = "Baseline_vs_Modified"
    test_result = tests[pair_key]
    effect_result = effects[pair_key]

    return {
        'baseline_mean': baseline_mean,
        'modified_mean': modified_mean,
        'improvement_pct': improvement_pct,
        'p_value': test_result['p_value'],
        'significant': test_result['significant'],
        'test_used': test_result['test_used'],
        'cohens_d': effect_result['cohens_d'],
        'effect_size_interpretation': effect_result['interpretation']
    }


def plot_distribution(
    experiment_path: Union[str, Path],
    metric: str = 'accuracy',
    kind: str = 'violin',
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """
    Plot distribution of results for a single experiment.

    Args:
        experiment_path: Path to experiment directory
        metric: Metric to plot ('accuracy' or 'time')
        kind: Plot type ('violin', 'box', 'hist')
        output_path: If provided, save plot to this path
        show: If True, display plot interactively
    """
    if not VISUALIZATION_AVAILABLE:
        print("ERROR: Visualization libraries (matplotlib, seaborn) not available.")
        print("Install with: pip install matplotlib seaborn")
        return

    analyzer = ExperimentAnalyzer(experiment_path)
    results = analyzer.load_results()
    successful = [r for r in results if r.get('success', True)]

    if metric == 'accuracy':
        values = [r['accuracy'] for r in successful]
        ylabel = 'Accuracy'
    else:
        values = [r['time_seconds'] for r in successful]
        ylabel = 'Time (seconds)'

    # Set style
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    if kind == 'violin':
        sns.violinplot(y=values, ax=ax, color='steelblue')
    elif kind == 'box':
        sns.boxplot(y=values, ax=ax, color='steelblue')
    elif kind == 'hist':
        ax.hist(values, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel(ylabel)
        ax.set_ylabel('Frequency')
    else:
        raise ValueError(f"Invalid plot kind: {kind}")

    if kind != 'hist':
        ax.set_ylabel(ylabel)

    exp_name = analyzer.get_experiment_name()
    ax.set_title(f'{exp_name} - {ylabel} Distribution ({len(values)} runs)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    experiments: Dict[str, Union[str, Path]],
    metric: str = 'accuracy',
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """
    Create side-by-side comparison plot for multiple experiments.

    Args:
        experiments: Dictionary mapping names to experiment paths
        metric: Metric to compare ('accuracy' or 'time')
        output_path: If provided, save plot to this path
        show: If True, display plot interactively
    """
    if not VISUALIZATION_AVAILABLE:
        print("ERROR: Visualization libraries (matplotlib, seaborn) not available.")
        return

    # Load data from all experiments
    data = []
    for name, path in experiments.items():
        analyzer = ExperimentAnalyzer(path)
        results = analyzer.load_results()
        successful = [r for r in results if r.get('success', True)]

        if metric == 'accuracy':
            values = [r['accuracy'] for r in successful]
        else:
            values = [r['time_seconds'] for r in successful]

        for val in values:
            data.append({'Experiment': name, 'Value': val})

    # Create DataFrame for seaborn
    import pandas as pd
    df = pd.DataFrame(data)

    # Set style
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create violin plot
    sns.violinplot(data=df, x='Experiment', y='Value', ax=ax, palette='Set2')

    # Overlay box plot for quartiles
    sns.boxplot(data=df, x='Experiment', y='Value', ax=ax,
                width=0.3, palette='dark:gray', linewidth=1.5, fliersize=0)

    if metric == 'accuracy':
        ylabel = 'Accuracy'
    else:
        ylabel = 'Time (seconds)'

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_title(f'{ylabel} Comparison', fontsize=14, fontweight='bold')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def export_csv(
    experiments: Dict[str, Union[str, Path]],
    output_path: Union[str, Path],
    include_raw_data: bool = False
) -> None:
    """
    Export experiment statistics to CSV format.

    Args:
        experiments: Dictionary mapping names to experiment paths
        output_path: Output CSV file path
        include_raw_data: If True, also export raw per-run data
    """
    import csv

    rows = []

    for name, path in experiments.items():
        analyzer = ExperimentAnalyzer(path)
        stats = analyzer.compute_extended_statistics()

        if "error" in stats:
            print(f"Warning: Skipping {name} due to error: {stats['error']}")
            continue

        row = {
            'experiment': name,
            'n_runs': stats['n_runs'],
            'successful_runs': stats['successful_runs'],
            'failed_runs': stats['failed_runs'],
            'success_rate': stats['success_rate'],
            'acc_mean': stats['accuracy']['mean'],
            'acc_std': stats['accuracy']['std'],
            'acc_median': stats['accuracy']['median'],
            'acc_min': stats['accuracy']['min'],
            'acc_max': stats['accuracy']['max'],
            'acc_ci95_lower': stats['accuracy']['confidence_intervals']['ci_95'][0],
            'acc_ci95_upper': stats['accuracy']['confidence_intervals']['ci_95'][1],
            'acc_ci99_lower': stats['accuracy']['confidence_intervals']['ci_99'][0],
            'acc_ci99_upper': stats['accuracy']['confidence_intervals']['ci_99'][1],
            'time_mean': stats['time']['mean'],
            'time_std': stats['time']['std'],
            'time_median': stats['time']['median'],
            'time_min': stats['time']['min'],
            'time_max': stats['time']['max'],
        }
        rows.append(row)

    # Write CSV
    if rows:
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"Statistics exported to: {output_path}")

    # Optionally export raw data
    if include_raw_data:
        raw_output = Path(output_path).with_stem(Path(output_path).stem + '_raw')

        raw_rows = []
        for name, path in experiments.items():
            analyzer = ExperimentAnalyzer(path)
            results = analyzer.load_results()

            for r in results:
                raw_rows.append({
                    'experiment': name,
                    'run_id': r['run_id'],
                    'seed': r['seed'],
                    'accuracy': r['accuracy'],
                    'time_seconds': r['time_seconds'],
                    'success': r.get('success', True),
                })

        if raw_rows:
            with open(raw_output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=raw_rows[0].keys())
                writer.writeheader()
                writer.writerows(raw_rows)

            print(f"Raw data exported to: {raw_output}")


def export_latex_table(
    experiments: Dict[str, Union[str, Path]],
    output_path: Union[str, Path],
    metrics: List[str] = ['accuracy', 'time'],
    caption: str = "Experiment Results",
    label: str = "tab:results"
) -> None:
    """
    Export comparison table in LaTeX format for academic reports.

    Args:
        experiments: Dictionary mapping names to experiment paths
        output_path: Output .tex file path
        metrics: Metrics to include in table
        caption: LaTeX table caption
        label: LaTeX table label for referencing
    """
    lines = []

    # Table header
    n_metrics = len(metrics)
    col_spec = "l" + "c" * (n_metrics * 2)  # Name + 2 cols per metric (mean±std, CI)

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Column headers
    header_parts = ["Experiment"]
    for metric in metrics:
        if metric == 'accuracy':
            header_parts.extend(["Accuracy (mean±std)", "95\\% CI"])
        else:
            header_parts.extend(["Time (s) (mean±std)", "95\\% CI"])

    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for name, path in experiments.items():
        analyzer = ExperimentAnalyzer(path)
        stats = analyzer.compute_extended_statistics()

        if "error" in stats:
            continue

        row_parts = [name.replace("_", "\\_")]  # Escape underscores

        for metric in metrics:
            stat = stats[metric]
            mean = stat['mean']
            std = stat['std']
            ci = stat['confidence_intervals']['ci_95']

            if metric == 'accuracy':
                mean_std_str = f"{mean:.4f} ± {std:.4f}"
                ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            else:  # time
                mean_std_str = f"{mean:.3f} ± {std:.3f}"
                ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"

            row_parts.extend([mean_std_str, ci_str])

        lines.append(" & ".join(row_parts) + " \\\\")

    # Table footer
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    # Write file
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"LaTeX table exported to: {output_path}")
    print(f"\nTo use in your report, add to preamble:")
    print("  \\usepackage{booktabs}")
    print(f"\nThen reference in text with: \\ref{{{label}}}")


def quick_summary(experiment_path: Union[str, Path]) -> None:
    """
    Print a formatted summary of an experiment to console.

    Args:
        experiment_path: Path to experiment directory or summary.json
    """
    try:
        analyzer = ExperimentAnalyzer(experiment_path)
        stats = analyzer.compute_extended_statistics()

        # Header
        exp_name = analyzer.get_experiment_name()
        print(f"\n{'='*80}")
        print(f"EXPERIMENT SUMMARY: {exp_name}")
        print(f"{'='*80}\n")

        # Error check
        if "error" in stats:
            print(f"ERROR: {stats['error']}")
            print(f"Runs: {stats.get('n_runs', 0)} total, "
                  f"{stats.get('successful_runs', 0)} successful, "
                  f"{stats.get('failed_runs', 0)} failed")
            return

        # Run information
        print(f"Runs: {stats['n_runs']} total")
        if stats['failed_runs'] > 0:
            print(f"      {stats['successful_runs']} successful, {stats['failed_runs']} failed")
            print(f"      Success rate: {stats['success_rate']*100:.1f}%")
        print()

        # Accuracy statistics
        acc = stats['accuracy']
        print("ACCURACY STATISTICS:")
        print(f"  Mean:       {acc['mean']:.6f} ± {acc['std']:.6f} (std)")
        print(f"  Median:     {acc['median']:.6f}")
        print(f"  Range:      [{acc['min']:.6f}, {acc['max']:.6f}]")
        print(f"  CV:         {acc['cv']:.4f}")
        print()

        # Confidence intervals
        print("  Confidence Intervals:")
        for ci_key, ci_val in sorted(acc['confidence_intervals'].items()):
            conf_pct = ci_key.replace('ci_', '')
            print(f"    {conf_pct}% CI:  [{ci_val[0]:.6f}, {ci_val[1]:.6f}]")
        print()

        # Percentiles
        print("  Percentiles:")
        percs = acc['percentiles']
        print(f"    25th: {percs['p25']:.6f}  |  50th: {percs['p50']:.6f}  |  75th: {percs['p75']:.6f}")
        print(f"    90th: {percs['p90']:.6f}  |  95th: {percs['p95']:.6f}  |  99th: {percs['p99']:.6f}")
        print()

        # Time statistics
        time = stats['time']
        print("TIME STATISTICS:")
        print(f"  Mean:       {time['mean']:.4f}s ± {time['std']:.4f}s (std)")
        print(f"  Median:     {time['median']:.4f}s")
        print(f"  Range:      [{time['min']:.4f}s, {time['max']:.4f}s]")
        total_time = time['mean'] * stats['successful_runs']
        print(f"  Total:      {total_time:.2f}s ({total_time/60:.2f} min, {total_time/3600:.2f} hr)")
        print()

        # Metadata
        metadata = analyzer.get_metadata()
        if metadata:
            print("EXPERIMENT METADATA:")

            if 'git_info' in metadata:
                git = metadata['git_info']
                print(f"  Git Commit: {git.get('commit', 'N/A')[:12]}")
                print(f"  Branch:     {git.get('branch', 'N/A')}")
                if git.get('dirty', False):
                    print("  Status:     DIRTY (uncommitted changes)")

            if 'gpu_info' in metadata and metadata['gpu_info'].get('available'):
                gpu = metadata['gpu_info']
                if 'devices' in gpu and gpu['devices']:
                    device = gpu['devices'][0]
                    print(f"  GPU:        {device.get('name', 'Unknown')} "
                          f"({device.get('total_memory_gb', 0):.1f} GB)")

            if 'additional_info' in metadata:
                addl = metadata['additional_info']
                if 'config_path' in addl:
                    print(f"  Config:     {addl['config_path']}")

        print(f"\n{'='*80}\n")

    except Exception as e:
        print(f"ERROR: Failed to analyze experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Command-line interface for experiment analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze CIFAR-10 speedrun experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single experiment
  python analyze_experiments.py single experiment_logs/cifar10_baseline_*/

  # Using glob pattern
  python analyze_experiments.py single "experiment_logs/*baseline*"
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Analysis command')

    # Single experiment analysis
    single_parser = subparsers.add_parser(
        'single',
        help='Analyze a single experiment'
    )
    single_parser.add_argument(
        'experiment_path',
        type=str,
        help='Path to experiment directory or summary.json (supports glob patterns)'
    )
    single_parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON instead of formatted text'
    )

    # Compare experiments
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare multiple experiments'
    )
    compare_parser.add_argument(
        '--baseline',
        type=str,
        required=True,
        help='Path to baseline experiment (supports glob patterns)'
    )
    compare_parser.add_argument(
        '--modified',
        type=str,
        required=True,
        help='Path to modified experiment (supports glob patterns)'
    )
    compare_parser.add_argument(
        '--metric',
        type=str,
        default='accuracy',
        choices=['accuracy', 'time'],
        help='Metric to compare (default: accuracy)'
    )
    compare_parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON instead of formatted text'
    )

    args = parser.parse_args()

    # Show help if no command specified
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Handle single experiment analysis
    if args.command == 'single':
        if args.json:
            # JSON output
            analyzer = ExperimentAnalyzer(args.experiment_path)
            stats = analyzer.compute_extended_statistics()
            print(json.dumps(stats, indent=2))
        else:
            # Formatted text output
            quick_summary(args.experiment_path)

    # Handle comparison
    elif args.command == 'compare':
        result = compare_to_baseline(
            args.baseline,
            args.modified,
            metric=args.metric
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Formatted output
            print(f"\n{'='*80}")
            print(f"BASELINE vs MODIFIED COMPARISON ({args.metric.upper()})")
            print(f"{'='*80}\n")

            print(f"Baseline mean:  {result['baseline_mean']:.6f}")
            print(f"Modified mean:  {result['modified_mean']:.6f}")
            print(f"Improvement:    {result['improvement_pct']:+.2f}%")
            print()

            print(f"Statistical Test: {result['test_used']}")
            print(f"P-value:          {result['p_value']:.6f}")

            if result['significant']:
                print(f"Result:           STATISTICALLY SIGNIFICANT (p < 0.05)")
            else:
                print(f"Result:           NOT statistically significant (p >= 0.05)")
            print()

            print(f"Effect Size:")
            print(f"  Cohen's d:      {result['cohens_d']:.4f}")
            print(f"  Interpretation: {result['effect_size_interpretation']}")

            print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
