#!/usr/bin/env python3
"""
analyze_results.py

Comprehensive statistics analysis tool for NanoGPT speedrun experiments.
Computes extended statistics, performs comparisons with statistical tests,
and exports results suitable for academic reports.

Adapted from CIFAR-10 analyze_experiments.py for validation loss metric.

Usage:
    # Analyze single experiment
    python scripts/analyze_results.py single experiment_logs/nanogpt_baseline_*/

    # Compare two experiments
    python scripts/analyze_results.py compare \\
        --baseline experiment_logs/nanogpt_baseline_*/ \\
        --modified experiment_logs/nanogpt_modified_*/

    # Or use as module
    from analyze_results import ExperimentAnalyzer
    analyzer = ExperimentAnalyzer("experiment_logs/nanogpt_baseline_*/")
    stats = analyzer.compute_extended_statistics()
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from scipy import stats as scipy_stats
import glob


class ExperimentAnalyzer:
    """
    Loads and analyzes results from a single NanoGPT experiment.

    Supports loading from:
    - Experiment directory (reads results.jsonl and summary.json)
    - Direct path to summary.json

    Computes extended statistics:
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
            self.config_file = self.exp_dir / "config.json"
        elif self.experiment_path.name == "summary.json":
            self.exp_dir = self.experiment_path.parent
            self.results_file = self.exp_dir / "results.jsonl"
            self.summary_file = self.experiment_path
            self.config_file = self.exp_dir / "config.json"
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
        self._config = None

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

    def load_config(self) -> Optional[Dict]:
        """
        Load config.json if it exists.

        Returns:
            Config dictionary or None if file doesn't exist
        """
        if self._config is not None:
            return self._config

        if not self.config_file.exists():
            return None

        with open(self.config_file, 'r') as f:
            self._config = json.load(f)

        return self._config

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
            Dictionary with extended statistics for val_loss, train_loss, and time:
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
        val_losses = np.array([r['final_val_loss'] for r in successful])
        train_losses = np.array([r['final_train_loss'] for r in successful])
        times = np.array([r['time_seconds'] for r in successful])

        # Compute statistics
        stats = {
            "n_runs": n_total,
            "successful_runs": n_success,
            "failed_runs": n_failed,
            "success_rate": float(success_rate),
            "val_loss": self._compute_metric_stats(val_losses, confidence_levels),
            "train_loss": self._compute_metric_stats(train_losses, confidence_levels),
            "time": self._compute_metric_stats(times, confidence_levels),
        }

        return stats

    def _compute_metric_stats(
        self,
        values: np.ndarray,
        confidence_levels: List[float]
    ) -> Dict:
        """
        Compute statistics for a single metric (val_loss, train_loss, or time).

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
            "std": float(np.std(values, ddof=1)) if n > 1 else 0.0,
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
        if n > 1:
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


def compare_experiments(baseline_path: str, modified_path: str) -> Dict:
    """
    Compare two experiments with statistical tests.

    Args:
        baseline_path: Path to baseline experiment
        modified_path: Path to modified experiment

    Returns:
        Dictionary with comparison results including:
        - Statistics for both experiments
        - Statistical test results (t-test, p-value)
        - Effect size (Cohen's d)
        - Improvement metrics
    """
    baseline = ExperimentAnalyzer(baseline_path)
    modified = ExperimentAnalyzer(modified_path)

    baseline_stats = baseline.compute_extended_statistics()
    modified_stats = modified.compute_extended_statistics()

    # Get raw values for statistical tests
    baseline_results = baseline.load_results()
    modified_results = modified.load_results()

    baseline_successful = [r for r in baseline_results if r.get('success', True)]
    modified_successful = [r for r in modified_results if r.get('success', True)]

    baseline_val_losses = np.array([r['final_val_loss'] for r in baseline_successful])
    modified_val_losses = np.array([r['final_val_loss'] for r in modified_successful])

    baseline_times = np.array([r['time_seconds'] for r in baseline_successful])
    modified_times = np.array([r['time_seconds'] for r in modified_successful])

    # Perform Welch's t-test (for unequal variances)
    t_stat, p_value = scipy_stats.ttest_ind(
        baseline_val_losses,
        modified_val_losses,
        equal_var=False  # Welch's t-test
    )

    # Compute Cohen's d effect size
    pooled_std = np.sqrt(
        (np.std(baseline_val_losses, ddof=1)**2 + np.std(modified_val_losses, ddof=1)**2) / 2
    )
    cohens_d = (np.mean(baseline_val_losses) - np.mean(modified_val_losses)) / pooled_std

    # Compute improvements (for val_loss, lower is better)
    val_loss_improvement = np.mean(baseline_val_losses) - np.mean(modified_val_losses)
    val_loss_improvement_pct = (val_loss_improvement / np.mean(baseline_val_losses)) * 100

    # Time comparison
    time_improvement = np.mean(baseline_times) - np.mean(modified_times)
    time_improvement_pct = (time_improvement / np.mean(baseline_times)) * 100 if np.mean(baseline_times) > 0 else 0

    comparison = {
        "baseline": {
            "name": baseline.get_experiment_name(),
            "stats": baseline_stats
        },
        "modified": {
            "name": modified.get_experiment_name(),
            "stats": modified_stats
        },
        "statistical_tests": {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant_at_001": p_value < 0.01,
            "significant_at_005": p_value < 0.05,
            "significant_at_010": p_value < 0.10
        },
        "improvements": {
            "val_loss_improvement": float(val_loss_improvement),
            "val_loss_improvement_pct": float(val_loss_improvement_pct),
            "time_improvement_seconds": float(time_improvement),
            "time_improvement_pct": float(time_improvement_pct),
            "is_better": val_loss_improvement > 0  # Positive means modified is better (lower loss)
        }
    }

    return comparison


def print_analysis_summary(stats: Dict, experiment_name: str = "Experiment"):
    """
    Print human-readable summary of experiment statistics.

    Args:
        stats: Statistics dictionary from compute_extended_statistics()
        experiment_name: Name of experiment for display
    """
    print(f"\n{'='*80}")
    print(f"Analysis Summary: {experiment_name}")
    print(f"{'='*80}\n")

    print(f"Runs: {stats['successful_runs']}/{stats['n_runs']} successful ({stats['success_rate']*100:.1f}%)\n")

    if 'val_loss' in stats:
        vl = stats['val_loss']
        print(f"Validation Loss:")
        print(f"  Mean:   {vl['mean']:.4f} ± {vl['std']:.4f}")
        print(f"  Median: {vl['median']:.4f}")
        print(f"  Range:  [{vl['min']:.4f}, {vl['max']:.4f}]")
        if 'confidence_intervals' in vl and 'ci_95' in vl['confidence_intervals']:
            ci = vl['confidence_intervals']['ci_95']
            print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print()

    if 'time' in stats:
        t = stats['time']
        print(f"Training Time:")
        print(f"  Mean:   {t['mean']:.2f}s ± {t['std']:.2f}s")
        print(f"  Total:  {t['mean'] * stats['successful_runs']:.2f}s ({t['mean'] * stats['successful_runs'] / 60:.2f} minutes)")
        print()


def print_comparison_summary(comparison: Dict):
    """
    Print human-readable comparison between two experiments.

    Args:
        comparison: Comparison dictionary from compare_experiments()
    """
    baseline = comparison['baseline']
    modified = comparison['modified']
    tests = comparison['statistical_tests']
    improvements = comparison['improvements']

    print(f"\n{'='*80}")
    print(f"Experiment Comparison")
    print(f"{'='*80}\n")

    print(f"Baseline: {baseline['name']}")
    print(f"Modified: {modified['name']}\n")

    # Validation loss comparison
    baseline_vl = baseline['stats']['val_loss']['mean']
    baseline_vl_std = baseline['stats']['val_loss']['std']
    modified_vl = modified['stats']['val_loss']['mean']
    modified_vl_std = modified['stats']['val_loss']['std']

    print(f"Validation Loss:")
    print(f"  Baseline: {baseline_vl:.4f} ± {baseline_vl_std:.4f}")
    print(f"  Modified: {modified_vl:.4f} ± {modified_vl_std:.4f}")
    print(f"  Change:   {improvements['val_loss_improvement']:+.4f} ({improvements['val_loss_improvement_pct']:+.2f}%)")

    if improvements['is_better']:
        print(f"  → Modified is BETTER (lower validation loss)")
    else:
        print(f"  → Modified is WORSE (higher validation loss)")
    print()

    # Statistical significance
    print(f"Statistical Tests:")
    print(f"  t-statistic:  {tests['t_statistic']:.3f}")
    print(f"  p-value:      {tests['p_value']:.4f}")
    print(f"  Cohen's d:    {tests['cohens_d']:.3f}")
    print()

    print(f"Significance:")
    if tests['significant_at_001']:
        print(f"  ✓ HIGHLY SIGNIFICANT (p < 0.01) - Meets NanoGPT speedrun requirements")
    elif tests['significant_at_005']:
        print(f"  ✓ Significant (p < 0.05) - But NOT sufficient for speedrun claims")
    elif tests['significant_at_010']:
        print(f"  ~ Marginally significant (p < 0.10)")
    else:
        print(f"  ✗ NOT significant (p >= 0.10)")
    print()

    # Effect size interpretation
    abs_d = abs(tests['cohens_d'])
    print(f"Effect Size (Cohen's d = {tests['cohens_d']:.3f}):")
    if abs_d < 0.2:
        print(f"  Negligible effect")
    elif abs_d < 0.5:
        print(f"  Small effect")
    elif abs_d < 0.8:
        print(f"  Medium effect")
    else:
        print(f"  Large effect")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze NanoGPT speedrun experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single experiment
  python scripts/analyze_results.py single experiment_logs/nanogpt_baseline_*/

  # Compare baseline vs modification
  python scripts/analyze_results.py compare \\
      --baseline experiment_logs/nanogpt_baseline_*/ \\
      --modified experiment_logs/nanogpt_lr_high_*/
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Single experiment analysis
    single_parser = subparsers.add_parser('single', help='Analyze a single experiment')
    single_parser.add_argument('experiment', type=str, help='Path to experiment directory')

    # Comparison
    compare_parser = subparsers.add_parser('compare', help='Compare two experiments')
    compare_parser.add_argument('--baseline', type=str, required=True, help='Path to baseline experiment')
    compare_parser.add_argument('--modified', type=str, required=True, help='Path to modified experiment')
    compare_parser.add_argument('--output', type=str, help='Optional output file for results (JSON)')

    args = parser.parse_args()

    if args.command == 'single':
        try:
            analyzer = ExperimentAnalyzer(args.experiment)
            stats = analyzer.compute_extended_statistics()
            experiment_name = analyzer.get_experiment_name()

            print_analysis_summary(stats, experiment_name)

        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == 'compare':
        try:
            comparison = compare_experiments(args.baseline, args.modified)

            print_comparison_summary(comparison)

            # Save to file if requested
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(comparison, f, indent=2)
                print(f"\nResults saved to: {output_path}")

        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
