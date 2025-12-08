#!/usr/bin/env python3
"""
compare_experiments.py

Generate comparison reports between baseline and modified NanoGPT experiments.
Creates markdown reports suitable for inclusion in writeups and academic reports.

Usage:
    python scripts/compare_experiments.py \\
        --baseline experiment_logs/nanogpt_baseline_*/ \\
        --modified experiment_logs/nanogpt_modified_*/ \\
        --output results/comparison_report.md
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Import from analyze_results.py
from analyze_results import ExperimentAnalyzer, compare_experiments


def generate_markdown_report(comparison: dict, output_path: str = None) -> str:
    """
    Generate a comprehensive markdown comparison report.

    Args:
        comparison: Comparison dictionary from compare_experiments()
        output_path: Optional path to save report to

    Returns:
        Markdown report as string
    """
    baseline = comparison['baseline']
    modified = comparison['modified']
    tests = comparison['statistical_tests']
    improvements = comparison['improvements']

    # Build markdown report
    lines = []

    # Title and metadata
    lines.append(f"# NanoGPT Experiment Comparison Report")
    lines.append(f"")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"")
    lines.append(f"**Baseline:** {baseline['name']}")
    lines.append(f"**Modified:** {modified['name']}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # Summary
    lines.append(f"## Summary")
    lines.append(f"")

    baseline_vl = baseline['stats']['val_loss']['mean']
    baseline_vl_std = baseline['stats']['val_loss']['std']
    modified_vl = modified['stats']['val_loss']['mean']
    modified_vl_std = modified['stats']['val_loss']['std']

    if improvements['is_better']:
        verdict = "✅ **IMPROVEMENT** - Modified experiment achieved lower validation loss"
    else:
        verdict = "❌ **REGRESSION** - Modified experiment had higher validation loss"

    lines.append(verdict)
    lines.append(f"")

    # Statistical significance verdict
    if tests['significant_at_001']:
        sig_verdict = "✅ **HIGHLY SIGNIFICANT (p < 0.01)** - Meets NanoGPT speedrun requirements"
    elif tests['significant_at_005']:
        sig_verdict = "⚠️ **SIGNIFICANT (p < 0.05)** - But NOT sufficient for speedrun claims (requires p < 0.01)"
    else:
        sig_verdict = "❌ **NOT SIGNIFICANT** - Difference could be due to random chance"

    lines.append(sig_verdict)
    lines.append(f"")

    # Key metrics table
    lines.append(f"## Key Metrics")
    lines.append(f"")
    lines.append(f"| Metric | Baseline | Modified | Change | Improvement |")
    lines.append(f"|--------|----------|----------|--------|-------------|")

    # Validation loss
    change_str = f"{improvements['val_loss_improvement']:+.4f}"
    pct_str = f"{improvements['val_loss_improvement_pct']:+.2f}%"
    lines.append(
        f"| **Validation Loss** | {baseline_vl:.4f} ± {baseline_vl_std:.4f} | "
        f"{modified_vl:.4f} ± {modified_vl_std:.4f} | {change_str} | {pct_str} |"
    )

    # Time
    baseline_time = baseline['stats']['time']['mean']
    baseline_time_std = baseline['stats']['time']['std']
    modified_time = modified['stats']['time']['mean']
    modified_time_std = modified['stats']['time']['std']
    time_change = improvements['time_improvement_seconds']
    time_pct = improvements['time_improvement_pct']

    lines.append(
        f"| **Training Time (s)** | {baseline_time:.2f} ± {baseline_time_std:.2f} | "
        f"{modified_time:.2f} ± {modified_time_std:.2f} | {time_change:+.2f} | {time_pct:+.2f}% |"
    )

    lines.append(f"")
    lines.append(f"*Note: For validation loss, negative change (lower loss) is better.*")
    lines.append(f"")

    # Statistical tests
    lines.append(f"## Statistical Analysis")
    lines.append(f"")
    lines.append(f"### Test Results")
    lines.append(f"")
    lines.append(f"| Test | Value |")
    lines.append(f"|------|-------|")
    lines.append(f"| t-statistic | {tests['t_statistic']:.3f} |")
    lines.append(f"| p-value | {tests['p_value']:.4f} |")
    lines.append(f"| Cohen's d | {tests['cohens_d']:.3f} |")
    lines.append(f"")

    lines.append(f"### Significance Levels")
    lines.append(f"")
    lines.append(f"| Level | Result |")
    lines.append(f"|-------|--------|")
    lines.append(f"| p < 0.01 | {'✅ YES' if tests['significant_at_001'] else '❌ NO'} |")
    lines.append(f"| p < 0.05 | {'✅ YES' if tests['significant_at_005'] else '❌ NO'} |")
    lines.append(f"| p < 0.10 | {'✅ YES' if tests['significant_at_010'] else '❌ NO'} |")
    lines.append(f"")

    # Effect size interpretation
    abs_d = abs(tests['cohens_d'])
    if abs_d < 0.2:
        effect_size = "Negligible"
    elif abs_d < 0.5:
        effect_size = "Small"
    elif abs_d < 0.8:
        effect_size = "Medium"
    else:
        effect_size = "Large"

    lines.append(f"### Effect Size")
    lines.append(f"")
    lines.append(f"Cohen's d = {tests['cohens_d']:.3f} indicates a **{effect_size}** effect size.")
    lines.append(f"")

    # Detailed statistics
    lines.append(f"## Detailed Statistics")
    lines.append(f"")

    # Baseline stats
    lines.append(f"### Baseline: {baseline['name']}")
    lines.append(f"")
    baseline_stats = baseline['stats']
    lines.append(f"**Runs:** {baseline_stats['successful_runs']}/{baseline_stats['n_runs']} successful")
    lines.append(f"")

    lines.append(f"**Validation Loss:**")
    lines.append(f"- Mean: {baseline_stats['val_loss']['mean']:.4f}")
    lines.append(f"- Std: {baseline_stats['val_loss']['std']:.4f}")
    lines.append(f"- Median: {baseline_stats['val_loss']['median']:.4f}")
    lines.append(f"- Min: {baseline_stats['val_loss']['min']:.4f}")
    lines.append(f"- Max: {baseline_stats['val_loss']['max']:.4f}")

    if 'confidence_intervals' in baseline_stats['val_loss']:
        if 'ci_95' in baseline_stats['val_loss']['confidence_intervals']:
            ci = baseline_stats['val_loss']['confidence_intervals']['ci_95']
            lines.append(f"- 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        if 'ci_99' in baseline_stats['val_loss']['confidence_intervals']:
            ci = baseline_stats['val_loss']['confidence_intervals']['ci_99']
            lines.append(f"- 99% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    lines.append(f"")

    # Modified stats
    lines.append(f"### Modified: {modified['name']}")
    lines.append(f"")
    modified_stats = modified['stats']
    lines.append(f"**Runs:** {modified_stats['successful_runs']}/{modified_stats['n_runs']} successful")
    lines.append(f"")

    lines.append(f"**Validation Loss:**")
    lines.append(f"- Mean: {modified_stats['val_loss']['mean']:.4f}")
    lines.append(f"- Std: {modified_stats['val_loss']['std']:.4f}")
    lines.append(f"- Median: {modified_stats['val_loss']['median']:.4f}")
    lines.append(f"- Min: {modified_stats['val_loss']['min']:.4f}")
    lines.append(f"- Max: {modified_stats['val_loss']['max']:.4f}")

    if 'confidence_intervals' in modified_stats['val_loss']:
        if 'ci_95' in modified_stats['val_loss']['confidence_intervals']:
            ci = modified_stats['val_loss']['confidence_intervals']['ci_95']
            lines.append(f"- 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        if 'ci_99' in modified_stats['val_loss']['confidence_intervals']:
            ci = modified_stats['val_loss']['confidence_intervals']['ci_99']
            lines.append(f"- 99% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    lines.append(f"")

    # Interpretation
    lines.append(f"## Interpretation")
    lines.append(f"")

    if improvements['is_better'] and tests['significant_at_001']:
        lines.append(f"The modified experiment shows a **statistically significant improvement** "
                     f"over the baseline (p < 0.01). The validation loss decreased by "
                     f"{abs(improvements['val_loss_improvement']):.4f} "
                     f"({abs(improvements['val_loss_improvement_pct']):.2f}%), "
                     f"which meets the requirements for NanoGPT speedrun claims.")
    elif improvements['is_better'] and tests['significant_at_005']:
        lines.append(f"The modified experiment shows improvement over the baseline with p < 0.05, "
                     f"but this does **NOT meet the p < 0.01 requirement** for NanoGPT speedrun claims. "
                     f"Additional runs may be needed for stronger statistical evidence.")
    elif improvements['is_better']:
        lines.append(f"While the modified experiment has lower mean validation loss, "
                     f"the difference is **not statistically significant** (p = {tests['p_value']:.4f}). "
                     f"This could be due to random variation. More runs are recommended.")
    else:
        lines.append(f"The modified experiment performed **worse** than the baseline, "
                     f"with higher validation loss. This modification should not be used.")

    lines.append(f"")

    # Methodology
    lines.append(f"## Methodology")
    lines.append(f"")
    lines.append(f"- **Statistical Test:** Welch's t-test (unequal variances)")
    lines.append(f"- **Effect Size:** Cohen's d")
    lines.append(f"- **Confidence Intervals:** t-distribution based")
    lines.append(f"- **Significance Threshold:** p < 0.01 (NanoGPT speedrun requirement)")
    lines.append(f"")

    # Footer
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"*Report generated by NanoGPT experiment infrastructure*")
    lines.append(f"")

    # Join all lines
    report = "\n".join(lines)

    # Save to file if requested
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_file}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Generate comparison report between NanoGPT experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/compare_experiments.py \\
      --baseline experiment_logs/nanogpt_baseline_*/ \\
      --modified experiment_logs/nanogpt_lr_high_*/ \\
      --output results/comparison_report.md
        """
    )

    parser.add_argument(
        '--baseline',
        type=str,
        required=True,
        help='Path to baseline experiment directory'
    )
    parser.add_argument(
        '--modified',
        type=str,
        required=True,
        help='Path to modified experiment directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for markdown report'
    )

    args = parser.parse_args()

    try:
        # Run comparison
        print(f"Comparing experiments...")
        print(f"  Baseline: {args.baseline}")
        print(f"  Modified: {args.modified}")
        print()

        comparison = compare_experiments(args.baseline, args.modified)

        # Generate markdown report
        report = generate_markdown_report(comparison, args.output)

        # Also print to console
        print()
        print("="*80)
        print("COMPARISON REPORT")
        print("="*80)
        print()
        print(report)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
