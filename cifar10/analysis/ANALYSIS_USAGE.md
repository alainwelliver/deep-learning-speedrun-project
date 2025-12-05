# analyze_experiments.py - Usage Guide

Comprehensive statistics analysis tool for CIFAR-10 speedrun experiments.

## Quick Start

### 1. Analyze a Single Experiment

```bash
# View formatted summary
python analyze_experiments.py single experiment_logs/cifar10_baseline_20251205_222844/

# Output as JSON
python analyze_experiments.py single experiment_logs/cifar10_baseline_20251205_222844/ --json
```

**Output includes:**
- Mean, std, median, min, max
- 95% and 99% confidence intervals
- Percentiles (25th, 50th, 75th, 90th, 95th, 99th)
- Coefficient of variation
- Success rate
- Experiment metadata (git commit, GPU info, config)

### 2. Compare Two Experiments

```bash
# Compare modified experiment to baseline
python analyze_experiments.py compare \
    --baseline experiment_logs/cifar10_baseline_20251205_222844/ \
    --modified experiment_logs/cifar10_modified_*/ \
    --metric accuracy

# Compare training times
python analyze_experiments.py compare \
    --baseline experiment_logs/cifar10_baseline_*/ \
    --modified experiment_logs/cifar10_optimized_*/ \
    --metric time
```

**Output includes:**
- Percentage improvement
- Statistical significance (p-value)
- Automatic test selection (Welch's t-test, Mann-Whitney U, etc.)
- Effect size (Cohen's d)

### 3. Use as Python Module

```python
from analyze_experiments import (
    ExperimentAnalyzer,
    ExperimentComparator,
    compare_to_baseline,
    quick_summary,
    export_csv,
    export_latex_table,
    plot_distribution,
    plot_comparison
)

# Analyze single experiment
analyzer = ExperimentAnalyzer("experiment_logs/cifar10_baseline_20251205_222844/")
stats = analyzer.compute_extended_statistics()

print(f"Mean accuracy: {stats['accuracy']['mean']:.4f}")
print(f"99% CI: {stats['accuracy']['confidence_intervals']['ci_99']}")
print(f"95th percentile: {stats['accuracy']['percentiles']['p95']:.4f}")

# Quick summary to console
quick_summary("experiment_logs/cifar10_baseline_20251205_222844/")

# Compare to baseline
result = compare_to_baseline(
    baseline_path="experiment_logs/cifar10_baseline_20251205_222844/",
    modified_path="experiment_logs/cifar10_modified_*/",
    metric='accuracy'
)

print(f"Improvement: {result['improvement_pct']:.2f}%")
print(f"P-value: {result['p_value']:.4f}")
print(f"Significant: {result['significant']}")
print(f"Effect size: {result['cohens_d']:.3f} ({result['effect_size_interpretation']})")
```

### 4. Advanced Comparison

```python
# Compare multiple experiments
baseline = ExperimentAnalyzer("experiment_logs/baseline_*/")
modified1 = ExperimentAnalyzer("experiment_logs/lr_tuned_*/")
modified2 = ExperimentAnalyzer("experiment_logs/augmented_*/")

comparator = ExperimentComparator({
    'Baseline': baseline,
    'LR Tuned': modified1,
    'Augmented': modified2
})

# Check statistical assumptions
assumptions = comparator.check_assumptions(metric='accuracy')
print("Normality test results:", assumptions['normality'])
print("Equal variances:", assumptions['equal_variances'])

# Pairwise tests
tests = comparator.compute_pairwise_tests(metric='accuracy', test='auto')
for pair, result in tests.items():
    print(f"{pair}:")
    print(f"  Test: {result['test_used']}")
    print(f"  P-value: {result['p_value']:.4f}")
    print(f"  Significant: {result['significant']}")

# Effect sizes
effects = comparator.compute_effect_sizes(metric='accuracy')
for pair, result in effects.items():
    print(f"{pair}: Cohen's d = {result['cohens_d']:.3f} ({result['interpretation']})")

# Rank experiments
rankings = comparator.rank_experiments(metric='accuracy')
for rank, (name, mean, std) in enumerate(rankings, 1):
    print(f"{rank}. {name}: {mean:.4f} ± {std:.4f}")
```

### 5. Visualization

```python
from analyze_experiments import plot_distribution, plot_comparison

# Single experiment distribution (violin plot)
plot_distribution(
    "experiment_logs/cifar10_baseline_20251205_222844/",
    metric='accuracy',
    kind='violin',
    output_path='figures/baseline_distribution.pdf',
    show=False  # Don't display, just save
)

# Box plot
plot_distribution(
    "experiment_logs/cifar10_baseline_20251205_222844/",
    metric='accuracy',
    kind='box',
    output_path='figures/baseline_boxplot.pdf',
    show=False
)

# Histogram
plot_distribution(
    "experiment_logs/cifar10_baseline_20251205_222844/",
    metric='accuracy',
    kind='hist',
    output_path='figures/baseline_histogram.pdf',
    show=False
)

# Compare multiple experiments
experiments = {
    'Baseline': 'experiment_logs/cifar10_baseline_20251205_222844/',
    'Modified': 'experiment_logs/cifar10_modified_*/'
}

plot_comparison(
    experiments,
    metric='accuracy',
    output_path='figures/accuracy_comparison.pdf',
    show=False
)
```

### 6. Export for Reports

```python
from analyze_experiments import export_csv, export_latex_table

experiments = {
    'Baseline': 'experiment_logs/cifar10_baseline_20251205_222844/',
    'LR Tuned': 'experiment_logs/cifar10_lr_tuned_*/',
    'Augmented': 'experiment_logs/cifar10_augmented_*/'
}

# Export to CSV
export_csv(
    experiments,
    output_path='results/experiment_stats.csv',
    include_raw_data=True  # Also exports raw per-run data
)

# Export LaTeX table
export_latex_table(
    experiments,
    output_path='tables/results.tex',
    metrics=['accuracy', 'time'],
    caption='Comparison of CIFAR-10 Training Modifications',
    label='tab:cifar_results'
)
```

The LaTeX table can be included in your report:
```latex
\usepackage{booktabs}

% In your document:
Table \ref{tab:cifar_results} shows the results of our experiments.

\input{tables/results.tex}
```

## Extended Statistics Explained

### Confidence Intervals
- **95% CI**: 95% probability that true mean lies in this range
- **99% CI**: 99% probability (wider, more conservative)

### Percentiles
- **25th/75th**: Interquartile range (IQR) - middle 50% of data
- **90th**: 90% of runs achieved this or better
- **95th/99th**: Top performers, useful for understanding variability

### Coefficient of Variation (CV)
- Ratio of std to mean: `CV = std / mean`
- Dimensionless measure of relative variability
- Lower CV = more consistent results

### Effect Size (Cohen's d)
- Standardized measure of difference between means
- Interpretation:
  - Negligible: |d| < 0.2
  - Small: 0.2 ≤ |d| < 0.5
  - Medium: 0.5 ≤ |d| < 0.8
  - Large: |d| ≥ 0.8

### Statistical Tests
The script automatically selects appropriate tests:
- **Independent t-test**: Normal data, equal variances
- **Welch's t-test**: Normal data, unequal variances
- **Mann-Whitney U**: Non-normal data or small samples (n < 30)

## Example Output

### Single Experiment Summary
```
================================================================================
EXPERIMENT SUMMARY: cifar10_baseline
================================================================================

Runs: 100 total

ACCURACY STATISTICS:
  Mean:       0.940137 ± 0.001367 (std)
  Median:     0.940150
  Range:      [0.936800, 0.944400]
  CV:         0.0015

  Confidence Intervals:
    95% CI:  [0.939866, 0.940408]
    99% CI:  [0.939778, 0.940496]

  Percentiles:
    25th: 0.939175  |  50th: 0.940150  |  75th: 0.941000
    90th: 0.941600  |  95th: 0.942405  |  99th: 0.943410

TIME STATISTICS:
  Mean:       4.3848s ± 0.0307s (std)
  Median:     4.3966s
  Range:      [4.2824s, 4.4357s]
  Total:      438.48s (7.31 min, 0.12 hr)

EXPERIMENT METADATA:
  Git Commit: f9382c562720
  Branch:     main
  GPU:        NVIDIA A100 80GB PCIe (85.1 GB)
  Config:     configs/baseline.json
================================================================================
```

### Comparison Output
```
================================================================================
BASELINE vs MODIFIED COMPARISON (ACCURACY)
================================================================================

Baseline mean:  0.940137
Modified mean:  0.942500
Improvement:    +0.25%

Statistical Test: Welch's t-test
P-value:          0.021456
Result:           STATISTICALLY SIGNIFICANT (p < 0.05)

Effect Size:
  Cohen's d:      0.3421
  Interpretation: small

================================================================================
```

## Tips for Your Final Report

1. **Baseline Statistics**: Use 95% CI and percentiles to show consistency
   ```
   Our baseline achieved 94.01% ± 0.14% accuracy (95% CI: [93.99%, 94.04%])
   across 100 runs, demonstrating high reproducibility.
   ```

2. **Comparison**: Report both statistical significance and effect size
   ```
   The modified approach achieved 94.25% accuracy, a +0.25% improvement
   over baseline (p=0.021, Cohen's d=0.34), which is statistically
   significant with a small effect size.
   ```

3. **Visualizations**: Use violin plots to show distribution shape
   - Shows full distribution, not just mean/std
   - Reveals multimodality if present
   - More informative than bar charts

4. **Table**: Use LaTeX export for professional formatting
   - Consistent number formatting
   - Proper scientific notation
   - Easy to update if you rerun experiments

## Troubleshooting

**"No such file or directory"**: Make sure glob pattern is in quotes
```bash
# Good
python analyze_experiments.py single "experiment_logs/cifar10_baseline_*/"

# Bad (shell expands wildcard)
python analyze_experiments.py single experiment_logs/cifar10_baseline_*/
```

**"Visualization libraries not available"**: Install matplotlib and seaborn
```bash
pip install matplotlib seaborn pandas
```

**"Too few samples for normality test"**: Need at least 3 runs for statistical tests. Use more runs for reliable statistics (20+ recommended, 100 ideal).
