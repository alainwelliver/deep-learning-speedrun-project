# NanoGPT Speedrun Experiments

Experiment infrastructure for ESE 3060 Final Project Part 2 - NanoGPT training speedrun.

**Goal**: Train GPT-2 (124M parameters) to ≤3.28 validation loss on FineWeb as fast as possible on 8x H100 GPUs.

## Setup

### 1. Install Dependencies

```bash
pip install torch numpy scipy
```

### 2. Download Dataset

For quick testing (first 900M tokens):
```bash
cd ..  # Go to project root
python data/cached_fineweb10B.py 9
```

For full dataset (needed for actual runs):
```bash
python data/cached_fineweb10B.py 8
```

This will download data to `data/fineweb10B/fineweb_train_*.bin` and `fineweb_val_*.bin`.

### 3. Verify Installation

Test that the experiment logger works:
```bash
cd nanogpt
python experiment_logger.py
```

## Quick Start

### Debug Mode (Single GPU, Single Run)

Test the infrastructure without using expensive multi-GPU resources:

```bash
python scripts/run_nanogpt_experiment.py \
    --config configs/baseline.json \
    --n_runs 1 \
    --n_gpus 1
```

This runs a single training run on 1 GPU for testing purposes.

### Dry Run

Preview the commands that will be executed without actually running:

```bash
python scripts/run_nanogpt_experiment.py \
    --config configs/baseline.json \
    --dry_run
```

### Full Baseline Run (8 GPUs, 5 Runs)

Run the full baseline experiment with statistical significance:

```bash
python scripts/run_nanogpt_experiment.py \
    --config configs/baseline.json \
    --n_runs 5 \
    --n_gpus 8
```

**Note**: Each run takes approximately 7 minutes on 8x H100 GPUs, so 5 runs will take ~35 minutes total.

## 3-Stage Experimental Workflow

For budget-constrained exploration, use the **staged workflow** to efficiently screen multiple variants before committing to expensive full runs.

### Overview

The staged approach progressively filters unpromising variants through 3 stages:

- **Stage A (Screening)**: Quick, cheap runs to identify promising variants
- **Stage B (Validation)**: Medium-length runs to confirm effects hold at scale
- **Stage C (Full)**: Complete runs for statistical significance and final results

**Budget Savings**: ~43% cost reduction vs. running all variants with full runs.

### Stage Specifications

| Stage | Iterations | GPUs | Cost/Run | Runs/Variant | Purpose |
|-------|-----------|------|----------|--------------|---------|
| **A** | 1000 (~1/5) | 1 L40 | ~$2 | 2 | Screen all variants |
| **B** | 3000 (~3/5) | 4 A100s | ~$10 | 3 | Validate best variant |
| **C** | 5100 (full) | 8 H100s | ~$8-9 | 5 | Final statistical proof |

### Directory Structure

All configs are organized by stage:

```
configs/
├── stage_a_screening/
│   ├── baseline_screening.json
│   ├── palm_screening.json
│   ├── curriculum_screening.json
│   └── depth_warm_screening.json
├── stage_b_validation/
│   ├── baseline_validation.json
│   ├── palm_validation.json
│   ├── curriculum_validation.json
│   └── depth_warm_validation.json
└── stage_c_full/
    ├── baseline.json
    ├── palm_parallel.json
    ├── curriculum_full.json
    └── depth_warm.json
```

### How to Run

#### Stage A: Screen All Variants

Run all 4 variants with short, cheap runs to identify the best performer(s):

```bash
cd nanogpt

# Screen baseline
python run_nanogpt_experiment.py \
    --config configs/stage_a_screening/baseline_screening.json \
    --n_runs 2

# Screen PaLM parallel architecture
python run_nanogpt_experiment.py \
    --config configs/stage_a_screening/palm_screening.json \
    --n_runs 2

# Screen curriculum learning
python run_nanogpt_experiment.py \
    --config configs/stage_a_screening/curriculum_screening.json \
    --n_runs 2

# Screen DepthWarm
python run_nanogpt_experiment.py \
    --config configs/stage_a_screening/depth_warm_screening.json \
    --n_runs 2
```

**Time**: ~30 minutes per variant (2 runs × 2 min/run × some overhead)
**Cost**: ~$4 per variant, ~$16 total for all 4
**Output**: 4 experiment directories in `experiment_logs/`

#### Compare Stage A Results

After all Stage A runs complete, compare validation losses:

```bash
# Check mean validation loss for each variant
for dir in experiment_logs/nanogpt_*_screening_*/; do
    echo "=== $(basename $dir) ==="
    jq '.statistics.val_loss.mean' "$dir/summary.json"
done
```

Look for:
- **Best mean validation loss** (lowest is best)
- **Low variance** (smaller std is better)
- **Promising trends** in training curves

#### Stage B: Validate Best Variant

Once you've identified the best 1-2 variants from Stage A, run Stage B to confirm the effect holds with more training:

```bash
# Replace 'palm' with your best Stage A variant
python run_nanogpt_experiment.py \
    --config configs/stage_b_validation/palm_validation.json \
    --n_runs 3 \
    --n_gpus 4
```

**Time**: ~15 minutes (3 runs × 5 min/run)
**Cost**: ~$30
**Output**: Single experiment directory

**Decision Point**: If Stage B results look promising (lower val loss than baseline, consistent improvement), proceed to Stage C. If Stage B looks bad, pivot to your next-best Stage A variant.

#### Stage C: Final Statistical Proof

If Stage B confirms your variant is promising, run the full experiment for publication-quality statistics:

```bash
# Run full experiment on best variant
python run_nanogpt_experiment.py \
    --config configs/stage_c_full/palm_parallel.json \
    --n_runs 5 \
    --n_gpus 8
```

**Time**: ~35 minutes (5 runs × 7 min/run)
**Cost**: ~$42
**Output**: Final results with statistical significance

### Example Workflow

Here's a complete workflow showing how to use the staging approach:

```bash
# 1. Run Stage A on all 4 variants (~1 hour total, ~$16)
for variant in baseline palm curriculum depth_warm; do
    python run_nanogpt_experiment.py \
        --config configs/stage_a_screening/${variant}_screening.json \
        --n_runs 2
done

# 2. Analyze Stage A results
echo "Stage A Results Summary:"
for dir in experiment_logs/nanogpt_*_screening_*/; do
    variant=$(basename "$dir" | cut -d_ -f2)
    mean=$(jq -r '.statistics.val_loss.mean' "$dir/summary.json")
    std=$(jq -r '.statistics.val_loss.std' "$dir/summary.json")
    echo "$variant: $mean ± $std"
done

# 3. Manually identify best variant (let's say it's 'palm')
# Run Stage B validation
python run_nanogpt_experiment.py \
    --config configs/stage_b_validation/palm_validation.json \
    --n_runs 3 \
    --n_gpus 4

# 4. Check Stage B results
cat experiment_logs/nanogpt_palm_validation_*/summary.json | jq '.statistics.val_loss'

# 5. If promising, proceed to Stage C
python run_nanogpt_experiment.py \
    --config configs/stage_c_full/palm_parallel.json \
    --n_runs 5 \
    --n_gpus 8

# 6. Final results
cat experiment_logs/nanogpt_palm_parallel_*/summary.json | jq '.statistics'
```

### Decision Criteria

**Stage A → Stage B**:
- Pick the variant(s) with **lowest mean validation loss**
- Consider **variance** - lower std is more reliable
- Can advance 1-2 variants if multiple look promising
- Budget allows for 1 pivot if first choice fails

**Stage B → Stage C**:
- Validate that **mean val loss < baseline** (or target threshold)
- Check that **trend is consistent** across runs
- Ensure **CI doesn't overlap** with baseline too much
- If Stage B fails, pivot to next-best Stage A variant

**Pivoting in Stage B**:
If your first Stage B choice performs poorly, you have budget to try one more:

```bash
# If palm failed in Stage B, try curriculum
python run_nanogpt_experiment.py \
    --config configs/stage_b_validation/curriculum_validation.json \
    --n_runs 3 \
    --n_gpus 4
```

### Budget Summary

**Total budget**: ~$95 (fits in $96 constraint)

- Stage A: 4 variants × 2 runs × $2 = **$16**
- Stage B: 1 variant × 3 runs × $10 = **$30**
- Stage C: 1 variant × 5 runs × $8.40 = **$42**
- **Reserve**: $7 for potential Stage B pivot

**vs. Naive approach**: Running all 4 variants fully would cost ~$168 (4 × $42)

**Savings**: 43% cost reduction

### How Stage Configs Work

The staging system uses **environment variable overrides** to modify hyperparameters without duplicating training scripts:

**Stage A config example** (`baseline_screening.json`):
```json
{
  "experiment_name": "nanogpt_baseline_screening",
  "script": "experiments/train_gpt.py",
  "stage": "A_screening",
  "n_gpus": 1,
  "stage_config": {
    "num_iterations": 1000,
    "warmdown_iters": 284,
    "batch_size": 128,
    "device_batch_size": 16,
    "val_loss_every": 25
  }
}
```

The `stage_config` section is passed to the training script via environment variables (`STAGE_NUM_ITERATIONS`, etc.), overriding the default values.

### Interpreting Partial Runs

**Stage A (1000 iters)**: Loss will be higher than full training, but relative rankings should hold. Focus on **comparing between variants**, not absolute loss values.

**Stage B (3000 iters)**: Loss should be closer to final values (~85-90% of full training). More reliable for predicting Stage C outcomes.

**Extrapolating**: If Stage A shows variant X is 0.05 better than baseline at 1000 iters, it's a good sign, but not guaranteed to hold at 5100 iters. Stage B confirms whether the advantage persists.

### Variant Descriptions

Your 4 variants to test:

1. **Baseline** (`baseline_screening.json`)
   - Unmodified `train_gpt.py`
   - Reference point for comparisons

2. **PaLM Parallel** (`palm_screening.json`)
   - Parallel attention + MLP blocks
   - Shared RMSNorm, 1/√2 scaling
   - Architecture modification

3. **Curriculum Learning** (`curriculum_screening.json`)
   - Progressive sequence cropping (256→512→768→1024)
   - Loss plateau detection
   - Training methodology modification

4. **DepthWarm** (`depth_warm_screening.json`)
   - Progressive layer activation
   - Stochastic depth warmup
   - Architecture modification

### Tips

- **Run Stage A overnight** to save time
- **Use `--dry_run`** to verify configs before expensive runs
- **Monitor first run** of each stage to catch issues early
- **Save experiment directories** - you'll need them for analysis
- **Git commit** before each stage for reproducibility
- **Track costs** to avoid budget overruns

## Configuration Files

### Baseline Configuration (`configs/baseline.json`)

```json
{
  "experiment_name": "nanogpt_baseline",
  "description": "Baseline NanoGPT training - unmodified train_gpt.py",
  "script": "train_gpt.py",
  "n_gpus": 8,
  "base_seed": 42,
  "target_val_loss": 3.28,
  "modification_type": "none",
  "hyperparameters": {
    "notes": "Using default hyperparameters from train_gpt.py"
  }
}
```

### Creating Modified Configurations

To test a modification:

1. Create a modified version of the training script (e.g., `train_gpt_modified.py`)
2. Edit hyperparameters or make code changes in the new file
3. Create a new config file pointing to it:

```json
{
  "experiment_name": "nanogpt_lr_high",
  "description": "Higher learning rate experiment",
  "script": "train_gpt_modified.py",
  "n_gpus": 8,
  "base_seed": 42,
  "modification_type": "hyperparameter",
  "modification_details": "Increased learning_rate from 0.0036 to 0.005",
  "target_val_loss": 3.28,
  "baseline_val_loss": 3.30,
  "baseline_std": 0.02
}
```

4. Run the experiment:
```bash
python scripts/run_nanogpt_experiment.py --config configs/your_config.json --n_runs 5
```

## Experiment Outputs

Each experiment creates a timestamped directory in `experiment_logs/`:

```
experiment_logs/
└── nanogpt_baseline_20241207_143022/
    ├── experiment.log           # Human-readable log
    ├── config.json              # Configuration used
    ├── results.jsonl           # Per-run final results
    ├── metrics.jsonl           # Per-step training metrics
    ├── summary.json            # Statistical summary
    ├── git_diff.patch          # Git diff if repo was dirty
    └── runs/                   # Per-run directories
        ├── run_0_seed_42/
        │   └── stdout.log      # Full stdout from training
        ├── run_1_seed_43/
        │   └── stdout.log
        └── ...
```

### File Formats

**results.jsonl** - One JSON object per line, one per run:
```json
{"run_id": 0, "seed": 42, "final_val_loss": 3.275, "final_train_loss": 2.891, "time_seconds": 420.5, "final_step": 5100, "success": true}
{"run_id": 1, "seed": 43, "final_val_loss": 3.282, "final_train_loss": 2.895, "time_seconds": 418.2, "final_step": 5100, "success": true}
```

**metrics.jsonl** - Per-step metrics for training curves:
```json
{"run_id": 0, "step": 0, "val_loss": 10.123, "train_time_ms": 0, "step_avg_ms": 0.0}
{"run_id": 0, "step": 1, "train_loss": 9.876, "train_time_ms": 120, "step_avg_ms": 120.0}
{"run_id": 0, "step": 125, "val_loss": 5.234, "train_time_ms": 15000, "step_avg_ms": 120.0}
```

**summary.json** - Statistical summary with git/GPU metadata:
```json
{
  "experiment_name": "nanogpt_baseline",
  "end_time": "2024-12-07T14:35:22.123456",
  "statistics": {
    "n_runs": 5,
    "successful_runs": 5,
    "val_loss": {
      "mean": 3.278,
      "std": 0.015,
      "min": 3.265,
      "max": 3.295,
      "median": 3.275,
      "ci_95": [3.260, 3.296]
    },
    "time": {
      "mean": 419.5,
      "std": 12.3,
      "total": 2097.5
    }
  },
  "git_info": { ... },
  "gpu_info": { ... }
}
```

## Analyzing Results

### View Summary

The summary is printed at the end of each experiment run:

```
Summary Statistics:
  Runs: 5/5
  Mean Validation Loss: 3.278 ± 0.015
  95% CI: [3.260, 3.296]
  Best Validation Loss: 3.265
```

### Read Summary JSON

```bash
cat experiment_logs/nanogpt_baseline_*/summary.json | jq '.statistics.val_loss'
```

### Compare Experiments

To compare baseline vs modification, look at the mean validation loss and confidence intervals:

```bash
# Baseline
cat experiment_logs/nanogpt_baseline_*/summary.json | jq '.statistics.val_loss.mean'

# Your modification
cat experiment_logs/nanogpt_your_mod_*/summary.json | jq '.statistics.val_loss.mean'
```

For statistical testing (Phase 2 - if time permits), use `scripts/analyze_results.py` and `scripts/compare_experiments.py`.

## Statistical Requirements

From the NanoGPT speedrun rules:

- **Significance threshold**: p < 0.01 for claims about validation loss improvements
- **Recommended runs**: 5-10 for statistical power
- **Report**: Mean ± standard deviation, 95% confidence intervals
- **Target**: Mean validation loss ≤ 3.28 with p < 0.01

## Troubleshooting

### "torchrun: command not found"

Make sure PyTorch is installed with distributed support:
```bash
pip install torch torchvision torchaudio
```

### "Training script not found"

Make sure you're running from the project root or the nanogpt/ directory, and that `train_gpt.py` exists in the nanogpt/ directory.

### "No validation loss captured"

This can happen if:
- Training crashed before first validation
- Output format changed (check regex patterns in run_nanogpt_experiment.py)
- Check the stdout.log in the run directory for actual output

### Out of memory errors

If running on single GPU for debugging and hitting OOM:
- The baseline uses batch_size=512 which requires 8 GPUs
- Single GPU runs may need reduced batch size
- Consider modifying train_gpt.py to reduce batch_size for debugging

## Development Workflow

1. **Test locally** with `--dry_run` to verify configuration
2. **Debug** with `--n_runs 1 --n_gpus 1` on cheaper instance
3. **Run baseline** with full settings on 8x H100
4. **Create modification** by copying and editing train_gpt.py
5. **Run experiment** with new config pointing to modified script
6. **Compare results** using summary.json files
7. **Iterate** based on results

## File Reference

- [experiment_logger.py](experiment_logger.py) - Logging infrastructure
- [scripts/run_nanogpt_experiment.py](scripts/run_nanogpt_experiment.py) - Main experiment runner
- [configs/baseline.json](configs/baseline.json) - Baseline configuration
- [train_gpt.py](train_gpt.py) - Original training script (do not modify)
- `train_gpt_modified.py` - Create this for your modifications

## Tips

- Always run baseline first to establish statistical baseline
- Use `--dry_run` to test configurations before expensive runs
- Keep git repo clean (or at least committed) before runs for reproducibility
- Check `experiment.log` for detailed execution logs
- Per-step metrics in `metrics.jsonl` can be used for training curve plots
- Each run uses seed = base_seed + run_id for reproducibility

## Next Steps (Phase 2 - Analysis Tools)

If time permits, implement:
- `scripts/analyze_results.py` - Statistical analysis and comparison
- `scripts/compare_experiments.py` - Generate comparison reports
- Training curve plotting from metrics.jsonl
