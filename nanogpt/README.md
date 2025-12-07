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
