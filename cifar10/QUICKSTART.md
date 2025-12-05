# Quick Start Checklist - ESE 3060 Final Project

Follow this checklist to get started with your experiments.

## 📁 File Structure

```
cifar10/
├── configs/               # Experiment configuration files
│   ├── baseline.json
│   ├── deterministic_translate.json
│   └── ablation_random_translate.json
├── experiments/           # Training code and variants
│   ├── airbench94.py     # Baseline training code
│   └── airbench94_deterministic_translate.py  # Example variant
├── experiment_logs/       # Results from runs (auto-generated)
├── analysis/             # Analysis scripts
├── run_cifar_experiment.py  # Main experiment runner
├── experiment_logger.py  # Logging utilities
└── test_setup.py         # Setup verification

../requirements.txt       # Dependencies (one level up)
```

## ✅ Pre-Flight Checklist

### Step 1: Clone and Setup
```bash
# Clone YOUR fork (not the original!)
git clone https://github.com/alainwelliver/deep-learning-speedrun-project.git
cd deep-learning-speedrun-project/cifar10

# Install dependencies
pip install -r ../requirements.txt

# Verify setup
python test_setup.py
```

### Step 2: Start Tmux Session
```bash
# Start a tmux session
tmux new -s baseline_exp

# Inside tmux, you can run long experiments
# Detach anytime with: Ctrl+B, then D
# Reattach with: tmux attach -t baseline_exp
```

### Step 3: Run Baseline Experiment
```bash
# Start with a small test (5 runs, ~30 seconds)
python run_cifar_experiment.py --config configs/baseline.json --n_runs 5

# If that works, run the full baseline (100 runs, ~7 minutes)
python run_cifar_experiment.py --config configs/baseline.json --n_runs 100
```

### Step 4: Check Results
```bash
# View the latest experiment log
ls -lt experiment_logs/
cat experiment_logs/cifar10_baseline_*/summary.json | python -m json.tool

# Should see something like:
# "mean_accuracy": 0.9401
# "std_accuracy": 0.0014
```

### Step 5: Commit and Push Results
```bash
# While still on runpod:
git add experiment_logs/
git commit -m "Baseline experiment: 94.01% ± 0.14% (n=100 runs)"
git push origin main

# Now your results are safely backed up!
```

---

## 🔬 Running Your First Modification

### Two Approaches:

**Approach A: Modify the baseline directly**
- Edit `experiments/airbench94.py`
- Use `--module airbench94` (default)

**Approach B: Create a new variant** (recommended for testing)
- Copy `experiments/airbench94.py` to `experiments/airbench94_mymod.py`
- Modify the new file
- Use `--module airbench94_mymod`
- Keeps baseline intact for comparison

### Step 1: Make Your Code Changes
```bash
# Option A: Edit the baseline directly
nano experiments/airbench94.py

# Option B: Create a new variant (recommended)
cp experiments/airbench94.py experiments/airbench94_mymod.py
nano experiments/airbench94_mymod.py
```

### Step 2: Create Config File
```bash
# Create a config for your experiment
# Use configs/example_config.json or configs/deterministic_translate.json as templates
cat > configs/my_modification.json << 'EOF'
{
  "experiment_name": "cifar10_my_modification",
  "hypothesis": "Your hypothesis here",
  "modification": "Description of what you changed",
  "description": "Detailed explanation of your approach",

  "base_seed": 42,
  "batch_size": 1024,
  "learning_rate": 11.5,
  "epochs": 9.9,
  "momentum": 0.85,
  "weight_decay": 0.0153,

  "target_accuracy": 0.94,
  "baseline_accuracy": null,
  "baseline_std": null,
  "baseline_n": 100
}
EOF
```

### Step 3: Commit Code BEFORE Running
```bash
# If you edited the baseline:
git add experiments/airbench94.py configs/my_modification.json
git commit -m "Add [your modification description]"

# If you created a new variant:
git add experiments/airbench94_mymod.py configs/my_modification.json
git commit -m "Add [your modification description]"
```

### Step 4: Run Experiment
```bash
# In tmux - Run with default module (experiments/airbench94.py):
python run_cifar_experiment.py --config configs/my_modification.json --n_runs 100

# Or run with a different module variant:
python run_cifar_experiment.py --config configs/deterministic_translate.json --module airbench94_deterministic_translate --n_runs 100
```

### Step 5: Commit Results
```bash
git add experiment_logs/
git commit -m "Results: [modification] - mean=XX.X%, std=X.XX%"
git push origin main
```

---

## 📋 Example Experiments Provided

The repository includes example configs you can run:

### 1. Baseline Experiment
```bash
python run_cifar_experiment.py --config configs/baseline.json --n_runs 100
```
Establishes baseline performance (~94% accuracy).

### 2. Deterministic Translation
```bash
python run_cifar_experiment.py --config configs/deterministic_translate.json --module airbench94_deterministic_translate --n_runs 100
```
Tests deterministic translation augmentation (similar to alternating flip).

### 3. Random Translation Control
```bash
python run_cifar_experiment.py --config configs/ablation_random_translate.json --module airbench94_deterministic_translate --n_runs 100
```
Control experiment with deterministic_translate=false to verify implementation.

---

## 📊 Analysis and Write-Up

### Analyze Results
```bash
# Use the provided analysis script
python analysis/analyze_experiments.py

# Or manually compare experiments
python -c "
import json
import glob

# Load all summaries
for summary_file in sorted(glob.glob('experiment_logs/*/summary.json')):
    with open(summary_file) as f:
        data = json.load(f)
        name = data['experiment_name']
        stats = data['statistics']
        acc = stats['accuracy']
        print(f'{name}: {acc[\"mean\"]:.4f} ± {acc[\"std\"]:.4f}')
"
```

### Create Visualizations
```bash
# The analysis/ directory contains scripts for visualization
cd analysis/
# Add your plotting scripts here

# Example: Plot accuracy distributions
python plot_results.py
```

### Write Report
- [ ] 1-page document (Part 1) or 2-page (Part 2)
- [ ] Hypothesis and motivation
- [ ] Description of modification
- [ ] Experimental methodology
- [ ] Results with statistics
- [ ] Plots and visualizations
- [ ] Discussion and conclusions

---

## 🆘 Troubleshooting

### **IMPORTANT: Always run commands from cifar10/ directory**
All commands in this guide assume you're in the `cifar10/` directory:
```bash
cd deep-learning-speedrun-project/cifar10
pwd  # Should show: .../deep-learning-speedrun-project/cifar10
```

### "Import Error: No module named 'airbench94'"
- Make sure you're in the cifar10/ directory (see above)
- Check that airbench94.py exists in the experiments/ folder
- The run_cifar_experiment.py automatically adds experiments/ to the Python path

### "CUDA not available"
- Make sure you're on a GPU instance
- Check: `nvidia-smi`

### "Accuracy much lower than expected"
- Check if your modification broke something
- Verify hyperparameters in config
- Try running baseline again

### Runpod disconnected / instance died
- This is why we use tmux!
- Reattach: `tmux attach -t baseline_exp`
- This is why we commit frequently!

### Git merge conflicts
```bash
# If you have conflicts:
git status
git diff
# Fix conflicts in the files
git add <fixed-files>
git commit
```

---

## 📝 Submission Checklist

Before submitting:
- [ ] All code is committed and pushed (experiments/, configs/, analysis/)
- [ ] Experiment logs are committed (experiment_logs/)
- [ ] README includes instructions to run your code
- [ ] Write-up PDF is complete (1 page Part 1, 2 pages Part 2)
- [ ] Code runs from cifar10/ directory with: `python run_cifar_experiment.py --config configs/your_config.json`
- [ ] GitHub repository is public or instructor has access
- [ ] Submit GitHub link + PDF write-up

**File Structure Check:**
```bash
# Your repo should have this structure:
deep-learning-speedrun-project/
├── requirements.txt
└── cifar10/
    ├── experiments/        # Your modified code here
    ├── configs/           # Your experiment configs here
    ├── experiment_logs/   # Results auto-saved here
    └── analysis/          # Analysis scripts here
```

---

## 🎯 Success Metrics

Part 1 (CIFAR-10):
- [ ] Baseline running successfully: ~94% accuracy
- [ ] At least one novel modification implemented
- [ ] 100+ runs for statistical significance
- [ ] Clear ablation study (tested variants)
- [ ] Write-up with hypothesis, results, analysis

Part 2 (NanoGPT):
- Similar but with train_gpt.py
- More expensive, so fewer runs may be acceptable
- Focus on one good idea rather than many mediocre ones

---

## 💡 Tips for Success

1. **Start early** - things will break, you'll need time to debug
2. **Test locally first** - use a small number of runs (n=5) to debug
3. **Commit obsessively** - every code change, every result
4. **Use tmux** - don't trust your SSH connection
5. **Read the paper** - the CIFAR speedrun paper has great ideas
6. **Ask questions** - use office hours, Ed Discussion
7. **Track compute usage** - screenshot your runpod billing
8. **Use the organized structure**:
   - Keep experiments in `experiments/` folder
   - Keep configs in `configs/` folder
   - All results auto-save to `experiment_logs/`
9. **Create variants, not copies** - Use `--module` flag to test different implementations
10. **Fill in baseline stats** - After running baseline, update your config files with actual baseline_accuracy and baseline_std

---

## 🔄 Workflow Summary

```bash
# 1. Enter cifar10 directory
cd deep-learning-speedrun-project/cifar10

# 2. Create/edit experiment code
nano experiments/airbench94_mymod.py

# 3. Create config
nano configs/mymod.json

# 4. Commit before running
git add experiments/airbench94_mymod.py configs/mymod.json
git commit -m "Add my modification"

# 5. Test with small run
python run_cifar_experiment.py --config configs/mymod.json --module airbench94_mymod --n_runs 5

# 6. Run full experiment in tmux
python run_cifar_experiment.py --config configs/mymod.json --module airbench94_mymod --n_runs 100

# 7. Commit results
git add experiment_logs/
git commit -m "Results: mymod - XX.XX% accuracy"
git push
```

Good luck! 🚀