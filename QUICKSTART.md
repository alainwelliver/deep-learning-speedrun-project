# Quick Start Checklist - ESE 3060 Final Project

Follow this checklist to get started with your experiments.

## ✅ Pre-Flight Checklist

### Step 1: Clone and Setup
```bash
# Clone YOUR fork (not the original!)
git clone https://github.com/YOUR-USERNAME/ese-3060-project.git
cd ese-3060-project

# Install dependencies
pip install -r requirements.txt

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

### Step 1: Make Your Code Changes
```bash
# Edit airbench94.py with your modification
# For example, add a residual connection

# Example: nano airbench94.py
# Or: vim airbench94.py
```

### Step 2: Create Config File
```bash
# Create a config for your experiment
cat > configs/my_modification.json << 'EOF'
{
  "experiment_name": "cifar10_my_modification",
  "hypothesis": "Your hypothesis here",
  "modification": "Description of what you changed",
  "base_seed": 42,
  "target_accuracy": 0.94,
  "baseline_accuracy": 0.9401,
  "baseline_std": 0.0014
}
EOF
```

### Step 3: Commit Code BEFORE Running
```bash
git add airbench94.py configs/my_modification.json
git commit -m "Add [your modification description]"
```

### Step 4: Run Experiment
```bash
# In tmux:
python run_cifar_experiment.py --config configs/my_modification.json --n_runs 100
```

### Step 5: Commit Results
```bash
git add experiment_logs/
git commit -m "Results: [modification] - mean=XX.X%, std=X.XX%"
git push origin main
```

---

## 📊 Analysis and Write-Up

### Analyze Results
```bash
# Compare experiments
python -c "
import json
import glob

# Load all summaries
for summary_file in glob.glob('experiment_logs/*/summary.json'):
    with open(summary_file) as f:
        data = json.load(f)
        name = data['experiment_name']
        stats = data['statistics']
        acc = stats['accuracy']
        print(f'{name}: {acc[\"mean\"]:.4f} ± {acc[\"std\"]:.4f}')
"
```

### Create Visualizations
```python
# Create plots/analysis script
# See examples in analysis/ directory
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

### "Import Error: No module named 'airbench94'"
- Make sure you're in the correct directory
- Check that airbench94.py exists in the current folder

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
- [ ] All code is committed and pushed
- [ ] Experiment logs are committed
- [ ] README includes instructions to run your code
- [ ] Write-up PDF is complete (1 page Part 1, 2 pages Part 2)
- [ ] Code runs with: `python run_cifar_experiment.py --config configs/your_config.json`
- [ ] GitHub repository is public or instructor has access
- [ ] Submit GitHub link + PDF write-up

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

Good luck! 🚀