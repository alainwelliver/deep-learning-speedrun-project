# Deep Learning Training Speedup Research

**Author:** Alain Welliver
**Project:** ESE 3060 Final Project, Fall 2025

## Overview

This repository contains rigorous experimental research on accelerating deep learning training pipelines through novel modifications to data augmentation and model architecture. The project comprises two independent studies:

1. **CIFAR-10 Fast Training**: Deterministic cropping strategies for 2.8% speedup on image classification
2. **NanoGPT Training Acceleration**: PaLM-style parallel transformer blocks for 2.66% speedup on language model training

Both studies employ multi-stage experimental methodology with statistical rigor, achieving measurable performance improvements validated across hundreds to thousands of training runs.

---

## Project 1: CIFAR-10 Deterministic Cropping

📄 **Paper:** [CIFAR10_SpeedUp_Final_Paper.pdf](CIFAR10_SpeedUp_Final_Paper.pdf)

### Abstract

Building on the success of deterministic augmentation techniques (alternating flip), this work explores derandomizing translation augmentation to reduce training time while maintaining model accuracy. By cycling through crop positions deterministically using `hash(image_index) + epoch`, each training image experiences all 25 possible crop positions systematically, maximizing augmentation diversity without redundancy. A backoff mechanism reverts to random translation for the final 40% of training to improve convergence.

### Key Results

- **2.8% training speedup** (statistically significant, p < 0.001)
- **0.064% accuracy decrease** (minimal trade-off)
- **2,400 total training runs** across multiple A100 instances for statistical confidence
- Validation across two independent hardware instances for robustness

**Use Case:** Speed-focused scenarios where marginal accuracy cost is acceptable, such as hyperparameter sweeps or rapid prototyping.

### Quick Start

```bash
cd cifar10

# Install dependencies
pip install -r ../requirements.txt

# Run baseline experiment (100 runs, ~7 minutes on A100)
python run_cifar_experiment.py \
    --config configs/baseline.json \
    --n_runs 100

# Run deterministic cropping with backoff (100 runs)
python run_cifar_experiment.py \
    --config configs/deterministic_translate_backoff.json \
    --module airbench94_deterministic_with_backoff \
    --n_runs 100
```

**Hardware Requirements:**
- NVIDIA A100 GPU (or similar)
- CUDA 11.7+
- ~5 seconds per training run

### Key Implementation Details

- Deterministic translation for first 60% of training epochs
- Backoff to random augmentation for final 40% to improve convergence
- Vectorized PyTorch operations for minimal overhead
- Rigorous experimental controls with matched instances and git commits

---

## Project 2: NanoGPT PaLM-Style Parallel Blocks

📄 **Paper:** [NanoGPT_SpeedUp_Final_Paper.pdf](NanoGPT_SpeedUp_Final_Paper.pdf)

### Abstract

Standard transformer blocks process attention and MLP layers sequentially, creating computational bottlenecks and extending gradient path length to 2L for L layers. This work applies Google's PaLM architecture modification—computing attention and MLP in parallel from shared normalized input with 1/√2 scaling—to NanoGPT training. The parallel formulation enables better GPU kernel fusion, eliminates one LayerNorm per block, and halves effective gradient path length to L.

### Key Results

- **2.66% training speedup** (p = 0.046, statistically significant)
- **7.5% peak memory reduction** (28.3 GB vs 30.6 GB on 8×H100)
- **1.04% validation loss regression** (speed-quality trade-off)
- Three-stage experimental pipeline (screening → validation → final comparison)

**Use Case:** Throughput-focused applications, hyperparameter sweeps, or memory-constrained systems. Not recommended for accuracy-critical final model training.

### Quick Start

```bash
cd nanogpt

# Install dependencies
pip install torch numpy scipy

# Download FineWeb dataset (first 900M tokens for testing)
python cached_fineweb10B.py 9

# Run baseline on 8×H100 GPUs (5 runs, ~35 minutes)
python run_nanogpt_experiment.py \
    --config configs/stage_c_full/baseline.json \
    --n_runs 5 \
    --n_gpus 8

# Run PaLM-parallel modification
python run_nanogpt_experiment.py \
    --config configs/stage_c_full/palm_parallel.json \
    --n_runs 5 \
    --n_gpus 8
```

**Hardware Requirements:**
- 8× NVIDIA H100 GPUs (for full runs)
- PyTorch 2.4.1+ with CUDA 12.1
- ~7 minutes per run on 8×H100

### Three-Stage Methodology

The project employs a budget-efficient experimental workflow:

| Stage | Hardware | Iterations | Cost/Run | Purpose |
|-------|----------|------------|----------|---------|
| **A (Screening)** | 1×L40 | 1000 (~1/5) | ~$2 | Quick validation of multiple variants |
| **B (Validation)** | 4×A100 | 3000 (~3/5) | ~$10 | Confirm speedup generalizes to more compute |
| **C (Final)** | 8×H100 | 5100 (full) | ~$8-9 | Statistical analysis with publication-quality results |

**Budget Savings:** ~43% cost reduction vs. running all variants with full H100 runs.

---

## Repository Structure

```
deep-learning-speedrun-project/
├── README.md                          # This file
├── CIFAR10_SpeedUp_Final_Paper.pdf    # CIFAR-10 research paper
├── NanoGPT_SpeedUp_Final_Paper.pdf    # NanoGPT research paper
├── requirements.txt                    # Python dependencies
│
├── cifar10/                           # CIFAR-10 experiments
│   ├── experiments/                   # Training code variants
│   ├── configs/                       # Experiment configurations
│   ├── experiment_logs/               # Results and metrics
│   ├── analysis/                      # Statistical analysis scripts
│   ├── run_cifar_experiment.py        # Main experiment runner
│   └── QUICKSTART.md                  # Detailed setup guide
│
└── nanogpt/                           # NanoGPT experiments
    ├── experiments/                   # Training scripts
    ├── configs/                       # Config files by stage
    │   ├── stage_a_screening/
    │   ├── stage_b_validation/
    │   └── stage_c_full/
    ├── experiment_logs/               # Results and logs
    ├── run_nanogpt_experiment.py      # Experiment orchestrator
    └── README.md                      # Detailed documentation
```

---

## Key Findings

Both projects demonstrate a consistent pattern: **architectural and algorithmic optimizations can achieve measurable training speedups (2-3%) with minimal accuracy trade-offs (0.06-1.04%)**.

### Cross-Project Insights

1. **Statistical Rigor Matters:** Small speedups require large sample sizes (100s-1000s of runs) to achieve statistical confidence (p < 0.05)
2. **Hardware Variation:** Instance-to-instance differences necessitate matched experimental controls and per-instance baseline comparisons
3. **Speed-Accuracy Trade-offs:** Both modifications trade minor accuracy for consistent speedup—suitable for different use cases
4. **Multi-Stage Validation:** Budget-efficient staged experimentation (screening → validation → final) prevents premature commitment to expensive full-scale runs

### Methodological Contributions

- **Reproducibility:** All experiments include git commit hashes, hardware specifications, and full configuration files
- **Transparency:** Documented failed experiments (ReLU² activation, curriculum learning, depth warming) alongside successes
- **Statistical Testing:** Proper normality tests, variance checks, effect size calculations (Cohen's d), and non-parametric alternatives

---

## Technical Skills Demonstrated

- **Distributed Training:** PyTorch DDP on multi-GPU systems (8×H100, 4×A100)
- **Experiment Infrastructure:** Automated experiment runners, logging systems, statistical analysis pipelines
- **Performance Optimization:** Memory profiling, GPU kernel fusion analysis, vectorization
- **Research Methodology:** Hypothesis formation, ablation studies, multi-stage validation, power analysis
- **Statistical Analysis:** t-tests, permutation tests, normality checks, confidence intervals, effect sizes
- **Cloud Computing:** Runpod.io GPU instance management, multi-instance experiments, budget tracking
- **Version Control:** Git-based reproducibility with commit hashes for all experiments

---

## Papers and Citations

**CIFAR-10 Speedup:**
> Welliver, A. (2025). *Deterministic Cropping to Speed Up CIFAR10 Training*. ESE 3060 Final Project Report. [PDF](CIFAR10_SpeedUp_Final_Paper.pdf)

**NanoGPT Speedup:**
> Welliver, A. (2025). *PaLM-Style Parallel Transformer Blocks to Speed Up NanoGPT Training*. ESE 3060 Final Project Report. [PDF](NanoGPT_SpeedUp_Final_Paper.pdf)

---

## Compute Resources

**Total GPU Budget:** ~$150 across both projects

### CIFAR-10
- **35.85 A100 GPU-hours** (~$50)
- Baseline: 6.12 GPU-hours
- Modified: 29.73 GPU-hours
- Hardware: NVIDIA A100 80GB PCIe

### NanoGPT
- **~$107** across L40, A100, and H100 instances
- Stage A (screening): ~$16 across 4 variants
- Stage B (validation): ~$30 for best variant
- Stage C (final): ~$42 for 3 runs per condition (6 total)
- Hardware: NVIDIA L40 48GB, A100 80GB PCIe, H100 SXM

---

## Related Work

- **CIFAR-10 Baseline:** [Keller Jordan's airbench94](https://github.com/KellerJordan/cifar10-airbench) - 3.83s, 94.01% accuracy record
- **NanoGPT Baseline:** [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) - GPT-2 speedrun efforts
- **PaLM Architecture:** [Google's PaLM paper](https://arxiv.org/abs/2204.02311) - Parallel attention/MLP blocks

---
NanoGPT_SpeedUp_Final_Paper
## Contact

**Alain Welliver**
GitHub: [@alainwelliver](https://github.com/alainwelliver)
Project Repository: [deep-learning-speedrun-project](https://github.com/alainwelliver/deep-learning-speedrun-project)

For questions about reproducing experiments or technical implementation details, please open an issue in the repository.
