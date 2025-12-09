"""
experiment_logger.py

Comprehensive logging system for NanoGPT speedrun experiments.
Adapted from CIFAR-10 ExperimentLogger for validation loss tracking and multi-GPU training.
Tracks git state, GPU information, hyperparameters, per-step metrics, and results.
"""

import json
import os
import subprocess
import datetime
from pathlib import Path
import sys


class GPTExperimentLogger:
    """
    Logger for NanoGPT experiments that captures:
    - Git state (commit, branch, dirty status, diff)
    - GPU information
    - Hyperparameters and configuration
    - Per-step training and validation metrics
    - Per-run final results
    - Statistical summaries
    """

    def __init__(self, experiment_name, log_dir="experiment_logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create unique experiment folder with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.exp_dir.mkdir(exist_ok=True)

        # Initialize log files
        self.log_file = self.exp_dir / "experiment.log"
        self.results_file = self.exp_dir / "results.jsonl"  # Per-run final results
        self.metrics_file = self.exp_dir / "metrics.jsonl"  # Per-step metrics

        # Capture system info
        self.git_info = self.get_git_info()
        self.gpu_info = self.get_gpu_info()
        self.system_info = self.get_system_info()

        # Start experiment log
        self.start_log()

        print(f"\n{'='*80}")
        print(f"NanoGPT Experiment Logger Initialized: {experiment_name}")
        print(f"Log directory: {self.exp_dir}")
        print(f"{'='*80}\n")

    def get_git_info(self):
        """Capture current git state - CRITICAL for reproducibility"""
        try:
            # Get commit hash
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            # Get branch name
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            # Check for uncommitted changes
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL
            ).decode()
            dirty = len(status) > 0

            # Get diff if there are uncommitted changes
            diff = ""
            if dirty:
                diff = subprocess.check_output(
                    ["git", "diff"],
                    stderr=subprocess.DEVNULL
                ).decode()

            # Get remote URL
            try:
                remote = subprocess.check_output(
                    ["git", "config", "--get", "remote.origin.url"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
            except:
                remote = "N/A"

            return {
                "commit": commit,
                "branch": branch,
                "dirty": dirty,
                "diff": diff,
                "remote": remote,
                "status": status
            }
        except subprocess.CalledProcessError:
            return {
                "error": "Not a git repository or git not available",
                "commit": "N/A",
                "branch": "N/A",
                "dirty": False
            }

    def get_gpu_info(self):
        """Capture GPU information"""
        try:
            import torch
            gpu_info = {
                "available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
            }

            if torch.cuda.is_available():
                gpu_info["devices"] = []
                for i in range(torch.cuda.device_count()):
                    gpu_info["devices"].append({
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "capability": torch.cuda.get_device_capability(i),
                        "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / 1e9
                    })

            return gpu_info
        except Exception as e:
            return {"error": f"GPU info unavailable: {str(e)}"}

    def get_system_info(self):
        """Capture system information"""
        import platform
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        }

    def start_log(self):
        """Write initial experiment header"""
        header = f"""
{'='*80}
NANOGPT EXPERIMENT LOG
{'='*80}
Experiment Name: {self.experiment_name}
Start Time: {datetime.datetime.now().isoformat()}

GIT INFORMATION:
  Commit:  {self.git_info.get('commit', 'N/A')}
  Branch:  {self.git_info.get('branch', 'N/A')}
  Remote:  {self.git_info.get('remote', 'N/A')}
  Dirty:   {self.git_info.get('dirty', False)}

GPU INFORMATION:
  CUDA Available: {self.gpu_info.get('available', False)}
  Device Count:   {self.gpu_info.get('device_count', 0)}
  CUDA Version:   {self.gpu_info.get('cuda_version', 'N/A')}
  PyTorch:        {self.gpu_info.get('pytorch_version', 'N/A')}
"""

        if self.gpu_info.get('available') and 'devices' in self.gpu_info:
            for device in self.gpu_info['devices']:
                header += f"  Device {device['index']}: {device['name']} ({device['total_memory_gb']:.1f} GB)\n"

        header += f"\n{'='*80}\n\n"

        with open(self.log_file, 'w') as f:
            f.write(header)

        # Save git diff if dirty
        if self.git_info.get('dirty') and self.git_info.get('diff'):
            with open(self.exp_dir / "git_diff.patch", 'w') as f:
                f.write(self.git_info['diff'])

    def log(self, message, level="INFO"):
        """Log a message with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] [{level}] {message}"

        # Print to console
        print(log_line)

        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_line + "\n")

    def log_config(self, config):
        """Log experiment configuration as JSON"""
        config_file = self.exp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        self.log(f"Configuration saved to {config_file.name}")
        self.log(f"Configuration: {json.dumps(config, indent=2)}")

    def log_step_metrics(self, run_id, step, train_loss=None, val_loss=None,
                         train_time_ms=None, step_avg_ms=None, **kwargs):
        """
        Log per-step training or validation metrics.

        Args:
            run_id (int): Run identifier
            step (int): Training step number
            train_loss (float, optional): Training loss at this step
            val_loss (float, optional): Validation loss at this step
            train_time_ms (float, optional): Cumulative training time in ms
            step_avg_ms (float, optional): Average time per step in ms
            **kwargs: Additional metrics to log
        """
        record = {
            "run_id": run_id,
            "step": step,
            "timestamp": datetime.datetime.now().isoformat()
        }

        if train_loss is not None:
            record["train_loss"] = float(train_loss)
        if val_loss is not None:
            record["val_loss"] = float(val_loss)
        if train_time_ms is not None:
            record["train_time_ms"] = float(train_time_ms)
        if step_avg_ms is not None:
            record["step_avg_ms"] = float(step_avg_ms)

        # Add any additional metrics
        record.update(kwargs)

        # Append to metrics file (JSONL format)
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(record) + "\n")

    def log_run_result(self, run_id, seed, final_val_loss, final_train_loss,
                       time_seconds, final_step, success=True, **kwargs):
        """
        Log the final result of a single training run.

        Args:
            run_id (int): Run identifier
            seed (int): Random seed used
            final_val_loss (float): Final validation loss achieved
            final_train_loss (float): Final training loss
            time_seconds (float): Total training time in seconds
            final_step (int): Final step number reached
            success (bool): Whether the run completed successfully
            **kwargs: Additional metrics to log
        """
        result = {
            "run_id": run_id,
            "seed": seed,
            "final_val_loss": float(final_val_loss),
            "final_train_loss": float(final_train_loss),
            "time_seconds": float(time_seconds),
            "final_step": int(final_step),
            "success": success,
            "timestamp": datetime.datetime.now().isoformat()
        }
        result.update(kwargs)

        # Append to results file
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(result) + "\n")

        status = "complete" if success else "FAILED"
        self.log(f"Run {run_id} {status}: val_loss={final_val_loss:.4f}, train_loss={final_train_loss:.4f}, time={time_seconds:.2f}s, seed={seed}")

    def load_results(self):
        """Load all results from the results file"""
        results = []
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        results.append(json.loads(line))
        return results

    def compute_statistics(self):
        """Compute statistics from all logged results"""
        results = self.load_results()

        if not results:
            return {"error": "No results to compute statistics from"}

        # Filter successful runs
        successful_runs = [r for r in results if r.get('success', True)]

        if not successful_runs:
            return {
                "error": "No successful runs to compute statistics from",
                "n_runs": len(results),
                "successful_runs": 0,
                "failed_runs": len(results)
            }

        val_losses = [r['final_val_loss'] for r in successful_runs]
        train_losses = [r['final_train_loss'] for r in successful_runs]
        times = [r['time_seconds'] for r in successful_runs]

        import numpy as np

        stats = {
            "n_runs": len(results),
            "successful_runs": len(successful_runs),
            "failed_runs": len(results) - len(successful_runs),
            "val_loss": {
                "mean": float(np.mean(val_losses)),
                "std": float(np.std(val_losses, ddof=1)) if len(val_losses) > 1 else 0.0,
                "min": float(np.min(val_losses)),
                "max": float(np.max(val_losses)),
                "median": float(np.median(val_losses)),
            },
            "train_loss": {
                "mean": float(np.mean(train_losses)),
                "std": float(np.std(train_losses, ddof=1)) if len(train_losses) > 1 else 0.0,
                "min": float(np.min(train_losses)),
                "max": float(np.max(train_losses)),
                "median": float(np.median(train_losses)),
            },
            "time": {
                "mean": float(np.mean(times)),
                "std": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
                "min": float(np.min(times)),
                "max": float(np.max(times)),
                "total": float(np.sum(times)),
            }
        }

        # 95% confidence interval for val_loss
        if len(val_losses) > 1:
            from scipy import stats as scipy_stats
            ci = scipy_stats.t.interval(
                0.95,
                len(val_losses) - 1,
                loc=stats["val_loss"]["mean"],
                scale=scipy_stats.sem(val_losses)
            )
            stats["val_loss"]["ci_95"] = [float(ci[0]), float(ci[1])]

        return stats

    def finalize(self, additional_info=None):
        """
        Finalize the experiment and write summary.

        Args:
            additional_info (dict, optional): Additional information to include in summary
        """
        self.log("="*80)
        self.log("FINALIZING EXPERIMENT")
        self.log("="*80)

        # Compute statistics
        stats = self.compute_statistics()

        self.log(f"Experiment completed with {stats.get('n_runs', 0)} runs")
        self.log(f"Successful runs: {stats.get('successful_runs', 0)}")
        self.log(f"Failed runs: {stats.get('failed_runs', 0)}")

        if 'val_loss' in stats:
            self.log(f"Mean validation loss: {stats['val_loss']['mean']:.4f} ± {stats['val_loss']['std']:.4f}")
            if 'ci_95' in stats['val_loss']:
                ci = stats['val_loss']['ci_95']
                self.log(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
            self.log(f"Best validation loss: {stats['val_loss']['min']:.4f}")

        if 'time' in stats:
            self.log(f"Mean time per run: {stats['time']['mean']:.2f}s")
            self.log(f"Total training time: {stats['time']['total']:.2f}s ({stats['time']['total']/60:.2f} minutes)")

        # Create final summary
        summary = {
            "experiment_name": self.experiment_name,
            "end_time": datetime.datetime.now().isoformat(),
            "git_info": self.git_info,
            "gpu_info": self.gpu_info,
            "system_info": self.system_info,
            "statistics": stats,
        }

        if additional_info:
            summary["additional_info"] = additional_info

        # Save summary
        summary_file = self.exp_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.log(f"Summary saved to {summary_file.name}")
        self.log(f"End Time: {datetime.datetime.now()}")
        self.log("="*80)

        print(f"\n{'='*80}")
        print(f"NanoGPT Experiment Complete!")
        print(f"Results saved to: {self.exp_dir}")
        if 'val_loss' in stats:
            print(f"Mean validation loss: {stats['val_loss']['mean']:.4f} ± {stats['val_loss']['std']:.4f}")
        print(f"{'='*80}\n")

        return self.exp_dir, summary


if __name__ == "__main__":
    # Test the logger
    print("Testing GPTExperimentLogger...")

    logger = GPTExperimentLogger("test_nanogpt_experiment")

    logger.log_config({
        "experiment_name": "test",
        "script": "train_gpt.py",
        "n_gpus": 1,
        "base_seed": 42
    })

    # Simulate some runs
    import random
    for i in range(3):
        # Simulate per-step metrics
        for step in [0, 125, 250, 375, 500]:
            if step == 0 or step % 125 == 0:
                # Validation step
                logger.log_step_metrics(
                    run_id=i,
                    step=step,
                    val_loss=10.0 - step * 0.01 + random.random() * 0.1,
                    train_time_ms=step * 100,
                    step_avg_ms=95 + random.random() * 10
                )
            else:
                # Training step
                logger.log_step_metrics(
                    run_id=i,
                    step=step,
                    train_loss=10.0 - step * 0.012 + random.random() * 0.1,
                    train_time_ms=step * 100,
                    step_avg_ms=95 + random.random() * 10
                )

        # Log final result
        final_val_loss = 3.25 + random.random() * 0.1
        final_train_loss = 2.85 + random.random() * 0.1
        logger.log_run_result(
            run_id=i,
            seed=42 + i,
            final_val_loss=final_val_loss,
            final_train_loss=final_train_loss,
            time_seconds=300 + random.random() * 60,
            final_step=500,
            success=True
        )

    logger.finalize({"test": "This was a test"})

    print("Test complete!")
