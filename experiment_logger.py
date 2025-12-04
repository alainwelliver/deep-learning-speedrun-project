"""
experiment_logger.py

Comprehensive logging system for ESE 3060 Final Project experiments.
Tracks git state, GPU information, hyperparameters, metrics, and results.
"""

import json
import os
import subprocess
import datetime
import torch
from pathlib import Path
import sys


class ExperimentLogger:
    """
    Logger for ML experiments that captures:
    - Git state (commit, branch, dirty status, diff)
    - GPU information
    - Hyperparameters
    - Per-run metrics
    - Final statistics and results
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
        self.results_file = self.exp_dir / "results.jsonl"  # JSON lines format
        self.metrics_file = self.exp_dir / "metrics.jsonl"
        
        # Capture system info
        self.git_info = self.get_git_info()
        self.gpu_info = self.get_gpu_info()
        self.system_info = self.get_system_info()
        
        # Start experiment log
        self.start_log()
        
        print(f"\n{'='*80}")
        print(f"Experiment Logger Initialized: {experiment_name}")
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
EXPERIMENT LOG
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
    
    def log_hyperparameters(self, hyperparams):
        """Log hyperparameters as JSON"""
        hp_file = self.exp_dir / "hyperparameters.json"
        with open(hp_file, 'w') as f:
            json.dump(hyperparams, f, indent=2)
        
        self.log(f"Hyperparameters saved to {hp_file.name}")
        self.log(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")
    
    def log_metrics(self, metrics, step=None, run_id=None):
        """
        Log metrics (accuracy, loss, time, etc.)
        
        Args:
            metrics (dict): Dictionary of metric name -> value
            step (int, optional): Training step or epoch
            run_id (int, optional): Run/seed identifier
        """
        timestamp = datetime.datetime.now().isoformat()
        
        record = {
            "timestamp": timestamp,
            "metrics": metrics
        }
        
        if step is not None:
            record["step"] = step
        if run_id is not None:
            record["run_id"] = run_id
        
        # Append to metrics file (JSONL format - one JSON per line)
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(record) + "\n")
        
        # Also log to main log file
        metric_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        log_msg = f"Metrics: {metric_str}"
        if step is not None:
            log_msg = f"Step {step} - {log_msg}"
        if run_id is not None:
            log_msg = f"Run {run_id} - {log_msg}"
        
        self.log(log_msg)
    
    def log_run_result(self, run_id, seed, accuracy, time_seconds, **kwargs):
        """
        Log the result of a single training run.
        
        Args:
            run_id (int): Run identifier
            seed (int): Random seed used
            accuracy (float): Final accuracy achieved
            time_seconds (float): Training time in seconds
            **kwargs: Additional metrics to log
        """
        result = {
            "run_id": run_id,
            "seed": seed,
            "accuracy": accuracy,
            "time_seconds": time_seconds,
            "timestamp": datetime.datetime.now().isoformat()
        }
        result.update(kwargs)
        
        # Append to results file
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(result) + "\n")
        
        self.log(f"Run {run_id} complete: accuracy={accuracy:.4f}, time={time_seconds:.2f}s, seed={seed}")
    
    def save_model(self, model, filename="model.pt"):
        """Save model checkpoint"""
        path = self.exp_dir / filename
        torch.save(model.state_dict(), path)
        self.log(f"Model saved to {filename}")
        return path
    
    def load_results(self):
        """Load all results from the results file"""
        results = []
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                for line in f:
                    results.append(json.loads(line))
        return results
    
    def compute_statistics(self):
        """Compute statistics from all logged results"""
        results = self.load_results()
        
        if not results:
            return {"error": "No results to compute statistics from"}
        
        accuracies = [r['accuracy'] for r in results]
        times = [r['time_seconds'] for r in results]
        
        import numpy as np
        
        stats = {
            "n_runs": len(results),
            "accuracy": {
                "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies, ddof=1)),
                "min": float(np.min(accuracies)),
                "max": float(np.max(accuracies)),
                "median": float(np.median(accuracies)),
            },
            "time": {
                "mean": float(np.mean(times)),
                "std": float(np.std(times, ddof=1)),
                "min": float(np.min(times)),
                "max": float(np.max(times)),
                "total": float(np.sum(times)),
            }
        }
        
        # 95% confidence interval for accuracy
        from scipy import stats as scipy_stats
        ci = scipy_stats.t.interval(
            0.95,
            len(accuracies) - 1,
            loc=stats["accuracy"]["mean"],
            scale=scipy_stats.sem(accuracies)
        )
        stats["accuracy"]["ci_95"] = [float(ci[0]), float(ci[1])]
        
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
        if 'accuracy' in stats:
            self.log(f"Mean accuracy: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")
            if 'ci_95' in stats['accuracy']:
                ci = stats['accuracy']['ci_95']
                self.log(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        
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
        print(f"Experiment Complete!")
        print(f"Results saved to: {self.exp_dir}")
        print(f"{'='*80}\n")
        
        return self.exp_dir, summary


if __name__ == "__main__":
    # Test the logger
    print("Testing ExperimentLogger...")
    
    logger = ExperimentLogger("test_experiment")
    
    logger.log_hyperparameters({
        "learning_rate": 0.1,
        "batch_size": 128,
        "epochs": 10
    })
    
    # Simulate some runs
    import random
    for i in range(5):
        acc = 0.92 + random.random() * 0.03
        time = 100 + random.random() * 20
        logger.log_run_result(
            run_id=i,
            seed=42 + i,
            accuracy=acc,
            time_seconds=time
        )
    
    logger.finalize({"test": "This was a test"})
    
    print("Test complete!")