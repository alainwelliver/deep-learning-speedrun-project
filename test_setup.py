#!/usr/bin/env python3
"""
test_setup.py

Test script to verify that the experiment infrastructure is working correctly.
Run this before starting your actual experiments.

Usage:
    python test_setup.py
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required packages are available."""
    print("Testing imports...")
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'json': 'JSON (built-in)',
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\nERROR: Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("  All imports OK!\n")
    return True


def test_cuda():
    """Test CUDA availability."""
    print("Testing CUDA...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA is available")
        print(f"  ✓ Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  ✓ Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"  ✗ CUDA is NOT available")
        print("  Warning: You need a GPU to run these experiments efficiently")
        return False
    
    print()
    return True


def test_git():
    """Test git availability and repository status."""
    print("Testing git...")
    
    import subprocess
    
    try:
        # Check if git is available
        result = subprocess.run(['git', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"  ✓ Git is available: {result.stdout.strip()}")
        
        # Check if we're in a git repository
        result = subprocess.run(['git', 'rev-parse', '--git-dir'],
                              capture_output=True, text=True, check=True)
        print(f"  ✓ Inside a git repository")
        
        # Get current commit
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                        stderr=subprocess.DEVNULL).decode().strip()
        print(f"  ✓ Current commit: {commit[:8]}")
        
        # Check for uncommitted changes
        status = subprocess.check_output(['git', 'status', '--porcelain'],
                                        stderr=subprocess.DEVNULL).decode()
        if status:
            print(f"  ! Warning: You have uncommitted changes")
            print(f"    (This is okay for testing, but commit before real experiments)")
        else:
            print(f"  ✓ No uncommitted changes")
            
    except subprocess.CalledProcessError:
        print(f"  ✗ Git not available or not in a git repository")
        print(f"  Make sure you've cloned the repository with git")
        return False
    except FileNotFoundError:
        print(f"  ✗ Git not installed")
        return False
    
    print()
    return True


def test_files():
    """Test that required files exist."""
    print("Testing file structure...")
    
    required_files = [
        'experiment_logger.py',
        'run_cifar_experiment.py',
        'configs/baseline.json',
    ]
    
    optional_files = [
        'airbench94.py',
        'train_gpt.py',
    ]
    
    all_ok = True
    
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            all_ok = False
    
    for file in optional_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ! {file} - not found (you'll need this for experiments)")
    
    # Check if experiment_logs directory exists or can be created
    log_dir = Path('experiment_logs')
    if not log_dir.exists():
        print(f"  ℹ Creating experiment_logs directory...")
        log_dir.mkdir(exist_ok=True)
        print(f"  ✓ experiment_logs/ created")
    else:
        print(f"  ✓ experiment_logs/")
    
    print()
    return all_ok


def test_logger():
    """Test the experiment logger."""
    print("Testing experiment logger...")
    
    try:
        from experiment_logger import ExperimentLogger
        
        # Create a test logger
        logger = ExperimentLogger("test_setup", log_dir="experiment_logs")
        
        print(f"  ✓ Logger initialized")
        
        # Log some test data
        logger.log("Test message")
        logger.log_hyperparameters({"test_param": 1.0})
        logger.log_run_result(0, 42, 0.94, 3.5)
        
        print(f"  ✓ Logging functions work")
        
        # Finalize
        exp_dir, summary = logger.finalize({"test": True})
        
        print(f"  ✓ Logger finalized")
        print(f"  ✓ Test logs saved to: {exp_dir}")
        
        # Clean up test logs
        import shutil
        shutil.rmtree(exp_dir)
        print(f"  ✓ Test logs cleaned up")
        
    except Exception as e:
        print(f"  ✗ Logger test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_config():
    """Test loading configuration."""
    print("Testing configuration loading...")
    
    try:
        import json
        
        config_file = Path('configs/baseline.json')
        if not config_file.exists():
            print(f"  ✗ Baseline config not found")
            return False
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"  ✓ Config loaded successfully")
        print(f"  ✓ Experiment name: {config.get('experiment_name', 'N/A')}")
        print(f"  ✓ Base seed: {config.get('base_seed', 'N/A')}")
        
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False
    
    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ESE 3060 Experiment Infrastructure - Setup Test")
    print("="*80 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("Git", test_git),
        ("Files", test_files),
        ("Logger", test_logger),
        ("Config", test_config),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"UNEXPECTED ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Print summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:10s} {name}")
        if not passed:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to run experiments.\n")
        print("Next steps:")
        print("  1. Run baseline: python run_cifar_experiment.py --config configs/baseline.json --n_runs 20")
        print("  2. Check README_EXPERIMENTS.md for full workflow")
        print()
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues above before proceeding.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())