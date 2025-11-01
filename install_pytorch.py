#!/usr/bin/env python3
"""
Smart PyTorch installer that detects your platform and installs the appropriate version.

Supports:
- Linux + NVIDIA GPU (CUDA)
- macOS + Apple Silicon (MPS)
- CPU-only fallback
"""

import sys
import platform
import subprocess
import os

def run_command(cmd):
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"Running: {cmd}")
    print('='*80)
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def detect_platform():
    """Detect platform and GPU availability."""
    system = platform.system()
    machine = platform.machine()
    
    print("="*80)
    print("Platform Detection")
    print("="*80)
    print(f"System: {system}")
    print(f"Machine: {machine}")
    print(f"Python: {platform.python_version()}")
    
    # Check for NVIDIA GPU
    has_nvidia = False
    if system == "Linux":
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True)
        has_nvidia = result.returncode == 0
        if has_nvidia:
            print("✓ NVIDIA GPU detected")
    
    # Check for Apple Silicon
    is_apple_silicon = system == "Darwin" and machine == "arm64"
    if is_apple_silicon:
        print("✓ Apple Silicon (M1/M2/M3) detected")
    
    return system, machine, has_nvidia, is_apple_silicon

def install_pytorch_cuda():
    """Install PyTorch with CUDA support."""
    print("\n" + "="*80)
    print("Installing PyTorch with CUDA 11.8 support")
    print("="*80)
    
    commands = [
        # Install PyTorch with CUDA
        "pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118",
        
        # Install PyTorch Geometric with CUDA wheels
        "pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html",
        "pip install torch-geometric",
    ]
    
    for cmd in commands:
        if not run_command(cmd):
            print(f"\n❌ Failed to install. Command: {cmd}")
            return False
    
    return True

def install_pytorch_mps():
    """Install PyTorch with MPS (Apple Silicon) support."""
    print("\n" + "="*80)
    print("Installing PyTorch with MPS (Apple Silicon) support")
    print("="*80)
    
    commands = [
        # Install PyTorch for macOS
        "pip install torch==2.0.0 torchvision==0.15.0",
        
        # Install PyTorch Geometric (CPU wheels work for MPS)
        "pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html",
        "pip install torch-geometric",
    ]
    
    for cmd in commands:
        if not run_command(cmd):
            print(f"\n❌ Failed to install. Command: {cmd}")
            return False
    
    return True

def install_pytorch_cpu():
    """Install CPU-only PyTorch."""
    print("\n" + "="*80)
    print("Installing PyTorch (CPU-only)")
    print("="*80)
    
    commands = [
        # Install PyTorch CPU
        "pip install torch==2.0.0 torchvision==0.15.0 --extra-index-url https://download.pytorch.org/whl/cpu",
        
        # Install PyTorch Geometric CPU wheels
        "pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html",
        "pip install torch-geometric",
    ]
    
    for cmd in commands:
        if not run_command(cmd):
            print(f"\n❌ Failed to install. Command: {cmd}")
            return False
    
    return True

def verify_installation():
    """Verify PyTorch installation."""
    print("\n" + "="*80)
    print("Verifying Installation")
    print("="*80)
    
    try:
        import torch
        import torch_geometric
        
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ PyTorch Geometric version: {torch_geometric.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.is_available()}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif torch.backends.mps.is_available():
            print(f"✓ MPS (Apple Silicon GPU) available: {torch.backends.mps.is_available()}")
        else:
            print(f"✓ Running on CPU")
        
        return True
        
    except ImportError as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main installation flow."""
    print("\n" + "="*80)
    print("PyTorch Smart Installer for GraphML-IMTS")
    print("="*80)
    
    # Detect platform
    system, machine, has_nvidia, is_apple_silicon = detect_platform()
    
    # Determine installation type
    if has_nvidia:
        print("\n📦 Recommendation: Install with CUDA support for maximum performance")
        install_func = install_pytorch_cuda
    elif is_apple_silicon:
        print("\n📦 Recommendation: Install with MPS support for Apple Silicon GPU")
        install_func = install_pytorch_mps
    else:
        print("\n📦 Recommendation: Install CPU-only version")
        install_func = install_pytorch_cpu
    
    # Ask for confirmation
    response = input("\nProceed with installation? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("Installation cancelled.")
        return 1
    
    # Install
    success = install_func()
    
    if not success:
        print("\n" + "="*80)
        print("❌ Installation failed!")
        print("="*80)
        return 1
    
    # Verify
    if not verify_installation():
        print("\n" + "="*80)
        print("⚠️  Installation completed but verification failed")
        print("="*80)
        return 1
    
    print("\n" + "="*80)
    print("✅ Installation successful!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Test the setup: python test_raindrop_training.py")
    print("  2. Start training: python src/models/raindrop/train_raindrop_forecasting.py --epochs 50")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

