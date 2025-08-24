#!/usr/bin/env python3
"""
Simple test script to verify tensor-atelier functionality.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import torch
    print(f"✓ PyTorch imported successfully (version: {torch.__version__})")
except ImportError as e:
    print(f"✗ Failed to import PyTorch: {e}")
    sys.exit(1)

try:
    import tensoratelier
    print("✓ tensoratelier imported successfully")
except ImportError as e:
    print(f"✗ Failed to import tensoratelier: {e}")
    sys.exit(1)

try:
    from tensoratelier.core import AtelierModule, AtelierTrainer, AtelierDataLoader, AtelierOptimizer
    print("✓ Core classes imported successfully")
except ImportError as e:
    print(f"✗ Failed to import core classes: {e}")
    sys.exit(1)

try:
    from tensoratelier.profilers import BaseProfiler
    print("✓ Profilers imported successfully")
except ImportError as e:
    print(f"✗ Failed to import profilers: {e}")
    sys.exit(1)

try:
    from tensoratelier.accelerators import BaseAccelerator
    print("✓ Accelerators imported successfully")
except ImportError as e:
    print(f"✗ Failed to import accelerators: {e}")
    sys.exit(1)

print("\n🎉 All imports successful! Tensor Atelier is ready to use.")
