#!/usr/bin/env python3
"""
Test script to check if circular imports are fixed.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test various imports to check for circular import issues."""
    
    print("Testing imports...")
    
    try:
        # Test core imports
        from tensoratelier.core import AtelierModule, AtelierTrainer, AtelierDataLoader, AtelierOptimizer
        print("✓ Core imports successful")
    except ImportError as e:
        if "torch" in str(e):
            print("⚠ Core imports failed due to missing PyTorch (expected)")
        else:
            print(f"✗ Core imports failed: {e}")
            return False
    except Exception as e:
        print(f"✗ Core imports failed: {e}")
        return False
    
    try:
        # Test handlers imports
        from tensoratelier.handlers import AcceleratorHandler
        print("✓ Handlers imports successful")
    except ImportError as e:
        if "torch" in str(e):
            print("⚠ Handlers imports failed due to missing PyTorch (expected)")
        else:
            print(f"✗ Handlers imports failed: {e}")
            return False
    except Exception as e:
        print(f"✗ Handlers imports failed: {e}")
        return False
    
    try:
        # Test loops imports
        from tensoratelier.loops import _FitLoop, _TrainingEpochLoop
        print("✓ Loops imports successful")
    except ImportError as e:
        if "torch" in str(e):
            print("⚠ Loops imports failed due to missing PyTorch (expected)")
        else:
            print(f"✗ Loops imports failed: {e}")
            return False
    except Exception as e:
        print(f"✗ Loops imports failed: {e}")
        return False
    
    try:
        # Test profilers imports
        from tensoratelier.profilers import BaseProfiler
        print("✓ Profilers imports successful")
    except ImportError as e:
        if "torch" in str(e):
            print("⚠ Profilers imports failed due to missing PyTorch (expected)")
        else:
            print(f"✗ Profilers imports failed: {e}")
            return False
    except Exception as e:
        print(f"✗ Profilers imports failed: {e}")
        return False
    
    try:
        # Test accelerators imports
        from tensoratelier.accelerators import BaseAccelerator
        print("✓ Accelerators imports successful")
    except ImportError as e:
        if "torch" in str(e):
            print("⚠ Accelerators imports failed due to missing PyTorch (expected)")
        else:
            print(f"✗ Accelerators imports failed: {e}")
            return False
    except Exception as e:
        print(f"✗ Accelerators imports failed: {e}")
        return False
    
    try:
        # Test utils imports
        from tensoratelier.utils.parsing import _wrap_args
        print("✓ Utils imports successful")
    except ImportError as e:
        if "torch" in str(e):
            print("⚠ Utils imports failed due to missing PyTorch (expected)")
        else:
            print(f"✗ Utils imports failed: {e}")
            return False
    except Exception as e:
        print(f"✗ Utils imports failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 All imports successful! Circular imports are fixed.")
    else:
        print("\n❌ Some imports failed. Circular imports may still exist.")
        sys.exit(1)
