#!/usr/bin/env python3
"""Test script to check if circular imports are fixed."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test various imports to check for circular import issues."""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import tensoratelier
        print("✓ tensoratelier imported successfully")
        
        # Test core imports
        from tensoratelier.core import AtelierModule, AtelierTrainer
        print("✓ core modules imported successfully")
        
        # Test accelerator imports
        from tensoratelier.accelerators import Accelerator
        print("✓ accelerators imported successfully")
        
        # Test profiler imports
        from tensoratelier.profilers import BaseProfiler
        print("✓ profilers imported successfully")
        
        print("All imports successful! No circular import issues detected.")
        
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    test_imports()
