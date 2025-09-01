#!/usr/bin/env python3
"""Simple test script to verify tensor-atelier functionality."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import torch
    print("✓ PyTorch imported successfully")
except ImportError:
    print("⚠ PyTorch not available - some functionality may be limited")

try:
    import tensoratelier
    print("✓ tensoratelier imported successfully")
    
    # Test basic functionality
    print(f"tensoratelier version: {getattr(tensoratelier, '__version__', 'unknown')}")
    
except ImportError as e:
    print(f"✗ Failed to import tensoratelier: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

print("Import test completed!")
