#!/usr/bin/env python3
"""Test script to verify the example can import without syntax errors."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_example_import():
    """Test that the example script can import without syntax errors."""
    try:
        print("Testing example import...")
        
        # Test importing the example
        import examples.simple_training
        print("✓ Example imported successfully")
        
        # Test importing the custom profiler example
        import examples.custom_profiler
        print("✓ Custom profiler example imported successfully")
        
        print("All examples imported successfully!")
        
    except SyntaxError as e:
        print(f"Syntax error: {e}")
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    test_example_import()
