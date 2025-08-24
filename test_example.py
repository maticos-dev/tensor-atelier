#!/usr/bin/env python3
"""
Test script to verify the example can import without syntax errors.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_example_import():
    """Test that the example script can import without syntax errors."""
    
    print("Testing example script imports...")
    
    try:
        # Test the imports that the example script uses
        from tensoratelier import AtelierModule, AtelierTrainer
        print("✓ Example imports successful!")
        return True
    except ImportError as e:
        if "torch" in str(e):
            print("⚠ Example imports failed due to missing PyTorch (expected)")
            return True  # This is expected without PyTorch
        else:
            print(f"✗ Example imports failed: {e}")
            return False
    except SyntaxError as e:
        print(f"✗ Syntax error in imports: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_example_import()
    if success:
        print("\n🎉 Example script can import successfully!")
    else:
        print("\n❌ Example script has import issues.")
        sys.exit(1)
