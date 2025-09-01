"""Test to verify profiler syntax is correct."""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_profiler_syntax():
    """Test that the profiler classes can be imported and used."""
    try:
        # Import the profiler module directly
        from tensoratelier.profilers.profiler import AtelierBaseProfiler, ProfilerContext
        
        # Test creating a simple profiler
        class TestProfiler(AtelierBaseProfiler):
            def __init__(self):
                super().__init__()
                self.timings = {}
            
            def start(self, desc: str, **kwargs):
                if desc not in self.timings:
                    self.timings[desc] = []
                self.active_profiles[desc] = time.perf_counter()
            
            def stop(self, desc: str, context, **kwargs):
                if desc in self.active_profiles:
                    start_time = self.active_profiles[desc]
                    elapsed = time.perf_counter() - start_time
                    self.timings[desc].append(elapsed)
                    del self.active_profiles[desc]
                    print(f"✓ {desc}: {elapsed:.4f}s")
        
        # Test the profiler
        profiler = TestProfiler()
        
        # Test basic profiling
        with profiler.profile("test_operation"):
            time.sleep(0.1)  # Simulate some work
        
        print("✓ Profiler syntax test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Profiler syntax test failed: {e}")
        return False

if __name__ == "__main__":
    test_profiler_syntax()
