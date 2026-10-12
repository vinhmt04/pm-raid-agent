#!/usr/bin/env python
"""Debug script to test pytest import behavior."""

import sys
import importlib.util

def test_import_like_pytest():
    """Test importing the test module like pytest would."""
    print("Testing pytest-like import of test_base.py...")
    
    # Try to import the test module
    test_file = "tests/unit/llm_providers/test_base.py"
    spec = importlib.util.spec_from_file_location("test_base", test_file)
    
    if spec is None:
        print(f"Could not create spec for {test_file}")
        return False
    
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("✓ Test module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Test module import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_import_like_pytest()
    sys.exit(0 if success else 1)