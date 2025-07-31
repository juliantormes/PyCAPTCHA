#!/usr/bin/env python3
"""
Launcher script to run all model tests from project root
"""
import subprocess
import sys
import os

def main():
    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Add project root to Python path
    sys.path.insert(0, script_dir)
    
    # Run the main test script
    test_script = os.path.join("scripts", "testing", "test_all_models_v2.py")
    
    if os.path.exists(test_script):
        print("🚀 Running PyCAPTCHA Model Testing Suite...")
        print("=" * 60)
        # Import and run directly instead of subprocess to maintain path context
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_all_models", test_script)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        test_module.main()
    else:
        print(f"❌ Test script not found: {test_script}")

if __name__ == "__main__":
    main()
