#!/usr/bin/env python3
"""Quick start script for the Incremental Learning System."""

import sys
import subprocess
from pathlib import Path

def main():
    """Main function for quick start."""
    print("🧠 Incremental Learning System - Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   The 'src' directory should be present.")
        sys.exit(1)
    
    print("✅ Project structure detected")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Error: Python 3.10+ required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Show available options
    print("\n🚀 Available options:")
    print("1. Run training experiments")
    print("2. Launch interactive demo")
    print("3. Run tests")
    print("4. Show help")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n📊 Running training experiments...")
        try:
            subprocess.run([sys.executable, "train.py", "--experiment", "all"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
        except FileNotFoundError:
            print("❌ train.py not found. Make sure you're in the project root.")
    
    elif choice == "2":
        print("\n🎨 Launching interactive demo...")
        try:
            subprocess.run([sys.executable, "run_demo.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Demo failed: {e}")
        except FileNotFoundError:
            print("❌ run_demo.py not found. Make sure you're in the project root.")
    
    elif choice == "3":
        print("\n🧪 Running tests...")
        try:
            subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Tests failed: {e}")
        except FileNotFoundError:
            print("❌ pytest not found. Install with: pip install pytest")
    
    elif choice == "4":
        show_help()
    
    else:
        print("❌ Invalid choice. Please select 1-4.")
        sys.exit(1)

def show_help():
    """Show help information."""
    print("\n📚 Help - Incremental Learning System")
    print("=" * 40)
    print("""
This is a research and education focused incremental learning system.

Key Features:
- Multiple baseline algorithms (SGD, Naive Bayes)
- Advanced continual learning methods (EWC, Experience Replay)
- Comprehensive evaluation metrics
- Interactive Streamlit demo
- Safety and ethics considerations

Quick Commands:
- python train.py --experiment all          # Run all experiments
- python run_demo.py                       # Launch interactive demo
- python -m pytest tests/ -v               # Run test suite
- streamlit run demo/app.py                # Direct demo launch

Configuration:
- Edit configs/default.yaml for parameters
- Modify src/models.py for new algorithms
- Add tests in tests/ directory

Safety Notice:
This system is for research and education only.
Not intended for production use or critical decisions.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
""")

if __name__ == "__main__":
    main()
