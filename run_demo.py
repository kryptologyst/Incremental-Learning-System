#!/usr/bin/env python3
"""Simple script to run the incremental learning demo."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit demo."""
    demo_path = Path(__file__).parent / "demo" / "app.py"
    
    if not demo_path.exists():
        print("Error: Demo app not found!")
        sys.exit(1)
    
    print("Starting Incremental Learning System Demo...")
    print("This will open in your web browser.")
    print("Press Ctrl+C to stop the demo.")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path), "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        print(f"Error running demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
