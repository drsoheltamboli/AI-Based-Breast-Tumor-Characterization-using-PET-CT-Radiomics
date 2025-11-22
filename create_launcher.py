"""
Simple Launcher Script for GUI
Creates a convenient way to launch the GUI application
"""

import sys
import os
from pathlib import Path

def launch_gui():
    """Launch the GUI application"""
    try:
        # Check if we're in the right directory
        if not Path("gui_predictor.py").exists():
            print("Error: gui_predictor.py not found!")
            print("Please run this script from the project root directory.")
            input("Press Enter to exit...")
            return
        
        # Check if executable exists
        exe_path = Path("dist/BreastCancerDiagnosisGUI.exe")
        if exe_path.exists():
            print("Launching standalone executable...")
            os.system(f'"{exe_path}"')
        else:
            # Launch using Python
            print("Launching GUI using Python...")
            os.system(f'{sys.executable} gui_predictor.py')
    
    except Exception as e:
        print(f"Error launching GUI: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    launch_gui()

