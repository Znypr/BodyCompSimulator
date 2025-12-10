# launcher.py

import subprocess
import sys
import os

# --- CRITICAL GUARD ---
# Check if the script is running inside a Streamlit server process.
# This prevents the infinite loop when Streamlit re-executes its entry point.
if os.environ.get("STREAMLIT_SERVER_PORT") is not None:
    # If the environment variable is set, we are running INSIDE the server.
    # We should NOT try to launch Streamlit again. 
    # The actual 'main.py' script will be loaded by the server itself.
    sys.exit(0)
# ----------------------

# Get the path to the temporary folder where PyInstaller extracts files
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller executable
    base_dir = sys._MEIPASS
    # Change the current working directory to the directory where the files are extracted
    os.chdir(base_dir)
else:
    # Running as a script (for testing)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

# Construct the full command to run Streamlit
# FIX: Use sys.executable and the -m flag to run streamlit as a Python module.
command = [
    sys.executable,        # The bundled Python interpreter
    "-m",                  # Run the following as a module
    "streamlit",           # The streamlit module
    "run", 
    "main.py", 
    "--server.port", "8501", 
    "--server.headless", "true", 
    "--browser.gatherUsageStats", "false"
]

print("Starting Streamlit app...")
try:
    # Use subprocess.run to execute the command
    # NOTE: This command will block until the user closes the app's console window.
    subprocess.run(command, check=True)
    
except subprocess.CalledProcessError as e:
    print(f"\n--- Streamlit Process Ended with Error ---")
    print(f"Error Code: {e.returncode}")
    print(f"Details: {e}")
    input("Press Enter to close...") # Keep window open on error
except FileNotFoundError:
    print("\n--- CRITICAL ERROR ---")
    print("The bundled Python interpreter or required modules were not found.") 
    input("Press Enter to close...")