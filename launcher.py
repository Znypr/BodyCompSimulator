# launcher.py
import subprocess
import sys
import os

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
# We assume 'streamlit' executable is available in the environment path or PyInstaller's path
# --server.headless True: Prevents opening a browser window (optional, but often cleaner for PyInstaller)
# --browser.gatherUsageStats False: Reduces unnecessary network activity
command = [
    "streamlit", 
    "run", 
    "main.py", 
    "--server.port", "8501", 
    "--server.headless", "true", 
    "--browser.gatherUsageStats", "false"
]

print("Starting Streamlit app...")
try:
    # Use subprocess.run to execute the command
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error starting Streamlit: {e}")
    input("Press Enter to close...") # Keep window open on error
except FileNotFoundError:
    print("Error: 'streamlit' command not found. Ensure PyInstaller included it correctly.")
    input("Press Enter to close...")