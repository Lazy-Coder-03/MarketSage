# src/marketsage.py

import os
import sys
import argparse
import subprocess
import shutil

# Define the paths for your project based on the directory structure.
# This ensures the script works correctly no matter where it's executed from.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
MAIN_APP_SCRIPT = os.path.join(SRC_DIR, 'main.py')
STREAMLIT_APP_SCRIPT = os.path.join(SRC_DIR, 'app.py')
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, 'requirements.txt')

def check_and_install_dependencies():
    """
    Checks for and installs required dependencies in a virtual environment.
    This function handles both virtualenv creation and package installation.
    """
    print("🛠️ Checking for and installing dependencies...")
    venv_dir = os.path.join(PROJECT_ROOT, 'venv')
    
    # Check if a virtual environment exists
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        try:
            # Check if virtualenv is installed globally
            subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True, cwd=PROJECT_ROOT)
            print("Virtual environment created successfully.")
        except FileNotFoundError:
            print("Error: 'venv' module not found. Please ensure Python is correctly installed or try 'pip install virtualenv'.", file=sys.stderr)
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Determine the path to the pip executable inside the venv
    if sys.platform == "win32":
        pip_path = os.path.join(venv_dir, 'Scripts', 'pip.exe')
    else:
        pip_path = os.path.join(venv_dir, 'bin', 'pip')
    
    # Check if requirements.txt exists
    if not os.path.exists(REQUIREMENTS_FILE):
        print(f"Error: requirements.txt not found at {REQUIREMENTS_FILE}. Please ensure it exists.", file=sys.stderr)
        sys.exit(1)

    print("Installing packages from requirements.txt...")
    try:
        subprocess.run([pip_path, 'install', '-r', REQUIREMENTS_FILE], check=True, cwd=PROJECT_ROOT)
        print("✅ Dependencies installed successfully.")
    except FileNotFoundError:
        print("Error: 'pip' executable not found in the virtual environment. Please check the virtual environment setup.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}", file=sys.stderr)
        print("Please review requirements.txt and your network connection.", file=sys.stderr)
        sys.exit(1)

def normalize_symbol(symbol):
    """
    Normalizes a user-input stock symbol to a standard format.
    Handles various cases like RELIANCE, reliance.ns, etc.
    """
    # Convert to uppercase
    symbol = symbol.upper()
    
    # Check for common Indian market suffixes and add if missing
    if not symbol.endswith(".NS"):
        if "NS" not in symbol:
            symbol += ".NS"
    
    # Remove any extra spaces
    return symbol.strip()

def train_model(model_name, lookback_days):
    """
    Calls the main.py training script with the specified symbol and lookback days.
    """
    print(f"--- Starting training for model '{model_name}' with lookback of {lookback_days} days ---")
    
    # Determine the Python executable path inside the venv
    venv_dir = os.path.join(PROJECT_ROOT, 'venv')
    if sys.platform == "win32":
        python_path = os.path.join(venv_dir, 'Scripts', 'python.exe')
    else:
        python_path = os.path.join(venv_dir, 'bin', 'python')

    try:
        # Use subprocess to run the main.py script with arguments
        command = [python_path, MAIN_APP_SCRIPT, '--symbol', model_name, '--lookback', str(lookback_days)]
        subprocess.run(command, check=True)
        print("--- Model training complete ---")
        return True
    except FileNotFoundError:
        print(f"Error: Python executable or training script was not found. Please ensure dependencies are installed.", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during model training: {e}", file=sys.stderr)
        print("Please check the error messages above for details on the failure.", file=sys.stderr)
        return False

def run_streamlit_app():
    """
    Runs the Streamlit application using a subprocess.
    """
    print("--- Loading and running Streamlit app ---")
    
    # Determine the Streamlit executable path inside the venv
    venv_dir = os.path.join(PROJECT_ROOT, 'venv')
    if sys.platform == "win32":
        streamlit_path = os.path.join(venv_dir, 'Scripts', 'streamlit.exe')
    else:
        streamlit_path = os.path.join(venv_dir, 'bin', 'streamlit')

    try:
        # This command starts the Streamlit server and should open a browser window.
        subprocess.run([streamlit_path, 'run', STREAMLIT_APP_SCRIPT], check=True)
    except FileNotFoundError:
        print("Error: 'streamlit' not found in the virtual environment. Please ensure it's in your requirements.txt and you have run the setup script.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running Streamlit: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """
    Main function to handle the CLI logic, model management, and app launch.
    """
    # First, handle all setup and installation logic
    check_and_install_dependencies()
    
    # Now, proceed with the main workflow
    raw_symbol = input("What ticker/symbol do you want to train/load the model for (e.g., RELIANCE.NS)? ")
    
    if not raw_symbol:
        print("Error: A stock symbol must be provided.", file=sys.stderr)
        sys.exit(1)
    
    stock_symbol = normalize_symbol(raw_symbol)
    
    model_path = os.path.join(MODELS_DIR, stock_symbol.replace(".", "_"))

    if os.path.exists(model_path) and os.path.isdir(model_path):
        print(f"Model for '{stock_symbol}' is already present.")
        while True:
            choice = input("Do you want to retrain it again? (y/n): ").lower()
            if choice == 'y':
                try:
                    lookback_str = input("Enter the 'lookback' parameter (e.g., 60): ")
                    lookback_days = int(lookback_str)
                    
                    print(f"Removing old model directory: {model_path}")
                    shutil.rmtree(model_path)
                    
                    if train_model(stock_symbol, lookback_days):
                        break
                    else:
                        print("Training failed. Cannot proceed.", file=sys.stderr)
                        sys.exit(1)
                except ValueError:
                    print("Invalid input. Please enter a valid number for 'lookback'.")
                except OSError as e:
                    print(f"Error removing old model directory: {e}", file=sys.stderr)
                    sys.exit(1)
            elif choice == 'n':
                print("Using the existing model. Skipping training.")
                break
            else:
                print("Invalid choice. Please enter 'y' or 'n'.")
    else:
        print(f"Model for '{stock_symbol}' not found. Training a new model with default lookback of 60 days.")
        if not train_model(stock_symbol, 60):
            print("Training failed. Cannot proceed.", file=sys.stderr)
            sys.exit(1)
            
    run_streamlit_app()

if __name__ == "__main__":
    main()