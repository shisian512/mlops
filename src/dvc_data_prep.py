# src/dvc_data_prep.py
"""
DVC Data Preparation Module

This module provides functionality to prepare data tracked by Data Version Control (DVC).
It handles checking out a specific DVC version (git commit hash or tag) and pulling
the associated data, ensuring reproducibility in the ML pipeline.

The module uses Git for version control of DVC files and DVC for actual data versioning,
allowing the ML pipeline to work with specific versions of datasets.

Typical usage:
    python dvc_data_prep.py <dvc_data_path> <dvc_version>

Where:
    - dvc_data_path: Path to the DVC-tracked data folder
    - dvc_version: Git commit hash, branch, or tag for the DVC files
"""

# Standard library imports
import os
import subprocess
import sys

def prepare_dvc_data(dvc_data_path, dvc_version):
    """
    Checks out a specific DVC version and pulls the corresponding data.
    
    This function performs the following steps:
    1. Verifies DVC is installed and available
    2. Checks out the specified git version containing the DVC files
    3. Pulls the data tracked by DVC at the specified path
    
    Args:
        dvc_data_path (str): The path to the DVC-tracked folder relative to project root
        dvc_version (str): The git reference (commit hash, branch, or tag) to checkout
        
    Returns:
        None
        
    Raises:
        SystemExit: If DVC is not installed or if Git/DVC commands fail
    """
    print(f"Preparing data version '{dvc_version}' for training...")
    
    # Check if DVC is installed and configured
    try:
        # Run DVC version command to verify installation
        subprocess.run(["dvc", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        # Exit if DVC command is not found in PATH
        print("Error: DVC command not found. Please ensure DVC is installed.")
        sys.exit(1)

    # Change to the project directory containing the .dvc files
    # The DVC command needs to be run from the root of the DVC project
    try:
        # Checkout the specified git version
        # This ensures the .dvc files match the version we want to pull
        subprocess.run(["git", "checkout", dvc_version], check=True)
        print(f"Git checkout successful for DVC version {dvc_version}.")

        # Pull the data tracked by DVC
        # This downloads the actual data files based on the .dvc file references
        subprocess.run(["dvc", "pull", dvc_data_path], check=True)
        print(f"DVC pull successful for data at '{dvc_data_path}'.")

    except subprocess.CalledProcessError as e:
        # Handle errors from Git or DVC commands
        print(f"Error executing DVC or Git command: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
        sys.exit(1)
    
    print("Data preparation complete.")


if __name__ == "__main__":
    # Command-line argument validation
    if len(sys.argv) != 3:
        print("Usage: python dvc_data_prep.py <dvc_data_path> <dvc_version>")
        print("  dvc_data_path: Path to the DVC-tracked data folder")
        print("  dvc_version: Git commit hash, branch, or tag for the DVC files")
        sys.exit(1)
    
    # Execute the data preparation with command-line arguments
    prepare_dvc_data(sys.argv[1], sys.argv[2])

