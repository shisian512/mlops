# src/dvc_data_prep.py
import subprocess
import sys
import os


def prepare_dvc_data(dvc_data_path, dvc_version):
    """
    Checks out a specific DVC version and pulls the data.

    Args:
        dvc_data_path (str): The path to the DVC-tracked folder.
        dvc_version (str): The DVC version (git commit hash or tag) to pull.
    """
    print(f"Preparing data version '{dvc_version}' for training...")

    # Check if DVC is installed and configured
    try:
        subprocess.run(["dvc", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        print("Error: DVC command not found. Please ensure DVC is installed.")
        sys.exit(1)

    # Change to the project directory containing the .dvc files
    # The DVC command needs to be run from the root of the DVC project
    try:
        # Assumes the script is run from a location where `os.getcwd()` is the project root
        # and the .dvc folder is in the same directory as `dvc_data_path`
        subprocess.run(["git", "checkout", dvc_version], check=True)
        print(f"Git checkout successful for DVC version {dvc_version}.")

        # Pull the data
        subprocess.run(["dvc", "pull", dvc_data_path], check=True)
        print(f"DVC pull successful for data at '{dvc_data_path}'.")

    except subprocess.CalledProcessError as e:
        print(f"Error executing DVC or Git command: {e.stderr.decode()}")
        sys.exit(1)

    print("Data preparation complete.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python dvc_data_prep.py <dvc_data_path> <dvc_version>")
        sys.exit(1)

    prepare_dvc_data(sys.argv[1], sys.argv[2])
