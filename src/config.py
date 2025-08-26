"""Configuration Module

This module provides utilities for loading configuration from YAML files.
It serves as a central configuration handler for the MLOps pipeline,
allowing different components to access consistent configuration settings.
"""

import yaml


def load_yaml_config(path: str) -> dict:
    """Load configuration from a YAML file.
    
    Args:
        path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Dictionary containing the configuration parameters loaded from the YAML file.
        
    Raises:
        FileNotFoundError: If the specified configuration file doesn't exist.
        yaml.YAMLError: If the YAML file has invalid syntax.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
