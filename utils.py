import json
import os

def get_model_path():
    """
    Retrieve the path of the last logged model from a JSON file.
    
    Returns:
        str: The absolute path to the last logged model.
    """
    with open("last_model_path.json") as f:
        model_path = json.load(f)["model_path"]
    
    return os.path.abspath(model_path)