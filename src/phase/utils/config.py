import os

import yaml


def load_config(config_path):
    """Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        dict: Loaded configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            return os.path.expanduser(os.path.expandvars(value))
        return value

    return expand(config)


def save_config(config, file_path):
    """Save configuration to a YAML file.

    Args:
        config (dict): Configuration dictionary
        file_path (str): Path to save the config file
    """
    import yaml

    with open(file_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
