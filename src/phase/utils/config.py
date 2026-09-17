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
    return config


def save_config(config, file_path):
    """Save configuration to a YAML file.

    Args:
        config (dict): Configuration dictionary
        file_path (str): Path to save the config file
    """
    import yaml

    with open(file_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
