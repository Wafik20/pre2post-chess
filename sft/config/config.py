from omegaconf import OmegaConf

def load_config(config_path: str):
    """Load configuration from YAML file using OmegaConf."""
    return OmegaConf.load(config_path)