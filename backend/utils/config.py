import yaml
import os

class Config:
    def __init__(self, config_path="configs/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get(self, section, key=None):
        if section not in self.config:
            raise KeyError(f"Section '{section}' not found in config.")
        if key:
            if key not in self.config[section]:
                raise KeyError(f"Key '{key}' not found in section '{section}'.")
            return self.config[section][key]
        return self.config[section]

