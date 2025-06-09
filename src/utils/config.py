import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "model": {
                "name": "microsoft/DialoGPT-medium",
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.9
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 5e-5,
                "epochs": 3,
                "warmup_steps": 100
            },
            "data": {
                "max_context_length": 2048,
                "overlap": 200
            },
            "paths": {
                "data_dir": "data/",
                "models_dir": "data/models/",
                "plots_dir": "plots/"
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default

config = Config()
