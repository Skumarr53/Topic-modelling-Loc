from loguru import logger
from dataclasses import dataclass, field
from typing import Any, Dict, List
import hydra
from typing import Optional
from omegaconf import DictConfig,OmegaConf, MISSING

_config: Optional[DictConfig] = None

def get_config() -> DictConfig:
    global _config
    # If the configuration is not already loaded, initialize and compose it
    if _config is None:
        try:
            with hydra.initialize(config_path="../config"):
                _config = hydra.compose(config_name="config.yaml")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    return _config