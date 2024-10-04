# centralized_nlp_package/utils/config.py

from dataclasses import dataclass, field
from typing import Any, Dict, List
import hydra
from omegaconf import OmegaConf, MISSING


@dataclass(frozen=True)
class PathsConfig:
    model_artifacts_path: str = ""

@dataclass(frozen=True)
class BlobFilenameConfig:
    litigious_flnm: str = ""
    complex_flnm: str = ""
    uncertianity_flnm: str = ""
    syllable_flnm: str = ""
    vocab_pos_flnm: str = ""
    vocab_neg_flnm: str = ""
    contraction_flnm: str = ""
    stop_words_flnm: str = ""
    negate_words_flnm: str = ""

@dataclass
class SnowflakeConfig:
    user: str = MISSING
    password: str = MISSING
    account: str = MISSING
    warehouse: str = MISSING
    database: str = MISSING
    schema: str = MISSING

@dataclass
class DaskConfig:
    n_workers: int = 32

@dataclass
class Word2VecConfig:
    vector_size: int = 300
    window: int = 5
    min_count: int = 10
    workers: int = 16
    epochs: int = 15
    gen_bigram: bool = False
    bigram_threshold: int = 2
    model_save_path: str = "path/to/save/model/{min_year}_{min_month}_{max_year}_{max_month}_v1.model"

@dataclass
class PreprocessingConfig:
    spacy_model: str = ""
    additional_stop_words: List[str] = ""
    max_length: int = 1000000000

@dataclass
class PsycholinguisticsConfig:
    filt_sections: List[str] = field(default_factory=list)

@dataclass
class Config:
    snowflake: SnowflakeConfig = SnowflakeConfig()
    dask: DaskConfig = DaskConfig()
    word2vec: Word2VecConfig = Word2VecConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    blob_filenames: BlobFilenameConfig = BlobFilenameConfig()
    paths: PathsConfig = PathsConfig()

def get_config(config_path: str = "../configs", config_name: str = "config") -> Config:
    """
    Loads the Hydra configuration.

    Args:
        config_path (str): Path to the configuration directory.
        config_name (str): Name of the main configuration file.

    Returns:
        Config: Configuration object.
    """
    cfg = hydra.compose(config_name=config_name, config_path=config_path)
    # Convert OmegaConf to a structured Config dataclass
    config = hydra.utils.instantiate(cfg, _recursive_=False)
    return config
