# backend/__init__.py — NAPRAWIONY: minimalny, bez circular imports
"""Backend moduły TMIV - podstawowe importy."""

# Import tylko najważniejszych klas, bez zagnieżdżania
from .ml_integration import (
    ModelConfig,
    TrainingResult, 
    train_model_comprehensive,
    save_model_artifacts,
    load_model_artifacts
)

from .utils import (
    seed_everything,
    hash_dataframe_signature,
    infer_problem_type,
    validate_dataframe,
    get_openai_key_from_envs
)

__all__ = [
    # ML Integration
    'ModelConfig',
    'TrainingResult',
    'train_model_comprehensive', 
    'save_model_artifacts',
    'load_model_artifacts',
    
    # Utils (podstawowe)
    'seed_everything',
    'hash_dataframe_signature', 
    'infer_problem_type',
    'validate_dataframe',
    'get_openai_key_from_envs'
]