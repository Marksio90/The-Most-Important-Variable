# backend/__init__.py — NAPRAWIONY: tylko istniejące importy
"""Backend moduły TMIV - ML integration, smart target, utilities."""

# Import tylko tych funkcji które rzeczywiście istnieją w naszych paczach

from .ml_integration import (
    ModelConfig,
    TrainingResult, 
    train_model_comprehensive,
    save_model_artifacts,
    load_model_artifacts
)

from .smart_target import (
    SmartTargetSelector,
    TargetRecommendation,
    format_target_explanation,
    format_alternatives_list
)

from .utils import (
    seed_everything,
    hash_dataframe_signature,
    infer_problem_type,
    is_id_like,
    SmartTargetDetector,
    validate_dataframe,
    preprocess_column_names,
    detect_data_types,
    generate_ml_report,
    debug_dataframe,
    compare_dataframes,
    # Compatibility functions (nowo dodane)
    get_openai_key_from_envs,
    auto_pick_target,
    to_local,
    utc_now_iso_z
)

__all__ = [
    # ML Integration
    'ModelConfig',
    'TrainingResult',
    'train_model_comprehensive', 
    'save_model_artifacts',
    'load_model_artifacts',
    
    # Smart Target
    'SmartTargetSelector',
    'TargetRecommendation',
    'format_target_explanation',
    'format_alternatives_list',
    
    # Utils
    'seed_everything',
    'hash_dataframe_signature', 
    'infer_problem_type',
    'is_id_like',
    'SmartTargetDetector',
    'validate_dataframe',
    'preprocess_column_names',
    'detect_data_types',
    'generate_ml_report',
    'debug_dataframe',
    'compare_dataframes',
    'get_openai_key_from_envs',
    'auto_pick_target', 
    'to_local',
    'utc_now_iso_z'
]