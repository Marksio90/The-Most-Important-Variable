# backend/__init__.py — punkt wejścia pakietu backend
from __future__ import annotations

"""
Pakiet backend TMIV.

Zapewnia stabilne importy dla warstwy aplikacji:
    from backend import (
        SmartTargetDetector, auto_pick_target, infer_problem_type,
        get_openai_key_from_envs, seed_everything,
        utc_now_iso_z, to_utc_iso_z, to_local,
        HAS_MATPLOTLIB, HAS_SEABORN, HAS_XGBOOST, HAS_LGBM, HAS_CATBOOST,
        optional_dep_message, hash_dataframe_signature, flatten_dict
    )
"""

__version__ = "0.1.0"

# Re-eksport helperów z backend.utils
from .utils import (
    # klucze/sekrety
    get_openai_key_from_envs,
    # czas
    utc_now_iso_z, to_utc_iso_z, to_local,
    # seed
    seed_everything,
    # target/problem
    SmartTargetDetector, auto_pick_target, infer_problem_type, is_id_like,
    # opcjonalne zależności
    HAS_MATPLOTLIB, HAS_SEABORN, HAS_XGBOOST, HAS_LGBM, HAS_CATBOOST,
    optional_dep_message,
    # drobiazgi
    hash_dataframe_signature, flatten_dict,
)

__all__ = [
    # meta
    "__version__",
    # klucze/sekrety
    "get_openai_key_from_envs",
    # czas
    "utc_now_iso_z", "to_utc_iso_z", "to_local",
    # seed
    "seed_everything",
    # target/problem
    "SmartTargetDetector", "auto_pick_target", "infer_problem_type", "is_id_like",
    # opcjonalne zależności
    "HAS_MATPLOTLIB", "HAS_SEABORN", "HAS_XGBOOST", "HAS_LGBM", "HAS_CATBOOST",
    "optional_dep_message",
    # drobiazgi
    "hash_dataframe_signature", "flatten_dict",
]
