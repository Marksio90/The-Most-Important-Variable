# settings.py — centralna konfiguracja TMIV (DEV/PROD, flagi, limity, silniki)
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import List, Optional, Any, Dict
import os

# optional: Streamlit secrets (gdy aplikacja działa w Streamlit)
try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except Exception:
    HAS_STREAMLIT = False

# optional: pydantic v1/v2 (jeśli brak, użyjemy fallbacku z dataclass)
try:
    try:
        from pydantic_settings import BaseSettings  # v2 style
        PydanticBase = BaseSettings  # type: ignore
        PYD_VER = 2
    except Exception:
        from pydantic import BaseSettings  # type: ignore
        PydanticBase = BaseSettings  # type: ignore
        PYD_VER = 1
    HAS_PYDANTIC = True
except Exception:
    HAS_PYDANTIC = False
    PydanticBase = object  # type: ignore
    PYD_VER = 0

# optional: dotenv (lokalne .env podczas dev)
try:
    from dotenv import load_dotenv  # type: ignore
    HAS_DOTENV = True
except Exception:
    HAS_DOTENV = False


# ==========================
#  Enum silników ML dla UI
# ==========================
class MLEngine(Enum):
    AUTO = "auto"
    SKLEARN = "sklearn"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    PYCARET = "pycaret"


# ==========================
#  Fallback bez Pydantic
# ==========================
@dataclass
class _DataclassSettings:
    # Ogólne
    app_name: str = "TMIV"
    app_env: str = "DEV"  # DEV | PROD
    debug: bool = True

    # Dane / Upload
    data_max_file_size_mb: int = 200
    data_supported_formats: List[str] = (".csv", ".xlsx", ".xls")

    # ML / funkcje opcjonalne
    enable_shap: bool = False
    enable_xgboost: bool = True
    enable_lightgbm: bool = True
    enable_catboost: bool = True

    # Trening / walidacja
    default_random_state: int = 42
    default_cv_folds: int = 3
    default_test_size: float = 0.2

    # Telemetria / diagnostyka
    track_telemetry: bool = False
    show_debug_panel: bool = True

    # Ścieżki
    output_dir: str = "tmiv_out"
    history_db_path: str = "tmiv_out/history.sqlite"

    # LLM / klucze (tylko flaga; realny klucz pobieramy w utils.get_openai_key_from_envs)
    llm_enabled_by_default: bool = False

    # Rejestr modeli
    models_dir: str = "tmiv_out/models"


# ==========================
#  Settings przez Pydantic
# ==========================
if HAS_PYDANTIC:

    class Settings(PydanticBase):  # type: ignore[misc]
        # Ogólne
        app_name: str = "TMIV"
        app_env: str = "DEV"  # DEV | PROD
        debug: bool = True

        # Dane / Upload
        data_max_file_size_mb: int = 200
        data_supported_formats: List[str] = [".csv", ".xlsx", ".xls"]

        # ML / funkcje opcjonalne
        enable_shap: bool = False
        enable_xgboost: bool = True
        enable_lightgbm: bool = True
        enable_catboost: bool = True

        # Trening / walidacja
        default_random_state: int = 42
        default_cv_folds: int = 3
        default_test_size: float = 0.2

        # Telemetria / diagnostyka
        track_telemetry: bool = False
        show_debug_panel: bool = True

        # Ścieżki
        output_dir: str = "tmiv_out"
        history_db_path: str = "tmiv_out/history.sqlite"

        # LLM
        llm_enabled_by_default: bool = False

        # Rejestr modeli
        models_dir: str = "tmiv_out/models"

        class Config:  # pydantic v1
            env_file = ".env"
            env_prefix = "TMIV_"
            case_sensitive = False

        model_config = {
            "env_file": ".env",
            "extra": "ignore"  # pydantic v2
        }

else:
    # brak pydantic – używamy dataclass
    Settings = _DataclassSettings  # type: ignore[misc]


# ==========================
#  Wczytanie .env / secrets
# ==========================
def _load_env_chain() -> None:
    """
    Porządek:
      1) st.secrets (o ile Streamlit) — na potrzeby 'logic', ale te wartości i tak sczyta Pydantic/ENV
      2) .env przez dotenv (DEV)
      3) os.environ (PROD)
    Pydantic (jeśli jest) i tak wczyta env automatycznie; ta funkcja jest po to,
    żeby w dev try rozprowadzić zmienne z .env do os.environ (gdy dotenv dostępny).
    """
    # 1) Streamlit secrets nie trafiają automatycznie do os.environ – traktujemy informacyjnie
    # (klucze LLM pobieramy bezpośrednio przez utils.get_openai_key_from_envs)

    # 2) .env → os.environ (tylko jeśli mamy python-dotenv)
    if HAS_DOTENV:
        # domyślne .env w katalogu projektu
        load_dotenv(override=False)


def _override_from_secrets(s: Settings) -> Settings:
    """
    Drobne nadpisania z st.secrets (jeśli Streamlit udostępnia), np. profile/limity.
    Nie nadpisuje agresywnie – tylko jeśli są dostępne i typu bool/int/str/list.
    """
    if not HAS_STREAMLIT:
        return s

    try:
        sec = st.secrets  # type: ignore[attr-defined]
    except Exception:
        return s

    def _maybe_set(attr: str, key: str):
        nonlocal s
        if key in sec:
            try:
                val = sec[key]
                # lekkie rzutowania na typy prymitywne
                if isinstance(getattr(s, attr), bool):
                    setattr(s, attr, bool(val))
                elif isinstance(getattr(s, attr), int):
                    setattr(s, attr, int(val))
                elif isinstance(getattr(s, attr), float):
                    setattr(s, attr, float(val))
                elif isinstance(getattr(s, attr), list) and isinstance(val, (list, tuple)):
                    setattr(s, attr, list(val))
                elif isinstance(getattr(s, attr), str):
                    setattr(s, attr, str(val))
            except Exception:
                pass

    # przykładowe mapowania
    _maybe_set("app_env", "APP_ENV")
    _maybe_set("data_max_file_size_mb", "DATA_MAX_FILE_SIZE_MB")
    _maybe_set("track_telemetry", "TRACK_TELEMETRY")
    _maybe_set("show_debug_panel", "SHOW_DEBUG_PANEL")
    _maybe_set("enable_shap", "ENABLE_SHAP")
    _maybe_set("enable_xgboost", "ENABLE_XGBOOST")
    _maybe_set("enable_lightgbm", "ENABLE_LIGHTGBM")
    _maybe_set("enable_catboost", "ENABLE_CATBOOST")

    return s


# ==========================
#  Public API
# ==========================
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Singleton konfiguracyjny.
    - ładuje .env (jeśli dotenv),
    - tworzy Settings (Pydantic albo dataclass),
    - delikatnie nadpisuje z st.secrets (gdy Streamlit).
    """
    _load_env_chain()

    if HAS_PYDANTIC:
        # Pydantic sam przeczyta ENV + .env (jeśli obecne)
        s = Settings()  # type: ignore[call-arg]
    else:
        # Fallback: dataclass + ręczne mapowanie minimalnych ENV
        s = Settings()  # type: ignore[call-arg]
        s.app_name = os.getenv("TMIV_APP_NAME", s.app_name)
        s.app_env = os.getenv("TMIV_APP_ENV", s.app_env)
        s.debug = os.getenv("TMIV_DEBUG", str(s.debug)).lower() == "true"

        s.data_max_file_size_mb = int(os.getenv("TMIV_DATA_MAX_FILE_SIZE_MB", s.data_max_file_size_mb))
        fmts = os.getenv("TMIV_DATA_SUPPORTED_FORMATS", None)
        if fmts:
            s.data_supported_formats = [p.strip() for p in fmts.split(",") if p.strip()]

        s.enable_shap = os.getenv("TMIV_ENABLE_SHAP", str(s.enable_shap)).lower() == "true"
        s.enable_xgboost = os.getenv("TMIV_ENABLE_XGBOOST", str(s.enable_xgboost)).lower() == "true"
        s.enable_lightgbm = os.getenv("TMIV_ENABLE_LIGHTGBM", str(s.enable_lightgbm)).lower() == "true"
        s.enable_catboost = os.getenv("TMIV_ENABLE_CATBOOST", str(s.enable_catboost)).lower() == "true"

        s.default_random_state = int(os.getenv("TMIV_DEFAULT_RANDOM_STATE", s.default_random_state))
        s.default_cv_folds = int(os.getenv("TMIV_DEFAULT_CV_FOLDS", s.default_cv_folds))
        s.default_test_size = float(os.getenv("TMIV_DEFAULT_TEST_SIZE", s.default_test_size))

        s.track_telemetry = os.getenv("TMIV_TRACK_TELEMETRY", str(s.track_telemetry)).lower() == "true"
        s.show_debug_panel = os.getenv("TMIV_SHOW_DEBUG_PANEL", str(s.show_debug_panel)).lower() == "true"

        s.output_dir = os.getenv("TMIV_OUTPUT_DIR", s.output_dir)
        s.history_db_path = os.getenv("TMIV_HISTORY_DB_PATH", s.history_db_path)
        s.models_dir = os.getenv("TMIV_MODELS_DIR", s.models_dir)

        s.llm_enabled_by_default = os.getenv("TMIV_LLM_ENABLED_BY_DEFAULT", str(s.llm_enabled_by_default)).lower() == "true"

    # Nadpisz z secrets (o ile Streamlit)
    s = _override_from_secrets(s)

    # Tryb PROD → domyślnie mniej hałasu
    if (getattr(s, "app_env", "DEV") or "DEV").upper() == "PROD":
        try:
            s.debug = False
            s.show_debug_panel = False
        except Exception:
            pass

    return s


__all__ = ["get_settings", "Settings", "MLEngine"]
