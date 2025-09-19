# settings.py — centralna konfiguracja TMIV (NAPRAWIONA: lepsze ładowanie .env, debug info)
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import List, Optional, Any, Dict, Union
from pathlib import Path
import json
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
        from pydantic_settings import BaseSettings  # v2
        from pydantic import Field
        PydanticBase = BaseSettings  # type: ignore
        PYD_VER = 2
    except Exception:
        from pydantic import BaseSettings, Field  # v1  # type: ignore
        PydanticBase = BaseSettings  # type: ignore
        PYD_VER = 1
    HAS_PYDANTIC = True
except Exception:
    HAS_PYDANTIC = False
    PydanticBase = object  # type: ignore
    PYD_VER = 0
    def Field(*args, **kwargs): return None  # type: ignore

# optional: dotenv (lokalne .env podczas dev)
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    HAS_DOTENV = True
except Exception:
    HAS_DOTENV = False


# ==========================
#  Pomocnicze konwersje ENV
# ==========================
TRUE_SET  = {"1", "true", "t", "yes", "y", "on"}
FALSE_SET = {"0", "false", "f", "no", "n", "off"}

def _env_bool(val: Union[str, bool, int, None], default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val != 0
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in TRUE_SET: return True
    if s in FALSE_SET: return False
    return default

def _env_list(val: Union[str, List[str], None], default: List[str]) -> List[str]:
    if val is None:
        return list(default)
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if not s:
        return list(default)
    # spróbuj JSON
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    # fallback: CSV
    return [p.strip() for p in s.split(",") if p.strip()]


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
    data_supported_formats: List[str] = None  # np. [".csv", ".xlsx"]

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
    database_url: str = "sqlite:///tmiv_data.db"  # Dodane dla kompatybilności z app.py
    models_dir: str = "tmiv_out/models"

    # LLM / klucze (tylko flaga; realny klucz pobieramy w utils.get_openai_key_from_envs)
    llm_enabled_by_default: bool = False

    # Nowe ustawienia UI
    default_color_theme: str = "default"
    default_detail_level: str = "intermediate"
    default_chart_height: int = 500
    show_grid: bool = True
    interactive_charts: bool = True

    # Wyliczenia pomocnicze (uzupełniane po inicjalizacji)
    is_prod: bool = False
    is_dev: bool = True

    def __post_init__(self):
        if self.data_supported_formats is None:
            self.data_supported_formats = [".csv", ".xlsx", ".xls"]


# ==========================
#  Settings przez Pydantic
# ==========================
if HAS_PYDANTIC:
    if PYD_VER == 2:
        class Settings(PydanticBase):  # type: ignore[misc]
            # Ogólne
            app_name: str = "TMIV"
            app_env: str = "DEV"  # DEV | PROD
            debug: bool = True

            # Dane / Upload
            data_max_file_size_mb: int = 200
            data_supported_formats: List[str] = Field(default_factory=lambda: [".csv", ".xlsx", ".xls"])

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
            database_url: str = "sqlite:///tmiv_data.db"  # Dodane dla kompatybilności z app.py
            models_dir: str = "tmiv_out/models"

            # LLM
            llm_enabled_by_default: bool = False

            # Nowe ustawienia UI
            default_color_theme: str = "default"
            default_detail_level: str = "intermediate"
            default_chart_height: int = 500
            show_grid: bool = True
            interactive_charts: bool = True

            # pomocnicze (ustawiane po utworzeniu)
            is_prod: bool = False
            is_dev: bool = True

            # Pydantic v2 config
            model_config = {
                "env_file": ".env",
                "env_prefix": "TMIV_",
                "case_sensitive": False,
                "extra": "ignore",
            }
    else:
        class Settings(PydanticBase):  # type: ignore[misc]
            # Ogólne
            app_name: str = "TMIV"
            app_env: str = "DEV"  # DEV | PROD
            debug: bool = True

            # Dane / Upload
            data_max_file_size_mb: int = 200
            data_supported_formats: List[str] = Field(default_factory=lambda: [".csv", ".xlsx", ".xls"])

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
            database_url: str = "sqlite:///tmiv_data.db"  # Dodane dla kompatybilności z app.py
            models_dir: str = "tmiv_out/models"

            # LLM
            llm_enabled_by_default: bool = False

            # Nowe ustawienia UI
            default_color_theme: str = "default"
            default_detail_level: str = "intermediate"
            default_chart_height: int = 500
            show_grid: bool = True
            interactive_charts: bool = True

            # pomocnicze (ustawiane po utworzeniu)
            is_prod: bool = False
            is_dev: bool = True

            class Config:
                env_file = ".env"
                env_prefix = "TMIV_"
                case_sensitive = False
else:
    Settings = _DataclassSettings  # type: ignore[misc]


# ==========================
#  Wczytanie .env / secrets - NAPRAWIONE
# ==========================
def _load_env_chain() -> None:
    """
    NAPRAWIONA wersja: Próbuje znaleźć .env w bieżącym katalogu i katalogach nadrzędnych.
    Następnie ładuje zmienne do os.environ.
    """
    if not HAS_DOTENV:
        return
        
    try:
        # Znajdź .env file (w bieżącym katalogu lub wyżej)
        env_file = find_dotenv()
        if env_file:
            print(f"[SETTINGS] Znaleziono .env: {env_file}")
            load_dotenv(env_file, override=False)
            
            # Debug: sprawdź czy klucz został wczytany
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key and openai_key.startswith("sk-"):
                print(f"[SETTINGS] ✅ Wczytano klucz OpenAI z .env")
            else:
                print(f"[SETTINGS] ⚠️ Brak prawidłowego klucza OpenAI w .env")
        else:
            print(f"[SETTINGS] Nie znaleziono pliku .env")
            
        # Próbuj też z konkretną ścieżką
        local_env = Path(".env")
        if local_env.exists():
            load_dotenv(local_env, override=False)
            print(f"[SETTINGS] Wczytano .env z {local_env.absolute()}")
            
    except Exception as e:
        print(f"[SETTINGS] Błąd wczytywania .env: {e}")


def _has_streamlit_secrets() -> bool:
    """Bezpiecznie sprawdza czy Streamlit secrets są dostępne."""
    if not HAS_STREAMLIT:
        return False
    try:
        _ = st.secrets  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


def _override_from_secrets(s: Settings) -> Settings:
    """
    Delikatne nadpisania z st.secrets (jeśli dostępne), np. profile/limity.
    Nie nadpisuje agresywnie – tylko jeśli są dostępne i typu bool/int/str/list.
    """
    if not _has_streamlit_secrets():
        return s

    try:
        sec = st.secrets  # type: ignore[attr-defined]
    except Exception:
        return s

    def _maybe_set(attr: str, key: str):
        try:
            if key in sec:
                val = sec[key]
                cur = getattr(s, attr, None)
                if isinstance(cur, bool):
                    setattr(s, attr, _env_bool(val, cur))
                elif isinstance(cur, int):
                    setattr(s, attr, int(val))
                elif isinstance(cur, float):
                    setattr(s, attr, float(val))
                elif isinstance(cur, list):
                    setattr(s, attr, _env_list(val, cur))
                elif isinstance(cur, str):
                    setattr(s, attr, str(val))
        except Exception:
            pass

    # przykładowe mapowania (opcjonalne)
    _maybe_set("app_env", "APP_ENV")
    _maybe_set("data_max_file_size_mb", "DATA_MAX_FILE_SIZE_MB")
    _maybe_set("track_telemetry", "TRACK_TELEMETRY")
    _maybe_set("show_debug_panel", "SHOW_DEBUG_PANEL")
    _maybe_set("enable_shap", "ENABLE_SHAP")
    _maybe_set("enable_xgboost", "ENABLE_XGBOOST")
    _maybe_set("enable_lightgbm", "ENABLE_LIGHTGBM")
    _maybe_set("enable_catboost", "ENABLE_CATBOOST")
    _maybe_set("output_dir", "OUTPUT_DIR")
    _maybe_set("models_dir", "MODELS_DIR")
    _maybe_set("history_db_path", "HISTORY_DB_PATH")
    _maybe_set("database_url", "DATABASE_URL")  # Dodane

    return s


def _normalize_after_load(s: Settings) -> Settings:
    """Ujednolica typy/formaty po wczytaniu z ENV/secrets – niezależnie od backendu (pydantic/dataclass)."""
    # app_env + flagi trybu
    env_up = (getattr(s, "app_env", "DEV") or "DEV").upper()
    try:
        s.is_prod = env_up == "PROD"  # type: ignore[attr-defined]
        s.is_dev = not s.is_prod      # type: ignore[attr-defined]
    except Exception:
        pass

    # listy i booleany z .env (np. "1"/"0", ".csv,.xlsx")
    s.data_supported_formats = _env_list(getattr(s, "data_supported_formats", None),
                                         [".csv", ".xlsx", ".xls"])

    # upewnij się, że wartości liczbowe są poprawne
    try:
        s.data_max_file_size_mb = int(s.data_max_file_size_mb)
    except Exception:
        s.data_max_file_size_mb = 200

    # upewnij się, że ścieżki istnieją (tworzymy dopiero przy starcie aplikacji UI/CLI)
    for p in (getattr(s, "output_dir", "tmiv_out"),
              getattr(s, "models_dir", "tmiv_out/models")):
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    return s


# ==========================
#  Public API
# ==========================
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Singleton konfiguracyjny.
    - ładuje .env (jeśli dotenv) - NAPRAWIONE
    - tworzy Settings (Pydantic albo dataclass),
    - delikatnie nadpisuje z st.secrets (gdy Streamlit),
    - normalizuje typy i listy niezależnie od backendu.
    """
    print("[SETTINGS] Inicjalizacja konfiguracji...")
    _load_env_chain()

    if HAS_PYDANTIC:
        # Pydantic sam przeczyta ENV + .env (jeśli obecne)
        s = Settings()  # type: ignore[call-arg]
    else:
        # Fallback: dataclass + ręczne mapowanie ENV
        s = Settings()  # type: ignore[call-arg]
        s.app_name = os.getenv("TMIV_APP_NAME", s.app_name)
        s.app_env = os.getenv("TMIV_APP_ENV", s.app_env)
        s.debug = _env_bool(os.getenv("TMIV_DEBUG"), s.debug)

        s.data_max_file_size_mb = int(os.getenv("TMIV_DATA_MAX_FILE_SIZE_MB", s.data_max_file_size_mb))
        s.data_supported_formats = _env_list(os.getenv("TMIV_DATA_SUPPORTED_FORMATS"),
                                             s.data_supported_formats)

        s.enable_shap = _env_bool(os.getenv("TMIV_ENABLE_SHAP"), s.enable_shap)
        s.enable_xgboost = _env_bool(os.getenv("TMIV_ENABLE_XGBOOST"), s.enable_xgboost)
        s.enable_lightgbm = _env_bool(os.getenv("TMIV_ENABLE_LIGHTGBM"), s.enable_lightgbm)
        s.enable_catboost = _env_bool(os.getenv("TMIV_ENABLE_CATBOOST"), s.enable_catboost)

        s.default_random_state = int(os.getenv("TMIV_DEFAULT_RANDOM_STATE", s.default_random_state))
        s.default_cv_folds = int(os.getenv("TMIV_DEFAULT_CV_FOLDS", s.default_cv_folds))
        s.default_test_size = float(os.getenv("TMIV_DEFAULT_TEST_SIZE", s.default_test_size))

        s.track_telemetry = _env_bool(os.getenv("TMIV_TRACK_TELEMETRY"), s.track_telemetry)
        s.show_debug_panel = _env_bool(os.getenv("TMIV_SHOW_DEBUG_PANEL"), s.show_debug_panel)

        s.output_dir = os.getenv("TMIV_OUTPUT_DIR", s.output_dir)
        s.history_db_path = os.getenv("TMIV_HISTORY_DB_PATH", s.history_db_path)
        s.database_url = os.getenv("TMIV_DATABASE_URL", s.database_url)  # Dodane
        s.models_dir = os.getenv("TMIV_MODELS_DIR", s.models_dir)

        s.llm_enabled_by_default = _env_bool(os.getenv("TMIV_LLM_ENABLED_BY_DEFAULT"),
                                             s.llm_enabled_by_default)

    # Nadpisz z secrets (o ile Streamlit) - BEZPIECZNIE
    s = _override_from_secrets(s)

    # Normalize / sanity
    s = _normalize_after_load(s)

    # Tryb PROD → domyślnie mniej hałasu
    if getattr(s, "is_prod", False):
        try:
            s.debug = False
            s.show_debug_panel = False
        except Exception:
            pass

    print(f"[SETTINGS] ✅ Konfiguracja gotowa (debug: {getattr(s, 'debug', False)})")
    return s


def clear_settings_cache():
    """Czyści cache ustawień - przydatne po zmianie konfiguracji."""
    get_settings.cache_clear()
    print("[SETTINGS] Cache ustawień wyczyszczony")


# ==========================
#  Convenience helpers
# ==========================
def engines_enabled(settings: Optional[Settings] = None) -> Dict[str, bool]:
    s = settings or get_settings()
    return {
        "sklearn": True,  # zawsze
        "lightgbm": bool(getattr(s, "enable_lightgbm", True)),
        "xgboost": bool(getattr(s, "enable_xgboost", True)),
        "catboost": bool(getattr(s, "enable_catboost", True)),
        "pycaret": False,  # domyślnie wyłączony; włącz, jeśli używasz
    }

def as_dict(settings: Optional[Settings] = None) -> Dict[str, Any]:
    s = settings or get_settings()
    # Minimalny słownik do debug panelu
    return {
        "app_name": s.app_name,
        "app_env": s.app_env,
        "debug": s.debug,
        "data_max_file_size_mb": s.data_max_file_size_mb,
        "data_supported_formats": s.data_supported_formats,
        "default_random_state": s.default_random_state,
        "default_cv_folds": s.default_cv_folds,
        "default_test_size": s.default_test_size,
        "output_dir": s.output_dir,
        "models_dir": s.models_dir,
        "history_db_path": s.history_db_path,
        "database_url": getattr(s, "database_url", "sqlite:///tmiv_data.db"),  # Dodane
        "enable_xgboost": s.enable_xgboost,
        "enable_lightgbm": s.enable_lightgbm,
        "enable_catboost": s.enable_catboost,
        "enable_shap": s.enable_shap,
        "is_prod": getattr(s, "is_prod", False),
        "is_dev": getattr(s, "is_dev", True),
        "default_color_theme": getattr(s, "default_color_theme", "default"),
        "default_detail_level": getattr(s, "default_detail_level", "intermediate"),
        "default_chart_height": getattr(s, "default_chart_height", 500),
    }


__all__ = ["get_settings", "Settings", "MLEngine", "engines_enabled", "as_dict", "clear_settings_cache"]