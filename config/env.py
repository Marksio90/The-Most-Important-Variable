from pathlib import Path
import os
import streamlit as st
from dotenv import load_dotenv

_ENV_PATHS = [
    Path(".env"),
    Path("config/.env"),
]

def load_env_files() -> None:
    # Wczytaj wszystkie .env, kolejno (późniejsze nadpisują wcześniejsze)
    for p in _ENV_PATHS:
        if p.exists():
            load_dotenv(dotenv_path=p, override=True)

def _pull_from_st_secrets() -> str | None:
    # Obsłuż różne warianty klucza w secrets
    for k in ("OPENAI_API_KEY", "openai_api_key", "openai", "openaiKey"):
        try:
            v = st.secrets.get(k)  # nie rzuca KeyError
        except Exception:
            v = None
        if v:
            return str(v).strip()
    return None

def _pull_from_environ() -> str | None:
    for k in ("OPENAI_API_KEY", "openai_api_key"):
        v = os.environ.get(k)
        if v:
            return v.strip()
    return None

def get_openai_api_key() -> str | None:
    """
    Kolejność:
    1) st.secrets (deployment / Streamlit Cloud)
    2) os.environ (po load_env_files())
    Zabezpiecza i synchronizuje do os.environ + st.session_state.
    """
    key = _pull_from_st_secrets() or _pull_from_environ()
    if key:
        os.environ["OPENAI_API_KEY"] = key  # normalizujemy nazwę
        st.session_state["openai_api_key"] = key
        return key
    return None

def ensure_openai_api_key() -> bool:
    """
    Główna funkcja pomocnicza:
    - ładuje .env,
    - pobiera klucz,
    - zwraca True/False czy klucz jest dostępny.
    """
    load_env_files()
    key = get_openai_api_key()
    return bool(key)
