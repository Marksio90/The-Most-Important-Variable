# backend/utils.py — uniwersalne helpery TMIV (target, problem type, klucze, czas, seed)
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Any, List
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sys
import random

import numpy as np
import pandas as pd

# streamlit jest opcjonalne w niektórych skryptach CLI — import warunkowy
try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except Exception:
    HAS_STREAMLIT = False


# ==============================
#  Opcjonalne zależności — status
# ==============================
def _soft_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False

HAS_MATPLOTLIB = _soft_import("matplotlib")
HAS_SEABORN    = _soft_import("seaborn")
HAS_XGBOOST    = _soft_import("xgboost")
HAS_LGBM       = _soft_import("lightgbm")
HAS_CATBOOST   = _soft_import("catboost")

OPTIONALS = {
    "matplotlib": HAS_MATPLOTLIB,
    "seaborn": HAS_SEABORN,
    "xgboost": HAS_XGBOOST,
    "lightgbm": HAS_LGBM,
    "catboost": HAS_CATBOOST,
}


# =======================================
#  Klucze i konfiguracje (LLM / sekrety)
# =======================================
OPENAI_KEY_ENV_NAMES = [
    "OPENAI_API_KEY",
    "OPENAI_KEY",
    "TMIV_OPENAI_API_KEY",
]

def get_openai_key_from_envs() -> Optional[str]:
    """
    Pobiera klucz z (1) st.session_state.openai_api_key,
    (2) st.secrets["openai_api_key"|"OPENAI_API_KEY"],
    (3) zmienne środowiskowe.
    Zwraca None gdy brak lub nie przechodzi prostej walidacji.
    """
    key: Optional[str] = None

    # 1) session_state
    if HAS_STREAMLIT:
        try:
            key = st.session_state.get("openai_api_key") or st.session_state.get("OPENAI_API_KEY")
        except Exception:
            pass

    # 2) st.secrets
    if not key and HAS_STREAMLIT:
        try:
            for k in ("openai_api_key", "OPENAI_API_KEY"):
                if k in st.secrets:
                    key = st.secrets[k]  # type: ignore[index]
                    if key:
                        break
        except Exception:
            pass

    # 3) env
    if not key:
        for env_name in OPENAI_KEY_ENV_NAMES:
            v = os.environ.get(env_name)
            if v:
                key = v
                break

    # walidacja (prosta — unikamy fałszywych trafień)
    if key and _looks_like_openai_key(key):
        return key.strip()

    return None


def _looks_like_openai_key(value: str) -> bool:
    v = value.strip()
    # Najczęstsze prefiksy: sk-..., sk-proj-..., sk-org-...
    return bool(re.match(r"^sk-[-_A-Za-z0-9]{20,}$", v))


# =====================
#  Czas i strefy czasu
# =====================
def utc_now_iso_z() -> str:
    """Zwraca bieżący czas w UTC w formacie ISO-8601 z sufiksem 'Z'."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def to_utc_iso_z(dt: datetime) -> str:
    """Konwertuje datetime (aware/naive) do ISO-8601 UTC z 'Z'."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def to_local(dt_utc: datetime, tz_name: str = "Europe/Warsaw") -> datetime:
    """
    Konwertuje aware UTC -> lokalna strefa (domyślnie Europe/Warsaw).
    Jeśli dt jest naive, traktujemy jako UTC.
    """
    try:
        import zoneinfo  # py3.9+
        tz = zoneinfo.ZoneInfo(tz_name)
    except Exception:
        # fallback: pozostaw UTC
        tz = timezone.utc

    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(tz)


# ==========
#  Losowość
# ==========
def seed_everything(seed: int = 42) -> None:
    """Ustala seed dla numpy/random (sklearn zwykle przyjmuje random_state w configu)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False     # type: ignore
    except Exception:
        pass


# =============================
#  Target i typ problemu (ML)
# =============================

# ID-like name pattern
ID_LIKE_COL_PATTERNS = re.compile(
    r"(?:^|[_\- ])(id|uuid|guid|index|idx|kod|code|nr|no|number)(?:$|[_\- ])",
    re.IGNORECASE
)

# ---- NOWE: priorytet nazw cenowych (znormalizowanych) ----
PRICE_PRIORITY_ORDER = [
    "averageprice", "avgprice", "avg_price",  # różne warianty
    "targetprice", "target_price",
    "price",
    "closeprice", "close_price", "close",
]
# priorytet klasycznych nazw targetu (po cenowych)
CLASSIC_TARGET_ORDER = ["target", "label", "y"]

def _normalize_name(name: str) -> str:
    """Dolower + wywalenie znaków niealfanumerycznych (do porównań nazw)."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())

def is_id_like(series: pd.Series, colname: str) -> bool:
    """Heurystyka: kolumna wygląda jak identyfikator? (nazwa + wysoka unikalność)."""
    name_match = bool(ID_LIKE_COL_PATTERNS.search(colname or ""))
    try:
        n = len(series)
        nunique = series.nunique(dropna=True)
        high_uniqueness = n > 0 and (nunique / max(1, n)) > 0.98
    except Exception:
        high_uniqueness = False
    return name_match or high_uniqueness

def _all_missing(s: pd.Series) -> bool:
    try:
        return bool(s.isna().all())
    except Exception:
        return False

def _valid_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """Kolumny, które nie są ID-like i nie są w całości puste."""
    ex = set(exclude or [])
    out: List[str] = []
    for c in df.columns:
        if c in ex:
            continue
        try:
            if not is_id_like(df[c], c) and not _all_missing(df[c]):
                out.append(c)
        except Exception:
            continue
    return out

def _candidate_targets(df: pd.DataFrame) -> List[str]:
    """
    (Zachowane na potrzeby wewnętrzne) – nie używamy już jego priorytetu w detect_target,
    bo wprowadziliśmy własny porządek (price > classic > categorical > fallback).
    """
    if df is None or df.empty:
        return []
    return list(df.columns)

@dataclass
class SmartTargetDetector:
    """
    Heurystyczny detektor targetu.
    Priorytet:
      1) kolumny cenowe (AveragePrice/price/target_price/close itp.),
      2) 'target'/'label'/'y',
      3) kolumna kategoryczna o sensownej kardynalności,
      4) fallback: ostatnia sensowna kolumna z danych.
    Pomija ID-like i całe puste.
    """

    min_class_samples: int = 2           # aby uznać klasyfikację, każda klasa musi mieć >= 2 wystąpień
    max_class_cardinality: int = 50      # zbyt wiele klas -> mały priorytet dla klasyfikacji
    prefer_categorical: bool = True

    def detect_target(self, df: pd.DataFrame) -> Optional[str]:
        if df is None or df.empty:
            return None

        valid = _valid_columns(df)

        # 1) priorytet cenowy (wg listy, niezależnie od kolejności w df)
        norm_map = { _normalize_name(c): c for c in valid }
        for key in PRICE_PRIORITY_ORDER:
            k = _normalize_name(key)
            if k in norm_map:
                return norm_map[k]

        # 2) klasyczne nazwy targetu
        for key in CLASSIC_TARGET_ORDER:
            k = _normalize_name(key)
            if k in norm_map:
                return norm_map[k]

        # 3) preferuj kategoryczne (jeśli włączone)
        if self.prefer_categorical:
            cat_first = self._best_categorical(df, valid)
            if cat_first:
                return cat_first

        # 4) fallback – ostatnia „sensowna” kolumna (często target jest na końcu pliku)
        return valid[-1] if valid else None

    def _best_categorical(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        best: Optional[str] = None
        best_score = -1.0
        for c in candidates:
            s = df[c]
            try:
                nunique = s.nunique(dropna=True)
                n = len(s) if len(s) else 1
                if 1 < nunique <= self.max_class_cardinality:
                    vc = s.value_counts(dropna=True)
                    if (vc >= self.min_class_samples).all():
                        score = -(nunique)  # mniejsza kardynalność -> lepiej
                        if score > best_score:
                            best_score = score
                            best = c
            except Exception:
                continue
        return best

def auto_pick_target(df: pd.DataFrame) -> Optional[str]:
    """Szybki wybór targetu (alias dla detektora)."""
    return SmartTargetDetector().detect_target(df)

def infer_problem_type(df: pd.DataFrame, target: str) -> str:
    """
    Heurystyczna detekcja typu problemu:
    - jeśli dtype liczbowy i kardynalność > 20% liczby wierszy → 'regression'
    - jeśli dtype kategoryczny/tekstowy lub niska kardynalność → 'classification'
    """
    if target not in df.columns:
        return "other"
    s = df[target]

    # liczbowy?
    is_numeric = pd.api.types.is_numeric_dtype(s)
    try:
        nunique = s.nunique(dropna=True)
        n = len(s) if len(s) else 1
        high_card = nunique >= max(20, int(0.2 * n))
    except Exception:
        nunique, n, high_card = 2, 10, False

    if is_numeric and high_card:
        return "regression"

    # jeśli tylko dwie klasy → binary classification
    if nunique == 2:
        return "classification"

    # tekst/kategoria/mała kardynalność → classification
    if not is_numeric or not high_card:
        return "classification"

    return "other"


# ===========================
#  Drobne, ale przydatne I/O
# ===========================
def hash_dataframe_signature(df: pd.DataFrame, max_rows: int = 2000) -> str:
    """
    Tworzy krótki hash sygnatury danych (kolumny + sample wartości),
    przydatne do nazywania artefaktów modelu.
    """
    sample = df.head(max_rows).to_dict(orient="list")
    payload = json.dumps(
        {"columns": list(df.columns), "sample": sample},
        ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]  # krótszy identyfikator

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Spłaszcza słownik zagnieżdżony do kluczy 'a.b.c'."""
    items: List[Tuple[str, Any]] = []
    for k, v in (d or {}).items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ===========================
#  Przyjazne komunikaty/flags
# ===========================
def optional_dep_message() -> str:
    """
    Zwraca krótki status opcjonalnych zależności — można pokazać w zakładce Debug.
    """
    parts = []
    for mod, ok in OPTIONALS.items():
        parts.append(f"{mod}: {'OK' if ok else 'brak'}")
    return " | ".join(parts)


# ======= __all__ =======
__all__ = [
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
