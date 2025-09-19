# backend/utils.py — NAPRAWIONE: lepszy system kluczy OpenAI + timezone
from __future__ import annotations

import hashlib
import os
import random
import re
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ==============================
# Reproducibility
# ==============================
def seed_everything(seed: int = 42) -> None:
    """
    Ustawia seed dla wszystkich popularnych generatorów liczb losowych.
    Zapewnia powtarzalność wyników.
    """
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass

    # Opcjonalnie dla TensorFlow/PyTorch jeśli dostępne
    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(seed)
    except Exception:
        pass

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
    except Exception:
        pass


# ==============================
# Sygnatura danych
# ==============================
def hash_dataframe_signature(df: pd.DataFrame, max_rows: int = 5000) -> str:
    """
    Tworzy "treściową" sygnaturę DataFrame (hash SHA-256),
    bazując na pierwszych `max_rows` wierszach zapisanych do CSV bez indeksu.
    To podejście dobrze wykrywa zmiany danych i jest stabilne między sesjami.
    """
    try:
        sample_csv = df.head(max_rows).to_csv(index=False).encode("utf-8")
        return hashlib.sha256(sample_csv).hexdigest()
    except Exception:
        # fallback: strukturalna sygnatura, gdyby CSV się nie powiodło
        signature_data = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": [str(dtype) for dtype in df.dtypes],
        }
        signature_str = str(signature_data).encode("utf-8")
        return hashlib.sha256(signature_str).hexdigest()


# ==============================
# Heurystyka typu problemu
# ==============================
def infer_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Inteligentne wykrywanie typu problemu ML na podstawie targetu.
    Zwraca 'classification' lub 'regression'.
    Reguły:
      - nienumeryczny → classification
      - <=2 unikalne → classification
      - <=20 unikalnych i unikalność <=5% → classification
      - całkowite dane z małą liczbą unikalnych → classification
      - inaczej → regression
    """
    if target_col not in df.columns:
        raise ValueError(f"Kolumna '{target_col}' nie istnieje w DataFrame")

    s = df[target_col].dropna()
    if len(s) == 0:
        # gdy pusto, lepiej nie wysypywać całego flow
        return "classification"

    is_num = pd.api.types.is_numeric_dtype(s)
    nunique = int(s.nunique())
    total = int(len(s))

    if not is_num:
        return "classification"

    if nunique <= 2:
        return "classification"

    if nunique <= 20 and (nunique / max(total, 1)) <= 0.05:
        return "classification"

    if pd.api.types.is_integer_dtype(s) and nunique <= min(50, max(1, total // 10)):
        return "classification"

    return "regression"


# ==============================
# Walidacja danych
# ==============================
def is_id_like(series: pd.Series, column_name: str) -> bool:
    """
    Sprawdza czy kolumna wygląda jak identyfikator (ID).
    ID nie powinno być używane jako target/feature do nauki (zwykle powoduje przeciek).
    """
    name = str(column_name).lower()
    id_patterns = [
        "id",
        "uuid",
        "guid",
        "key",
        "index",
        "idx",
        "_id",
        "userid",
        "user_id",
        "customer_id",
        "row_id",
        "record_id",
        "seq",
        "sequence",
    ]
    if any(p in name for p in id_patterns):
        return True

    # Sprawdź unikalność (100% lub prawie 100%)
    nunique = series.nunique(dropna=True)
    if nunique >= max(2, int(len(series) * 0.99)):
        return True

    # Sprawdź prostą sekwencję integerów
    try:
        if pd.api.types.is_integer_dtype(series):
            clean = series.dropna().astype(int)
            if len(clean) >= 5:
                vals = np.sort(clean.unique())
                diffs = np.diff(vals)
                if len(diffs) > 0 and np.all(diffs == diffs[0]) and diffs[0] in (1,):
                    return True
    except Exception:
        pass

    return False


def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Waliduje DataFrame pod kątem gotowości do ML.
    Zwraca raport: {valid, errors, warnings, suggestions, stats}
    """
    report: Dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "suggestions": [],
        "stats": {},
    }

    # Podstawowe sanity
    if df is None or df.empty:
        report["valid"] = False
        report["errors"].append("DataFrame jest pusty.")
        return report

    if len(df.columns) < 2:
        report["valid"] = False
        report["errors"].append("Za mało kolumn (wymagane minimum 2).")
        return report

    # Statystyki
    report["stats"] = {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
        "null_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # Braki danych
    null_cols = [c for c in df.columns if df[c].isna().any()]
    if null_cols:
        total_nulls = int(df[null_cols].isna().sum().sum())
        null_ratio = total_nulls / max(1, (len(df) * len(null_cols)))
        if null_ratio > 0.5:
            report["errors"].append(f"Bardzo dużo braków danych (~{null_ratio:.1%}).")
        elif null_ratio > 0.2:
            report["warnings"].append(f"Dużo braków danych (~{null_ratio:.1%}).")
            report["suggestions"].append("Rozważ imputację lub usunięcie kolumn z dużą liczbą braków.")

    # Duplikaty
    dup = report["stats"]["duplicate_rows"]
    if dup > 0:
        dup_ratio = dup / max(1, len(df))
        if dup_ratio > 0.1:
            report["warnings"].append(f"Dużo duplikatów (~{dup_ratio:.1%}).")
        report["suggestions"].append("Rozważ usunięcie zduplikowanych wierszy.")

    # Kolumny stałe
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        report["warnings"].append(f"Kolumny stałe: {constant_cols}")
        report["suggestions"].append("Usuń kolumny stałe — nie wnoszą informacji.")

    # Potencjalne ID
    id_cols = [c for c in df.columns if is_id_like(df[c], c)]
    if id_cols:
        report["suggestions"].append(f"Potencjalne ID do usunięcia: {id_cols}")

    # Bardzo wysoka kardynalność stringów
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        high_card = [c for c in obj_cols if df[c].nunique(dropna=True) > len(df) * 0.5]
        if high_card:
            report["warnings"].append(f"Wysoka kardynalność w kolumnach: {high_card}")
            report["suggestions"].append("Kolumny o wysokiej kardynalności mogą utrudniać modelowanie (One-Hot).")

    return report


# ==============================
# Typy danych / kategoryzacja
# ==============================
def preprocess_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Czyści nazwy kolumn: usuwa znaki specjalne, podwójne spacje/underscore'y,
    dba o unikalność.
    """
    dfc = df.copy()
    mapping: Dict[str, str] = {}
    seen: set = set()

    for col in dfc.columns:
        new = str(col).strip()
        new = re.sub(r"[^\w\s]", "_", new)  # znaki specjalne → _
        new = re.sub(r"[\s_]+", "_", new)   # wielokrotne spacje/_ → pojedynczy _
        new = new.strip("_") or "col"

        base = new
        i = 1
        while new in seen:
            new = f"{base}_{i}"
            i += 1

        mapping[col] = new
        seen.add(new)

    return dfc.rename(columns=mapping)


def detect_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Kategoryzuje kolumny wg typu danych i charakterystyk.
    Zwraca słownik list: numeric, categorical, boolean, datetime, text, id_like, constant, high_cardinality
    """
    out: Dict[str, List[str]] = {
        "numeric": [],
        "categorical": [],
        "boolean": [],
        "datetime": [],
        "text": [],
        "id_like": [],
        "constant": [],
        "high_cardinality": [],
    }

    for col in df.columns:
        s = df[col]

        # stałe
        if s.nunique(dropna=True) <= 1:
            out["constant"].append(col)
            continue

        # id-like
        if is_id_like(s, col):
            out["id_like"].append(col)
            continue

        if pd.api.types.is_bool_dtype(s):
            out["boolean"].append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(s):
            out["datetime"].append(col)
            continue

        if pd.api.types.is_numeric_dtype(s):
            out["numeric"].append(col)
            continue

        # tekst/kategorie
        unique_ratio = s.nunique(dropna=True) / max(1, len(s))
        if unique_ratio > 0.5:
            out["high_cardinality"].append(col)
        elif s.nunique(dropna=True) <= 50:
            out["categorical"].append(col)
        else:
            out["text"].append(col)

    return out


# ==============================
# Raport ML
# ==============================
def generate_ml_report(df: pd.DataFrame, target_col: Optional[str] = None) -> str:
    """
    Generuje zwięzły raport gotowości danych do ML (markdown).
    """
    lines: List[str] = ["# 📊 Raport analizy danych dla Machine Learning\n"]
    lines.append(f"**Rozmiar danych:** {len(df):,} wierszy × {len(df.columns)} kolumn")
    lines.append(f"**Pamięć:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB\n")

    val = validate_dataframe(df)

    if not val["valid"]:
        lines.append("## ❌ Błędy krytyczne")
        for e in val["errors"]:
            lines.append(f"- {e}")
        lines.append("")

    if val["warnings"]:
        lines.append("## ⚠️ Ostrzeżenia")
        for w in val["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    if val["suggestions"]:
        lines.append("## 💡 Zalecenia")
        for s in val["suggestions"]:
            lines.append(f"- {s}")
        lines.append("")

    # Typy kolumn
    types = detect_data_types(df)
    lines.append("## 📋 Kategoryzacja kolumn")
    for cat, cols in types.items():
        if cols:
            lines.append(f"**{cat.title()}:** {', '.join(map(str, cols))}")
    lines.append("")

    # Target
    if target_col and target_col in df.columns:
        s = df[target_col]
        ptype = infer_problem_type(df, target_col)
        lines.append(f"## 🎯 Analiza targetu: `{target_col}`")
        lines.append(f"**Typ problemu:** {ptype}")
        lines.append(f"**Typ danych:** {s.dtype}")
        lines.append(f"**Unikalne wartości:** {s.nunique(dropna=True)}")
        lines.append(f"**Braki:** {s.isna().sum()} ({s.isna().mean():.1%})")
        if ptype == "classification":
            vc = s.value_counts(dropna=True)
            lines.append(f"**Rozkład klas (top):** {dict(vc.head())}")
            if len(vc) > 1 and vc.min() > 0:
                lines.append(f"**Balans klas:** {vc.max() / vc.min():.1f}:1")
        lines.append("")

    # Następne kroki
    lines.append("## 🚀 Następne kroki")
    if val["valid"]:
        lines.append("✅ Dane są gotowe do treningu ML")
        if target_col:
            lines.append(f"✅ Target `{target_col}` został wybrany")
        else:
            lines.append("🔸 Wybierz kolumnę targetu aby rozpocząć trening")
    else:
        lines.append("❌ Napraw błędy krytyczne przed treningiem")

    return "\n".join(lines)


# ==============================
# Debug tools
# ==============================
def debug_dataframe(df: pd.DataFrame) -> None:
    """Szybki debug info o DataFrame (print-based)."""
    try:
        print("📊 DataFrame Debug Info:")
        print(f"   Shape: {df.shape}")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        print(f"   Nulls: {df.isna().sum().sum()} cells")
        print(f"   Dtypes: {dict(df.dtypes.value_counts())}")
        print("\n🔍 Sample data:")
        print(df.head(3))
    except Exception:
        pass


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
    """Porównuje dwa DataFrames i zwraca różnice (shape/columns/signature)."""
    out: Dict[str, Any] = {
        "shape_changed": df1.shape != df2.shape,
        "columns_changed": list(df1.columns) != list(df2.columns),
        "signature_changed": hash_dataframe_signature(df1) != hash_dataframe_signature(df2),
    }
    if out["columns_changed"]:
        out["added_columns"] = list(set(df2.columns) - set(df1.columns))
        out["removed_columns"] = list(set(df1.columns) - set(df2.columns))
    return out


# ==============================
# Smart Target (heurystyka)
# ==============================
class SmartTargetDetector:
    """
    Zaawansowany detektor potencjalnych targetów w DataFrame.
    Używa heurystyk i wzorców do rankingu kolumn.
    """
    def __init__(self):
        self.price_keywords = [
            "price",
            "cost",
            "value",
            "amount",
            "sum",
            "total",
            "revenue",
            "sales",
            "income",
            "profit",
            "salary",
            "fee",
            "charge",
            "rate",
            "wage",
        ]
        self.target_keywords = [
            "target",
            "label",
            "y",
            "outcome",
            "result",
            "prediction",
            "class",
            "category",
            "response",
        ]
        self.exclusion_keywords = [
            "id",
            "uuid",
            "key",
            "index",
            "date",
            "time",
            "name",
            "description",
            "comment",
            "note",
        ]

    def _detect_outliers(self, s: pd.Series) -> bool:
        if not pd.api.types.is_numeric_dtype(s):
            return False
        clean = s.dropna()
        if len(clean) < 4:
            return False
        try:
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            return bool(((clean < lo) | (clean > hi)).any())
        except Exception:
            return False

    def _assess_data_quality(self, s: pd.Series) -> Dict[str, Any]:
        return {
            "null_ratio": float(s.isna().mean()),
            "unique_ratio": float(s.nunique(dropna=True) / max(1, len(s))),
            "dtype": str(s.dtype),
            "has_outliers": self._detect_outliers(s),
        }

    def _score_column(self, df: pd.DataFrame, col: str) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []
        s = df[col]

        # ID-like
        if is_id_like(s, col):
            return 0.0, ["Wygląda jak identyfikator"]

        name = col.lower()

        # Słowa-klucze
        if any(k in name for k in self.price_keywords):
            score += 3.0
            reasons.append("Nazwa sugeruje wartość liczbową do predykcji (np. price/cost).")

        if any(k in name for k in self.target_keywords):
            score += 2.5
            reasons.append("Nazwa sugeruje klasyczny target (np. target/label/outcome).")

        if any(k in name for k in self.exclusion_keywords):
            score -= 2.0
            reasons.append("Nazwa sugeruje raczej kolumnę pomocniczą (id/date/name/etc.).")

        # Pozycja kolumny (ostatnia kolumna bywa targetem)
        if list(df.columns).index(col) == len(df.columns) - 1:
            score += 1.0
            reasons.append("Ostatnia kolumna (częsta konwencja dla targetu).")

        # Charakterystyka danych
        if pd.api.types.is_numeric_dtype(s):
            nunique = s.nunique(dropna=True)
            total = len(s.dropna())
            if total > 0:
                ratio = nunique / total
                if ratio > 0.5:
                    score += 1.5
                    reasons.append("Wysokie zróżnicowanie wartości (kandydat na regresję).")
                elif 2 <= nunique <= 20:
                    score += 2.0
                    reasons.append(f"Umiarkowana liczba poziomów ({nunique}) — dobra dla klasyfikacji.")
        else:
            # nienumeryczne zwykle do klasyfikacji (jeśli sensowne klasy)
            nunique = s.nunique(dropna=True)
            if 2 <= nunique <= 50:
                score += 1.0
                reasons.append(f"Nienumeryczne z rozsądną liczbą klas ({nunique}).")

        # Braki danych
        null_ratio = s.isna().mean()
        if null_ratio > 0.3:
            score -= 1.0
            reasons.append(f"Dużo braków danych ({null_ratio:.1%}).")

        return max(0.0, score), reasons

    def analyze_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for col in df.columns:
            sc, why = self._score_column(df, col)
            if sc > 0:
                out.append(
                    {
                        "column": col,
                        "score": float(sc),
                        "reasons": why,
                        "problem_type": infer_problem_type(df, col),
                        "data_quality": self._assess_data_quality(df[col]),
                    }
                )
        return sorted(out, key=lambda x: x["score"], reverse=True)


# ==============================
# Smart Target (krótki wrapper)
# ==============================
def auto_pick_target(df: pd.DataFrame) -> Optional[str]:
    """
    Szybki wybór targetu: używa SmartTargetDetector; jeśli brak rezultatów,
    zwraca ostatnią kolumnę (fallback).
    """
    try:
        det = SmartTargetDetector()
        cands = det.analyze_columns(df)
        if cands:
            return cands[0]["column"]
    except Exception:
        pass
    return df.columns[-1] if len(df.columns) else None


# ==============================
# OpenAI key helpers - NAPRAWIONE
# ==============================
def get_openai_key_from_envs() -> Optional[str]:
    """
    NAPRAWIONA wersja z lepszym debugowaniem.
    Priorytety:
      1) st.session_state.temp_openai_key (ręcznie ustawiony w sesji)
      2) st.secrets["OPENAI_API_KEY"] (Streamlit Cloud)
      3) .env / zmienne środowiskowe: OPENAI_API_KEY | OPENAI_KEY | OPENAI_SECRET_KEY
    """
    # 1) session_state
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "session_state") and st.session_state.get("temp_openai_key"):
            key = st.session_state["temp_openai_key"]
            if key and key.startswith("sk-"):
                print(f"[UTILS] ✅ Klucz OpenAI z session_state")
                return key
    except Exception:
        pass

    # 2) secrets
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            val = st.secrets["OPENAI_API_KEY"]
            if isinstance(val, str) and val.startswith("sk-"):
                print(f"[UTILS] ✅ Klucz OpenAI z st.secrets")
                return val
    except Exception:
        pass

    # 3) .env / ENV
    for key_name in ("OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_SECRET_KEY"):
        val = os.getenv(key_name)
        if val and isinstance(val, str) and val.startswith("sk-"):
            print(f"[UTILS] ✅ Klucz OpenAI z {key_name}")
            return val
    
    print(f"[UTILS] ❌ Brak prawidłowego klucza OpenAI")
    return None


def set_openai_key_temp(key: str) -> bool:
    """
    NAPRAWIONA wersja - ustawia klucz tymczasowo i wymusza odświeżenie cache.
    Zwraca True jeśli wygląda poprawnie.
    """
    if not isinstance(key, str) or not key.startswith("sk-"):
        return False
    
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "session_state"):
            st.session_state["temp_openai_key"] = key
            print(f"[UTILS] ✅ Ustawiono klucz w session_state")
    except Exception:
        pass
        
    os.environ["OPENAI_API_KEY"] = key
    print(f"[UTILS] ✅ Ustawiono klucz w os.environ")
    
    # Wymuś odświeżenie cache konfiguracji
    try:
        from config.settings import clear_settings_cache
        clear_settings_cache()
    except Exception:
        pass
    
    return True


def clear_openai_key():
    """Czyści klucz OpenAI ze wszystkich miejsc."""
    try:
        import streamlit as st
        if hasattr(st, "session_state") and "temp_openai_key" in st.session_state:
            del st.session_state["temp_openai_key"]
            print(f"[UTILS] 🗑️ Usunięto klucz z session_state")
    except Exception:
        pass
        
    for key_name in ("OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_SECRET_KEY"):
        if key_name in os.environ:
            del os.environ[key_name]
            print(f"[UTILS] 🗑️ Usunięto klucz z {key_name}")


# ==============================
# Czas / strefy - NAPRAWIONE
# ==============================
def get_local_timezone() -> str:
    """Zwraca lokalną strefę czasową (na podstawie systemu)."""
    try:
        import time
        return time.tzname[0] if time.daylight == 0 else time.tzname[1]
    except Exception:
        return "Europe/Warsaw"  # fallback dla Polski


def to_local(dt: Any, timezone_str: str = None) -> Any:
    """
    NAPRAWIONA wersja - konwertuje datetime do lokalnej strefy czasowej.
    Obsługuje pandas.Timestamp i python datetime.
    """
    if timezone_str is None:
        timezone_str = get_local_timezone()
        
    try:
        from datetime import datetime
        import pytz  # type: ignore

        if isinstance(dt, str):
            dt = pd.to_datetime(dt)

        # pandas Timestamp
        if hasattr(dt, "tz_localize") or hasattr(dt, "tz_convert"):
            if getattr(dt, "tz", None) is None:
                dt = dt.tz_localize("UTC")  # type: ignore[attr-defined]
            return dt.tz_convert(timezone_str)  # type: ignore[attr-defined]

        # python datetime
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                dt = pytz.UTC.localize(dt)
            return dt.astimezone(pytz.timezone(timezone_str))

        return dt
    except Exception:
        return dt


def utc_now_iso_z() -> str:
    """Zwraca aktualny czas UTC w formacie ISO 8601 zakończony 'Z'."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def local_now_iso() -> str:
    """Zwraca aktualny czas lokalny w formacie ISO 8601."""
    return datetime.now().isoformat()


def format_datetime_for_display(dt: datetime, include_seconds: bool = True) -> str:
    """Formatuje datetime dla wyświetlenia użytkownikowi (lokalna strefa)."""
    try:
        local_dt = to_local(dt)
        if include_seconds:
            return local_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return local_dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(dt)