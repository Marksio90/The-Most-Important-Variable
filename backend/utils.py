# backend/utils.py — KOMPLETNE: utilities i helper functions
from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd


def seed_everything(seed: int = 42) -> None:
    """
    Ustawia seed dla wszystkich generatorów liczb losowych.
    Zapewnia powtarzalność wyników.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Opcjonalnie dla TensorFlow/PyTorch jeśli dostępne
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def hash_dataframe_signature(df: pd.DataFrame) -> str:
    """
    Tworzy hash signature DataFrame dla trackingu zmian.
    Uwzględnia kształt, nazwy kolumn i typ danych.
    """
    signature_data = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': [str(dtype) for dtype in df.dtypes],
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    signature_str = str(signature_data)
    return hashlib.md5(signature_str.encode()).hexdigest()[:16]


def infer_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Inteligentne wykrywanie typu problemu ML na podstawie targetu.
    Zwraca 'classification' lub 'regression'.
    """
    if target_col not in df.columns:
        raise ValueError(f"Kolumna '{target_col}' nie istnieje w DataFrame")
    
    target_series = df[target_col]
    
    # Usuń wartości puste dla analizy
    target_clean = target_series.dropna()
    
    if len(target_clean) == 0:
        raise ValueError(f"Kolumna '{target_col}' jest całkowicie pusta")
    
    # Sprawdź typ danych
    is_numeric = pd.api.types.is_numeric_dtype(target_clean)
    unique_count = target_clean.nunique()
    total_count = len(target_clean)
    
    # Reguły klasyfikacji
    if not is_numeric:
        return "classification"
    
    if unique_count <= 2:
        return "classification"
    
    if unique_count <= 20 and unique_count / total_count <= 0.05:
        return "classification"
    
    # Sprawdź czy wartości wyglądają jak klasy (integer w małym zakresie)
    if is_numeric and target_clean.dtype in ['int64', 'int32', 'int16', 'int8']:
        if unique_count <= min(50, total_count * 0.1):
            return "classification"
    
    # Domyślnie regresja dla numerycznych
    return "regression"


def is_id_like(series: pd.Series, column_name: str) -> bool:
    """
    Sprawdza czy kolumna wygląda jak identyfikator (ID).
    ID nie powinno być używane jako target.
    """
    col_name_lower = column_name.lower()
    
    # Wzorce nazw ID
    id_patterns = [
        'id', 'uuid', 'guid', 'key', 'index', 'idx', 
        '_id', 'userid', 'user_id', 'customer_id',
        'row_id', 'record_id', 'seq', 'sequence'
    ]
    
    for pattern in id_patterns:
        if pattern in col_name_lower:
            return True
    
    # Sprawdź charakterystykę danych
    if pd.api.types.is_numeric_dtype(series):
        # Sprawdź czy to sekwencja (np. 1,2,3,4,...)
        clean_series = series.dropna()
        if len(clean_series) > 1:
            sorted_values = np.sort(clean_series.unique())
            if len(sorted_values) > 1:
                # Sprawdź czy różnice są stałe (sekwencja)
                diffs = np.diff(sorted_values)
                if len(set(diffs)) == 1 and diffs[0] == 1:
                    return True
    
    # Sprawdź unikalność (ID powinno być unikalne)
    if series.nunique() == len(series):
        if series.nunique() > len(series) * 0.9:  # >90% unikalnych
            return True
    
    return False


class SmartTargetDetector:
    """
    Zaawansowany detektor potencjalnych targetów w DataFrame.
    Używa heurystyk i wzorców do rankingu kolumn.
    """
    
    def __init__(self):
        self.price_keywords = [
            'price', 'cost', 'value', 'amount', 'sum', 'total',
            'revenue', 'sales', 'income', 'profit', 'salary',
            'fee', 'charge', 'rate', 'wage'
        ]
        
        self.target_keywords = [
            'target', 'label', 'y', 'outcome', 'result',
            'prediction', 'class', 'category', 'response'
        ]
        
        self.exclusion_keywords = [
            'id', 'uuid', 'key', 'index', 'date', 'time',
            'name', 'description', 'comment', 'note'
        ]
    
    def analyze_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analizuje wszystkie kolumny i zwraca ranking potencjalnych targetów.
        """
        candidates = []
        
        for col in df.columns:
            score, reasons = self._score_column(df, col)
            if score > 0:
                candidates.append({
                    'column': col,
                    'score': score,
                    'reasons': reasons,
                    'problem_type': infer_problem_type(df, col),
                    'data_quality': self._assess_data_quality(df[col])
                })
        
        # Sortuj według score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates
    
    def _score_column(self, df: pd.DataFrame, col: str) -> Tuple[float, List[str]]:
        """Ocenia kolumnę jako potencjalny target."""
        score = 0.0
        reasons = []
        series = df[col]
        
        # Sprawdź wykluczenia
        if is_id_like(series, col):
            return 0.0, ["Wygląda jak identyfikator"]
        
        col_lower = col.lower()
        
        # Bonus za słowa kluczowe w nazwie
        for keyword in self.price_keywords:
            if keyword in col_lower:
                score += 3.0
                reasons.append(f"Nazwa zawiera '{keyword}' (prawdopodobnie wartość do predykcji)")
                break
        
        for keyword in self.target_keywords:
            if keyword in col_lower:
                score += 2.5
                reasons.append(f"Nazwa zawiera '{keyword}' (klasyczny target)")
                break
        
        # Malus za wykluczenia
        for keyword in self.exclusion_keywords:
            if keyword in col_lower:
                score -= 2.0
                reasons.append(f"Nazwa zawiera '{keyword}' (prawdopodobnie nie target)")
        
        # Bonus za pozycję (ostatnia kolumna często to target)
        if list(df.columns).index(col) == len(df.columns) - 1:
            score += 1.0
            reasons.append("Ostatnia kolumna (częsta konwencja dla targetu)")
        
        # Analiza danych
        if pd.api.types.is_numeric_dtype(series):
            nunique = series.nunique()
            total = len(series)
            
            # Regresja - wysokie zróżnicowanie
            if nunique / total > 0.5:
                score += 1.5
                reasons.append("Wysokie zróżnicowanie wartości (kandydat na regresję)")
            
            # Klasyfikacja - średnie zróżnicowanie
            elif 2 <= nunique <= 20:
                score += 2.0
                reasons.append(f"Optymalna liczba klas ({nunique}) dla klasyfikacji")
        
        # Sprawdź braki danych (malus)
        null_ratio = series.isna().mean()
        if null_ratio > 0.3:
            score -= 1.0
            reasons.append(f"Dużo braków danych ({null_ratio:.1%})")
        
        return max(0.0, score), reasons
    
    def _assess_data_quality(self, series: pd.Series) -> Dict[str, Any]:
        """Ocenia jakość danych w kolumnie."""
        return {
            'null_ratio': series.isna().mean(),
            'unique_ratio': series.nunique() / len(series),
            'dtype': str(series.dtype),
            'has_outliers': self._detect_outliers(series)
        }
    
    def _detect_outliers(self, series: pd.Series) -> bool:
        """Wykrywa obecność outlierów w serii numerycznej."""
        if not pd.api.types.is_numeric_dtype(series):
            return False
        
        clean_series = series.dropna()
        if len(clean_series) < 4:
            return False
        
        try:
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
            return len(outliers) > 0
        except Exception:
            return False


def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Waliduje DataFrame pod kątem gotowości do ML.
    Zwraca raport z problemami i zaleceniami.
    """
    report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'suggestions': [],
        'stats': {}
    }
    
    # Podstawowe sprawdzenia
    if df.empty:
        report['valid'] = False
        report['errors'].append("DataFrame jest pusty")
        return report
    
    if len(df.columns) < 2:
        report['valid'] = False
        report['errors'].append("Za mało kolumn (minimum 2 wymagane)")
        return report
    
    # Statystyki
    report['stats'] = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'null_cells': df.isna().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Sprawdź braki danych
    null_cols = df.columns[df.isna().any()].tolist()
    if null_cols:
        total_nulls = df[null_cols].isna().sum().sum()
        null_ratio = total_nulls / (len(df) * len(null_cols))
        
        if null_ratio > 0.5:
            report['errors'].append(f"Bardzo dużo braków danych ({null_ratio:.1%})")
        elif null_ratio > 0.2:
            report['warnings'].append(f"Dużo braków danych ({null_ratio:.1%})")
            report['suggestions'].append("Rozważ uzupełnienie lub usunięcie kolumn z brakami")
    
    # Sprawdź duplikaty
    if report['stats']['duplicate_rows'] > 0:
        dup_ratio = report['stats']['duplicate_rows'] / len(df)
        if dup_ratio > 0.1:
            report['warnings'].append(f"Dużo duplikatów ({dup_ratio:.1%})")
            report['suggestions'].append("Rozważ usunięcie duplikatów")
    
    # Sprawdź kolumny stałe
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        report['warnings'].append(f"Kolumny stałe: {constant_cols}")
        report['suggestions'].append("Usuń kolumny stałe - nie wnoszą informacji")
    
    # Sprawdź potencjalne ID
    id_cols = [col for col in df.columns if is_id_like(df[col], col)]
    if id_cols:
        report['suggestions'].append(f"Potencjalne ID do usunięcia: {id_cols}")
    
    # Sprawdź typy danych
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        high_cardinality = [col for col in object_cols if df[col].nunique() > len(df) * 0.5]
        if high_cardinality:
            report['warnings'].append(f"Wysoką kardynalność: {high_cardinality}")
            report['suggestions'].append("Kolumny o wysokiej kardynalności mogą być problematyczne")
    
    return report


def preprocess_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Czyści nazwy kolumn z problematycznych znaków.
    Zapewnia kompatybilność z algorytmami ML.
    """
    df_copy = df.copy()
    
    # Mapa czyszczenia
    new_names = {}
    
    for col in df_copy.columns:
        new_name = str(col).strip()
        
        # Usuń problematyczne znaki
        new_name = re.sub(r'[^\w\s]', '_', new_name)
        
        # Usuń wielokrotne spacje/underscores
        new_name = re.sub(r'[\s_]+', '_', new_name)
        
        # Usuń underscore na początku/końcu
        new_name = new_name.strip('_')
        
        # Sprawdź czy nie jest puste
        if not new_name:
            new_name = f"col_{list(df_copy.columns).index(col)}"
        
        # Sprawdź unikalnośc
        original_new_name = new_name
        counter = 1
        while new_name in new_names.values():
            new_name = f"{original_new_name}_{counter}"
            counter += 1
        
        new_names[col] = new_name
    
    return df_copy.rename(columns=new_names)


def detect_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Kategoryzuje kolumny według typu danych i charakterystyk.
    """
    categorization = {
        'numeric': [],
        'categorical': [],
        'boolean': [],
        'datetime': [],
        'text': [],
        'id_like': [],
        'constant': [],
        'high_cardinality': []
    }
    
    for col in df.columns:
        series = df[col]
        
        # Sprawdź czy stałe
        if series.nunique() <= 1:
            categorization['constant'].append(col)
            continue
        
        # Sprawdź czy ID-like
        if is_id_like(series, col):
            categorization['id_like'].append(col)
            continue
        
        # Sprawdź typy danych
        if pd.api.types.is_numeric_dtype(series):
            categorization['numeric'].append(col)
        
        elif pd.api.types.is_bool_dtype(series):
            categorization['boolean'].append(col)
        
        elif pd.api.types.is_datetime64_any_dtype(series):
            categorization['datetime'].append(col)
        
        else:
            # Object/string - sprawdź karakterystyki
            unique_ratio = series.nunique() / len(series)
            
            if unique_ratio > 0.5:
                categorization['high_cardinality'].append(col)
            elif series.nunique() <= 50:
                categorization['categorical'].append(col)
            else:
                categorization['text'].append(col)
    
    return categorization


def generate_ml_report(df: pd.DataFrame, target_col: Optional[str] = None) -> str:
    """
    Generuje kompleksowy raport gotowości danych do ML.
    """
    report_lines = ["# 📊 Raport analizy danych dla Machine Learning\n"]
    
    # Podstawowe info
    report_lines.append(f"**Rozmiar danych:** {len(df):,} wierszy × {len(df.columns)} kolumn")
    report_lines.append(f"**Pamięć:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB\n")
    
    # Walidacja
    validation = validate_dataframe(df)
    
    if not validation['valid']:
        report_lines.append("## ❌ Błędy krytyczne")
        for error in validation['errors']:
            report_lines.append(f"- {error}")
        report_lines.append("")
    
    if validation['warnings']:
        report_lines.append("## ⚠️ Ostrzeżenia")
        for warning in validation['warnings']:
            report_lines.append(f"- {warning}")
        report_lines.append("")
    
    if validation['suggestions']:
        report_lines.append("## 💡 Zalecenia")
        for suggestion in validation['suggestions']:
            report_lines.append(f"- {suggestion}")
        report_lines.append("")
    
    # Analiza typów
    types = detect_data_types(df)
    report_lines.append("## 📋 Kategoryzacja kolumn")
    
    for category, columns in types.items():
        if columns:
            report_lines.append(f"**{category.title()}:** {', '.join(columns)}")
    
    report_lines.append("")
    
    # Analiza targetu
    if target_col and target_col in df.columns:
        report_lines.append(f"## 🎯 Analiza targetu: {target_col}")
        
        target_series = df[target_col]
        problem_type = infer_problem_type(df, target_col)
        
        report_lines.append(f"**Typ problemu:** {problem_type}")
        report_lines.append(f"**Typ danych:** {target_series.dtype}")
        report_lines.append(f"**Unikalne wartości:** {target_series.nunique()}")
        report_lines.append(f"**Braki:** {target_series.isna().sum()} ({target_series.isna().mean():.1%})")
        
        if problem_type == "classification":
            value_counts = target_series.value_counts()
            report_lines.append(f"**Rozkład klas:** {dict(value_counts.head())}")
            
            if len(value_counts) > 1:
                imbalance = value_counts.max() / value_counts.min()
                report_lines.append(f"**Balans klas:** {imbalance:.1f}:1")
        
        report_lines.append("")
    
    # Zalecenia finalne
    report_lines.append("## 🚀 Następne kroki")
    
    if validation['valid']:
        report_lines.append("✅ Dane są gotowe do treningu ML")
        if target_col:
            report_lines.append(f"✅ Target '{target_col}' został wybrany")
        else:
            report_lines.append("🔸 Wybierz kolumnę targetu aby rozpocząć trening")
    else:
        report_lines.append("❌ Napraw błędy krytyczne przed treningiem")
    
    return "\n".join(report_lines)


# Narzędzia do debugowania
def debug_dataframe(df: pd.DataFrame) -> None:
    """Szybki debug info o DataFrame."""
    print(f"📊 DataFrame Debug Info:")
    print(f"   Shape: {df.shape}")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    print(f"   Nulls: {df.isna().sum().sum()} cells")
    print(f"   Dtypes: {dict(df.dtypes.value_counts())}")
    
    # Próbka danych
    print(f"\n🔍 Sample data:")
    print(df.head(3))


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
    """Porównuje dwa DataFrames i zwraca różnice."""
    comparison = {
        'shape_changed': df1.shape != df2.shape,
        'columns_changed': list(df1.columns) != list(df2.columns),
        'signature_changed': hash_dataframe_signature(df1) != hash_dataframe_signature(df2)
    }
    
    if comparison['columns_changed']:
        comparison['added_columns'] = list(set(df2.columns) - set(df1.columns))
        comparison['removed_columns'] = list(set(df1.columns) - set(df2.columns))
    
    return comparison


# ================== COMPATIBILITY FUNCTIONS ==================
def get_openai_key_from_envs() -> Optional[str]:
    """
    Pobiera klucz OpenAI z zmiennych środowiskowych.
    Kompatybilność z starym kodem + ładowanie .env
    """
    import os
    
    # Spróbuj załadować .env
    try:
        from dotenv import load_dotenv
        load_dotenv()  # Ładuje .env do os.environ
    except ImportError:
        # python-dotenv nie jest zainstalowane - to OK
        pass
    
    # Sprawdź różne możliwe nazwy kluczy
    possible_keys = [
        'OPENAI_API_KEY',
        'OPENAI_KEY', 
        'OPENAI_SECRET_KEY'
    ]
    
    for key_name in possible_keys:
        key_value = os.getenv(key_name)
        if key_value and key_value.startswith('sk-'):
            return key_value
    
    # Sprawdź session state Streamlit
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and 'temp_openai_key' in st.session_state:
            return st.session_state.temp_openai_key
    except:
        pass
    
    return None


def auto_pick_target(df: pd.DataFrame) -> Optional[str]:
    """
    Automatycznie wybiera target z DataFrame.
    Wrapper dla SmartTargetDetector dla kompatybilności.
    """
    try:
        detector = SmartTargetDetector()
        candidates = detector.analyze_columns(df)
        return candidates[0]['column'] if candidates else None
    except Exception:
        # Fallback - ostatnia kolumna
        return df.columns[-1] if len(df.columns) > 0 else None


def to_local(dt: Any, timezone: str = "Europe/Warsaw") -> Any:
    """
    Konwertuje datetime do lokalnej strefy czasowej.
    Kompatybilność z starym kodem.
    """
    try:
        from datetime import datetime
        import pytz
        
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        
        if hasattr(dt, 'tz_localize') or hasattr(dt, 'tz_convert'):
            # Pandas Timestamp
            if dt.tz is None:
                dt = dt.tz_localize('UTC')
            return dt.tz_convert(timezone)
        
        elif isinstance(dt, datetime):
            # Python datetime
            if dt.tzinfo is None:
                dt = pytz.UTC.localize(dt)
            local_tz = pytz.timezone(timezone)
            return dt.astimezone(local_tz)
        
        return dt
        
    except Exception:
        return dt


def utc_now_iso_z() -> str:
    """
    Zwraca aktualny czas UTC w formacie ISO z Z.
    Kompatybilność z starym kodem.
    """
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')