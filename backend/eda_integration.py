"""
Backend EDA Integration - Kompletna implementacja
Zawiera AdvancedColumnAnalyzer i SmartDataPreprocessor
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import warnings
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import time

# Importy dla OpenAI (opcjonalne)
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMERATORY I KONFIGURACJA
# ============================================================================

class ColumnType(Enum):
    """Typy kolumn dla lepszej kategoryzacji."""
    DATETIME = "Data/czas (timestamp)"
    PRICE = "Cena / koszt (zmienna ciągła)"
    CATEGORY = "Kategoria (zmienna kategoryczna)"
    VOLUME = "Wielkość / ilość (zmienna ciągła)"
    NUMERIC = "Zmienna numeryczna"
    TEXT = "Zmienna kategoryczna/tekstowa"
    ID = "Identyfikator"
    BINARY = "Zmienna binarna"

class DataQuality(Enum):
    """Poziomy jakości danych."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class PreprocessingConfig:
    """Konfiguracja dla automatycznego preprocessingu."""
    # Ogólne
    remove_duplicates: bool = True
    handle_missing: bool = True
    
    # Kolumny
    drop_constant_threshold: float = 0.95  # Usuń jeśli >95% wartości to ta sama
    drop_missing_threshold: float = 0.90   # Usuń jeśli >90% wartości to NaN
    
    # Kategorie wysokiej kardinalności
    max_categories: int = 30
    category_threshold: int = 50  # Zastosuj capping jeśli >50 unikalnych wartości
    
    # Wartości odstające
    enable_outlier_treatment: bool = True
    outlier_method: str = "winsorize"  # "winsorize", "clip", "remove"
    lower_quantile: float = 0.005
    upper_quantile: float = 0.995
    
    # Daty
    parse_dates: bool = True
    create_date_features: bool = True
    date_features: List[str] = field(default_factory=lambda: ["year", "month", "day", "dayofweek", "quarter", "is_weekend"])
    
    # Imputacja
    numeric_imputation: str = "median"  # "mean", "median", "mode"
    categorical_imputation: str = "mode"  # "mode", "constant"
    constant_value: str = "MISSING"

@dataclass
class ColumnAnalysis:
    """Wynik analizy pojedynczej kolumny."""
    name: str
    dtype: str
    column_type: ColumnType
    quality: DataQuality
    unique_count: int
    missing_count: int
    missing_percentage: float
    description: str
    recommendations: List[str] = field(default_factory=list)
    sample_values: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PreprocessingReport:
    """Raport z preprocessingu danych."""
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    dropped_columns: Dict[str, List[str]] = field(default_factory=dict)
    created_columns: List[str] = field(default_factory=list)
    transformations: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    
    def summary(self) -> str:
        """Zwraca podsumowanie preprocessingu."""
        dropped_total = sum(len(cols) for cols in self.dropped_columns.values())
        created_total = len(self.created_columns)
        
        return f"""
## 📊 Podsumowanie Preprocessingu

**Rozmiar danych:**
- Przed: {self.original_shape[0]:,} wierszy × {self.original_shape[1]} kolumn
- Po: {self.final_shape[0]:,} wierszy × {self.final_shape[1]} kolumn

**Zmiany kolumn:**
- Usunięte: {dropped_total} kolumn
- Utworzone: {created_total} kolumn

**Czas przetwarzania:** {self.processing_time:.2f}s
"""

# ============================================================================
# ADVANCED COLUMN ANALYZER
# ============================================================================

class AdvancedColumnAnalyzer:
    """Zaawansowany analizator kolumn z integracją OpenAI."""
    
    def __init__(self, enable_llm: bool = False, openai_api_key: Optional[str] = None):
        self.enable_llm = enable_llm and HAS_OPENAI
        self.openai_api_key = openai_api_key
        
        if self.enable_llm and self.openai_api_key:
            openai.api_key = self.openai_api_key
    
    def analyze_column(self, series: pd.Series, column_name: str) -> ColumnAnalysis:
        """Analizuje pojedynczą kolumnę."""
        
        # Podstawowe statystyki
        dtype = str(series.dtype)
        unique_count = series.nunique()
        missing_count = series.isna().sum()
        missing_percentage = (missing_count / len(series)) * 100
        
        # Określ typ kolumny
        column_type = self._determine_column_type(series, column_name)
        
        # Ocena jakości
        quality = self._assess_quality(series, missing_percentage, unique_count)
        
        # Statystyki specyficzne dla typu
        statistics = self._calculate_statistics(series, column_type)
        
        # Próbki wartości
        sample_values = self._get_sample_values(series)
        
        # Opis kolumny
        description = self._generate_description(series, column_name, column_type)
        
        # Rekomendacje
        recommendations = self._generate_recommendations(series, column_type, quality)
        
        return ColumnAnalysis(
            name=column_name,
            dtype=dtype,
            column_type=column_type,
            quality=quality,
            unique_count=unique_count,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            description=description,
            recommendations=recommendations,
            sample_values=sample_values,
            statistics=statistics
        )
    
    def analyze_dataset(self, df: pd.DataFrame) -> List[ColumnAnalysis]:
        """Analizuje cały dataset."""
        analyses = []
        
        for column in df.columns:
            try:
                analysis = self.analyze_column(df[column], column)
                analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Błąd podczas analizy kolumny {column}: {e}")
                # Dodaj podstawową analizę w przypadku błędu
                analyses.append(ColumnAnalysis(
                    name=column,
                    dtype=str(df[column].dtype),
                    column_type=ColumnType.TEXT,
                    quality=DataQuality.POOR,
                    unique_count=0,
                    missing_count=len(df),
                    missing_percentage=100.0,
                    description=f"Błąd analizy: {e}",
                    recommendations=["Sprawdź dane w tej kolumnie"]
                ))
        
        return analyses
    
    def _determine_column_type(self, series: pd.Series, column_name: str) -> ColumnType:
        """Określa typ kolumny na podstawie danych i nazwy."""
        
        name_lower = column_name.lower()
        
        # Sprawdź po nazwie kolumny
        if any(keyword in name_lower for keyword in ['id', 'key', 'index', 'uuid']):
            return ColumnType.ID
        
        if any(keyword in name_lower for keyword in ['date', 'time', 'timestamp', 'created', 'updated']):
            return ColumnType.DATETIME
        
        if any(keyword in name_lower for keyword in ['price', 'cost', 'amount', 'revenue', 'value', 'salary']):
            return ColumnType.PRICE
        
        if any(keyword in name_lower for keyword in ['volume', 'count', 'quantity', 'size', 'length']):
            return ColumnType.VOLUME
        
        # Sprawdź po typie danych
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnType.DATETIME
        
        if pd.api.types.is_bool_dtype(series):
            return ColumnType.BINARY
        
        # Dla numerycznych
        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = series.nunique() / len(series)
            
            # Jeśli mało unikalnych wartości, prawdopodobnie kategoria
            if unique_ratio < 0.05 or series.nunique() <= 10:
                return ColumnType.CATEGORY
            
            # Sprawdź czy binarne (0/1, True/False)
            unique_vals = set(series.dropna().unique())
            if unique_vals.issubset({0, 1}) or unique_vals.issubset({True, False}):
                return ColumnType.BINARY
            
            return ColumnType.NUMERIC
        
        # Dla tekstowych/object
        if series.dtype == 'object':
            unique_ratio = series.nunique() / len(series)
            
            # Wysoka unikalność = prawdopodobnie tekst/ID
            if unique_ratio > 0.8:
                return ColumnType.TEXT
            
            return ColumnType.CATEGORY
        
        return ColumnType.TEXT
    
    def _assess_quality(self, series: pd.Series, missing_percentage: float, unique_count: int) -> DataQuality:
        """Ocenia jakość kolumny."""
        
        total_rows = len(series)
        
        # Kryteria jakości
        if missing_percentage > 70:
            return DataQuality.POOR
        
        if unique_count == 0 or (unique_count == 1 and missing_percentage < 100):
            return DataQuality.POOR
        
        # Sprawdź czy wszystkie wartości unikalne (prawdopodobnie ID)
        if unique_count == total_rows and missing_percentage < 5:
            return DataQuality.FAIR  # Może być ID
        
        # Dobra jakość
        if missing_percentage < 5 and unique_count > 1:
            return DataQuality.EXCELLENT
        
        if missing_percentage < 20 and unique_count > 1:
            return DataQuality.GOOD
        
        return DataQuality.FAIR
    
    def _calculate_statistics(self, series: pd.Series, column_type: ColumnType) -> Dict[str, Any]:
        """Oblicza statystyki specyficzne dla typu kolumny."""
        
        stats = {}
        
        try:
            if column_type in [ColumnType.NUMERIC, ColumnType.PRICE, ColumnType.VOLUME]:
                numeric_series = pd.to_numeric(series, errors='coerce')
                stats.update({
                    'mean': float(numeric_series.mean()) if not numeric_series.isna().all() else None,
                    'median': float(numeric_series.median()) if not numeric_series.isna().all() else None,
                    'std': float(numeric_series.std()) if not numeric_series.isna().all() else None,
                    'min': float(numeric_series.min()) if not numeric_series.isna().all() else None,
                    'max': float(numeric_series.max()) if not numeric_series.isna().all() else None,
                    'skewness': float(numeric_series.skew()) if not numeric_series.isna().all() else None
                })
            
            elif column_type in [ColumnType.CATEGORY, ColumnType.TEXT, ColumnType.BINARY]:
                value_counts = series.value_counts()
                stats.update({
                    'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'unique_values': int(series.nunique()),
                    'mode': series.mode().iloc[0] if len(series.mode()) > 0 else None
                })
            
            elif column_type == ColumnType.DATETIME:
                try:
                    dt_series = pd.to_datetime(series, errors='coerce')
                    stats.update({
                        'min_date': dt_series.min().isoformat() if not dt_series.isna().all() else None,
                        'max_date': dt_series.max().isoformat() if not dt_series.isna().all() else None,
                        'date_range_days': (dt_series.max() - dt_series.min()).days if not dt_series.isna().all() else None
                    })
                except:
                    pass
                    
        except Exception as e:
            logger.warning(f"Błąd przy obliczaniu statystyk: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def _get_sample_values(self, series: pd.Series, n: int = 5) -> List[str]:
        """Zwraca próbki wartości z kolumny."""
        
        try:
            # Weź unikalne wartości (bez NaN)
            unique_values = series.dropna().unique()
            
            # Ogranicz do n wartości
            if len(unique_values) > n:
                sample = np.random.choice(unique_values, size=n, replace=False)
            else:
                sample = unique_values
            
            # Konwertuj do stringów
            return [str(val) for val in sample]
            
        except Exception:
            return []
    
    def _generate_description(self, series: pd.Series, column_name: str, column_type: ColumnType) -> str:
        """Generuje opis kolumny."""
        
        # Spróbuj użyć LLM jeśli dostępne
        if self.enable_llm and self.openai_api_key:
            try:
                return self._get_llm_description(series, column_name, column_type)
            except Exception as e:
                logger.warning(f"Błąd LLM dla kolumny {column_name}: {e}")
        
        # Fallback na heurystyki
        return self._get_heuristic_description(series, column_name, column_type)
    
    def _get_llm_description(self, series: pd.Series, column_name: str, column_type: ColumnType) -> str:
        """Generuje opis używając OpenAI."""
        
        # Przygotuj dane o kolumnie
        sample_values = self._get_sample_values(series, 3)
        unique_count = series.nunique()
        missing_percentage = (series.isna().sum() / len(series)) * 100
        
        prompt = f"""
Przeanalizuj kolumnę danych i opisz ją w 1-2 zdaniach po polsku.

Nazwa kolumny: {column_name}
Typ: {column_type.value}
Unikalne wartości: {unique_count}
Brakujące dane: {missing_percentage:.1f}%
Przykładowe wartości: {', '.join(sample_values)}

Opisz zwięźle co ta kolumna reprezentuje i jaka może być jej rola w analizie danych.
"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Błąd OpenAI: {e}")
            return self._get_heuristic_description(series, column_name, column_type)
    
    def _get_heuristic_description(self, series: pd.Series, column_name: str, column_type: ColumnType) -> str:
        """Generuje opis używając heurystyk."""
        
        unique_count = series.nunique()
        missing_percentage = (series.isna().sum() / len(series)) * 100
        
        # Bazowe opisy dla typów
        type_descriptions = {
            ColumnType.ID: f"Kolumna identyfikująca z {unique_count} unikalnymi wartościami.",
            ColumnType.DATETIME: f"Kolumna czasowa obejmująca {unique_count} różnych momentów.",
            ColumnType.PRICE: f"Wartości cenowe/finansowe z {unique_count} różnymi poziomami.",
            ColumnType.VOLUME: f"Wielkości/ilości z {unique_count} różnymi wartościami.",
            ColumnType.NUMERIC: f"Zmienna numeryczna z {unique_count} różnymi wartościami.",
            ColumnType.CATEGORY: f"Zmienna kategoryczna z {unique_count} kategoriami.",
            ColumnType.TEXT: f"Dane tekstowe z {unique_count} unikalnymi wartościami.",
            ColumnType.BINARY: f"Zmienna binarna (tak/nie, 0/1)."
        }
        
        base_desc = type_descriptions.get(column_type, f"Kolumna z {unique_count} unikalnymi wartościami.")
        
        # Dodaj informację o brakujących danych
        if missing_percentage > 0:
            base_desc += f" Brakuje {missing_percentage:.1f}% wartości."
        
        return base_desc
    
    def _generate_recommendations(self, series: pd.Series, column_type: ColumnType, quality: DataQuality) -> List[str]:
        """Generuje rekomendacje dla kolumny."""
        
        recommendations = []
        missing_percentage = (series.isna().sum() / len(series)) * 100
        unique_count = series.nunique()
        
        # Rekomendacje dla jakości
        if quality == DataQuality.POOR:
            if missing_percentage > 50:
                recommendations.append("Rozważ usunięcie tej kolumny ze względu na dużo brakujących danych")
            if unique_count <= 1:
                recommendations.append("Kolumna ma stałą wartość - można ją usunąć")
        
        # Rekomendacje dla braków danych
        if missing_percentage > 20:
            recommendations.append("Zastosuj imputację lub usuń wiersze z brakami")
        
        # Rekomendacje specyficzne dla typu
        if column_type == ColumnType.CATEGORY:
            if unique_count > 50:
                recommendations.append("Wysoka kardinalność - rozważ grupowanie rzadkich kategorii")
        
        elif column_type in [ColumnType.NUMERIC, ColumnType.PRICE, ColumnType.VOLUME]:
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')
                if not numeric_series.isna().all():
                    skewness = abs(numeric_series.skew())
                    if skewness > 2:
                        recommendations.append("Rozkład jest skośny - rozważ transformację log/sqrt")
            except:
                pass
        
        elif column_type == ColumnType.ID:
            recommendations.append("Kolumna ID - usuń przed trenowaniem modelu")
        
        if not recommendations:
            recommendations.append("Kolumna wygląda poprawnie")
        
        return recommendations

# ============================================================================
# SMART DATA PREPROCESSOR
# ============================================================================

class SmartDataPreprocessor:
    """Inteligentny preprocessor danych z konfigurowalnymi opcjami."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.analyzer = AdvancedColumnAnalyzer()
    
    def preprocess(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, PreprocessingReport]:
        """Główna metoda preprocessingu."""
        
        start_time = time.time()
        original_shape = df.shape
        report = PreprocessingReport(original_shape=original_shape, final_shape=original_shape)
        
        # Kopia danych
        df_processed = df.copy()
        
        try:
            # 1. Usuń duplikaty
            if self.config.remove_duplicates:
                before_rows = len(df_processed)
                df_processed = df_processed.drop_duplicates()
                after_rows = len(df_processed)
                if before_rows != after_rows:
                    report.transformations['duplicates_removed'] = before_rows - after_rows
            
            # 2. Analizuj kolumny
            column_analyses = self.analyzer.analyze_dataset(df_processed)
            
            # 3. Usuń kolumny stałe i z dużymi brakami
            df_processed, dropped_constant = self._remove_constant_columns(df_processed)
            df_processed, dropped_missing = self._remove_high_missing_columns(df_processed)
            
            report.dropped_columns['constant'] = dropped_constant
            report.dropped_columns['high_missing'] = dropped_missing
            
            # 4. Parsuj daty i twórz cechy czasowe
            if self.config.parse_dates:
                df_processed, date_features = self._process_date_columns(df_processed, column_analyses)
                report.created_columns.extend(date_features)
            
            # 5. Obsłuż kategorie wysokiej kardinalności
            df_processed = self._handle_high_cardinality_categories(df_processed, column_analyses)
            
            # 6. Imputacja braków danych
            if self.config.handle_missing:
                df_processed = self._impute_missing_values(df_processed, target_column)
                report.transformations['missing_imputed'] = True
            
            # 7. Obsłuż wartości odstające
            if self.config.enable_outlier_treatment:
                df_processed = self._handle_outliers(df_processed, column_analyses, target_column)
                report.transformations['outliers_treated'] = True
            
            # Finalizuj raport
            report.final_shape = df_processed.shape
            report.processing_time = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Błąd podczas preprocessingu: {e}")
            report.warnings.append(f"Błąd preprocessingu: {e}")
            # Zwróć oryginalne dane w przypadku błędu
            df_processed = df.copy()
        
        return df_processed, report
    
    def _remove_constant_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Usuwa kolumny z stałymi wartościami."""
        
        dropped = []
        for col in df.columns:
            try:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < (1 - self.config.drop_constant_threshold):
                    dropped.append(col)
            except:
                pass
        
        if dropped:
            df = df.drop(columns=dropped)
            logger.info(f"Usunięto kolumny stałe: {dropped}")
        
        return df, dropped
    
    def _remove_high_missing_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Usuwa kolumny z dużą liczbą braków."""
        
        dropped = []
        for col in df.columns:
            missing_ratio = df[col].isna().sum() / len(df)
            if missing_ratio > self.config.drop_missing_threshold:
                dropped.append(col)
        
        if dropped:
            df = df.drop(columns=dropped)
            logger.info(f"Usunięto kolumny z dużymi brakami: {dropped}")
        
        return df, dropped
    
    def _process_date_columns(self, df: pd.DataFrame, analyses: List[ColumnAnalysis]) -> Tuple[pd.DataFrame, List[str]]:
        """Parsuje daty i tworzy cechy czasowe."""
        
        created_features = []
        
        for analysis in analyses:
            if analysis.column_type == ColumnType.DATETIME:
                col = analysis.name
                try:
                    # Parsuj datę
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    if self.config.create_date_features:
                        # Twórz cechy czasowe
                        if 'year' in self.config.date_features:
                            df[f"{col}_year"] = df[col].dt.year
                            created_features.append(f"{col}_year")
                        
                        if 'month' in self.config.date_features:
                            df[f"{col}_month"] = df[col].dt.month
                            created_features.append(f"{col}_month")
                        
                        if 'day' in self.config.date_features:
                            df[f"{col}_day"] = df[col].dt.day
                            created_features.append(f"{col}_day")
                        
                        if 'dayofweek' in self.config.date_features:
                            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                            created_features.append(f"{col}_dayofweek")
                        
                        if 'quarter' in self.config.date_features:
                            df[f"{col}_quarter"] = df[col].dt.quarter
                            created_features.append(f"{col}_quarter")
                        
                        if 'is_weekend' in self.config.date_features:
                            df[f"{col}_is_weekend"] = (df[col].dt.dayofweek >= 5).astype(int)
                            created_features.append(f"{col}_is_weekend")
                
                except Exception as e:
                    logger.warning(f"Błąd przetwarzania daty w kolumnie {col}: {e}")
        
        return df, created_features
    
    def _handle_high_cardinality_categories(self, df: pd.DataFrame, analyses: List[ColumnAnalysis]) -> pd.DataFrame:
        """Obsługuje kategorie wysokiej kardinalności."""
        
        for analysis in analyses:
            if analysis.column_type == ColumnType.CATEGORY and analysis.unique_count > self.config.category_threshold:
                col = analysis.name
                try:
                    # Zachowaj top N kategorii, resztę grupuj jako "Other"
                    top_categories = df[col].value_counts().head(self.config.max_categories).index.tolist()
                    df[col] = df[col].apply(lambda x: x if x in top_categories else "Other")
                    logger.info(f"Ograniczono kardinalność kolumny {col} do {self.config.max_categories} kategorii")
                except Exception as e:
                    logger.warning(f"Błąd ograniczania kardinalności w kolumnie {col}: {e}")
        
        return df
    
    def _impute_missing_values(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Imputuje brakujące wartości."""
        
        for col in df.columns:
            if col == target_column:
                continue  # Nie imputuj targetu
            
            if df[col].isna().sum() == 0:
                continue  # Brak braków
            
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Imputacja numeryczna
                    if self.config.numeric_imputation == "mean":
                        df[col] = df[col].fillna(df[col].mean())
                    elif self.config.numeric_imputation == "median":
                        df[col] = df[col].fillna(df[col].median())
                    elif self.config.numeric_imputation == "mode":
                        mode_val = df[col].mode()
                        if len(mode_val) > 0:
                            df[col] = df[col].fillna(mode_val.iloc[0])
                
                else:
                    # Imputacja kategoryczna
                    if self.config.categorical_imputation == "mode":
                        mode_val = df[col].mode()
                        if len(mode_val) > 0:
                            df[col] = df[col].fillna(mode_val.iloc[0])
                    elif self.config.categorical_imputation == "constant":
                        df[col] = df[col].fillna(self.config.constant_value)
                        
            except Exception as e:
                logger.warning(f"Błąd imputacji w kolumnie {col}: {e}")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, analyses: List[ColumnAnalysis], target_column: Optional[str] = None) -> pd.DataFrame:
        """Obsługuje wartości odstające."""
        
        for analysis in analyses:
            if analysis.column_type in [ColumnType.NUMERIC, ColumnType.PRICE, ColumnType.VOLUME]:
                col = analysis.name
                
                if col == target_column:
                    continue  # Nie modyfikuj targetu
                
                try:
                    if self.config.outlier_method == "winsorize":
                        lower = df[col].quantile(self.config.lower_quantile)
                        upper = df[col].quantile(self.config.upper_quantile)
                        df[col] = df[col].clip(lower=lower, upper=upper)
                    
                    elif self.config.outlier_method == "clip":
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        df[col] = df[col].clip(lower=lower, upper=upper)
                    
                    elif self.config.outlier_method == "remove":
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        df = df[(df[col] >= lower) & (df[col] <= upper)]
                        
                except Exception as e:
                    logger.warning(f"Błąd obsługi outlierów w kolumnie {col}: {e}")
        
        return df

# ============================================================================
# KOMPATYBILNOŚĆ WSTECZNA - ORYGINALNE FUNKCJE
# ============================================================================

@st.cache_data
def describe_column(column_name: str, sample_values: list, unique_count: int, 
                   missing_count: int, dtype: str, use_llm: bool = False) -> str:
    """Opisuje kolumnę - kompatybilność wsteczna."""
    
    analyzer = AdvancedColumnAnalyzer(enable_llm=use_llm)
    
    # Stwórz tymczasową serię dla analizy
    sample_series = pd.Series(sample_values + [np.nan] * missing_count)
    
    try:
        analysis = analyzer.analyze_column(sample_series, column_name)
        return analysis.description
    except Exception as e:
        logger.warning(f"Błąd opisu kolumny {column_name}: {e}")
        return f"Kolumna {column_name} typu {dtype} z {unique_count} unikalnymi wartościami."

def smart_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Inteligentne podsumowanie danych - kompatybilność wsteczna."""
    
    analyzer = AdvancedColumnAnalyzer()
    analyses = analyzer.analyze_dataset(df)
    
    # Klasyfikuj kolumny
    summary = {
        'basic_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicates': df.duplicated().sum()
        },
        'columns_by_type': {},
        'data_quality': {
            'excellent': 0,
            'good': 0,
            'fair': 0,
            'poor': 0
        },
        'missing_data': {},
        'recommendations': []
    }
    
    # Grupuj według typu
    for analysis in analyses:
        col_type = analysis.column_type.value
        if col_type not in summary['columns_by_type']:
            summary['columns_by_type'][col_type] = []
        summary['columns_by_type'][col_type].append(analysis.name)
        
        # Jakość danych
        summary['data_quality'][analysis.quality.value] += 1
        
        # Braki danych
        if analysis.missing_percentage > 0:
            summary['missing_data'][analysis.name] = analysis.missing_percentage
    
    # Globalne rekomendacje
    poor_quality_cols = [a.name for a in analyses if a.quality == DataQuality.POOR]
    if poor_quality_cols:
        summary['recommendations'].append(f"Sprawdź jakość kolumn: {', '.join(poor_quality_cols[:3])}")
    
    high_missing_cols = [a.name for a in analyses if a.missing_percentage > 30]
    if high_missing_cols:
        summary['recommendations'].append(f"Kolumny z dużymi brakami: {', '.join(high_missing_cols[:3])}")
    
    return summary

def auto_prepare_data(df: pd.DataFrame, target_column: str, 
                     create_date_features: bool = True,
                     handle_outliers: bool = True,
                     max_categories: int = 30) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Automatyczne przygotowanie danych - kompatybilność wsteczna."""
    
    # Stwórz konfigurację na podstawie parametrów
    config = PreprocessingConfig(
        create_date_features=create_date_features,
        enable_outlier_treatment=handle_outliers,
        max_categories=max_categories
    )
    
    preprocessor = SmartDataPreprocessor(config)
    df_processed, report = preprocessor.preprocess(df, target_column)
    
    # Konwertuj raport do starego formatu
    old_format_report = {
        'original_shape': report.original_shape,
        'final_shape': report.final_shape,
        'processing_time': report.processing_time,
        'changes': {
            'dropped_columns': sum(len(cols) for cols in report.dropped_columns.values()),
            'created_columns': len(report.created_columns),
            'transformations': report.transformations
        },
        'summary': report.summary()
    }
    
    return df_processed, old_format_report

# ============================================================================
# UTILITIES I HELPER FUNCTIONS
# ============================================================================

def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Ocenia jakość całego datasetu."""
    
    analyzer = AdvancedColumnAnalyzer()
    analyses = analyzer.analyze_dataset(df)
    
    # Zlicz jakość kolumn
    quality_counts = {quality.value: 0 for quality in DataQuality}
    for analysis in analyses:
        quality_counts[analysis.quality.value] += 1
    
    # Oblicz ogólny score jakości
    total_cols = len(analyses)
    quality_score = (
        quality_counts['excellent'] * 1.0 +
        quality_counts['good'] * 0.7 +
        quality_counts['fair'] * 0.4 +
        quality_counts['poor'] * 0.0
    ) / total_cols if total_cols > 0 else 0.0
    
    return {
        'overall_quality_score': quality_score,
        'quality_distribution': quality_counts,
        'total_columns': total_cols,
        'problematic_columns': [a.name for a in analyses if a.quality == DataQuality.POOR],
        'missing_data_issues': [a.name for a in analyses if a.missing_percentage > 50],
        'recommendations': _generate_dataset_recommendations(analyses)
    }

def _generate_dataset_recommendations(analyses: List[ColumnAnalysis]) -> List[str]:
    """Generuje rekomendacje dla całego datasetu."""
    
    recommendations = []
    
    # Problemy z jakością
    poor_cols = [a for a in analyses if a.quality == DataQuality.POOR]
    if poor_cols:
        recommendations.append(f"Usuń lub napraw {len(poor_cols)} kolumn niskiej jakości")
    
    # Braki danych
    high_missing = [a for a in analyses if a.missing_percentage > 30]
    if high_missing:
        recommendations.append(f"Zastosuj imputację dla {len(high_missing)} kolumn z dużymi brakami")
    
    # Kategorie wysokiej kardinalności
    high_card = [a for a in analyses if a.column_type == ColumnType.CATEGORY and a.unique_count > 50]
    if high_card:
        recommendations.append(f"Ogranicz kardinalność {len(high_card)} kolumn kategorycznych")
    
    # Potencjalne ID
    id_cols = [a for a in analyses if a.column_type == ColumnType.ID]
    if id_cols:
        recommendations.append(f"Usuń {len(id_cols)} kolumn identyfikacyjnych przed treningiem")
    
    if not recommendations:
        recommendations.append("Dataset wygląda dobrze przygotowany!")
    
    return recommendations

def get_preprocessing_config(advanced: bool = False) -> PreprocessingConfig:
    """Zwraca konfigurację preprocessingu."""
    
    if advanced:
        return PreprocessingConfig(
            remove_duplicates=True,
            handle_missing=True,
            drop_constant_threshold=0.98,
            drop_missing_threshold=0.80,
            max_categories=50,
            enable_outlier_treatment=True,
            outlier_method="winsorize",
            create_date_features=True,
            date_features=["year", "month", "day", "dayofweek", "quarter", "is_weekend"]
        )
    else:
        return PreprocessingConfig()

# ============================================================================
# TESTOWANIE I DEMO
# ============================================================================

def demo_analysis():
    """Demonstracja możliwości analizatora."""
    
    # Stwórz przykładowe dane
    np.random.seed(42)
    demo_data = {
        'id': range(1000),
        'name': [f'Product_{i}' for i in range(1000)],
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'price': np.random.lognormal(3, 1, 1000),
        'date_created': pd.date_range('2023-01-01', periods=1000, freq='D'),
        'is_active': np.random.choice([True, False], 1000),
        'description': [f'Opis produktu {i}' for i in range(1000)]
    }
    
    # Dodaj braki danych
    demo_data['price'][::10] = np.nan  # 10% braków
    demo_data['description'][::5] = np.nan  # 20% braków
    
    df = pd.DataFrame(demo_data)
    
    print("=== DEMO: Advanced Column Analyzer ===")
    
    # Analiza kolumn
    analyzer = AdvancedColumnAnalyzer()
    analyses = analyzer.analyze_dataset(df)
    
    for analysis in analyses:
        print(f"\n🔍 Kolumna: {analysis.name}")
        print(f"   Typ: {analysis.column_type.value}")
        print(f"   Jakość: {analysis.quality.value}")
        print(f"   Opis: {analysis.description}")
        if analysis.recommendations:
            print(f"   Rekomendacje: {analysis.recommendations[0]}")
    
    # Preprocessing
    print("\n=== DEMO: Smart Data Preprocessor ===")
    
    preprocessor = SmartDataPreprocessor()
    df_processed, report = preprocessor.preprocess(df, 'price')
    
    print(f"Kształt przed: {report.original_shape}")
    print(f"Kształt po: {report.final_shape}")
    print(f"Czas: {report.processing_time:.2f}s")
    print(f"Utworzone kolumny: {len(report.created_columns)}")
    
    return df_processed, report

if __name__ == "__main__":
    # Uruchom demo
    demo_analysis()