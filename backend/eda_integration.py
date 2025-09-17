from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import logging
import warnings
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import hashlib

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class TransformationStatus(Enum):
    """Status transformacji danych."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TransformationResult:
    """Wynik pojedynczej transformacji."""
    status: TransformationStatus
    message: str
    data_changed: bool = False
    affected_columns: List[str] = field(default_factory=list)
    created_columns: List[str] = field(default_factory=list)
    dropped_columns: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PreprocessingConfig:
    """Uproszczona konfiguracja dla preprocessingu."""
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
class PreprocessingReport:
    """Kompleksowy raport z preprocessingu danych."""
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    transformation_results: List[TransformationResult] = field(default_factory=list)
    processing_time: float = 0.0
    
    def add_result(self, result: TransformationResult) -> None:
        """Dodaje wynik transformacji do raportu."""
        self.transformation_results.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Zwraca podsumowanie preprocessingu."""
        successful = sum(1 for r in self.transformation_results if r.status == TransformationStatus.SUCCESS)
        failed = sum(1 for r in self.transformation_results if r.status == TransformationStatus.FAILED)
        
        all_dropped = []
        all_created = []
        for result in self.transformation_results:
            all_dropped.extend(result.dropped_columns)
            all_created.extend(result.created_columns)
        
        return {
            "original_shape": self.original_shape,
            "final_shape": self.final_shape,
            "successful_transformations": successful,
            "failed_transformations": failed,
            "total_dropped_columns": len(set(all_dropped)),
            "total_created_columns": len(set(all_created)),
            "processing_time": self.processing_time
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje raport do słownika."""
        return {
            "summary": self.get_summary(),
            "transformations": [
                {
                    "status": result.status.value,
                    "message": result.message,
                    "data_changed": result.data_changed,
                    "affected_columns": result.affected_columns,
                    "created_columns": result.created_columns,
                    "dropped_columns": result.dropped_columns,
                    "metrics": result.metrics
                } for result in self.transformation_results
            ]
        }

# ============================================================================
# PROTOKOŁY I ABSTRAKCJE
# ============================================================================

class DataTransformer(Protocol):
    """Protokół dla transformatorów danych."""
    def transform(self, df: pd.DataFrame, config: PreprocessingConfig) -> Tuple[pd.DataFrame, TransformationResult]:
        """Wykonuje transformację danych."""
        ...

class ColumnAnalyzer(Protocol):
    """Protokół dla analizatorów kolumn."""
    def analyze_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analizuje kolumny i zwraca ich opisy."""
        ...

# ============================================================================
# TRANSFORMATORY DANYCH (PIPELINE PATTERN)
# ============================================================================

class BaseTransformer(ABC):
    """Bazowa klasa dla transformatorów danych."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def _execute_transform(self, df: pd.DataFrame, config: PreprocessingConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Wykonuje konkretną transformację."""
        pass
    
    def transform(self, df: pd.DataFrame, config: PreprocessingConfig) -> Tuple[pd.DataFrame, TransformationResult]:
        """Główna metoda transformacji z error handling."""
        try:
            df_transformed, metrics = self._execute_transform(df.copy(), config)
            
            # Sprawdź czy dane się zmieniły
            data_changed = not df.equals(df_transformed)
            
            result = TransformationResult(
                status=TransformationStatus.SUCCESS,
                message=f"{self.name} completed successfully",
                data_changed=data_changed,
                metrics=metrics
            )
            
            return df_transformed, result
            
        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            result = TransformationResult(
                status=TransformationStatus.FAILED,
                message=f"{self.name} failed: {str(e)}",
                data_changed=False
            )
            return df, result

class RemoveEmptyColumnsTransformer(BaseTransformer):
    """Usuwa kolumny z wysokim odsetkiem braków."""
    
    def __init__(self):
        super().__init__("Remove Empty Columns")
    
    def _execute_transform(self, df: pd.DataFrame, config: PreprocessingConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        empty_cols = []
        
        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df)
            if missing_pct >= config.drop_missing_threshold:
                empty_cols.append(col)
        
        if empty_cols:
            df = df.drop(columns=empty_cols)
        
        metrics = {
            "dropped_columns": empty_cols,
            "drop_threshold": config.drop_missing_threshold
        }
        
        return df, metrics

class RemoveConstantColumnsTransformer(BaseTransformer):
    """Usuwa kolumny stałe lub prawie stałe."""
    
    def __init__(self):
        super().__init__("Remove Constant Columns")
    
    def _execute_transform(self, df: pd.DataFrame, config: PreprocessingConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        constant_cols = []
        
        for col in df.columns:
            try:
                unique_ratio = df[col].nunique(dropna=True) / df[col].notna().sum()
                if unique_ratio < (1 - config.drop_constant_threshold) or df[col].nunique(dropna=True) <= 1:
                    constant_cols.append(col)
            except (ZeroDivisionError, ValueError):
                constant_cols.append(col)
        
        if constant_cols:
            df = df.drop(columns=constant_cols)
        
        metrics = {
            "dropped_columns": constant_cols,
            "constant_threshold": config.drop_constant_threshold
        }
        
        return df, metrics

class DateTimeTransformer(BaseTransformer):
    """Parsuje daty i tworzy cechy czasowe."""
    
    def __init__(self):
        super().__init__("DateTime Features")
    
    def _execute_transform(self, df: pd.DataFrame, config: PreprocessingConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if not config.parse_dates:
            return df, {"skipped": "Date parsing disabled"}
        
        date_cols = []
        created_features = []
        
        for col in df.columns:
            if self._is_datetime_column(df[col]):
                try:
                    dt_series = pd.to_datetime(df[col], errors='coerce')
                    
                    if config.create_date_features:
                        base_name = col.replace('_date', '').replace('_time', '').replace('_at', '')
                        
                        feature_mapping = {
                            "year": lambda x: x.dt.year,
                            "month": lambda x: x.dt.month,
                            "day": lambda x: x.dt.day,
                            "dayofweek": lambda x: x.dt.dayofweek,
                            "quarter": lambda x: x.dt.quarter,
                            "is_weekend": lambda x: (x.dt.dayofweek >= 5).astype(int)
                        }
                        
                        for feature_name in config.date_features:
                            if feature_name in feature_mapping:
                                new_col = f"{base_name}__{feature_name}"
                                df[new_col] = feature_mapping[feature_name](dt_series)
                                created_features.append(new_col)
                    
                    # Zachowaj oryginalną kolumnę jako datetime
                    df[col] = dt_series
                    date_cols.append(col)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse datetime column {col}: {e}")
        
        metrics = {
            "processed_date_columns": date_cols,
            "created_features": created_features
        }
        
        return df, metrics
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        """Sprawdza czy kolumna zawiera daty."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        if series.dtype == 'object':
            try:
                sample = series.dropna().head(min(50, len(series)))
                if len(sample) > 0:
                    pd.to_datetime(sample, errors='raise')
                    return True
            except (ValueError, TypeError):
                pass
        
        return False

class DuplicateRemovalTransformer(BaseTransformer):
    """Usuwa duplikaty."""
    
    def __init__(self):
        super().__init__("Remove Duplicates")
    
    def _execute_transform(self, df: pd.DataFrame, config: PreprocessingConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if not config.remove_duplicates:
            return df, {"skipped": "Duplicate removal disabled"}
        
        initial_rows = len(df)
        df_dedupe = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df_dedupe)
        
        metrics = {
            "duplicates_removed": duplicates_removed,
            "initial_rows": initial_rows,
            "final_rows": len(df_dedupe)
        }
        
        return df_dedupe, metrics

class HighCardinalityTransformer(BaseTransformer):
    """Obsługuje kategorie wysokiej kardynalności."""
    
    def __init__(self):
        super().__init__("High Cardinality Treatment")
    
    def _execute_transform(self, df: pd.DataFrame, config: PreprocessingConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        capped_cols = {}
        
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                unique_count = df[col].nunique()
                
                if unique_count > config.category_threshold:
                    # Zostaw top N kategorii, resztę zamień na 'OTHER'
                    value_counts = df[col].value_counts()
                    top_categories = set(value_counts.head(config.max_categories).index)
                    
                    df[col] = df[col].apply(lambda x: x if x in top_categories else 'OTHER')
                    capped_cols[col] = {
                        "original_unique": unique_count,
                        "capped_to": config.max_categories
                    }
        
        metrics = {
            "capped_columns": capped_cols,
            "category_threshold": config.category_threshold,
            "max_categories": config.max_categories
        }
        
        return df, metrics

class OutlierTransformer(BaseTransformer):
    """Obsługuje wartości odstające."""
    
    def __init__(self):
        super().__init__("Outlier Treatment")
    
    def _execute_transform(self, df: pd.DataFrame, config: PreprocessingConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if not config.enable_outlier_treatment:
            return df, {"skipped": "Outlier treatment disabled"}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        treated_cols = []
        
        for col in numeric_cols:
            try:
                if config.outlier_method == 'winsorize':
                    lower_bound = df[col].quantile(config.lower_quantile)
                    upper_bound = df[col].quantile(config.upper_quantile)
                    df[col] = df[col].clip(lower_bound, upper_bound)
                    treated_cols.append(col)
                    
                elif config.outlier_method == 'clip':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower_bound, upper_bound)
                    treated_cols.append(col)
                    
            except Exception as e:
                logger.warning(f"Failed to treat outliers in {col}: {e}")
        
        metrics = {
            "treated_columns": treated_cols,
            "method": config.outlier_method,
            "quantile_range": [config.lower_quantile, config.upper_quantile]
        }
        
        return df, metrics

class MissingValueTransformer(BaseTransformer):
    """Imputuje brakujące wartości."""
    
    def __init__(self):
        super().__init__("Missing Value Imputation")
    
    def _execute_transform(self, df: pd.DataFrame, config: PreprocessingConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if not config.handle_missing:
            return df, {"skipped": "Missing value handling disabled"}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        imputed_numeric = []
        imputed_categorical = []
        
        # Numeryczne
        for col in numeric_cols:
            if df[col].isna().any():
                if config.numeric_imputation == 'mean':
                    fill_value = df[col].mean()
                elif config.numeric_imputation == 'median':
                    fill_value = df[col].median()
                elif config.numeric_imputation == 'mode':
                    fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                else:
                    fill_value = 0
                
                df[col] = df[col].fillna(fill_value)
                imputed_numeric.append(col)
        
        # Kategoryczne
        for col in categorical_cols:
            if df[col].isna().any():
                if config.categorical_imputation == 'mode':
                    try:
                        fill_value = df[col].mode().iloc[0]
                    except (IndexError, AttributeError):
                        fill_value = config.constant_value
                else:
                    fill_value = config.constant_value
                
                df[col] = df[col].fillna(fill_value)
                imputed_categorical.append(col)
        
        metrics = {
            "imputed_numeric": imputed_numeric,
            "imputed_categorical": imputed_categorical,
            "numeric_strategy": config.numeric_imputation,
            "categorical_strategy": config.categorical_imputation
        }
        
        return df, metrics

# ============================================================================
# PIPELINE PREPROCESSINGU
# ============================================================================

class PreprocessingPipeline:
    """Pipeline składający się z komponentów transformacyjnych."""
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.transformers = self._create_default_pipeline()
    
    def _create_default_pipeline(self) -> List[BaseTransformer]:
        """Tworzy domyślny pipeline transformacji."""
        return [
            RemoveEmptyColumnsTransformer(),
            RemoveConstantColumnsTransformer(),
            DateTimeTransformer(),
            DuplicateRemovalTransformer(),
            HighCardinalityTransformer(),
            OutlierTransformer(),
            MissingValueTransformer(),
        ]
    
    def add_transformer(self, transformer: BaseTransformer) -> 'PreprocessingPipeline':
        """Dodaje transformer do pipeline."""
        self.transformers.append(transformer)
        return self
    
    def remove_transformer(self, transformer_name: str) -> 'PreprocessingPipeline':
        """Usuwa transformer z pipeline."""
        self.transformers = [t for t in self.transformers if t.name != transformer_name]
        return self
    
    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, PreprocessingReport]:
        """Wykonuje cały pipeline transformacji."""
        start_time = datetime.now()
        
        # Inicjalizuj raport
        report = PreprocessingReport(
            original_shape=df.shape,
            final_shape=df.shape
        )
        
        df_processed = df.copy()
        
        try:
            for transformer in self.transformers:
                df_processed, result = transformer.transform(df_processed, self.config)
                
                # Aktualizuj informacje o kolumnach w wyniku
                if hasattr(result, 'metrics') and isinstance(result.metrics, dict):
                    if 'dropped_columns' in result.metrics:
                        result.dropped_columns = result.metrics['dropped_columns']
                    if 'created_features' in result.metrics:
                        result.created_columns = result.metrics['created_features']
                
                report.add_result(result)
                
                # Loguj postęp
                if result.data_changed:
                    logger.info(f"{transformer.name}: {result.message}")
            
            # Aktualizuj finalne informacje
            report.final_shape = df_processed.shape
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            error_result = TransformationResult(
                status=TransformationStatus.FAILED,
                message=f"Pipeline error: {str(e)}",
                data_changed=False
            )
            report.add_result(error_result)
        
        finally:
            # Czas przetwarzania
            end_time = datetime.now()
            report.processing_time = (end_time - start_time).total_seconds()
        
        return df_processed, report

# ============================================================================
# ANALIZATORY KOLUMN
# ============================================================================

class SmartColumnAnalyzer:
    """Inteligentny analizator kolumn z konfigurowalnymi heurystykami."""
    
    def __init__(self, api_key: Optional[str] = None, use_ai: bool = True):
        self.api_key = api_key
        self.use_ai = use_ai and api_key
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Konfiguruje wzorce do rozpoznawania typów kolumn."""
        self.patterns = {
            ColumnType.DATETIME: {
                'keywords': ['date', 'time', 'timestamp', 'created', 'updated', 'dt'],
                'suffixes': ['_at', '_date', '_time'],
                'prefixes': ['date_', 'time_']
            },
            ColumnType.PRICE: {
                'keywords': ['price', 'cost', 'amount', 'value', 'revenue', 'fee', 'charge'],
                'suffixes': ['_price', '_cost', '_amount', '_fee'],
                'prefixes': ['price_', 'cost_', 'avg_']
            },
            ColumnType.VOLUME: {
                'keywords': ['volume', 'quantity', 'qty', 'count', 'total', 'sum', 'amount'],
                'suffixes': ['_qty', '_count', '_vol', '_total'],
                'prefixes': ['qty_', 'vol_', 'total_']
            },
            ColumnType.ID: {
                'keywords': ['id', 'key', 'index', 'identifier'],
                'suffixes': ['_id', '_key', '_idx'],
                'prefixes': ['id_', 'key_']
            },
            ColumnType.CATEGORY: {
                'keywords': ['type', 'category', 'class', 'group', 'segment', 'region', 'status'],
                'suffixes': ['_type', '_cat', '_class', '_group'],
                'prefixes': ['type_', 'cat_', 'class_']
            }
        }
    
    @st.cache_data(show_spinner=False)
    def analyze_columns(_self, df: pd.DataFrame) -> Dict[str, str]:
        """Analizuje kolumny i zwraca ich opisy."""
        descriptions = {}
        
        # Najpierw spróbuj AI jeśli dostępne
        if _self.use_ai and _self.api_key:
            ai_descriptions = _self._get_ai_descriptions(df)
            descriptions.update(ai_descriptions)
        
        # Uzupełnij heurystykami
        for column in df.columns:
            if column not in descriptions:
                col_type = _self._match_pattern(column, df[column])
                descriptions[column] = col_type.value
        
        return descriptions
    
    def _match_pattern(self, column_name: str, series: pd.Series) -> ColumnType:
        """Dopasowuje kolumnę do wzorca na podstawie nazwy i danych."""
        name_lower = column_name.lower()
        
        # Sprawdź wzorce nazw
        for col_type, config in self.patterns.items():
            # Słowa kluczowe
            if any(keyword in name_lower for keyword in config['keywords']):
                return col_type
            # Sufiksy
            if any(name_lower.endswith(suffix) for suffix in config['suffixes']):
                return col_type
            # Prefiksy
            if any(name_lower.startswith(prefix) for prefix in config['prefixes']):
                return col_type
        
        # Analiza danych
        if self._is_likely_datetime(series):
            return ColumnType.DATETIME
        elif self._is_likely_binary(series):
            return ColumnType.BINARY
        elif pd.api.types.is_numeric_dtype(series):
            return ColumnType.NUMERIC
        else:
            return ColumnType.TEXT
    
    def _is_likely_datetime(self, series: pd.Series) -> bool:
        """Sprawdza czy seria przypomina datę."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        if series.dtype == 'object' and len(series.dropna()) > 0:
            try:
                sample = series.dropna().head(min(100, len(series)))
                pd.to_datetime(sample, errors='raise')
                return True
            except (ValueError, TypeError):
                return False
        return False
    
    def _is_likely_binary(self, series: pd.Series) -> bool:
        """Sprawdza czy seria jest binarna."""
        unique_vals = series.dropna().unique()
        return len(unique_vals) <= 2
    
    def _get_ai_descriptions(self, df: pd.DataFrame) -> Dict[str, str]:
        """Pobiera opisy kolumn z OpenAI API."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            columns_info = []
            
            # Przygotuj informacje o kolumnach
            for col in df.columns:
                sample_values = df[col].dropna().head(3).tolist()
                dtype = str(df[col].dtype)
                unique_count = df[col].nunique()
                
                columns_info.append({
                    'name': col,
                    'dtype': dtype,
                    'unique_count': unique_count,
                    'sample_values': sample_values
                })
            
            prompt = f"""
            Przeanalizuj poniższe kolumny danych i zwróć CZYSTY obiekt JSON mapujący nazwa_kolumny -> krótki opis po polsku.
            Uwzględnij typ danych, przykładowe wartości i liczbę unikalnych wartości.
            
            Kolumny: {json.dumps(columns_info, ensure_ascii=False, indent=2)}
            
            Zwróć tylko JSON, bez dodatkowego tekstu.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            content = content.replace('```json', '').replace('```', '').strip()
            
            return json.loads(content)
            
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            return {}

# ============================================================================
# ANALIZATOR JAKOŚCI DANYCH
# ============================================================================

class DataQualityAnalyzer:
    """Zaawansowana analiza jakości danych z cachowaniem."""
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def comprehensive_quality_report(_df: pd.DataFrame) -> pd.DataFrame:
        """Generuje kompleksowy raport jakości danych z cachowaniem."""
        df = _df.copy()  # Kopiuj dla bezpieczeństwa
        n_rows = len(df)
        quality_metrics = []
        
        for col in df.columns:
            series = df[col]
            
            # Podstawowe metryki
            missing_count = series.isna().sum()
            missing_pct = missing_count / n_rows if n_rows > 0 else 0
            unique_count = series.nunique(dropna=True)
            
            # Zaawansowane metryki
            is_constant = unique_count <= 1
            is_mostly_missing = missing_pct > 0.9
            is_unique_identifier = unique_count == (n_rows - missing_count) and unique_count > 1
            
            base_metrics = {
                'column': col,
                'dtype': str(series.dtype),
                'missing_count': missing_count,
                'missing_pct': round(missing_pct, 4),
                'unique_count': unique_count,
                'is_constant': is_constant,
                'is_mostly_missing': is_mostly_missing,
                'is_unique_id': is_unique_identifier,
            }
            
            # Dla numerycznych
            if pd.api.types.is_numeric_dtype(series):
                try:
                    zeros_count = (series == 0).sum()
                    negatives_count = (series < 0).sum()
                    outliers_count = DataQualityAnalyzer._detect_outliers(series).sum()
                    
                    base_metrics.update({
                        'zeros_count': zeros_count,
                        'negatives_count': negatives_count,
                        'outliers_count': outliers_count,
                        'mean': round(series.mean(), 4) if not series.empty else None,
                        'std': round(series.std(), 4) if not series.empty else None,
                    })
                except Exception:
                    # Fallback dla problematycznych danych numerycznych
                    base_metrics.update({
                        'zeros_count': 0,
                        'negatives_count': 0,
                        'outliers_count': 0,
                        'mean': None,
                        'std': None,
                    })
            else:
                # Dla kategorycznych
                try:
                    value_counts = series.value_counts(dropna=False)
                    most_frequent_pct = value_counts.iloc[0] / n_rows if len(value_counts) > 0 else 0
                    
                    base_metrics.update({
                        'cardinality': 'high' if unique_count > 50 else 'medium' if unique_count > 10 else 'low',
                        'most_frequent_pct': round(most_frequent_pct, 4),
                        'top_values': dict(value_counts.head(3)) if len(value_counts) > 0 else {}
                    })
                except Exception:
                    # Fallback dla problematycznych danych kategorycznych
                    base_metrics.update({
                        'cardinality': 'unknown',
                        'most_frequent_pct': 0.0,
                        'top_values': {}
                    })
            
            quality_metrics.append(base_metrics)
        
        quality_df = pd.DataFrame(quality_metrics)
        return quality_df.sort_values(['missing_pct', 'is_constant'], ascending=[False, False])
    
    @staticmethod
    def _detect_outliers(series: pd.Series, method: str = 'iqr') -> pd.Series:
        """Wykrywa wartości odstające."""
        try:
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return (series < lower_bound) | (series > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((series - series.mean()) / series.std())
                return z_scores > 3
            
        except Exception:
            pass
        
        return pd.Series([False] * len(series), index=series.index)

# ============================================================================
# GŁÓWNA KLASA ORCHESTRATORA EDA
# ============================================================================

class SmartDataPreprocessor:
    """Główny orchestrator preprocessingu danych z pipeline pattern."""
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.pipeline = PreprocessingPipeline(self.config)
        self.column_analyzer = SmartColumnAnalyzer()
        self.quality_analyzer = DataQualityAnalyzer()
    
    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, PreprocessingReport]:
        """Główna metoda preprocessingu z pełnym pipeline."""
        return self.pipeline.fit_transform(df, target_column)
    
    def analyze_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analizuje jakość danych."""
        return self.quality_analyzer.comprehensive_quality_report(df)
    
    def get_column_descriptions(self, df: pd.DataFrame, api_key: Optional[str] = None) -> Dict[str, str]:
        """Pobiera opisy kolumn."""
        if api_key:
            self.column_analyzer.api_key = api_key
            self.column_analyzer.use_ai = True
        return self.column_analyzer.analyze_columns(df)
    
    def customize_pipeline(self) -> PreprocessingPipeline:
        """Zwraca pipeline do customizacji."""
        return self.pipeline
    
    def get_preprocessing_summary(self, report: PreprocessingReport) -> Dict[str, Any]:
        """Zwraca czytelne podsumowanie preprocessingu."""
        summary = report.get_summary()
        
        # Dodaj dodatkowe informacje
        transformations_by_status = {}
        for result in report.transformation_results:
            status = result.status.value
            if status not in transformations_by_status:
                transformations_by_status[status] = []
            transformations_by_status[status].append({
                "name": result.message.split(" completed")[0].split(" failed")[0],
                "data_changed": result.data_changed,
                "columns_affected": len(result.affected_columns),
                "columns_created": len(result.created_columns),
                "columns_dropped": len(result.dropped_columns)
            })
        
        summary["transformations_by_status"] = transformations_by_status
        return summary

# ============================================================================
# KOMPATYBILNOŚĆ WSTECZNA
# ============================================================================

def get_column_descriptions(df: pd.DataFrame, api_key: Optional[str] = None) -> Dict[str, str]:
    """Kompatybilność wsteczna - pobiera opisy kolumn."""
    analyzer = SmartColumnAnalyzer(api_key=api_key)
    return analyzer.analyze_columns(df)

def quick_eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Kompatybilność wsteczna - szybkie podsumowanie EDA."""
    return DataQualityAnalyzer.comprehensive_quality_report(df)

def auto_prepare_data(df: pd.DataFrame, target: Optional[str] = None, config: Optional[PreprocessingConfig] = None) -> Tuple[pd.DataFrame, dict]:
    """Kompatybilność wsteczna - automatyczne przygotowanie danych."""
    preprocessor = SmartDataPreprocessor(config or PreprocessingConfig())
    df_processed, report = preprocessor.fit_transform(df, target)
    
    # Konwertuj raport na stary format dla kompatybilności
    summary = report.get_summary()
    
    # Zbierz informacje z transformacji
    dropped_constant = []
    dropped_allnull = []
    parsed_dates = []
    capped_cats = {}
    dropped_duplicates = 0
    
    for result in report.transformation_results:
        if "Constant" in result.message and result.metrics:
            dropped_constant.extend(result.metrics.get('dropped_columns', []))
        elif "Empty" in result.message and result.metrics:
            dropped_allnull.extend(result.metrics.get('dropped_columns', []))
        elif "DateTime" in result.message and result.metrics:
            parsed_dates.extend(result.metrics.get('processed_date_columns', []))
        elif "Cardinality" in result.message and result.metrics:
            capped_cats.update(result.metrics.get('capped_columns', {}))
        elif "Duplicates" in result.message and result.metrics:
            dropped_duplicates = result.metrics.get('duplicates_removed', 0)
    
    legacy_info = {
        'dropped_constant': dropped_constant,
        'dropped_allnull': dropped_allnull,
        'parsed_dates': parsed_dates,
        'capped_cats': capped_cats,
        'dropped_duplicates': dropped_duplicates,
        'processing_time': report.processing_time,
        'warnings': [r.message for r in report.transformation_results if r.status != TransformationStatus.SUCCESS]
    }
    
    return df_processed, legacy_info

# ============================================================================
# ADVANCED EDA COMPONENTS
# ============================================================================

class EDAVisualizer:
    """Komponent do tworzenia wizualizacji EDA."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._cache_key = self._generate_cache_key(df)
    
    def _generate_cache_key(self, df: pd.DataFrame) -> str:
        """Generuje klucz cache na podstawie danych."""
        try:
            # Użyj hash z podstawowych właściwości DataFrame
            key_data = f"{df.shape}_{list(df.columns)}_{df.dtypes.to_dict()}"
            return hashlib.md5(key_data.encode()).hexdigest()[:16]
        except Exception:
            return "default_key"
    
    @st.cache_data(show_spinner=False)
    def get_correlation_matrix(_self, numeric_only: bool = True) -> Optional[pd.DataFrame]:
        """Zwraca macierz korelacji z cachowaniem."""
        try:
            if numeric_only:
                numeric_df = _self.df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) < 2:
                    return None
                return numeric_df.corr()
            else:
                return _self.df.corr(numeric_only=True)
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return None
    
    @st.cache_data(show_spinner=False)
    def get_missing_data_summary(_self) -> pd.DataFrame:
        """Zwraca podsumowanie braków danych."""
        missing_data = _self.df.isnull().sum()
        missing_pct = (missing_data / len(_self.df)) * 100
        
        summary = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_pct
        }).sort_values('Missing_Count', ascending=False)
        
        return summary[summary['Missing_Count'] > 0]
    
    @st.cache_data(show_spinner=False)
    def get_categorical_summary(_self, max_categories: int = 10) -> Dict[str, pd.DataFrame]:
        """Zwraca podsumowanie zmiennych kategorycznych."""
        categorical_cols = _self.df.select_dtypes(include=['object', 'category']).columns
        summaries = {}
        
        for col in categorical_cols:
            value_counts = _self.df[col].value_counts().head(max_categories)
            summaries[col] = value_counts.reset_index()
            summaries[col].columns = [col, 'Count']
        
        return summaries

class EDAReportGenerator:
    """Generator raportów EDA."""
    
    def __init__(self, df: pd.DataFrame, preprocessor: SmartDataPreprocessor):
        self.df = df
        self.preprocessor = preprocessor
        self.visualizer = EDAVisualizer(df)
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generuje komprehensywny raport EDA."""
        report = {
            "basic_info": self._get_basic_info(),
            "data_quality": self._get_data_quality_info(),
            "missing_data": self._get_missing_data_info(),
            "correlations": self._get_correlation_info(),
            "categorical_summary": self._get_categorical_info(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _get_basic_info(self) -> Dict[str, Any]:
        """Podstawowe informacje o danych."""
        return {
            "shape": self.df.shape,
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            "dtypes": self.df.dtypes.value_counts().to_dict(),
            "duplicates": self.df.duplicated().sum()
        }
    
    def _get_data_quality_info(self) -> Dict[str, Any]:
        """Informacje o jakości danych."""
        quality_df = self.preprocessor.analyze_data_quality(self.df)
        
        return {
            "constant_columns": quality_df[quality_df['is_constant']]['column'].tolist(),
            "mostly_missing_columns": quality_df[quality_df['is_mostly_missing']]['column'].tolist(),
            "high_missing_columns": quality_df[quality_df['missing_pct'] > 0.5]['column'].tolist(),
            "potential_ids": quality_df[quality_df['is_unique_id']]['column'].tolist()
        }
    
    def _get_missing_data_info(self) -> Dict[str, Any]:
        """Informacje o brakach danych."""
        missing_summary = self.visualizer.get_missing_data_summary()
        
        return {
            "columns_with_missing": len(missing_summary),
            "total_missing_values": int(missing_summary['Missing_Count'].sum()),
            "worst_columns": missing_summary.head(5).to_dict('records')
        }
    
    def _get_correlation_info(self) -> Dict[str, Any]:
        """Informacje o korelacjach."""
        corr_matrix = self.visualizer.get_correlation_matrix()
        
        if corr_matrix is None:
            return {"available": False}
        
        # Znajdź silne korelacje
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': round(corr_val, 3)
                    })
        
        return {
            "available": True,
            "strong_correlations": strong_corr,
            "max_correlation": float(corr_matrix.abs().unstack().sort_values(ascending=False).iloc[1])
        }
    
    def _get_categorical_info(self) -> Dict[str, Any]:
        """Informacje o zmiennych kategorycznych."""
        cat_summary = self.visualizer.get_categorical_summary()
        
        high_cardinality = []
        for col, summary_df in cat_summary.items():
            total_unique = self.df[col].nunique()
            if total_unique > 50:
                high_cardinality.append({
                    'column': col,
                    'unique_values': total_unique,
                    'data_ratio': total_unique / len(self.df)
                })
        
        return {
            "categorical_columns": len(cat_summary),
            "high_cardinality": high_cardinality,
            "summary": {k: v.to_dict('records') for k, v in cat_summary.items()}
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generuje rekomendacje na podstawie analizy."""
        recommendations = []
        
        # Sprawdź jakość danych
        quality_df = self.preprocessor.analyze_data_quality(self.df)
        
        # Rekomendacje dla braków danych
        high_missing = quality_df[quality_df['missing_pct'] > 0.5]
        if not high_missing.empty:
            recommendations.append(f"Rozważ usunięcie kolumn z wysokim odsetkiem braków: {', '.join(high_missing['column'].tolist())}")
        
        # Rekomendacje dla kolumn stałych
        constant = quality_df[quality_df['is_constant']]
        if not constant.empty:
            recommendations.append(f"Usuń kolumny stałe: {', '.join(constant['column'].tolist())}")
        
        # Rekomendacje dla wysokiej kardynalności
        high_card = quality_df[(quality_df['dtype'] == 'object') & (quality_df['unique_count'] > 100)]
        if not high_card.empty:
            recommendations.append(f"Rozważ grupowanie kategorii dla kolumn wysokiej kardynalności: {', '.join(high_card['column'].tolist())}")
        
        # Rekomendacje dla duplikatów
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            recommendations.append(f"Usuń {duplicates} duplikaty wierszy")
        
        if not recommendations:
            recommendations.append("Dane wyglądają na dobrze przygotowane do analizy")
        
        return recommendations

# ============================================================================
# PRZYKŁAD UŻYCIA I FACTORY
# ============================================================================

class EDAFactory:
    """Factory do tworzenia komponentów EDA."""
    
    @staticmethod
    def create_preprocessor(config: Optional[PreprocessingConfig] = None) -> SmartDataPreprocessor:
        """Tworzy preprocessor z domyślną lub custom konfiguracją."""
        return SmartDataPreprocessor(config)
    
    @staticmethod
    def create_custom_pipeline(steps: List[str], config: Optional[PreprocessingConfig] = None) -> PreprocessingPipeline:
        """Tworzy custom pipeline z wybranymi krokami."""
        pipeline = PreprocessingPipeline(config or PreprocessingConfig())
        
        # Mapowanie nazw kroków na transformatory
        step_mapping = {
            'remove_empty': RemoveEmptyColumnsTransformer(),
            'remove_constant': RemoveConstantColumnsTransformer(),
            'datetime': DateTimeTransformer(),
            'duplicates': DuplicateRemovalTransformer(),
            'high_cardinality': HighCardinalityTransformer(),
            'outliers': OutlierTransformer(),
            'missing_values': MissingValueTransformer()
        }
        
        # Wyczyść domyślny pipeline i dodaj wybrane kroki
        pipeline.transformers = []
        for step in steps:
            if step in step_mapping:
                pipeline.add_transformer(step_mapping[step])
        
        return pipeline
    
    @staticmethod
    def create_report_generator(df: pd.DataFrame, config: Optional[PreprocessingConfig] = None) -> EDAReportGenerator:
        """Tworzy generator raportów EDA."""
        preprocessor = EDAFactory.create_preprocessor(config)
        return EDAReportGenerator(df, preprocessor)

# Przykład użycia nowych funkcji
if __name__ == "__main__":
    # Przykład użycia nowego pipeline
    config = PreprocessingConfig(
        remove_duplicates=True,
        enable_outlier_treatment=True,
        outlier_method='winsorize',
        create_date_features=True,
        max_categories=25
    )
    
    # Stwórz preprocessor
    preprocessor = SmartDataPreprocessor(config)
    
    # Lub stwórz custom pipeline
    custom_pipeline = EDAFactory.create_custom_pipeline(
        ['remove_constant', 'datetime', 'missing_values'],
        config
    )
    
    # Przykład użycia (gdyby był DataFrame)
    # df_processed, report = preprocessor.fit_transform(df, target_column='price')
    # print(f"Preprocessing completed in {report.processing_time:.2f}s")
    # print("Summary:", preprocessor.get_preprocessing_summary(report))