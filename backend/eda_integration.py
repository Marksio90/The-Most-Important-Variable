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
        lines = [
            f"📊 Preprocessing Summary",
            f"━━━━━━━━━━━━━━━━━━━━━━",
            f"Original shape: {self.original_shape}",
            f"Final shape: {self.final_shape}",
            f"Processing time: {self.processing_time:.2f}s",
            "",
        ]
        
        if self.dropped_columns:
            lines.append("🗑️ Dropped columns:")
            for reason, cols in self.dropped_columns.items():
                if cols:
                    lines.append(f"  • {reason}: {len(cols)} columns")
        
        if self.created_columns:
            lines.append(f"✨ Created {len(self.created_columns)} new columns")
        
        if self.warnings:
            lines.append(f"⚠️  {len(self.warnings)} warnings generated")
            
        return "\n".join(lines)

class AdvancedColumnAnalyzer:
    """Zaawansowany analizator kolumn z konfigurowalnymi heurystykami."""
    
    def __init__(self, api_key: Optional[str] = None, use_ai: bool = True):
        self.api_key = api_key
        self.use_ai = use_ai
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
                # Testuj konwersję na próbce danych
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
            # Usuń potencjalne markdown formatting
            content = content.replace('```json', '').replace('```', '').strip()
            
            return json.loads(content)
            
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            return {}

class DataQualityAnalyzer:
    """Zaawansowana analiza jakości danych."""
    
    @staticmethod
    def comprehensive_quality_report(df: pd.DataFrame) -> pd.DataFrame:
        """Generuje kompleksowy raport jakości danych."""
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
            
            # Dla numerycznych
            if pd.api.types.is_numeric_dtype(series):
                zeros_count = (series == 0).sum()
                negatives_count = (series < 0).sum()
                outliers_count = DataQualityAnalyzer._detect_outliers(series).sum()
                
                quality_metrics.append({
                    'column': col,
                    'dtype': str(series.dtype),
                    'missing_count': missing_count,
                    'missing_pct': round(missing_pct, 4),
                    'unique_count': unique_count,
                    'is_constant': is_constant,
                    'is_mostly_missing': is_mostly_missing,
                    'is_unique_id': is_unique_identifier,
                    'zeros_count': zeros_count,
                    'negatives_count': negatives_count,
                    'outliers_count': outliers_count,
                    'mean': round(series.mean(), 4) if not series.empty else None,
                    'std': round(series.std(), 4) if not series.empty else None,
                })
            else:
                # Dla kategorycznych
                value_counts = series.value_counts(dropna=False)
                most_frequent_pct = value_counts.iloc[0] / n_rows if len(value_counts) > 0 else 0
                
                quality_metrics.append({
                    'column': col,
                    'dtype': str(series.dtype),
                    'missing_count': missing_count,
                    'missing_pct': round(missing_pct, 4),
                    'unique_count': unique_count,
                    'is_constant': is_constant,
                    'is_mostly_missing': is_mostly_missing,
                    'is_unique_id': is_unique_identifier,
                    'cardinality': 'high' if unique_count > 50 else 'medium' if unique_count > 10 else 'low',
                    'most_frequent_pct': round(most_frequent_pct, 4),
                    'top_values': dict(value_counts.head(3)) if len(value_counts) > 0 else {}
                })
        
        quality_df = pd.DataFrame(quality_metrics)
        return quality_df.sort_values(['missing_pct', 'is_constant'], ascending=[False, False])
    
    @staticmethod
    def _detect_outliers(series: pd.Series, method: str = 'iqr') -> pd.Series:
        """Wykrywa wartości odstające."""
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
        
        return pd.Series([False] * len(series), index=series.index)

class SmartDataPreprocessor:
    """Inteligentny preprocessor danych z konfigurowalnymi opcjami."""
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.report = PreprocessingReport((0, 0), (0, 0))
    
    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, PreprocessingReport]:
        """Główna metoda preprocessingu."""
        start_time = datetime.now()
        
        # Inicjalizacja raportu
        self.report = PreprocessingReport(
            original_shape=df.shape,
            final_shape=df.shape
        )
        
        df_processed = df.copy()
        
        try:
            # 1. Usuń całkowicie puste kolumny
            df_processed = self._handle_empty_columns(df_processed)
            
            # 2. Usuń kolumny stałe lub prawie stałe
            df_processed = self._handle_constant_columns(df_processed)
            
            # 3. Parsuj daty i twórz cechy czasowe
            if self.config.parse_dates:
                df_processed = self._handle_datetime_columns(df_processed)
            
            # 4. Usuń duplikaty
            if self.config.remove_duplicates:
                df_processed = self._remove_duplicates(df_processed)
            
            # 5. Obsłuż kategorie wysokiej kardinalności
            df_processed = self._handle_high_cardinality(df_processed)
            
            # 6. Obsłuż wartości odstające
            if self.config.enable_outlier_treatment:
                df_processed = self._handle_outliers(df_processed, target_column)
            
            # 7. Imputuj brakujące wartości
            if self.config.handle_missing:
                df_processed = self._handle_missing_values(df_processed)
            
            # Aktualizuj finalne informacje
            self.report.final_shape = df_processed.shape
            
        except Exception as e:
            self.report.warnings.append(f"Error during preprocessing: {str(e)}")
            logger.error(f"Preprocessing error: {e}")
        
        finally:
            # Czas przetwarzania
            end_time = datetime.now()
            self.report.processing_time = (end_time - start_time).total_seconds()
        
        return df_processed, self.report
    
    def _handle_empty_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Usuwa kolumny z wysokim odsetkiem braków."""
        empty_cols = []
        
        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df)
            if missing_pct >= self.config.drop_missing_threshold:
                empty_cols.append(col)
        
        if empty_cols:
            df = df.drop(columns=empty_cols)
            self.report.dropped_columns['high_missing'] = empty_cols
            logger.info(f"Dropped {len(empty_cols)} columns with >{self.config.drop_missing_threshold*100}% missing values")
        
        return df
    
    def _handle_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Usuwa kolumny stałe lub prawie stałe."""
        constant_cols = []
        
        for col in df.columns:
            try:
                # Sprawdź czy kolumna ma tylko jedną unikalną wartość (pomijając NaN)
                unique_ratio = df[col].nunique(dropna=True) / df[col].notna().sum()
                if unique_ratio < (1 - self.config.drop_constant_threshold) or df[col].nunique(dropna=True) <= 1:
                    constant_cols.append(col)
            except (ZeroDivisionError, ValueError):
                constant_cols.append(col)
        
        if constant_cols:
            df = df.drop(columns=constant_cols)
            self.report.dropped_columns['constant'] = constant_cols
            logger.info(f"Dropped {len(constant_cols)} constant/near-constant columns")
        
        return df
    
    def _handle_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parsuje daty i tworzy cechy czasowe."""
        date_cols = []
        
        for col in df.columns:
            if self._is_datetime_column(df[col]):
                try:
                    dt_series = pd.to_datetime(df[col], errors='coerce')
                    
                    if self.config.create_date_features:
                        # Twórz cechy czasowe
                        base_name = col.replace('_date', '').replace('_time', '').replace('_at', '')
                        
                        if 'year' in self.config.date_features:
                            df[f"{base_name}__year"] = dt_series.dt.year
                        if 'month' in self.config.date_features:
                            df[f"{base_name}__month"] = dt_series.dt.month
                        if 'day' in self.config.date_features:
                            df[f"{base_name}__day"] = dt_series.dt.day
                        if 'dayofweek' in self.config.date_features:
                            df[f"{base_name}__dayofweek"] = dt_series.dt.dayofweek
                        if 'quarter' in self.config.date_features:
                            df[f"{base_name}__quarter"] = dt_series.dt.quarter
                        if 'is_weekend' in self.config.date_features:
                            df[f"{base_name}__is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)
                        
                        # Zachowaj oryginalną kolumnę jako datetime
                        df[col] = dt_series
                        
                        created_features = [f"{base_name}__{feat}" for feat in self.config.date_features]
                        self.report.created_columns.extend(created_features)
                    
                    date_cols.append(col)
                    
                except Exception as e:
                    self.report.warnings.append(f"Failed to parse datetime column {col}: {e}")
        
        if date_cols:
            self.report.transformations['parsed_dates'] = date_cols
            logger.info(f"Processed {len(date_cols)} datetime columns")
        
        return df
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        """Sprawdza czy kolumna zawiera daty."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        if series.dtype == 'object':
            try:
                # Testuj na próbce
                sample = series.dropna().head(min(50, len(series)))
                if len(sample) > 0:
                    pd.to_datetime(sample, errors='raise')
                    return True
            except (ValueError, TypeError):
                pass
        
        return False
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Usuwa duplikaty."""
        initial_rows = len(df)
        df_dedupe = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df_dedupe)
        
        if duplicates_removed > 0:
            self.report.transformations['duplicates_removed'] = duplicates_removed
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        return df_dedupe
    
    def _handle_high_cardinality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Obsługuje kategorie wysokiej kardinalności."""
        capped_cols = {}
        
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                unique_count = df[col].nunique()
                
                if unique_count > self.config.category_threshold:
                    # Zostaw top N kategorii, resztę zamień na 'OTHER'
                    value_counts = df[col].value_counts()
                    top_categories = set(value_counts.head(self.config.max_categories).index)
                    
                    df[col] = df[col].apply(lambda x: x if x in top_categories else 'OTHER')
                    capped_cols[col] = unique_count
        
        if capped_cols:
            self.report.transformations['capped_categories'] = capped_cols
            logger.info(f"Capped {len(capped_cols)} high-cardinality categorical columns")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
        """Obsługuje wartości odstające."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Nie stosuj do kolumny docelowej
        if target_column and target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        treated_cols = []
        
        for col in numeric_cols:
            try:
                if self.config.outlier_method == 'winsorize':
                    lower_bound = df[col].quantile(self.config.lower_quantile)
                    upper_bound = df[col].quantile(self.config.upper_quantile)
                    df[col] = df[col].clip(lower_bound, upper_bound)
                    treated_cols.append(col)
                    
                elif self.config.outlier_method == 'clip':
                    # Bardziej agresywne clipping
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower_bound, upper_bound)
                    treated_cols.append(col)
                    
            except Exception as e:
                self.report.warnings.append(f"Failed to treat outliers in {col}: {e}")
        
        if treated_cols:
            self.report.transformations['outlier_treatment'] = {
                'method': self.config.outlier_method,
                'columns': treated_cols
            }
            logger.info(f"Applied {self.config.outlier_method} to {len(treated_cols)} numeric columns")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputuje brakujące wartości."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        imputed_numeric = []
        imputed_categorical = []
        
        # Numeryczne
        for col in numeric_cols:
            if df[col].isna().any():
                if self.config.numeric_imputation == 'mean':
                    fill_value = df[col].mean()
                elif self.config.numeric_imputation == 'median':
                    fill_value = df[col].median()
                elif self.config.numeric_imputation == 'mode':
                    fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                else:
                    fill_value = 0
                
                df[col] = df[col].fillna(fill_value)
                imputed_numeric.append(col)
        
        # Kategoryczne
        for col in categorical_cols:
            if df[col].isna().any():
                if self.config.categorical_imputation == 'mode':
                    try:
                        fill_value = df[col].mode().iloc[0]
                    except (IndexError, AttributeError):
                        fill_value = self.config.constant_value
                else:
                    fill_value = self.config.constant_value
                
                df[col] = df[col].fillna(fill_value)
                imputed_categorical.append(col)
        
        if imputed_numeric or imputed_categorical:
            self.report.transformations['imputation'] = {
                'numeric': {
                    'method': self.config.numeric_imputation,
                    'columns': imputed_numeric
                },
                'categorical': {
                    'method': self.config.categorical_imputation,
                    'columns': imputed_categorical
                }
            }
            logger.info(f"Imputed missing values in {len(imputed_numeric + imputed_categorical)} columns")
        
        return df


# Funkcje pomocnicze dla kompatybilności wstecznej
def get_column_descriptions(df: pd.DataFrame, api_key: Optional[str] = None) -> Dict[str, str]:
    """Kompatybilność wsteczna - pobiera opisy kolumn."""
    analyzer = AdvancedColumnAnalyzer(api_key=api_key)
    return analyzer.analyze_columns(df)

def quick_eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Kompatybilność wsteczna - szybkie podsumowanie EDA."""
    return DataQualityAnalyzer.comprehensive_quality_report(df)

def auto_prepare_data(df: pd.DataFrame, target: Optional[str] = None, config: Optional[PreprocessingConfig] = None) -> Tuple[pd.DataFrame, dict]:
    """Kompatybilność wsteczna - automatyczne przygotowanie danych."""
    preprocessor = SmartDataPreprocessor(config or PreprocessingConfig())
    df_processed, report = preprocessor.fit_transform(df, target)
    
    # Konwertuj raport na stary format dla kompatybilności
    legacy_info = {
        'dropped_constant': report.dropped_columns.get('constant', []),
        'dropped_allnull': report.dropped_columns.get('high_missing', []),
        'parsed_dates': report.transformations.get('parsed_dates', []),
        'capped_cats': report.transformations.get('capped_categories', {}),
        'dropped_duplicates': report.transformations.get('duplicates_removed', 0),
        'processing_time': report.processing_time,
        'warnings': report.warnings
    }
    
    return df_processed, legacy_info

# Przykład użycia nowych funkcji
if __name__ == "__main__":
    # Konfiguracja preprocessingu
    config = PreprocessingConfig(
        max_categories=25,
        enable_outlier_treatment=True,
        outlier_method='winsorize',
        create_date_features=True,
        date_features=['year', 'month', 'dayofweek', 'is_weekend']
    )
    
    # Przykład użycia
    # df_processed, report = SmartDataPreprocessor(config).fit_transform(df, target_column='price')
    # print(report.summary())