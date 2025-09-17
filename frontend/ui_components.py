from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import io
import csv
import re
import logging
from contextlib import contextmanager

import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime

# Konfiguracja logowania
logger = logging.getLogger(__name__)

# ==============================
# KONFIGURACJA I TYPY
# ==============================

class DataSource(Enum):
    """Źródła danych."""
    CSV = "csv"
    JSON = "json" 
    EXCEL = "excel"
    PARQUET = "parquet"
    DEMO = "demo"

class FileStatus(Enum):
    """Status przetwarzania pliku."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    TOO_LARGE = "too_large"
    INVALID_FORMAT = "invalid_format"
    ENCODING_ERROR = "encoding_error"
    EMPTY_FILE = "empty_file"

@dataclass
class FileMetadata:
    """Metadane wgranego pliku."""
    filename: str
    size_mb: float
    format: str
    rows: int = 0
    columns: int = 0
    status: FileStatus = FileStatus.SUCCESS
    warnings: List[str] = field(default_factory=list)
    encoding: Optional[str] = None
    separator: Optional[str] = None

@dataclass
class DataConfig:
    """Konfiguracja wczytywania danych."""
    max_file_size_mb: int = 200
    max_preview_rows: int = 20
    supported_formats: List[str] = field(default_factory=lambda: ["csv", "json", "xlsx", "parquet"])
    auto_detect_encoding: bool = True
    auto_detect_separator: bool = True

@dataclass 
class UIConfig:
    """Konfiguracja interfejsu użytkownika."""
    app_title: str = "TMIV — The Most Important Variables"
    app_subtitle: str = "AutoML • EDA • Historia eksperymentów"
    show_advanced_options: bool = True
    enable_llm: bool = True
    enable_demo_data: bool = True
    demo_data_paths: List[Path] = field(default_factory=lambda: [
        Path("data/avocado.csv"),
        Path("datasets/avocado.csv"), 
        Path("avocado.csv")
    ])

# ==============================
# PROTOKOŁY (INTERFACES)
# ==============================

class FileFormatDetector(Protocol):
    """Interface dla detektorów formatów plików."""
    def detect_encoding(self, data: bytes) -> str: ...
    def detect_delimiter(self, text: str) -> str: ...
    def detect_format(self, filename: str) -> str: ...

class FileValidator(Protocol):
    """Interface dla walidatorów plików."""
    def validate_size(self, size: int) -> bool: ...
    def validate_format(self, filename: str) -> bool: ...
    def validate_content(self, df: pd.DataFrame) -> List[str]: ...

class DataLoader(Protocol):
    """Interface dla loaderów danych."""
    def load_data(self, file_data: bytes, **kwargs) -> pd.DataFrame: ...

# ==============================
# IMPLEMENTACJE DETEKTORÓW
# ==============================

class SmartFileFormatDetector:
    """Inteligentny detektor formatów i parametrów plików."""
    
    def detect_encoding(self, sample: bytes, max_size: int = 64 * 1024) -> str:
        """Wykrywa encoding z fallbackiem."""
        sample = sample[:max_size]
        encodings = ["utf-8", "cp1250", "latin1", "utf-16", "ascii"]
        
        for encoding in encodings:
            try:
                decoded = sample.decode(encoding)
                # Sprawdź czy nie ma dziwnych znaków
                if not any(ord(c) > 127 and ord(c) < 160 for c in decoded[:1000]):
                    return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        return "utf-8"  # Fallback
    
    def detect_delimiter(self, sample: str, max_lines: int = 50) -> str:
        """Wykrywa separator CSV z analizą spójności."""
        lines = sample.split('\n')[:max_lines]
        
        try:
            # Użyj CSV Sniffer
            dialect = csv.Sniffer().sniff(sample[:8192], delimiters=",;\t|")
            return dialect.delimiter
        except Exception:
            pass
        
        # Fallback: analiza spójności
        delimiters = [",", ";", "\t", "|"]
        scores = {}
        
        for delim in delimiters:
            counts = [line.count(delim) for line in lines if line.strip()]
            if not counts:
                continue
                
            most_common_count = max(set(counts), key=counts.count) if counts else 0
            consistency = counts.count(most_common_count) / len(counts) if counts else 0
            scores[delim] = (most_common_count, consistency)
        
        if not scores:
            return ","
        
        # Wybierz delimiter z najlepszą spójnością
        best_delim = max(scores.items(), key=lambda x: (x[1][1], min(x[1][0], 20)))
        return best_delim[0]
    
    def detect_format(self, filename: str) -> str:
        """Wykrywa format pliku na podstawie rozszerzenia."""
        suffix = Path(filename).suffix.lower().lstrip('.')
        format_mapping = {
            'csv': 'csv',
            'txt': 'csv', 
            'json': 'json',
            'jsonl': 'json',
            'xlsx': 'excel',
            'xls': 'excel',
            'parquet': 'parquet'
        }
        return format_mapping.get(suffix, 'csv')

class BasicFileValidator:
    """Podstawowy walidator plików."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def validate_size(self, size_bytes: int) -> bool:
        """Waliduje rozmiar pliku."""
        size_mb = size_bytes / (1024 * 1024)
        return size_mb <= self.config.max_file_size_mb
    
    def validate_format(self, filename: str) -> bool:
        """Waliduje format pliku."""
        format_name = SmartFileFormatDetector().detect_format(filename)
        return format_name in self.config.supported_formats
    
    def validate_content(self, df: pd.DataFrame) -> List[str]:
        """Waliduje zawartość DataFrame."""
        warnings = []
        
        if df.empty:
            warnings.append("Plik jest pusty")
        
        if len(df.columns) == 0:
            warnings.append("Brak kolumn w danych")
        
        if len(df.columns) > 1000:
            warnings.append("Bardzo dużo kolumn (>1000) - może wpłynąć na wydajność")
        
        return warnings

# ==============================
# LOADERY DANYCH
# ==============================

class CSVDataLoader:
    """Loader dla plików CSV."""
    
    def __init__(self, detector: FileFormatDetector):
        self.detector = detector
    
    def load_data(self, file_data: bytes, **kwargs) -> pd.DataFrame:
        """Ładuje plik CSV z auto-detekcją lub override parametrów."""
        
        # Auto-detekcja lub użycie podanych parametrów
        encoding = kwargs.get('encoding') or self.detector.detect_encoding(file_data)
        
        try:
            decoded = file_data.decode(encoding, errors='ignore')
        except Exception:
            decoded = file_data.decode('utf-8', errors='ignore')
            encoding = 'utf-8'
        
        delimiter = kwargs.get('delimiter') or self.detector.detect_delimiter(decoded)
        decimal = kwargs.get('decimal', '.')
        header = kwargs.get('header', 0)
        
        # Wczytaj DataFrame
        df = pd.read_csv(
            io.BytesIO(file_data),
            encoding=encoding,
            sep=delimiter,
            decimal=decimal,
            header=header,
            low_memory=False,
            na_values=['', 'NULL', 'null', 'N/A', 'n/a', '#N/A'],
            keep_default_na=True
        )
        
        # Jeśli brak header, nadaj nazwy kolumn
        if header is None:
            df.columns = [f"col_{i}" for i in range(len(df.columns))]
        
        return df

class JSONDataLoader:
    """Loader dla plików JSON."""
    
    def load_data(self, file_data: bytes, **kwargs) -> pd.DataFrame:
        """Ładuje pliki JSON z obsługą różnych formatów."""
        
        decoded = file_data.decode('utf-8', errors='ignore')
        
        # Próbuj różne formaty JSON
        formats_to_try = [
            ('records', lambda: pd.read_json(io.StringIO(decoded), orient='records')),
            ('lines', lambda: pd.read_json(io.StringIO(decoded), lines=True)),
            ('auto', lambda: pd.read_json(io.StringIO(decoded))),
        ]
        
        for format_name, loader in formats_to_try:
            try:
                df = loader()
                if df is not None and not df.empty:
                    return df
            except Exception:
                continue
        
        raise ValueError("Nie udało się wczytać danych z pliku JSON")

class ExcelDataLoader:
    """Loader dla plików Excel."""
    
    def load_data(self, file_data: bytes, **kwargs) -> pd.DataFrame:
        """Ładuje pliki Excel."""
        sheet_name = kwargs.get('sheet_name', 0)
        header = kwargs.get('header', 0)
        
        df = pd.read_excel(
            io.BytesIO(file_data),
            sheet_name=sheet_name,
            header=header
        )
        
        return df

# ==============================
# FACTORY DLA LOADERÓW
# ==============================

class DataLoaderFactory:
    """Factory do tworzenia odpowiednich loaderów danych."""
    
    def __init__(self):
        self.detector = SmartFileFormatDetector()
        self.loaders = {
            'csv': CSVDataLoader(self.detector),
            'json': JSONDataLoader(),
            'excel': ExcelDataLoader()
        }
    
    def get_loader(self, format_name: str) -> DataLoader:
        """Zwraca odpowiedni loader dla formatu."""
        return self.loaders.get(format_name, self.loaders['csv'])

# ==============================
# GŁÓWNE KOMPONENTY UI
# ==============================

class FileUploadWidget:
    """Uproszczony widget do uploadu plików."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.detector = SmartFileFormatDetector()
        self.validator = BasicFileValidator(config)
        self.loader_factory = DataLoaderFactory()
    
    def render(self, key: str = "file_uploader") -> Tuple[Optional[pd.DataFrame], Optional[FileMetadata]]:
        """Renderuje widget upload z podstawowymi opcjami."""
        
        uploaded_file = st.file_uploader(
            "Wgraj plik danych",
            type=self.config.supported_formats,
            key=key,
            help=f"Maksymalny rozmiar: {self.config.max_file_size_mb}MB"
        )
        
        if not uploaded_file:
            return None, None
        
        # Walidacja podstawowa
        if not self.validator.validate_size(uploaded_file.size):
            st.error(f"Plik za duży! ({uploaded_file.size / 1024 / 1024:.1f}MB > {self.config.max_file_size_mb}MB)")
            return None, None
        
        if not self.validator.validate_format(uploaded_file.name):
            st.error(f"Nieobsługiwany format pliku")
            return None, None
        
        # Wczytywanie danych
        try:
            format_name = self.detector.detect_format(uploaded_file.name)
            loader = self.loader_factory.get_loader(format_name)
            
            file_data = uploaded_file.read()
            uploaded_file.seek(0)
            
            df = loader.load_data(file_data)
            
            # Walidacja zawartości
            content_warnings = self.validator.validate_content(df)
            
            # Metadane
            metadata = FileMetadata(
                filename=uploaded_file.name,
                size_mb=uploaded_file.size / 1024 / 1024,
                format=format_name,
                rows=len(df),
                columns=len(df.columns),
                status=FileStatus.WARNING if content_warnings else FileStatus.SUCCESS,
                warnings=content_warnings
            )
            
            # Wyświetl informacje o pliku
            self._display_file_info(df, metadata)
            
            return df, metadata
            
        except Exception as e:
            st.error(f"Błąd wczytywania pliku: {str(e)}")
            return None, None
    
    def _display_file_info(self, df: pd.DataFrame, metadata: FileMetadata):
        """Wyświetla informacje o wczytanym pliku."""
        
        # Status badge
        if metadata.status == FileStatus.SUCCESS:
            st.success(f"✅ **{metadata.filename}** wczytany pomyślnie")
        elif metadata.status == FileStatus.WARNING:
            st.warning(f"⚠️ **{metadata.filename}** wczytany z ostrzeżeniami")
        
        # Podstawowe metryki
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Wiersze", f"{metadata.rows:,}")
        with col2:
            st.metric("Kolumny", metadata.columns)
        with col3:
            st.metric("Rozmiar", f"{metadata.size_mb:.1f} MB")
        with col4:
            st.metric("Format", metadata.format.upper())
        
        # Ostrzeżenia
        if metadata.warnings:
            with st.expander("⚠️ Ostrzeżenia", expanded=False):
                for warning in metadata.warnings:
                    st.warning(warning)
        
        # Podgląd danych
        preview_rows = min(self.config.max_preview_rows, len(df))
        with st.expander("📊 Podgląd danych", expanded=True):
            st.dataframe(df.head(preview_rows), use_container_width=True)

class DemoDataWidget:
    """Widget do wyboru danych demonstracyjnych."""
    
    def __init__(self, demo_paths: List[Path]):
        self.demo_paths = demo_paths
    
    def render(self) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Renderuje opcję demo data."""
        
        st.info("📊 Używasz danych demonstracyjnych 'avocado'")
        
        try:
            for path in self.demo_paths:
                if path.exists():
                    df = pd.read_csv(path)
                    
                    # Podstawowe informacje
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Wiersze", f"{len(df):,}")
                    with col2:
                        st.metric("Kolumny", len(df.columns))
                    with col3:
                        st.metric("Źródło", "Demo")
                    
                    # Podgląd
                    with st.expander("📊 Podgląd danych", expanded=True):
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    return df, "avocado (demo)"
            
            raise FileNotFoundError("Nie znaleziono pliku demonstracyjnego")
            
        except Exception as e:
            st.error(f"Błąd ładowania danych demo: {str(e)}")
            return None, None

class TargetSelectorWidget:
    """Uproszczony widget do wyboru kolumny celu."""
    
    def __init__(self):
        self.target_keywords = [
            "target", "y", "label", "class", "price", "amount", "value", 
            "revenue", "sales", "outcome", "result", "prediction", "AveragePrice"
        ]
    
    def render(self, columns: List[str]) -> Optional[str]:
        """Renderuje selector kolumny celu z heurystyką."""
        
        if not columns:
            st.warning("Nie można ustalić kolumn — wczytaj dane.")
            return None
        
        # Heurystyka znajdowania targetu
        auto_target = self._detect_target(columns)
        
        # Selector z podpowiedzią
        default_index = columns.index(auto_target) if auto_target in columns else 0
        
        selected_target = st.selectbox(
            "🎯 Wybierz kolumnę celu",
            options=columns,
            index=default_index,
            help="Kolumna, którą chcesz przewidywać"
        )
        
        if selected_target != auto_target and auto_target:
            st.info(f"💡 Podpowiedź: wykryto potencjalny target '{auto_target}'")
        
        return selected_target
    
    def _detect_target(self, columns: List[str]) -> Optional[str]:
        """Heurystyka znajdowania kolumny celu."""
        column_lower = {c.lower(): c for c in columns}
        
        for keyword in self.target_keywords:
            if keyword.lower() in column_lower:
                return column_lower[keyword.lower()]
        
        return None

class DataSourceSelectorWidget:
    """Widget do wyboru źródła danych."""
    
    def __init__(self, config: DataConfig, ui_config: UIConfig):
        self.config = config
        self.ui_config = ui_config
        self.file_widget = FileUploadWidget(config)
        self.demo_widget = DemoDataWidget(ui_config.demo_data_paths) if ui_config.enable_demo_data else None
    
    def render(self) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Renderuje selector źródła danych."""
        
        # Opcje źródeł
        options = ["Wgraj własny plik"]
        if self.ui_config.enable_demo_data:
            options.append("Użyj danych demo 'avocado'")
        
        source_type = st.radio(
            "📦 Źródło danych", 
            options, 
            horizontal=True
        )
        
        if "demo" in source_type.lower() and self.demo_widget:
            return self.demo_widget.render()
        else:
            df, metadata = self.file_widget.render()
            return df, metadata.filename if metadata else None

# ==============================
# GŁÓWNA KLASA APLIKACJI UI
# ==============================

class TMIVApp:
    """Główna klasa aplikacji UI - tylko orchestracja komponentów."""
    
    def __init__(self, data_config: DataConfig = None, ui_config: UIConfig = None):
        self.data_config = data_config or DataConfig()
        self.ui_config = ui_config or UIConfig()
        
        # Komponenty UI
        self.data_selector = DataSourceSelectorWidget(self.data_config, self.ui_config)
        self.target_selector = TargetSelectorWidget()
    
    def render_header(self) -> None:
        """Renderuje nagłówek aplikacji."""
        st.title(self.ui_config.app_title)
        st.caption(self.ui_config.app_subtitle)
    
    def render_data_selection(self) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
        """Renderuje sekcję wyboru danych i targetu."""
        
        # Nagłówek
        self.render_header()
        
        # Wybór danych
        df, dataset_name = self.data_selector.render()
        
        if df is None or df.empty:
            return None, None, None
        
        # Zapisz kolumny do session state dla kompatybilności
        st.session_state["df_columns"] = list(df.columns)
        
        # Wybór targetu
        target = self.target_selector.render(list(df.columns))
        
        return df, dataset_name, target
    
    def render_configuration(self, available_engines: List[str]) -> Dict[str, Any]:
        """Renderuje uproszczony panel konfiguracji."""
        
        with st.sidebar:
            st.header("⚙️ Ustawienia")
            
            config = {}
            
            # Silnik ML
            config['ml_engine'] = st.selectbox(
                "Silnik ML",
                available_engines,
                index=0 if available_engines else 0,
                key="ml_engine"
            )
            
            # Podstawowe opcje
            config['cross_validation'] = st.checkbox("Cross-validation", value=True)
            config['feature_selection'] = st.checkbox("Selekcja cech", value=True)
            
            # Zaawansowane opcje
            with st.expander("🔧 Zaawansowane"):
                config['hyperparameter_tuning'] = st.checkbox("Tuning hiperparametrów", value=False)
                config['outlier_treatment'] = st.checkbox("Obsługa outlierów", value=True)
        
        return config

class DataQualitySummaryWidget:
    """Widget do wyświetlania podsumowania jakości danych."""
    
    @staticmethod
    def render(df: pd.DataFrame) -> None:
        """Pokazuje podsumowanie jakości danych."""
        
        if df.empty:
            return
        
        st.subheader("📊 Jakość danych")
        
        # Podstawowe statystyki
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Braki danych", f"{missing_pct:.1f}%")
        
        with col2:
            duplicates = df.duplicated().sum()
            st.metric("Duplikaty", duplicates)
        
        with col3:
            numeric_cols = len(df.select_dtypes(include=['number']).columns)
            st.metric("Kolumny numeryczne", numeric_cols)
        
        with col4:
            categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
            st.metric("Kolumny kategoryczne", categorical_cols)
        
        # Szczegółowa analiza braków
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if not missing_data.empty:
            with st.expander(f"🔍 Szczegóły braków danych ({len(missing_data)} kolumn)", expanded=False):
                missing_df = pd.DataFrame({
                    'Kolumna': missing_data.index,
                    'Braki': missing_data.values,
                    'Procent': (missing_data.values / len(df) * 100).round(1)
                })
                st.dataframe(missing_df, use_container_width=True)

# ==============================
# FUNKCJE KOMPATYBILNOŚCI WSTECZNEJ
# ==============================

def header() -> None:
    """Kompatybilność wsteczna - nagłówek aplikacji."""
    config = UIConfig()
    st.title(config.app_title)
    st.caption(config.app_subtitle)

def dataset_selector(sample_data_path: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, str]:
    """Kompatybilność wsteczna - selector danych."""
    
    ui_config = UIConfig()
    if sample_data_path:
        ui_config.demo_data_paths = [Path(sample_data_path) / "avocado.csv"]
    
    data_config = DataConfig()
    selector = DataSourceSelectorWidget(data_config, ui_config)
    
    df, name = selector.render()
    return df or pd.DataFrame(), name or ""

def show_detected_target(auto_target: Optional[str], columns: Optional[List[str]] = None) -> Optional[str]:
    """Kompatybilność wsteczna - selector targetu."""
    
    cols = columns or st.session_state.get("df_columns", [])
    if not cols:
        return None
    
    target_selector = TargetSelectorWidget()
    return target_selector.render(cols)

def sidebar_config(
    available_ml: List[str],
    default_engine: str = "auto",
    show_eda_engine: bool = True,
) -> Dict[str, Any]:
    """Kompatybilność wsteczna - panel konfiguracji."""
    
    with st.sidebar:
        st.header("⚙️ Ustawienia")
        
        config = {}
        
        if show_eda_engine:
            config["eda_engine"] = st.selectbox(
                "Silnik EDA",
                ["Szybkie podsumowanie", "Rozkłady", "Korelacje", "Analiza zaawansowana"],
                key="eda_engine"
            )
        
        # Silnik ML
        default_idx = available_ml.index(default_engine) if default_engine in available_ml else 0
        config['ml_engine'] = st.selectbox(
            "Silnik ML",
            available_ml,
            index=default_idx,
            key="ml_engine"
        )
        
        # LLM (dla kompatybilności)
        config['llm_enabled'] = False
        config['llm_api_key'] = ""
        config['llm_prompt'] = ""
    
    return config

def upload_widget(
    config: Optional[DataConfig] = None,
    key: str = "main_uploader",
    label: str = "Wgraj plik danych"
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Prosty wrapper dla kompatybilności wstecznej."""
    
    upload_widget = FileUploadWidget(config or DataConfig())
    df, metadata = upload_widget.render(key)
    
    filename = metadata.filename if metadata else None
    return df, filename

def advanced_upload_widget(
    max_size_mb: int = 200,
    formats: Optional[List[str]] = None,
    enable_preview: bool = True,
    key: str = "advanced_uploader"
) -> Tuple[Optional[pd.DataFrame], Optional[FileMetadata]]:
    """Zaawansowany wrapper z customizacją."""
    
    config = DataConfig(
        max_file_size_mb=max_size_mb,
        supported_formats=formats or ["csv", "json", "xlsx"],
        max_preview_rows=20 if enable_preview else 0
    )
    
    uploader = FileUploadWidget(config)
    return uploader.render(key)

def list_saved_runs(out_dir: Union[str, Path] = "tmiv_out") -> List[str]:
    """Kompatybilność wsteczna - lista zapisanych runów."""
    base = Path(out_dir)
    if not base.exists():
        return []
    
    runs = [p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")]
    return sorted(runs)

# Funkcje pomocnicze zachowane dla kompatybilności
def _is_date_like_series(s: pd.Series) -> bool:
    """Sprawdza czy seria przypomina daty."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object:
        try:
            parsed = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
            return parsed.notna().mean() >= 0.9
        except Exception:
            return False
    return False

def _auto_side_options(df: pd.DataFrame, target: Optional[str], task: Optional[str]) -> Dict[str, Any]:
    """Heurystyczne domyślne opcje preprocessingu."""
    side = {
        "drop_constant": True,
        "auto_dates": True,
        "limit_cardinality": True,
        "high_card_topk": 50,
        "target_log1p": "auto",
        "target_winsor": "auto",
    }

    has_cat = any(
        (df[c].dtype == "object") or str(df[c].dtype).startswith("category")
        for c in df.columns if c != (target or "")
    )
    if not has_cat:
        side["limit_cardinality"] = False

    has_maybe_date = any(_is_date_like_series(df[c]) for c in df.columns if c != (target or ""))
    if not has_maybe_date:
        side["auto_dates"] = False

    if task == "clf":
        side["target_log1p"] = False
        side["target_winsor"] = False

    return side

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    # Przykład nowego API
    st.set_page_config(page_title="TMIV Demo", layout="wide")
    
    # Konfiguracja
    data_config = DataConfig(max_file_size_mb=500, max_preview_rows=50)
    ui_config = UIConfig(app_title="TMIV — Enhanced Version", enable_llm=True)
    
    # Aplikacja
    app = TMIVApp(data_config, ui_config)
    df, dataset_name, target = app.render_data_selection()
    
    if df is not None and not df.empty:
        # Konfiguracja
        config = app.render_configuration(["auto", "xgboost", "lightgbm"])
        
        # Podsumowanie jakości
        DataQualitySummaryWidget.render(df)
        
        # Pokaż wybraną konfigurację
        st.subheader("⚙️ Wybrana konfiguracja")
        st.json({
            "dataset": dataset_name,
            "target": target,
            "rows": len(df),
            "columns": len(df.columns),
            **config
        })
        
        st.success("✅ Wszystko gotowe do analizy!")
    
    # Przykład starego API (kompatybilność wsteczna)
    st.markdown("---")
    st.subheader("Kompatybilność wsteczna")
    
    # Stary sposób użycia nadal działa
    if st.button("Test starego API"):
        header()
        old_df, old_name = dataset_selector("data")
        if not old_df.empty:
            target = show_detected_target(None, list(old_df.columns))
            config = sidebar_config(["auto", "xgboost"])
            st.write("Stare API działa!", {"dataset": old_name, "target": target})

# --- TMIVApp shim: zachowuje kompatybilność z app.py, gdy ktoś importuje klasę ---
# Jeśli w tym module istnieją funkcje o nazwach takich jak 'header', 'glossary_box',
# wywołania app.header() będą delegowane do header() itd.

from typing import Any, Callable

# Jeśli nie masz tych funkcji w module, dodaj proste wersje awaryjne:
try:
    header  # type: ignore[name-defined]
except NameError:  # brak header() w module -> prosta wersja
    import streamlit as st
    def header(title: str = "TMIV"):
        st.set_page_config(page_title=title, layout="wide")
        st.markdown(f"# {title}")

try:
    glossary_box  # type: ignore[name-defined]
except NameError:  # brak glossary_box() -> prosta wersja
    import streamlit as st
    GLOSSARY = {}
    def glossary_box(location: str = "sidebar"):
        where = st.sidebar if location == "sidebar" else st
        with where:
            where.markdown("### 🧠 Słowniczek")
            if not GLOSSARY:
                where.caption("Brak haseł.")

class TMIVApp:
    """Lekki adapter: app.header() -> header(), app.glossary_box() -> glossary_box(), itd."""
    def __getattr__(self, name: str) -> Any:
        obj = globals().get(name)
        if callable(obj):
            return obj
        raise AttributeError(f"TMIVApp: brak atrybutu lub funkcji '{name}' w frontend.ui_components")

__all__ = [*([n for n in globals() if not n.startswith("_")]), "TMIVApp"]
