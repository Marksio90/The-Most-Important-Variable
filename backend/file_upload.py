from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import io
import json
import logging
import mimetypes
import hashlib
from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileFormat(Enum):
    """Obsługiwane formaty plików."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    EXCEL = "excel"
    FEATHER = "feather"
    PICKLE = "pickle"

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
    size_bytes: int
    size_mb: float
    format: FileFormat
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    hash_md5: Optional[str] = None
    
    # Metadane DataFrame
    rows: int = 0
    columns: int = 0
    memory_mb: float = 0.0
    dtypes: Dict[str, str] = field(default_factory=dict)
    
    # Status i diagnostyka
    status: FileStatus = FileStatus.SUCCESS
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    upload_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje metadane na słownik."""
        return {
            'filename': self.filename,
            'size_mb': self.size_mb,
            'format': self.format.value,
            'rows': self.rows,
            'columns': self.columns,
            'memory_mb': self.memory_mb,
            'status': self.status.value,
            'warnings_count': len(self.warnings),
            'processing_time': self.processing_time,
            'upload_timestamp': self.upload_timestamp.isoformat()
        }

@dataclass
class UploadConfig:
    """Konfiguracja uploadera plików."""
    # Limity
    max_size_mb: int = 200
    max_rows: int = 1_000_000
    max_columns: int = 1000
    
    # CSV settings
    csv_separator: str = ","
    csv_encoding: str = "utf-8"
    csv_auto_detect: bool = True
    csv_low_memory: bool = True
    
    # JSON settings
    json_lines_fallback: bool = True
    json_encoding: str = "utf-8"
    
    # Excel settings
    excel_sheet: Union[str, int, None] = 0
    excel_header: Union[int, List[int], None] = 0
    
    # Ogólne
    sample_rows: int = 1000  # Dla podglądu
    enable_caching: bool = True
    enable_preview: bool = True
    enable_metadata: bool = True
    
    # Walidacja
    require_non_empty: bool = True
    check_duplicates: bool = False
    allowed_extensions: List[str] = field(default_factory=lambda: ["csv", "json", "parquet", "xlsx", "xls", "feather", "pkl"])

class SmartFileUploader:
    """Zaawansowany uploader plików z inteligentną detekcją i walidacją."""
    
    def __init__(self, config: UploadConfig = None):
        self.config = config or UploadConfig()
        self.supported_formats = {
            'csv': FileFormat.CSV,
            'json': FileFormat.JSON,
            'jsonl': FileFormat.JSON,
            'parquet': FileFormat.PARQUET,
            'xlsx': FileFormat.EXCEL,
            'xls': FileFormat.EXCEL,
            'feather': FileFormat.FEATHER,
            'pkl': FileFormat.PICKLE,
            'pickle': FileFormat.PICKLE
        }
    
    def create_upload_widget(
        self, 
        key: str = "file_uploader",
        label: str = "Wgraj plik danych",
        help_text: Optional[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[FileMetadata]]:
        """Tworzy widget do wgrywania plików z zaawansowanymi opcjami."""
        
        # Przygotuj help text
        if not help_text:
            help_text = self._generate_help_text()
        
        # Główny widget
        uploaded_file = st.file_uploader(
            label=label,
            type=self.config.allowed_extensions,
            key=key,
            help=help_text
        )
        
        if not uploaded_file:
            return None, None
        
        # Przetwórz plik
        df, metadata = self._process_uploaded_file(uploaded_file)
        
        # Wyświetl informacje o pliku
        if metadata:
            self._display_file_info(metadata)
            
            # Podgląd danych
            if df is not None and self.config.enable_preview:
                self._display_preview(df, metadata)
        
        return df, metadata
    
    def _generate_help_text(self) -> str:
        """Generuje tekst pomocy dla widgetu."""
        formats = ", ".join(self.config.allowed_extensions)
        return f"Obsługiwane formaty: {formats}. Maksymalny rozmiar: {self.config.max_size_mb}MB"
    
    def _process_uploaded_file(self, uploaded_file) -> Tuple[Optional[pd.DataFrame], FileMetadata]:
        """Przetwarza wgrany plik."""
        start_time = datetime.now()
        
        # Podstawowe metadane
        metadata = FileMetadata(
            filename=uploaded_file.name,
            size_bytes=uploaded_file.size,
            size_mb=round(uploaded_file.size / (1024 * 1024), 3),
            format=self._detect_format(uploaded_file.name),
            mime_type=uploaded_file.type
        )
        
        # Walidacja rozmiaru
        if metadata.size_mb > self.config.max_size_mb:
            metadata.status = FileStatus.TOO_LARGE
            st.error(f"❌ Plik za duży! ({metadata.size_mb}MB > {self.config.max_size_mb}MB)")
            return None, metadata
        
        # Wygeneruj hash pliku
        metadata.hash_md5 = self._calculate_hash(uploaded_file)
        
        try:
            # Wczytaj DataFrame
            df = self._load_dataframe(uploaded_file, metadata)
            
            if df is None:
                return None, metadata
            
            # Waliduj DataFrame
            df, metadata = self._validate_dataframe(df, metadata)
            
            # Oblicz czas przetwarzania
            processing_time = (datetime.now() - start_time).total_seconds()
            metadata.processing_time = round(processing_time, 3)
            
            logger.info(f"Successfully processed {metadata.filename}: {metadata.rows}x{metadata.columns}")
            
            return df, metadata
            
        except Exception as e:
            metadata.status = FileStatus.ERROR
            metadata.warnings.append(f"Błąd przetwarzania: {str(e)}")
            st.error(f"❌ Błąd podczas wczytywania pliku: {str(e)}")
            return None, metadata
    
    def _detect_format(self, filename: str) -> FileFormat:
        """Wykrywa format pliku na podstawie rozszerzenia."""
        suffix = Path(filename).suffix.lower().lstrip('.')
        return self.supported_formats.get(suffix, FileFormat.CSV)
    
    def _calculate_hash(self, uploaded_file) -> str:
        """Oblicza hash MD5 pliku."""
        try:
            uploaded_file.seek(0)
            content = uploaded_file.read()
            uploaded_file.seek(0)
            return hashlib.md5(content).hexdigest()[:8]  # Krótki hash
        except Exception:
            return "unknown"
    
    def _load_dataframe(self, uploaded_file, metadata: FileMetadata) -> Optional[pd.DataFrame]:
        """Wczytuje DataFrame z pliku."""
        uploaded_file.seek(0)
        
        try:
            if metadata.format == FileFormat.CSV:
                return self._load_csv(uploaded_file, metadata)
            elif metadata.format == FileFormat.JSON:
                return self._load_json(uploaded_file, metadata)
            elif metadata.format == FileFormat.PARQUET:
                return self._load_parquet(uploaded_file, metadata)
            elif metadata.format == FileFormat.EXCEL:
                return self._load_excel(uploaded_file, metadata)
            elif metadata.format == FileFormat.FEATHER:
                return self._load_feather(uploaded_file, metadata)
            elif metadata.format == FileFormat.PICKLE:
                return self._load_pickle(uploaded_file, metadata)
            else:
                metadata.status = FileStatus.INVALID_FORMAT
                st.error(f"❌ Nieobsługiwany format: {metadata.format}")
                return None
                
        except UnicodeDecodeError as e:
            metadata.status = FileStatus.ENCODING_ERROR
            metadata.warnings.append(f"Błąd kodowania: {str(e)}")
            st.error("❌ Problem z kodowaniem znaków. Spróbuj zapisać plik w UTF-8.")
            return None
        except pd.errors.EmptyDataError:
            metadata.status = FileStatus.EMPTY_FILE
            st.error("❌ Plik jest pusty lub ma nieprawidłowy format")
            return None
    
    def _load_csv(self, uploaded_file, metadata: FileMetadata) -> pd.DataFrame:
        """Wczytuje plik CSV z inteligentną detekcją."""
        params = {
            'encoding': self.config.csv_encoding,
            'low_memory': self.config.csv_low_memory
        }
        
        if self.config.csv_auto_detect:
            # Próbuj wykryć separator
            sample = uploaded_file.read(1024).decode(self.config.csv_encoding, errors='ignore')
            uploaded_file.seek(0)
            
            separators = [',', ';', '\t', '|']
            best_sep = ','
            max_columns = 0
            
            for sep in separators:
                try:
                    sample_df = pd.read_csv(io.StringIO(sample), sep=sep, nrows=5)
                    if len(sample_df.columns) > max_columns:
                        max_columns = len(sample_df.columns)
                        best_sep = sep
                except Exception:
                    continue
            
            params['sep'] = best_sep
            metadata.warnings.append(f"Auto-wykryty separator: '{best_sep}'")
        else:
            params['sep'] = self.config.csv_separator
        
        return pd.read_csv(uploaded_file, **params)
    
    def _load_json(self, uploaded_file, metadata: FileMetadata) -> pd.DataFrame:
        """Wczytuje plik JSON z obsługą różnych formatów."""
        try:
            # Standardowy JSON
            df = pd.read_json(uploaded_file, encoding=self.config.json_encoding)
            metadata.warnings.append("Wczytano jako standardowy JSON")
            return df
        except ValueError:
            if self.config.json_lines_fallback:
                # JSONL (JSON Lines)
                uploaded_file.seek(0)
                df = pd.read_json(uploaded_file, lines=True, encoding=self.config.json_encoding)
                metadata.warnings.append("Wczytano jako JSON Lines")
                return df
            else:
                raise
    
    def _load_parquet(self, uploaded_file, metadata: FileMetadata) -> pd.DataFrame:
        """Wczytuje plik Parquet."""
        try:
            import pyarrow.parquet as pq
            return pd.read_parquet(uploaded_file)
        except ImportError:
            st.error("❌ Brak biblioteki pyarrow. Zainstaluj: pip install pyarrow")
            return None
    
    def _load_excel(self, uploaded_file, metadata: FileMetadata) -> pd.DataFrame:
        """Wczytuje plik Excel."""
        try:
            import openpyxl  # Dla .xlsx
        except ImportError:
            st.warning("⚠️ Brak biblioteki openpyxl. Niektóre pliki Excel mogą nie działać.")
        
        params = {}
        if self.config.excel_sheet is not None:
            params['sheet_name'] = self.config.excel_sheet
        if self.config.excel_header is not None:
            params['header'] = self.config.excel_header
        
        df = pd.read_excel(uploaded_file, **params)
        
        # Informacja o arkuszu
        if isinstance(self.config.excel_sheet, str):
            metadata.warnings.append(f"Wczytano arkusz: {self.config.excel_sheet}")
        
        return df
    
    def _load_feather(self, uploaded_file, metadata: FileMetadata) -> pd.DataFrame:
        """Wczytuje plik Feather."""
        try:
            import pyarrow.feather as feather
            return pd.read_feather(uploaded_file)
        except ImportError:
            st.error("❌ Brak biblioteki pyarrow. Zainstaluj: pip install pyarrow")
            return None
    
    def _load_pickle(self, uploaded_file, metadata: FileMetadata) -> pd.DataFrame:
        """Wczytuje plik Pickle (z ostrzeżeniem bezpieczeństwa)."""
        st.warning("⚠️ **Ostrzeżenie bezpieczeństwa**: Pliki pickle mogą zawierać niebezpieczny kod!")
        
        if st.checkbox("Rozumiem ryzyko i chcę wczytać plik pickle", key=f"pickle_warning_{metadata.hash_md5}"):
            return pd.read_pickle(uploaded_file)
        else:
            st.stop()
    
    def _validate_dataframe(self, df: pd.DataFrame, metadata: FileMetadata) -> Tuple[pd.DataFrame, FileMetadata]:
        """Waliduje wczytany DataFrame."""
        
        # Podstawowe metadane
        metadata.rows = len(df)
        metadata.columns = len(df.columns)
        metadata.memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3)
        metadata.dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Walidacja pustości
        if self.config.require_non_empty and df.empty:
            metadata.status = FileStatus.EMPTY_FILE
            metadata.warnings.append("DataFrame jest pusty")
            return df, metadata
        
        # Walidacja limitów
        if metadata.rows > self.config.max_rows:
            metadata.warnings.append(f"Plik ma {metadata.rows} wierszy (limit: {self.config.max_rows})")
            df = df.head(self.config.max_rows)
            metadata.rows = len(df)
        
        if metadata.columns > self.config.max_columns:
            metadata.warnings.append(f"Plik ma {metadata.columns} kolumn (limit: {self.config.max_columns})")
            df = df.iloc[:, :self.config.max_columns]
            metadata.columns = len(df.columns)
        
        # Sprawdź duplikaty
        if self.config.check_duplicates:
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                metadata.warnings.append(f"Znaleziono {duplicates} duplikatów wierszy")
        
        # Sprawdź problematyczne kolumny
        self._analyze_data_quality(df, metadata)
        
        return df, metadata
    
    def _analyze_data_quality(self, df: pd.DataFrame, metadata: FileMetadata):
        """Analizuje jakość danych i dodaje ostrzeżenia."""
        
        # Kolumny z wysokim odsetkiem braków
        missing_threshold = 0.8
        high_missing = []
        
        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df)
            if missing_pct > missing_threshold:
                high_missing.append(f"{col} ({missing_pct:.1%})")
        
        if high_missing:
            metadata.warnings.append(f"Kolumny z wysokim odsetkiem braków: {', '.join(high_missing)}")
        
        # Kolumny stałe
        constant_cols = []
        for col in df.columns:
            if df[col].nunique(dropna=True) <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            metadata.warnings.append(f"Kolumny stałe: {', '.join(constant_cols)}")
        
        # Sprawdź dziwne znaki w kolumnach tekstowych
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols[:5]:  # Sprawdź tylko pierwsze 5
            if df[col].astype(str).str.contains(r'[^\x00-\x7F]').any():
                metadata.warnings.append(f"Kolumna '{col}' zawiera znaki niestandardowe")
                break
    
    def _display_file_info(self, metadata: FileMetadata):
        """Wyświetla informacje o wczytanym pliku."""
        
        # Status badge
        if metadata.status == FileStatus.SUCCESS:
            st.success(f"✅ **{metadata.filename}** wczytany pomyślnie")
        elif metadata.status == FileStatus.WARNING:
            st.warning(f"⚠️ **{metadata.filename}** wczytany z ostrzeżeniami")
        else:
            st.error(f"❌ Problem z plikiem **{metadata.filename}**")
            return
        
        # Podstawowe informacje
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Wiersze", f"{metadata.rows:,}")
        with col2:
            st.metric("Kolumny", metadata.columns)
        with col3:
            st.metric("Rozmiar", f"{metadata.size_mb} MB")
        with col4:
            st.metric("Pamięć", f"{metadata.memory_mb} MB")
        
        # Ostrzeżenia
        if metadata.warnings:
            with st.expander(f"⚠️ Ostrzeżenia ({len(metadata.warnings)})", expanded=False):
                for warning in metadata.warnings:
                    st.warning(warning)
        
        # Szczegółowe informacje
        if self.config.enable_metadata:
            with st.expander("📋 Szczegółowe informacje", expanded=False):
                info_data = {
                    "Nazwa pliku": metadata.filename,
                    "Format": metadata.format.value,
                    "Hash MD5": metadata.hash_md5,
                    "Typ MIME": metadata.mime_type,
                    "Czas przetwarzania": f"{metadata.processing_time}s",
                    "Data wgrania": metadata.upload_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.json(info_data)
    
    def _display_preview(self, df: pd.DataFrame, metadata: FileMetadata):
        """Wyświetla podgląd danych."""
        
        st.subheader("🔍 Podgląd danych")
        
        # Opcje podglądu
        col1, col2 = st.columns([3, 1])
        
        with col2:
            sample_size = st.selectbox(
                "Wierszy do pokazania:",
                [5, 10, 20, 50, 100],
                index=1,
                key=f"preview_size_{metadata.hash_md5}"
            )
        
        # Pokaż dane
        st.dataframe(
            df.head(sample_size),
            use_container_width=True,
            height=min(400, (sample_size + 1) * 35)
        )
        
        # Podstawowe statystyki
        if st.checkbox("Pokaż statystyki", key=f"show_stats_{metadata.hash_md5}"):
            st.subheader("📊 Podstawowe statystyki")
            
            # Numeryczne kolumny
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("**Kolumny numeryczne:**")
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            # Kategoryczne kolumny
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                st.write("**Kolumny kategoryczne:**")
                cat_stats = []
                for col in cat_cols[:10]:  # Maksymalnie 10 kolumn
                    stats = {
                        'Kolumna': col,
                        'Unikalne': df[col].nunique(),
                        'Najczęstsza': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A',
                        'Braki': df[col].isna().sum()
                    }
                    cat_stats.append(stats)
                
                st.dataframe(pd.DataFrame(cat_stats), use_container_width=True)


# Funkcje pomocnicze dla łatwego użycia
def upload_widget(
    config: Optional[UploadConfig] = None,
    key: str = "main_uploader",
    label: str = "Wgraj plik danych"
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Prosty wrapper dla kompatybilności wstecznej."""
    uploader = SmartFileUploader(config or UploadConfig())
    df, metadata = uploader.create_upload_widget(key=key, label=label)
    
    filename = metadata.filename if metadata else None
    return df, filename

def advanced_upload_widget(
    max_size_mb: int = 200,
    formats: Optional[List[str]] = None,
    enable_preview: bool = True,
    key: str = "advanced_uploader"
) -> Tuple[Optional[pd.DataFrame], Optional[FileMetadata]]:
    """Zaawansowany wrapper z customizacją."""
    
    config = UploadConfig(
        max_size_mb=max_size_mb,
        enable_preview=enable_preview,
        allowed_extensions=formats or ["csv", "json", "parquet", "xlsx"]
    )
    
    uploader = SmartFileUploader(config)
    return uploader.create_upload_widget(key=key)


# Przykład użycia w aplikacji Streamlit
if __name__ == "__main__":
    st.set_page_config(page_title="Smart File Uploader", layout="wide")
    st.title("🚀 Zaawansowany uploader plików")
    
    # Konfiguracja
    with st.sidebar:
        st.subheader("⚙️ Konfiguracja")
        
        max_size = st.slider("Maksymalny rozmiar (MB)", 1, 500, 200)
        enable_preview = st.checkbox("Podgląd danych", True)
        check_duplicates = st.checkbox("Sprawdź duplikaty", False)
        
        config = UploadConfig(
            max_size_mb=max_size,
            enable_preview=enable_preview,
            check_duplicates=check_duplicates
        )
    
    # Upload widget
    df, metadata = advanced_upload_widget(
        max_size_mb=max_size,
        enable_preview=enable_preview,
        key="demo_uploader"
    )
    
    if df is not None and metadata is not None:
        st.success(f"Gotowe! Masz {len(df)} wierszy i {len(df.columns)} kolumn do analizy.")
        
        # Możesz teraz użyć df do dalszej analizy
        # Przykład: st.dataframe(df)