"""
TMIV - The Most Important Variables
Kompletna aplikacja AutoML z wykorzystaniem wszystkich modułów
"""
from __future__ import annotations

import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import json
import logging
import io
import time
from datetime import datetime

# Konfiguracja strony - musi być pierwsza
st.set_page_config(
    page_title="TMIV - The Most Important Variables",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importy modułów aplikacji - z obsługą błędów
missing_modules = []

# Plotly imports with error handling
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError as e:
    missing_modules.append(f"plotly: {e}")
    px = None
    go = None
    HAS_PLOTLY = False

# Seaborn and matplotlib imports with fallback
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    HAS_SEABORN = True
except ImportError as e:
    missing_modules.append(f"seaborn/matplotlib: {e}")
    sns = None
    plt = None
    HAS_SEABORN = False

# Other utility imports with fallbacks
try:
    import zipfile
    HAS_ZIPFILE = True
except ImportError as e:
    missing_modules.append(f"zipfile: {e}")
    HAS_ZIPFILE = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError as e:
    missing_modules.append(f"joblib: {e}")
    HAS_JOBLIB = False

# 1. Settings - naprawiony import z kompatybilnością
try:
    from config.settings import get_settings
    test_settings = get_settings()
    # Dodaj property dla kompatybilności z starym kodem
    if not hasattr(test_settings, 'data'):
        class DataCompat:
            def __init__(self, parent):
                self.max_file_size_mb = getattr(parent, 'data_max_file_size_mb', 200)
                self.supported_formats = getattr(parent, 'data_supported_formats', ['.csv', '.xlsx', '.json'])
        test_settings.data = DataCompat(test_settings)
    print("✅ Real settings loaded")
except ImportError as e:
    missing_modules.append(f"config.settings: {e}")
    def get_settings():
        class MockData:
            max_file_size_mb = 200
            supported_formats = [".csv", ".xlsx", ".json"]
        
        class MockSettings:
            app_name = "TMIV - The Most Important Variables"
            data = MockData()
            data_max_file_size_mb = 200
            data_supported_formats = [".csv", ".xlsx", ".json"]
            def get_feature_flag(self, flag): 
                return True
        
        return MockSettings()

# 2. Frontend components - naprawiony import
try:
    from frontend.ui_components import TMIVApp, DataConfig, UIConfig
    # Test czy klasy działają
    test_config = DataConfig()
    test_ui = UIConfig()
    print("✅ Real UI components loaded")
except ImportError as e:
    missing_modules.append(f"frontend.ui_components: {e}")
    class DataConfig:
        def __init__(self, **kwargs): 
            self.max_file_size_mb = kwargs.get('max_file_size_mb', 200)
            self.supported_formats = kwargs.get('supported_formats', ['.csv', '.xlsx'])
            self.auto_detect_encoding = kwargs.get('auto_detect_encoding', True)
            self.max_preview_rows = kwargs.get('max_preview_rows', 50)
    
    class UIConfig:
        def __init__(self, **kwargs):
            self.app_title = kwargs.get("app_title", "TMIV")
            self.app_subtitle = kwargs.get("app_subtitle", "AutoML Tool")
            self.enable_llm = kwargs.get("enable_llm", False)
            self.show_advanced_options = kwargs.get("show_advanced_options", True)
    
    class TMIVApp:
        def __init__(self, data_config, ui_config): 
            self.data_config = data_config
            self.ui_config = ui_config
        
        def render_data_selection(self):
            st.info("Upload CSV file to continue")
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                # Auto detect target
                target = None
                for col in df.columns:
                    if any(word in col.lower() for word in ['target', 'y', 'price', 'label', 'averageprice']):
                        target = col
                        break
                if not target and len(df.columns) > 1:
                    target = df.columns[-1]
                return df, uploaded_file.name, target
            return None, None, None

# 3. ML Integration - użyj prawdziwej implementacji
try:
    from backend.ml_integration import (
        train_sklearn_enhanced, ModelConfig, MLTrainingOrchestrator,
        AdvancedMLTrainingOrchestrator
    )
    print("✅ Real ML integration loaded")
    USE_REAL_ML = True
except ImportError as e:
    missing_modules.append(f"backend.ml_integration: {e}")
    USE_REAL_ML = False
    
    class ModelConfig:
        def __init__(self, **kwargs): 
            self.target = kwargs.get('target')
            self.cv_folds = kwargs.get('cv_folds', 5)
            self.hyperopt_trials = kwargs.get('hyperopt_trials', 50)
            self.outlier_detection = kwargs.get('outlier_detection', True)
    
    def train_sklearn_enhanced(df, config):
        # Mock training z lepszym feedbackiem
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
        
        target = config.target
        X = df.drop(columns=[target])
        y = df[target]
        
        # Determine problem type
        is_classification = y.nunique() < 20 and y.dtype == 'object' or y.nunique() <= 10
        
        # Simple preprocessing
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_processed = X[numeric_cols].fillna(X[numeric_cols].median())
        
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        
        if is_classification:
            model = RandomForestClassifier(random_state=42, n_estimators=100)
        else:
            model = RandomForestRegressor(random_state=42, n_estimators=100)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if is_classification:
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        else:
            metrics = {
                "r2": r2_score(y_test, y_pred),
                "rmse": mean_squared_error(y_test, y_pred, squared=False)
            }
        
        feature_importance = pd.DataFrame({
            'feature': X_processed.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metadata = {
            "engine": "mock_sklearn", 
            "run_time": 2.0,
            "problem_type": "classification" if is_classification else "regression",
            "n_features": len(X_processed.columns),
            "model_params": model.get_params()
        }
        
        return model, metrics, feature_importance, metadata

# 4. EDA Integration - użyj prawdziwej implementacji  
try:
    from backend.eda_integration import SmartDataPreprocessor, AdvancedColumnAnalyzer
    print("✅ Real EDA integration loaded")
    USE_REAL_EDA = True
except ImportError as e:
    missing_modules.append(f"backend.eda_integration: {e}")
    USE_REAL_EDA = False
    
    class SmartDataPreprocessor:
        def preprocess(self, df, target=None):
            # Enhanced preprocessing mock
            df_clean = df.copy()
            
            # Drop columns with >90% missing
            for col in df_clean.columns:
                if df_clean[col].isna().sum() / len(df_clean) > 0.9:
                    df_clean = df_clean.drop(columns=[col])
            
            # Fill numeric missing with median
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != target:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
            class Report:
                def __init__(self):
                    self.original_shape = df.shape
                    self.final_shape = df_clean.shape
                    self.processing_time = 1.5
                    self.dropped_columns = {"high_missing": []}
                    self.created_columns = []
                    self.transformations = {"preprocessing": "completed"}
                    self.warnings = []
                
                def to_dict(self):
                    return {
                        "original_shape": self.original_shape,
                        "final_shape": self.final_shape,
                        "processing_time": self.processing_time,
                        "transformations": self.transformations
                    }
            
            return df_clean, Report()

# 5. Utils - użyj prawdziwej implementacji
try:
    from backend.utils import SmartTargetDetector, MLRecommendationEngine
    print("✅ Real utils loaded")
    USE_REAL_UTILS = True  
except ImportError as e:
    missing_modules.append(f"backend.utils: {e}")
    USE_REAL_UTILS = False
    
    class SmartTargetDetector:
        def detect_target(self, df, preferred=None):
            if preferred and preferred in df.columns:
                return preferred
            for col in df.columns:
                if any(word in col.lower() for word in ['target', 'y', 'price', 'label', 'averageprice']):
                    return col
            return df.columns[-1] if len(df.columns) > 0 else None
        
        def analyze_target(self, df, target_col):
            class Analysis:
                def __init__(self):
                    self.problem_type = type('obj', (object,), {'value': 'regression'})()
                    self.unique_values = df[target_col].nunique()
                    self.missing_ratio = df[target_col].isna().sum() / len(df)
                    self.is_balanced = True
                    self.needs_transformation = False
                    self.confidence_score = 0.8
                    self.recommendations = ["Target wygląda dobrze przygotowany"]
            return Analysis()
    
    class MLRecommendationEngine:
        def generate_recommendations(self, analysis=None, **kwargs):
            return """
## 💡 Rekomendacje ML

**Dla regresji:**
- Sprawdź rozkład targetu - jeśli skoś ny, zastosuj log/sqrt transform
- Usuń outliery lub zastosuj winsorization - poprawi RMSE/MAE

**Ogólne:**
- Przetestuj różne algorytmy ML
- Użyj cross-validation do oceny modelu
- Monitoruj performance w czasie
            """

# 6. Database - użyj prawdziwej implementacji
try:
    from db.db_utils import MLExperimentTracker, RunRecord, ProblemType, RunStatus
    print("✅ Real database loaded")
    USE_REAL_DB = True
except ImportError as e:
    missing_modules.append(f"db.db_utils: {e}")
    USE_REAL_DB = False
    
    class ProblemType:
        REGRESSION = "regression"
        BINARY_CLASSIFICATION = "classification"
        MULTICLASS_CLASSIFICATION = "classification"
    
    class RunStatus:
        COMPLETED = "completed"
        RUNNING = "running"
        FAILED = "failed"
    
    class RunRecord:
        def __init__(self, **kwargs): 
            self.run_id = kwargs.get('run_id', 'test')
            self.dataset = kwargs.get('dataset', 'unknown')
            self.target = kwargs.get('target', 'unknown')
            self.problem_type = kwargs.get('problem_type')
            self.engine = kwargs.get('engine')
            self.status = kwargs.get('status', RunStatus.COMPLETED)
            self.metrics = kwargs.get('metrics', {})
            self.parameters = kwargs.get('parameters', {})
            self.notes = kwargs.get('notes', '')
            self.duration_seconds = kwargs.get('duration_seconds', 0)
    
    class MLExperimentTracker:
        def __init__(self): 
            if 'experiment_history' not in st.session_state:
                st.session_state.experiment_history = []
        
        def get_history(self): 
            history = st.session_state.get('experiment_history', [])
            if history:
                return pd.DataFrame(history)
            return pd.DataFrame()
        
        def log_run(self, record): 
            if 'experiment_history' not in st.session_state:
                st.session_state.experiment_history = []
            
            run_data = {
                "run_id": getattr(record, 'run_id', 'test'),
                "dataset": getattr(record, 'dataset', 'unknown'),
                "target": getattr(record, 'target', 'unknown'),
                "problem_type": getattr(record.problem_type, 'value', 'unknown') if hasattr(record, 'problem_type') and record.problem_type else 'unknown',
                "engine": getattr(record, 'engine', 'unknown'),
                "status": getattr(record.status, 'value', 'completed') if hasattr(record, 'status') and record.status else 'completed',
                "created_at": pd.Timestamp.now().isoformat(),
                "duration": getattr(record, 'duration_seconds', 0)
            }
            st.session_state.experiment_history.append(run_data)
            return True
        
        def get_statistics(self): 
            return {"total_runs": len(st.session_state.get('experiment_history', []))}
        
        def clear_history(self, confirm=False):
            if confirm:
                st.session_state.experiment_history = []
        
        def backup_database(self): 
            return Path("backup_session.json")

# Logger
logger = logging.getLogger(__name__)

# Enhanced CSS
st.markdown("""
<style>
[data-testid="stMetricLabel"] > div {
    white-space: normal;
    word-wrap: break-word;
    text-align: center;
}
.stSuccess > div { border-left: 4px solid #28a745; }
.stError > div { border-left: 4px solid #dc3545; }
.stWarning > div { border-left: 4px solid #ffc107; }
.stInfo > div { border-left: 4px solid #17a2b8; }

/* Hide selectbox label duplication */
.stSelectbox > label:first-child { display: none; }

/* Improve spacing */
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

@dataclass
class AppState:
    """Stan aplikacji - centralne zarządzanie"""
    model: Optional[Any] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: Dict[str, Any] = field(default_factory=dict)
    dataset: Optional[pd.DataFrame] = None
    dataset_name: str = ""
    target_column: str = ""
    preprocessing_info: Dict[str, Any] = field(default_factory=dict)
    training_completed: bool = False
    selected_analysis_type: str = "Podsumowanie statystyczne"

def init_app_state() -> AppState:
    """Inicjalizuje stan aplikacji z lepszą trwałością"""
    # NAPRAWKA: Użyj bardziej stabilnego klucza
    if "tmiv_app_state" not in st.session_state:
        st.session_state.tmiv_app_state = AppState()
    
    # NAPRAWKA: Sprawdź spójność stanu
    app_state = st.session_state.tmiv_app_state
    
    # Debug: loguj stan
    if hasattr(st, 'write'):  # Only in debug mode
        pass  # Można dodać logowanie
    
    return app_state

def get_openai_key() -> str:
    """Pobiera klucz OpenAI z różnych źródeł"""
    import os
    import re
    
    # Sprawdź session_state, secrets, env
    sources = [
        st.session_state.get("openai_key", ""),
        getattr(st.secrets, "OPENAI_API_KEY", "") if hasattr(st, 'secrets') else "",
        os.getenv("OPENAI_API_KEY", "")
    ]
    
    for key in sources:
        if key and re.match(r"^sk-[a-zA-Z0-9]{20,}$", key):
            return key
    return ""

class TMIVApplication:
    """Główna klasa aplikacji TMIV wykorzystująca wszystkie moduły"""
    
    def __init__(self):
        self.settings = get_settings()
        # NAPRAWKA: Inicjalizuj stan przed innymi komponentami
        self.state = init_app_state()
        self.experiment_tracker = MLExperimentTracker()
        self._setup_configs()
        
        # NAPRAWKA: Sprawdź stan przy inicjalizacji
        if hasattr(st, 'sidebar'):  # Tylko gdy UI jest dostępne
            if self.state.training_completed:
                st.sidebar.success("✅ Model załadowany z poprzedniej sesji")
        
    def _setup_configs(self):
        """Konfiguruje komponenty aplikacji"""
        # Bezpieczne pobieranie atrybutów dla nowej struktury settings
        max_size = getattr(self.settings, 'data_max_file_size_mb', 
                          getattr(getattr(self.settings, 'data', None), 'max_file_size_mb', 200))
        formats = getattr(self.settings, 'data_supported_formats', 
                         getattr(getattr(self.settings, 'data', None), 'supported_formats', ['.csv', '.xlsx']))
        app_title = getattr(self.settings, 'app_name', 'TMIV')
        
        self.data_config = DataConfig(
            max_file_size_mb=max_size,
            supported_formats=formats,
            auto_detect_encoding=True,
            max_preview_rows=50
        )
        
        self.ui_config = UIConfig(
            app_title=app_title,
            app_subtitle="AutoML • EDA • Historia eksperymentów",
            enable_llm=bool(get_openai_key()),
            show_advanced_options=True
        )
        
        self.tmiv_app = TMIVApp(self.data_config, self.ui_config)
        
        # Setup advanced analyzers if available
        openai_key = get_openai_key()
        if USE_REAL_EDA and openai_key:
            self.column_analyzer = AdvancedColumnAnalyzer(
                enable_llm=True, 
                openai_api_key=openai_key
            )
        elif USE_REAL_EDA:
            self.column_analyzer = AdvancedColumnAnalyzer(enable_llm=False)
        else:
            self.column_analyzer = None
        
    def run(self):
        """Główny punkt wejścia aplikacji - naprawiona logika"""
        # Pokaż ostrzeżenia o brakujących modułach
        if missing_modules:
            with st.expander("⚠️ Informacje o modułach", expanded=False):
                st.warning(f"Niektóre moduły używają mock implementacji:")
                for module in missing_modules:
                    st.text(f"• {module}")
                st.info("Aplikacja działa z ograniczoną funkcjonalnością. Wszystkie podstawowe funkcje są dostępne.")
        
        self._render_header()
        self._render_openai_status()
        
        # NAPRAWKA: Zawsze renderuj sekcję danych
        self._data_loading_phase()
        
        # Advanced ML Section - DODAJ TĘ LINIĘ
        self._render_advanced_ml_section()
        
        # NAPRAWKA: Pokazuj wyniki jeśli są dostępne
        if self.state.training_completed and self.state.model is not None:
            st.markdown("---")  # Separator
            self._results_phase()
        
        # Sekcje dostępne zawsze
        st.markdown("---")  # Separator
        self._render_history_section()
        self._render_sidebar_tools()
        
    def _render_header(self):
        """Renderuje nagłówek aplikacji"""
        st.title(self.ui_config.app_title)
        st.caption(self.ui_config.app_subtitle)
        
        # Status aplikacji - ulepszona wersja
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            if self.state.dataset is not None:
                dataset_info = f"{self.state.dataset_name} ({len(self.state.dataset):,} wierszy × {len(self.state.dataset.columns)} kolumn)"
                st.success(f"📊 **Dane:** {dataset_info}")
            else:
                st.info("📁 Wczytaj dane aby rozpocząć analizę")
                
        with col2:
            if self.state.training_completed:
                st.success("🎯 Model gotowy")
            else:
                st.info("⏳ Oczekuje")
                
        with col3:
            if self.state.target_column:
                st.info(f"🎯 **Target:** {self.state.target_column}")
            else:
                st.info("🎯 Brak targetu")
                
        with col4:
            history = self.experiment_tracker.get_history()
            experiments_count = len(history) if history is not None and not history.empty else 0
            st.metric("Eksperymenty", experiments_count)
    
    def _render_openai_status(self):
        """Status klucza OpenAI bez problemów z rerun"""
        openai_key = get_openai_key()
        
        if openai_key:
            st.success("🤖 **OpenAI aktywny** - Inteligentne opisy kolumn i rekomendacje dostępne")
        else:
            with st.expander("🔑 Konfiguracja OpenAI (opcjonalne - dla lepszych rekomendacji)"):
                st.info("OpenAI umożliwia:")
                st.markdown("""
                - 🧠 Automatyczne rozpoznawanie znaczenia kolumn
                - 💡 Inteligentne rekomendacje ML
                - 📝 Szczegółowe opisy wyników
                """)
                
                key_input = st.text_input(
                    "Klucz OpenAI API",
                    type="password",
                    placeholder="sk-...",
                    help="Wklej klucz z https://platform.openai.com/account/api-keys",
                    key="openai_key_input"
                )
                
                # NAPRAWKA: Przycisk zamiast auto-rerun
                if key_input and st.button("💾 Zapisz klucz OpenAI", key="save_openai_key"):
                    st.session_state["openai_key"] = key_input
                    st.success("✅ Klucz OpenAI zapisany! Funkcje AI są teraz aktywne.")
            
    def _data_loading_phase(self):
        """Faza wczytywania i konfiguracji danych"""
        st.markdown("## 📊 Przygotowanie danych")
        
        # Wczytywanie danych z ulepszonym interfejsem
        df, dataset_name, target = self.tmiv_app.render_data_selection()
        
        if df is None or df.empty:
            st.info("👆 **Wczytaj dane aby kontynuować**")
            st.markdown("""
            **Obsługiwane formaty:** CSV, JSON, Excel, Parquet
            
            **Demo data:** Zestaw danych o avocado zawierający informacje o cenach w różnych regionach USA
            """)
            return
            
        # Aktualizuj stan
        self.state.dataset = df
        self.state.dataset_name = dataset_name
        self.state.target_column = target or ""
        
        # Enhanced EDA section
        self._render_eda_section(df)
        
        # Trening z walidacją
        if target and target in df.columns:
            self._render_training_section(df, target)
        else:
            st.warning("⚠️ Wybierz prawidłową kolumnę docelową aby rozpocząć trening modelu")
    
    def _render_eda_section(self, df: pd.DataFrame):
        """Ulepszona sekcja analizy eksploracyjnej"""
        st.markdown("## 🔬 Analiza eksploracyjna")
        
        # Szybkie podsumowanie z dodatkowymi metrykami
        with st.expander("📋 Podsumowanie danych", expanded=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Wiersze", f"{len(df):,}")
            with col2:
                st.metric("Kolumny", len(df.columns))
            with col3:
                missing_pct = (df.isna().sum().sum() / df.size) * 100
                st.metric("Braki danych", f"{missing_pct:.1f}%")
            with col4:
                duplicates = df.duplicated().sum()
                st.metric("Duplikaty", duplicates)
            with col5:
                memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
                st.metric("Pamięć", f"{memory_mb:.1f} MB")
        
        # Persistent selectbox using session state
        if "eda_analysis_type" not in st.session_state:
            st.session_state.eda_analysis_type = "Podsumowanie statystyczne"
        
        analysis_options = [
            "Podsumowanie statystyczne",
            "Rozkłady zmiennych", 
            "Korelacje",
            "Analiza targetu",
            "Jakość danych",
            "Analiza kolumn (AI)" if self.ui_config.enable_llm else None
        ]
        analysis_options = [opt for opt in analysis_options if opt is not None]
        
        # Use session state for persistent selection
        current_index = analysis_options.index(st.session_state.eda_analysis_type) if st.session_state.eda_analysis_type in analysis_options else 0
        
        analysis_type = st.selectbox(
            "**Wybierz rodzaj analizy:**",
            analysis_options,
            index=current_index,
            key="eda_analysis_selector"
        )
        
        # Update session state
        st.session_state.eda_analysis_type = analysis_type
        
        # Render analysis based on selection
        if analysis_type == "Podsumowanie statystyczne":
            self._render_statistical_summary(df)
            
        elif analysis_type == "Rozkłady zmiennych":
            self._render_distributions(df)
            
        elif analysis_type == "Korelacje":
            self._render_correlations(df)
            
        elif analysis_type == "Analiza targetu":
            self._render_target_analysis(df)
            
        elif analysis_type == "Jakość danych":
            self._render_data_quality(df)
            
        elif analysis_type == "Analiza kolumn (AI)" and self.ui_config.enable_llm:
            self._render_ai_column_analysis(df)
    
    def _render_statistical_summary(self, df: pd.DataFrame):
        """Renderuje podsumowanie statystyczne"""
        st.subheader("📈 Statystyki opisowe")
        
        # Separate numeric and non-numeric
        numeric_df = df.select_dtypes(include=[np.number])
        non_numeric_df = df.select_dtypes(exclude=[np.number])
        
        if not numeric_df.empty:
            st.write("**Zmienne numeryczne:**")
            st.dataframe(numeric_df.describe(), use_container_width=True)
        
        if not non_numeric_df.empty:
            st.write("**Zmienne kategoryczne:**")
            categorical_stats = []
            for col in non_numeric_df.columns:
                stats = {
                    'Kolumna': col,
                    'Unikalne wartości': df[col].nunique(),
                    'Najczęstsza wartość': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A',
                    'Częstość najczęstszej': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0,
                    'Braki danych': df[col].isna().sum()
                }
                categorical_stats.append(stats)
            
            if categorical_stats:
                st.dataframe(pd.DataFrame(categorical_stats), use_container_width=True)

    def _render_distributions(self, df: pd.DataFrame):
        """Renderuje rozkłady zmiennych z ulepszeniami"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.info("Brak zmiennych numerycznych do analizy rozkładu")
            return
            
        st.subheader(f"📊 Rozkłady zmiennych ({len(numeric_cols)} dostępnych)")
        
        # Multi-select with better default
        default_cols = numeric_cols[:min(3, len(numeric_cols))]
        selected_cols = st.multiselect(
            "Wybierz kolumny do analizy:", 
            numeric_cols, 
            default=default_cols,
            key="distribution_columns"
        )
        
        if selected_cols:
            if HAS_PLOTLY:  # ZMIENIONE: użyj HAS_PLOTLY zamiast px
                # Create subplots for multiple columns
                for i, col in enumerate(selected_cols):
                    st.write(f"**{col}**")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig = px.histogram(df, x=col, marginal="box", 
                                        title=f"Rozkład: {col}")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Statistics
                        stats = df[col].describe()
                        st.dataframe(stats, use_container_width=True)
                        
                        # Additional info
                        skewness = df[col].skew()
                        st.metric("Skośność", f"{skewness:.3f}")
                        
                        if abs(skewness) > 1:
                            st.warning("Rozkład jest skośny - rozważ transformację")
                        elif abs(skewness) < 0.5:
                            st.success("Rozkład zbliżony do normalnego")
                        else:
                            st.info("Rozkład umiarkowanie skośny")
            else:
                # Fallback bez plotly
                for col in selected_cols:
                    st.subheader(f"Statystyki: {col}")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write(df[col].describe())
                    with col2:
                        st.write(f"Skośność: {df[col].skew():.3f}")
                        st.write(f"Kurtoza: {df[col].kurtosis():.3f}")

    def _render_correlations(self, df: pd.DataFrame):
        """Renderuje ulepszony analizę korelacji"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            st.info("Potrzeba co najmniej 2 zmiennych numerycznych do analizy korelacji")
            return
            
        st.subheader("🔗 Analiza korelacji")
        
        corr_matrix = numeric_df.corr()
        
        if HAS_PLOTLY:  # ZMIENIONE: użyj HAS_PLOTLY
            # Enhanced heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Macierz korelacji",
                zmin=-1, zmax=1
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(corr_matrix.round(3), use_container_width=True)
        
        # Enhanced strong correlations analysis
        strong_corr = []
        threshold = st.slider("Próg silnej korelacji", 0.5, 0.95, 0.7, 0.05)
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    strong_corr.append({
                        'Zmienna 1': corr_matrix.columns[i],
                        'Zmienna 2': corr_matrix.columns[j],
                        'Korelacja': round(corr_val, 3),
                        'Siła': 'Bardzo silna' if abs(corr_val) > 0.9 else 'Silna'
                    })
        
        if strong_corr:
            st.subheader(f"Silne korelacje (|r| > {threshold})")
            df_corr = pd.DataFrame(strong_corr)
            st.dataframe(df_corr, use_container_width=True)
            
            if any(abs(item['Korelacja']) > 0.95 for item in strong_corr):
                st.warning("⚠️ Wykryto bardzo silne korelacje - rozważ usunięcie niektórych zmiennych")
        else:
            st.info(f"Brak korelacji silniejszych niż {threshold}")

    def _render_target_analysis(self, df: pd.DataFrame):
        """Ulepszona analiza zmiennej docelowej"""
        if not self.state.target_column:
            st.info("Wybierz zmienną docelową w sekcji wyboru danych")
            return
            
        target_col = self.state.target_column
        st.subheader(f"🎯 Analiza targetu: **{target_col}**")
        
        if USE_REAL_UTILS:
            detector = SmartTargetDetector()
            
            try:
                analysis = detector.analyze_target(df, target_col)
                
                # Enhanced target info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Typ problemu", getattr(analysis.problem_type, 'value', 'unknown'))
                with col2:
                    st.metric("Unikalne wartości", analysis.unique_values)
                with col3:
                    st.metric("Braki danych", f"{analysis.missing_ratio:.1%}")
                with col4:
                    quality_color = {
                        'excellent': '🟢', 'good': '🟡', 'poor': '🔴', 'unknown': '⚪'
                    }
                    quality_emoji = quality_color.get(getattr(analysis, 'quality', 'unknown').value if hasattr(analysis, 'quality') else 'unknown', '⚪')
                    st.metric("Jakość", f"{quality_emoji} {getattr(analysis, 'quality', 'unknown').value if hasattr(analysis, 'quality') else 'unknown'}")
                
                # Visualization with problem type detection
                problem_type = getattr(analysis.problem_type, 'value', 'unknown')
                
                if HAS_PLOTLY and problem_type == "regression":  # ZMIENIONE: dodaj HAS_PLOTLY
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig = px.histogram(df, x=target_col, marginal="box", 
                                        title=f"Rozkład targetu: {target_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        target_stats = df[target_col].describe()
                        st.dataframe(target_stats)
                        
                elif HAS_PLOTLY and problem_type in ["classification", "binary_classification"]:  # ZMIENIONE
                    value_counts = df[target_col].value_counts()
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                title=f"Rozkład klas: {target_col}")
                        fig.update_xaxis(title="Klasy")
                        fig.update_yaxis(title="Liczebność")
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.write("**Rozkład klas:**")
                        class_dist = pd.DataFrame({
                            'Klasa': value_counts.index,
                            'Liczebność': value_counts.values,
                            'Procent': (value_counts.values / len(df) * 100).round(1)
                        })
                        st.dataframe(class_dist, use_container_width=True)
                else:
                    # Fallback
                    st.write("**Podstawowe statystyki:**")
                    if df[target_col].dtype in ['object', 'category']:
                        st.write(df[target_col].value_counts().head(10))
                    else:
                        st.write(df[target_col].describe())
                        
                # Show recommendations if available
                if hasattr(analysis, 'recommendations') and analysis.recommendations:
                    st.subheader("💡 Rekomendacje dla targetu")
                    for rec in analysis.recommendations:
                        st.write(f"• {rec}")
                        
            except Exception as e:
                st.error(f"Błąd analizy targetu: {e}")
                # Fallback analysis
                self._render_basic_target_analysis(df, target_col)
        else:
            self._render_basic_target_analysis(df, target_col)

    def create_predictions_vs_actual_plot(y_test, y_pred, model_type):
        """Create predictions vs actual plot - with fallback if seaborn/plotly unavailable"""
        if model_type == 'regression':
            # Scatter plot for regression
            if HAS_PLOTLY:  # ZMIENIONE
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=y_test, y=y_pred,
                    mode='markers',
                    name='Predictions',
                    marker=dict(color='blue', opacity=0.6)
                ))
                
                # Perfect prediction line
                min_val = min(min(y_test), min(y_pred))
                max_val = max(max(y_test), max(y_pred))
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title="Regression: Predicted vs Actual Values",
                    xaxis_title="Actual Values",
                    yaxis_title="Predicted Values"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Plotly not available - showing basic statistics instead")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Actual values stats:**")
                    st.write(pd.Series(y_test).describe())
                with col2:
                    st.write("**Predicted values stats:**")
                    st.write(pd.Series(y_pred).describe())
                
        else:  # classification
            # Confusion Matrix
            try:
                from sklearn.metrics import confusion_matrix
                
                cm = confusion_matrix(y_test, y_pred)
                
                if HAS_SEABORN and HAS_PLOTLY:  # ZMIENIONE
                    # Use seaborn if available
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                    
                elif HAS_PLOTLY:  # ZMIENIONE
                    # Fallback to plotly heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        text=cm,
                        texttemplate="%{text}",
                        textfont={"size": 20},
                        colorscale='Blues'
                    ))
                    fig.update_layout(
                        title="Confusion Matrix",
                        xaxis_title="Predicted",
                        yaxis_title="Actual"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Basic text display
                    st.write("**Confusion Matrix:**")
                    st.dataframe(pd.DataFrame(cm), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Could not create confusion matrix: {str(e)}")
                st.write("**Classification Results Summary:**")
                st.write(f"Total predictions: {len(y_pred)}")

    def plot_feature_importance(feature_importance_df: pd.DataFrame, top_n: int = 20):
        """Create interactive feature importance plot - with fallback"""
        if feature_importance_df.empty:
            st.warning("Feature importance not available for this model")
            return
        
        st.subheader("🎯 Feature Importance")
        
        # Take top N features
        top_features = feature_importance_df.head(top_n)
        
        if HAS_PLOTLY:  # ZMIENIONE
            # Create horizontal bar chart
            fig = go.Figure(go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker_color='lightblue',
                text=[f'{v:.4f}' for v in top_features['importance']],
                textposition='auto',
            ))
            
            fig.update_layout(
                title=f"Top {len(top_features)} Most Important Features",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=max(400, len(top_features) * 25),
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to simple display
            st.bar_chart(top_features.set_index('feature')['importance'])
            
            # Show as table
            st.dataframe(
                top_features[['feature', 'importance']].rename(columns={
                    'feature': 'Feature', 
                    'importance': 'Importance'
                }),
                use_container_width=True,
                hide_index=True
            )

    def create_model_export_section(model, metrics_dict, feature_importance_df, X_test=None, y_test=None):
        """Advanced model export functionality - with fallback if joblib unavailable"""
        st.subheader("📦 Export & Download")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Model pickle export
            if st.button("💾 Download Model") and model is not None:
                if HAS_JOBLIB:  # ZMIENIONE
                    buffer = io.BytesIO()
                    joblib.dump(model, buffer)
                    buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download .joblib",
                        data=buffer.getvalue(),
                        file_name=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib",
                        mime="application/octet-stream"
                    )
                else:
                    st.warning("Joblib not available - model export disabled")
    
    
    def _render_basic_target_analysis(self, df: pd.DataFrame, target_col: str):
        """Podstawowa analiza targetu (fallback)"""
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unikalne wartości", df[target_col].nunique())
        with col2:
            missing_pct = df[target_col].isna().sum() / len(df) * 100
            st.metric("Braki danych", f"{missing_pct:.1f}%")
        with col3:
            if df[target_col].dtype in ['int64', 'float64']:
                st.metric("Typ", "Numeryczny")
            else:
                st.metric("Typ", "Kategoryczny")
        
        if df[target_col].dtype in ['object', 'category']:
            st.write("**Rozkład wartości:**")
            st.write(df[target_col].value_counts().head(10))
        else:
            st.write("**Statystyki opisowe:**")
            st.write(df[target_col].describe())
    
    def _render_data_quality(self, df: pd.DataFrame):
        """Ulepszona analiza jakości danych"""
        st.subheader("🔍 Analiza jakości danych")
        
        quality_issues = []
        recommendations = []
        
        # Enhanced missing data analysis
        missing = df.isnull().sum()
        for col, miss_count in missing.items():
            if miss_count > 0:
                miss_pct = (miss_count / len(df)) * 100
                severity = "🔴 Krytyczne" if miss_pct > 50 else "🟡 Umiarkowane" if miss_pct > 20 else "🟢 Niskie"
                quality_issues.append({
                    'Kolumna': col,
                    'Problem': 'Braki danych',
                    'Wartość': f"{miss_pct:.1f}% ({miss_count} wierszy)",
                    'Priorytet': severity
                })
                
                if miss_pct > 50:
                    recommendations.append(f"Rozważ usunięcie kolumny '{col}' (>50% braków)")
                elif miss_pct > 20:
                    recommendations.append(f"Zastosuj imputację dla kolumny '{col}'")
        
        # Constant columns analysis
        for col in df.columns:
            unique_count = df[col].nunique()
            if unique_count <= 1:
                quality_issues.append({
                    'Kolumna': col,
                    'Problem': 'Kolumna stała',
                    'Wartość': f"{unique_count} unikatowych wartości",
                    'Priorytet': "🔴 Krytyczne"
                })
                recommendations.append(f"Usuń kolumnę stałą '{col}'")
        
        # High cardinality analysis
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.8:
                quality_issues.append({
                    'Kolumna': col,
                    'Problem': 'Wysoka kardynalność',
                    'Wartość': f"{df[col].nunique()} unikalnych z {len(df)} ({unique_ratio:.1%})",
                    'Priorytet': "🟡 Umiarkowane"
                })
                recommendations.append(f"Rozważ grupowanie wartości w kolumnie '{col}'")
        
        # Data type issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if should be numeric
                try:
                    numeric_convert = pd.to_numeric(df[col], errors='coerce')
                    if numeric_convert.notna().sum() / len(df) > 0.8:
                        quality_issues.append({
                            'Kolumna': col,
                            'Problem': 'Typ danych',
                            'Wartość': 'Prawdopodobnie numeryczna',
                            'Priorytet': "🟢 Niskie"
                        })
                        recommendations.append(f"Konwertuj '{col}' na typ numeryczny")
                except:
                    pass
        
        # Display results
        if quality_issues:
            st.dataframe(pd.DataFrame(quality_issues), use_container_width=True)
        else:
            st.success("✅ Nie wykryto poważnych problemów z jakością danych!")
        
        if recommendations:
            st.subheader("💡 Rekomendacje")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
    
    def _render_ai_column_analysis(self, df: pd.DataFrame):
        """AI-powered column analysis"""
        if not self.column_analyzer:
            st.info("Analiza AI niedostępna - brak integracji z OpenAI")
            return
            
        st.subheader("🤖 Inteligentna analiza kolumn")
        
        # Column selection for analysis
        selected_cols = st.multiselect(
            "Wybierz kolumny do analizy AI:",
            df.columns.tolist(),
            default=df.columns.tolist()[:5],
            key="ai_analysis_columns"
        )
        
        if selected_cols and st.button("🚀 Uruchom analizę AI", type="primary"):
            try:
                with st.spinner("Analizuję kolumny z pomocą AI..."):
                    analyses = []
                    for col in selected_cols:
                        analysis = self.column_analyzer.analyze_column(df[col], col)
                        analyses.append(analysis)
                
                # Display results
                for analysis in analyses:
                    with st.expander(f"📊 {analysis.name} ({analysis.column_type.value})", expanded=True):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write("**Opis AI:**")
                            st.write(analysis.description)
                            
                            if analysis.recommendations:
                                st.write("**Rekomendacje:**")
                                for rec in analysis.recommendations:
                                    st.write(f"• {rec}")
                        
                        with col2:
                            st.write("**Szczegóły:**")
                            st.write(f"Typ: {analysis.column_type.value}")
                            st.write(f"Jakość: {analysis.quality.value}")
                            st.write(f"Unikalne: {analysis.unique_values}")
                            st.write(f"Braki: {analysis.missing_percentage:.1f}%")
                            
                            if analysis.sample_values:
                                st.write("**Przykłady:**")
                                st.write(", ".join(map(str, analysis.sample_values[:3])))
                                
            except Exception as e:
                st.error(f"Błąd analizy AI: {e}")
    
    def _render_training_section(self, df: pd.DataFrame, target: str):
        """Ulepszona sekcja treningu modelu"""
        st.markdown("## 🚀 Trening modelu ML")
        
        # Validate target
        if target not in df.columns:
            st.error(f"Kolumna docelowa '{target}' nie istnieje w danych!")
            return
        
        # Enhanced model configuration
        with st.expander("⚙️ Konfiguracja treningu", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                cv_folds = st.slider("Cross-validation (folds)", 2, 10, 5)
                remove_outliers = st.checkbox("Usuń outliery", value=True)
                test_size = st.slider("Rozmiar zbioru testowego", 0.1, 0.4, 0.2, 0.05)
                
            with col2:
                hyperopt_trials = st.slider("Optymalizacja hiperparametrów", 10, 200, 50)
                compute_shap = st.checkbox("Oblicz SHAP values", value=len(df) < 1000)
                random_state = st.number_input("Random seed", value=42, min_value=0)
                
                # DODANE: Wybór silnika ML
                ml_engine = st.selectbox("Silnik ML:", [
                    "auto_pycaret" if 'pycaret' not in missing_modules else "auto",
                    "sklearn", 
                    "lightgbm", 
                    "xgboost",
                    "pycaret"
                ], help="auto_pycaret = PyCaret z porównaniem wielu modeli")
        
        # Enhanced training button with validation
        target_valid = df[target].notna().sum() > 10
        features_valid = len(df.select_dtypes(include=[np.number]).columns) > 0

        if not target_valid:
            st.warning("Za mało prawidłowych wartości w kolumnie docelowej")
        elif not features_valid:
            st.warning("Brak zmiennych numerycznych do treningu")
        else:
            if st.button("Rozpocznij trening modelu", type="primary", use_container_width=True):
                self._train_model_enhanced(df, target, cv_folds, remove_outliers, 
                                        hyperopt_trials, compute_shap, test_size, random_state, ml_engine)  # DODANE ml_engine
            
    def _render_advanced_ml_section(self):
        """Advanced ML section integrated with TMIV"""
        st.markdown("## 🤖 Advanced Machine Learning Suite")
        
        # Check if we have data
        if self.state.dataset is None:
            st.warning("⚠️ Please upload data first in the data loading section above!")
            return
        
        data = self.state.dataset
        
        # Model configuration section
        st.subheader("🎯 Enhanced Model Configuration")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Target selection with current state
            current_target_idx = 0
            if self.state.target_column and self.state.target_column in data.columns:
                current_target_idx = list(data.columns).index(self.state.target_column)
            
            target_col = st.selectbox(
                "Select Target Variable", 
                data.columns, 
                index=current_target_idx,
                key="advanced_ml_target_select"
            )
            
            # Update state
            self.state.target_column = target_col
            
            # Feature selection
            all_features = [col for col in data.columns if col != target_col]
            feature_cols = st.multiselect(
                "Select Features (leave empty for all)",
                all_features,
                default=[],
                key="advanced_ml_feature_select"
            )
            
            if not feature_cols:
                feature_cols = all_features
        
        with col2:
            st.subheader("⚙️ Advanced Settings")
            
            # Engine selection based on available modules
            engine_options = ['auto']
            if not any('pycaret' in str(m) for m in missing_modules):
                engine_options.extend(['pycaret', 'auto_pycaret'])
            engine_options.extend(['sklearn', 'lightgbm', 'xgboost'])
            
            engine = st.selectbox("ML Engine", engine_options, index=0, key="advanced_ml_engine")
            
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, key="advanced_ml_test_size")
            random_state = st.number_input("Random State", 0, 1000, 42, key="advanced_ml_random_state")
            
            cv_folds = st.slider("Cross Validation", 2, 10, 5, key="advanced_ml_cv")
            hyperopt_trials = st.slider("Hyperparameter Optimization", 10, 100, 50, key="advanced_ml_hyperopt")
        
        # Auto-detect problem type
        if target_col and target_col in data.columns:
            try:
                if USE_REAL_ML:
                    from backend.ml_integration import detect_problem_type
                    detected_problem_type = detect_problem_type(data[target_col])
                else:
                    # Fallback detection
                    unique_vals = data[target_col].nunique()
                    if unique_vals <= 10 and data[target_col].dtype in ['object', 'category']:
                        detected_problem_type = "classification"
                    elif unique_vals <= 20 and data[target_col].dtype in ['int64']:
                        detected_problem_type = "classification"
                    else:
                        detected_problem_type = "regression"
                
                st.info(f"🔍 Detected problem type: **{detected_problem_type}**")
            except:
                detected_problem_type = "unknown"
        
        # Training section
        if len(feature_cols) > 0 and target_col:
            st.divider()
            st.subheader("🏋️ Enhanced Model Training")
            
            # Training tabs for different approaches
            train_tab1, train_tab2, train_tab3 = st.tabs([
                "🚀 Quick Training", "🎯 AutoML (PyCaret)", "🔧 Advanced Options"
            ])
            
            with train_tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    use_advanced = st.checkbox(
                        "Use Advanced Training Pipeline", 
                        value=True, 
                        key="advanced_ml_use_advanced"
                    )
                    
                with col2:
                    if st.button("🚀 Train Model", type="primary", key="advanced_ml_train_button"):
                        self._train_advanced_ml_model(
                            data, target_col, feature_cols, engine, test_size, 
                            random_state, cv_folds, hyperopt_trials, use_advanced, detected_problem_type
                        )
            
            with train_tab2:
                if not any('pycaret' in str(m) for m in missing_modules):
                    st.write("**🎯 Automated ML with PyCaret**")
                    st.info("PyCaret will compare multiple algorithms and select the best one automatically")
                    
                    if st.button("🔍 Compare & Train Best Model", key="pycaret_auto_train"):
                        self._train_pycaret_automl(data, target_col, feature_cols, detected_problem_type)
                else:
                    st.warning("PyCaret not available - using fallback AutoML")
                    if st.button("🔍 Train AutoML (Fallback)", key="fallback_auto_train"):
                        self._train_advanced_ml_model(
                            data, target_col, feature_cols, 'auto', test_size, 
                            random_state, cv_folds, hyperopt_trials, True, detected_problem_type
                        )
            
            with train_tab3:
                st.write("**🔧 Advanced Configuration**")
                
                with st.expander("Preprocessing Options"):
                    handle_outliers = st.checkbox("Remove Outliers", value=True)
                    feature_scaling = st.selectbox("Feature Scaling", ['standard', 'robust', 'minmax', 'none'])
                    handle_imbalance = st.checkbox("Handle Class Imbalance", value=False)
                
                with st.expander("Model Options"):
                    ensemble_methods = st.checkbox("Use Ensemble Methods", value=False)
                    compute_shap = st.checkbox("Compute SHAP Values", value=False)
                    early_stopping = st.checkbox("Early Stopping", value=True)
                
                if st.button("🚀 Train Advanced Model", key="advanced_custom_train"):
                    advanced_config = {
                        'handle_outliers': handle_outliers,
                        'feature_scaling': feature_scaling,
                        'handle_imbalance': handle_imbalance,
                        'ensemble_methods': ensemble_methods,
                        'compute_shap': compute_shap,
                        'early_stopping': early_stopping
                    }
                    self._train_advanced_ml_model(
                        data, target_col, feature_cols, engine, test_size, 
                        random_state, cv_folds, hyperopt_trials, True, detected_problem_type, advanced_config
                    )
        
        # Display results if training completed
        if hasattr(st.session_state, 'advanced_ml_result') and st.session_state.advanced_ml_result:
            self._display_advanced_ml_results()
        
        # Model comparison dashboard
        self._display_ml_model_history()

    def _train_advanced_ml_model(self, data, target_col, feature_cols, engine, test_size, 
                                random_state, cv_folds, hyperopt_trials, use_advanced, 
                                detected_problem_type, advanced_config=None):
        """Train advanced ML model with comprehensive pipeline"""
        
        with st.spinner("Training advanced ML model... This may take several minutes"):
            try:
                # Prepare training data
                training_data = data[feature_cols + [target_col]].copy()
                
                start_time = time.time()
                
                if USE_REAL_ML:
                    # Use real ML integration
                    from backend.ml_integration import ModelConfig, train_model_comprehensive
                    
                    config = ModelConfig(
                        target=target_col,
                        engine=engine,
                        cv_folds=cv_folds,
                        test_size=test_size,
                        random_state=random_state,
                        hyperopt_trials=hyperopt_trials,
                        use_optuna=True,
                        feature_engineering=True,
                        outlier_detection=advanced_config.get('handle_outliers', True) if advanced_config else True
                    )
                    
                    result = train_model_comprehensive(
                        df=training_data, 
                        config=config, 
                        use_advanced=use_advanced
                    )
                    
                else:
                    # Enhanced mock training
                    config = ModelConfig(
                        target=target_col,
                        cv_folds=cv_folds,
                        hyperopt_trials=hyperopt_trials,
                        outlier_detection=True
                    )
                    
                    model, metrics, fi_df, metadata = train_sklearn_enhanced(training_data, config)
                    
                    # Create result-like object
                    class MockResult:
                        def __init__(self):
                            self.model = model
                            self.metrics = metrics
                            self.feature_importance = fi_df
                            self.metadata = metadata
                            self.training_time = time.time() - start_time
                            self.preprocessing_info = {}
                    
                    result = MockResult()
                
                # Store results
                st.session_state.advanced_ml_result = result
                st.session_state.advanced_ml_feature_cols = feature_cols
                st.session_state.advanced_ml_target_col = target_col
                st.session_state.advanced_ml_problem_type = detected_problem_type
                
                # Update main app state as well
                self.state.model = result.model
                self.state.metrics = result.metrics
                self.state.feature_importance = result.feature_importance
                self.state.metadata = result.metadata
                self.state.training_completed = True
                
                st.success(f"✅ Advanced ML model trained successfully! Training time: {result.training_time:.2f}s")
                st.balloons()
                
            except Exception as e:
                st.error(f"Advanced ML training failed: {str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

    def _train_pycaret_automl(self, data, target_col, feature_cols, detected_problem_type):
        """Train model using PyCaret AutoML"""
        
        with st.spinner("Running PyCaret AutoML... Comparing multiple models"):
            try:
                training_data = data[feature_cols + [target_col]].copy()
                
                if USE_REAL_ML:
                    from backend.ml_integration import PyCaretTrainer
                    
                    trainer = PyCaretTrainer()
                    trainer.setup_pycaret(training_data.drop(columns=[target_col]), training_data[target_col])
                    
                    # Compare models
                    comparison_results = trainer.compare_models()
                    st.subheader("🏆 Model Comparison Results")
                    st.dataframe(comparison_results.head(10))
                    
                    # Train best model
                    trainer.fit(training_data.drop(columns=[target_col]), training_data[target_col], model_name='best')
                    
                    # Create result object
                    class PyCaretResult:
                        def __init__(self):
                            self.model = trainer.model
                            self.metrics = trainer.metrics
                            self.feature_importance = trainer.get_feature_importance()
                            self.metadata = {'engine': 'pycaret', 'comparison_results': comparison_results}
                            self.training_time = 0  # PyCaret handles timing internally
                            self.preprocessing_info = {}
                    
                    result = PyCaretResult()
                    
                    # Store results
                    st.session_state.advanced_ml_result = result
                    st.session_state.advanced_ml_feature_cols = feature_cols
                    st.session_state.advanced_ml_target_col = target_col
                    st.session_state.advanced_ml_problem_type = detected_problem_type
                    
                    st.success("✅ PyCaret AutoML completed! Best model selected and trained.")
                    
                else:
                    st.warning("PyCaret not available - using fallback")
                    
            except Exception as e:
                st.error(f"PyCaret AutoML failed: {str(e)}")

    def _display_advanced_ml_results(self):
        """Display comprehensive ML results"""
        st.divider()
        st.markdown("## 📊 Advanced ML Results")
        
        result = st.session_state.advanced_ml_result
        feature_cols = st.session_state.advanced_ml_feature_cols
        target_col = st.session_state.advanced_ml_target_col
        problem_type = st.session_state.advanced_ml_problem_type
        
        if result.model is None:
            st.error("No trained model available")
            return
        
        # Create comprehensive tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Performance", "🎯 Predictions", "📊 Visualizations", 
            "🔍 Model Analysis", "📦 Export", "ℹ️ Details"
        ])
        
        with tab1:
            # Performance metrics
            display_model_metrics(result.metrics, problem_type)
            
            # Cross-validation if available
            if hasattr(result, 'metrics') and 'cv_scores' in result.metrics:
                st.subheader("🔄 Cross-Validation Results")
                cv_scores = result.metrics['cv_scores']
                if cv_scores:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("CV Mean", f"{np.mean(cv_scores):.4f}")
                    with col2:
                        st.metric("CV Std", f"{np.std(cv_scores):.4f}")
                    with col3:
                        st.metric("CV Range", f"±{1.96 * np.std(cv_scores):.4f}")
        
        with tab2:
            # Predictions interface
            self._render_advanced_predictions_interface(result, feature_cols, target_col, problem_type)
        
        with tab3:
            # Advanced visualizations
            self._render_advanced_visualizations(result, feature_cols, target_col, problem_type)
        
        with tab4:
            # Model analysis
            self._render_model_analysis(result, feature_cols)
        
        with tab5:
            # Export functionality
            self._render_export_functionality(result, feature_cols, target_col)
        
        with tab6:
            # Detailed information
            self._render_detailed_model_info(result)

    def _render_advanced_predictions_interface(self, result, feature_cols, target_col, problem_type):
        """Advanced predictions interface"""
        st.subheader("🎯 Model Predictions")
        
        # Split data for testing
        from sklearn.model_selection import train_test_split
        
        X = self.state.dataset[feature_cols]
        y = self.state.dataset[target_col]
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if problem_type == "classification" and y.nunique() > 1 else None
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pred_col1, pred_col2 = st.columns([2, 1])
        
        with pred_col1:
            if st.button("📊 Predict on Test Set", key="advanced_predict_test"):
                try:
                    predictions = result.model.predict(X_test)
                    
                    # Create comprehensive results
                    results_df = X_test.copy()
                    results_df['Actual'] = y_test.values
                    results_df['Predicted'] = predictions
                    
                    if problem_type == 'regression':
                        results_df['Error'] = results_df['Actual'] - results_df['Predicted']
                        results_df['Absolute_Error'] = abs(results_df['Error'])
                        results_df['Percentage_Error'] = (results_df['Error'] / results_df['Actual'] * 100).round(2)
                    
                    st.dataframe(results_df.head(20), use_container_width=True)
                    
                    # Performance summary
                    if problem_type == 'classification':
                        accuracy = (predictions == y_test).mean()
                        st.success(f"🎯 Test Accuracy: {accuracy:.4f}")
                    else:
                        mae = np.mean(abs(predictions - y_test))
                        rmse = np.sqrt(np.mean((predictions - y_test)**2))
                        mape = np.mean(abs((predictions - y_test) / y_test)) * 100
                        st.success(f"📊 MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")
                        
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
        
        with pred_col2:
            # Single prediction interface
            st.write("**Single Prediction Interface**")
            single_prediction_interface(result.model, feature_cols, X_train, problem_type)

    def _render_advanced_visualizations(self, result, feature_cols, target_col, problem_type):
        """Advanced visualizations for ML results"""
        st.subheader("📊 Advanced Visualizations")
        
        # Feature importance
        if hasattr(result, 'feature_importance') and not result.feature_importance.empty:
            plot_feature_importance(result.feature_importance)
        
        # Predictions vs Actual
        try:
            from sklearn.model_selection import train_test_split
            X = self.state.dataset[feature_cols]
            y = self.state.dataset[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            predictions = result.model.predict(X_test)
            
            create_predictions_vs_actual_plot(y_test, predictions, problem_type)
            
        except Exception as e:
            st.error(f"Visualization failed: {str(e)}")

    def _render_model_analysis(self, result, feature_cols):
        """Advanced model analysis"""
        st.subheader("🔍 Model Analysis & Insights")
        
        # Model complexity
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Features Used", len(feature_cols))
            st.metric("Model Type", type(result.model).__name__)
        
        with col2:
            if hasattr(result, 'training_time'):
                st.metric("Training Time", f"{result.training_time:.2f}s")
            st.metric("Data Points", f"{len(self.state.dataset):,}")
        
        with col3:
            # Model performance assessment
            if 'r2' in result.metrics:
                r2 = result.metrics['r2']
                performance = "Excellent" if r2 > 0.8 else "Good" if r2 > 0.6 else "Poor"
                st.metric("Performance", performance)
            elif 'accuracy' in result.metrics:
                acc = result.metrics['accuracy']
                performance = "Excellent" if acc > 0.85 else "Good" if acc > 0.7 else "Poor"
                st.metric("Performance", performance)

    def _render_export_functionality(self, result, feature_cols, target_col):
        """Advanced export functionality"""
        st.subheader("📦 Advanced Export Options")
        
        # Prepare test data
        from sklearn.model_selection import train_test_split
        X = self.state.dataset[feature_cols]
        y = self.state.dataset[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        create_model_export_section(
            result.model, 
            result.metrics, 
            getattr(result, 'feature_importance', pd.DataFrame()),
            X_test, 
            y_test
        )

    def _render_detailed_model_info(self, result):
        """Detailed model information"""
        st.subheader("ℹ️ Detailed Model Information")
        
        with st.expander("🔍 Model Metadata", expanded=True):
            if hasattr(result, 'metadata'):
                st.json(result.metadata)
        
        with st.expander("📊 Model Parameters"):
            if hasattr(result.model, 'get_params'):
                st.json(result.model.get_params())
        
        with st.expander("🔧 Training Configuration"):
            config_info = {
                "Features": st.session_state.advanced_ml_feature_cols,
                "Target": st.session_state.advanced_ml_target_col,
                "Problem Type": st.session_state.advanced_ml_problem_type,
                "Training Time": getattr(result, 'training_time', 'Unknown'),
                "Model Class": type(result.model).__name__
            }
            st.json(config_info)

    def _display_ml_model_history(self):
        """Display ML model history and comparison"""
        st.divider()
        st.subheader("🏆 ML Model History & Comparison")
        
        # Save current model to history
        if hasattr(st.session_state, 'advanced_ml_result') and st.session_state.advanced_ml_result:
            if st.button("📝 Save Model to History", key="save_advanced_ml_history"):
                result = st.session_state.advanced_ml_result
                
                model_record = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_name': type(result.model).__name__,
                    'engine': getattr(result.metadata, 'engine', 'unknown') if hasattr(result, 'metadata') else 'unknown',
                    'problem_type': st.session_state.advanced_ml_problem_type,
                    'features': len(st.session_state.advanced_ml_feature_cols),
                    'training_time': getattr(result, 'training_time', 0),
                    **{k: v for k, v in result.metrics.items() if isinstance(v, (int, float))}
                }
                
                if 'advanced_ml_history' not in st.session_state:
                    st.session_state.advanced_ml_history = []
                
                st.session_state.advanced_ml_history.append(model_record)
                st.success("✅ Model saved to history!")
        
        # Display history
        if 'advanced_ml_history' in st.session_state and st.session_state.advanced_ml_history:
            history_df = pd.DataFrame(st.session_state.advanced_ml_history)
            
            # Show history table
            st.dataframe(history_df, use_container_width=True)
            
            # Comparison chart
            if len(history_df) > 1:
                metric_cols = [col for col in ['accuracy', 'r2', 'mae', 'rmse', 'f1_weighted'] 
                              if col in history_df.columns and history_df[col].notna().any()]
                
                if metric_cols and go:
                    selected_metric = st.selectbox(
                        "Choose metric for comparison", 
                        metric_cols, 
                        key="advanced_ml_metric_comparison"
                    )
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=history_df['model_name'],
                        y=history_df[selected_metric],
                        name=selected_metric.upper(),
                        marker_color='lightblue',
                        text=[f"{v:.4f}" for v in history_df[selected_metric]],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title=f"Model Comparison - {selected_metric.upper()}",
                        xaxis_title="Models",
                        yaxis_title=selected_metric.upper(),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("💡 Train models to build comparison history!")

    def _train_model_enhanced(self, df: pd.DataFrame, target: str, cv_folds: int, 
                            remove_outliers: bool, hyperopt_trials: int, compute_shap: bool,
                            test_size: float, random_state: int, ml_engine: str = "auto"):  # DODANE ml_engine
        """Ulepszony trening modelu z PyCaret support"""
        
        with st.status("Trening modelu w toku...", expanded=True) as status:
            try:
                # Step 1: Data preprocessing
                status.update(label="Przygotowywanie danych...")
                time.sleep(0.5)
                
                preprocessor = SmartDataPreprocessor()
                df_processed, prep_report = preprocessor.preprocess(df, target)
                
                st.write(f"Dane przygotowane: {prep_report.original_shape[0]:,} → {prep_report.final_shape[0]:,} wierszy")
                
                # Step 2: Model configuration  
                status.update(label="Konfiguracja modelu...")
                time.sleep(0.5)
                
                model_config = ModelConfig(
                    target=target,
                    engine=ml_engine,  # ZMIENIONE: używaj wybranego engine
                    cv_folds=cv_folds,
                    hyperopt_trials=hyperopt_trials,
                    outlier_detection=remove_outliers,
                    test_size=test_size,
                    random_state=random_state
                )
                
                st.write(f"Model skonfigurowany (engine: {ml_engine})")
            
            # Reszta funkcji bez zmian...
                
                # Step 3: Training
                status.update(label="🎯 Trenowanie modelu...")
                time.sleep(0.5)
                
                if USE_REAL_ML:
                    # Use real ML orchestrator
                    orchestrator = MLTrainingOrchestrator(model_config)
                    result = orchestrator.train(df_processed)
                    model = result.model
                    metrics = result.metrics
                    fi_df = result.feature_importance
                    metadata = result.metadata
                    metadata['preprocessing_info'] = result.preprocessing_info
                else:
                    # Use enhanced mock
                    model, metrics, fi_df, metadata = train_sklearn_enhanced(df_processed, model_config)
                
                st.write(f"✅ Model wytrenowany ({metadata.get('engine', 'unknown')})")
                
                # Step 4: Update application state
                status.update(label="💾 Zapisywanie wyników...")
                time.sleep(0.3)
                
                self.state.model = model
                self.state.metrics = metrics
                self.state.feature_importance = fi_df
                self.state.metadata = metadata
                self.state.preprocessing_info = prep_report.to_dict()
                self.state.training_completed = True
                
                # Step 5: Save to history
                self._save_experiment_enhanced(metrics, metadata)
                
                status.update(label="✅ Trening zakończony pomyślnie!", state="complete")
                st.balloons()
                
                # NAPRAWKA: Wymuś wyświetlenie wyników
                st.success("🎉 Model gotowy! Wyniki wyświetlone poniżej.")
                
                # NAPRAWKA: Nie używaj st.rerun() - powoduje resetowanie
                # st.rerun()  # Usuń tę linię jeśli istnieje
                
            except Exception as e:
                status.update(label="❌ Błąd podczas treningu", state="error")
                st.error(f"Szczegóły błędu: {str(e)}")
                logger.exception("Enhanced training failed")
    
    def _save_experiment_enhanced(self, metrics: Dict, metadata: Dict):
        """Enhanced experiment saving with better error handling"""
        try:
            # Determine problem type more accurately
            problem_type = ProblemType.REGRESSION
            if "accuracy" in metrics or "f1" in str(metrics).lower():
                problem_type = ProblemType.BINARY_CLASSIFICATION
            elif metadata.get('problem_type') == 'classification':
                problem_type = ProblemType.MULTICLASS_CLASSIFICATION
            
            # Create enhanced record
            run_id = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{metadata.get('engine', 'unknown')}"
            
            record = RunRecord(
                dataset=self.state.dataset_name,
                target=self.state.target_column,
                run_id=run_id,
                problem_type=problem_type,
                engine=metadata.get("engine", "auto"),
                status=RunStatus.COMPLETED,
                metrics=metrics,
                parameters=metadata.get("model_params", {}),
                notes=f"Eksperyment z interfejsu TMIV - {len(self.state.dataset):,} wierszy, {len(self.state.dataset.columns)} kolumn",
                duration_seconds=metadata.get("run_time", 0)
            )
            
            success = self.experiment_tracker.log_run(record)
            if success:
                st.toast(f"💾 Eksperyment {run_id} zapisany!", icon="✅")
            
        except Exception as e:
            st.warning(f"Nie udało się zapisać eksperymentu: {e}")
            logger.exception("Failed to save enhanced experiment")
    
    def _results_phase(self):
        """Ulepszona faza wyników"""
        st.markdown("## 📊 Wyniki treningu modelu")
        
        if not self.state.model:
            st.error("Brak wytrenowanego modelu!")
            return
        
        # Enhanced metrics display
        self._render_metrics_enhanced()
        
        # Feature importance
        self._render_feature_importance_enhanced()
        
        # Model insights
        self._render_model_insights()
        
        # Recommendations
        self._render_recommendations_enhanced()
        
        # Predictions
        self._render_predictions_enhanced()
        
        # Action buttons
        self._render_action_buttons()
    
    def _render_metrics_enhanced(self):
        """Enhanced metrics rendering"""
        st.subheader("📈 Metryki modelu")
        metrics = self.state.metrics
        
        if not metrics:
            st.warning("Brak metryk do wyświetlenia")
            return
        
        # Main metrics in columns with better formatting
        metric_items = list(metrics.items())
        n_cols = min(len(metric_items), 4)
        cols = st.columns(n_cols)
        
        for i, (name, value) in enumerate(metric_items):
            if isinstance(value, (int, float)) and not pd.isna(value):
                with cols[i % n_cols]:
                    # Format based on metric type
                    if name.lower() in ['r2', 'accuracy', 'f1_weighted', 'f1_macro']:
                        formatted_value = f"{value:.3f}"
                        if value > 0.8:
                            delta_color = "normal"
                        elif value > 0.6:
                            delta_color = "normal" 
                        else:
                            delta_color = "inverse"
                    else:
                        formatted_value = f"{value:.4f}"
                        delta_color = "normal"
                    
                    st.metric(
                        name.upper().replace('_', ' '),
                        formatted_value
                    )
        
        # Additional metrics info
        with st.expander("📊 Wszystkie metryki i szczegóły"):
            col1, col2 = st.columns(2)
            with col1:
                st.json(metrics)
            with col2:
                if self.state.metadata:
                    st.write("**Metadane modelu:**")
                    st.json(self.state.metadata)
    
    def _render_feature_importance_enhanced(self):
        """Enhanced feature importance rendering"""
        fi_df = self.state.feature_importance
        
        if fi_df.empty:
            st.info("Brak informacji o ważności cech")
            return
            
        st.subheader("🏆 Najważniejsze cechy w modelu")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            # Controls
            max_features = min(20, len(fi_df))
            n_features = st.slider("Liczba cech do wyświetlenia", 5, max_features, 10)
            show_percentage = st.checkbox("Pokaż jako procenty", value=True)
        
        with col1:
            # Enhanced visualization
            top_features = fi_df.head(n_features).copy()
            
            if show_percentage and 'importance' in top_features.columns:
                total_importance = top_features['importance'].sum()
                top_features['importance_pct'] = (top_features['importance'] / total_importance * 100).round(1)
            
            if px and not top_features.empty:
                y_col = 'importance_pct' if show_percentage else 'importance'
                title = f"Top {n_features} najważniejszych cech"
                
                fig = px.bar(
                    top_features,
                    x=y_col,
                    y='feature',
                    orientation='h',
                    title=title,
                    labels={y_col: 'Ważność (%' if show_percentage else 'Ważność)'}
                )
                fig.update_yaxes(categoryorder='total ascending')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced table
        display_df = top_features.copy()
        if show_percentage and 'importance_pct' in display_df.columns:
            display_df = display_df[['feature', 'importance_pct']].rename(columns={'importance_pct': 'Ważność (%)'})
        else:
            display_df = display_df[['feature', 'importance']].rename(columns={'importance': 'Ważność'})
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Insights
        if len(fi_df) > 0:
            top_3_pct = (fi_df.head(3)['importance'].sum() / fi_df['importance'].sum() * 100)
            st.info(f"💡 Top 3 cechy odpowiadają za {top_3_pct:.1f}% ważności modelu")
    
    def _render_model_insights(self):
        """Render model insights and interpretation"""
        st.subheader("🔍 Wgląd w model")
        
        metadata = self.state.metadata
        problem_type = metadata.get('problem_type', 'unknown')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Typ problemu", problem_type.title())
            st.metric("Silnik ML", metadata.get('engine', 'Unknown'))
        
        with col2:
            st.metric("Liczba cech", metadata.get('n_features', 'N/A'))
            st.metric("Czas treningu", f"{metadata.get('run_time', 0):.1f}s")
        
        with col3:
            # Performance assessment
            if 'r2' in self.state.metrics:
                r2 = self.state.metrics['r2']
                if r2 > 0.8:
                    st.success("🎯 Świetna wydajność")
                elif r2 > 0.6:
                    st.info("👍 Dobra wydajność")
                else:
                    st.warning("⚠️ Słaba wydajność")
            elif 'accuracy' in self.state.metrics:
                acc = self.state.metrics['accuracy']
                if acc > 0.85:
                    st.success("🎯 Świetna dokładność")
                elif acc > 0.7:
                    st.info("👍 Dobra dokładność")
                else:
                    st.warning("⚠️ Słaba dokładność")
        
        # Model complexity assessment
        if not self.state.feature_importance.empty:
            with st.expander("📊 Analiza złożoności modelu"):
                fi_df = self.state.feature_importance
                
                # Feature distribution analysis
                if len(fi_df) > 0:
                    # Calculate feature importance distribution
                    total_importance = fi_df['importance'].sum()
                    top_10_pct = fi_df.head(10)['importance'].sum() / total_importance * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Rozkład ważności cech:**")
                        st.write(f"• Top 10 cech: {top_10_pct:.1f}% całkowitej ważności")
                        st.write(f"• Pozostałe {len(fi_df)-10} cech: {100-top_10_pct:.1f}%")
                    
                    with col2:
                        # Complexity indicators
                        if top_10_pct > 80:
                            st.success("✅ Model skupia się na kilku kluczowych cechach")
                        elif top_10_pct > 60:
                            st.info("ℹ️ Model wykorzystuje umiarkowaną liczbę cech")
                        else:
                            st.warning("⚠️ Model może być zbyt złożony")
    
    def _render_recommendations_enhanced(self):
        """Enhanced AI/rule-based recommendations"""
        st.subheader("💡 Rekomendacje i wnioski")
        
        openai_key = get_openai_key()
        
        if USE_REAL_UTILS and openai_key and not self.state.feature_importance.empty:
            # AI-powered recommendations
            try:
                detector = SmartTargetDetector()
                analysis = detector.analyze_target(self.state.dataset, self.state.target_column)
                
                engine = MLRecommendationEngine()
                recommendations = engine.generate_recommendations(
                    analysis=analysis,
                    top_features=self.state.feature_importance.head(5)['feature'].tolist(),
                    dataset_size=len(self.state.dataset)
                )
                
                st.markdown(recommendations)
                
            except Exception as e:
                st.error(f"Błąd generowania rekomendacji AI: {e}")
                self._render_rule_based_recommendations_enhanced()
        else:
            # Enhanced rule-based recommendations
            self._render_rule_based_recommendations_enhanced()
    
    def _render_rule_based_recommendations_enhanced(self):
        """Enhanced rule-based recommendations"""
        recommendations = []
        metrics = self.state.metrics
        fi_df = self.state.feature_importance
        
        st.write("**🔍 Automatyczne rekomendacje na podstawie wyników:**")
        
        # Performance-based recommendations
        if 'r2' in metrics:
            r2 = metrics['r2']
            if r2 < 0.6:
                recommendations.append("📈 **Niska wydajność (R² < 0.6)**: Rozważ dodanie nowych cech, transformację danych lub wypróbowanie innych algorytmów")
            elif r2 > 0.95:
                recommendations.append("⚠️ **Bardzo wysoka wydajność (R² > 0.95)**: Sprawdź czy nie ma overfittingu. Przetestuj na nowych danych")
            else:
                recommendations.append("✅ **Dobra wydajność modelu**: Model wykazuje satysfakcjonującą zdolność przewidywania")
        
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            if acc < 0.8:
                recommendations.append("📊 **Niska dokładność**: Sprawdź balans klas, jakość cech i rozważ inne algorytmy")
            else:
                recommendations.append("✅ **Dobra dokładność klasyfikacji**: Model skutecznie rozpoznaje klasy")
        
        # Feature importance recommendations
        if not fi_df.empty and len(fi_df) > 0:
            top_3_importance = fi_df.head(3)['importance'].sum() / fi_df['importance'].sum()
            if top_3_importance > 0.8:
                recommendations.append(f"🎯 **Koncentracja cech**: Top 3 cechy mają {top_3_importance:.1%} wpływu. Skup się na ich jakości")
            
            # Suggest feature engineering
            top_features = fi_df.head(3)['feature'].tolist()
            recommendations.append(f"🔧 **Najważniejsze cechy**: {', '.join(top_features)}. Rozważ tworzenie nowych cech na ich podstawie")
        
        # Data quality recommendations  
        if self.state.dataset is not None:
            n_rows, n_cols = self.state.dataset.shape
            if n_rows < 1000:
                recommendations.append("📊 **Mały zbiór danych**: Rozważ zbieranie większej ilości danych lub użycie prostszych modeli")
            if n_cols > 50:
                recommendations.append("🔍 **Dużo cech**: Rozważ selekcję cech lub redukcję wymiarowości")
        
        # General recommendations
        recommendations.extend([
            "🔬 **Walidacja**: Przetestuj model na nowych, nieznanych danych",
            "📈 **Monitorowanie**: Śledź wydajność modelu w czasie rzeczywistym",
            "🔄 **Iteracja**: Regularnie aktualizuj model nowymi danymi"
        ])
        
        for rec in recommendations:
            st.markdown(rec)
            
        # Action items
        st.write("**🎯 Następne kroki:**")
        st.markdown("""
        1. **Sprawdź wyniki na danych testowych**
        2. **Przeanalizuj błędnie sklasyfikowane przypadki**
        3. **Rozważ ensemble methods dla lepszych wyników**
        4. **Dokumentuj parametry najlepszego modelu**
        """)
    
    def _render_predictions_enhanced(self):
        """Enhanced predictions section"""
        if not self.state.model or self.state.dataset is None:
            return
        
        st.subheader("🔮 Predykcje modelu")
        
        with st.expander("📊 Predykcje na próbce danych", expanded=False):
            # Enhanced sample selection
            sample_size = st.selectbox("Rozmiar próbki:", [5, 10, 20, 50], index=1)
            sample_method = st.radio("Metoda próbkowania:", ["Pierwsze wiersze", "Losowe"], horizontal=True)
            
            if sample_method == "Losowe":
                if st.button("🎲 Nowa losowa próbka"):
                    st.rerun()
                sample_data = self.state.dataset.sample(n=min(sample_size, len(self.state.dataset))).drop(
                    columns=[self.state.target_column], errors='ignore'
                )
                actual_values = self.state.dataset.sample(n=min(sample_size, len(self.state.dataset)))[self.state.target_column]
            else:
                sample_data = self.state.dataset.head(sample_size).drop(
                    columns=[self.state.target_column], errors='ignore'
                )
                actual_values = self.state.dataset.head(sample_size)[self.state.target_column]
            
            try:
                # Make predictions
                predictions = self.state.model.predict(sample_data)
                
                # Create results dataframe
                results_df = sample_data.copy()
                results_df['Prawdziwa wartość'] = actual_values.values
                results_df['Predykcja'] = predictions
                
                # Calculate errors for regression
                if 'r2' in self.state.metrics:
                    results_df['Błąd'] = results_df['Prawdziwa wartość'] - results_df['Predykcja']
                    results_df['Błąd %'] = (results_df['Błąd'] / results_df['Prawdziwa wartość'] * 100).round(2)
                
                # Display results
                st.dataframe(results_df, use_container_width=True)
                
                # Prediction quality assessment
                if len(predictions) > 0:
                    if 'r2' in self.state.metrics:  # Regression
                        from sklearn.metrics import mean_absolute_error, mean_squared_error
                        mae = mean_absolute_error(actual_values, predictions)
                        rmse = mean_squared_error(actual_values, predictions, squared=False)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("MAE na próbce", f"{mae:.3f}")
                        with col2:
                            st.metric("RMSE na próbce", f"{rmse:.3f}")
                    else:  # Classification
                        from sklearn.metrics import accuracy_score
                        acc = accuracy_score(actual_values, predictions)
                        st.metric("Dokładność na próbce", f"{acc:.3f}")
                
            except Exception as e:
                st.error(f"Błąd podczas generowania predykcji: {e}")
                st.info("Model może wymagać preprocessingu danych zgodnego z treningiem")
    
    def _render_action_buttons(self):
        """Enhanced action buttons"""
        st.markdown("---")
        st.subheader("🎛️ Akcje")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Nowy eksperyment", use_container_width=True, help="Wyczyść obecne wyniki i rozpocznij od nowa"):
                self._reset_state()
        
        with col2:
            if st.button("📥 Pobierz wyniki", use_container_width=True, help="Zapisz wyniki w formacie JSON"):
                self._export_results_enhanced()
        
        with col3:
            if st.button("📊 Porównaj z historią", use_container_width=True, help="Zobacz jak obecny model wypada na tle poprzednich"):
                self._show_model_comparison()
                
        with col4:
            if st.button("🚀 Optymalizacja", use_container_width=True, help="Uruchom zaawansowaną optymalizację hiperparametrów"):
                if USE_REAL_ML:
                    st.info("Funkcja optymalizacji dostępna w pełnej wersji")
                else:
                    st.info("Zaawansowana optymalizacja wymaga modułów ML")
    
    def _show_model_comparison(self):
        """Show comparison with historical models"""
        st.subheader("📊 Porównanie z poprzednimi modelami")
        
        history = self.experiment_tracker.get_history()
        
        if history.empty or len(history) < 2:
            st.info("Potrzeba co najmniej 2 eksperymentów do porównania")
            return
        
        # Filter for same target and dataset
        relevant_history = history[
            (history['target'] == self.state.target_column) & 
            (history['dataset'] == self.state.dataset_name)
        ] if 'target' in history.columns and 'dataset' in history.columns else history
        
        if len(relevant_history) < 2:
            st.info("Brak porównywalnych eksperymentów (ten sam dataset i target)")
            return
        
        st.write(f"Znaleziono {len(relevant_history)} porównywalnych eksperymentów:")
        
        # Show comparison table
        comparison_df = relevant_history[['run_id', 'engine', 'created_at']].copy() if all(col in relevant_history.columns for col in ['run_id', 'engine', 'created_at']) else relevant_history
        st.dataframe(comparison_df, use_container_width=True)
    
    def _render_history_section(self):
        """Enhanced history section"""
        st.markdown("## 📚 Historia eksperymentów")
        
        try:
            history = self.experiment_tracker.get_history()
            
            if history.empty:
                st.info("🆕 Brak historii eksperymentów. Przeprowadź pierwszy eksperyment aby zobaczyć historię!")
                return
            
            # Enhanced statistics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                total_runs = len(history)
                st.metric("Łączne eksperymenty", total_runs)
            
            with col2:
                completed = len(history[history['status'] == 'completed']) if 'status' in history.columns else len(history)
                st.metric("Ukończone", completed)
            
            with col3:
                unique_datasets = history['dataset'].nunique() if 'dataset' in history.columns else 1
                st.metric("Unikalne datasety", unique_datasets)
            
            with col4:
                unique_targets = history['target'].nunique() if 'target' in history.columns else 1
                st.metric("Unikalne targety", unique_targets)
                
            with col5:
                if 'created_at' in history.columns:
                    try:
                        latest = pd.to_datetime(history['created_at']).max()
                        st.metric("Ostatni eksperyment", latest.strftime("%m/%d %H:%M"))
                    except:
                        st.metric("Ostatni", "Dziś")
                else:
                    st.metric("Ostatni", "Dziś")
            
            # Enhanced history table
            with st.expander("📋 Szczegóły eksperymentów", expanded=True):
                # Column selection
                available_columns = history.columns.tolist()
                display_columns = ['run_id', 'dataset', 'target', 'engine', 'status', 'created_at']
                selected_columns = [col for col in display_columns if col in available_columns]
                
                if not selected_columns:
                    selected_columns = available_columns[:5]  # Show first 5 columns
                
                # Sorting options
                col1, col2 = st.columns([3, 1])
                with col2:
                    sort_by = st.selectbox("Sortuj według:", selected_columns, 
                                         index=selected_columns.index('created_at') if 'created_at' in selected_columns else 0)
                    sort_desc = st.checkbox("Malejąco", value=True)
                
                # Display sorted data
                display_df = history[selected_columns].copy()
                if sort_by in display_df.columns:
                    display_df = display_df.sort_values(sort_by, ascending=not sort_desc)
                
                st.dataframe(display_df.head(20), use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Błąd podczas ładowania historii: {e}")
            logger.exception("Failed to render enhanced history section")

    def _render_sidebar_tools(self):
        """Enhanced sidebar tools"""
        with st.sidebar:
            st.header("🛠 Narzędzia")
            
            # DODAJ: Debug info
            with st.expander("🔍 Debug Info"):
                st.write(f"Training completed: {self.state.training_completed}")
                st.write(f"Model exists: {self.state.model is not None}")
                st.write(f"Metrics count: {len(self.state.metrics)}")
                st.write(f"Feature importance: {len(self.state.feature_importance)} rows")
                
                if 'experiment_history' in st.session_state:
                    st.write(f"History: {len(st.session_state.experiment_history)} experiments")
     
        
            # Export section
            st.subheader("💾 Eksport")
            if self.state.training_completed:
                if st.button("📥 Eksportuj wyniki", use_container_width=True):
                    self._export_results_enhanced()
                    
                if st.button("📊 Eksportuj model", use_container_width=True):
                    st.info("Funkcja eksportu modeli będzie dostępna wkrótce")
            else:
                st.info("Wytrenuj model aby udostępnić eksport")
            
            # History management
            st.subheader("📚 Historia")
            
            try:
                stats = self.experiment_tracker.get_statistics()
                total_runs = stats.get('total_runs', 0)
                st.metric("Wszystkie eksperymenty", total_runs)
                
                if total_runs > 0:
                    if st.button("📊 Export historii CSV", use_container_width=True):
                        try:
                            if USE_REAL_DB:
                                csv_data = self.experiment_tracker.export_to_csv()
                                st.download_button(
                                    label="📁 Pobierz historię CSV",
                                    data=csv_data,
                                    file_name=f"tmiv_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            else:
                                # Fallback for session-based history
                                history = self.experiment_tracker.get_history()
                                if not history.empty:
                                    csv_data = history.to_csv(index=False)
                                    st.download_button(
                                        label="📁 Pobierz historię CSV",
                                        data=csv_data,
                                        file_name=f"tmiv_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                        except Exception as e:
                            st.error(f"Błąd eksportu: {e}")
                            
            except Exception:
                st.metric("Eksperymenty", "❌")
            
            # Advanced tools
            with st.expander("⚙️ Zarządzanie"):
                if st.button("🗑 Wyczyść historię", type="secondary", use_container_width=True):
                    if st.checkbox("Potwierdź usunięcie wszystkich eksperymentów"):
                        self.experiment_tracker.clear_history(confirm=True)
                        st.success("Historia wyczyszczona!")
                        st.rerun()
                
                if st.button("💾 Backup bazy", type="secondary", use_container_width=True):
                    backup_path = self.experiment_tracker.backup_database()
                    if backup_path:
                        st.success(f"Backup: {backup_path}")
                        
            # System info
            st.subheader("ℹ️ System")
            st.write(f"**Wersja:** {self.settings.app_name}")
            if hasattr(self.settings, 'version'):
                st.write(f"**Build:** {self.settings.version}")
            st.write(f"**Moduły:** {len(missing_modules)} mock")
            
            # Feature flags
            if hasattr(self.settings, 'get_feature_flag'):
                with st.expander("🚩 Funkcje"):
                    st.write("AI Analiza:", "✅" if self.ui_config.enable_llm else "❌")
                    st.write("Real ML:", "✅" if USE_REAL_ML else "❌")
                    st.write("Advanced EDA:", "✅" if USE_REAL_EDA else "❌")

    def _reset_state(self):
        """Enhanced state reset"""
        # Clear training state
        self.state.model = None
        self.state.metrics = {}
        self.state.feature_importance = pd.DataFrame()
        self.state.metadata = {}
        self.state.preprocessing_info = {}
        self.state.training_completed = False
        
        # Clear session state for analysis
        if "eda_analysis_type" in st.session_state:
            st.session_state.eda_analysis_type = "Podsumowanie statystyczne"
        
        st.success("🔄 Stan aplikacji został zresetowany - możesz rozpocząć nowy eksperyment!")
        time.sleep(1)
        st.rerun()

    def _export_results_enhanced(self):
        """Enhanced results export"""
        if not self.state.training_completed:
            st.warning("Brak wyników do eksportu")
            return
        
        try:
            # Prepare comprehensive export data
            export_data = {
                "experiment_info": {
                    "export_timestamp": pd.Timestamp.now().isoformat(),
                    "dataset_name": self.state.dataset_name,
                    "target_column": self.state.target_column,
                    "dataset_shape": list(self.state.dataset.shape) if self.state.dataset is not None else None,
                },
                "model_results": {
                    "metrics": self.state.metrics,
                    "metadata": self.state.metadata,
                    "feature_importance": self.state.feature_importance.to_dict('records') if not self.state.feature_importance.empty else [],
                    "preprocessing_info": self.state.preprocessing_info
                },
                "system_info": {
                    "app_version": getattr(self.settings, 'version', '2.0.0'),
                    "modules_used": {
                        "real_ml": USE_REAL_ML,
                        "real_eda": USE_REAL_EDA,
                        "real_utils": USE_REAL_UTILS,
                        "real_db": USE_REAL_DB
                    },
                    "missing_modules": missing_modules
                }
            }
            
            # Create download data
            json_data = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
            
            # Enhanced download button
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f"tmiv_results_{self.state.dataset_name}_{self.state.target_column}_{timestamp}.json"
            
            st.download_button(
                label="📁 Pobierz kompletne wyniki (JSON)",
                data=json_data,
                file_name=filename,
                mime="application/json",
                use_container_width=True
            )
            
            st.success("✅ Wyniki gotowe do pobrania!")
            
            # Show export preview
            with st.expander("👀 Podgląd eksportowanych danych"):
                st.json(export_data)
            
        except Exception as e:
            st.error(f"Błąd eksportu: {e}")
            logger.exception("Failed to export enhanced results")

# ===== DODAJ TE FUNKCJE PRZED def main(): W app.py =====

def display_model_metrics(metrics_dict, model_type: str):
    """Display model performance metrics in a nice layout"""
    st.subheader("📊 Model Performance Metrics")
    
    if model_type == "classification":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'accuracy' in metrics_dict:
                st.metric("Accuracy", f"{metrics_dict['accuracy']:.4f}")
        with col2:
            if 'f1_weighted' in metrics_dict:
                st.metric("F1-Score", f"{metrics_dict['f1_weighted']:.4f}")
        with col3:
            if 'roc_auc' in metrics_dict:
                st.metric("ROC AUC", f"{metrics_dict['roc_auc']:.4f}")
        with col4:
            if 'balanced_accuracy' in metrics_dict:
                st.metric("Balanced Accuracy", f"{metrics_dict['balanced_accuracy']:.4f}")
                
        # Second row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'precision' in metrics_dict:
                st.metric("Precision", f"{metrics_dict.get('precision', 0):.4f}")
        with col2:
            if 'recall' in metrics_dict:
                st.metric("Recall", f"{metrics_dict.get('recall', 0):.4f}")
        with col3:
            if 'mcc' in metrics_dict:
                st.metric("Matthews Corr Coef", f"{metrics_dict.get('mcc', 0):.4f}")
                
    else:  # regression
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'r2' in metrics_dict:
                st.metric("R² Score", f"{metrics_dict['r2']:.4f}")
        with col2:
            if 'mae' in metrics_dict:
                st.metric("MAE", f"{metrics_dict['mae']:.4f}")
        with col3:
            if 'rmse' in metrics_dict:
                st.metric("RMSE", f"{metrics_dict['rmse']:.4f}")

def plot_feature_importance(feature_importance_df: pd.DataFrame, top_n: int = 20):
    """Create interactive feature importance plot"""
    if feature_importance_df.empty:
        st.warning("Feature importance not available for this model")
        return
    
    st.subheader("🎯 Feature Importance")
    
    # Take top N features
    top_features = feature_importance_df.head(top_n)
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker_color='lightblue',
        text=[f'{v:.4f}' for v in top_features['importance']],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f"Top {len(top_features)} Most Important Features",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=max(400, len(top_features) * 25),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_model_export_section(model, metrics_dict, feature_importance_df, X_test=None, y_test=None):
    """Advanced model export functionality"""
    st.subheader("📦 Export & Download")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Model pickle export
        if st.button("💾 Download Model") and model is not None:
            import joblib
            
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            
            st.download_button(
                label="📥 Download .joblib",
                data=buffer.getvalue(),
                file_name=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib",
                mime="application/octet-stream"
            )
    
    with col2:
        # Model report export
        if st.button("📊 Download Report"):
            report = f"""
MODEL TRAINING REPORT
====================

Model Information:
- Type: {type(model).__name__ if model else 'Unknown'}
- Training Status: {'✅ Trained' if model else '❌ Not Trained'}

Performance Metrics:
{chr(10).join([f"- {k}: {v:.4f}" for k, v in metrics_dict.items() if isinstance(v, (int, float))])}

Feature Importance:
{feature_importance_df.head(10).to_string() if not feature_importance_df.empty else 'Not available'}
"""
            
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col3:
        # Predictions export
        if st.button("🎯 Download Predictions") and X_test is not None and model is not None:
            try:
                predictions = model.predict(X_test)
                
                # Create predictions DataFrame
                pred_df = X_test.copy()
                pred_df['predictions'] = predictions
                if y_test is not None:
                    pred_df['actual'] = y_test
                
                csv_buffer = io.StringIO()
                pred_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Prediction export failed: {str(e)}")

def create_predictions_vs_actual_plot(y_test, y_pred, model_type):
    """Create predictions vs actual plot"""
    if model_type == 'regression':
        # Scatter plot for regression
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Regression: Predicted vs Actual Values",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # classification
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        st.pyplot(fig)

def single_prediction_interface(model, feature_columns, X_train, model_type):
    """Interface for single instance prediction"""
    st.write("Enter values for prediction:")
    
    input_data = {}
    
    # Limit to first 10 features for UI manageability
    display_features = feature_columns[:10]
    
    for feature in display_features:
        if feature in X_train.select_dtypes(include=[np.number]).columns:
            # Numeric input
            min_val = float(X_train[feature].min())
            max_val = float(X_train[feature].max())
            mean_val = float(X_train[feature].mean())
            
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                key=f"single_pred_{feature}"
            )
        else:
            # Categorical input
            unique_values = X_train[feature].unique()
            input_data[feature] = st.selectbox(
                f"{feature}",
                options=unique_values,
                key=f"single_pred_{feature}"
            )
    
    if st.button("🎯 Make Prediction"):
        try:
            # Create input dataframe with all features
            input_df = pd.DataFrame([input_data])
            
            # Add missing features with default values
            for col in feature_columns:
                if col not in input_df.columns:
                    if col in X_train.select_dtypes(include=[np.number]).columns:
                        input_df[col] = X_train[col].mean()
                    else:
                        mode_val = X_train[col].mode()
                        input_df[col] = mode_val[0] if not mode_val.empty else "MISSING"
            
            # Reorder columns to match training data
            input_df = input_df[feature_columns]
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            st.success(f"Prediction: **{prediction}**")
            
            # Show probability if classification
            if model_type == 'classification' and hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(input_df)
                    if proba is not None:
                        proba_dict = {f"Class_{i}": prob for i, prob in enumerate(proba[0])}
                        st.write("**Prediction Probabilities:**")
                        st.json(proba_dict)
                except:
                    pass
                    
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

def main():
    """Enhanced main function with better error handling"""
    try:
        # Initialize and run application
        app = TMIVApplication()
        app.run()
        
    except Exception as e:
        st.error("🚨 Krytyczny błąd aplikacji")
        
        with st.expander("🔍 Szczegóły błędu", expanded=False):
            st.code(str(e))
            
        st.info("""
        **Możliwe rozwiązania:**
        1. Odśwież stronę (F5)
        2. Wyczyść cache przeglądarki
        3. Sprawdź czy wszystkie wymagane pliki są dostępne
        4. Skontaktuj się z administratorem
        """)
        
        # Emergency fallback
        st.markdown("---")
        st.subheader("🆘 Tryb awaryjny")
        if st.button("🔄 Spróbuj ponownie", type="primary"):
            st.rerun()

if __name__ == "__main__":
    main()