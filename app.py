"""
TMIV - The Most Important Variables
Ulepszona aplikacja AutoML z czystą architekturą
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

# Konfiguracja strony - musi być pierwsza
st.set_page_config(
    page_title="TMIV - The Most Important Variables",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importy modułów aplikacji
try:
    from config.settings import get_settings
    from frontend.ui_components import TMIVApp, DataConfig, UIConfig
    from backend.ml_integration import train_sklearn_enhanced, ModelConfig
    from backend.eda_integration import SmartDataPreprocessor, AdvancedColumnAnalyzer
    from backend.utils import SmartTargetDetector, MLRecommendationEngine
    from db.db_utils import MLExperimentTracker, RunRecord, ProblemType, RunStatus
except ImportError as e:
    st.error(f"Błąd importu modułów: {e}")
    st.stop()

# Logger
logger = logging.getLogger(__name__)

# CSS dla lepszego wyglądu
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

def init_app_state() -> AppState:
    """Inicjalizuje stan aplikacji"""
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()
    return st.session_state.app_state

def get_openai_key() -> str:
    """Pobiera klucz OpenAI z różnych źródeł"""
    import os
    import re
    
    # Sprawdź session_state, secrets, env
    sources = [
        st.session_state.get("openai_key", ""),
        getattr(st.secrets, "OPENAI_API_KEY", ""),
        os.getenv("OPENAI_API_KEY", "")
    ]
    
    for key in sources:
        if key and re.match(r"^sk-[a-zA-Z0-9]{20,}$", key):
            return key
    return ""

class TMIVApplication:
    """Główna klasa aplikacji TMIV"""
    
    def __init__(self):
        self.settings = get_settings()
        self.state = init_app_state()
        self.experiment_tracker = MLExperimentTracker()
        self._setup_configs()
        
    def _setup_configs(self):
        """Konfiguruje komponenty aplikacji"""
        self.data_config = DataConfig(
            max_file_size_mb=self.settings.data.max_file_size_mb,
            supported_formats=self.settings.data.supported_formats,
            auto_detect_encoding=True,
            max_preview_rows=50
        )
        
        self.ui_config = UIConfig(
            app_title=self.settings.app_name,
            app_subtitle="AutoML • EDA • Historia eksperymentów • Jeden eksport ZIP",
            enable_llm=bool(get_openai_key()),
            show_advanced_options=self.settings.get_feature_flag("advanced_preprocessing")
        )
        
        self.tmiv_app = TMIVApp(self.data_config, self.ui_config)
        
    def run(self):
        """Główny punkt wejścia aplikacji"""
        self._render_header()
        self._render_openai_status()
        
        # Główny przepływ
        if not self.state.training_completed:
            self._data_loading_phase()
        else:
            self._results_phase()
        
        # Sekcje dostępne zawsze
        self._render_history_section()
        self._render_sidebar_tools()
        
    def _render_header(self):
        """Renderuje nagłówek aplikacji"""
        st.title(self.ui_config.app_title)
        st.caption(self.ui_config.app_subtitle)
        
        # Status aplikacji
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if self.state.dataset is not None:
                st.success(f"📊 Dane: {self.state.dataset_name} ({len(self.state.dataset)} wierszy)")
            else:
                st.info("📁 Wczytaj dane aby rozpocząć")
                
        with col2:
            if self.state.training_completed:
                st.success("🎯 Model gotowy")
            else:
                st.info("⏳ Oczekuje na trening")
                
        with col3:
            experiments_count = len(self.experiment_tracker.get_history() or [])
            st.metric("Eksperymenty", experiments_count)
    
    def _render_openai_status(self):
        """Status klucza OpenAI"""
        openai_key = get_openai_key()
        
        if openai_key:
            st.success("🤖 OpenAI aktywny - opisy kolumn i rekomendacje dostępne")
        else:
            with st.expander("🔑 Konfiguracja OpenAI (opcjonalne)"):
                key_input = st.text_input(
                    "Klucz OpenAI API",
                    type="password",
                    placeholder="sk-...",
                    help="Dla opisów kolumn i rekomendacji AI"
                )
                if key_input:
                    st.session_state["openai_key"] = key_input
                    st.rerun()
    
    def _data_loading_phase(self):
        """Faza wczytywania i konfiguracji danych"""
        st.markdown("## 📊 Przygotowanie danych")
        
        # Wczytywanie danych
        df, dataset_name, target = self.tmiv_app.render_data_selection()
        
        if df is None or df.empty:
            st.info("👆 Wczytaj dane aby kontynuować")
            return
            
        # Aktualizuj stan
        self.state.dataset = df
        self.state.dataset_name = dataset_name
        self.state.target_column = target or ""
        
        # EDA sekcja
        self._render_eda_section(df)
        
        # Trening
        if target:
            self._render_training_section(df, target)
    
    def _render_eda_section(self, df: pd.DataFrame):
        """Sekcja analizy eksploracyjnej"""
        st.markdown("## 🔬 Analiza eksploracyjna")
        
        # Szybkie podsumowanie
        with st.expander("📋 Podsumowanie danych", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
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
        
        # Wybór typu analizy
        analysis_type = st.selectbox(
            "Rodzaj analizy",
            [
                "Podsumowanie statystyczne",
                "Rozkłady zmiennych", 
                "Korelacje",
                "Analiza targetu",
                "Jakość danych"
            ]
        )
        
        if analysis_type == "Podsumowanie statystyczne":
            st.dataframe(df.describe(include='all'), use_container_width=True)
            
        elif analysis_type == "Rozkłady zmiennych":
            self._render_distributions(df)
            
        elif analysis_type == "Korelacje":
            self._render_correlations(df)
            
        elif analysis_type == "Analiza targetu":
            self._render_target_analysis(df)
            
        elif analysis_type == "Jakość danych":
            self._render_data_quality(df)
    
    def _render_distributions(self, df: pd.DataFrame):
        """Renderuje rozkłady zmiennych"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.info("Brak zmiennych numerycznych")
            return
            
        selected_cols = st.multiselect(
            "Wybierz kolumny", 
            numeric_cols, 
            default=numeric_cols[:3]
        )
        
        if selected_cols:
            for col in selected_cols:
                st.subheader(f"Rozkład: {col}")
                fig = px.histogram(df, x=col, marginal="box")
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_correlations(self, df: pd.DataFrame):
        """Renderuje analizę korelacji"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            st.info("Potrzeba co najmniej 2 zmiennych numerycznych")
            return
            
        corr_matrix = numeric_df.corr()
        
        # Heatmapa
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Macierz korelacji"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Silne korelacje
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corr.append({
                        'Zmienna 1': corr_matrix.columns[i],
                        'Zmienna 2': corr_matrix.columns[j],
                        'Korelacja': round(corr_val, 3)
                    })
        
        if strong_corr:
            st.subheader("Silne korelacje (|r| > 0.7)")
            st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
    
    def _render_target_analysis(self, df: pd.DataFrame):
        """Analiza zmiennej docelowej"""
        if not self.state.target_column:
            st.info("Wybierz zmienną docelową")
            return
            
        target_col = self.state.target_column
        detector = SmartTargetDetector()
        
        try:
            analysis = detector.analyze_target(df, target_col)
            
            # Podstawowe informacje
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Typ problemu", analysis.problem_type.value)
            with col2:
                st.metric("Unikalne wartości", analysis.unique_values)
            with col3:
                st.metric("Braki danych", f"{analysis.missing_ratio:.1%}")
            
            # Wizualizacja rozkładu
            if analysis.problem_type.value == "regression":
                fig = px.histogram(df, x=target_col, marginal="box")
                st.plotly_chart(fig, use_container_width=True)
            else:
                value_counts = df[target_col].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values)
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Błąd analizy targetu: {e}")
    
    def _render_data_quality(self, df: pd.DataFrame):
        """Analiza jakości danych"""
        quality_issues = []
        
        # Sprawdzenie braków danych
        missing = df.isnull().sum()
        for col, miss_count in missing.items():
            if miss_count > 0:
                miss_pct = (miss_count / len(df)) * 100
                quality_issues.append({
                    'Kolumna': col,
                    'Problem': 'Braki danych',
                    'Wartość': f"{miss_pct:.1f}% ({miss_count} wierszy)"
                })
        
        # Sprawdzenie kolumn stałych
        for col in df.columns:
            if df[col].nunique() <= 1:
                quality_issues.append({
                    'Kolumna': col,
                    'Problem': 'Kolumna stała',
                    'Wartość': f"{df[col].nunique()} unikatowych wartości"
                })
        
        # Sprawdzenie wysokiej kardynalności
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.8:
                quality_issues.append({
                    'Kolumna': col,
                    'Problem': 'Wysoka kardynalność',
                    'Wartość': f"{df[col].nunique()} unikalnych z {len(df)}"
                })
        
        if quality_issues:
            st.dataframe(pd.DataFrame(quality_issues), use_container_width=True)
        else:
            st.success("Nie wykryto problemów z jakością danych")
    
    def _render_training_section(self, df: pd.DataFrame, target: str):
        """Sekcja treningu modelu"""
        st.markdown("## 🚀 Trening modelu")
        
        # Konfiguracja treningu
        with st.expander("⚙️ Konfiguracja treningu", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                cv_folds = st.slider("Cross-validation (folds)", 0, 10, 5)
                remove_outliers = st.checkbox("Usuń outliery", value=True)
                
            with col2:
                hyperopt_trials = st.slider("Optymalizacja hiperparametrów", 10, 200, 50)
                compute_shap = st.checkbox("Oblicz SHAP values", value=len(df) < 1000)
        
        # Przycisk treningu
        if st.button("🎯 Rozpocznij trening", type="primary", use_container_width=True):
            self._train_model(df, target, cv_folds, remove_outliers, hyperopt_trials, compute_shap)
    
    def _train_model(self, df: pd.DataFrame, target: str, cv_folds: int, 
                    remove_outliers: bool, hyperopt_trials: int, compute_shap: bool):
        """Wykonuje trening modelu"""
        
        with st.status("🔄 Trening w toku...", expanded=True) as status:
            try:
                # Preprocessing
                st.write("📋 Przygotowanie danych...")
                preprocessor = SmartDataPreprocessor()
                df_processed, prep_report = preprocessor.fit_transform(df, target)
                
                # Konfiguracja modelu
                st.write("⚙️ Konfiguracja modelu...")
                model_config = ModelConfig(
                    target=target,
                    cv_folds=cv_folds,
                    hyperopt_trials=hyperopt_trials,
                    outlier_detection=remove_outliers
                )
                
                # Trening
                st.write("🎯 Trening modelu...")
                model, metrics, fi_df, metadata = train_sklearn_enhanced(df_processed, model_config)
                
                # Aktualizacja stanu
                self.state.model = model
                self.state.metrics = metrics
                self.state.feature_importance = fi_df
                self.state.metadata = metadata
                self.state.preprocessing_info = prep_report.to_dict()
                self.state.training_completed = True
                
                # Zapis do historii
                st.write("💾 Zapisywanie eksperymentu...")
                self._save_experiment(metrics, metadata)
                
                status.update(label="✅ Trening zakończony pomyślnie", state="complete")
                st.balloons()
                
            except Exception as e:
                status.update(label="❌ Błąd treningu", state="error")
                st.error(f"Błąd podczas treningu: {e}")
                logger.exception("Training failed")
    
    def _save_experiment(self, metrics: Dict, metadata: Dict):
        """Zapisuje eksperyment do bazy"""
        try:
            # Określ typ problemu
            problem_type = ProblemType.REGRESSION
            if "accuracy" in metrics or "f1" in str(metrics).lower():
                problem_type = ProblemType.BINARY_CLASSIFICATION
            
            # Utwórz rekord
            record = RunRecord(
                dataset=self.state.dataset_name,
                target=self.state.target_column,
                run_id=metadata.get("run_id", f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"),
                problem_type=problem_type,
                engine=metadata.get("engine", "auto"),
                status=RunStatus.COMPLETED,
                metrics=metrics,
                parameters=metadata.get("model_params", {}),
                notes="Eksperyment z interfejsu TMIV",
                duration_seconds=metadata.get("run_time", 0)
            )
            
            success = self.experiment_tracker.log_run(record)
            if success:
                st.toast("💾 Eksperyment zapisany w historii", icon="✅")
            
        except Exception as e:
            st.warning(f"Nie udało się zapisać eksperymentu: {e}")
            logger.exception("Failed to save experiment")
    
    def _results_phase(self):
        """Faza wyników po treningu"""
        st.markdown("## 📊 Wyniki treningu")
        
        # Metryki
        self._render_metrics()
        
        # Feature importance
        self._render_feature_importance()
        
        # Rekomendacje
        self._render_recommendations()
        
        # Predykcje
        self._render_predictions()
        
        # Reset przyciski
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Nowy eksperyment", use_container_width=True):
                self._reset_state()
        with col2:
            if st.button("📥 Pobierz wyniki", use_container_width=True):
                self._export_results()
    
    def _render_metrics(self):
        """Renderuje metryki modelu"""
        metrics = self.state.metrics
        
        if not metrics:
            return
            
        # Główne metryki w kolumnach
        metric_cols = st.columns(len(metrics))
        
        for i, (name, value) in enumerate(metrics.items()):
            if isinstance(value, (int, float)) and not pd.isna(value):
                with metric_cols[i % len(metric_cols)]:
                    st.metric(
                        name.upper(),
                        f"{value:.4f}" if isinstance(value, float) else str(value)
                    )
        
        # Szczegółowe metryki
        with st.expander("📊 Wszystkie metryki"):
            st.json(metrics)
    
    def _render_feature_importance(self):
        """Renderuje ważność cech"""
        fi_df = self.state.feature_importance
        
        if fi_df.empty:
            return
            
        st.subheader("🏆 Najważniejsze cechy")
        
        # Slider dla liczby cech
        max_features = min(20, len(fi_df))
        n_features = st.slider("Liczba cech do wyświetlenia", 5, max_features, 10)
        
        # Wykres
        top_features = fi_df.head(n_features)
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title="Ważność cech"
        )
        fig.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela
        st.dataframe(top_features, use_container_width=True)
    
    def _render_recommendations(self):
        """Renderuje rekomendacje AI/reguły"""
        st.subheader("💡 Rekomendacje")
        
        # Sprawdź czy jest klucz OpenAI
        openai_key = get_openai_key()
        
        if openai_key and not self.state.feature_importance.empty:
            # AI rekomendacje
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
        else:
            # Podstawowe rekomendacje reguły
            self._render_rule_based_recommendations()
    
    def _render_rule_based_recommendations(self):
        """Podstawowe rekomendacje oparte na regułach"""
        recommendations = []
        metrics = self.state.metrics
        
        # Rekomendacje dla regresji
        if 'r2' in metrics:
            r2 = metrics['r2']
            if r2 < 0.6:
                recommendations.append("• R² < 0.6 - rozważ dodanie nowych cech lub transformacje danych")
            elif r2 > 0.95:
                recommendations.append("• R² > 0.95 - sprawdź czy nie ma overfittingu")
        
        # Rekomendacje dla klasyfikacji
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            if acc < 0.8:
                recommendations.append("• Accuracy < 80% - sprawdź balans klas i jakość cech")
        
        # Ogólne rekomendacje
        recommendations.extend([
            "• Przetestuj model na nowych danych spoza zbioru treningowego",
            "• Rozważ zbieranie dodatkowych danych dla poprawy wyników",
            "• Monitoruj performance modelu w czasie"
        ])
        
        for rec in recommendations:
            st.markdown(rec)
    
    def _render_predictions(self):
        """Sekcja predykcji"""
        st.subheader("🔮 Predykcje")
        
        if self.state.model is None or self.state.dataset is None:
            st.info("Brak wytrenowanego modelu")
            return
        
        # Wybór danych do predykcji
        data_source = st.radio(
            "Źródło danych",
            ["Pierwszy 10 wierszy z danych treningowych", "Wklej własne dane CSV"]
        )
        
        if data_source.startswith("Pierwszy"):
            # Predykcje na próbce treningowej
            X_pred = self.state.dataset.head(10).drop(columns=[self.state.target_column])
            predictions = self.state.model.predict(X_pred)
            
            results_df = X_pred.copy()
            results_df['Predykcja'] = predictions
            
            st.dataframe(results_df, use_container_width=True)
            
        else:
            # Własne dane CSV
            csv_input = st.text_area(
                "Wklej dane CSV",
                placeholder="kolumna1,kolumna2\nwartość1,wartość2",
                height=150
            )
            
            if csv_input.strip():
                try:
                    import io
                    custom_df = pd.read_csv(io.StringIO(csv_input))
                    predictions = self.state.model.predict(custom_df)
                    
                    results_df = custom_df.copy()
                    results_df['Predykcja'] = predictions
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Export predykcji
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Pobierz predykcje CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Błąd przetwarzania danych: {e}")
    
    def _render_history_section(self):
        """Sekcja historii eksperymentów"""
        st.markdown("## 📚 Historia eksperymentów")
        
        # Pobierz historię
        history_df = self.experiment_tracker.get_history()
        
        if history_df is None or history_df.empty:
            st.info("Brak zapisanych eksperymentów")
            return
        
        # Wyświetl historię
        st.dataframe(history_df, use_container_width=True)
        
        # Przyciski akcji
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Statystyki"):
                stats = self.experiment_tracker.get_statistics()
                st.json(stats)
                
        with col2:
            csv_data = self.experiment_tracker.export_to_csv()
            if csv_data:
                st.download_button(
                    "📥 Eksport CSV",
                    data=csv_data,
                    file_name="tmiv_history.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("🗑️ Wyczyść historię", type="secondary"):
                if st.confirmation_dialog("Czy na pewno wyczyścić historię?"):
                    self.experiment_tracker.clear_history(confirm=True)
                    st.rerun()
    
    def _render_sidebar_tools(self):
        """Narzędzia w sidebarze"""
        with st.sidebar:
            st.markdown("### 🛠️ Narzędzia")
            
            # Cache management
            if st.button("🧹 Wyczyść cache"):
                st.cache_data.clear()
                st.success("Cache wyczyszczony")
            
            # Backup database
            if st.button("💾 Backup bazy"):
                try:
                    backup_path = self.experiment_tracker.backup_database()
                    if backup_path:
                        st.success(f"Backup: {backup_path.name}")
                except Exception as e:
                    st.error(f"Błąd backup: {e}")
            
            # Pomoc
            with st.expander("ℹ️ Pomoc"):
                st.markdown("""
                **TMIV AutoML**
                
                1. **Wczytaj dane** - CSV, JSON, Excel
                2. **Wybierz target** - automatyczna detekcja
                3. **Eksploruj dane** - EDA i wizualizacje  
                4. **Trenuj model** - jeden klik
                5. **Analizuj wyniki** - metryki i wykresy
                6. **Wykonaj predykcje** - nowe dane
                
                Wszystkie eksperymenty są zapisywane w historii.
                """)
    
    def _reset_state(self):
        """Resetuje stan aplikacji"""
        self.state.model = None
        self.state.metrics = {}
        self.state.feature_importance = pd.DataFrame()
        self.state.metadata = {}
        self.state.dataset = None
        self.state.dataset_name = ""
        self.state.target_column = ""
        self.state.training_completed = False
        st.rerun()
    
    def _export_results(self):
        """Export wyników"""
        if not self.state.training_completed:
            st.warning("Brak wyników do eksportu")
            return
            
        # Przygotuj dane do eksportu
        export_data = {
            "metadata": self.state.metadata,
            "metrics": self.state.metrics,
            "feature_importance": self.state.feature_importance.to_dict('records') if not self.state.feature_importance.empty else [],
            "preprocessing_info": self.state.preprocessing_info,
            "export_timestamp": pd.Timestamp.now().isoformat()}