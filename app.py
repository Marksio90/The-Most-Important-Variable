# app.py — ZMODERNIZOWANA APLIKACJA TMIV z pełną integracją wszystkich modułów
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import traceback
import time

# ====== NASZE MODUŁY (z paczek 1-8) ======
from config.settings import get_settings
from frontend.ui_components import (
    render_sidebar, render_footer, render_upload_section,
    render_model_config_section, render_training_results,
    render_model_registry_section, render_data_preview_enhanced
)
from frontend.advanced_eda import render_eda_section
from backend.smart_target import SmartTargetSelector, format_target_explanation, format_alternatives_list
from backend.smart_target_llm import (
    LLMTargetSelector, render_openai_config, 
    render_smart_target_section_with_llm
)
from backend.ml_integration import (
    ModelConfig, train_model_comprehensive, save_model_artifacts, 
    load_model_artifacts, TrainingResult
)
from backend.utils import (
    infer_problem_type, validate_dataframe, seed_everything,
    hash_dataframe_signature, get_openai_key_from_envs
)
from backend.report_generator import (
    export_model_comprehensive, generate_quick_report, ModelReportGenerator
)
from db.db_utils import (
    DatabaseManager, TrainingRecord, create_training_record,
    save_training_record, get_training_history
)

# ====== KONFIGURACJA STRONY ======
st.set_page_config(
    page_title="TMIV - The Most Important Variables",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/tmiv',
        'Report a bug': "https://github.com/your-repo/tmiv/issues",
        'About': "TMIV - Zaawansowana platforma AutoML z inteligentnym wyborem targetu i automatyczną optymalizacją modeli uczenia maszynowego."
    }
)

# ====== CSS STYLING ======
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
    }
    .target-recommendation {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: 2px solid #2196f3;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border: 2px solid #ff9800;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(255, 152, 0, 0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border: 2px solid #4caf50;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.1);
    }
    .error-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border: 2px solid #f44336;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(244, 67, 54, 0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #667eea;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online { background-color: #4caf50; }
    .status-offline { background-color: #f44336; }
    .status-warning { background-color: #ff9800; }
</style>
""", unsafe_allow_html=True)

class TMIVApp:
    """Główna klasa aplikacji TMIV z pełną integracją wszystkich modułów."""
    
    def __init__(self):
        self.settings = get_settings()
        self.db_manager = DatabaseManager(self.settings.database_url)
        self.smart_target = SmartTargetSelector()
        self.llm_target = LLMTargetSelector()
        
        # Inicjalizacja session state
        self._init_session_state()
        
        # Seeding dla reproducibility
        seed_everything(self.settings.random_seed)
    
    def _init_session_state(self):
        """Inicjalizuje wszystkie zmienne session state."""
        defaults = {
            'df': None,
            'dataset_name': '',
            'target_recommendations': [],
            'selected_target': None,
            'training_result': None,
            'last_training_id': None,
            'openai_key_set': False,
            'data_processed': False,
            'model_trained': False,
            'export_files': {},
            'current_tab': 'upload',
            'settings_changed': False
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def run(self):
        """Główna pętla aplikacji."""
        try:
            self._render_header()
            self._render_main_content()
            self._render_footer()
        except Exception as e:
            st.error(f"❌ Błąd aplikacji: {str(e)}")
            if self.settings.debug:
                st.exception(e)
    
    def _render_header(self):
        """Renderuje nagłówek aplikacji."""
        st.markdown("""
        <div class="main-header">
            <h1>🎯 TMIV - The Most Important Variables</h1>
            <p>Zaawansowana platforma AutoML z inteligentnym wyborem targetu, automatyczną optymalizacją modeli uczenia maszynowego, eksploracyjną analizą danych oraz komprehensywnym systemem raportowania z możliwością eksportu modeli do środowisk produkcyjnych</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_main_content(self):
        """Renderuje główną zawartość aplikacji."""
        # Sidebar
        with st.sidebar:
            render_sidebar(self.settings)
            
            # Konfiguracja OpenAI
            openai_configured = render_openai_config()
            st.session_state.openai_key_set = openai_configured
        
        # Main content tabs
        tab_upload, tab_data, tab_eda, tab_target, tab_model, tab_results, tab_history = st.tabs([
            "📤 Wczytywanie", "📊 Dane", "🔍 EDA", "🎯 Target", "⚙️ Model", "📈 Wyniki", "📚 Historia"
        ])
        
        with tab_upload:
            self._render_upload_tab()
        
        with tab_data:
            self._render_data_tab()
        
        with tab_eda:
            self._render_eda_tab()
        
        with tab_target:
            self._render_target_tab()
        
        with tab_model:
            self._render_model_tab()
        
        with tab_results:
            self._render_results_tab()
        
        with tab_history:
            self._render_history_tab()
    
    def _render_upload_tab(self):
        """Renderuje zakładkę wczytywania danych."""
        st.header("📤 Wczytywanie danych")
        
        # Upload section
        uploaded_data = render_upload_section()
        
        if uploaded_data:
            df, dataset_name, status_msg = uploaded_data
            
            if df is not None:
                st.session_state.df = df
                st.session_state.dataset_name = dataset_name
                st.session_state.data_processed = True
                
                # Walidacja danych
                validation_result = validate_dataframe(df)
                if validation_result['valid']:
                    st.markdown(f"""
                    <div class="success-box">
                        ✅ <strong>Dane wczytane pomyślnie!</strong><br>
                        📊 Dataset: <code>{dataset_name}</code><br>
                        📏 Rozmiar: {len(df):,} wierszy × {len(df.columns):,} kolumn<br>
                        💾 Pamięć: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                        ⚠️ <strong>Ostrzeżenia dotyczące danych:</strong><br>
                        {chr(10).join(f'• {issue}' for issue in validation_result['issues'])}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-box">
                    ❌ <strong>Błąd wczytywania:</strong> {status_msg}
                </div>
                """, unsafe_allow_html=True)
    
    def _render_data_tab(self):
        """Renderuje zakładkę danych z rozbudowanym podglądem."""
        st.header("📊 Analiza danych")
        
        if st.session_state.df is None:
            st.info("🔼 Najpierw wczytaj dane w zakładce 'Wczytywanie'")
            return
        
        df = st.session_state.df
        
        # Enhanced data preview
        render_data_preview_enhanced(df, st.session_state.dataset_name)
    
    def _render_eda_tab(self):
        """Renderuje zakładkę EDA."""
        st.header("🔍 Eksploracyjna Analiza Danych (EDA)")
        
        if st.session_state.df is None:
            st.info("🔼 Najpierw wczytaj dane w zakładce 'Wczytywanie'")
            return
        
        df = st.session_state.df
        
        # Rozbudowane EDA
        render_eda_section(df)
    
    def _render_target_tab(self):
        """Renderuje zakładkę wyboru targetu."""
        st.header("🎯 Wybór zmiennej docelowej (Target)")
        
        if st.session_state.df is None:
            st.info("🔼 Najpierw wczytaj dane w zakładce 'Wczytywanie'")
            return
        
        df = st.session_state.df
        
        # Smart Target Selection z LLM
        if st.session_state.openai_key_set:
            selected_target = render_smart_target_section_with_llm(
                df, self.llm_target, self.smart_target
            )
        else:
            st.info("💡 Skonfiguruj klucz OpenAI w sidebar, aby uzyskać inteligentne rekomendacje targetu")
            
            # Fallback do podstawowego smart target
            st.subheader("🎯 Automatyczna rekomendacja targetu")
            
            recommendations = self.smart_target.recommend_targets(df)
            if recommendations:
                st.session_state.target_recommendations = recommendations
                
                # Display recommendations
                for i, rec in enumerate(recommendations[:3]):
                    with st.container():
                        st.markdown(f"""
                        <div class="target-recommendation">
                            <h4>🥇 Rekomendacja #{i+1}: {rec['column']}</h4>
                            <p><strong>Typ problemu:</strong> {rec['problem_type']}</p>
                            <p><strong>Wynik:</strong> {rec['score']:.3f}</p>
                            <p><strong>Uzasadnienie:</strong> {rec['explanation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Manual target selection
            st.subheader("📋 Wybór ręczny")
            available_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64', 'object', 'category']]
            selected_target = st.selectbox(
                "Wybierz zmienną docelową:",
                options=available_columns,
                index=0 if recommendations else 0,
                help="Wybierz kolumnę, którą chcesz przewidywać"
            )
        
        if selected_target:
            st.session_state.selected_target = selected_target
            
            # Analiza wybranego targetu
            target_series = df[selected_target]
            problem_type = infer_problem_type(target_series)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Wybrana zmienna", selected_target)
                st.metric("Typ problemu", problem_type.title())
            
            with col2:
                if pd.api.types.is_numeric_dtype(target_series):
                    st.metric("Min", f"{target_series.min():.3f}")
                    st.metric("Max", f"{target_series.max():.3f}")
                    st.metric("Średnia", f"{target_series.mean():.3f}")
                else:
                    st.metric("Unikalne klasy", target_series.nunique())
                    most_common = target_series.mode().iloc[0] if len(target_series.mode()) > 0 else "N/A"
                    st.metric("Najczęstsza klasa", str(most_common))
    
    def _render_model_tab(self):
        """Renderuje zakładkę konfiguracji i treningu modelu."""
        st.header("⚙️ Konfiguracja i trening modelu")
        
        if st.session_state.df is None:
            st.info("🔼 Najpierw wczytaj dane w zakładce 'Wczytywanie'")
            return
        
        if not st.session_state.selected_target:
            st.info("🔼 Najpierw wybierz zmienną docelową w zakładce 'Target'")
            return
        
        df = st.session_state.df
        target = st.session_state.selected_target
        problem_type = infer_problem_type(df[target])
        
        # Model configuration
        model_config = render_model_config_section(df, target, problem_type)
        
        if model_config and st.button("🚀 Rozpocznij trening modelu", type="primary", use_container_width=True):
            self._train_model(df, model_config)
    
    def _train_model(self, df: pd.DataFrame, config: ModelConfig):
        """Trenuje model z progress barem."""
        with st.container():
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Przygotowanie danych
                status_text.text("🔄 Przygotowywanie danych...")
                progress_bar.progress(10)
                time.sleep(0.5)
                
                # Trening
                status_text.text("🤖 Trenowanie modelu...")
                progress_bar.progress(30)
                
                # Główny trening
                result = train_model_comprehensive(df, config)
                progress_bar.progress(70)
                
                # Zapis do bazy
                status_text.text("💾 Zapisywanie wyników...")
                training_record = create_training_record(
                    dataset_name=st.session_state.dataset_name,
                    target=config.target,
                    config=config,
                    result=result
                )
                
                training_id = save_training_record(self.db_manager, training_record)
                st.session_state.last_training_id = training_id
                st.session_state.training_result = result
                st.session_state.model_trained = True
                progress_bar.progress(90)
                
                # Export comprehensive
                status_text.text("📤 Generowanie eksportów...")
                export_files = export_model_comprehensive(
                    model=result.model,
                    result=result,
                    config=config,
                    df=df,
                    dataset_name=st.session_state.dataset_name,
                    run_id=training_record.run_id,
                    db_manager=self.db_manager
                )
                
                st.session_state.export_files = export_files
                progress_bar.progress(100)
                
                # Success message
                status_text.empty()
                progress_bar.empty()
                
                st.markdown(f"""
                <div class="success-box">
                    🎉 <strong>Model wytrenowany pomyślnie!</strong><br>
                    📊 R² Score: {result.metrics.get('r2', result.metrics.get('accuracy', 0)):.4f}<br>
                    ⏱️ Czas treningu: {result.metadata.get('training_time_seconds', 0):.2f}s<br>
                    📁 Wygenerowano {len(export_files)} plików eksportowych
                </div>
                """, unsafe_allow_html=True)
                
                # Szybki raport
                quick_report = generate_quick_report(result, config, df, st.session_state.dataset_name)
                with st.expander("📋 Szybki raport", expanded=True):
                    st.markdown(quick_report)
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                
                st.markdown(f"""
                <div class="error-box">
                    ❌ <strong>Błąd treningu:</strong> {str(e)}
                </div>
                """, unsafe_allow_html=True)
                
                if self.settings.debug:
                    st.exception(e)
    
    def _render_results_tab(self):
        """Renderuje zakładkę z wynikami treningu."""
        st.header("📈 Wyniki treningu modelu")
        
        if not st.session_state.model_trained or not st.session_state.training_result:
            st.info("🔼 Najpierw wytrenuj model w zakładce 'Model'")
            return
        
        result = st.session_state.training_result
        
        # Renderuj wyniki
        render_training_results(result, st.session_state.export_files)
        
        # Registry modeli
        st.subheader("💾 Rejestr modeli")
        render_model_registry_section(self.db_manager, st.session_state.last_training_id)
    
    def _render_history_tab(self):
        """Renderuje zakładkę z historią treningów."""
        st.header("📚 Historia treningów")
        
        # Pobierz historię
        try:
            history = get_training_history(self.db_manager, limit=50)
            
            if not history:
                st.info("📝 Brak zapisanych treningów")
                return
            
            # Wyświetl historię
            st.subheader(f"📋 Ostatnie {len(history)} treningów")
            
            for record in history:
                with st.expander(f"🎯 {record.dataset_name} → {record.target} ({record.run_id[:8]}...)", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Problem", record.problem_type.title())
                        st.metric("Silnik", record.engine)
                    
                    with col2:
                        st.metric("Data", record.created_at.strftime("%Y-%m-%d %H:%M"))
                        st.metric("Czas treningu", f"{record.training_time_seconds:.2f}s")
                    
                    with col3:
                        primary_metric = record.metrics.get('r2') or record.metrics.get('accuracy')
                        if primary_metric:
                            st.metric("Główna metryka", f"{primary_metric:.4f}")
                        
                        st.metric("Cechy", record.n_features)
                    
                    # Metryki szczegółowe
                    if record.metrics:
                        st.write("**Wszystkie metryki:**")
                        metrics_text = " | ".join([f"{k}: {v:.4f}" for k, v in record.metrics.items() if isinstance(v, (int, float))])
                        st.text(metrics_text)
                    
                    # Przycisk do pobrania modelu
                    if st.button(f"📥 Pobierz model {record.run_id[:8]}", key=f"download_{record.id}"):
                        try:
                            model_artifacts = load_model_artifacts(record.run_id)
                            if model_artifacts:
                                st.success("✅ Model załadowany pomyślnie!")
                                # Tu możesz dodać logikę pobierania
                            else:
                                st.error("❌ Nie znaleziono plików modelu")
                        except Exception as e:
                            st.error(f"❌ Błąd ładowania: {e}")
        
        except Exception as e:
            st.error(f"❌ Błąd pobierania historii: {e}")
            if self.settings.debug:
                st.exception(e)
    
    def _render_footer(self):
        """Renderuje stopkę aplikacji."""
        render_footer()

def main():
    """Główna funkcja aplikacji."""
    try:
        app = TMIVApp()
        app.run()
    except Exception as e:
        st.error("❌ Krytyczny błąd aplikacji")
        st.exception(e)

if __name__ == "__main__":
    main()