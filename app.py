# app.py — NAPRAWIONY: kompatybilny z naszymi paczkami 1-6
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff

# ====== NASZE MODUŁY (z paczek 1-6) ======
from config.settings import get_settings
from frontend.ui_components import (
    render_upload_section, render_data_preview, render_model_config_section,
    render_training_results, render_sidebar, render_footer
)
from frontend.advanced_eda import render_eda_section
from backend.smart_target import SmartTargetSelector, format_target_explanation
from backend.ml_integration import (
    ModelConfig, train_model_comprehensive, save_model_artifacts, 
    load_model_artifacts, TrainingResult
)
from backend.utils import (
    infer_problem_type, validate_dataframe, seed_everything,
    hash_dataframe_signature
)
from db.db_utils import (
    DatabaseManager, TrainingRecord, create_training_record,
    save_training_record, get_training_history
)

# ================== STAN APLIKACJI ==================
@dataclass
class AppState:
    """Stan aplikacji TMIV."""
    dataset: Optional[pd.DataFrame] = None
    dataset_name: str = ""
    target_column: Optional[str] = None
    target_recommendations: List[Any] = field(default_factory=list)
    training_result: Optional[TrainingResult] = None
    training_completed: bool = False
    last_run_id: Optional[str] = None
    model_registry: Dict[str, Any] = field(default_factory=dict)


def get_app_state() -> AppState:
    """Pobiera stan aplikacji z session_state."""
    if "tmiv_app_state" not in st.session_state:
        st.session_state.tmiv_app_state = AppState()
    return st.session_state.tmiv_app_state


def reset_app_state():
    """Resetuje stan aplikacji."""
    if "tmiv_app_state" in st.session_state:
        del st.session_state.tmiv_app_state


# ================== POMOCNICZE FUNKCJE WYKRESÓW ==================
def plot_regression_results(y_true, y_pred, title="Predykcje vs Rzeczywistość"):
    """Renderuje scatter plot dla regresji."""
    if y_true is None or y_pred is None or len(y_true) == 0:
        st.info("Brak danych do wizualizacji regresji")
        return
    
    try:
        df_plot = pd.DataFrame({
            "y_true": np.array(y_true).flatten(), 
            "y_pred": np.array(y_pred).flatten()
        })
        
        fig = px.scatter(
            df_plot, 
            x="y_true", 
            y="y_pred", 
            title=title,
            labels={"y_true": "Wartości rzeczywiste", "y_pred": "Predykcje"}
        )
        
        # Linia idealna
        min_val = min(df_plot["y_true"].min(), df_plot["y_pred"].min())
        max_val = max(df_plot["y_true"].max(), df_plot["y_pred"].max())
        
        fig.add_shape(
            type="line",
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(color="red", dash="dash"),
            name="Idealna predykcja"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Błąd renderowania wykresu regresji: {str(e)}")


def plot_confusion_matrix(y_true, y_pred, title="Macierz pomyłek"):
    """Renderuje macierz pomyłek dla klasyfikacji."""
    if y_true is None or y_pred is None or len(y_true) == 0:
        st.info("Brak danych do wizualizacji klasyfikacji")
        return
    
    try:
        from sklearn.metrics import confusion_matrix
        
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=[str(label) for label in labels],
            y=[str(label) for label in labels],
            annotation_text=cm,
            showscale=True,
            colorscale='Blues'
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Predykcje",
            yaxis_title="Wartości rzeczywiste",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Błąd renderowania macierzy pomyłek: {str(e)}")


def show_metrics_dashboard(result: TrainingResult):
    """Wyświetla dashboard z metrykami."""
    if not result or not result.metrics:
        st.info("Brak metryk do wyświetlenia")
        return
    
    st.subheader("📈 Metryki modelu")
    
    # Główne metryki w kolumnach
    metrics_items = list(result.metrics.items())
    if metrics_items:
        cols = st.columns(min(4, len(metrics_items)))
        
        for i, (metric_name, metric_value) in enumerate(metrics_items[:4]):
            with cols[i]:
                try:
                    if isinstance(metric_value, (int, float)) and not pd.isna(metric_value):
                        formatted_name = metric_name.replace('_', ' ').title()
                        st.metric(formatted_name, f"{metric_value:.4f}")
                    else:
                        st.metric(metric_name, str(metric_value))
                except Exception:
                    st.metric(metric_name, "N/A")
    
    # Ostrzeżenia z metadanych
    if result.metadata and result.metadata.get("warnings"):
        st.subheader("⚠️ Ostrzeżenia")
        for warning in result.metadata["warnings"]:
            st.warning(warning)
    
    # Szczegóły w expander
    with st.expander("🔍 Szczegółowe metryki i metadane"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Metryki:**")
            st.json(result.metrics)
        
        with col2:
            st.write("**Metadane:**")
            st.json(result.metadata or {})


# ================== ZARZĄDZANIE MODELAMI ==================
def get_model_path(models_dir: Path, dataset_name: str, target: str, run_id: str) -> Path:
    """Generuje ścieżkę do modelu."""
    safe_dataset = dataset_name.replace("/", "_").replace("\\", "_")
    safe_target = target.replace("/", "_").replace("\\", "_")
    return models_dir / f"{safe_dataset}__{safe_target}__{run_id}"


def list_saved_models(models_dir: Path, dataset_name: str, target: str) -> List[Path]:
    """Listuje zapisane modele dla danego dataset/target."""
    if not models_dir.exists():
        return []
    
    safe_dataset = dataset_name.replace("/", "_").replace("\\", "_")
    safe_target = target.replace("/", "_").replace("\\", "_")
    pattern = f"{safe_dataset}__{safe_target}__"
    
    return sorted([
        p for p in models_dir.iterdir() 
        if p.is_dir() and p.name.startswith(pattern)
    ], reverse=True)


def render_model_registry(state: AppState, models_dir: Path):
    """Renderuje sekcję rejestru modeli."""
    st.header("💾 Rejestr modeli")
    
    if not state.target_column or not state.dataset_name:
        st.info("Wybierz dataset i target aby zarządzać modelami")
        return
    
    col1, col2 = st.columns(2)
    
    # Eksport modelu
    with col1:
        st.subheader("📤 Eksport modelu")
        
        can_export = (state.training_completed and 
                     state.training_result and 
                     state.training_result.model is not None)
        
        if st.button("💾 Zapisz model", disabled=not can_export, use_container_width=True):
            if can_export:
                try:
                    # Generuj run_id
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    run_id = f"run_{timestamp}"
                    
                    model_path = get_model_path(
                        models_dir, 
                        state.dataset_name, 
                        state.target_column, 
                        run_id
                    )
                    
                    save_model_artifacts(
                        model_path, 
                        state.training_result.model, 
                        state.training_result.metadata or {}
                    )
                    
                    state.last_run_id = run_id
                    st.success(f"✅ Model zapisany: {model_path.name}")
                    
                except Exception as e:
                    st.error(f"❌ Błąd zapisu modelu: {str(e)}")
            else:
                st.warning("Najpierw wytrenuj model")
    
    # Import/wczytywanie modeli
    with col2:
        st.subheader("📥 Wczytywanie modeli")
        
        saved_models = list_saved_models(models_dir, state.dataset_name, state.target_column)
        
        if not saved_models:
            st.info("Brak zapisanych modeli dla tego dataset/target")
        else:
            selected_model = st.selectbox(
                "Wybierz model:",
                options=[p.name for p in saved_models],
                help="Lista zapisanych modeli"
            )
            
            col_load, col_predict = st.columns(2)
            
            with col_load:
                if st.button("📂 Wczytaj model", use_container_width=True):
                    try:
                        model_path = models_dir / selected_model
                        model, metadata = load_model_artifacts(model_path)
                        
                        st.success("✅ Model wczytany!")
                        st.json(metadata)
                        
                    except Exception as e:
                        st.error(f"❌ Błąd wczytywania: {str(e)}")
            
            with col_predict:
                if st.button("🔮 Predykcje", use_container_width=True):
                    try:
                        model_path = models_dir / selected_model
                        model, metadata = load_model_artifacts(model_path)
                        
                        # Przygotuj dane do predykcji
                        if state.target_column in state.dataset.columns:
                            X = state.dataset.drop(columns=[state.target_column])
                        else:
                            X = state.dataset
                        
                        predictions = model.predict(X)
                        
                        # Pokaż wyniki
                        result_df = state.dataset.copy()
                        result_df["prediction"] = predictions
                        
                        st.write("**Przykładowe predykcje:**")
                        st.dataframe(result_df.head(10), use_container_width=True)
                        
                        # Download
                        csv_data = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "⬇️ Pobierz predykcje CSV",
                            csv_data,
                            file_name=f"predictions_{state.dataset_name}_{state.target_column}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Błąd predykcji: {str(e)}")


# ================== HISTORIA URUCHOMIEŃ ==================
def render_training_history(db_manager: DatabaseManager):
    """Renderuje historię treningów."""
    st.header("📚 Historia treningów")
    
    try:
        history = get_training_history(db_manager, limit=50)
        
        if not history:
            st.info("Brak historii treningów. Wytrenuj pierwszy model.")
            return
        
        # Statystyki
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Łącznie uruchomień", len(history))
        
        with col2:
            unique_datasets = len(set(record.dataset_name for record in history))
            st.metric("Różnych datasetów", unique_datasets)
        
        with col3:
            unique_targets = len(set(record.target_column for record in history))
            st.metric("Różnych targetów", unique_targets)
        
        with col4:
            completed = sum(1 for record in history if record.status == "completed")
            st.metric("Zakończone", completed)
        
        # Tabela historii
        st.subheader("📋 Lista uruchomień")
        
        history_data = []
        for record in history:
            history_data.append({
                "Dataset": record.dataset_name,
                "Target": record.target_column,
                "Engine": record.engine,
                "Problem": record.problem_type,
                "Status": record.status,
                "Data": record.created_at.strftime("%Y-%m-%d %H:%M") if record.created_at else "N/A"
            })
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Błąd wczytywania historii: {str(e)}")


# ================== GŁÓWNA APLIKACJA ==================
def main():
    """Główna funkcja aplikacji TMIV."""
    
    # Konfiguracja strony
    st.set_page_config(
        page_title="TMIV - The Most Important Variables",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    try:
        # Inicjalizacja
        settings = get_settings()
        state = get_app_state()
        
        # Przygotuj katalogi
        output_dir = Path("./tmiv_output")
        output_dir.mkdir(exist_ok=True)
        
        models_dir = output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Database manager
        db_manager = DatabaseManager(str(output_dir / "tmiv_history.db"))
        
        # Smart target selector
        smart_target = SmartTargetSelector()
        
        # Header
        st.title("🎯 TMIV - The Most Important Variables")
        st.markdown("**Zaawansowana platforma AutoML z inteligentnym wyborem targetu**")
        
        # Sidebar
        with st.sidebar:
            render_sidebar()
            
            if state.dataset is not None:
                st.divider()
                st.metric("Wiersze", f"{len(state.dataset):,}")
                st.metric("Kolumny", f"{len(state.dataset.columns):,}")
                
                if st.button("🗑️ Wyczyść dane", type="secondary"):
                    reset_app_state()
                    st.rerun()
        
        # Główna zawartość w tabsach
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📁 Dane", 
            "🎯 Target", 
            "📊 EDA", 
            "🤖 Trening", 
            "📈 Wyniki"
        ])
        
        # TAB 1: Wczytywanie danych
        with tab1:
            st.header("📁 Wczytywanie danych")
            
            uploaded_file = st.file_uploader(
                "Wybierz plik danych",
                type=['csv', 'xlsx', 'xls'],
                help="Obsługiwane formaty: CSV, Excel"
            )
            
            if uploaded_file is not None:
                try:
                    with st.spinner("Wczytywanie danych..."):
                        # Wczytaj dane
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                        
                        state.dataset = df
                        state.dataset_name = uploaded_file.name.split('.')[0]
                        
                        # Reset innych stanów
                        state.target_column = None
                        state.target_recommendations = []
                        state.training_result = None
                        state.training_completed = False
                    
                    st.success(f"✅ Wczytano {len(df)} wierszy i {len(df.columns)} kolumn")
                    
                    # Podgląd danych
                    render_data_preview(df)
                    
                    # Walidacja
                    validation = validate_dataframe(df)
                    if not validation['valid']:
                        st.error("❌ Problemy z danymi:")
                        for error in validation['errors']:
                            st.error(f"• {error}")
                    
                    if validation['warnings']:
                        for warning in validation['warnings']:
                            st.warning(f"⚠️ {warning}")
                
                except Exception as e:
                    st.error(f"❌ Błąd wczytywania: {str(e)}")
        
        # TAB 2: Wybór targetu
        with tab2:
            if state.dataset is None:
                st.info("📁 Najpierw wczytaj dane w zakładce 'Dane'")
            else:
                st.header("🎯 Inteligentny wybór targetu")
                
                # Analiza targetu przez AI
                if not state.target_recommendations:
                    with st.spinner("Analizuję potencjalne targety..."):
                        try:
                            recommendations = smart_target.analyze_and_recommend(state.dataset)
                            state.target_recommendations = recommendations
                        except Exception as e:
                            st.error(f"Błąd analizy targetu: {str(e)}")
                            state.target_recommendations = []
                
                # Pokaż rekomendacje
                if state.target_recommendations:
                    best_recommendation = state.target_recommendations[0]
                    
                    st.markdown("### 🎯 Najlepsza rekomendacja")
                    explanation = format_target_explanation(best_recommendation)
                    st.markdown(explanation)
                    
                    # Alternatywy
                    if len(state.target_recommendations) > 1:
                        with st.expander("🔄 Alternatywne opcje"):
                            for i, rec in enumerate(state.target_recommendations[1:4], 2):
                                st.write(f"**{i}. {rec.column}** ({rec.confidence:.1%}) - {rec.reason}")
                
                # Wybór targetu
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Domyślny wybór - najlepsza rekomendacja
                    default_idx = 0
                    if state.target_recommendations:
                        best_col = state.target_recommendations[0].column
                        if best_col in state.dataset.columns:
                            default_idx = list(state.dataset.columns).index(best_col)
                    
                    selected_target = st.selectbox(
                        "Wybierz kolumnę targetu:",
                        state.dataset.columns,
                        index=default_idx,
                        help="⭐ = zalecane przez AI"
                    )
                
                with col2:
                    if st.button("✅ Zatwierdź", type="primary"):
                        state.target_column = selected_target
                        st.success(f"Target: {selected_target}")
                        st.rerun()
                
                # Podgląd targetu
                if selected_target:
                    with st.expander("👀 Podgląd targetu", expanded=True):
                        target_series = state.dataset[selected_target]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Typ", str(target_series.dtype))
                        with col2:
                            st.metric("Unikalne", target_series.nunique())
                        with col3:
                            st.metric("Braki", target_series.isna().sum())
                        with col4:
                            missing_pct = target_series.isna().mean() * 100
                            st.metric("Braki %", f"{missing_pct:.1f}%")
                        
                        # Wykryj typ problemu
                        problem_type = infer_problem_type(state.dataset, selected_target)
                        st.info(f"🔍 Wykryty typ problemu: **{problem_type}**")
                        
                        # Podgląd rozkładu
                        if problem_type.lower() == "classification":
                            value_counts = target_series.value_counts().head(10)
                            st.bar_chart(value_counts)
                        else:
                            fig = px.histogram(target_series.dropna(), title="Rozkład wartości")
                            st.plotly_chart(fig, use_container_width=True)
        
        # TAB 3: EDA
        with tab3:
            if state.dataset is None:
                st.info("📁 Najpierw wczytaj dane")
            else:
                render_eda_section(state.dataset, state.target_column)
        
        # TAB 4: Trening
        with tab4:
            if state.dataset is None:
                st.info("📁 Najpierw wczytaj dane")
            elif not state.target_column:
                st.info("🎯 Najpierw wybierz target")
            else:
                st.header("🤖 Trening modelu AutoML")
                
                # Konfiguracja
                config = render_model_config_section(state.dataset, state.target_column)
                
                # Przycisk treningu
                if st.button("🚀 Trenuj model", type="primary", use_container_width=True):
                    with st.spinner("Trenuję model... To może potrwać kilka minut."):
                        try:
                            # Konfiguracja modelu
                            model_config = ModelConfig(
                                target=state.target_column,
                                engine=config['engine'],
                                test_size=config['test_size'],
                                cv_folds=config['cv_folds'],
                                random_state=config['random_state'],
                                stratify=config['stratify'],
                                enable_probabilities=config['enable_probabilities']
                            )
                            
                            # Trening
                            start_time = time.time()
                            result = train_model_comprehensive(state.dataset, model_config)
                            training_time = time.time() - start_time
                            
                            # Zapisz wynik
                            state.training_result = result
                            state.training_completed = True
                            
                            # Zapisz do historii
                            training_record = create_training_record(
                                model_config=model_config,
                                result=result,
                                df=state.dataset
                            )
                            training_record.training_time = training_time
                            training_record.dataset_name = state.dataset_name
                            
                            save_training_record(db_manager, training_record)
                            
                            st.success("✅ Trening zakończony pomyślnie!")
                            
                        except Exception as e:
                            st.error(f"❌ Błąd treningu: {str(e)}")
                            st.code(traceback.format_exc())
        
        # TAB 5: Wyniki
        with tab5:
            if not state.training_completed or not state.training_result:
                st.info("🤖 Najpierw wytrenuj model")
            else:
                st.header("📈 Wyniki treningu")
                
                # Dashboard metryk
                show_metrics_dashboard(state.training_result)
                
                # Wizualizacje
                if state.training_result.metadata:
                    validation_info = state.training_result.metadata.get('validation_info', {})
                    problem_type = state.training_result.metadata.get('problem_type', '').lower()
                    
                    if validation_info.get('y_true') and validation_info.get('y_pred'):
                        st.subheader("📊 Wizualizacje wyników")
                        
                        if problem_type == "regression":
                            plot_regression_results(
                                validation_info['y_true'], 
                                validation_info['y_pred']
                            )
                        elif problem_type == "classification":
                            plot_confusion_matrix(
                                validation_info['y_true'], 
                                validation_info['y_pred']
                            )
                
                # Feature importance
                if not state.training_result.feature_importance.empty:
                    st.subheader("🏆 Ważność cech")
                    
                    n_features = min(20, len(state.training_result.feature_importance))
                    top_features = state.training_result.feature_importance.head(n_features)
                    
                    fig = px.bar(
                        top_features,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title=f"Top {n_features} najważniejszych cech"
                    )
                    fig.update_layout(height=max(400, n_features * 25))
                    st.plotly_chart(fig, use_container_width=True)
        
        # Sekcje dodatkowe poniżej tabsów
        st.divider()
        
        # Model Registry
        render_model_registry(state, models_dir)
        
        st.divider()
        
        # Historia
        render_training_history(db_manager)
        
        # Footer
        render_footer()
    
    except Exception as e:
        st.error(f"❌ Krytyczny błąd aplikacji: {str(e)}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()