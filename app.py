# app.py — TMIV (naprawiony, spójne ścieżki z settings, bez duplikatów sekcji)
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

# ====== MODUŁY PROJEKTU ======
from config.settings import get_settings  # musi zwracać m.in. output_dir, models_dir, history_db_path
from frontend.ui_components import (
    render_upload_section, render_data_preview, render_model_config_section,
    render_training_results, render_sidebar, render_footer
)
from frontend.advanced_eda import render_eda_section
from backend.smart_target import SmartTargetSelector, format_target_explanation
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


# ================== POMOCNICZE WYKRESY ==================
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
        min_val = float(min(df_plot["y_true"].min(), df_plot["y_pred"].min()))
        max_val = float(max(df_plot["y_true"].max(), df_plot["y_pred"].max()))

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

        # annotation_text musi być listą list stringów
        ann = [[str(v) for v in row] for row in cm.tolist()]

        fig = ff.create_annotated_heatmap(
            z=cm,
            x=[str(label) for label in labels],
            y=[str(label) for label in labels],
            annotation_text=ann,
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
    """Renderuje sekcję rejestru modeli (zakładka 💾 Modele)."""
    st.header("💾 Rejestr modeli")

    if not state.target_column or not state.dataset_name:
        st.info("Wybierz dataset i target aby zarządzać modelami.")
        return

    col1, col2 = st.columns(2)

    # Eksport modelu
    with col1:
        st.subheader("📤 Eksport modelu")

        can_export = (
            state.training_completed and
            state.training_result and
            getattr(state.training_result, "model", None) is not None
        )

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
                st.warning("Najpierw wytrenuj model.")

    # Import/wczytywanie modeli
    with col2:
        st.subheader("📥 Wczytywanie modeli")

        saved_models = list_saved_models(models_dir, state.dataset_name, state.target_column)

        if not saved_models:
            st.info("Brak zapisanych modeli dla tego dataset/target.")
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
                        with st.expander("Metadane modelu", expanded=False):
                            st.json(metadata)

                    except Exception as e:
                        st.error(f"❌ Błąd wczytywania: {str(e)}")

            with col_predict:
                if st.button("🔮 Predykcje", use_container_width=True):
                    try:
                        model_path = models_dir / selected_model
                        model, metadata = load_model_artifacts(model_path)

                        # Przygotuj dane do predykcji
                        if state.dataset is None:
                            st.info("Wczytaj dane, aby wykonać predykcje.")
                            return

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
    """Renderuje historię treningów (zakładka 📚 Historia)."""
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
        seed_everything(42)

        # Przygotuj katalogi wg settings
        output_dir = Path(getattr(settings, "output_dir", "./tmiv_output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        models_dir = Path(getattr(settings, "models_dir", output_dir / "models"))
        models_dir.mkdir(parents=True, exist_ok=True)

        history_db_path = Path(getattr(settings, "history_db_path", output_dir / "tmiv_history.db"))
        db_manager = DatabaseManager(str(history_db_path))

        # Smart target selector (klasyczny; LLM używamy w UI w tabie Target)
        smart_target = SmartTargetSelector()

        # Header
        st.title("🎯 TMIV - The Most Important Variables")
        st.markdown("**Zaawansowana platforma AutoML z inteligentnym wyborem targetu**")

        # Sidebar
        with st.sidebar:
            render_sidebar()

            # Konfiguracja OpenAI (pobieranie z .env/st.secrets lub ręczny input)
            render_openai_config()

            if state.dataset is not None:
                st.divider()
                st.metric("Wiersze", f"{len(state.dataset):,}")
                st.metric("Kolumny", f"{len(state.dataset.columns):,}")

                if st.button("🗑️ Wyczyść dane", type="secondary"):
                    reset_app_state()
                    st.rerun()

        # Główne zakładki (dodane „Modele” i „Historia”)
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📁 Dane",
            "🎯 Target",
            "📊 EDA",
            "🤖 Trening",
            "📈 Wyniki",
            "💾 Modele",
            "📚 Historia"
        ])

        # TAB 1: Wczytywanie danych — STABILNE
    with tab1:
        st.header("📁 Wczytywanie danych")

        # zapamiętujemy wybór źródła między rerunami
        data_source = st.radio(
            "Wybierz źródło danych:",
            ["📁 Upload pliku", "🥑 Demo: Avocado", "🌸 Demo: Iris", "📊 Demo: Wine"],
            horizontal=True,
            key="tmiv_data_source",
        )

        # --- helper: cache odczytu pliku na podstawie bajtów ---
        @st.cache_data(show_spinner="Wczytywanie danych…")
        def _read_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
            import io
            bio = io.BytesIO(file_bytes)
            if filename.lower().endswith(".csv"):
                return pd.read_csv(bio)
            elif filename.lower().endswith((".xlsx", ".xls")):
                return pd.read_excel(bio)
            else:
                raise ValueError("Nieobsługiwany format pliku.")

        # --- helper: ustaw dataset w stanie TYLKO gdy się zmienił (po podpisie) ---
        def _set_dataset_if_changed(new_df: pd.DataFrame, new_name: str):
            sig_new = hash_dataframe_signature(new_df)
            sig_old = hash_dataframe_signature(state.dataset) if state.dataset is not None else None
            if sig_new != sig_old:
                state.dataset = new_df
                state.dataset_name = new_name
                # reset tylko przy NOWYM zbiorze
                state.target_column = None
                state.target_recommendations = []
                state.training_result = None
                state.training_completed = False

        new_df = None
        new_name = ""

        if data_source == "📁 Upload pliku":
            uploaded = st.file_uploader(
                "Wybierz plik danych",
                type=['csv', 'xlsx', 'xls'],
                help="Obsługiwane formaty: CSV, Excel",
                key="tmiv_uploader",
            )

            if uploaded is not None:
                try:
                    file_bytes = uploaded.getvalue()
                    new_df = _read_table(file_bytes, uploaded.name)
                    new_name = uploaded.name.rsplit(".", 1)[0]
                except Exception as e:
                    st.error(f"❌ Błąd wczytywania: {e}")

        elif data_source == "🥑 Demo: Avocado":
            if st.button("📥 Wczytaj Avocado Dataset", type="primary", key="btn_avocado"):
                try:
                    avocado_path = Path("data/avocado.csv")
                    if avocado_path.exists():
                        new_df = pd.read_csv(avocado_path)
                        if 'Unnamed: 0' in new_df.columns:
                            new_df = new_df.drop('Unnamed: 0', axis=1)
                        new_name = "avocado"
                        st.success("✅ Demo dataset Avocado wczytany!")
                        st.info("🎯 **Idealny dla regresji** – `AveragePrice` jako target.")
                    else:
                        st.error("❌ Nie znaleziono data/avocado.csv")
                        st.info("💡 Umieść plik avocado.csv w folderze data/")
                except Exception as e:
                    st.error(f"❌ Błąd: {e}")

            if state.dataset is None:
                with st.expander("ℹ️ O datasecie Avocado"):
                    st.write("""
                    **Avocado Prices Dataset**
                    - 📊 18 249 wierszy × 13 kolumn
                    - 🎯 Target: `AveragePrice`
                    - 📈 Regresja
                    """)

        elif data_source == "🌸 Demo: Iris":
            if st.button("📥 Wczytaj Iris Dataset", type="primary", key="btn_iris"):
                try:
                    from sklearn.datasets import load_iris
                    iris = load_iris(as_frame=True)
                    new_df = iris.frame
                    new_name = "iris"
                    st.success("✅ Demo dataset Iris wczytany!")
                    st.info("🎯 **Klasyfikacja** – gatunek kwiatu.")
                except Exception as e:
                    st.error(f"❌ Błąd: {e}")

            if state.dataset is None:
                with st.expander("ℹ️ O datasecie Iris"):
                    st.write("""
                    **Iris Flower Classification**
                    - 📊 150 wierszy × 5 kolumn
                    - 🎯 Target: `target` (3 klasy)
                    - 📈 Klasyfikacja
                    """)

        elif data_source == "📊 Demo: Wine":
            if st.button("📥 Wczytaj Wine Dataset", type="primary", key="btn_wine"):
                try:
                    from sklearn.datasets import load_wine
                    wine = load_wine(as_frame=True)
                    new_df = wine.frame
                    new_name = "wine"
                    st.success("✅ Demo dataset Wine wczytany!")
                    st.info("🎯 **Klasyfikacja** – gatunek wina.")
                except Exception as e:
                    st.error(f"❌ Błąd: {e}")

            if state.dataset is None:
                with st.expander("ℹ️ O datasecie Wine"):
                    st.write("""
                    **Wine Classification**
                    - 📊 178 wierszy × 14 kolumn
                    - 🎯 Target: `target` (3 klasy)
                    - 📈 Klasyfikacja
                    """)

        # Jeśli coś nowego wczytano — ustaw do stanu tylko gdy inny niż obecny
        if new_df is not None:
            _set_dataset_if_changed(new_df, new_name)

        # --- PREVIEW: zawsze bazuj na state.dataset (nie na lokalnym df!) ---
        if state.dataset is not None:
            st.success(f"✅ Dane: {len(state.dataset)} wierszy × {len(state.dataset.columns)} kolumn")
            render_data_preview(state.dataset)

            # Walidacja + ostrzeżenia (lekka, bez rerenderów)
            validation = validate_dataframe(state.dataset)
            if not validation['valid']:
                st.error("❌ Problemy z danymi:")
                for err in validation['errors']:
                    st.error(f"• {err}")
            for warn in validation['warnings']:
                st.warning(f"⚠️ {warn}")

            # narzędzia
            col_l, col_r = st.columns(2)
            with col_l:
                if st.button("🗑️ Wyczyść dane", type="secondary", key="btn_reset_data"):
                    reset_app_state()
                    st.rerun()


        # TAB 2: Wybór targetu
        with tab2:
            if state.dataset is None:
                st.info("📁 Najpierw wczytaj dane w zakładce **Dane**.")
            else:
                selected_target = render_smart_target_section_with_llm(
                    state.dataset,
                    state.dataset_name
                )

                if selected_target:
                    state.target_column = selected_target
                    st.success(f"✅ Target ustawiony: {selected_target}")

                    with st.expander("👀 Podgląd wybranego targetu", expanded=True):
                        target_series = state.dataset[selected_target]

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Typ", str(target_series.dtype))
                        with col2:
                            st.metric("Unikalne", int(target_series.nunique()))
                        with col3:
                            st.metric("Braki", int(target_series.isna().sum()))
                        with col4:
                            missing_pct = target_series.isna().mean() * 100
                            st.metric("Braki %", f"{missing_pct:.1f}%")

                        problem_type = infer_problem_type(state.dataset, selected_target)
                        st.info(f"🔍 Wykryty typ problemu: **{problem_type}**")

                        # Podgląd rozkładu
                        if problem_type.lower() == "classification":
                            value_counts = target_series.value_counts().head(20)
                            st.bar_chart(value_counts)
                            if len(value_counts) > 1 and value_counts.min() > 0:
                                imbalance_ratio = value_counts.max() / value_counts.min()
                                if imbalance_ratio > 3:
                                    st.warning(f"⚠️ Niebalans klas: {imbalance_ratio:.1f}:1")
                        else:
                            fig = px.histogram(target_series.dropna(), title="Rozkład wartości")
                            st.plotly_chart(fig, use_container_width=True)

        # TAB 3: EDA
        with tab3:
            if state.dataset is None:
                st.info("📁 Najpierw wczytaj dane.")
            else:
                render_eda_section(state.dataset, state.target_column)

        # TAB 4: Trening
        with tab4:
            if state.dataset is None:
                st.info("📁 Najpierw wczytaj dane.")
            elif not state.target_column:
                st.info("🎯 Najpierw wybierz target.")
            else:
                st.header("🤖 Trening modelu AutoML")

                # Konfiguracja
                config = render_model_config_section(state.dataset, state.target_column)

                # Przycisk treningu
                if st.button("🚀 Trenuj model", type="primary", use_container_width=True):
                    with st.spinner("Trenuję model…"):
                        try:
                            model_config = ModelConfig(
                                target=state.target_column,
                                engine=config['engine'],
                                test_size=config['test_size'],
                                cv_folds=config['cv_folds'],
                                random_state=config['random_state'],
                                stratify=config['stratify'],
                                enable_probabilities=config['enable_probabilities']
                            )

                            start_time = time.time()
                            result = train_model_comprehensive(state.dataset, model_config)
                            training_time = time.time() - start_time

                            state.training_result = result
                            state.training_completed = True

                            # Zapis rekordu do historii
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
                st.info("🤖 Najpierw wytrenuj model.")
            else:
                st.header("📈 Wyniki treningu")

                # Dashboard metryk
                show_metrics_dashboard(state.training_result)

                # Wizualizacje
                validation_info = (state.training_result.metadata or {}).get('validation_info', {})
                problem_type = (state.training_result.metadata or {}).get('problem_type', '').lower()

                if validation_info.get('y_true') is not None and validation_info.get('y_pred') is not None:
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
                if hasattr(state.training_result, "feature_importance") and \
                   isinstance(state.training_result.feature_importance, pd.DataFrame) and \
                   not state.training_result.feature_importance.empty:
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

        # TAB 6: Modele (Rejestr)
        with tab6:
            render_model_registry(state, models_dir)

        # TAB 7: Historia
        with tab7:
            render_training_history(db_manager)

        # Stopka
        st.divider()
        render_footer()

    except Exception as e:
        st.error(f"❌ Krytyczny błąd aplikacji: {str(e)}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
