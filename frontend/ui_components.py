# frontend/ui_components.py — NAPRAWIONE: działające ustawienia, profesjonalna stopka, mniej duplikacji
from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend.ml_integration import TrainingResult
from backend.utils import infer_problem_type
from config.settings import MLEngine


def render_sidebar() -> Dict[str, Any]:
    """Renderuje sidebar z działającymi ustawieniami globalnymi."""
    st.sidebar.header("⚙️ Ustawienia aplikacji")
    
    # Inicjalizacja wartości domyślnych jeśli nie ma w session_state
    if "tmiv_color_theme" not in st.session_state:
        st.session_state.tmiv_color_theme = "default"
    if "tmiv_detail_level" not in st.session_state:
        st.session_state.tmiv_detail_level = "intermediate"
    if "tmiv_chart_height" not in st.session_state:
        st.session_state.tmiv_chart_height = 500
    if "tmiv_show_grid" not in st.session_state:
        st.session_state.tmiv_show_grid = True
    if "tmiv_interactive_charts" not in st.session_state:
        st.session_state.tmiv_interactive_charts = True
    
    settings = {}
    
    # Tema kolorów - NAPRAWIONA: używa session_state
    settings['color_theme'] = st.sidebar.selectbox(
        "Paleta kolorów wykresów:",
        ["default", "viridis", "plasma", "blues", "reds", "greens"],
        index=["default", "viridis", "plasma", "blues", "reds", "greens"].index(st.session_state.tmiv_color_theme),
        help="Wybierz paletę kolorów dla wszystkich wykresów",
        key="tmiv_color_theme"
    )
    
    # Poziom szczegółowości - NAPRAWIONA: używa session_state
    settings['detail_level'] = st.sidebar.selectbox(
        "Poziom szczegółowości:",
        ["basic", "intermediate", "advanced"],
        index=["basic", "intermediate", "advanced"].index(st.session_state.tmiv_detail_level),
        help="Kontroluje ilość wyświetlanych informacji i opcji",
        key="tmiv_detail_level"
    )
    
    # Informacja o aktualnych ustawieniach
    if settings['detail_level'] != "basic":
        st.sidebar.success(f"🎨 Tema: {settings['color_theme']}")
        st.sidebar.info(f"📊 Poziom: {settings['detail_level']}")
    
    # Ustawienia wykresów - NAPRAWIONE: używa session_state
    with st.sidebar.expander("📊 Parametry wizualizacji", expanded=False):
        settings['chart_height'] = st.slider(
            "Wysokość wykresów (px):", 
            200, 1000, 
            st.session_state.tmiv_chart_height,
            step=50,
            key="tmiv_chart_height"
        )
        settings['show_grid'] = st.checkbox(
            "Pokaż siatkę na wykresach", 
            value=st.session_state.tmiv_show_grid,
            key="tmiv_show_grid"
        )
        settings['interactive_charts'] = st.checkbox(
            "Interaktywne wykresy", 
            value=st.session_state.tmiv_interactive_charts,
            key="tmiv_interactive_charts"
        )
    
    # Reset ustawień
    if st.sidebar.button("🔄 Resetuj ustawienia"):
        st.session_state.tmiv_color_theme = "default"
        st.session_state.tmiv_detail_level = "intermediate"
        st.session_state.tmiv_chart_height = 500
        st.session_state.tmiv_show_grid = True
        st.session_state.tmiv_interactive_charts = True
        st.rerun()
    
    return settings


def render_upload_section() -> Optional[pd.DataFrame]:
    """Renderuje sekcję upload z zaawansowanymi opcjami."""
    st.header("📁 Źródło danych")
    st.markdown("**Wczytaj dane do analizy i treningu modelu ML**")
    
    # Taby dla różnych źródeł
    tab1, tab2, tab3 = st.tabs(["📄 Upload pliku", "🔗 URL", "🎲 Dane przykładowe"])
    
    df = None
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Wybierz plik danych:",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            help="Obsługiwane formaty: CSV, Excel, JSON, Parquet (max 200MB)"
        )
        
        if uploaded_file is not None:
            # Opcje parsowania dla CSV
            if uploaded_file.name.endswith('.csv'):
                with st.expander("🔧 Opcje parsowania CSV", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        separator = st.selectbox("Separator:", [',', ';', '\t', '|'], index=0)
                        encoding = st.selectbox("Kodowanie:", ['utf-8', 'latin-1', 'cp1252', 'cp1250'], index=0)
                    with col2:
                        decimal = st.selectbox("Separator dziesiętny:", ['.', ','], index=0)
                        skip_rows = st.number_input("Pomiń wiersze na początku:", 0, 20, 0)
                    
                df = _load_csv_with_options(uploaded_file, separator, encoding, decimal, skip_rows)
            else:
                df = _load_file(uploaded_file)
    
    with tab2:
        url = st.text_input(
            "URL do pliku CSV/JSON:",
            placeholder="https://raw.githubusercontent.com/example/data.csv",
            help="Podaj bezpośredni link do pliku danych"
        )
        
        if url and st.button("📥 Wczytaj z URL"):
            df = _load_from_url(url)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            dataset_choice = st.selectbox(
                "Wybierz dataset przykładowy:",
                ["", "iris", "wine", "diabetes", "boston_housing", "breast_cancer"],
                help="Gotowe datasety z sklearn do testowania algorytmów"
            )
        
        with col2:
            if dataset_choice:
                st.write(f"**Dataset: {dataset_choice}**")
                descriptions = {
                    "iris": "🌸 Klasyfikacja gatunków koszczców (150 wierszy, 4 cechy)",
                    "wine": "🍷 Klasyfikacja win (178 wierszy, 13 cech)",  
                    "diabetes": "💊 Regresja postępu cukrzycy (442 wiersze, 10 cech)",
                    "boston_housing": "🏠 Regresja cen domów w Bostonie (506 wierszy, 13 cech)",
                    "breast_cancer": "🎗️ Klasyfikacja nowotworu piersi (569 wierszy, 30 cech)"
                }
                st.caption(descriptions.get(dataset_choice, "Opis niedostępny"))
        
        if dataset_choice and st.button("🎲 Wczytaj przykład"):
            df = _load_sample_dataset(dataset_choice)
    
    # Podgląd wczytanych danych - NAPRAWIONA: bez duplikacji metryk
    if df is not None:
        _render_data_success_message(df)
        render_data_preview(df, show_metrics=False)  # wyłączamy duplikację metryk
    
    return df


def render_data_preview(df: pd.DataFrame, show_metrics: bool = True) -> None:
    """Renderuje podgląd danych z opcjonalnymi metrykami."""
    st.subheader("👀 Podgląd danych")

    # zapamiętaj liczbę wierszy między rerunami
    default_n = min(100, len(df))
    n = st.number_input(
        "Liczba wierszy do podglądu",
        min_value=5, max_value=int(min(5000, len(df))),
        value=int(st.session_state.get("tmiv_preview_rows", default_n)),
        step=5,
        key="tmiv_preview_rows",
        help="Zmiana nie wczytuje ponownie pliku – podgląd jest natychmiastowy."
    )

    # oddzielny key dla tabeli, żeby Streamlit nie „mrugał" komponentem
    st.dataframe(
        df.head(int(n)),
        use_container_width=True,
        hide_index=False,
        key="tmiv_preview_table"
    )

    # NAPRAWIONA: opcjonalne metadane tylko gdy show_metrics=True
    if show_metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Wiersze (całość)", f"{len(df):,}")
        with col2:
            st.metric("Kolumny", f"{len(df.columns):,}")
        with col3:
            missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Braki", f"{missing_pct:.1f}%")
        with col4:
            st.metric("Pamięć", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")


def render_model_config_section(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Renderuje konfigurację modelu z inteligentnymi rekomendacjami."""
    st.subheader("⚙️ Konfiguracja treningu modelu")
    
    # Analiza danych do inteligentnych rekomendacji
    n_rows = len(df)
    n_cols = len(df.columns)
    problem_type = infer_problem_type(df, target_col)
    missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
    
    config = {}
    
    # Podstawowe ustawienia
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Podstawowe parametry**")
        
        # Silnik ML z rekomendacją
        engines = ["auto", "sklearn", "lightgbm", "xgboost", "catboost"]
        recommended_engine = _get_recommended_engine(n_rows, n_cols, problem_type)
        engine_index = engines.index(recommended_engine) if recommended_engine in engines else 0
        
        config['engine'] = st.selectbox(
            f"Silnik ML: (💡 Polecany: {recommended_engine})",
            engines,
            index=engine_index,
            help="auto = automatyczny wybór najlepszego dostępnego silnika"
        )
        
        # Test size z rekomendacją  
        optimal_test_size = _get_optimal_test_size(n_rows)
        config['test_size'] = st.slider(
            f"Rozmiar zbioru testowego: (💡 Optymalny: {optimal_test_size})",
            0.1, 0.4, optimal_test_size, 0.05,
            help="Część danych przeznaczona do testowania modelu"
        )
        
        config['random_state'] = st.number_input(
            "Random seed:",
            1, 999999, 42,
            help="Zapewnia powtarzalność wyników między uruchomieniami"
        )
    
    with col2:
        st.write("**Walidacja krzyżowa**")
        
        # CV folds z rekomendacją
        optimal_cv = _get_optimal_cv_folds(n_rows)
        config['cv_folds'] = st.slider(
            f"Folds cross-validation: (💡 Optymalny: {optimal_cv})",
            3, 10, optimal_cv,
            help="Liczba części do walidacji krzyżowej"
        )
        
        # Stratyfikacja z rekomendacją
        stratify_recommended = _should_stratify(df, target_col, problem_type)
        config['stratify'] = st.checkbox(
            f"Stratyfikacja {'✅ Zalecane' if stratify_recommended else '⚠️ Opcjonalne'}",
            value=stratify_recommended,
            help="Zachowanie proporcji klas w podziale (dla klasyfikacji)"
        )
        
        # Prawdopodobieństwa 
        proba_recommended = problem_type.lower() == "classification"
        config['enable_probabilities'] = st.checkbox(
            f"Prawdopodobieństwa {'✅ Zalecane' if proba_recommended else ''}",
            value=proba_recommended,
            help="Obliczanie prawdopodobieństw dla klasyfikacji"
        )
    
    # Zaawansowane ustawienia - ROZBUDOWANE
    with st.expander("🔧 Zaawansowane ustawienia treningu", expanded=False):
        advanced_col1, advanced_col2 = st.columns(2)
        
        with advanced_col1:
            st.write("**Preprocessing**")
            
            # Feature engineering
            config['feature_engineering'] = st.checkbox(
                f"Inżynieria cech {'✅ Zalecane' if n_cols < 50 else '⚠️ Ostrożnie'}",
                value=(n_cols < 50),
                help="Automatyczne tworzenie nowych cech (interakcje, transformacje)"
            )
            
            # Selekcja cech
            config['feature_selection'] = st.checkbox(
                f"Selekcja cech {'✅ Zalecane' if n_cols > 20 else ''}",
                value=(n_cols > 20),
                help="Automatyczna selekcja najważniejszych cech"
            )
            
            # Balansowanie klas
            config['handle_imbalance'] = st.checkbox(
                f"Balansowanie klas {'✅ Zalecane' if _has_class_imbalance(df, target_col) else ''}",
                value=_has_class_imbalance(df, target_col),
                help="Automatyczne balansowanie niezrównoważonych klas"
            )
        
        with advanced_col2:
            st.write("**Optymalizacja**")
            
            # Tuning hiperparametrów
            config['hyperparameter_tuning'] = st.checkbox(
                f"Tuning hiperparametrów {'✅ Zalecane' if n_rows > 1000 else '⚠️ Może być wolne'}",
                value=(n_rows > 1000 and n_rows < 50000),
                help="Optymalizacja hiperparametrów (wydłuża czas treningu 3-10x)"
            )
            
            # Early stopping
            config['early_stopping'] = st.checkbox(
                f"Early stopping {'✅ Zalecane' if n_rows > 5000 else ''}",
                value=(n_rows > 5000),
                help="Zatrzymanie treningu gdy model przestaje się poprawiać"
            )
            
            # Ensembling
            config['ensemble_methods'] = st.checkbox(
                f"Metody zespołowe {'✅ Zalecane' if n_rows > 2000 else '⚠️ Może być wolne'}",
                value=False,  # domyślnie wyłączone bo czasochłonne
                help="Łączenie wielu modeli dla lepszych wyników (voting, stacking)"
            )
    
    # Przewidywany czas treningu - ULEPSZONY
    estimated_time = _estimate_training_time(df, config)
    performance_score = _estimate_performance_score(df, config, target_col)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"⏱️ **Przewidywany czas:** {estimated_time}")
    with col2:
        st.info(f"🎯 **Oczekiwane wyniki:** {performance_score}")
    
    # Ostrzeżenia i zalecenia
    warnings = _get_config_warnings(df, config, target_col)
    if warnings:
        with st.expander("⚠️ Ostrzeżenia i zalecenia", expanded=True):
            for warning in warnings:
                st.warning(warning)
    
    return config


def render_training_results(result: Any, export_files: Dict[str, str]) -> None:
    """Renderuje wyniki treningu modelu."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Metryki modelu")
        
        # Wyświetl wszystkie metryki w grid
        metrics = result.metrics
        if metrics:
            # Grupuj metryki w rzędy po 2
            metric_items = list(metrics.items())
            for i in range(0, len(metric_items), 2):
                subcol1, subcol2 = st.columns(2)
                
                # Pierwsza metryka
                if i < len(metric_items):
                    name, value = metric_items[i]
                    display_name = name.replace('_', ' ').title()
                    if isinstance(value, (int, float)):
                        subcol1.metric(display_name, f"{value:.4f}")
                
                # Druga metryka (jeśli istnieje)
                if i + 1 < len(metric_items):
                    name, value = metric_items[i + 1]
                    display_name = name.replace('_', ' ').title()
                    if isinstance(value, (int, float)):
                        subcol2.metric(display_name, f"{value:.4f}")
    
    with col2:
        st.subheader("⏱️ Metadane treningu")
        
        metadata = result.metadata
        if metadata:
            st.metric("Czas treningu", f"{metadata.get('training_time_seconds', 0):.2f}s")
            st.metric("Silnik ML", metadata.get('engine', 'Unknown'))
            st.metric("Problem", metadata.get('problem_type', 'Unknown').title())
    
    # Feature importance
    if not result.feature_importance.empty:
        st.subheader("🏆 Ważność cech")
        
        # Wykres
        fig = px.bar(
            result.feature_importance.head(15),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 15 najważniejszych cech'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Eksportowane pliki
    if export_files:
        st.subheader("📁 Eksportowane pliki")
        
        for file_type, file_path in export_files.items():
            if file_path:
                col_icon, col_desc = st.columns([1, 4])
                with col_icon:
                    if file_type == 'model':
                        st.write("🤖")
                    elif file_type == 'html_report':
                        st.write("📊")
                    elif file_type == 'feature_importance':
                        st.write("📈")
                    else:
                        st.write("📄")
                
                with col_desc:
                    st.write(f"**{file_type.replace('_', ' ').title()}**: `{file_path}`")


def render_model_registry_section(db_manager: Any, current_training_id: Optional[str] = None) -> None:
    """Renderuje sekcję rejestru modeli."""
    st.subheader("💾 Rejestr modeli")
    
    try:
        # Pobierz ostatnie modele z bazy
        from db.db_utils import get_training_history
        recent_models = get_training_history(db_manager, limit=10)
        
        if not recent_models:
            st.info("📝 Brak zapisanych modeli")
            return
        
        # Wyświetl modele w tabeli
        model_data = []
        for record in recent_models:
            # Oznacz aktualny model
            indicator = "🔥" if record.run_id == current_training_id else "📊"
            
            model_data.append({
                "Status": indicator,
                "Run ID": record.run_id[:8] + "...",
                "Dataset": record.dataset_name,
                "Target": record.target,
                "Engine": record.engine,
                "R²/Acc": f"{record.metrics.get('r2', record.metrics.get('accuracy', 0)):.3f}",
                "Data": record.created_at.strftime("%m-%d %H:%M"),
                "Czas": f"{record.training_time_seconds:.1f}s"
            })
        
        # Wyświetl jako DataFrame
        if model_data:
            df = pd.DataFrame(model_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Opcje zarządzania
        st.subheader("🔧 Zarządzanie modelami")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Wyczyść stare modele", help="Usuń modele starsze niż 30 dni"):
                st.info("Funkcja w przygotowaniu")
        
        with col2:
            if st.button("📊 Porównaj modele", help="Porównaj metryki różnych modeli"):
                st.info("Funkcja w przygotowaniu")
        
        with col3:
            if st.button("📥 Eksportuj rejestr", help="Pobierz rejestr jako CSV"):
                if model_data:
                    csv = pd.DataFrame(model_data).to_csv(index=False)
                    st.download_button(
                        label="💾 Pobierz CSV",
                        data=csv,
                        file_name=f"tmiv_models_registry_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Brak danych do eksportu")
    
    except Exception as e:
        st.error(f"❌ Błąd rejestru modeli: {e}")


def render_data_preview_enhanced(df: pd.DataFrame, dataset_name: str) -> None:
    """Renderuje rozbudowany podgląd danych bez powtórzeń."""
    st.subheader(f"📊 Analiza datasetu: {dataset_name}")
    
    # Podstawowe statystyki w metrikach
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📏 Wiersze", f"{len(df):,}")
    with col2:
        st.metric("📊 Kolumny", f"{len(df.columns):,}")
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("💾 Pamięć", f"{memory_mb:.1f} MB")
    with col4:
        missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns)) * 100)
        st.metric("❌ Braki", f"{missing_pct:.1f}%")
    
    # Tabs dla różnych widoków
    tab_preview, tab_types, tab_quality, tab_stats = st.tabs([
        "👀 Podgląd", "🏷️ Typy danych", "🔍 Jakość", "📈 Statystyki"
    ])
    
    with tab_preview:
        # Slider dla liczby wierszy
        max_rows = min(1000, len(df))
        num_rows = st.slider("Liczba wierszy do wyświetlenia:", 5, max_rows, min(100, max_rows))
        
        # Podgląd danych
        st.dataframe(df.head(num_rows), use_container_width=True)
    
    with tab_types:
        # Analiza typów danych
        type_info = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            null_count = df[col].isna().sum()
            unique_count = df[col].nunique()
            
            type_info.append({
                "Kolumna": col,
                "Typ": col_type,
                "Unikalne": unique_count,
                "Braki": null_count,
                "% Braków": f"{(null_count/len(df)*100):.1f}%"
            })
        
        st.dataframe(pd.DataFrame(type_info), use_container_width=True, hide_index=True)
    
    with tab_quality:
        # Analiza jakości danych
        st.write("**Podsumowanie jakości:**")
        
        # Duplikaty
        duplicates = df.duplicated().sum()
        st.write(f"• Duplikaty: {duplicates:,} wierszy ({(duplicates/len(df)*100):.1f}%)")
        
        # Kompletnie puste wiersze
        empty_rows = df.isna().all(axis=1).sum()
        st.write(f"• Puste wiersze: {empty_rows:,}")
        
        # Kolumny z wysokim % braków
        high_missing = df.columns[df.isna().mean() > 0.5].tolist()
        if high_missing:
            st.write(f"• Kolumny z >50% braków: {', '.join(high_missing)}")
        
        # Kolumny z jedną wartością
        single_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if single_value_cols:
            st.write(f"• Kolumny z jedną wartością: {', '.join(single_value_cols)}")
    
    with tab_stats:
        # Statystyki dla kolumn numerycznych
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.write("**Statystyki kolumn numerycznych:**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Rozkład dla kolumn kategorycznych
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.write("**Najczęstsze wartości w kolumnach kategorycznych:**")
            for col in categorical_cols[:5]:  # Max 5 kolumn
                top_values = df[col].value_counts().head(3)
                st.write(f"• **{col}**: {', '.join([f'{v} ({c})' for v, c in top_values.items()])}")

def render_footer() -> None:
    """Renderuje profesjonalną stopkę aplikacji."""
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**🎯 TMIV Platform**")
        st.caption("Zaawansowana platforma AutoML z inteligentnym wyborem targetu i automatyczną optymalizacją modeli uczenia maszynowego")
    
    with col2:
        st.markdown("**🧠 Inteligencja**")
        st.caption("• AI-powered target selection\n• Smart hyperparameter tuning\n• Automated feature engineering")
    
    with col3:
        st.markdown("**📊 Możliwości**")
        st.caption("• Multi-engine ML training\n• Advanced EDA & visualization\n• Model registry & versioning")
    
    with col4:
        st.markdown("**⚡ Status systemu**")
        st.caption("🟢 **Online** | ✅ **Gotowy do pracy**")
        st.caption(f"📅 Wersja: 4.2 | 🔄 Ostatnia aktualizacja: 2025")


# ===== FUNKCJE POMOCNICZE - NOWE/ROZBUDOWANE =====

def _render_data_success_message(df: pd.DataFrame) -> None:
    """Renderuje komunikat o pomyślnym wczytaniu danych."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"✅ **Dane wczytane pomyślnie:** {len(df):,} wierszy × {len(df.columns)} kolumn")
    with col2:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Pamięć", f"{memory_mb:.1f} MB")


def _get_recommended_engine(n_rows: int, n_cols: int, problem_type: str) -> str:
    """Zwraca zalecany silnik ML na podstawie charakterystyk danych."""
    if n_rows < 1000:
        return "sklearn"
    elif n_rows < 10000:
        return "lightgbm"  
    elif problem_type.lower() == "regression":
        return "xgboost"
    else:
        return "lightgbm"


def _get_optimal_test_size(n_rows: int) -> float:
    """Zwraca optymalny rozmiar zbioru testowego."""
    if n_rows < 1000:
        return 0.3
    elif n_rows < 5000:
        return 0.25  
    else:
        return 0.2


def _get_optimal_cv_folds(n_rows: int) -> int:
    """Zwraca optymalną liczbę folds dla CV."""
    if n_rows < 1000:
        return 3
    elif n_rows < 5000:
        return 5
    else:
        return 5  # stabilna wartość dla większych zbiorów


def _should_stratify(df: pd.DataFrame, target_col: str, problem_type: str) -> bool:
    """Sprawdza czy stratyfikacja jest zalecana."""
    if problem_type.lower() != "classification":
        return False
    
    try:
        value_counts = df[target_col].value_counts()
        if len(value_counts) < 2:
            return False
        
        # Sprawdź czy każda klasa ma co najmniej 2 próbki
        return (value_counts >= 2).all()
    except:
        return False


def _has_class_imbalance(df: pd.DataFrame, target_col: str) -> bool:
    """Sprawdza czy występuje niebalans klas."""
    try:
        if target_col not in df.columns:
            return False
        
        value_counts = df[target_col].value_counts()
        if len(value_counts) < 2:
            return False
        
        ratio = value_counts.max() / value_counts.min()
        return ratio > 3.0  # niebalans gdy stosunek > 3:1
    except:
        return False


def _get_config_warnings(df: pd.DataFrame, config: Dict[str, Any], target_col: str) -> List[str]:
    """Generuje ostrzeżenia i zalecenia dla konfiguracji."""
    warnings = []
    n_rows = len(df)
    
    # Ostrzeżenia o czasie treningu
    if config.get('hyperparameter_tuning') and config.get('ensemble_methods'):
        warnings.append("⏱️ Włączenie jednocześnie tuning hiperparametrów i metod zespołowych znacznie wydłuży trening.")
    
    # Ostrzeżenia o jakości
    if n_rows < 100:
        warnings.append("📊 Mały zbiór danych (<100 wierszy) może prowadzić do overfittingu.")
    
    if config.get('test_size', 0.2) > 0.3 and n_rows < 1000:
        warnings.append("🎯 Duży rozmiar zbioru testowego przy małym zbiorze może obniżyć jakość treningu.")
    
    # Zalecenia optymalizacji
    if not config.get('feature_selection') and len(df.columns) > 50:
        warnings.append("🔧 Dla zbiorów z wieloma cechami zalecana jest selekcja cech.")
    
    return warnings


def _estimate_training_time(df: pd.DataFrame, config: Dict[str, Any]) -> str:
    """Szacuje czas treningu z uwzględnieniem zaawansowanych opcji."""
    n_rows = len(df)
    n_cols = len(df.columns)
    
    # Bazowy czas w sekundach
    base_time = 5 + (n_rows / 1000) + (n_cols / 10)
    
    # Mnożniki dla różnych opcji
    if config.get('hyperparameter_tuning'):
        base_time *= 8
    if config.get('ensemble_methods'):  
        base_time *= 3
    if config.get('feature_engineering'):
        base_time *= 1.5
    if config.get('feature_selection'):
        base_time *= 1.3
    
    # Mnożniki dla silników
    engine = config.get('engine', 'sklearn')
    if engine == 'catboost':
        base_time *= 1.5
    elif engine == 'xgboost':
        base_time *= 1.2
        
    # Formatowanie
    if base_time < 60:
        return f"~{int(base_time)} sekund"
    elif base_time < 3600:
        return f"~{int(base_time / 60)} minut"
    else:
        return f"~{base_time / 3600:.1f} godzin"


def _estimate_performance_score(df: pd.DataFrame, config: Dict[str, Any], target_col: str) -> str:
    """Szacuje oczekiwaną jakość modelu."""
    score = 70  # bazowa jakość
    
    n_rows = len(df)
    n_cols = len(df.columns)
    
    # Bonusy za rozmiar danych
    if n_rows > 5000:
        score += 10
    elif n_rows > 1000:
        score += 5
    elif n_rows < 100:
        score -= 15
        
    # Bonusy za proporcję cech do próbek
    feature_ratio = n_cols / n_rows
    if feature_ratio < 0.1:
        score += 5
    elif feature_ratio > 0.5:
        score -= 10
        
    # Bonusy za optymalizacje
    if config.get('hyperparameter_tuning'):
        score += 8
    if config.get('feature_selection'):
        score += 5
    if config.get('ensemble_methods'):
        score += 10
        
    # Kara za niebalans klas
    if _has_class_imbalance(df, target_col):
        if not config.get('handle_imbalance'):
            score -= 10
        else:
            score += 3  # bonus za handling
    
    score = max(40, min(95, score))  # ograniczenie 40-95%
    
    if score >= 85:
        return f"Wysokie ({score}%)"
    elif score >= 70:
        return f"Dobre ({score}%)"
    elif score >= 55:
        return f"Średnie ({score}%)"
    else:
        return f"Niskie ({score}%)"


def _render_enhanced_metrics(result: TrainingResult) -> None:
    """Renderuje rozbudowane metryki główne."""
    st.write("### 🎯 Główne metryki modelu")
    
    if not result.metrics:
        st.warning("Brak dostępnych metryk")
        return
    
    # Dodatkowe metryki obliczone z metadanych
    enhanced_metrics = result.metrics.copy()
    
    # Oblicz dodatkowe metryki jeśli dostępne dane
    validation_info = result.metadata.get('validation_info', {})
    if 'y_true' in validation_info and 'y_pred' in validation_info:
        y_true = np.array(validation_info['y_true'])
        y_pred = np.array(validation_info['y_pred'])
        
        problem_type = result.metadata.get('problem_type', '').lower()
        
        if problem_type == 'regression':
            # Dodatkowe metryki regresji
            enhanced_metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
            enhanced_metrics['max_error'] = np.max(np.abs(y_true - y_pred))
            
        elif problem_type == 'classification':
            # Dodatkowe metryki klasyfikacji
            from sklearn.metrics import precision_score, recall_score
            try:
                enhanced_metrics['precision'] = precision_score(y_true, y_pred, average='macro')
                enhanced_metrics['recall'] = recall_score(y_true, y_pred, average='macro')
            except:
                pass
    
    # Wyświetl metryki w kolumnach
    metrics_items = list(enhanced_metrics.items())
    if metrics_items:
        # Maksymalnie 4 kolumny na wiersz
        n_cols = min(4, len(metrics_items))
        n_rows = (len(metrics_items) + n_cols - 1) // n_cols
        
        for row in range(n_rows):
            cols = st.columns(n_cols)
            for col_idx in range(n_cols):
                metric_idx = row * n_cols + col_idx
                if metric_idx < len(metrics_items):
                    metric_name, metric_value = metrics_items[metric_idx]
                    with cols[col_idx]:
                        formatted_name = _format_metric_name(metric_name)
                        formatted_value = _format_metric_value(metric_value)
                        delta_color = _get_metric_color(metric_name, metric_value)
                        
                        st.metric(
                            label=formatted_name,
                            value=formatted_value,
                            delta=None,
                            delta_color=delta_color
                        )


def _render_enhanced_feature_importance(fi_df: pd.DataFrame) -> None:
    """Renderuje rozbudowany wykres ważności cech."""
    st.write("### 🏆 Analiza ważności cech")
    
    if fi_df.empty:
        st.info("Brak danych o ważności cech")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Liczba cech do pokazania
        n_features = st.slider("Liczba najważniejszych cech:", 5, min(50, len(fi_df)), 15)
        top_features = fi_df.head(n_features)
        
        # Wykres słupkowy z kolorami
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f"Top {n_features} najważniejszych cech",
            color='importance',
            color_continuous_scale=st.session_state.get('tmiv_color_theme', 'viridis')
        )
        
        fig.update_layout(
            height=max(400, n_features * 25),
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Ważność względna",
            yaxis_title="Cechy",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**📊 Statystyki ważności**")
        st.metric("Najważniejsza cecha", fi_df.iloc[0]['feature'])
        st.metric("Max ważność", f"{fi_df.iloc[0]['importance']:.4f}")
        st.metric("Suma top 10", f"{fi_df.head(10)['importance'].sum():.4f}")
        
        # Procent kumulatywny
        cumsum = fi_df['importance'].cumsum()
        total = fi_df['importance'].sum()
        for n in [5, 10, 20]:
            if len(fi_df) >= n:
                pct = (cumsum.iloc[n-1] / total) * 100
                st.metric(f"Top {n} cech", f"{pct:.1f}% całk. ważności")
    
    # Tabela z wartościami
    with st.expander("📋 Szczegółowa tabela ważności", expanded=False):
        display_df = fi_df.copy()
        display_df['importance'] = display_df['importance'].round(6)
        display_df['cumulative_%'] = (display_df['importance'].cumsum() / display_df['importance'].sum() * 100).round(1)
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def _render_model_diagnostics(result: TrainingResult) -> None:
    """Renderuje diagnostykę modelu."""
    st.write("### 🔬 Diagnostyka modelu")
    
    # Model complexity analysis
    metadata = result.metadata or {}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Kompleksowość**")
        n_features_raw = metadata.get('n_features_raw', 0)
        n_features_after = metadata.get('n_features_after_preproc', 0)
        st.metric("Cechy pierwotne", f"{n_features_raw}")
        st.metric("Cechy po preprocessingu", f"{n_features_after}")
        
        if n_features_raw > 0 and n_features_after > 0:
            reduction = ((n_features_raw - n_features_after) / n_features_raw) * 100
            st.metric("Redukcja cech", f"{reduction:.1f}%")
    
    with col2:
        st.write("**Preprocessing**")
        st.metric("Kolumny numeryczne", metadata.get('num_cols_count', 'N/A'))
        st.metric("Kolumny kategoryczne", metadata.get('cat_cols_count', 'N/A'))
        stratified = metadata.get('stratified', False)
        st.metric("Stratyfikacja", "✅ Tak" if stratified else "❌ Nie")
    
    with col3:
        st.write("**Jakość danych**")
        n_rows = metadata.get('n_rows', 0)
        st.metric("Wiersze treningowe", f"{n_rows:,}")
        
        # Class distribution dla klasyfikacji
        class_dist = metadata.get('class_distribution', {})
        if class_dist and len(class_dist) > 1:
            values = list(class_dist.values())
            imbalance = max(values) / min(values)
            st.metric("Stosunek klas", f"{imbalance:.1f}:1")


# Pozostałe funkcje pomocnicze (bez zmian lub drobne poprawki)...

def _render_detailed_results(result: TrainingResult) -> None:
    """Renderuje szczegółowe wyniki modelu."""
    st.write("### 🔍 Szczegółowa analiza wyników")
    
    # Taby dla różnych typów analiz
    detail_tab1, detail_tab2, detail_tab3 = st.tabs([
        "📊 Wizualizacje predykcji", 
        "🎲 Macierz pomyłek", 
        "📈 Analiza residuów"
    ])
    
    with detail_tab1:
        _render_prediction_plots(result)
    
    with detail_tab2:
        _render_confusion_matrix(result)
    
    with detail_tab3:
        _render_residual_analysis(result)


def _render_prediction_plots(result: TrainingResult) -> None:
    """Renderuje wykresy predykcji."""
    validation_info = result.metadata.get('validation_info', {})
    
    if 'y_true' not in validation_info or 'y_pred' not in validation_info:
        st.info("Brak danych predykcji do wizualizacji")
        return
    
    y_true = np.array(validation_info['y_true'])
    y_pred = np.array(validation_info['y_pred'])
    
    problem_type = result.metadata.get('problem_type', 'unknown')
    
    if problem_type.lower() == 'regression':
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot dla regresji
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=y_true,
                y=y_pred,
                mode='markers',
                name='Predykcje',
                marker=dict(
                    color='blue',
                    opacity=0.6,
                    size=6
                )
            ))
            
            # Linia doskonałych predykcji
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Idealna predykcja',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="Predykcje vs Wartości rzeczywiste",
                xaxis_title="Wartości rzeczywiste",
                yaxis_title="Predykcje",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Histogram błędów
            errors = y_true - y_pred
            fig_hist = px.histogram(x=errors, title="Rozkład błędów predykcji", nbins=30)
            fig_hist.update_layout(
                xaxis_title="Błąd (true - pred)",
                yaxis_title="Częstość",
                height=400
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    else:
        # Histogram dla klasyfikacji
        fig = px.histogram(
            x=y_true,
            title="Rozkład rzeczywistych vs predykcji",
            nbins=min(20, len(np.unique(y_true)))
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_confusion_matrix(result: TrainingResult) -> None:
    """Renderuje macierz pomyłek dla klasyfikacji."""
    validation_info = result.metadata.get('validation_info', {})
    
    if 'confusion_matrix' not in validation_info:
        st.info("Macierz pomyłek dostępna tylko dla problemów klasyfikacji")
        return
    
    cm = np.array(validation_info['confusion_matrix'])
    labels = validation_info.get('labels', [f"Klasa {i}" for i in range(len(cm))])
    
    # Heatmapa macierzy pomyłek
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Pred: {label}" for label in labels],
        y=[f"True: {label}" for label in labels],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Macierz pomyłek",
        xaxis_title="Predykcje",
        yaxis_title="Wartości rzeczywiste",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statystyki z macierzy
    col1, col2, col3 = st.columns(3)
    with col1:
        accuracy = np.trace(cm) / np.sum(cm)
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        total_predictions = np.sum(cm)
        st.metric("Total predictions", f"{total_predictions:,}")
    with col3:
        errors = total_predictions - np.trace(cm) 
        st.metric("Błędne predykcje", f"{errors:,}")


def _render_residual_analysis(result: TrainingResult) -> None:
    """Renderuje analizę residuów (dla regresji)."""
    validation_info = result.metadata.get('validation_info', {})
    problem_type = result.metadata.get('problem_type', '').lower()
    
    if problem_type != 'regression' or 'y_true' not in validation_info:
        st.info("Analiza residuów dostępna tylko dla problemów regresji")
        return
        
    y_true = np.array(validation_info['y_true'])
    y_pred = np.array(validation_info['y_pred'])
    residuals = y_true - y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Residuals vs Fitted
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(color='blue', opacity=0.6),
            name='Residuals'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Residuals vs Fitted Values",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Q-Q plot (approximate)
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)
        
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode='markers',
            name='Sample quantiles'
        ))
        
        # Reference line
        fig_qq.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=theoretical_quantiles * np.std(residuals) + np.mean(residuals),
            mode='lines',
            name='Perfect normal',
            line=dict(color='red', dash='dash')
        ))
        
        fig_qq.update_layout(
            title="Q-Q Plot (Normalność residuów)",
            xaxis_title="Theoretical quantiles",
            yaxis_title="Sample quantiles",
            height=400
        )
        st.plotly_chart(fig_qq, use_container_width=True)


def _render_model_metadata(metadata: Dict[str, Any]) -> None:
    """Renderuje metadane modelu."""
    with st.expander("ℹ️ Metadane i szczegóły techniczne", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Podstawowe informacje:**")
            st.write(f"- Silnik: {metadata.get('engine', 'N/A')}")
            st.write(f"- Problem: {metadata.get('problem_type', 'N/A')}")
            st.write(f"- Wiersze: {metadata.get('n_rows', 'N/A'):,}")
            st.write(f"- Cechy przed: {metadata.get('n_features_raw', 'N/A')}")
            st.write(f"- Cechy po: {metadata.get('n_features_after_preproc', 'N/A')}")
        
        with col2:
            st.write("**Preprocessing:**")
            st.write(f"- Cechy numeryczne: {metadata.get('num_cols_count', 'N/A')}")
            st.write(f"- Cechy kategoryczne: {metadata.get('cat_cols_count', 'N/A')}")
            st.write(f"- Stratyfikacja: {'Tak' if metadata.get('stratified', False) else 'Nie'}")
            st.write(f"- Sygnatura danych: {metadata.get('data_signature', 'N/A')[:12]}...")
        
        # Ostrzeżenia
        warnings = metadata.get('warnings', [])
        if warnings:
            st.write("**⚠️ Ostrzeżenia:**")
            for warning in warnings:
                st.warning(warning)
        
        # Pełne metadane w JSON
        with st.expander("🔍 Pełne metadane (JSON)", expanded=False):
            st.json(metadata)


# Funkcje pomocnicze dla ładowania danych
def _load_csv_with_options(file, separator: str, encoding: str, decimal: str, skip_rows: int = 0) -> Optional[pd.DataFrame]:
    """Wczytuje CSV z opcjami."""
    try:
        return pd.read_csv(file, sep=separator, encoding=encoding, decimal=decimal, skiprows=skip_rows)
    except Exception as e:
        st.error(f"Błąd wczytywania CSV: {str(e)}")
        return None


def _load_file(file) -> Optional[pd.DataFrame]:
    """Wczytuje plik w automatycznym trybie."""
    try:
        if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            return pd.read_excel(file)
        elif file.name.endswith('.json'):
            return pd.read_json(file)
        elif file.name.endswith('.parquet'):
            return pd.read_parquet(file)
        else:
            return pd.read_csv(file)
    except Exception as e:
        st.error(f"Błąd wczytywania pliku: {str(e)}")
        return None


def _load_from_url(url: str) -> Optional[pd.DataFrame]:
    """Wczytuje dane z URL."""
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Błąd wczytywania z URL: {str(e)}")
        return None


def _load_sample_dataset(dataset_name: str) -> Optional[pd.DataFrame]:
    """Wczytuje przykładowy dataset."""
    try:
        if dataset_name == "iris":
            from sklearn.datasets import load_iris
            data = load_iris(as_frame=True)
            return data.frame
        elif dataset_name == "wine":
            from sklearn.datasets import load_wine
            data = load_wine(as_frame=True)  
            return data.frame
        elif dataset_name == "diabetes":
            from sklearn.datasets import load_diabetes
            data = load_diabetes(as_frame=True)
            return data.frame
        elif dataset_name == "boston_housing":
            # Boston housing jest deprecated, używamy alternatywy
            st.warning("Dataset Boston Housing jest przestarzały. Użyj własnych danych lub inny przykład.")
            return None
        elif dataset_name == "breast_cancer":
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer(as_frame=True)
            return data.frame
        else:
            st.error(f"Nieznany dataset: {dataset_name}")
            return None
    except Exception as e:
        st.error(f"Błąd wczytywania datasetu {dataset_name}: {str(e)}")
        return None


def _format_metric_name(metric_name: str) -> str:
    """Formatuje nazwę metryki."""
    formatting_map = {
        'accuracy': 'Dokładność',
        'f1_macro': 'F1 Score (macro)',
        'precision': 'Precision (macro)',
        'recall': 'Recall (macro)', 
        'roc_auc': 'ROC AUC',
        'roc_auc_ovr_macro': 'ROC AUC (OvR)',
        'mae': 'MAE',
        'rmse': 'RMSE',
        'r2': 'R²',
        'mape': 'MAPE (%)',
        'max_error': 'Max Error'
    }
    return formatting_map.get(metric_name, metric_name.replace('_', ' ').title())


def _format_metric_value(value: Any) -> str:
    """Formatuje wartość metryki."""
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return "N/A"
        if 0 <= abs(value) <= 1:
            return f"{value:.4f}"
        elif abs(value) < 10:
            return f"{value:.3f}"
        else:
            return f"{value:.2f}"
    return str(value)


def _get_metric_color(metric_name: str, value: Any) -> Optional[str]:
    """Zwraca kolor metryki w zależności od wartości."""
    if not isinstance(value, (int, float)) or pd.isna(value):
        return None
    
    # Metryki gdzie wyższe = lepsze
    higher_better = ['accuracy', 'f1_macro', 'precision', 'recall', 'roc_auc', 'roc_auc_ovr_macro', 'r2']
    # Metryki gdzie niższe = lepsze  
    lower_better = ['mae', 'rmse', 'mape', 'max_error']
    
    if metric_name in higher_better:
        return "normal" if value > 0.7 else "inverse"
    elif metric_name in lower_better:
        # Dla błędów względnych - próg zależy od typu
        if metric_name == 'mape':
            return "normal" if value < 20 else "inverse"
        else:
            return "normal" if value < np.mean([abs(value), 100]) else "inverse"
    
    return None