# frontend/ui_components.py — ZMODERNIZOWANE: lepszy UX + nowe komponenty
from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend.ml_integration import TrainingResult
from config.settings import MLEngine


def render_sidebar() -> Dict[str, Any]:
    """Renderuje sidebar z ustawieniami globalnymi."""
    st.sidebar.header("⚙️ Ustawienia")
    
    settings = {}
    
    # Tema kolorów
    settings['color_theme'] = st.sidebar.selectbox(
        "Tema kolorów:",
        ["default", "viridis", "plasma", "blues"],
        help="Paleta kolorów dla wykresów"
    )
    
    # Poziom szczegółowości
    settings['detail_level'] = st.sidebar.selectbox(
        "Poziom szczegółowości:",
        ["basic", "intermediate", "advanced"],
        index=1,
        help="Ilość wyświetlanych informacji"
    )
    
    # Ustawienia wykresów
    with st.sidebar.expander("📊 Ustawienia wykresów", expanded=False):
        settings['chart_height'] = st.slider("Wysokość wykresów:", 300, 800, 500)
        settings['show_grid'] = st.checkbox("Pokaż siatkę", value=True)
        settings['interactive_charts'] = st.checkbox("Interaktywne wykresy", value=True)
    
    return settings


def render_upload_section() -> Optional[pd.DataFrame]:
    """Renderuje sekcję upload z zaawansowanymi opcjami."""
    st.header("📁 Wczytywanie danych")
    
    # Taby dla różnych źródeł
    tab1, tab2, tab3 = st.tabs(["📄 Upload pliku", "🔗 URL", "🎲 Dane przykładowe"])
    
    df = None
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Wybierz plik danych:",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            help="Obsługiwane formaty: CSV, Excel, JSON, Parquet"
        )
        
        if uploaded_file is not None:
            # Opcje parsowania dla CSV
            if uploaded_file.name.endswith('.csv'):
                with st.expander("🔧 Opcje parsowania CSV", expanded=False):
                    separator = st.selectbox("Separator:", [',', ';', '\t', '|'], index=0)
                    encoding = st.selectbox("Kodowanie:", ['utf-8', 'latin-1', 'cp1252'], index=0)
                    decimal = st.selectbox("Separator dziesiętny:", ['.', ','], index=0)
                    
                df = _load_csv_with_options(uploaded_file, separator, encoding, decimal)
            else:
                df = _load_file(uploaded_file)
    
    with tab2:
        url = st.text_input(
            "URL do pliku CSV:",
            placeholder="https://example.com/data.csv",
            help="Podaj bezpośredni link do pliku CSV"
        )
        
        if url and st.button("📥 Wczytaj z URL"):
            df = _load_from_url(url)
    
    with tab3:
        dataset_choice = st.selectbox(
            "Wybierz dataset przykładowy:",
            ["", "iris", "titanic", "boston_housing", "wine", "diabetes"],
            help="Gotowe datasety do testowania"
        )
        
        if dataset_choice and st.button("🎲 Wczytaj przykład"):
            df = _load_sample_dataset(dataset_choice)
    
    # Podgląd wczytanych danych
    if df is not None:
        render_data_preview(df)
    
    return df


def render_data_preview(df: pd.DataFrame) -> None:
    """Renderuje podgląd wczytanych danych."""
    st.subheader("👀 Podgląd danych")
    
    # Szybkie statystyki
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Wiersze", f"{len(df):,}")
    with col2:
        st.metric("Kolumny", f"{len(df.columns):,}")
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Pamięć", f"{memory_mb:.1f} MB")
    with col4:
        missing_cells = df.isna().sum().sum()
        st.metric("Braki", f"{missing_cells:,}")
    with col5:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeryczne", numeric_cols)
    
    # Taby podglądu
    preview_tab1, preview_tab2, preview_tab3 = st.tabs(["📋 Pierwsze wiersze", "ℹ️ Info", "📊 Statystyki"])
    
    with preview_tab1:
        n_rows = st.slider(
            "Liczba wierszy do podglądu:", 
            min_value=5, 
            max_value=min(50, len(df)), 
            value=10,
            key="data_preview_slider"  
        )
        st.dataframe(df.head(n_rows), use_container_width=True)
    
    with preview_tab2:
        info_df = _create_info_dataframe(df)
        st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    with preview_tab3:
        if len(df.select_dtypes(include=[np.number]).columns) > 0:
            st.dataframe(df.describe(), use_container_width=True)
        else:
            st.info("Brak kolumn numerycznych do statystyk")


def render_model_config_section(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Renderuje sekcję konfiguracji modelu z zaawansowanymi opcjami."""
    st.subheader("⚙️ Konfiguracja modelu")
    
    config = {}
    
    # Podstawowe ustawienia
    col1, col2 = st.columns(2)
    
    with col1:
        config['engine'] = st.selectbox(
            "Silnik ML:",
            ["auto", "sklearn", "lightgbm", "xgboost", "catboost"],
            help="auto = automatyczny wybór najlepszego dostępnego"
        )
        
        config['test_size'] = st.slider(
            "Rozmiar zbioru testowego:",
            0.1, 0.4, 0.2, 0.05,
            help="Część danych przeznaczona do testowania"
        )
        
        config['random_state'] = st.number_input(
            "Random seed:",
            1, 999999, 42,
            help="Zapewnia powtarzalność wyników"
        )
    
    with col2:
        config['cv_folds'] = st.slider(
            "Folds cross-validation:",
            3, 10, 5,
            help="Liczba części do walidacji krzyżowej"
        )
        
        config['stratify'] = st.checkbox(
            "Stratyfikacja",
            value=True,
            help="Zachowanie proporcji klas w podziale (dla klasyfikacji)"
        )
        
        config['enable_probabilities'] = st.checkbox(
            "Prawdopodobieństwa",
            value=True,
            help="Obliczanie prawdopodobieństw dla klasyfikacji"
        )
    
    # Zaawansowane ustawienia
    with st.expander("🔧 Zaawansowane ustawienia", expanded=False):
        advanced_col1, advanced_col2 = st.columns(2)
        
        with advanced_col1:
            config['max_categories'] = st.number_input(
                "Max kategorii:",
                10, 1000, 200,
                help="Maksymalna liczba kategorii do One-Hot Encoding"
            )
            
            config['handle_imbalance'] = st.checkbox(
                "Balansowanie klas",
                value=False,
                help="Automatyczne balansowanie niezrównoważonych klas"
            )
        
        with advanced_col2:
            config['feature_selection'] = st.checkbox(
                "Selekcja cech",
                value=False,
                help="Automatyczna selekcja najważniejszych cech"
            )
            
            config['hyperparameter_tuning'] = st.checkbox(
                "Tuning hiperparametrów",
                value=False,
                help="Optymalizacja hiperparametrów (wydłuża czas treningu)"
            )
    
    # Przewidywany czas treningu
    estimated_time = _estimate_training_time(df, config)
    st.info(f"⏱️ Przewidywany czas treningu: {estimated_time}")
    
    return config


def render_training_results(result: TrainingResult, show_details: bool = True) -> None:
    """Renderuje wyniki treningu z interaktywnymi wykresami."""
    if not result:
        st.warning("Brak wyników do wyświetlenia")
        return
    
    st.subheader("📈 Wyniki treningu modelu")
    
    # Metryki główne
    _render_main_metrics(result)
    
    # Feature importance
    if not result.feature_importance.empty:
        _render_feature_importance(result.feature_importance)
    
    # Szczegółowe wyniki
    if show_details:
        _render_detailed_results(result)
    
    # Metadane modelu
    _render_model_metadata(result.metadata)


def _render_main_metrics(result: TrainingResult) -> None:
    """Renderuje główne metryki modelu."""
    st.write("### 🎯 Główne metryki")
    
    if not result.metrics:
        st.warning("Brak dostępnych metryk")
        return
    
    # Automatyczne formatowanie metryk
    metrics_cols = st.columns(min(4, len(result.metrics)))
    
    for i, (metric_name, metric_value) in enumerate(result.metrics.items()):
        with metrics_cols[i % len(metrics_cols)]:
            formatted_name = _format_metric_name(metric_name)
            formatted_value = _format_metric_value(metric_value)
            delta_color = _get_metric_color(metric_name, metric_value)
            
            st.metric(
                label=formatted_name,
                value=formatted_value,
                delta=None,
                delta_color=delta_color
            )


def _render_feature_importance(fi_df: pd.DataFrame) -> None:
    """Renderuje wykres ważności cech."""
    st.write("### 🏆 Ważność cech")
    
    if fi_df.empty:
        st.info("Brak danych o ważności cech")
        return
    
    # Liczba cech do pokazania
    n_features = min(20, len(fi_df))
    top_features = fi_df.head(n_features)
    
    # Wykres słupkowy
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=f"Top {n_features} najważniejszych cech",
        color='importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=max(400, n_features * 25),
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Ważność",
        yaxis_title="Cechy"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela z wartościami
    with st.expander("📋 Tabela ważności cech", expanded=False):
        st.dataframe(
            top_features.round(4),
            use_container_width=True,
            hide_index=True
        )


def _render_detailed_results(result: TrainingResult) -> None:
    """Renderuje szczegółowe wyniki modelu."""
    st.write("### 🔍 Szczegółowe wyniki")
    
    # Taby dla różnych typów analiz
    detail_tab1, detail_tab2, detail_tab3 = st.tabs([
        "📊 Wizualizacje predykcji", 
        "🎲 Macierz pomyłek", 
        "📈 Krzywa uczenia"
    ])
    
    with detail_tab1:
        _render_prediction_plots(result)
    
    with detail_tab2:
        _render_confusion_matrix(result)
    
    with detail_tab3:
        _render_learning_curves(result)


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
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Histogram dla klasyfikacji
        fig = px.histogram(
            x=y_true,
            title="Rozkład predykcji vs rzeczywistych klas",
            nbins=min(20, len(np.unique(y_true)))
        )
        
        st.plotly_chart(fig, use_container_width=True)


def _render_confusion_matrix(result: TrainingResult) -> None:
    """Renderuje macierz pomyłek dla klasyfikacji."""
    validation_info = result.metadata.get('validation_info', {})
    
    if 'confusion_matrix' not in validation_info:
        st.info("Brak macierzy pomyłek (dostępna tylko dla klasyfikacji)")
        return
    
    cm = np.array(validation_info['confusion_matrix'])
    labels = validation_info.get('labels', [f"Klasa {i}" for i in range(len(cm))])
    
    # Heatmapa macierzy pomyłek
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Macierz pomyłek",
        xaxis_title="Predykcje",
        yaxis_title="Rzeczywiste",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_learning_curves(result: TrainingResult) -> None:
    """Renderuje krzywe uczenia (placeholder)."""
    st.info("🚧 Krzywe uczenia będą dostępne w następnej wersji")


def _render_model_metadata(metadata: Dict[str, Any]) -> None:
    """Renderuje metadane modelu."""
    with st.expander("ℹ️ Metadane modelu", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Podstawowe informacje:**")
            st.write(f"- Silnik: {metadata.get('engine', 'N/A')}")
            st.write(f"- Problem: {metadata.get('problem_type', 'N/A')}")
            st.write(f"- Wiersze: {metadata.get('n_rows', 'N/A'):,}")
            st.write(f"- Cechy: {metadata.get('n_features_raw', 'N/A')}")
        
        with col2:
            st.write("**Preprocessing:**")
            st.write(f"- Cechy numeryczne: {metadata.get('num_cols_count', 'N/A')}")
            st.write(f"- Cechy kategoryczne: {metadata.get('cat_cols_count', 'N/A')}")
            st.write(f"- Stratyfikacja: {'Tak' if metadata.get('stratified', False) else 'Nie'}")
        
        # Ostrzeżenia
        warnings = metadata.get('warnings', [])
        if warnings:
            st.write("**⚠️ Ostrzeżenia:**")
            for warning in warnings:
                st.warning(warning)


def render_model_registry_section() -> None:
    """Renderuje sekcję rejestru modeli."""
    st.subheader("📚 Rejestr modeli")
    st.info("🚧 Rejestr modeli będzie dostępny w następnej wersji")


def render_footer() -> None:
    """Renderuje stopkę aplikacji."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🎯 TMIV v2.0**")
        st.markdown("AutoML Platform")
    
    with col2:
        st.markdown("**📈 Funkcje**")
        st.markdown("Smart Target • EDA • ML • Historia")
    
    with col3:
        st.markdown("**📊 Status**")
        st.markdown("🟢 Online | ✅ Gotowy")


# Funkcje pomocnicze
def _load_csv_with_options(file, separator: str, encoding: str, decimal: str) -> Optional[pd.DataFrame]:
    """Wczytuje CSV z opcjami."""
    try:
        return pd.read_csv(file, sep=separator, encoding=encoding, decimal=decimal)
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
            df = data.frame
            return df
        elif dataset_name == "titanic":
            # Można dodać inne datasety
            st.info("Dataset Titanic będzie dostępny wkrótce")
            return None
        # Dodaj więcej datasetów według potrzeb
        else:
            st.error(f"Nieznany dataset: {dataset_name}")
            return None
    except Exception as e:
        st.error(f"Błąd wczytywania datasetu {dataset_name}: {str(e)}")
        return None


def _create_info_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Tworzy DataFrame z informacjami o kolumnach."""
    info_data = []
    
    for col in df.columns:
        series = df[col]
        info_data.append({
            'Kolumna': col,
            'Typ': str(series.dtype),
            'Braki': series.isna().sum(),
            'Braki %': f"{series.isna().mean() * 100:.1f}%",
            'Unikalne': series.nunique(),
            'Przykład': str(series.dropna().iloc[0]) if len(series.dropna()) > 0 else 'N/A'
        })
    
    return pd.DataFrame(info_data)


def _estimate_training_time(df: pd.DataFrame, config: Dict[str, Any]) -> str:
    """Szacuje czas treningu."""
    n_rows = len(df)
    n_cols = len(df.columns)
    
    # Prosta heurystyka
    base_time = 10  # sekundy
    
    if n_rows > 10000:
        base_time *= 2
    if n_rows > 100000:
        base_time *= 3
    
    if n_cols > 50:
        base_time *= 1.5
    
    if config.get('hyperparameter_tuning', False):
        base_time *= 5
    
    if base_time < 60:
        return f"{int(base_time)} sekund"
    elif base_time < 3600:
        return f"{int(base_time / 60)} minut"
    else:
        return f"{base_time / 3600:.1f} godzin"


def _format_metric_name(metric_name: str) -> str:
    """Formatuje nazwę metryki."""
    formatting_map = {
        'accuracy': 'Dokładność',
        'f1_macro': 'F1 Score',
        'roc_auc': 'ROC AUC',
        'mae': 'MAE',
        'rmse': 'RMSE',
        'r2': 'R²'
    }
    return formatting_map.get(metric_name, metric_name.upper())


def _format_metric_value(value: Any) -> str:
    """Formatuje wartość metryki."""
    if isinstance(value, (int, float)):
        if 0 <= value <= 1:
            return f"{value:.3f}"
        else:
            return f"{value:.2f}"
    return str(value)


def _get_metric_color(metric_name: str, value: Any) -> Optional[str]:
    """Zwraca kolor metryki w zależności od wartości."""
    if not isinstance(value, (int, float)):
        return None
    
    # Metryki gdzie wyższe = lepsze
    higher_better = ['accuracy', 'f1_macro', 'roc_auc', 'r2']
    # Metryki gdzie niższe = lepsze  
    lower_better = ['mae', 'rmse']
    
    if metric_name in higher_better:
        return "normal" if value > 0.7 else "inverse"
    elif metric_name in lower_better:
        return "inverse" if value < 100 else "normal"
    
    return None