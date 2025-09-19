# frontend/advanced_eda.py — Zaawansowane komponenty EDA z rozwijalnymi widokami
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas.api.types import is_datetime64_any_dtype  # <-- ważne dla bezpiecznego describe()


# -----------------------------
# USTAWIENIA GLOBALNE (lekko-konfigurowalne)
# -----------------------------
FAST_LIMIT_ROWS = 50_000       # ile wierszy zostawić w trybie szybkim
MISSING_HEATMAP_MAX_ROWS = 800 # próbkowanie wierszy na mapie braków
MISSING_HEATMAP_MAX_COLS = 120 # limit kolumn na mapie braków
TOP_CATEG_LEVELS = 50          # przycinanie liczby kategorii
TOP_CORR_PAIRS = 2000          # ile par korelacyjnych liczyć / wyświetlać maks.
TOP_CORR_SHOW = 30             # ile top korelacji pokazać w tabeli


# -----------------------------
# POMOCNICZE FUNKCJE (cache + sampling)
# -----------------------------
@st.cache_data(show_spinner=False)
def _describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bezpieczny opis DataFrame:
    - Na nowszych Pandas próbuje datetime_is_numeric=True.
    - Na starszych: każdą kolumnę datetime konwertuje do liczby ns od epoki (float, NaT->NaN),
      po czym wywołuje describe(include="all").
    """
    try:
        # Działa na Pandas 1.5+ / 2.x
        return df.copy().describe(include="all", datetime_is_numeric=True)
    except TypeError:
        # Starsze Pandas: rzutuj kolumny datetime na int64 ns -> float (NaT=>NaN)
        df2 = df.copy()
        for c in df2.columns:
            s = df2[c]
            try:
                if is_datetime64_any_dtype(s):
                    # Konwersja do UTC i ns
                    s2 = pd.to_datetime(s, errors="coerce", utc=True)
                    # view("int64") może dawać iNaT dla NaT; rzutujemy na float i zamieniamy iNaT->NaN
                    arr = s2.view("int64").astype("float64")
                    arr[arr == np.iinfo("int64").min] = np.nan
                    df2[c] = arr
            except Exception:
                # W razie wątpliwości zostaw kolumnę bez zmian
                pass
        return df2.describe(include="all")


@st.cache_data(show_spinner=False)
def _value_counts_head(s: pd.Series, top: int = TOP_CATEG_LEVELS) -> pd.DataFrame:
    """Zwraca top-n value_counts jako DataFrame (cache'owane)."""
    vc = s.value_counts(dropna=False)
    if len(vc) > top:
        vc = vc.head(top)
    return vc.to_frame("count").reset_index().rename(columns={"index": "value"})


@st.cache_data(show_spinner=False)
def _corr_matrix(numeric_df: pd.DataFrame) -> pd.DataFrame:
    """Macierz korelacji (cache'owana)."""
    # zabezpieczenie: usuń kolumny stałe (var=0) aby uniknąć NaN
    nunique = numeric_df.nunique(dropna=False)
    keep_cols = nunique.index[nunique > 1]
    safe_df = numeric_df[keep_cols]
    if safe_df.shape[1] < 2:
        return pd.DataFrame()
    return safe_df.corr()


def _maybe_sample(df: pd.DataFrame, fast_mode: bool, limit: int = FAST_LIMIT_ROWS) -> pd.DataFrame:
    """Jeśli fast_mode aktywny, próbkowanie wierszy dla płynności."""
    if not fast_mode:
        return df
    if len(df) > limit:
        return df.sample(limit, random_state=42)
    return df


def _safe_mode(series: pd.Series) -> str:
    """Zwraca najczęstszą wartość lub 'N/A' bez wyjątków."""
    try:
        m = series.mode(dropna=True)
        return str(m.iloc[0]) if len(m) > 0 else "N/A"
    except Exception:
        return "N/A"


def _first_non_null(series: pd.Series) -> str:
    """Zwraca pierwszy niepusty przykład lub 'N/A'."""
    try:
        drop = series.dropna()
        return str(drop.iloc[0]) if len(drop) > 0 else "N/A"
    except Exception:
        return "N/A"


class AdvancedEDAComponents:
    """
    Zaawansowane komponenty EDA z interaktywnymi wizualizacjami.
    Automatycznie dostosowuje się do typu danych i problemu.
    """

    def __init__(self):
        self.color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
            '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43'
        ]

    def render_comprehensive_eda(self, df: pd.DataFrame, target_col: Optional[str] = None) -> None:
        """Renderuje kompletny pakiet EDA z rozwijalnymi sekcjami."""
        if df is None or df.empty:
            st.warning("Brak danych do analizy EDA.")
            return

        st.subheader("📊 Zaawansowana Analiza Eksploracyjna (EDA)")

        # Tryb szybki (sampling)
        fast_mode = st.toggle(
            "⚡ Tryb szybki (sampling i ograniczenia dla dużych plików)",
            value=True,
            help=f"Próbkowanie do ~{FAST_LIMIT_ROWS:,} wierszy, limity kategorii i korelacji dla płynności."
        )
        df_view = _maybe_sample(df, fast_mode, FAST_LIMIT_ROWS)

        # Szybkie statystyki w jednej linii
        self._render_quick_stats_row(df_view)

        # Rozwijalne sekcje EDA
        with st.expander("🔍 Profil danych i jakość", expanded=False):
            self._render_data_quality_profile(df_view, fast_mode)

        with st.expander("📈 Rozkłady i histogramy", expanded=False):
            self._render_distributions(df_view, target_col, fast_mode)

        with st.expander("🌐 Macierz korelacji", expanded=False):
            self._render_correlation_analysis(df_view, target_col, fast_mode)

        with st.expander("📊 Analiza kategorii", expanded=False):
            self._render_categorical_analysis(df_view, target_col, fast_mode)

        with st.expander("🎯 Analiza targetu", expanded=bool(target_col)):
            if target_col and target_col in df_view.columns:
                self._render_target_analysis(df_view, target_col)
            else:
                st.info("Wybierz target aby zobaczyć analizę.")

        with st.expander("🔗 Interakcje między cechami", expanded=False):
            self._render_feature_interactions(df_view, target_col, fast_mode)

        with st.expander("⚠️ Anomalie i wartości odstające", expanded=False):
            self._render_outlier_detection(df_view, fast_mode)

    def _render_quick_stats_row(self, df: pd.DataFrame) -> None:
        """Renderuje szybkie statystyki w jednym wierszu."""
        cols = st.columns(6)

        with cols[0]:
            st.metric("Wiersze", f"{len(df):,}")
        with cols[1]:
            st.metric("Kolumny", f"{len(df.columns):,}")
        with cols[2]:
            numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
            st.metric("Numeryczne", numeric_cols)
        with cols[3]:
            categorical_cols = df.select_dtypes(include=['object', 'category']).shape[1]
            st.metric("Kategoryczne", categorical_cols)
        with cols[4]:
            missing_cells = int(df.isna().sum().sum())
            st.metric("Braki", f"{missing_cells:,}")
        with cols[5]:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Pamięć", f"{memory_mb:.1f} MB")

    def _render_data_quality_profile(self, df: pd.DataFrame, fast_mode: bool) -> None:
        """Renderuje szczegółowy profil jakości danych."""
        st.write("### Profil jakości danych")

        # Profil kolumn
        profile_data = []
        for col in df.columns:
            series = df[col]
            profile_data.append({
                'Kolumna': col,
                'Typ': str(series.dtype),
                'Braki': int(series.isna().sum()),
                'Braki %': f"{series.isna().mean() * 100:.1f}%",
                'Unikalne': int(series.nunique(dropna=True)),
                'Unikalne %': f"{(series.nunique(dropna=True) / max(len(series), 1)) * 100:.1f}%",
                'Najczęstsza': _safe_mode(series),
                'Przykład': _first_non_null(series)
            })

        profile_df = pd.DataFrame(profile_data)
        st.dataframe(profile_df, use_container_width=True, hide_index=True)

        # Mapa braków danych (opcjonalna, próbkowana)
        if df.isna().any().any():
            show_heatmap = st.checkbox(
                "Pokaż mapę braków danych (próbkowaną dla płynności)",
                value=False,
                help="Dla bardzo szerokich lub długich danych wizualizacja może być ciężka."
            )
            if show_heatmap:
                missing_data = df.isna()

                # Przycinanie kolumn/wierszy
                cols = list(missing_data.columns)
                if len(cols) > MISSING_HEATMAP_MAX_COLS:
                    cols = cols[:MISSING_HEATMAP_MAX_COLS]
                md_cols = missing_data[cols]

                if len(md_cols) > MISSING_HEATMAP_MAX_ROWS:
                    md_cols = md_cols.sample(MISSING_HEATMAP_MAX_ROWS, random_state=42)

                fig = go.Figure(data=go.Heatmap(
                    z=md_cols.astype(int).values,
                    x=list(md_cols.columns),
                    y=[f"Row {i}" for i in md_cols.index],
                    colorscale=[[0, '#f0f0f0'], [1, '#ff4444']],
                    showscale=False
                ))

                fig.update_layout(
                    title="Mapa braków danych (czerwone = brak) – próbkowana",
                    height=400,
                    xaxis_title="Kolumny",
                    yaxis_title="Wiersze (próbka)"
                )

                st.plotly_chart(fig, use_container_width=True)

    def _render_distributions(self, df: pd.DataFrame, target_col: Optional[str], fast_mode: bool) -> None:
        """Renderuje rozkłady zmiennych numerycznych i kategorycznych."""
        st.write("### Rozkłady zmiennych")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Usuń target z list jeśli jest wybrany
        if target_col:
            numeric_cols = [col for col in numeric_cols if col != target_col]
            categorical_cols = [col for col in categorical_cols if col != target_col]

        tab1, tab2 = st.tabs(["📊 Numeryczne", "📋 Kategoryczne"])

        with tab1:
            if numeric_cols:
                # Wybór kolumn do analizy
                default_num = numeric_cols[: min(4, len(numeric_cols))]
                selected_numeric = st.multiselect(
                    "Wybierz kolumny numeryczne:",
                    numeric_cols,
                    default=default_num
                )

                if selected_numeric:
                    # Histogramy
                    n_cols = min(2, len(selected_numeric))
                    n_rows = (len(selected_numeric) + n_cols - 1) // n_cols

                    fig = make_subplots(
                        rows=n_rows, cols=n_cols,
                        subplot_titles=selected_numeric
                    )

                    for i, col in enumerate(selected_numeric):
                        row = i // n_cols + 1
                        col_pos = i % n_cols + 1

                        fig.add_trace(
                            go.Histogram(
                                x=df[col].dropna(),
                                name=col,
                                marker_color=self.color_palette[i % len(self.color_palette)]
                            ),
                            row=row, col=col_pos
                        )

                    fig.update_layout(
                        title="Rozkłady zmiennych numerycznych",
                        height=300 * n_rows,
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Statystyki opisowe
                    st.write("#### Statystyki opisowe")
                    try:
                        desc_stats = _describe_df(df[selected_numeric])[selected_numeric]
                    except Exception:
                        desc_stats = _describe_df(df[selected_numeric])
                    st.dataframe(desc_stats, use_container_width=True)
            else:
                st.info("Brak kolumn numerycznych do analizy.")

        with tab2:
            if categorical_cols:
                selected_categorical = st.selectbox(
                    "Wybierz kolumnę kategoryczną:",
                    categorical_cols
                )

                if selected_categorical:
                    # Bezpieczne value_counts: obsługa NaN, None, mieszanych typów i bardzo długich etykiet
                    s = df[selected_categorical]
                    s_display = s.fillna("(NaN)").astype(str)

                    # top-N, ale z możliwością rozszerzenia
                    top_n = st.slider("Ile najczęstszych kategorii pokazać", 5, 50, 20)
                    vc = s_display.value_counts(dropna=False).head(top_n).reset_index()
                    # Gwarantowane nazwy kolumn:
                    vc.columns = ["label", "count"]

                    # wykres (poziomy bar) – odporny na egzotyczne etykiety
                    fig = px.bar(
                        vc,
                        x="count",
                        y="label",
                        orientation="h",
                        title=f"Rozkład: {selected_categorical}",
                    )
                    fig.update_layout(
                        yaxis_title=selected_categorical,
                        xaxis_title="Liczba wystąpień",
                        height=max(400, len(vc) * 25),
                        margin=dict(l=10, r=10, t=60, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # metryki dla wybranej kolumny
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Unikalne kategorie", int(s.nunique(dropna=True)))
                        st.metric("Najczęstsza", str(vc.iloc[0]["label"]) if len(vc) else "—")
                    with col2:
                        st.metric("Częstość najczęstszej", f"{int(vc.iloc[0]['count']):,}" if len(vc) else "—")
                        missing_pct = s.isna().mean() * 100
                        st.metric("Braki", f"{missing_pct:.1f}%")
            else:
                st.info("Brak kolumn kategorycznych do analizy.")

    def _render_correlation_analysis(self, df: pd.DataFrame, target_col: Optional[str], fast_mode: bool) -> None:
        """Renderuje analizę korelacji."""
        st.write("### Analiza korelacji")

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            st.info("Za mało kolumn numerycznych do analizy korelacji.")
            return

        # Macierz korelacji (cache)
        corr_matrix = _corr_matrix(numeric_df)
        if corr_matrix.empty or corr_matrix.shape[1] < 2:
            st.info("Brak wystarczającej zmienności do policzenia korelacji.")
            return

        # Heatmapa (przycinana dla bardzo szerokich danych)
        cols = list(corr_matrix.columns)
        max_cols = 80 if fast_mode else 160
        if len(cols) > max_cols:
            cols = cols[:max_cols]
        cm_small = corr_matrix.loc[cols, cols]

        fig = go.Figure(data=go.Heatmap(
            z=cm_small.values,
            x=cm_small.columns,
            y=cm_small.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(cm_small.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title=f"Macierz korelacji (pierwsze {len(cols)} kolumn)",
            height=600,
            width=700
        )

        st.plotly_chart(fig, use_container_width=True)

        # Korelacje z targetem (jeśli numeryczny)
        if target_col and target_col in corr_matrix.columns:
            st.write(f"#### Korelacje z targetem: {target_col}")
            target_corrs = corr_matrix[target_col].drop(target_col, errors="ignore").dropna()
            target_corrs = target_corrs.abs().sort_values(ascending=False)

            top_show = min(30, len(target_corrs))
            if top_show > 0:
                fig_bar = px.bar(
                    x=target_corrs.values[:top_show],
                    y=target_corrs.index[:top_show],
                    orientation='h',
                    title=f"Korelacje z {target_col} (top {top_show})",
                    color=target_corrs.values[:top_show],
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Brak istotnych korelacji z targetem.")

        # Tabela najsilniejszych korelacji (wszystkie pary – limitowana)
        st.write("#### Najsilniejsze korelacje (sparowane)")
        # Bezpieczne tworzenie listy par (limitowanie liczby kolumn dla złożoności O(n^2))
        cols_for_pairs = list(corr_matrix.columns)
        # twardy limit kolumn, żeby nie liczyć dziesiątek milionów par
        max_cols_pairs = 200 if fast_mode else 400
        if len(cols_for_pairs) > max_cols_pairs:
            cols_for_pairs = cols_for_pairs[:max_cols_pairs]
        cm_pairs = corr_matrix.loc[cols_for_pairs, cols_for_pairs]

        corr_pairs = []
        ncols = len(cm_pairs.columns)
        # zlicz maksymalnie TOP_CORR_PAIRS par (dla wydajności)
        counted = 0
        for i in range(ncols):
            for j in range(i + 1, ncols):
                corr_val = cm_pairs.iloc[i, j]
                if pd.notna(corr_val):
                    corr_pairs.append({
                        'Kolumna 1': cm_pairs.columns[i],
                        'Kolumna 2': cm_pairs.columns[j],
                        'Korelacja': corr_val
                    })
                    counted += 1
                    if counted >= TOP_CORR_PAIRS:
                        break
            if counted >= TOP_CORR_PAIRS:
                break

        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.reindex(corr_df['Korelacja'].abs().sort_values(ascending=False).index)
            st.dataframe(corr_df.head(TOP_CORR_SHOW), use_container_width=True, hide_index=True)
            if counted >= TOP_CORR_PAIRS:
                st.caption(f"Pokazano top {TOP_CORR_SHOW} z ograniczonych {TOP_CORR_PAIRS} par (dla wydajności).")
        else:
            st.info("Nie udało się wyznaczyć znaczących par korelacyjnych.")

    def _render_categorical_analysis(self, df: pd.DataFrame, target_col: Optional[str], fast_mode: bool) -> None:
        """Renderuje analizę zmiennych kategorycznych."""
        st.write("### Analiza zmiennych kategorycznych")

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_col and target_col in categorical_cols:
            categorical_cols.remove(target_col)

        if not categorical_cols:
            st.info("Brak zmiennych kategorycznych do analizy.")
            return

        selected_cat = st.selectbox("Wybierz zmienną kategoryczną:", categorical_cols)

        if selected_cat:
            # Podstawowe statystyki
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unikalne kategorie", int(df[selected_cat].nunique()))
            with col2:
                st.metric("Najczęstsza", _safe_mode(df[selected_cat]))
            with col3:
                missing_pct = df[selected_cat].isna().mean() * 100
                st.metric("Braki", f"{missing_pct:.1f}%")

            # Wykres rozkładu (przycięty do TOP_CATEG_LEVELS)
            vc_df = _value_counts_head(df[selected_cat], TOP_CATEG_LEVELS)
            if vc_df.empty:
                st.info("Brak danych do wizualizacji rozkładu kategorii.")
            else:
                value_counts = vc_df.set_index("value")["count"]
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Rozkład kategorii: {selected_cat}"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Analiza względem targetu (jeśli target istnieje)
            if target_col and target_col in df.columns:
                st.write(f"#### Analiza {selected_cat} vs {target_col}")

                # Jeśli target ma zbyt wiele poziomów, przytnij do TOP_CATEG_LEVELS
                tgt = df[target_col].astype("category") if df[target_col].dtype == "O" else df[target_col]
                if hasattr(tgt, "nunique") and tgt.nunique() > TOP_CATEG_LEVELS:
                    st.info(f"Target ma wiele kategorii – przycinam do top {TOP_CATEG_LEVELS}.")
                    top_tgt = tgt.value_counts().index[:TOP_CATEG_LEVELS]
                    crosstab = pd.crosstab(df[selected_cat], tgt.where(tgt.isin(top_tgt)), normalize='index') * 100
                    crosstab = crosstab.fillna(0.0)
                else:
                    crosstab = pd.crosstab(df[selected_cat], tgt, normalize='index') * 100

                # Dla nadmiaru kategorii po stronie selected_cat – przytnij do TOP_CATEG_LEVELS
                if crosstab.shape[0] > TOP_CATEG_LEVELS:
                    st.info(f"Kolumna {selected_cat} ma wiele poziomów – pokazuję top {TOP_CATEG_LEVELS} wg supportu.")
                    keep_rows = df[selected_cat].value_counts().index[:TOP_CATEG_LEVELS]
                    crosstab = crosstab.loc[crosstab.index.intersection(keep_rows)]

                if crosstab.empty:
                    st.info("Brak danych do wykresu rozkładu względem targetu.")
                else:
                    fig_stack = go.Figure()
                    for col in crosstab.columns:
                        fig_stack.add_trace(go.Bar(
                            name=str(col),
                            x=crosstab.index.astype(str),
                            y=crosstab[col],
                        ))

                    fig_stack.update_layout(
                        barmode='stack',
                        title=f"Rozkład {target_col} w grupach {selected_cat}",
                        yaxis_title="Procent",
                        xaxis_title=selected_cat
                    )

                    st.plotly_chart(fig_stack, use_container_width=True)

    def _render_target_analysis(self, df: pd.DataFrame, target_col: str) -> None:
        """Renderuje szczegółową analizę targetu."""
        st.write(f"### Analiza targetu: {target_col}")

        target_series = df[target_col]

        # Podstawowe info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Typ", str(target_series.dtype))
        with col2:
            st.metric("Unikalne", int(target_series.nunique()))
        with col3:
            st.metric("Braki", f"{int(target_series.isna().sum()):,}")
        with col4:
            st.metric("Braki %", f"{target_series.isna().mean()*100:.1f}%")

        # Analiza w zależności od typu
        if pd.api.types.is_numeric_dtype(target_series):
            self._render_numeric_target_analysis(target_series)
        else:
            self._render_categorical_target_analysis(target_series)

    def _render_numeric_target_analysis(self, target_series: pd.Series) -> None:
        """Analiza numerycznego targetu."""
        # Histogram
        if target_series.dropna().empty:
            st.info("Brak danych do wykresu histogramu.")
        else:
            fig = px.histogram(
                target_series.dropna(),
                title=f"Rozkład {target_series.name}",
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)

        # Box plot
        if target_series.dropna().empty:
            st.info("Brak danych do wykresu pudełkowego.")
        else:
            fig_box = px.box(
                y=target_series.dropna(),
                title=f"Box plot - {target_series.name}"
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Statystyki
        stats = target_series.describe()
        st.dataframe(stats.to_frame().T, use_container_width=True)

    def _render_categorical_target_analysis(self, target_series: pd.Series) -> None:
        """Analiza kategorycznego targetu."""
        value_counts = target_series.value_counts()

        if value_counts.empty:
            st.info("Brak danych do analizy rozkładu klas.")
            return

        # Bar chart
        fig = px.bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            title=f"Rozkład klas - {target_series.name}"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabela z liczebnościami
        st.dataframe(value_counts.to_frame('Liczebność'), use_container_width=True)

        # Sprawdź balans klas
        if len(value_counts) > 1 and value_counts.min() > 0:
            imbalance_ratio = value_counts.max() / value_counts.min()
            if imbalance_ratio > 3:
                st.warning(f"⚠️ Wykryto niebalans klas (ratio: {imbalance_ratio:.1f}:1)")
                st.info("💡 Rozważ techniki balansowania klas przed treningiem.")

    def _render_feature_interactions(self, df: pd.DataFrame, target_col: Optional[str], fast_mode: bool) -> None:
        """Renderuje analizę interakcji między cechami."""
        st.write("### Interakcje między cechami")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)

        if len(numeric_cols) < 2:
            st.info("Za mało cech numerycznych do analizy interakcji.")
            return

        col1, col2 = st.columns(2)
        with col1:
            feature1 = st.selectbox("Cecha 1:", numeric_cols, key="feat1")
        with col2:
            feature2 = st.selectbox(
                "Cecha 2:",
                [col for col in numeric_cols if col != feature1],
                key="feat2"
            )

        if feature1 and feature2:
            # Scatter plot
            if target_col and target_col in df.columns:
                fig = px.scatter(
                    df, x=feature1, y=feature2, color=target_col,
                    title=f"Interakcja: {feature1} vs {feature2} (kolor: {target_col})"
                )
            else:
                fig = px.scatter(
                    df, x=feature1, y=feature2,
                    title=f"Interakcja: {feature1} vs {feature2}"
                )

            st.plotly_chart(fig, use_container_width=True)

            # Korelacja między wybranymi cechami
            try:
                correlation = df[feature1].corr(df[feature2])
                st.metric("Korelacja między wybranymi cechami", f"{correlation:.3f}")
            except Exception:
                st.metric("Korelacja między wybranymi cechami", "N/A")

    def _render_outlier_detection(self, df: pd.DataFrame, fast_mode: bool) -> None:
        """Renderuje detekcję wartości odstających."""
        st.write("### Detekcja wartości odstających")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            st.info("Brak kolumn numerycznych do analizy.")
            return

        selected_col = st.selectbox("Wybierz kolumnę do analizy:", numeric_cols)

        if selected_col:
            series = df[selected_col].dropna()

            if len(series) == 0:
                st.info("Brak danych w wybranej kolumnie.")
                return

            # Metoda IQR
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = series[(series < lower_bound) | (series > upper_bound)]

            st.metric("Liczba wartości odstających (IQR)", len(outliers))
            st.metric("% wartości odstających", f"{len(outliers)/len(series)*100:.1f}%")

            # Box plot z oznaczonymi outlierami
            fig = px.box(y=series, title=f"Box plot z outlierami - {selected_col}")
            st.plotly_chart(fig, use_container_width=True)

            if len(outliers) > 0:
                st.write("#### Próbka wartości odstających:")
                st.write(outliers.head(10).values)


def render_eda_section(df: pd.DataFrame, target_col: Optional[str] = None) -> None:
    """Główna funkcja renderująca EDA - do użycia w app.py"""
    eda = AdvancedEDAComponents()
    eda.render_comprehensive_eda(df, target_col)
