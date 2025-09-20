from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# USTAWIENIA GLOBALNE
# -----------------------------
FAST_LIMIT_ROWS = 50_000       
MISSING_HEATMAP_MAX_ROWS = 800 
MISSING_HEATMAP_MAX_COLS = 120 
TOP_CATEG_LEVELS = 50          
TOP_CORR_PAIRS = 2000          
TOP_CORR_SHOW = 30             

# -----------------------------
# CACHE FUNCTIONS
# -----------------------------
@st.cache_data(show_spinner=False)
def _describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Bezpieczny opis DataFrame z obsługą datetime."""
    try:
        return df.copy().describe(include="all", datetime_is_numeric=True)
    except TypeError:
        df2 = df.copy()
        for c in df2.columns:
            s = df2[c]
            try:
                if is_datetime64_any_dtype(s):
                    s2 = pd.to_datetime(s, errors="coerce", utc=True)
                    arr = s2.view("int64").astype("float64")
                    arr[arr == np.iinfo("int64").min] = np.nan
                    df2[c] = arr
            except Exception:
                pass
        return df2.describe(include="all")

@st.cache_data(show_spinner=False)
def _value_counts_head(s: pd.Series, top: int = TOP_CATEG_LEVELS) -> pd.DataFrame:
    """Zwraca top-n value_counts jako DataFrame."""
    vc = s.value_counts(dropna=False)
    if len(vc) > top:
        vc = vc.head(top)
    return vc.to_frame("count").reset_index().rename(columns={"index": "value"})

@st.cache_data(show_spinner=False)
def _corr_matrix(numeric_df: pd.DataFrame) -> pd.DataFrame:
    """Macierz korelacji (cache'owana)."""
    nunique = numeric_df.nunique(dropna=False)
    keep_cols = nunique.index[nunique > 1]
    safe_df = numeric_df[keep_cols]
    if safe_df.shape[1] < 2:
        return pd.DataFrame()
    return safe_df.corr()

@st.cache_data(show_spinner=False)
def _compute_pca(df: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Oblicza PCA dla danych numerycznych."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return np.array([]), np.array([])
    
    # Wypełnij braki medianą
    filled_df = numeric_df.fillna(numeric_df.median())
    
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(filled_df)
    explained_variance = pca.explained_variance_ratio_
    
    return components, explained_variance

@st.cache_data(show_spinner=False)
def _detect_outliers_iqr(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Wykrywa outliers metodą IQR dla kolumn numerycznych."""
    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
            
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        if outlier_mask.sum() > 0:
            outliers[col] = df[col][outlier_mask]
    
    return outliers

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def _maybe_sample(df: pd.DataFrame, fast_mode: bool, limit: int = FAST_LIMIT_ROWS) -> pd.DataFrame:
    """Próbkowanie dla szybkości."""
    if not fast_mode:
        return df
    if len(df) > limit:
        return df.sample(limit, random_state=42)
    return df

def _safe_mode(series: pd.Series) -> str:
    """Zwraca najczęstszą wartość lub 'N/A'."""
    try:
        m = series.mode(dropna=True)
        return str(m.iloc[0]) if len(m) > 0 else "N/A"
    except Exception:
        return "N/A"

def _first_non_null(series: pd.Series) -> str:
    """Zwraca pierwszy niepusty przykład."""
    try:
        drop = series.dropna()
        return str(drop.iloc[0]) if len(drop) > 0 else "N/A"
    except Exception:
        return "N/A"

def _get_color_palette():
    """Zwraca paletę kolorów z session_state."""
    theme = st.session_state.get('tmiv_color_theme', 'default')
    palette_map = {
        'default': px.colors.qualitative.Set1,
        'viridis': px.colors.sequential.Viridis,
        'plasma': px.colors.sequential.Plasma,
        'blues': px.colors.sequential.Blues,
        'reds': px.colors.sequential.Reds,
        'greens': px.colors.sequential.Greens
    }
    return palette_map.get(theme, px.colors.qualitative.Set1)


class AdvancedEDAComponents:
    """Zaawansowane komponenty EDA z nowymi funkcjami analitycznymi."""
    
    def __init__(self):
        self.color_palette = _get_color_palette()

    def render_comprehensive_eda(self, df: pd.DataFrame, target_col: Optional[str] = None) -> None:
        if df is None or df.empty:
            st.warning("Brak danych do analizy EDA.")
            return

        st.subheader("📊 Zaawansowana Analiza Eksploracyjna (EDA)")
        st.markdown("**Komprehensywna analiza danych z wizualizacjami i statystykami opisowymi**")

        # Tryb szybki
        fast_mode = st.toggle(
            "⚡ Tryb szybki (optymalizacja wydajności)",
            value=True,
            help=f"Próbkowanie do {FAST_LIMIT_ROWS:,} wierszy i ograniczenia dla płynności"
        )
        df_view = _maybe_sample(df, fast_mode, FAST_LIMIT_ROWS)

        if fast_mode and len(df) > FAST_LIMIT_ROWS:
            st.info(f"🔬 Analiza oparta na próbce {len(df_view):,} z {len(df):,} wierszy")

        # Szybkie statystyki
        self._render_quick_stats_row(df_view)

        # ROZBUDOWANE SEKCJE EDA
        st.markdown("---")
        
        # Organizacja w zakładkach - NOWE ZAKŁADKI
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "🔍 Profil danych",
            "📈 Rozkłady", 
            "🌐 Korelacje",
            "📊 Kategorie",
            "🎯 Target",
            "🔗 Interakcje",
            "⚠️ Anomalie",
            "🧬 Redukcja wymiarów",  # NOWA
            "📋 Raport jakości"       # NOWA
        ])

        with tab1:
            self._render_data_quality_profile(df_view, fast_mode)

        with tab2:
            self._render_distributions(df_view, target_col, fast_mode)

        with tab3:
            self._render_correlation_analysis(df_view, target_col, fast_mode)

        with tab4:
            self._render_categorical_analysis(df_view, target_col, fast_mode)

        with tab5:
            if target_col and target_col in df_view.columns:
                self._render_target_analysis(df_view, target_col)
            else:
                st.info("Wybierz target aby zobaczyć analizę.")

        with tab6:
            self._render_feature_interactions(df_view, target_col, fast_mode)

        with tab7:
            self._render_outlier_detection(df_view, fast_mode)

        with tab8:  # NOWA ZAKŁADKA
            self._render_dimensionality_reduction(df_view, target_col, fast_mode)

        with tab9:  # NOWA ZAKŁADKA  
            self._render_data_quality_report(df_view, target_col)

    def _render_quick_stats_row(self, df: pd.DataFrame) -> None:
        """Rozbudowany rząd szybkich statystyk."""
        cols = st.columns(7)  # Dodana jedna kolumna
        
        with cols[0]:
            st.metric("Wiersze", f"{len(df):,}")
        with cols[1]:
            st.metric("Kolumny", f"{len(df.columns):,}")
        with cols[2]:
            numeric_count = df.select_dtypes(include=[np.number]).shape[1]
            st.metric("Numeryczne", numeric_count)
        with cols[3]:
            cat_count = df.select_dtypes(include=['object', 'category']).shape[1]
            st.metric("Kategoryczne", cat_count)
        with cols[4]:
            missing_count = int(df.isna().sum().sum())
            st.metric("Braki", f"{missing_count:,}")
        with cols[5]:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Pamięć", f"{memory_mb:.1f} MB")
        with cols[6]:  # NOWA METRYKA
            duplicate_count = df.duplicated().sum()
            st.metric("Duplikaty", f"{duplicate_count:,}")

    def _render_data_quality_profile(self, df: pd.DataFrame, fast_mode: bool) -> None:
        """Rozbudowany profil jakości danych."""
        st.write("### 🔍 Szczegółowy profil danych")
        
        # Profil kolumn z dodatkowymi metrykami
        profile_data = []
        for col in df.columns:
            series = df[col]
            dtype_info = str(series.dtype)
            
            # Dodatkowe analizy
            memory_usage = series.memory_usage(deep=True) / 1024  # KB
            
            if pd.api.types.is_numeric_dtype(series):
                try:
                    skewness = series.skew()
                    kurtosis = series.kurtosis()
                    additional_info = f"Skew: {skewness:.2f}, Kurt: {kurtosis:.2f}"
                except:
                    additional_info = "N/A"
            else:
                additional_info = f"Najdłuższy: {series.astype(str).str.len().max()} znaków"
            
            profile_data.append({
                'Kolumna': col,
                'Typ': dtype_info,
                'Braki': int(series.isna().sum()),
                'Braki %': f"{series.isna().mean() * 100:.1f}%",
                'Unikalne': int(series.nunique(dropna=True)),
                'Unikalność %': f"{(series.nunique(dropna=True) / max(len(series), 1)) * 100:.1f}%",
                'Pamięć (KB)': f"{memory_usage:.1f}",
                'Najczęstsza': _safe_mode(series),
                'Info dodatkowe': additional_info,
                'Przykład': _first_non_null(series)
            })
        
        profile_df = pd.DataFrame(profile_data)
        st.dataframe(profile_df, use_container_width=True, hide_index=True)

        # Analiza braków danych - ROZBUDOWANA
        if df.isna().any().any():
            st.write("### 🕳️ Analiza braków danych")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Statystyki braków
                missing_stats = df.isna().sum().sort_values(ascending=False)
                missing_stats = missing_stats[missing_stats > 0]
                
                if not missing_stats.empty:
                    fig_missing = px.bar(
                        x=missing_stats.values,
                        y=missing_stats.index,
                        orientation='h',
                        title="Braki danych według kolumn",
                        color=missing_stats.values,
                        color_continuous_scale='Reds'
                    )
                    fig_missing.update_layout(height=max(300, len(missing_stats) * 25))
                    st.plotly_chart(fig_missing, use_container_width=True)
            
            with col2:
                # Wzorce braków
                if st.checkbox("Pokaż wzorce braków danych", help="Analiza kombinacji braków"):
                    missing_combinations = df.isna().groupby(list(df.columns)).size().sort_values(ascending=False)
                    if len(missing_combinations) > 1:
                        st.write("**Top kombinacje braków:**")
                        for pattern, count in missing_combinations.head(5).items():
                            missing_cols = [col for col, is_missing in zip(df.columns, pattern) if is_missing]
                            if missing_cols:
                                st.write(f"- {count:,} wierszy: {', '.join(missing_cols)}")
            
            # Heatmapa braków - ulepszona
            show_heatmap = st.checkbox("Pokaż heatmapę braków danych")
            if show_heatmap:
                missing_data = df.isna()
                cols = list(missing_data.columns)
                if len(cols) > MISSING_HEATMAP_MAX_COLS:
                    cols = cols[:MISSING_HEATMAP_MAX_COLS]
                    st.info(f"Pokazuję pierwsze {MISSING_HEATMAP_MAX_COLS} kolumn")
                
                md_cols = missing_data[cols]
                if len(md_cols) > MISSING_HEATMAP_MAX_ROWS:
                    md_cols = md_cols.sample(MISSING_HEATMAP_MAX_ROWS, random_state=42)

                fig = go.Figure(data=go.Heatmap(
                    z=md_cols.astype(int).values,
                    x=list(md_cols.columns),
                    y=[f"Row {i}" for i in md_cols.index],
                    colorscale=[[0, '#f0f0f0'], [1, '#ff4444']],
                    showscale=True,
                    colorbar=dict(title="Brak danych")
                ))
                fig.update_layout(
                    title="Mapa braków danych (czerwone = brak)",
                    height=400,
                    xaxis_title="Kolumny",
                    yaxis_title="Wiersze"
                )
                st.plotly_chart(fig, use_container_width=True)

    def _render_dimensionality_reduction(self, df: pd.DataFrame, target_col: Optional[str], fast_mode: bool) -> None:
        """NOWA: Analiza redukcji wymiarów."""
        st.write("### 🧬 Redukcja wymiarów i analiza skupień")
        st.markdown("**Wizualizacja danych w przestrzeni o obniżonej wymiarowości**")
        
        # Sprawdź czy są dane numeryczne
        numeric_df = df.select_dtypes(include=[np.number])
        if target_col and target_col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[target_col])
        
        if numeric_df.shape[1] < 2:
            st.warning("Za mało cech numerycznych do analizy redukcji wymiarów (wymagane minimum 2).")
            return
        
        # Opcje analizy
        reduction_method = st.selectbox(
            "Wybierz metodę redukcji:",
            ["PCA", "t-SNE"],
            help="PCA - liniowa, szybka; t-SNE - nieliniowa, wolniejsza ale lepiej pokazuje klastry"
        )
        
        col1, col2 = st.columns(2)
        
        # Parametry
        with col1:
            if reduction_method == "PCA":
                n_components = st.slider("Liczba komponentów PCA:", 2, min(5, numeric_df.shape[1]), 2)
            else:
                perplexity = st.slider("Perplexity (t-SNE):", 5, min(50, len(df)//4), 30)
        
        with col2:
            show_clusters = st.checkbox("Pokaż klastry (K-means)", value=False)
            if show_clusters:
                n_clusters = st.slider("Liczba klastrów:", 2, 10, 3)
        
        # Oblicz redukcję wymiarów
        try:
            if reduction_method == "PCA":
                components, explained_var = _compute_pca(numeric_df, n_components)
                
                if components.size == 0:
                    st.error("Nie udało się obliczyć PCA")
                    return
                
                # Wykres PCA
                if n_components == 2:
                    fig = go.Figure()
                    
                    # Kolorowanie według targetu jeśli dostępny
                    if target_col and target_col in df.columns:
                        target_values = df[target_col]
                        fig.add_trace(go.Scatter(
                            x=components[:, 0],
                            y=components[:, 1],
                            mode='markers',
                            marker=dict(
                                color=target_values,
                                colorscale='viridis',
                                showscale=True,
                                colorbar=dict(title=target_col)
                            ),
                            text=target_values,
                            name='Data points'
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=components[:, 0],
                            y=components[:, 1],
                            mode='markers',
                            marker=dict(color='blue', opacity=0.6),
                            name='Data points'
                        ))
                    
                    # Dodaj klastry jeśli wybrane
                    if show_clusters:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        cluster_labels = kmeans.fit_predict(components)
                        
                        fig.add_trace(go.Scatter(
                            x=components[:, 0],
                            y=components[:, 1],
                            mode='markers',
                            marker=dict(
                                color=cluster_labels,
                                colorscale='Set1',
                                symbol='circle-open',
                                size=10,
                                line=dict(width=2)
                            ),
                            name='Klastry',
                            showlegend=False
                        ))
                        
                        # Centra klastrów
                        centers = kmeans.cluster_centers_
                        fig.add_trace(go.Scatter(
                            x=centers[:, 0],
                            y=centers[:, 1],
                            mode='markers',
                            marker=dict(
                                color='red',
                                size=15,
                                symbol='x',
                                line=dict(width=3)
                            ),
                            name='Centra klastrów'
                        ))
                    
                    fig.update_layout(
                        title=f"PCA - Analiza głównych składowych",
                        xaxis_title=f"PC1 ({explained_var[0]:.1%} wariancji)",
                        yaxis_title=f"PC2 ({explained_var[1]:.1%} wariancji)",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statystyki PCA
                    st.write("#### 📊 Statystyki PCA")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Wariancja PC1", f"{explained_var[0]:.1%}")
                    with col2:
                        st.metric("Wariancja PC2", f"{explained_var[1]:.1%}")
                    with col3:
                        total_var = explained_var[:2].sum()
                        st.metric("Całkowita wariancja", f"{total_var:.1%}")
                
                # Wykres wszystkich komponentów
                if n_components > 2:
                    st.write("#### 📈 Wariancja wyjaśniona przez komponenty")
                    fig_var = px.bar(
                        x=range(1, len(explained_var) + 1),
                        y=explained_var,
                        title="Wariancja wyjaśniona przez każdy komponent PCA"
                    )
                    fig_var.update_layout(
                        xaxis_title="Numer komponentu",
                        yaxis_title="Wariancja wyjaśniona"
                    )
                    st.plotly_chart(fig_var, use_container_width=True)
            
            else:  # t-SNE
                st.info("🔄 Obliczanie t-SNE... To może potrwać chwilę.")
                
                # Przygotowanie danych
                filled_df = numeric_df.fillna(numeric_df.median())
                
                # Próbkowanie dla t-SNE jeśli za duże
                if len(filled_df) > 5000:
                    filled_df = filled_df.sample(5000, random_state=42)
                    st.info(f"Użyto próbki 5000 wierszy dla t-SNE")
                
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
                components = tsne.fit_transform(filled_df)
                
                # Wykres t-SNE
                fig = go.Figure()
                
                if target_col and target_col in df.columns:
                    target_sample = df[target_col].iloc[:len(components)]
                    fig.add_trace(go.Scatter(
                        x=components[:, 0],
                        y=components[:, 1],
                        mode='markers',
                        marker=dict(
                            color=target_sample,
                            colorscale='viridis',
                            showscale=True,
                            colorbar=dict(title=target_col)
                        ),
                        text=target_sample,
                        name='Data points'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=components[:, 0],
                        y=components[:, 1],
                        mode='markers',
                        marker=dict(color='blue', opacity=0.6),
                        name='Data points'
                    ))
                
                fig.update_layout(
                    title=f"t-SNE - Nieliniowa redukcja wymiarów",
                    xaxis_title="t-SNE 1",
                    yaxis_title="t-SNE 2",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Błąd podczas redukcji wymiarów: {str(e)}")

    def _render_data_quality_report(self, df: pd.DataFrame, target_col: Optional[str]) -> None:
        """NOWA: Komprehensywny raport jakości danych."""
        st.write("### 📋 Raport jakości danych")
        st.markdown("**Automatyczna ocena gotowości danych do modelowania ML**")
        
        # Ogólna ocena jakości
        quality_score = self._calculate_quality_score(df, target_col)
        
        # Wizualny wskaźnik jakości
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if quality_score >= 80:
                st.success(f"🏆 **Wysoka jakość danych: {quality_score}/100**")
                quality_color = "green"
            elif quality_score >= 60:
                st.info(f"✅ **Dobra jakość danych: {quality_score}/100**")
                quality_color = "blue"
            elif quality_score >= 40:
                st.warning(f"⚠️ **Średnia jakość danych: {quality_score}/100**")
                quality_color = "orange"
            else:
                st.error(f"❌ **Niska jakość danych: {quality_score}/100**")
                quality_color = "red"
        
        # Szczegółowa analiza problemów
        st.write("#### 🔍 Szczegółowa analiza problemów")
        
        problems = []
        recommendations = []
        
        # 1. Braki danych
        missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 20:
            problems.append(f"Wysokie procent braków danych: {missing_pct:.1f}%")
            recommendations.append("Rozważ imputację lub usunięcie kolumn z wieloma brakami")
        elif missing_pct > 5:
            problems.append(f"Umiarkowany procent braków: {missing_pct:.1f}%")
            recommendations.append("Zastosuj strategię imputacji przed treningiem")
        
        # 2. Duplikaty
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            dup_pct = (duplicates / len(df)) * 100
            problems.append(f"Duplikaty: {duplicates:,} wierszy ({dup_pct:.1f}%)")
            recommendations.append("Usuń duplikaty aby uniknąć data leakage")
        
        # 3. Kolumny stałe
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            problems.append(f"Kolumny stałe: {len(constant_cols)} ({', '.join(constant_cols[:3])}{'...' if len(constant_cols) > 3 else ''})")
            recommendations.append("Usuń kolumny stałe - nie wnoszą informacji")
        
        # 4. Wysoka kardynalność
        high_card_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > len(df) * 0.5:
                high_card_cols.append(col)
        
        if high_card_cols:
            problems.append(f"Wysoka kardynalność: {len(high_card_cols)} kolumn")
            recommendations.append("Rozważ feature hashing lub target encoding dla kolumn tekstowych")
        
        # 5. Niebalans klas (jeśli target jest kategoryczny)
        if target_col and target_col in df.columns:
            if pd.api.types.is_categorical_dtype(df[target_col]) or df[target_col].dtype == 'object':
                value_counts = df[target_col].value_counts()
                if len(value_counts) > 1:
                    imbalance_ratio = value_counts.max() / value_counts.min()
                    if imbalance_ratio > 10:
                        problems.append(f"Silny niebalans klas: {imbalance_ratio:.1f}:1")
                        recommendations.append("Użyj technik balansowania klas (SMOTE, undersampling)")
                    elif imbalance_ratio > 3:
                        problems.append(f"Niebalans klas: {imbalance_ratio:.1f}:1")
                        recommendations.append("Monitoruj metryki dla każdej klasy oddzielnie")
        
        # Wyświetl problemy i rekomendacje
        if problems:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**⚠️ Wykryte problemy:**")
                for i, problem in enumerate(problems, 1):
                    st.write(f"{i}. {problem}")
            
            with col2:
                st.write("**💡 Rekomendacje:**")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
        else:
            st.success("✅ Nie wykryto poważnych problemów z jakością danych!")
        
        # Raport przydatności kolumn
        st.write("#### 📊 Przydatność kolumn do modelowania")
        
        utility_data = []
        for col in df.columns:
            if col == target_col:
                continue
                
            series = df[col]
            utility_score = self._calculate_column_utility(series)
            
            issues = []
            if series.isna().mean() > 0.5:
                issues.append("Dużo braków")
            if series.nunique() <= 1:
                issues.append("Stała wartość")
            if pd.api.types.is_object_dtype(series) and series.nunique() > len(df) * 0.5:
                issues.append("Wysoka kardynalność")
            
            utility_data.append({
                'Kolumna': col,
                'Ocena przydatności': utility_score,
                'Typ': str(series.dtype),
                'Unikalne': series.nunique(),
                'Braki %': f"{series.isna().mean() * 100:.1f}%",
                'Potencjalne problemy': ', '.join(issues) if issues else 'Brak'
            })
        
        utility_df = pd.DataFrame(utility_data)
        utility_df = utility_df.sort_values('Ocena przydatności', ascending=False)
        
        # Kolorowanie według przydatności
        def color_utility(val):
            if val >= 80:
                return 'background-color: lightgreen'
            elif val >= 60:
                return 'background-color: lightyellow'  
            elif val >= 40:
                return 'background-color: lightorange'
            else:
                return 'background-color: lightcoral'
        
        styled_df = utility_df.style.applymap(color_utility, subset=['Ocena przydatności'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Podsumowanie gotowości
        st.write("#### 🎯 Gotowość do treningu ML")
        
        ready_cols = len([col for col in utility_df['Kolumna'] if utility_df[utility_df['Kolumna'] == col]['Ocena przydatności'].iloc[0] >= 60])
        total_cols = len(utility_df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ogólna jakość", f"{quality_score}/100")
        with col2:
            st.metric("Przydatne kolumny", f"{ready_cols}/{total_cols}")
        with col3:
            st.metric("% gotowości", f"{(ready_cols/max(total_cols,1)*100):.0f}%")
        with col4:
            if quality_score >= 70 and ready_cols/max(total_cols,1) >= 0.6:
                st.success("✅ Gotowy")
            else:
                st.warning("⚠️ Wymaga poprawek")

    def _calculate_quality_score(self, df: pd.DataFrame, target_col: Optional[str]) -> int:
        """Oblicza ogólną ocenę jakości danych (0-100)."""
        score = 100
        
        # Kara za braki danych
        missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        score -= min(30, missing_pct)
        
        # Kara za duplikaty
        dup_pct = (df.duplicated().sum() / len(df)) * 100
        score -= min(20, dup_pct * 2)
        
        # Kara za kolumny stałe
        constant_cols = sum(1 for col in df.columns if df[col].nunique() <= 1)
        score -= min(15, constant_cols * 5)
        
        # Kara za zbyt małą próbkę
        if len(df) < 100:
            score -= 20
        elif len(df) < 1000:
            score -= 10
        
        # Bonus za dobrą proporcję cech/próbek
        feature_ratio = len(df.columns) / len(df)
        if feature_ratio < 0.1:
            score += 5
        elif feature_ratio > 0.5:
            score -= 10
        
        return max(0, min(100, int(score)))
    
    def _calculate_column_utility(self, series: pd.Series) -> int:
        """Oblicza przydatność kolumny do modelowania (0-100)."""
        score = 100
        
        # Kara za braki
        missing_pct = series.isna().mean() * 100
        score -= min(40, missing_pct)
        
        # Kara za stałość
        if series.nunique() <= 1:
            score = 0
        
        # Kara za bardzo wysoką kardynalność (dla tekstowych)
        if pd.api.types.is_object_dtype(series):
            unique_ratio = series.nunique() / len(series)
            if unique_ratio > 0.5:
                score -= 30
        
        # Bonus za rozsądną zmienność
        if pd.api.types.is_numeric_dtype(series):
            try:
                std = series.std()
                if pd.notna(std) and std > 0:
                    score += 5
            except:
                pass
        
        return max(0, min(100, int(score)))

    # Pozostałe metody bez zmian lub z drobnymi ulepszeniami...
    def _render_distributions(self, df: pd.DataFrame, target_col: Optional[str], fast_mode: bool) -> None:
        """Ulepszone rozkłady z dodatkowymi statystykami."""
        st.write("### 📈 Rozkłady zmiennych")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if target_col:
            numeric_cols = [c for c in numeric_cols if c != target_col]
            categorical_cols = [c for c in categorical_cols if c != target_col]

        tab1, tab2 = st.tabs(["📊 Numeryczne", "📋 Kategoryczne"])

        with tab1:
            if numeric_cols:
                default_num = numeric_cols[: min(4, len(numeric_cols))]
                selected_numeric = st.multiselect("Wybierz kolumny numeryczne:", numeric_cols, default=default_num)
                
                if selected_numeric:
                    # Opcje wizualizacji
                    col1, col2 = st.columns(2)
                    with col1:
                        chart_type = st.selectbox("Typ wykresu:", ["Histogram", "Box plot", "Violin plot"])
                    with col2:
                        show_stats = st.checkbox("Pokaż statystyki opisowe", value=True)
                    
                    # Wykresy
                    n_cols = min(2, len(selected_numeric))
                    n_rows = (len(selected_numeric) + n_cols - 1) // n_cols
                    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=selected_numeric)
                    
                    for i, col in enumerate(selected_numeric):
                        row = i // n_cols + 1
                        col_pos = i % n_cols + 1
                        
                        if chart_type == "Histogram":
                            fig.add_trace(
                                go.Histogram(
                                    x=df[col].dropna(),
                                    name=col,
                                    marker_color=self.color_palette[i % len(self.color_palette)]
                                ),
                                row=row, col=col_pos
                            )
                        elif chart_type == "Box plot":
                            fig.add_trace(
                                go.Box(
                                    y=df[col].dropna(),
                                    name=col,
                                    marker_color=self.color_palette[i % len(self.color_palette)]
                                ),
                                row=row, col=col_pos
                            )
                        else:  # Violin plot
                            fig.add_trace(
                                go.Violin(
                                    y=df[col].dropna(),
                                    name=col,
                                    fillcolor=self.color_palette[i % len(self.color_palette)],
                                    opacity=0.6
                                ),
                                row=row, col=col_pos
                            )
                    
                    fig.update_layout(title=f"Rozkłady zmiennych numerycznych ({chart_type})", height=300 * n_rows, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    if show_stats:
                        st.write("#### 📊 Statystyki opisowe")
                        try:
                            desc_stats = _describe_df(df[selected_numeric])
                            st.dataframe(desc_stats.round(4), use_container_width=True)
                        except Exception:
                            st.info("Nie można wyświetlić statystyk opisowych")
            else:
                st.info("Brak kolumn numerycznych do analizy.")

        with tab2:
            if categorical_cols:
                selected_categorical = st.selectbox("Wybierz kolumnę kategoryczną:", categorical_cols)
                if selected_categorical:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        s = df[selected_categorical]
                        s_display = s.fillna("(NaN)").astype(str)
                        top_n = st.slider("Liczba kategorii do pokazania:", 5, min(50, s.nunique()), 15)
                        vc = s_display.value_counts(dropna=False).head(top_n).reset_index()
                        vc.columns = ["label", "count"]

                        chart_type = st.selectbox("Typ wykresu:", ["Bar plot", "Pie chart", "Donut chart"])
                        
                        if chart_type == "Pie chart":
                            fig = px.pie(values=vc["count"], names=vc["label"], title=f"Rozkład: {selected_categorical}")
                        elif chart_type == "Donut chart":
                            fig = px.pie(values=vc["count"], names=vc["label"], title=f"Rozkład: {selected_categorical}", hole=0.4)
                        else:  # Bar plot
                            fig = px.bar(vc, x="count", y="label", orientation="h", title=f"Rozkład: {selected_categorical}")
                            fig.update_layout(height=max(400, len(vc) * 25))
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write("**📊 Statystyki:**")
                        st.metric("Unikalne kategorie", int(s.nunique(dropna=True)))
                        st.metric("Najczęstsza", str(vc.iloc[0]["label"]) if len(vc) else "—")
                        st.metric("Częstość najczęstszej", f"{int(vc.iloc[0]['count']):,}" if len(vc) else "—")
                        st.metric("Braki", f"{s.isna().mean() * 100:.1f}%")
                        
                        # Entropia (miara różnorodności)
                        if len(vc) > 1:
                            proportions = vc["count"] / vc["count"].sum()
                            entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
                            st.metric("Entropia", f"{entropy:.2f}")
            else:
                st.info("Brak kolumn kategorycznych do analizy.")

    def _render_correlation_analysis(self, df: pd.DataFrame, target_col: Optional[str], fast_mode: bool) -> None:
        """Rozbudowana analiza korelacji."""
        st.write("### 🌐 Analiza korelacji")

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            st.info("Za mało kolumn numerycznych do analizy korelacji (wymagane minimum 2).")
            return

        # Opcje analizy korelacji
        col1, col2 = st.columns(2)
        with col1:
            corr_method = st.selectbox("Metoda korelacji:", ["pearson", "spearman", "kendall"])
        with col2:
            min_corr = st.slider("Minimalna korelacja do pokazania:", 0.0, 1.0, 0.1, 0.05)

        corr_matrix = numeric_df.corr(method=corr_method)
        if corr_matrix.empty or corr_matrix.shape[1] < 2:
            st.info("Brak wystarczającej zmienności do policzenia korelacji.")
            return

        # Heatmapa korelacji - ulepsziona
        cols = list(corr_matrix.columns)
        max_cols = 30 if fast_mode else 50
        if len(cols) > max_cols:
            cols = cols[:max_cols]
            st.info(f"Pokazuję korelacje dla pierwszych {max_cols} kolumn")
        
        cm_display = corr_matrix.loc[cols, cols]

        fig = go.Figure(data=go.Heatmap(
            z=cm_display.values,
            x=cm_display.columns,
            y=cm_display.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(cm_display.values, 3),
            texttemplate="%{text}",
            textfont={"size": 8},
            hoverongaps=False,
            colorbar=dict(title="Korelacja")
        ))
        
        fig.update_layout(
            title=f"Macierz korelacji ({corr_method})",
            height=max(500, len(cols) * 15),
            width=max(500, len(cols) * 15)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Analiza korelacji z targetem
        if target_col and target_col in corr_matrix.columns:
            st.write(f"#### 🎯 Korelacje z targetem: {target_col}")
            target_corrs = corr_matrix[target_col].drop(target_col, errors="ignore").dropna()
            target_corrs_abs = target_corrs.abs().sort_values(ascending=False)
            
            # Filtruj po minimalnej korelacji
            significant_corrs = target_corrs_abs[target_corrs_abs >= min_corr]
            
            if len(significant_corrs) > 0:
                top_show = min(20, len(significant_corrs))
                
                # Wykres korelacji z targetem
                fig_target = go.Figure()
                
                colors = ['red' if corr < 0 else 'blue' for corr in target_corrs[significant_corrs.index[:top_show]]]
                
                fig_target.add_trace(go.Bar(
                    y=significant_corrs.index[:top_show],
                    x=target_corrs[significant_corrs.index[:top_show]],
                    orientation='h',
                    marker_color=colors,
                    text=[f"{val:.3f}" for val in target_corrs[significant_corrs.index[:top_show]]],
                    textposition='auto'
                ))
                
                fig_target.update_layout(
                    title=f"Korelacje z {target_col} (|r| >= {min_corr})",
                    xaxis_title="Korelacja",
                    yaxis_title="Cechy",
                    height=max(400, top_show * 25)
                )
                
                st.plotly_chart(fig_target, use_container_width=True)
                
                # Tabela szczegółowa
                corr_details = pd.DataFrame({
                    'Cecha': significant_corrs.index[:top_show],
                    'Korelacja': target_corrs[significant_corrs.index[:top_show]].round(4),
                    'Korelacja bezwzględna': significant_corrs[:top_show].round(4)
                })
                
                st.dataframe(corr_details, use_container_width=True, hide_index=True)
            else:
                st.info(f"Brak korelacji >= {min_corr} z targetem {target_col}")

        # Top korelacje między cechami
        st.write("#### 🔗 Najsilniejsze korelacje między cechami")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if i != j:
                    corr_val = corr_matrix.iloc[i, j]
                    if pd.notna(corr_val) and abs(corr_val) >= min_corr:
                        corr_pairs.append({
                            'Cecha 1': corr_matrix.columns[i],
                            'Cecha 2': corr_matrix.columns[j],
                            'Korelacja': corr_val,
                            'Korelacja bezwzględna': abs(corr_val)
                        })

        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.sort_values('Korelacja bezwzględna', ascending=False)
            
            top_pairs = min(15, len(corr_df))
            display_df = corr_df.head(top_pairs).round(4)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            if len(corr_df) > top_pairs:
                st.info(f"Pokazano {top_pairs} z {len(corr_df)} par o korelacji >= {min_corr}")
        else:
            st.info(f"Brak znaczących korelacji >= {min_corr}")

    def _render_categorical_analysis(self, df: pd.DataFrame, target_col: Optional[str], fast_mode: bool) -> None:
        """Analiza zmiennych kategorycznych."""
        st.write("### 📊 Analiza zmiennych kategorycznych")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_col and target_col in categorical_cols:
            categorical_cols.remove(target_col)

        if not categorical_cols:
            st.info("Brak zmiennych kategorycznych do analizy.")
            return

        selected_cat = st.selectbox("Wybierz zmienną kategoryczną:", categorical_cols)
        if selected_cat:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unikalne kategorie", int(df[selected_cat].nunique()))
            with col2:
                st.metric("Najczęstsza", _safe_mode(df[selected_cat]))
            with col3:
                st.metric("Braki", f"{df[selected_cat].isna().mean() * 100:.1f}%")

            s = df[selected_cat]
            s_display = s.fillna("(NaN)").astype(str)
            vc = s_display.value_counts(dropna=False).head(TOP_CATEG_LEVELS).reset_index()
            vc.columns = ["label", "count"]

            if vc.empty:
                st.info("Brak danych do wizualizacji rozkładu kategorii.")
            else:
                fig = px.pie(values=vc["count"], names=vc["label"], title=f"Rozkład kategorii: {selected_cat}")
                st.plotly_chart(fig, use_container_width=True)

            # Analiza względem targetu
            if target_col and target_col in df.columns:
                st.write(f"#### Analiza {selected_cat} vs {target_col}")
                self._render_categorical_vs_target(df, selected_cat, target_col)

    def _render_categorical_vs_target(self, df: pd.DataFrame, cat_col: str, target_col: str) -> None:
        """Analiza kategorii względem targetu."""
        cat_series = df[cat_col].fillna("(NaN)").astype(str)
        target_series = df[target_col]
        
        if pd.api.types.is_numeric_dtype(target_series):
            # Target numeryczny - box plot
            fig = px.box(
                x=cat_series, 
                y=target_series,
                title=f"Rozkład {target_col} według {cat_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statystyki grupowe
            group_stats = df.groupby(cat_series)[target_col].agg(['mean', 'median', 'std', 'count']).round(4)
            st.dataframe(group_stats, use_container_width=True)
            
        else:
            # Target kategoryczny - crosstab
            crosstab = pd.crosstab(cat_series, target_series, normalize='index') * 100
            
            if not crosstab.empty:
                fig = go.Figure()
                for col in crosstab.columns:
                    fig.add_trace(go.Bar(
                        name=str(col), 
                        x=crosstab.index, 
                        y=crosstab[col]
                    ))
                
                fig.update_layout(
                    barmode='stack',
                    title=f"Rozkład {target_col} w grupach {cat_col} (%)",
                    yaxis_title="Procent",
                    xaxis_title=cat_col
                )
                st.plotly_chart(fig, use_container_width=True)

    def _render_target_analysis(self, df: pd.DataFrame, target_col: str) -> None:
        """Szczegółowa analiza targetu."""
        st.write(f"### 🎯 Analiza targetu: {target_col}")
        target_series = df[target_col]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Typ", str(target_series.dtype))
        with col2:
            st.metric("Unikalne", int(target_series.nunique()))
        with col3:
            st.metric("Braki", f"{int(target_series.isna().sum()):,}")
        with col4:
            st.metric("Braki %", f"{target_series.isna().mean()*100:.1f}%")

        if pd.api.types.is_numeric_dtype(target_series):
            self._render_numeric_target_analysis(target_series)
        else:
            self._render_categorical_target_analysis(target_series)

    def _render_numeric_target_analysis(self, target_series: pd.Series) -> None:
        """Analiza targetu numerycznego."""
        clean_series = target_series.dropna()
        
        if clean_series.empty:
            st.info("Brak danych do analizy.")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig_hist = px.histogram(
                clean_series, 
                title=f"Rozkład {target_series.name}",
                nbins=min(50, int(np.sqrt(len(clean_series))))
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col2:
            # Box plot
            fig_box = px.box(y=clean_series, title=f"Box plot - {target_series.name}")
            st.plotly_chart(fig_box, use_container_width=True)

        # Statystyki opisowe
        st.write("#### 📊 Statystyki opisowe")
        stats = clean_series.describe()
        
        # Dodatkowe statystyki
        try:
            stats['skewness'] = clean_series.skew()
            stats['kurtosis'] = clean_series.kurtosis()
        except:
            pass
            
        st.dataframe(stats.to_frame().T, use_container_width=True)

    def _render_categorical_target_analysis(self, target_series: pd.Series) -> None:
        """Analiza targetu kategorycznego."""
        value_counts = target_series.value_counts(dropna=False)
        if value_counts.empty:
            st.info("Brak danych do analizy rozkładu klas.")
            return

        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig = px.bar(
                x=value_counts.index.astype(str), 
                y=value_counts.values, 
                title=f"Rozkład klas - {target_series.name}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart
            fig_pie = px.pie(
                values=value_counts.values,
                names=value_counts.index.astype(str),
                title="Proporcje klas"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Tabela z proporçjami
        st.write("#### 📋 Szczegóły rozkładu klas")
        class_df = pd.DataFrame({
            'Klasa': value_counts.index,
            'Liczebność': value_counts.values,
            'Procent': (value_counts.values / value_counts.sum() * 100).round(2)
        })
        st.dataframe(class_df, use_container_width=True, hide_index=True)

        # Analiza niebalansu
        if len(value_counts) > 1 and value_counts.min() > 0:
            imbalance_ratio = value_counts.max() / value_counts.min()
            if imbalance_ratio > 10:
                st.error(f"⚠️ Bardzo silny niebalans klas (ratio: {imbalance_ratio:.1f}:1)")
                st.info("💡 Rozważ techniki balansowania klas przed treningiem (SMOTE, undersampling).")
            elif imbalance_ratio > 3:
                st.warning(f"⚠️ Niebalans klas (ratio: {imbalance_ratio:.1f}:1)")
                st.info("💡 Monitoruj metryki dla każdej klasy oddzielnie.")

    def _render_feature_interactions(self, df: pd.DataFrame, target_col: Optional[str], fast_mode: bool) -> None:
        """Analiza interakcji między cechami."""
        st.write("### 🔗 Interakcje między cechami")
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
            feature2 = st.selectbox("Cecha 2:", [c for c in numeric_cols if c != feature1], key="feat2")

        if feature1 and feature2:
            # Scatter plot z opcjonalnym kolorem według targetu
            if target_col and target_col in df.columns:
                fig = px.scatter(
                    df, 
                    x=feature1, 
                    y=feature2, 
                    color=target_col,
                    title=f"Interakcja: {feature1} vs {feature2} (kolor: {target_col})",
                    opacity=0.6
                )
            else:
                fig = px.scatter(
                    df, 
                    x=feature1, 
                    y=feature2, 
                    title=f"Interakcja: {feature1} vs {feature2}",
                    opacity=0.6
                )
            
            # Dodaj linię trendu
            if len(df) < 10000:  # Tylko dla mniejszych zbiorów
                fig.add_scatter(
                    x=df[feature1], 
                    y=df[feature2], 
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                )
            
            st.plotly_chart(fig, use_container_width=True)

            # Statystyki interakcji
            col1, col2, col3 = st.columns(3)
            
            with col1:
                try:
                    correlation = df[feature1].corr(df[feature2])
                    st.metric("Korelacja Pearson", f"{correlation:.4f}")
                except Exception:
                    st.metric("Korelacja", "N/A")
            
            with col2:
                try:
                    spearman_corr = df[feature1].corr(df[feature2], method='spearman')
                    st.metric("Korelacja Spearman", f"{spearman_corr:.4f}")
                except Exception:
                    st.metric("Spearman", "N/A")
            
            with col3:
                # Mutual information (jeśli dostępne)
                try:
                    from sklearn.feature_selection import mutual_info_regression
                    clean_data = df[[feature1, feature2]].dropna()
                    if len(clean_data) > 10:
                        mi_score = mutual_info_regression(
                            clean_data[[feature1]], 
                            clean_data[feature2],
                            random_state=42
                        )[0]
                        st.metric("Mutual Information", f"{mi_score:.4f}")
                    else:
                        st.metric("Mutual Information", "N/A")
                except:
                    st.metric("Mutual Information", "N/A")

    def _render_outlier_detection(self, df: pd.DataFrame, fast_mode: bool) -> None:
        """Zaawansowana detekcja outliers."""
        st.write("### ⚠️ Detekcja wartości odstających")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.info("Brak kolumn numerycznych do analizy.")
            return

        # Wybór metody detekcji
        detection_method = st.selectbox(
            "Metoda detekcji:",
            ["IQR (Interquartile Range)", "Z-Score", "Modified Z-Score", "Isolation Forest"]
        )

        selected_col = st.selectbox("Wybierz kolumnę do analizy:", numeric_cols)
        
        if selected_col:
            series = df[selected_col].dropna()
            if len(series) == 0:
                st.info("Brak danych w wybranej kolumnie.")
                return

            # Wykryj outliers według wybranej metody
            outliers_mask = self._detect_outliers_by_method(series, detection_method)
            outliers = series[outliers_mask] if outliers_mask is not None else pd.Series([])
            
            # Statystyki outliers
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Liczba outliers", len(outliers))
            with col2:
                outlier_pct = (len(outliers) / len(series)) * 100
                st.metric("% outliers", f"{outlier_pct:.1f}%")
            with col3:
                if len(outliers) > 0:
                    st.metric("Min outlier", f"{outliers.min():.3f}")

            # Wizualizacje
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot z outliers
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=series,
                    name=selected_col,
                    boxpoints='outliers'
                ))
                fig_box.update_layout(
                    title=f"Box plot z outliers - {selected_col}",
                    height=400
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                # Histogram z oznaczonymi outliers
                fig_hist = go.Figure()
                
                # Normalne dane
                normal_data = series[~outliers_mask] if outliers_mask is not None else series
                fig_hist.add_trace(go.Histogram(
                    x=normal_data,
                    name="Normalne dane",
                    opacity=0.7,
                    marker_color='blue'
                ))
                
                # Outliers
                if len(outliers) > 0:
                    fig_hist.add_trace(go.Histogram(
                        x=outliers,
                        name="Outliers",
                        opacity=0.7,
                        marker_color='red'
                    ))
                
                fig_hist.update_layout(
                    title=f"Rozkład z outliers - {selected_col}",
                    height=400,
                    barmode='overlay'
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # Szczegóły outliers
            if len(outliers) > 0:
                with st.expander("📋 Szczegóły wartości odstających", expanded=False):
                    outlier_details = pd.DataFrame({
                        'Index': outliers.index,
                        'Wartość': outliers.values,
                        'Z-Score': np.abs((outliers - series.mean()) / series.std()),
                        'Percentyl': [series.quantile(0.01) <= val <= series.quantile(0.99) for val in outliers.values]
                    })
                    st.dataframe(outlier_details.head(20), use_container_width=True, hide_index=True)
                    
                    if len(outliers) > 20:
                        st.info(f"Pokazano pierwsze 20 z {len(outliers)} outliers")

        # Analiza outliers dla wszystkich kolumn
        st.write("#### 📊 Podsumowanie outliers dla wszystkich kolumn")
        all_outliers = _detect_outliers_iqr(df)
        
        if all_outliers:
            outlier_summary = pd.DataFrame([
                {
                    'Kolumna': col,
                    'Liczba outliers': len(outliers),
                    '% outliers': f"{(len(outliers) / len(df[col].dropna())) * 100:.1f}%",
                    'Min outlier': outliers.min(),
                    'Max outlier': outliers.max()
                }
                for col, outliers in all_outliers.items()
            ])
            
            st.dataframe(outlier_summary, use_container_width=True, hide_index=True)
        else:
            st.success("✅ Nie wykryto znaczących outliers w żadnej kolumnie numerycznej")

    def _detect_outliers_by_method(self, series: pd.Series, method: str) -> Optional[pd.Series]:
        """Wykrywa outliers różnymi metodami."""
        try:
            if method == "IQR (Interquartile Range)":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return (series < lower_bound) | (series > upper_bound)
            
            elif method == "Z-Score":
                z_scores = np.abs((series - series.mean()) / series.std())
                return z_scores > 3
            
            elif method == "Modified Z-Score":
                median = series.median()
                mad = (series - median).abs().median()
                modified_z_scores = 0.6745 * (series - median) / mad
                return np.abs(modified_z_scores) > 3.5
            
            elif method == "Isolation Forest":
                try:
                    from sklearn.ensemble import IsolationForest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_pred = iso_forest.fit_predict(series.values.reshape(-1, 1))
                    return pd.Series(outlier_pred == -1, index=series.index)
                except ImportError:
                    st.warning("Isolation Forest wymaga sklearn. Używam IQR.")
                    return self._detect_outliers_by_method(series, "IQR (Interquartile Range)")
            
            return None
            
        except Exception as e:
            st.error(f"Błąd podczas detekcji outliers: {e}")
            return None


def render_eda_section(df: pd.DataFrame, target_col: Optional[str] = None) -> None:
    """Główna funkcja renderująca EDA."""
    eda = AdvancedEDAComponents()
    eda.render_comprehensive_eda(df, target_col)