# frontend/advanced_eda.py — Zaawansowane komponenty EDA z rozwijalnymi widokami
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
        
        # Szybkie statystyki w jednej linii
        self._render_quick_stats_row(df)
        
        # Rozwijalne sekcje EDA
        with st.expander("🔍 Profil danych i jakość", expanded=False):
            self._render_data_quality_profile(df)
        
        with st.expander("📈 Rozkłady i histogramy", expanded=False):
            self._render_distributions(df, target_col)
        
        with st.expander("🌐 Macierz korelacji", expanded=False):
            self._render_correlation_analysis(df, target_col)
        
        with st.expander("📊 Analiza kategorii", expanded=False):
            self._render_categorical_analysis(df, target_col)
        
        with st.expander("🎯 Analiza targetu", expanded=bool(target_col)):
            if target_col and target_col in df.columns:
                self._render_target_analysis(df, target_col)
            else:
                st.info("Wybierz target aby zobaczyć analizę.")
        
        with st.expander("🔗 Interakcje między cechami", expanded=False):
            self._render_feature_interactions(df, target_col)
        
        with st.expander("⚠️ Anomalie i wartości odstające", expanded=False):
            self._render_outlier_detection(df)
    
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
            missing_cells = df.isna().sum().sum()
            st.metric("Braki", f"{missing_cells:,}")
        with cols[5]:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Pamięć", f"{memory_mb:.1f} MB")
    
    def _render_data_quality_profile(self, df: pd.DataFrame) -> None:
        """Renderuje szczegółowy profil jakości danych."""
        st.write("### Profil jakości danych")
        
        # Utworz profil dla każdej kolumny
        profile_data = []
        for col in df.columns:
            series = df[col]
            
            profile_data.append({
                'Kolumna': col,
                'Typ': str(series.dtype),
                'Braki': series.isna().sum(),
                'Braki %': f"{series.isna().mean() * 100:.1f}%",
                'Unikalne': series.nunique(),
                'Unikalne %': f"{series.nunique() / len(series) * 100:.1f}%",
                'Najczęstsza': str(series.mode().iloc[0]) if len(series.mode()) > 0 else 'N/A',
                'Przykład': str(series.dropna().iloc[0]) if len(series.dropna()) > 0 else 'N/A'
            })
        
        profile_df = pd.DataFrame(profile_data)
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
        
        # Mapa braków danych
        if df.isna().any().any():
            st.write("### Mapa braków danych")
            missing_data = df.isna()
            
            fig = go.Figure(data=go.Heatmap(
                z=missing_data.astype(int),
                x=missing_data.columns,
                y=[f"Row {i}" for i in range(len(missing_data))],
                colorscale=[[0, '#f0f0f0'], [1, '#ff4444']],
                showscale=False
            ))
            
            fig.update_layout(
                title="Mapa braków danych (czerwone = brak)",
                height=400,
                xaxis_title="Kolumny",
                yaxis_title="Wiersze (próbka)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_distributions(self, df: pd.DataFrame, target_col: Optional[str] = None) -> None:
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
                selected_numeric = st.multiselect(
                    "Wybierz kolumny numeryczne:",
                    numeric_cols,
                    default=numeric_cols[:4]  # Pierwsze 4
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
                    desc_stats = df[selected_numeric].describe()
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
                    value_counts = df[selected_categorical].value_counts().head(20)
                    
                    fig = px.bar(
                        x=value_counts.values,
                        y=value_counts.index,
                        orientation='h',
                        title=f"Rozkład: {selected_categorical}",
                        color=value_counts.values,
                        color_continuous_scale='viridis'
                    )
                    
                    fig.update_layout(
                        yaxis_title=selected_categorical,
                        xaxis_title="Liczba wystąpień",
                        height=max(400, len(value_counts) * 25)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statystyki kategorii
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Unikalne kategorie", df[selected_categorical].nunique())
                        st.metric("Najczęstsza", str(value_counts.index[0]))
                    with col2:
                        st.metric("Częstość najczęstszej", f"{value_counts.iloc[0]:,}")
                        missing_pct = df[selected_categorical].isna().mean() * 100
                        st.metric("Braki", f"{missing_pct:.1f}%")
            else:
                st.info("Brak kolumn kategorycznych do analizy.")
    
    def _render_correlation_analysis(self, df: pd.DataFrame, target_col: Optional[str] = None) -> None:
        """Renderuje analizę korelacji."""
        st.write("### Analiza korelacji")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            st.info("Za mało kolumn numerycznych do analizy korelacji.")
            return
        
        # Macierz korelacji
        corr_matrix = numeric_df.corr()
        
        # Heatmapa
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Macierz korelacji",
            height=600,
            width=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Najsilniejsze korelacje
        if target_col and target_col in corr_matrix.columns:
            st.write(f"#### Korelacje z targetem: {target_col}")
            target_corrs = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
            
            fig_bar = px.bar(
                x=target_corrs.values,
                y=target_corrs.index,
                orientation='h',
                title=f"Korelacje z {target_col}",
                color=target_corrs.values,
                color_continuous_scale='viridis'
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Tabela najsilniejszych korelacji
        st.write("#### Najsilniejsze korelacje (wszystkie pary)")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Kolumna 1': corr_matrix.columns[i],
                    'Kolumna 2': corr_matrix.columns[j],
                    'Korelacja': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df = corr_df.reindex(corr_df['Korelacja'].abs().sort_values(ascending=False).index)
        
        st.dataframe(corr_df.head(10), use_container_width=True, hide_index=True)
    
    def _render_categorical_analysis(self, df: pd.DataFrame, target_col: Optional[str] = None) -> None:
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
                st.metric("Unikalne kategorie", df[selected_cat].nunique())
            with col2:
                most_common = df[selected_cat].mode().iloc[0] if len(df[selected_cat].mode()) > 0 else "N/A"
                st.metric("Najczęstsza", str(most_common))
            with col3:
                missing_pct = df[selected_cat].isna().mean() * 100
                st.metric("Braki", f"{missing_pct:.1f}%")
            
            # Wykres rozkładu
            value_counts = df[selected_cat].value_counts().head(15)
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Rozkład kategorii: {selected_cat}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analiza względem targetu (jeśli target jest kategoryczny)
            if target_col and target_col in df.columns:
                st.write(f"#### Analiza {selected_cat} vs {target_col}")
                
                crosstab = pd.crosstab(df[selected_cat], df[target_col], normalize='index') * 100
                
                fig_stack = go.Figure()
                
                for col in crosstab.columns:
                    fig_stack.add_trace(go.Bar(
                        name=str(col),
                        x=crosstab.index,
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
            st.metric("Unikalne", target_series.nunique())
        with col3:
            st.metric("Braki", f"{target_series.isna().sum():,}")
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
        fig = px.histogram(
            target_series.dropna(),
            title=f"Rozkład {target_series.name}",
            nbins=50
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plot
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
        
        # Bar chart
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Rozkład klas - {target_series.name}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela z liczebnościami
        st.dataframe(value_counts.to_frame('Liczebność'), use_container_width=True)
        
        # Sprawdź balans klas
        if len(value_counts) > 1:
            imbalance_ratio = value_counts.max() / value_counts.min()
            if imbalance_ratio > 3:
                st.warning(f"⚠️ Wykryto niebalans klas (ratio: {imbalance_ratio:.1f}:1)")
                st.info("💡 Rozważ techniki balansowania klas przed treningiem.")
    
    def _render_feature_interactions(self, df: pd.DataFrame, target_col: Optional[str] = None) -> None:
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
            feature2 = st.selectbox("Cecha 2:", 
                                   [col for col in numeric_cols if col != feature1], 
                                   key="feat2")
        
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
            correlation = df[feature1].corr(df[feature2])
            st.metric("Korelacja między wybranymi cechami", f"{correlation:.3f}")
    
    def _render_outlier_detection(self, df: pd.DataFrame) -> None:
        """Renderuje detekcję wartości odstających."""
        st.write("### Detekcja wartości odstających")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.info("Brak kolumn numerycznych do analizy.")
            return
        
        selected_col = st.selectbox("Wybierz kolumnę do analizy:", numeric_cols)
        
        if selected_col:
            series = df[selected_col].dropna()
            
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