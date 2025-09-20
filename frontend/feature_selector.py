from __future__ import annotations

import pandas as pd
import streamlit as st

def render_interactive_feature_selection(df, target):
    st.subheader("🎯 Interaktywna selekcja cech")
    
    # Correlation matrix
    corr_matrix = df.corr()
    target_corr = corr_matrix[target].abs().sort_values(ascending=False)[1:]
    
    # Multiselect z domyślnymi top cechami
    default_features = target_corr.head(10).index.tolist()
    selected_features = st.multiselect(
        "Wybierz cechy do modelu:",
        options=df.columns.tolist(),
        default=default_features
    )
    
    # Pokazuj korelację w czasie rzeczywistym
    if selected_features:
        fig = px.bar(
            x=selected_features,
            y=[target_corr.get(f, 0) for f in selected_features],
            title="Korelacja z targetem"
        )
        st.plotly_chart(fig)
    
    return selected_features