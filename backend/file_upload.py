
from __future__ import annotations
from typing import Tuple, Optional
import io
import pandas as pd
import streamlit as st

def upload_widget() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    file = st.file_uploader("Wgraj plik danych (CSV / JSON / Parquet)", type=["csv","json","parquet"])
    if not file:
        return None, None
    name = file.name
    try:
        if name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        elif name.lower().endswith(".json"):
            try:
                df = pd.read_json(file)
            except ValueError:
                file.seek(0)
                df = pd.read_json(io.StringIO(file.read().decode("utf-8")), lines=True)
        else:
            import pyarrow.parquet as pq  # optional
            import pyarrow as pa
            file.seek(0)
            df = pd.read_parquet(file)
        return df, name
    except Exception as e:
        st.error(f"Nie udało się odczytać pliku {name}: {e}")
        return None, None
