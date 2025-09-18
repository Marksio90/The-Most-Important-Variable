# ui_components.py — spójny UI dla TMIV (Dane → Target → Uruchomienia)
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

# Opcjonalna integracja z rozszerzonym uploaderem
try:
    from file_upload import SmartFileUploader, UploadConfig
    HAS_SMART_UPLOADER = True
except Exception:
    SmartFileUploader = None  # type: ignore
    UploadConfig = None       # type: ignore
    HAS_SMART_UPLOADER = False


# =========================
# KONFIGURACJE UI / DANYCH
# =========================
@dataclass
class DataConfig:
    max_file_size_mb: int = 200
    supported_formats: List[str] = (".csv", ".xlsx", ".xls")
    auto_detect_encoding: bool = True
    max_preview_rows: int = 50


@dataclass
class UIConfig:
    app_title: str = "TMIV"
    app_subtitle: str = "AutoML • EDA • Historia uruchomień"
    enable_llm: bool = False
    show_advanced_options: bool = True


# =========================
# GŁÓWNA WARSTWA KOMPONENTÓW
# =========================
class TMIVApp:
    """
    Warstwa widżetów dla danych i podstawowego EDA.
    Zapewnia jeden, spójny „wejściowy” krok:
      - wybór źródła: plik lub demo
      - walidacja i podgląd
      - gotowy DataFrame + nazwa
    """

    def __init__(self, data_cfg: DataConfig, ui_cfg: UIConfig):
        self.data_cfg = data_cfg
        self.ui_cfg = ui_cfg

    # ---------- Public API ----------
    def render_data_selection(self) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Renderuje sekcję wyboru danych i zwraca (df, dataset_name).
        Zawsze jedna ścieżka uploadu. Jeśli brak pliku – można wybrać demo.
        """
        src = st.radio(
            "Źródło danych",
            options=["📁 Wgraj plik", "🧪 Demo: Breast Cancer", "🧪 Demo: Iris"],
            horizontal=True
        )

        df: Optional[pd.DataFrame] = None
        dataset_name: str = ""

        if src == "📁 Wgraj plik":
            df, dataset_name = self._render_file_input()
        elif src == "🧪 Demo: Breast Cancer":
            df, dataset_name = self._load_demo("breast_cancer")
        else:
            df, dataset_name = self._load_demo("iris")

        # Podgląd i walidacje
        if df is not None:
            self._render_quick_profile(df, dataset_name)

        return df, dataset_name

    # ---------- Internal helpers ----------
    def _render_file_input(self) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Preferuje SmartFileUploader (jeśli dostępny), w przeciwnym razie fallback.
        """
        st.write("Wspierane formaty:", ", ".join(self.data_cfg.supported_formats))
        if HAS_SMART_UPLOADER and UploadConfig is not None:
            # Konfiguracja zaawansowanego uploadu
            cfg = UploadConfig(
                max_file_size_mb=self.data_cfg.max_file_size_mb,
                guess_encoding=self.data_cfg.auto_detect_encoding,
                allow_xlsx=True,
                allow_csv=True,
                max_preview_rows=self.data_cfg.max_preview_rows,
                # Lekka heurystyka delimiterów i decimali jest już w SmartFileUploader
            )
            uploader = SmartFileUploader(cfg)
            out = uploader.render("Wgraj plik z danymi")
            if out and out.df is not None and isinstance(out.df, pd.DataFrame):
                dataset_name = out.filename or "dataset"
                df = self._coerce_dataframe(out.df)
                return df, dataset_name
            return None, ""
        else:
            # Fallback: pojedynczy uploader Streamlit
            file = st.file_uploader(
                "Wgraj plik (.csv, .xlsx)",
                type=["csv", "xlsx", "xls"],
                accept_multiple_files=False
            )
            if not file:
                return None, ""

            # Walidacja rozmiaru (jeśli dostępne)
            try:
                size_mb = getattr(file, "size", None)
                if size_mb is not None:
                    size_mb = size_mb / (1024 * 1024)
                    if size_mb > self.data_cfg.max_file_size_mb:
                        st.error(f"Plik ma {size_mb:.1f} MB i przekracza limit {self.data_cfg.max_file_size_mb} MB.")
                        return None, ""
            except Exception:
                pass

            # Wczytanie CSV/XLSX
            name = getattr(file, "name", "dataset")
            suffix = (name or "").lower()
            try:
                if suffix.endswith(".csv"):
                    df = self._read_csv_safely(file)
                else:
                    # xlsx/xls
                    data = BytesIO(file.read())
                    df = pd.read_excel(data)
                df = self._coerce_dataframe(df)
                return df, name
            except Exception as e:
                st.error(f"Nie udało się wczytać pliku: {e}")
                return None, ""

    def _read_csv_safely(self, file_obj: Any) -> pd.DataFrame:
        """
        CSV reader z prostą próbą wykrycia kodowania i separatora.
        Bez dodatkowych zależności.
        """
        # Spróbuj najpierw standardowo
        try:
            file_obj.seek(0)
        except Exception:
            pass

        # 1. Próbujemy pandas domyślnie (często złapie delimiter)
        try:
            return pd.read_csv(file_obj)
        except Exception:
            pass

        # 2. Szereg podejść: delimitery i encodings
        candidates_sep = [",", ";", "\t", "|"]
        candidates_enc = ["utf-8", "cp1250", "latin-1"]
        for enc in candidates_enc:
            for sep in candidates_sep:
                try:
                    try:
                        file_obj.seek(0)
                    except Exception:
                        pass
                    return pd.read_csv(file_obj, encoding=enc, sep=sep, engine="python")
                except Exception:
                    continue

        # 3. Ostatnia próba – odczyt binarny do pamięci i analiza linii
        try:
            try:
                file_obj.seek(0)
            except Exception:
                pass
            raw = file_obj.read()
            for enc in candidates_enc:
                try:
                    text = raw.decode(enc, errors="replace")
                    # heurystyka: wybierz sep dający najrówniejszą liczbę kolumn w top 50 liniach
                    lines = [l for l in text.splitlines()[:50] if l.strip()]
                    best_sep, best_score = ",", -1
                    for sep in candidates_sep:
                        counts = [len(l.split(sep)) for l in lines]
                        if not counts:
                            continue
                        score = -np.std(counts)  # mniejsze zróżnicowanie = lepiej
                        if score > best_score:
                            best_score, best_sep = score, sep
                    buf = BytesIO(text.encode(enc))
                    return pd.read_csv(buf, sep=best_sep, encoding=enc, engine="python")
                except Exception:
                    continue
        except Exception:
            pass

        raise ValueError("Nie udało się zinterpretować CSV (kodowanie/separator).")

    def _coerce_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Porządki po wczytaniu: przycinanie nazw, deduplikacja kolumn, normalize dtypes.
        """
        df = df.copy()
        # Oczyść nagłówki
        df.columns = [str(c).strip() for c in df.columns]
        # Usuń zduplikowane kolumny (zostaw pierwszą)
        _, idx = np.unique(df.columns, return_index=True)
        if len(idx) != len(df.columns):
            df = df.iloc[:, sorted(idx)]
        # Strip stringi
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for c in object_cols:
            try:
                df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
            except Exception:
                pass
        return df

    def _load_demo(self, which: str) -> Tuple[pd.DataFrame, str]:
        """
        Dwa lekkie zbiory demo: breast_cancer, iris (ze sklearn).
        """
        try:
            from sklearn.datasets import load_breast_cancer, load_iris
        except Exception as e:
            st.error(f"Brak sklearn do wczytania demo: {e}")
            return None, ""

        if which == "breast_cancer":
            d = load_breast_cancer(as_frame=True)
            df = d.frame.copy()
            # zgodność nazw
            target_name = d.target.name if hasattr(d, "target") else "target"
            df.rename(columns={target_name: "target"}, inplace=True)
            return df, "breast_cancer"
        else:
            d = load_iris(as_frame=True)
            df = d.frame.copy()
            target_name = d.target.name if hasattr(d, "target") else "target"
            df.rename(columns={target_name: "target"}, inplace=True)
            return df, "iris"

    def _render_quick_profile(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Krótki, szybki profil danych — minimum informacji bez ciężkich bibliotek.
        """
        st.subheader("Podgląd danych")
        n = min(len(df), self.data_cfg.max_preview_rows)
        st.dataframe(df.head(n), use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Wiersze", f"{len(df):,}")
        with c2:
            st.metric("Kolumny", f"{len(df.columns):,}")
        with c3:
            missing = int(df.isna().sum().sum())
            st.metric("Braki (łącznie)", f"{missing:,}")

        with st.expander("Typy kolumn i braki (Top)"):
            info = self._dtypes_and_missing(df)
            st.dataframe(info, use_container_width=True, hide_index=True)

        st.caption(f"Zbiór: **{dataset_name}** • Gotowe do **Uruchomienia** (treningu).")

    def _dtypes_and_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prosta tabela: kolumna, dtype, #na, %na
        """
        miss = df.isna().sum()
        pct = (miss / len(df) * 100.0).round(2)
        out = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "na_cnt": miss.values,
            "na_pct": pct.values
        })
        # sortuj po brakach desc, potem alfabetycznie
        out = out.sort_values(["na_cnt", "column"], ascending=[False, True]).reset_index(drop=True)
        # pokaż tylko top 50, żeby nie zawalać UI
        return out.head(50)
