# file_upload.py — SmartFileUploader (detekcja kodowania, separatora, limity, preview)
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional, List, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Konfiguracja i wyniki
# =========================
@dataclass
class UploadConfig:
    max_file_size_mb: int = 200
    guess_encoding: bool = True
    allow_csv: bool = True
    allow_xlsx: bool = True
    max_preview_rows: int = 50
    sample_mb: float = 2.0        # ile MB próbki do heurystyki
    hard_row_cap: Optional[int] = None  # np. 2_000_000 — twardy limit wierszy (None = brak)


@dataclass
class UploadOutput:
    df: Optional[pd.DataFrame]
    filename: str = ""
    encoding: Optional[str] = None
    delimiter: Optional[str] = None
    warnings: List[str] = None  # type: ignore


# =========================
# Główny uploader
# =========================
class SmartFileUploader:
    def __init__(self, cfg: UploadConfig):
        self.cfg = cfg

    # ---- public API ----
    def render(self, label: str = "Wgraj plik z danymi") -> Optional[UploadOutput]:
        types = []
        if self.cfg.allow_csv: types += ["csv"]
        if self.cfg.allow_xlsx: types += ["xlsx", "xls"]

        file = st.file_uploader(label, type=types, accept_multiple_files=False)
        if not file:
            return None

        # rozmiar
        size_b = getattr(file, "size", None)
        if size_b is not None:
            size_mb = size_b / (1024 * 1024)
            if size_mb > self.cfg.max_file_size_mb:
                st.error(f"Plik ma {size_mb:.1f} MB i przekracza limit {self.cfg.max_file_size_mb} MB.")
                return UploadOutput(df=None, filename=getattr(file, "name", ""), warnings=["limit_size_exceeded"])

        name = getattr(file, "name", "") or ""
        suffix = name.lower()

        # CSV
        if suffix.endswith(".csv"):
            return self._handle_csv(file, name)

        # XLSX/XLS
        if suffix.endswith(".xlsx") or suffix.endswith(".xls"):
            return self._handle_excel(file, name)

        # Inne – nieobsługiwane
        st.error("Nieobsługiwany format. Dozwolone: .csv, .xlsx.")
        return UploadOutput(df=None, filename=name, warnings=["unsupported_format"])

    # ---- internals ----
    def _handle_excel(self, file_obj: Any, name: str) -> UploadOutput:
        try:
            data = BytesIO(file_obj.read())
            df = pd.read_excel(data)
            df = self._coerce_dataframe(df)
            warn: List[str] = []

            if self.cfg.hard_row_cap and len(df) > self.cfg.hard_row_cap:
                warn.append(f"row_cap_reached_{self.cfg.hard_row_cap}")
                df = df.head(self.cfg.hard_row_cap)

            self._preview(df)
            return UploadOutput(df=df, filename=name, warnings=warn or [])
        except Exception as e:
            st.error(f"Nie udało się wczytać arkusza: {e}")
            return UploadOutput(df=None, filename=name, warnings=[f"excel_error:{e}"])

    def _handle_csv(self, file_obj: Any, name: str) -> UploadOutput:
        # 1) pobierz próbkę bajtów
        sample_bytes = self._read_sample(file_obj, int(self.cfg.sample_mb * 1024 * 1024))

        # 2) heurystyka kodowania i separatora
        enc = self._detect_encoding(sample_bytes) if self.cfg.guess_encoding else "utf-8"
        sample_text = self._safe_decode(sample_bytes, enc)
        sep = self._detect_delimiter(sample_text)

        # 3) właściwy odczyt
        try:
            # cofnij wskaźnik i czytaj cały plik
            self._seek_start(file_obj)
            df = pd.read_csv(file_obj, encoding=enc, sep=sep, engine="python")
        except Exception:
            # fallbacki na kodowanie/delimitery
            candidates_enc = [enc, "utf-8", "cp1250", "latin-1"]
            candidates_sep = [sep, ",", ";", "\t", "|"]
            df = None
            for e in candidates_enc:
                for s in candidates_sep:
                    try:
                        self._seek_start(file_obj)
                        df = pd.read_csv(file_obj, encoding=e, sep=s, engine="python")
                        enc, sep = e, s
                        raise StopIteration  # „wychodzimy” z obu pętli
                    except StopIteration:
                        break
                    except Exception:
                        continue
                if df is not None:
                    break

        if df is None:
            st.error("Nie udało się zinterpretować CSV (kodowanie/separator).")
            return UploadOutput(df=None, filename=name, warnings=["csv_parse_error"])

        df = self._coerce_dataframe(df)
        warns: List[str] = []
        if self.cfg.hard_row_cap and len(df) > self.cfg.hard_row_cap:
            warns.append(f"row_cap_reached_{self.cfg.hard_row_cap}")
            df = df.head(self.cfg.hard_row_cap)

        self._preview(df)
        return UploadOutput(df=df, filename=name, encoding=enc, delimiter=sep, warnings=warns)

    # ---- helpers ----
    @staticmethod
    def _seek_start(file_obj: Any) -> None:
        try:
            file_obj.seek(0)
        except Exception:
            pass

    @staticmethod
    def _read_sample(file_obj: Any, nbytes: int) -> bytes:
        try:
            SmartFileUploader._seek_start(file_obj)
            return file_obj.read(nbytes)
        except Exception:
            return b""

    @staticmethod
    def _safe_decode(b: bytes, enc: str) -> str:
        try:
            return b.decode(enc, errors="replace")
        except Exception:
            try:
                return b.decode("utf-8", errors="replace")
            except Exception:
                return ""

    def _detect_encoding(self, sample: bytes) -> str:
        # lekka heurystyka bez chardet: preferuj utf-8, potem cp1250, latin-1
        for enc in ("utf-8", "cp1250", "latin-1"):
            try:
                sample.decode(enc)
                return enc
            except Exception:
                continue
        return "utf-8"

    def _detect_delimiter(self, sample_text: str) -> str:
        # wybór separatora po najmniejszej wariancji liczby kolumn w liniach
        candidates = [",", ";", "\t", "|"]
        lines = [l for l in sample_text.splitlines()[:100] if l.strip()]
        if not lines:
            return ","
        best_sep, best_score = ",", float("inf")
        for sep in candidates:
            counts = [len(l.split(sep)) for l in lines]
            if not counts:
                continue
            var = float(np.var(counts))
            if var < best_score and max(counts) > 1:
                best_score, best_sep = var, sep
        return best_sep

    @staticmethod
    def _coerce_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        # usuń duplikaty kolumn
        _, idx = np.unique(df.columns, return_index=True)
        if len(idx) != len(df.columns):
            df = df.iloc[:, sorted(idx)]
        # strip stringi
        for c in df.select_dtypes(include=["object"]).columns.tolist():
            try:
                df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
            except Exception:
                pass
        return df

    def _preview(self, df: pd.DataFrame) -> None:
        n = min(len(df), self.cfg.max_preview_rows)
        with st.expander("Podgląd danych (pierwsze wiersze)"):
            st.dataframe(df.head(n), use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Wiersze", f"{len(df):,}")
        with c2:
            st.metric("Kolumny", f"{len(df.columns):,}")
        with c3:
            st.metric("Braki (łącznie)", f"{int(df.isna().sum().sum()):,}")
