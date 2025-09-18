# eda_integration.py — lekki auto-preprocessing + raport (kolumna → [operacje])
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


# ==============================
# Raport z przetwarzania danych
# ==============================
@dataclass
class PreprocessingReport:
    rows_before: int
    cols_before: int
    rows_after: int
    cols_after: int

    dropped_columns: List[str] = field(default_factory=list)
    added_columns: List[str] = field(default_factory=list)
    renamed_columns: Dict[str, str] = field(default_factory=dict)

    # klucz: nazwa kolumny, wartość: lista operacji wykonanych na kolumnie
    transformation_map: Dict[str, List[str]] = field(default_factory=dict)

    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    dataset_name: Optional[str] = None
    target: Optional[str] = None

    def add_op(self, col: str, op: str) -> None:
        self.transformation_map.setdefault(str(col), []).append(op)

    def summary(self) -> Dict[str, Any]:
        return {
            "shape_before": [self.rows_before, self.cols_before],
            "shape_after": [self.rows_after, self.cols_after],
            "dropped_columns": self.dropped_columns,
            "added_columns": self.added_columns,
            "renamed_columns": self.renamed_columns,
            "transformations": self.transformation_map,
            "warnings": self.warnings,
            "notes": self.notes,
            "target": self.target,
            "dataset_name": self.dataset_name,
        }


# =====================================
# Główny, szybki preprocessor do EDA
# =====================================
class SmartDataPreprocessor:
    """
    Szybki, bezpieczny pre-processor do porządków przed treningiem:
      - oczyszczanie nagłówków i duplikatów kolumn,
      - lekkie parsowanie kolumn datowych (+ opcjonalne featury daty),
      - przycinanie wartości odstających (winsoryzacja percentylowa),
      - uzupełnianie pojedynczych ewidentnych braków (opcjonalne),
      - detekcja kolumn stałych / quasi-zerowej zmienności (NZV) i ich wycięcie,
      - raport kolumna → [operacje].
    Imputacja i kodowanie kategorii wchodzą potem w skład Pipeline z ml_integration.
    """

    def __init__(
        self,
        *,
        enable_datetime_features: bool = True,
        datetime_guess_limit: int = 30,           # maks. liczba kolumn sprawdzanych „na datę”
        winsorize_pct: float = 0.005,             # 0.5% z każdej strony (prosta winsoryzacja)
        drop_constant: bool = True,
        drop_nzv_threshold: float = 0.01,         # <1% unikalnych wartości => NZV
        light_impute: bool = True,                # uzupełnij pojedyncze braki prostą heurystyką
        dataset_name: Optional[str] = None,
    ):
        self.enable_datetime_features = enable_datetime_features
        self.datetime_guess_limit = max(0, int(datetime_guess_limit))
        self.winsorize_pct = max(0.0, min(0.2, winsorize_pct))
        self.drop_constant = drop_constant
        self.drop_nzv_threshold = max(0.0, min(0.2, drop_nzv_threshold))
        self.light_impute = light_impute
        self.dataset_name = dataset_name

    # --------------- API ---------------
    def preprocess(self, df: pd.DataFrame, *, target: Optional[str] = None) -> Tuple[pd.DataFrame, PreprocessingReport]:
        if df is None or df.empty:
            raise ValueError("Pusty DataFrame — brak danych do preprocessingu.")

        df = df.copy()
        rows_before, cols_before = df.shape
        rpt = PreprocessingReport(
            rows_before=rows_before,
            cols_before=cols_before,
            rows_after=rows_before,
            cols_after=cols_before,
            dataset_name=self.dataset_name,
            target=target
        )

        # 1) Nagłówki: trim, prostsze nazwy
        old_cols = list(df.columns)
        new_cols = [self._clean_col_name(c) for c in old_cols]
        if new_cols != old_cols:
            df.columns = new_cols
            for o, n in zip(old_cols, new_cols):
                if o != n:
                    rpt.renamed_columns[o] = n

        # 2) Kolumny zduplikowane -> zostaw pierwszą
        df, dropped_dupes = self._drop_duplicated_columns(df)
        if dropped_dupes:
            rpt.dropped_columns.extend(dropped_dupes)
            for c in dropped_dupes:
                rpt.add_op(c, "drop_duplicate_column")

        # 3) ID-like i kolumny stałe / NZV (opcjonalnie drop)
        dropped_static = []
        if self.drop_constant or self.drop_nzv_threshold > 0:
            to_drop = self._constant_or_nzv_columns(df, nzv_threshold=self.drop_nzv_threshold, exclude=[target] if target else [])
            if to_drop:
                dropped_static = to_drop
                df = df.drop(columns=to_drop, errors="ignore")
                rpt.dropped_columns.extend(to_drop)
                for c in to_drop:
                    rpt.add_op(c, "drop_constant_or_nzv")

        # 4) Parsowanie kolumn datowych (+featury)
        if self.enable_datetime_features:
            df, added_dt_cols, parsed_cols = self._parse_and_expand_datetimes(df, limit=self.datetime_guess_limit, exclude=[target] if target else [])
            rpt.added_columns.extend(added_dt_cols)
            for c in parsed_cols:
                rpt.add_op(c, "parsed_datetime")
            for c in added_dt_cols:
                rpt.add_op(c, "date_feature")

        # 5) Lekkie winsoryzowanie numeryków (robust to extreme outliers)
        if self.winsorize_pct > 0:
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            for c in num_cols:
                try:
                    lo = df[c].quantile(self.winsorize_pct)
                    hi = df[c].quantile(1 - self.winsorize_pct)
                    df[c] = df[c].clip(lower=lo, upper=hi)
                    rpt.add_op(c, f"winsorize[{self.winsorize_pct:.3f}]")
                except Exception:
                    continue

        # 6) „Light impute” (opcjonalnie) — dla pojedynczych braków
        if self.light_impute:
            df = self._light_impute(df, rpt, exclude=[target] if target else [])

        # Aktualizacja rozmiaru
        rpt.rows_after, rpt.cols_after = df.shape

        # Notatki końcowe
        if target and target not in df.columns:
            rpt.warnings.append(f"Uwaga: kolumna target '{target}' nie istnieje po preprocessingu.")
        if rpt.cols_after < 1:
            rpt.warnings.append("Uwaga: po preprocessingu nie pozostały żadne kolumny wejściowe.")
        if dropped_static:
            rpt.notes.append(f"Usunięto {len(dropped_static)} kolumn stałych/NZV: {', '.join(dropped_static[:8])}{'…' if len(dropped_static) > 8 else ''}")

        return df, rpt

    # --------------- Helpers ---------------
    @staticmethod
    def _clean_col_name(col: Any) -> str:
        s = str(col).strip()
        # zamień whitespace i dziwne separatory na podkreślenia, usuń powtórzenia
        s = s.replace("\n", " ").replace("\r", " ")
        s = "_".join(s.split())
        return s

    @staticmethod
    def _drop_duplicated_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        cols = list(df.columns)
        _, first_idx = np.unique(cols, return_index=True)
        keep = set(cols[i] for i in sorted(first_idx))
        dropped = [c for c in cols if c not in keep]
        if dropped:
            df = df.loc[:, [c for c in cols if c in keep]]
        return df, dropped

    @staticmethod
    def _constant_or_nzv_columns(df: pd.DataFrame, *, nzv_threshold: float, exclude: List[str]) -> List[str]:
        """
        Zwraca listę kolumn stałych lub o quasi-zerowej zmienności.
        NZV: unikalnych/kart. <= nzv_threshold (np. 1%), ale > 0.
        """
        out: List[str] = []
        n = len(df)
        if n == 0:
            return out
        for c in df.columns:
            if c in exclude:
                continue
            try:
                nun = df[c].nunique(dropna=True)
                if nun <= 1:
                    out.append(c)
                else:
                    ratio = nun / float(n)
                    if 0 < nzv_threshold and ratio <= nzv_threshold:
                        out.append(c)
            except Exception:
                # jak kolumna jest egzotyczna i sypnie – nie usuwamy
                continue
        return out

    @staticmethod
    def _maybe_parse_datetime(s: pd.Series) -> Optional[pd.Series]:
        """
        Próba parsowania na datetime (jeśli sukces na >70% nie-NaN).
        """
        try:
            parsed = pd.to_datetime(s, errors="coerce", utc=True, infer_datetime_format=True)
            non_na = (~s.isna()).sum()
            if non_na == 0:
                return None
            ok = (~parsed.isna()).sum()
            if ok / max(1, non_na) >= 0.7:
                return parsed
        except Exception:
            pass
        return None

    def _parse_and_expand_datetimes(
        self, df: pd.DataFrame, *, limit: int, exclude: List[str]
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Parsuje „teksty-jak-daty” i dodaje featury: rok, miesiąc, dzień, dzień_tygodnia.
        Zwraca: df, lista_dodanych_kolumn, lista_parsed_cols.
        """
        added: List[str] = []
        parsed_cols: List[str] = []
        obj_cols = [c for c in df.select_dtypes(include=["object"]).columns.tolist() if c not in exclude]
        # ogranicz liczbę sprawdzanych kolumn dla szybkości
        for c in obj_cols[:limit]:
            s_parsed = self._maybe_parse_datetime(df[c])
            if s_parsed is None:
                continue
            df[c] = s_parsed  # nadpisz kolumnę datetime64[ns, UTC]
            parsed_cols.append(c)

            # featury
            try:
                df[f"{c}__year"] = df[c].dt.year
                df[f"{c}__month"] = df[c].dt.month
                df[f"{c}__day"] = df[c].dt.day
                df[f"{c}__dow"] = df[c].dt.dayofweek
                added.extend([f"{c}__year", f"{c}__month", f"{c}__day", f"{c}__dow"])
            except Exception:
                # jeśli kolumna jest cała NaT – pomiń
                continue

        return df, added, parsed_cols

    def _light_impute(self, df: pd.DataFrame, rpt: PreprocessingReport, *, exclude: List[str]) -> pd.DataFrame:
        """
        Uzupełnia proste, pojedyncze braki:
          - numeryczne: medianą, gdy braków jest mało (<=10%),
          - kategoryczne: modą, gdy braków jest mało (<=10%).
        Imputacja „poważna” dzieje się potem w Pipeline (ml_integration).
        """
        n = len(df)
        if n == 0:
            return df

        num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in exclude]
        cat_cols = [c for c in df.columns if c not in num_cols and c not in exclude]

        # numeryczne
        for c in num_cols:
            try:
                na_ratio = df[c].isna().mean()
                if 0 < na_ratio <= 0.10:
                    med = df[c].median()
                    df[c] = df[c].fillna(med)
                    rpt.add_op(c, "light_impute_median")
            except Exception:
                continue

        # kategoryczne
        for c in cat_cols:
            try:
                na_ratio = df[c].isna().mean()
                if 0 < na_ratio <= 0.10:
                    mode = df[c].mode(dropna=True)
                    if len(mode) > 0:
                        df[c] = df[c].fillna(mode.iloc[0])
                        rpt.add_op(c, "light_impute_mode")
            except Exception:
                continue

        return df


# ==============================
# Dodatkowy szybki „profil”
# ==============================
def profile_dataframe(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    Szybki profil tabelaryczny (bez ciężkich bibliotek):
      kolumna | dtype | #na | %na | #unikalnych | przykład
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["column", "dtype", "na_cnt", "na_pct", "nunique", "sample"])

    miss = df.isna().sum()
    pct = (miss / len(df) * 100.0).round(2)
    nun = df.nunique(dropna=True)

    samples: List[str] = []
    for c in df.columns:
        try:
            # przykładowa nie-NaN wartość do wglądu
            val = df.loc[~df[c].isna(), c].iloc[0] if (~df[c].isna()).any() else None
            samples.append(str(val)[:80] if val is not None else "")
        except Exception:
            samples.append("")

    out = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "na_cnt": miss.values,
        "na_pct": pct.values,
        "nunique": nun.values,
        "sample": samples
    })
    out = out.sort_values(["na_cnt", "nunique", "column"], ascending=[False, True, True]).reset_index(drop=True)
    return out.head(top_n)


__all__ = [
    "SmartDataPreprocessor",
    "PreprocessingReport",
    "profile_dataframe",
]
