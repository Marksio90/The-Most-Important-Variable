# frontend/ui_components.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import io
import csv
import re

import pandas as pd
import streamlit as st


# ==============================
# NIEWIELKIE POMOCNICZE
# ==============================
def _is_date_like_series(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object:
        parsed = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
        return parsed.notna().mean() >= 0.9
    return False


def _auto_side_options(df: pd.DataFrame, target: Optional[str], task: Optional[str]) -> Dict[str, Any]:
    """
    Heurystyczne domyślne opcje (gdybyś chciał je mieć pod ręką).
    Ten helper NIE jest używany przez UI automatycznie – to tylko narzędzie,
    które możesz wywołać z app.py jeśli chcesz.
    """
    side: Dict[str, Any] = {
        "drop_constant": True,
        "auto_dates": True,
        "limit_cardinality": True,
        "high_card_topk": 50,
        "target_log1p": "auto",   # dla regresji: auto (jeśli dodatni i skośny)
        "target_winsor": "auto",  # dla regresji: auto (gdy outlierów >5%)
    }

    has_cat = any(
        (df[c].dtype == "object") or str(df[c].dtype).startswith("category")
        for c in df.columns if c != (target or "")
    )
    if not has_cat:
        side["limit_cardinality"] = False

    has_maybe_date = any(_is_date_like_series(df[c]) for c in df.columns if c != (target or ""))
    if not has_maybe_date:
        side["auto_dates"] = False

    if task == "clf":
        side["target_log1p"] = False
        side["target_winsor"] = False

    return side


# ==============================
# HEADER
# ==============================
def header() -> None:
    st.title("TMIV — The Most Important Variables")
    st.caption("Silnik EDA • Trening modeli • Historia uruchomień • Jeden eksport ZIP")


# ==============================
# POMOCNICZE — auto-wykrywanie CSV
# ==============================
def _detect_encoding(sample: bytes) -> str:
    for enc in ("utf-8", "cp1250", "latin-1"):
        try:
            sample.decode(enc)
            return enc
        except Exception:
            continue
    return "utf-8"


def _detect_delimiter(decoded_sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(decoded_sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        counts = {d: decoded_sample.count(d) for d in [",", ";", "\t", "|"]}
        return max(counts, key=counts.get) if any(counts.values()) else ","


_num_token = re.compile(r"(?<!\w)(\d{1,3}([.,]\d{3})*[.,]\d+)(?!\w)")


def _detect_decimal(decoded_sample: str, delimiter: str) -> str:
    if delimiter == ";":
        return ","
    dot = comma = 0
    for m in _num_token.finditer(decoded_sample):
        s = m.group(1)
        if "," in s and "." in s:
            comma += int(s.rfind(",") > s.rfind("."))
            dot += int(s.rfind(".") > s.rfind(","))
        elif "," in s:
            comma += 1
        elif "." in s:
            dot += 1
    if comma > dot:
        return ","
    return "."


def _detect_header(decoded_sample: str) -> bool:
    try:
        return bool(csv.Sniffer().has_header(decoded_sample))
    except Exception:
        return True


# ==============================
# WYBÓR DANYCH: CSV / JSON / demo 'avocado'
# ==============================
def dataset_selector(sample_data_path: Optional[str | Path] = None) -> Tuple[pd.DataFrame, str]:
    """
    Wybór danych:
      - Wgraj własny CSV (auto-wykrywanie enc/sep/decimal/header z możliwością nadpisania),
      - Wgraj własny JSON (kilka popularnych formatów),
      - Zbiór 'avocado' (demo).
    Zwraca: (df, nazwa_zbioru). Jeśli brak — df pusty i nazwa informacyjna.
    """
    st.subheader("📦 Dane wejściowe")

    mode = st.radio(
        "Źródło danych",
        ["Wgraj własny plik CSV", "Wgraj własny plik JSON", "Zbiór 'avocado' (demo)"],
        horizontal=True,
    )

    # --- WŁASNY CSV (uniwersalny) ---
    if mode == "Wgraj własny plik CSV":
        up = st.file_uploader("Wgraj plik CSV", type=["csv"])
        if up is None:
            st.info("Wgraj plik CSV lub przełącz na demo 'avocado'.")
            return pd.DataFrame(), "(czekam na CSV)"

        raw = up.read()
        # Auto-wykrywanie
        enc_auto = _detect_encoding(raw[:64 * 1024])
        decoded = raw.decode(enc_auto, errors="ignore")
        sep_auto = _detect_delimiter(decoded[:64 * 1024])
        dec_auto = _detect_decimal(decoded[:64 * 1024], sep_auto)
        header_auto = _detect_header(decoded[:8 * 1024])

        with st.expander("⚙️ Zaawansowane opcje wczytywania", expanded=False):
            encoding = st.selectbox("Kodowanie", [enc_auto, "utf-8", "cp1250", "latin-1"], index=0)
            sep = st.selectbox("Separator", [sep_auto, ",", ";", "\\t", "|"], index=0)
            if sep == "\\t":
                sep = "\t"
            decimal = st.selectbox("Separator dziesiętny", [dec_auto, ".", ","], index=0)
            header = st.checkbox("Pierwszy wiersz to nagłówki", value=header_auto)
        try:
            buf = io.BytesIO(raw)
            if header:
                df = pd.read_csv(buf, sep=sep, decimal=decimal, encoding=encoding)
            else:
                df = pd.read_csv(buf, sep=sep, decimal=decimal, encoding=encoding, header=None)
                df.columns = [f"col_{i}" for i in range(df.shape[1])]
        except Exception as e:
            st.error(f"Nie udało się wczytać CSV: {e}")
            return pd.DataFrame(), "(błąd CSV)"

        sep_label = "TAB" if sep == "\t" else sep
        st.caption(f"Załadowano: {up.name}  •  kodowanie={encoding}, sep='{sep_label}', decimal='{decimal}'")
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"Kolumny ({len(df.columns)}): " + ", ".join(map(str, df.columns[:30])) + ("…" if len(df.columns) > 30 else ""))

        st.session_state["df_columns"] = list(df.columns)
        return df, up.name

    # --- WŁASNY JSON ---
    elif mode == "Wgraj własny plik JSON":
        upj = st.file_uploader("Wgraj plik JSON", type=["json"])
        if upj is None:
            st.info("Wgraj plik JSON lub przełącz na inne źródło.")
            return pd.DataFrame(), "(czekam na JSON)"

        raw = upj.read()
        decoded = raw.decode("utf-8", errors="ignore")

        # Spróbuj różnych popularnych wariantów
        df = None
        try:
            # 1) lista rekordów (najczęstsze)
            df = pd.read_json(io.StringIO(decoded), orient="records")
            if df is None or df.empty:
                raise ValueError("puste records")
        except Exception:
            try:
                # 2) NDJSON (po jednej linii na rekord)
                df = pd.read_json(io.StringIO(decoded), lines=True)
                if df is None or df.empty:
                    raise ValueError("puste lines")
            except Exception:
                try:
                    # 3) auto (pandas niech sam spróbuje)
                    df = pd.read_json(io.StringIO(decoded))
                except Exception as e:
                    st.error(f"Nie udało się wczytać JSON: {e}")
                    return pd.DataFrame(), "(błąd JSON)"

        if df is None or df.empty:
            st.warning("Plik JSON nie zawiera danych tabelarycznych do wyświetlenia.")
            return pd.DataFrame(), "(pusty JSON)"

        st.caption(f"Załadowano: {upj.name}  •  wiersze={len(df)}, kolumny={len(df.columns)}")
        st.dataframe(df.head(20), use_container_width=True)
        st.session_state["df_columns"] = list(df.columns)
        return df, upj.name

    # --- DEMO: avocado.csv ---
    else:
        candidates: List[Path] = []
        if sample_data_path:
            p = Path(sample_data_path)
            candidates += [p / "avocado.csv", p / "Avocado.csv"]
        candidates += [Path("data/avocado.csv"), Path("datasets/avocado.csv"), Path("avocado.csv")]
        src_path: Optional[Path] = next((p for p in candidates if p.exists()), None)

        if src_path is None:
            st.error("Nie znaleziono pliku **avocado.csv** w paczce projektu. Wgraj własny CSV/JSON.")
            return pd.DataFrame(), "(brak demo)"

        df = pd.read_csv(src_path)
        st.caption(f"Załadowano: {src_path}")
        st.dataframe(df.head(20), use_container_width=True)
        st.session_state["df_columns"] = list(df.columns)
        return df, "avocado"


# ==============================
# WYBÓR / PODPOWIEDŹ TARGETU (uniwersalna)
# ==============================
def show_detected_target(
    auto_target: Optional[str],
    columns: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Ustala kolumnę celu. Heurystyka nazw: ['target','y','label','class','price','amount','value'].
    Zawsze pozwala użytkownikowi wybrać kolumnę.
    """
    cols = columns or st.session_state.get("df_columns", [])
    st.subheader("🎯 Kolumna celu (target)")

    prefer = ["target", "y", "label", "class", "price", "amount", "value"]
    if not auto_target and cols:
        lower = {c.lower(): c for c in cols}
        for k in prefer:
            if k in lower:
                auto_target = lower[k]
                break

    if cols:
        default_ix = cols.index(auto_target) if (auto_target in cols) else 0
        picked = st.selectbox("Wybierz kolumnę celu", options=cols, index=default_ix)
        return picked

    st.warning("Nie mogę ustalić kolumn — wczytaj dane.")
    return None


# ==============================
# SIDEBAR Z USTAWIENIAMI (LLM zawsze ON, bez crasha gdy brak secrets)
# ==============================
def sidebar_config(
    available_ml: List[str],
    default_engine: str = "auto",
    show_eda_engine: bool = True,
) -> Dict[str, Any]:
    """
    Panel boczny. LLM zawsze aktywne (bez checkboxa).
    Status „Klucz dodany” pojawia się tylko przy poprawnym formacie klucza lub gdy istnieje poprawny klucz w secrets.
    """
    import re

    st.sidebar.header("Ustawienia")

    if show_eda_engine:
        st.sidebar.selectbox(
            "Silnik EDA",
            ["Szybkie podsumowanie", "Rozkłady", "Korelacje"],
            key="eda_engine",
        )

    if default_engine not in available_ml and available_ml:
        default_engine = available_ml[0]

    st.sidebar.subheader("Model")
    st.sidebar.selectbox(
        "Silnik ML",
        available_ml,
        index=max(0, available_ml.index(default_engine)),
        key="ml_engine",
    )

    st.sidebar.subheader("Wielkość danych")
    st.sidebar.selectbox(
        "Zakres danych do treningu",
        ["Cały zbiór", "Próbka 5k", "Próbka 1k"],
        key="data_sampler",
    )

    # LLM – zawsze aktywne, bez crasha gdy brak secrets.toml
    with st.sidebar.expander("🔑 Integracja LLM"):
        st.text_input(
            "Klucz API",
            key="llm_api_key",
            type="password",
            placeholder="sk-... lub sk-proj-...",
            help="Klucz trzymamy tylko w bieżącej sesji (st.session_state).",
        )

        def _looks_like_openai_key(s: str) -> bool:
            return bool(re.match(r"^(sk-|sk-proj-)[A-Za-z0-9_-]{10,}$", (s or "").strip()))

        try:
            secrets_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            secrets_key = ""

        raw_key = (st.session_state.get("llm_api_key") or "").strip()
        has_key = _looks_like_openai_key(raw_key) or _looks_like_openai_key(secrets_key)

        if has_key:
            st.success("Klucz dodany (sesja/secrets).")
        else:
            st.warning("Brak prawidłowego klucza. Wpisz swój klucz API.")

    data: Dict[str, Any] = {
        "ml_engine": st.session_state.get("ml_engine", default_engine),
        "data_sampler": st.session_state.get("data_sampler", "Cały zbiór"),
        "llm_enabled": True,
        "llm_api_key": raw_key if _looks_like_openai_key(raw_key) else "",
        "llm_prompt": "",
    }
    if show_eda_engine:
        data["eda_engine"] = st.session_state.get("eda_engine", "Szybkie podsumowanie")

    return data


# ==============================
# LISTA ZAPISANYCH RUNÓW
# ==============================
def list_saved_runs(out_dir: str | Path = "tmiv_out") -> List[str]:
    base = Path(out_dir)
    if not base.exists():
        return []
    runs = [p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")]
    runs.sort()
    return runs
