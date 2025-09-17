from __future__ import annotations

# ==============================
# IMPORTY — standard / third-party / projekt
# ==============================
import io
import json
import math
import os
import re
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZipFile, ZIP_DEFLATED

from plotly.subplots import make_subplots
import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)


# ===== Konfiguracja/komponenty UI =====
try:
    from config.settings import get_settings
except Exception:
    def get_settings():
        class _S:
            sample_data_path = None
        return _S()

# ===== importy modułów aplikacji =====
from frontend.ui_components import header, dataset_selector
from backend.eda_integration import quick_eda_summary  # jeśli nieużywane, można usunąć
from backend.ml_integration import (
    detect_problem_type,
    export_visualizations,
    train_sklearn,
)
from db.db_utils import (
    clear_history,
    ensure_db,
    export_history_csv,
    get_history,
    log_run,
    migrate_runs_table,
)

# --- Ładowanie klucza OpenAI z różnych źródeł ---
# (opcjonalnie) .env
try:
    from dotenv import load_dotenv
    load_dotenv()  # wczyta plik .env jeśli jest obok app.py
except Exception:
    pass

# ========= SŁOWNICZEK / OBJAŚNIENIA =========
GLOSSARY: Dict[str, str] = {
    # uczenie / walidacja
    "3-fold CV (Cross-Validation)": (
        "Walidacja krzyżowa: dane są dzielone na 3 części. Każda część raz pełni rolę walidacji, "
        "a pozostałe dwie – trenowania. Raportujemy średnią i odchylenie metryki."
    ),
    "Krzywa uczenia": (
        "Wykres jakości modelu w funkcji rozmiaru zbioru treningowego (train vs cv). "
        "Pomaga ocenić, czy model cierpi na high bias/variance i czy warto dodać dane."
    ),
    "Outliery (>3σ)": (
        "Obserwacje odstające: wartości znacznie odbiegające od średniej (|z-score|>3). "
        "Ich usunięcie może ustabilizować trening i metryki."
    ),
    "PCA 2D (podgląd)": (
        "Rzut danych do 2 wymiarów metodą PCA – tylko pogląd struktury (klastry, separowalność). "
        "Nie zmienia treningu modelu."
    ),
    "SHAP": (
        "Miara wpływu cech na predykcję. U nas liczona na żądanie i na małej próbce, aby nie spowalniać aplikacji."
    ),

    # metryki regresji
    "RMSE": "Pierwiastek z błędu średniokwadratowego – większe błędy są mocniej karane. Im niższy, tym lepiej.",
    "MAE":  "Średni błąd bezwzględny w jednostkach celu. Im niższy, tym lepiej.",
    "R²":   "Udział wariancji wyjaśnionej przez model (0–1). Im wyższy, tym lepiej.",
    "MAPE": "Średni błąd procentowy względem wartości rzeczywistych. Uwaga na wartości bliskie 0.",
    "SMAPE":(
        "Symetryczny MAPE – stabilniejszy w pobliżu zera; wartości ~0–100%. Im niższy, tym lepiej."
    ),

    # metryki klasyfikacji
    "Accuracy": "Odsetek poprawnych klasyfikacji. Może mylić przy niezbalansowanych klasach.",
    "Precision": "Udział trafnych pozytywów wśród wszystkich pozytywów modelu.",
    "Recall": "Udział wykrytych pozytywów wśród wszystkich prawdziwych pozytywów.",
    "F1_weighted": "Średnia harmoniczna precision i recall z wagami klas.",
    "ROC AUC": "Pole pod krzywą ROC – miara rozdzielczości klasyfikatora (wyższy = lepszy).",

    # przygotowanie danych
    "Winsoryzacja": (
        "Przycięcie skrajnych wartości (np. poniżej 1. i powyżej 99. percentyla), aby ograniczyć wpływ outlierów."
    ),
    "Transformacja log1p": (
        "log(1+x) – łagodzi prawostronną skośność rozkładu. Przy interpretacji prognoz trzeba odlogować."
    ),
    "Wysoka kardynalność": (
        "Kolumna kategoryczna z bardzo wieloma unikatowymi wartościami. Często grupujemy rzadkie do OTHER."
    ),
    "One-hot/encoding": (
        "Zamiana kategorii na cechy binarne (lub inne kodowanie). Może zwiększać wymiar danych."
    ),

    # EDA
    "Heatmapa korelacji": (
        "Macierz zależności liniowych (−1..1) między zmiennymi numerycznymi. Jasne pola = silne korelacje."
    ),
    "Scatter-matrix": "Macierz wykresów rozrzutu (para cech vs para cech) + histogramy na przekątnej.",
    "Mapa braków": "Mapa obecności/nieobecności wartości – pomaga znaleźć kolumny/wiersze z brakami.",
    "QQ-plot": (
        "Porównanie kwantyli danych z kwantylami rozkładu normalnego. Prosta ≈ dane ~N(μ,σ²)."
    ),
}

def glossary_box(location: str = "sidebar"):
    """Słowniczek pojęć jako lista wyboru (pokazuje tylko jedno hasło naraz)."""
    def _render():
        st.markdown("### 🧠 Słowniczek pojęć")
        terms = ["(wybierz hasło)"] + sorted(GLOSSARY.keys())
        sel = st.selectbox("Hasło", terms, index=0, key="glossary_sel")
        if sel != "(wybierz hasło)":
            st.markdown(f"**{sel}**")
            st.write(GLOSSARY.get(sel, "—"))
    if location == "sidebar":
        with st.sidebar:
            _render()
    else:
        _render()


# --- Kwantyle rozkładu normalnego: ppf z fallbackiem bez SciPy ---
try:
    from scipy.stats import norm as _scipy_norm
    def _norm_ppf(p: np.ndarray) -> np.ndarray:
        """Kwantyle rozkładu normalnego N(0,1) przy użyciu SciPy."""
        return _scipy_norm.ppf(p)
except Exception:
    import math
    def _erfinv(x: np.ndarray) -> np.ndarray:
        """Przybliżenie erfinv (odwrócona funkcja błędu) — fallback bez SciPy."""
        a = 0.147
        x = np.asarray(x, dtype=float)
        sgn = np.sign(x)
        ln = np.log(1 - x**2)
        first = 2/(np.pi*a) + ln/2
        second = ln/a
        inside = first**2 - second
        y = sgn * np.sqrt(np.sqrt(inside) - first)
        # korekta Halley'a dla lepszej dokładności
        err = np.vectorize(math.erf)(y) - x
        y = y - err / (2/np.sqrt(np.pi) * np.exp(-y*y))
        return y

    def _norm_ppf(p: np.ndarray) -> np.ndarray:
        """Kwantyle rozkładu normalnego N(0,1) przez przybliżenie erfinv."""
        return np.sqrt(2.0) * _erfinv(2.0 * np.asarray(p, dtype=float) - 1.0)

def _looks_like_openai_key(x: Optional[str]) -> bool:
    if not x or not isinstance(x, str):
        return False
    x = x.strip()
    # prosta heurystyka: sk-... albo sk-proj-...
    return bool(re.match(r"^(sk-|sk-proj-)[A-Za-z0-9_-]{16,}$", x))


def get_openai_key_from_envs() -> str:
    """
    Priorytety:
    1) st.session_state["llm_api_key"] (jeśli wcześniej wpisany)
    2) st.secrets["OPENAI_API_KEY"] (Streamlit Secrets)
    3) os.environ["OPENAI_API_KEY"] (w tym z .env po load_dotenv)
    """
    ss_key = st.session_state.get("llm_api_key") or st.session_state.get("llm_api_key_main") or ""
    try:
        secrets_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secrets_key = ""
    env_key = os.getenv("OPENAI_API_KEY", "")

    for candidate in [ss_key, secrets_key, env_key]:
        if _looks_like_openai_key(candidate):
            return candidate.strip()
    return ""


def set_openai_key_for_runtime(key: str):
    """Zapisz do session_state i do os.environ (dla bibliotek OpenAI)."""
    st.session_state["llm_api_key"] = key
    os.environ["OPENAI_API_KEY"] = key


# ==============================
# USTAWIENIA STRONY + STYL
# ==============================
st.set_page_config(page_title="TMIV — The Most Important Variables", layout="wide")

# Bezpieczny default – nadpiszemy po wczytaniu danych
perf_mode = False


# STYLE: zawijanie etykiet w st.metric
st.markdown(
    """
<style>
[data-testid="stMetricLabel"] > div {
    white-space: normal;
    word-wrap: break-word;
    text-align: center;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==============================
# POMOCNICZE
# ==============================
def _init_state():
    # Bezpieczna inicjalizacja — brak KeyError przed pierwszym treningiem
    defaults = {
        "model": None,
        "metrics": {},
        "fi_df": pd.DataFrame(),
        "meta": {},
        "X_last": None,
        "y_last": None,
        "extra_figs": {},
        "last_metrics": {},
        "_prep_info": {},
        "history_cleared": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init_state()


def _fmt_val(v: float, is_pct: bool = False, dec: int = 5) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return (f"{v:.{2 if is_pct else dec}f}%" if is_pct else f"{v:.{dec}f}")


def render_metric(
    col,
    *,
    label: str,
    value: float,
    key: str,
    lower_is_better: bool = False,
    is_pct: bool = False,
    dec: int = 5,
):
    last = st.session_state.get("last_metrics", {})
    prev = last.get(key, None)

    delta_str = None
    delta_color = "inverse" if lower_is_better else "normal"
    if prev is not None and prev == prev and value == value:
        diff = value - prev
        if prev != 0 and not math.isinf(prev):
            pct = (diff / abs(prev)) * 100.0
            delta_str = f"{'+' if diff>=0 else ''}{diff:.2f} ({'+' if pct>=0 else ''}{pct:.2f}%)"
        else:
            delta_str = f"{'+' if diff>=0 else ''}{diff:.2f}"

    col.metric(
        label=label,
        value=_fmt_val(value, is_pct=is_pct, dec=dec),
        delta=delta_str,
        delta_color=delta_color if delta_str else "off",
    )

    last[key] = value
    st.session_state["last_metrics"] = last


def safe_mape(y_true, y_pred, *, zero_policy="skip", eps=1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if zero_policy == "skip":
        mask = y_true != 0
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
    else:  # 'epsilon'
        denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
        return float(np.mean(np.abs((y_true - y_pred) / denom)))


def smape(y_true, y_pred, eps=1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def to_native(obj: Any):
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_native(x) for x in obj]
    try:
        if isinstance(obj, pd.DataFrame):
            return to_native(obj.to_dict(orient="records"))
        if isinstance(obj, (pd.Series, pd.Index)):
            return to_native(obj.tolist())
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    return obj


@st.cache_data(show_spinner=False)
def build_eda_report(df_raw: pd.DataFrame) -> dict:
    n_rows, n_cols = df_raw.shape
    miss = df_raw.isna().sum()
    miss_cols = miss[miss > 0].sort_values(ascending=False)
    miss_top = (
        (miss_cols.head(20) / max(n_rows, 1) * 100).round(2) if len(miss_cols) else pd.Series(dtype=float)
    )
    dup_rows = int(df_raw.duplicated().sum())
    constant_cols = [c for c in df_raw.columns if df_raw[c].nunique(dropna=False) <= 1]
    num_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
    cat_cols = [
        c
        for c in df_raw.columns
        if (df_raw[c].dtype == "object") or str(df_raw[c].dtype).startswith("category")
    ]
    dt_cols = [c for c in df_raw.columns if pd.api.types.is_datetime64_any_dtype(df_raw[c])]
    threshold = min(100, int(0.5 * n_rows))
    high_card = []
    for c in cat_cols:
        u = int(df_raw[c].nunique(dropna=True))
        if u > threshold:
            high_card.append({"column": c, "unique": u})
    report = {
        "shape": {"rows": int(n_rows), "cols": int(n_cols)},
        "missing_top": (
            [{"column": col, "percent": float(miss_top[col])} for col in miss_top.index] if len(miss_top) else []
        ),
        "duplicate_rows": dup_rows,
        "constant_cols": constant_cols[:50],
        "high_cardinality": high_card[:50],
        "types": {
            "numeric": {"count": len(num_cols)},
            "categorical": {"count": len(cat_cols)},
            "datetime": {"count": len(dt_cols)},
        },
    }
    return report


# ---- Altair helpers
def _escape_col(col: str) -> str:
    return str(col).replace(":", r"\:")


def _atype_col(df: pd.DataFrame, col: str) -> str:
    s = df[col]
    if is_datetime64_any_dtype(s):
        return "T"
    if is_numeric_dtype(s):
        return "Q"
    return "N"


def _tooltips(df: pd.DataFrame, cols: List[str]):
    return [alt.Tooltip(f"{_escape_col(c)}:{_atype_col(df, c)}", title=c) for c in cols]


# ---- Korelacje: heatmapa z wartościami
def _show_full_corr_heatmap(df: pd.DataFrame):
    """Pełnoekranowa mapa korelacji z wartościami w komórkach."""
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        st.info("Za mało kolumn numerycznych do korelacji.")
        return

    # lekki cache na podstawie hash-a ramki
    try:
        h = hashlib.md5(pd.util.hash_pandas_object(num.fillna(0), index=True).values).hexdigest()
    except Exception:
        h = str(len(num)) + "-" + str(tuple(num.columns))

    @st.cache_data(show_spinner=False)
    def _cached_corr(_h, _num):
        return _num.corr(numeric_only=True)

    corr = _cached_corr(h, num)

    x_labels = corr.columns.tolist()
    y_labels = corr.index.tolist()

    n = corr.shape[0]
    cell_px = 34
    height = max(350, n * cell_px + 140)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=x_labels,
            y=y_labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title=""),
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=12),
        )
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=x_labels,
        tickangle=45,
        automargin=True,
        showgrid=False,
    )
    fig.update_yaxes(
        type="category",
        categoryorder="array",
        categoryarray=y_labels,
        automargin=True,
        showgrid=False,
        autorange="reversed",
    )
    fig.update_layout(margin=dict(l=120, r=30, t=10, b=10), height=height, hovermode="closest")
    fig.update_traces(hovertemplate="%{y} ↔ %{x}<br>corr=%{z:.2f}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)


# ---- Auto wybór targetu
def _auto_pick_target(df: pd.DataFrame) -> tuple[str, str, list[str]]:
    reasons: list[str] = []
    cols = list(df.columns)
    preferred = ["AveragePrice", "target", "y", "label", "class", "price", "amount", "value"]
    for name in preferred:
        if name in cols:
            s = df[name]
            if (str(s.dtype) in ("object", "category")) or pd.api.types.is_bool_dtype(s) or s.nunique(dropna=True) <= 20:
                reasons.append(
                    f"nazwa '{name}' pasuje do listy typowych celów; liczba klas ≤ 20 ⇒ klasyfikacja"
                )
                return name, "classification", reasons
            reasons.append(
                f"nazwa '{name}' pasuje do listy typowych celów; kolumna numeryczna z wystarczającą zmiennością ⇒ regresja"
            )
            return name, "regression", reasons

    # kandydaci do klasyfikacji
    cand_cls = []
    for c in cols:
        s = df[c]
        nunq = s.nunique(dropna=True)
        if (str(s.dtype) in ("object", "category")) or (pd.api.types.is_integer_dtype(s) and 2 <= nunq <= 20):
            miss = float(s.isna().mean())
            cand_cls.append((c, nunq, miss))
    cand_cls.sort(key=lambda x: (x[1], x[2]))
    if cand_cls:
        c, nunq, miss = cand_cls[0]
        reasons.append(
            f"kolumna '{c}' wygląda na kategoryczną / int z {nunq} klasami (braki {miss:.1%}) ⇒ klasyfikacja"
        )
        return c, "classification", reasons

    # kandydaci do regresji
    cand_reg = []
    nrows = max(1, len(df))
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            nunq = s.nunique(dropna=True)
            if 10 <= nunq <= 0.95 * nrows:
                var = float(pd.to_numeric(s, errors="coerce").var(skipna=True) or 0.0)
                miss = float(s.isna().mean())
                cand_reg.append((c, var, miss, nunq))
    cand_reg.sort(key=lambda x: (-x[1], x[2]))
    if cand_reg:
        c, var, miss, nunq = cand_reg[0]
        reasons.append(
            f"kolumna '{c}' numeryczna (wariancja {var:.3g}), nie wygląda na ID (unikatów {nunq}) ⇒ regresja"
        )
        return c, "regression", reasons

    # fallback
    for c in cols:
        s = df[c]
        if not pd.api.types.is_bool_dtype(s):
            pt = "regression" if pd.api.types.is_numeric_dtype(s) else "classification"
            reasons.append(f"fallback: pierwsza nie-bool kolumna '{c}' ⇒ {pt}")
            return c, pt, reasons
    reasons.append("fallback: brak kandydatów — wybór pierwszej kolumny")
    return (cols[0] if cols else "target"), "regression", reasons


# ==============================
# DB — inicjalizacja
# ==============================
settings = get_settings()
conn = ensure_db()
try:
    migrate_runs_table(conn)
except Exception:
    pass


# ==============================
# SIDEBAR — narzędzia + health + słowniczek
# ==============================
with st.sidebar:
    st.markdown("### 🛠️ Narzędzia")
    if st.button("🧹 Wyczyść cache"):
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache wyczyszczony.")
        except Exception as e:
            st.warning(f"Nie udało się wyczyścić cache: {e}")

    # Słowniczek (jako lista wyboru)
    glossary_box("sidebar")

# ==============================
# HEADER + DANE
# ==============================
header()

# --- 🔑 Klucz OpenAI dostępny od razu na 1. stronie ---
current_key = get_openai_key_from_envs()

# Badge ze statusem klucza
c1, c2 = st.columns([0.7, 0.3])
with c1:
    st.caption("Status klucza OpenAI")
with c2:
    if _looks_like_openai_key(current_key):
        st.markdown(
            "<span style='color: white; background-color: green; padding: 4px 10px; border-radius: 10px;'>🟢 OpenAI aktywny</span>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<span style='color: white; background-color: red; padding: 4px 10px; border-radius: 10px;'>🔴 Brak klucza OpenAI</span>",
            unsafe_allow_html=True
        )

# Pole do wpisania tylko jeśli nie wykryto automatycznie
if not _looks_like_openai_key(current_key):
    st.markdown("#### 🔑 Wklej klucz OpenAI")
    typed = st.text_input(
        "Klucz OpenAI",
        placeholder="sk-... lub sk-proj-...",
        type="password",
        key="llm_api_key_main",
        help="Możesz też dodać go do .env jako OPENAI_API_KEY lub do st.secrets."
    )
    if _looks_like_openai_key(typed):
        set_openai_key_for_runtime(typed)
        st.success("Klucz OpenAI zapisany w tej sesji ✅")
else:
    set_openai_key_for_runtime(current_key)
    st.info("Wykryto klucz OpenAI z .env / środowiska / st.secrets.")

# --- Wybór i wczytanie danych (tylko raz) ---
df, dataset_name = dataset_selector(settings.sample_data_path)
if df is None or df.empty:
    st.stop()

# Automatyczny „tryb szybki” tylko dla EDA (bez przełącznika)
# Heurystyka: jeśli >20k wierszy, EDA działa na próbce.
perf_mode = len(df) > 20_000
df_view = df.sample(20_000, random_state=42) if perf_mode else df


# ==============================
# 🔬 EDA
# ==============================
st.markdown("## 🔬 EDA")
tgt_auto, _ptype_auto, _tgt_why = _auto_pick_target(df_view)

EDA_MODES = [
    "Szybkie podsumowanie",
    "Rozkłady (histogramy)",
    "Korelacje (heatmapa)",
    "Boxplot: kategoria → target",
    "Scatter: num → target (+trend)",
    "Szereg czasowy (data → target)",
    "Top kategorie (częstości)",
    "Krzywa uczenia (z ostatniego modelu)",
    "PCA 2D (z ostatniego modelu)",
    "Parowy podgląd (2 zmienne)",
    "Macierz par (scatter-matrix)",
    "Mapa braków (missingness)",
    "Rozkład targetu wg kategorii (top-k)",
    "QQ-plot (normalność)",
]

c_main, c_ctrl = st.columns([3, 1])
with c_ctrl:
    eda_choice = st.selectbox("Widok EDA", EDA_MODES, help="Wybierz widok analizy danych.")
with c_main:
    try:
        summary = quick_eda_summary(df_view)
    except Exception:
        summary = None

if eda_choice == "Szybkie podsumowanie":
    if summary is None or (hasattr(summary, "empty") and summary.empty) or (hasattr(summary, "__len__") and len(summary) == 0):
        st.info("Brak podsumowania – pokazuję podstawowe informacje.")
        st.write(df_view.describe(include="all").transpose())
    else:
        summary

elif eda_choice == "Rozkłady (histogramy)":
    num_cols = [c for c in df_view.columns if pd.api.types.is_numeric_dtype(df_view[c])]
    if not num_cols:
        st.info("Brak kolumn numerycznych.")
    else:
        sel = st.multiselect("Kolumny numeryczne", num_cols, default=num_cols[:6])
        bins = st.slider("Liczba koszy (bins)", 10, 80, 40)
        for c in sel:
            st.plotly_chart(px.histogram(df_view, x=c, nbins=bins, title=None), use_container_width=True)

elif eda_choice == "Korelacje (heatmapa)":
    _show_full_corr_heatmap(df_view)

elif eda_choice == "Boxplot: kategoria → target":
    if tgt_auto not in df_view.columns or not pd.api.types.is_numeric_dtype(df_view[tgt_auto]):
        st.info(f"Target `{tgt_auto}` nie jest numeryczny – wybierz inny zbiór/target.")
    else:
        cat_cols = [c for c in df_view.columns if c != tgt_auto and df_view[c].dtype.name in {"object", "category", "bool"}]
        if not cat_cols:
            st.info("Brak kolumn kategorycznych.")
        else:
            cat = st.selectbox("Kategoria", cat_cols, key="eda_box_cat2")
            topk = st.slider("Ile kategorii (TOP)", 3, 20, 12)
            vc = df_view[cat].astype(str).value_counts().head(topk).index.tolist()
            tmp = df_view[df_view[cat].astype(str).isin(vc)][[cat, tgt_auto]].dropna()
            if tmp.empty:
                st.info("Brak danych po filtrze.")
            else:
                # box + średnia (punkt)
                base = alt.Chart(tmp).encode(
                    x=alt.X(f"{_escape_col(cat)}:N", sort="-y", title=cat),
                    y=alt.Y(f"{_escape_col(tgt_auto)}:Q", title=tgt_auto),
                    tooltip=_tooltips(df_view, [cat, tgt_auto]),
                )
                st.altair_chart(
                    (base.mark_boxplot() + base.mark_point(color="red", filled=True, size=60, opacity=0.7).encode())
                    .properties(),
                    use_container_width=True,
                )

elif eda_choice == "Scatter: num → target (+trend)":
    if tgt_auto not in df_view.columns or not pd.api.types.is_numeric_dtype(df_view[tgt_auto]):
        st.info(f"Target `{tgt_auto}` nie jest numeryczny – wybierz inny zbiór/target.")
    else:
        num_cols = [c for c in df_view.columns if c != tgt_auto and pd.api.types.is_numeric_dtype(df_view[c])]
        if not num_cols:
            st.info("Brak dodatkowych kolumn numerycznych.")
        else:
            feat = st.selectbox("Cecha numeryczna", num_cols, key="eda_scatter_feat2")
            tmp = df_view[[feat, tgt_auto]].dropna()
            if len(tmp) > (3000 if perf_mode else 5000):
                tmp = tmp.sample(3000 if perf_mode else 5000, random_state=42)
            st.altair_chart(
                alt.Chart(tmp)
                .mark_circle(size=24, opacity=0.5)
                .encode(
                    x=alt.X(f"{_escape_col(feat)}:Q", title=feat),
                    y=alt.Y(f"{_escape_col(tgt_auto)}:Q", title=tgt_auto),
                    tooltip=_tooltips(df_view, [feat, tgt_auto]),
                )
                .transform_regression(feat, tgt_auto)
                .mark_line(),
                use_container_width=True,
            )

elif eda_choice == "Szereg czasowy (data → target)":
    dt_cols = [c for c in df_view.columns if pd.api.types.is_datetime64_any_dtype(df_view[c])]
    if not dt_cols:
        st.info("Brak kolumn typu data/czas.")
    elif tgt_auto not in df_view.columns or not pd.api.types.is_numeric_dtype(df_view[tgt_auto]):
        st.info(f"Target `{tgt_auto}` nie jest numeryczny – linia czasowa pominięta.")
    else:
        dtc = st.selectbox("Kolumna daty", dt_cols, key="eda_time_col2")
        tmp = df_view[[dtc, tgt_auto]].dropna().sort_values(dtc)
        if tmp.empty:
            st.info("Brak danych po czyszczeniu.")
        else:
            if len(tmp) > (5000 if perf_mode else 10000):
                step = max(1, len(tmp) // (5000 if perf_mode else 10000))
                tmp = tmp.iloc[::step]
            st.altair_chart(
                alt.Chart(tmp)
                .mark_line()
                .encode(
                    x=alt.X(f"{_escape_col(dtc)}:T", title=dtc),
                    y=alt.Y(f"{_escape_col(tgt_auto)}:Q", title=tgt_auto),
                    tooltip=_tooltips(df_view, [dtc, tgt_auto]),
                )
                .interactive(),
                use_container_width=True,
            )

elif eda_choice == "Top kategorie (częstości)":
    cat_cols = [c for c in df_view.columns if df_view[c].dtype.name in {"object", "category", "bool"}]
    if not cat_cols:
        st.info("Brak kolumn kategorycznych.")
    else:
        cols_pick = st.multiselect("Kolumny", cat_cols[:10], default=cat_cols[:3])
        topk = st.slider("TOP-k dla każdej kolumny", 3, 20, 12)
        for col in cols_pick[:6]:
            top = df_view[col].astype(str).value_counts().head(topk).reset_index()
            top.columns = [col, "count"]
            st.plotly_chart(px.bar(top, x=col, y="count", title=col), use_container_width=True)

elif eda_choice == "Parowy podgląd (2 zmienne)":
    cols_num = [c for c in df_view.columns if pd.api.types.is_numeric_dtype(df_view[c])]
    if len(cols_num) < 2:
        st.info("Potrzeba co najmniej dwóch kolumn numerycznych.")
    else:
        c1_, c2_ = st.columns(2)
        with c1_:
            xcol = st.selectbox("Oś X", cols_num, key="eda_pair_x")
        with c2_:
            ycol = st.selectbox("Oś Y", [c for c in cols_num if c != xcol], key="eda_pair_y")
        tmp = df_view[[xcol, ycol]].dropna()
        if len(tmp) > (4000 if perf_mode else 6000):
            tmp = tmp.sample(4000 if perf_mode else 6000, random_state=42)
        st.altair_chart(
            alt.Chart(tmp)
            .mark_circle(size=20, opacity=0.5)
            .encode(
                x=alt.X(f"{_escape_col(xcol)}:Q", title=xcol),
                y=alt.Y(f"{_escape_col(ycol)}:Q", title=ycol),
                tooltip=_tooltips(df_view, [xcol, ycol]),
            )
            .interactive(),
            use_container_width=True,
        )

elif eda_choice == "Macierz par (scatter-matrix)":
    num_cols = [c for c in df_view.columns if pd.api.types.is_numeric_dtype(df_view[c])]
    if len(num_cols) < 2:
        st.info("Potrzeba co najmniej dwóch kolumn numerycznych.")
    else:
        pick = st.multiselect("Wybierz kolumny (max 6 dla czytelności)", num_cols, default=num_cols[:4])
        if len(pick) >= 2:
            df_sm = df_view[pick].dropna()
            if len(df_sm) > (3000 if perf_mode else 8000):
                df_sm = df_sm.sample(3000 if perf_mode else 8000, random_state=42)
            fig = px.scatter_matrix(df_sm, dimensions=pick, title=None)
            fig.update_traces(diagonal_visible=True, showupperhalf=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Zaznacz co najmniej dwie kolumny.")

elif eda_choice == "Mapa braków (missingness)":
    miss_df = df_view.isna()
    if miss_df.empty:
        st.info("Brak danych.")
    else:
        # ogranicz do max 1000 wierszy dla wydajności
        if len(miss_df) > (1000 if perf_mode else 3000):
            miss_df = miss_df.sample(1000 if perf_mode else 3000, random_state=42)
        fig = go.Figure(data=go.Heatmap(
            z=miss_df.values.astype(int),
            x=list(miss_df.columns),
            y=[str(i) for i in miss_df.index],
            colorscale=[[0, "#1f77b4"], [1, "#ff7f0e"]],
            showscale=False,
        ))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

elif eda_choice == "Rozkład targetu wg kategorii (top-k)":
    # wybierz kategorię i pokaż box+średnia dla targetu
    cat_cols = [c for c in df_view.columns if c != tgt_auto and df_view[c].dtype.name in {"object", "category", "bool"}]
    if tgt_auto not in df_view.columns or not pd.api.types.is_numeric_dtype(df_view[tgt_auto]):
        st.info(f"Target `{tgt_auto}` nie jest numeryczny – wybierz inny widok.")
    elif not cat_cols:
        st.info("Brak kolumn kategorycznych.")
    else:
        cat = st.selectbox("Kategoria", cat_cols, key="eda_target_cat")
        k = st.slider("TOP-k kategorii", 3, 30, 12)
        vc = df_view[cat].astype(str).value_counts().head(k).index.tolist()
        tmp = df_view[df_view[cat].astype(str).isin(vc)][[cat, tgt_auto]].dropna()
        if tmp.empty:
            st.info("Brak danych po filtrze.")
        else:
            base = alt.Chart(tmp).encode(
                x=alt.X(f"{_escape_col(cat)}:N", sort="-y", title=cat),
                y=alt.Y(f"{_escape_col(tgt_auto)}:Q", title=tgt_auto),
                tooltip=_tooltips(df_view, [cat, tgt_auto]),
            )
            st.altair_chart(
                (base.mark_boxplot() +
                 base.transform_aggregate(mean_val=f"mean({_escape_col(tgt_auto)})", groupby=[cat])
                     .mark_point(color="orange", filled=True, size=80, opacity=0.9)
                     .encode(y="mean_val:Q"))
                , use_container_width=True
            )

elif eda_choice == "Krzywa uczenia (z ostatniego modelu)":
    mdl = st.session_state.get("model")
    X_last = st.session_state.get("X_last")
    y_last = st.session_state.get("y_last")
    meta = st.session_state.get("meta") or {}
    if mdl is None or X_last is None or y_last is None:
        st.info("Brak wytrenowanego modelu w bieżącej sesji.")
    else:
        fig = _plot_learning_curve(mdl, X_last, y_last, task=(meta.get("problem_type") or ""))
        st.plotly_chart(fig, use_container_width=True)

elif eda_choice == "PCA 2D (z ostatniego modelu)":
    X_last = st.session_state.get("X_last")
    y_last = st.session_state.get("y_last")
    meta = st.session_state.get("meta") or {}
    if X_last is None or y_last is None:
        st.info("Brak danych z ostatniego treningu.")
    else:
        fig = _pca_preview_2d(X_last, y_last, task=(meta.get("problem_type") or ""))
        st.plotly_chart(fig, use_container_width=True)
           

elif eda_choice == "QQ-plot (normalność)":
    # wybierz kolumnę numeryczną
    num_cols = [c for c in df_view.columns if pd.api.types.is_numeric_dtype(df_view[c])]
    if not num_cols:
        st.info("Brak kolumn numerycznych.")
    else:
        col = st.selectbox("Kolumna numeryczna", num_cols, key="eda_qq_col")
        s = pd.to_numeric(df_view[col], errors="coerce").dropna()
        if len(s) < 10:
            st.info("Za mało danych do QQ-plot.")
        else:
            s_sorted = np.sort(s.values)
            n = len(s_sorted)
            # teoretyczne kwantyle dla N(0,1)
            probs = (np.arange(1, n + 1) - 0.5) / n
            theor = _norm_ppf(probs)  # kwantyle normalne

            mu = np.mean(s_sorted)
            sigma = np.std(s_sorted)

            fig = go.Figure()
            fig.add_scatter(x=theor, y=s_sorted, mode="markers", name="Dane", opacity=0.7)
            fig.add_scatter(x=theor, y=(theor * sigma + mu), mode="lines", name="Linia referencyjna")
            fig.update_layout(
                title=f"QQ-plot — {col}",
                xaxis_title="Teoretyczne kwantyle N(0,1)",
                yaxis_title="Dane (posortowane)"
            )
            st.plotly_chart(fig, use_container_width=True)

# pomocnik dla QQ-plot — erfinv (bez SciPy)
def erfinv(x):
    # przybliżenie Winitzki + Halley step (wystarczające do QQ-plot)
    a = 0.147
    sgn = np.sign(x)
    ln = np.log(1 - x**2)
    first = 2/(np.pi*a) + ln/2
    second = ln/a
    inside = first**2 - second
    y = sgn * np.sqrt(np.sqrt(inside) - first)
    # pojedyncza korekta Halley'a
    err = math.erf(y) - x
    y = y - err / (2/np.sqrt(np.pi) * np.exp(-y*y))
    return y

# ==============================
# HELPERY: outliery / learning curve / PCA
# ==============================
def _remove_outliers_sigma(df: pd.DataFrame, target: str, sigma: float = 3.0) -> pd.DataFrame:
    """Usuwa wiersze, które dla dowolnej kolumny numerycznej mają |z-score| > sigma."""
    tmp = df.copy()
    num_cols = [c for c in tmp.columns if pd.api.types.is_numeric_dtype(tmp[c])]
    if not num_cols:
        return tmp
    for c in num_cols:
        s = pd.to_numeric(tmp[c], errors="coerce")
        mu = float(np.nanmean(s))
        sd = float(np.nanstd(s))
        if sd <= 0 or not np.isfinite(sd):
            continue
        z = (s - mu) / sd
        mask_ok = np.abs(z) <= sigma
        tmp = tmp.loc[mask_ok.fillna(False)]
    return tmp


def _plot_learning_curve(model, X: pd.DataFrame, y: pd.Series | np.ndarray, task: str, random_state: int = 42):
    """Zwraca Figure z krzywą uczenia (train vs CV), bez wyświetlania."""
    try:
        if isinstance(X, pd.DataFrame) and len(X) > 5000:
            Xs = X.sample(5000, random_state=random_state)
            ys = pd.Series(y).loc[Xs.index]
        else:
            Xs, ys = X, y
    except Exception:
        Xs, ys = X, y

    scorer = "neg_root_mean_squared_error" if "reg" in (task or "").lower() else "f1_weighted"
    sizes = np.linspace(0.1, 1.0, 5)

    try:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model, X=Xs, y=ys, train_sizes=sizes, cv=3, scoring=scorer, n_jobs=-1, shuffle=True, random_state=random_state
        )
    except Exception:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model, X=Xs, y=ys, train_sizes=sizes, cv=3, n_jobs=-1, shuffle=True, random_state=random_state
        )

    tr = np.mean(train_scores, axis=1)
    te = np.mean(test_scores, axis=1)
    metric_name = "RMSE" if "reg" in (task or "").lower() else "F1_weighted"
    if metric_name == "RMSE":
        tr = -tr
        te = -te

    fig = px.line(
        x=np.concatenate([train_sizes, train_sizes]),
        y=np.concatenate([tr, te]),
        color=(["train"] * len(train_sizes)) + (["cv"] * len(train_sizes)),
        labels={"x": "Rozmiar próby", "y": metric_name, "color": "Zbiór"},
        title="Krzywa uczenia",
    )
    return fig


def _pca_preview_2d(X: pd.DataFrame, y: pd.Series | np.ndarray, task: str, random_state: int = 42):
    """Zwraca Figure z szybkim rzutem PCA do 2D, bez wyświetlania."""
    Xp = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    num = Xp.select_dtypes(include=[np.number])
    cat = Xp.select_dtypes(exclude=[np.number])

    if not cat.empty:
        cat_enc = pd.get_dummies(cat.astype(str), drop_first=True)
        M = pd.concat([num, cat_enc], axis=1)
    else:
        M = num.copy()

    if M.empty:
        return px.scatter(pd.DataFrame(columns=["PC1", "PC2", "y"]), x="PC1", y="PC2", title="PCA — brak danych")

    idx = M.index
    if len(M) > 3000:
        idx = M.sample(3000, random_state=random_state).index
    M = M.loc[idx]
    yv = pd.Series(y).loc[idx]

    M = M.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pca = PCA(n_components=2, random_state=random_state)
    comp = pca.fit_transform(M.values)
    dfp = pd.DataFrame({"PC1": comp[:, 0], "PC2": comp[:, 1], "y": yv.values})

    if "reg" in (task or "").lower():
        try:
            bins = pd.qcut(dfp["y"], q=5, duplicates="drop")
            dfp["y_bin"] = bins.astype(str)
            color_col = "y_bin"
        except Exception:
            color_col = "y"
    else:
        color_col = "y"

    fig = px.scatter(dfp, x="PC1", y="PC2", color=color_col, opacity=0.7, title="PCA — podgląd 2D")
    return fig

def make_model_report_figure(
    dataset_name: str,
    target_name: str,
    problem_type: str,
    metrics: Dict[str, Any],
    fi_df: Optional[pd.DataFrame],
    prep_info: Dict[str, Any],
    engine_name: str = "auto",
) -> go.Figure:
    """Czytelniejszy mini-raport: tabela + TOP cechy (bar)."""

    # --- Tabela po lewej: zgrabne wiersze
    def _kv(label: str, value: Any) -> tuple[str, str]:
        if isinstance(value, float):
            v = f"{value:.6g}"
        else:
            v = str(value)
        return f"**{label}**", v

    info_rows = [
        _kv("Zbiór", dataset_name or "-"),
        _kv("Target", target_name or "-"),
        _kv("Problem", (problem_type or "-")),
        _kv("Silnik", engine_name or "-"),
        _kv("Przetw.", f"usun. state kol.: {prep_info.get('dropped_state_cols', 0)}"),
    ]

    # metryki — tylko kilka najważniejszych; resztę skracamy
    metric_rows = []
    for k, v in (metrics or {}).items():
        try:
            metric_rows.append(_kv(k, float(v)))
        except Exception:
            metric_rows.append(_kv(k, v))

    table_header = ["", ""]
    table_cells_left = [*[r[0] for r in info_rows], "—", *[r[0] for r in metric_rows]]
    table_cells_right = [*[r[1] for r in info_rows], "—", *[r[1] for r in metric_rows]]

    # --- Feature importance
    fi_top = pd.DataFrame()
    if isinstance(fi_df, pd.DataFrame) and not fi_df.empty and {"feature","importance"}.issubset(fi_df.columns):
        fi_top = (
            fi_df.head(15)  # mniej, ale czytelnie
                .copy()
        )
        fi_top["feature"] = fi_top["feature"].astype(str)
        # odwracamy kolejność, żeby najwyżej była TOP cecha
        fi_top = fi_top.iloc[::-1]

    # --- Subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "table"}, {"type": "xy"}]],
        column_widths=[0.46, 0.54],
        horizontal_spacing=0.08,
        subplot_titles=("Podsumowanie", "Najważniejsze cechy"),
    )

    # Tabela
    fig.add_trace(
        go.Table(
            header=dict(
                values=table_header,
                fill_color="#1f2937",  # dark slate
                font=dict(color="white", size=13),
                align="left",
                height=30,
            ),
            cells=dict(
                values=[table_cells_left, table_cells_right],
                align="left",
                fill_color=[["#0f172a" if i%2==0 else "#111827" for i in range(len(table_cells_left))],
                           ["#0f172a" if i%2==0 else "#111827" for i in range(len(table_cells_right))]],
                font=dict(color="rgba(255,255,255,0.95)", size=12),
                height=28,
            ),
            columnwidth=[0.50, 0.50],
        ),
        row=1, col=1
    )

    # Bar chart (jeśli są dane)
    if not fi_top.empty:
        fig.add_trace(
            go.Bar(
                x=fi_top["importance"].values,
                y=fi_top["feature"].values,
                orientation="h",
                text=[f"{v:.3f}" for v in fi_top["importance"].values],
                textposition="outside",  # wartości na końcach słupków
                marker=dict(color="#ff6b4a"),  # kontrast z dark
                hovertemplate="cecha=%{y}<br>importance=%{x:.6f}<extra></extra>",
            ),
            row=1, col=2
        )
        fig.update_yaxes(
            automargin=True,
            tickfont=dict(size=12),
            row=1, col=2
        )
        fig.update_xaxes(
            showgrid=True, gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            row=1, col=2
        )

    # --- Layout globalny: więcej "oddechu"
    fig.update_layout(
        template="plotly_dark",
        height=720,
        width=1280,
        bargap=0.18,
        margin=dict(l=20, r=20, t=70, b=30),
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        hoverlabel=dict(bgcolor="#111827"),
        uniformtext=dict(minsize=10, mode="hide"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Dodatkowe marginesy, żeby nic się nie “zlewało”
    fig.update_layout(
        annotations=[a.update(font=dict(size=14)) for a in fig.layout.annotations]
    )
    fig.update_yaxes(title=None, row=1, col=2)
    fig.update_xaxes(title=None, row=1, col=2)

    return fig

def save_model_report(
    *,
    fig: go.Figure,
    out_dir: Path
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Zapis raportu do PNG (jeśli dostępne Kaleido) oraz fallback do HTML.
    Zwraca: (png_path, html_path)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "model_report.png"
    html_path = out_dir / "model_report.html"

    png_ok = False
    try:
        # Wymaga: pip install -U kaleido
        fig.write_image(str(png_path), scale=2, width=1200, height=700)
        png_ok = True
    except Exception:
        png_ok = False

    try:
        fig.write_html(str(html_path), include_plotlyjs="cdn")
    except Exception:
        html_path = None

    return (png_path if png_ok else None), html_path

# ==============================
# Pełny raport wizualny (PNG/HTML) + helper zapisu dowolnych figur
# ==============================
def _save_plotly(fig: go.Figure, out_dir: Path, base_name: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Zapisuje wykres do PNG (jeśli Kaleido) oraz HTML (fallback).
    Zwraca: (png_path|None, html_path|None)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{base_name}.png"
    html_path = out_dir / f"{base_name}.html"

    png_ok = False
    try:
        fig.write_image(str(png_path), scale=2, width=1200, height=700)
        png_ok = True
    except Exception:
        pass

    try:
        fig.write_html(str(html_path), include_plotlyjs="cdn")
    except Exception:
        html_path = None

    return (png_path if png_ok else None), html_path


def build_and_save_full_reports(out_dir: Path) -> Dict[str, List[str]]:
    """
    Buduje i zapisuje komplet raportów graficznych na bazie bieżącej sesji:
      - model_report (tabela + TOP cechy) — już masz, ale robimy go na świeżo
      - feature_importance (bar)
      - residuals_hist (regresja)
      - confusion_matrix (klasyfikacja binarna)
      - roc_curve i precision_recall (jeśli mamy proby)
      - learning_curve i pca_preview (jeśli były włączone lub da się policzyć na szybko)

    Zwraca słownik: {nazwa_raportu: [lista_plików]}
    """
    created: Dict[str, List[str]] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    model = st.session_state.get("model")
    metrics = st.session_state.get("metrics") or {}
    fi_df: pd.DataFrame = st.session_state.get("fi_df") if isinstance(st.session_state.get("fi_df"), pd.DataFrame) else pd.DataFrame()
    meta = st.session_state.get("meta") or {}
    X_last: Optional[pd.DataFrame] = st.session_state.get("X_last")
    y_last = st.session_state.get("y_last")

    if model is None or X_last is None:
        return created  # nic do roboty

    problem_type = (meta.get("problem_type") or "").lower()
    dataset_name = meta.get("dataset", "") or "dataset"
    target_name = meta.get("target", "") or "target"

    # 1) Główny raport (Twoja funkcja)
    try:
        main_fig = make_model_report_figure(
            dataset_name=dataset_name,
            target_name=target_name,
            problem_type=meta.get("problem_type", ""),
            metrics=metrics,
            fi_df=fi_df,
            prep_info=st.session_state.get("_prep_info") or {},
            engine_name=meta.get("engine", "auto"),
        )
        p1, h1 = save_model_report(fig=main_fig, out_dir=out_dir)
        files = [str(p) for p in [p1, h1] if p]
        if files: created["model_report"] = files
    except Exception:
        pass

    # 2) Feature importance (bardziej surowy bar)
    try:
        if not fi_df.empty and {"feature", "importance"}.issubset(fi_df.columns):
            top = fi_df.head(25).iloc[::-1]  # h-bar od góry
            fig_fi = go.Figure(go.Bar(
                x=top["importance"].values, y=top["feature"].astype(str).values,
                orientation="h", hovertemplate="cecha=%{y}<br>importance=%{x:.6f}<extra></extra>"
            ))
            fig_fi.update_layout(title="Feature importance (TOP 25)", height=700, template="plotly_dark", margin=dict(l=120,r=30,t=60,b=20))
            p, h = _save_plotly(fig_fi, out_dir, "feature_importance")
            files = [str(x) for x in [p, h] if x]
            if files: created["feature_importance"] = files
    except Exception:
        pass

    # 3) Residua (tylko regresja)
    try:
        if "reg" in problem_type and y_last is not None:
            y_true = pd.Series(y_last, index=X_last.index)
            y_pred = pd.Series(model.predict(X_last), index=X_last.index)
            res = (y_true - y_pred).dropna()
            if not res.empty:
                fig_res = px.histogram(res.to_frame("residual"), x="residual", nbins=40, title="Rozkład residuów", template="plotly_dark")
                fig_res.update_layout(height=500, margin=dict(l=20,r=20,t=60,b=20))
                p, h = _save_plotly(fig_res, out_dir, "residuals_hist")
                files = [str(x) for x in [p, h] if x]
                if files: created["residuals_hist"] = files
    except Exception:
        pass

    # 4) Klasyfikacja: macierz pomyłek, ROC i PR
    try:
        if "class" in problem_type and y_last is not None:
            y_true = pd.Series(y_last, index=X_last.index).values
            # proby
            proba = None
            if hasattr(model, "predict_proba"):
                P = model.predict_proba(X_last)
                proba = P[:, 1] if (P.ndim == 2 and P.shape[1] >= 2) else P.ravel()
            # macierz pomyłek przy progu 0.5 (lub predict)
            if proba is not None:
                y_pred = (proba >= 0.5).astype(int)
            else:
                y_pred = model.predict(X_last)
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"],
                colorscale="Blues", showscale=True, text=cm, texttemplate="%{text}"
            ))
            fig_cm.update_layout(title="Macierz pomyłek (próg 0.5)", height=500, template="plotly_dark", margin=dict(l=100,r=20,t=60,b=20))
            p, h = _save_plotly(fig_cm, out_dir, "confusion_matrix")
            files = [str(x) for x in [p, h] if x]
            if files: created["confusion_matrix"] = files

            # ROC/PR, jeśli mamy proby
            if proba is not None and len(np.unique(y_true)) == 2:
                fpr, tpr, _ = roc_curve(y_true, proba)
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC={roc_auc:.3f}"))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="baseline", line=dict(dash="dash")))
                fig_roc.update_layout(title="ROC curve", xaxis_title="FPR", yaxis_title="TPR", height=500, template="plotly_dark")
                p, h = _save_plotly(fig_roc, out_dir, "roc_curve")
                files = [str(x) for x in [p, h] if x]
                if files: created["roc_curve"] = files

                prec, rec, _ = precision_recall_curve(y_true, proba)
                ap = average_precision_score(y_true, proba)
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR curve"))
                fig_pr.update_layout(title=f"Precision–Recall curve (AP={ap:.3f})",
                                    xaxis_title="Recall", yaxis_title="Precision",
                                    height=500, template="plotly_dark")
                p, h = _save_plotly(fig_pr, out_dir, "precision_recall_curve")
                files = [str(x) for x in [p, h] if x]
                if files: created["precision_recall_curve"] = files
    except Exception:
        pass

    # 5) Learning curve i PCA — skorzystaj z istniejących extra_figs, a jeśli nie ma spróbuj policzyć
    try:
        extra_figs = st.session_state.get("extra_figs", {})
        # LC
        if "lc" in extra_figs and isinstance(extra_figs["lc"], go.Figure):
            p, h = _save_plotly(extra_figs["lc"], out_dir, "learning_curve")
            files = [str(x) for x in [p, h] if x]
            if files: created["learning_curve"] = files
        else:
            # szybki fallback (bez przełączania UI)
            lc_fig = _plot_learning_curve(model, X_last, y_last, task=problem_type)
            p, h = _save_plotly(lc_fig, out_dir, "learning_curve")
            files = [str(x) for x in [p, h] if x]
            if files: created["learning_curve"] = files

        # PCA
        if "pca" in extra_figs and isinstance(extra_figs["pca"], go.Figure):
            p, h = _save_plotly(extra_figs["pca"], out_dir, "pca_preview")
            files = [str(x) for x in [p, h] if x]
            if files: created["pca_preview"] = files
        else:
            pca_fig = _pca_preview_2d(X_last, y_last, task=problem_type)
            p, h = _save_plotly(pca_fig, out_dir, "pca_preview")
            files = [str(x) for x in [p, h] if x]
            if files: created["pca_preview"] = files
    except Exception:
        pass

    # 6) Krótki README.md z meta
    try:
        readme = out_dir / "README_reports.md"
        lines = [
            f"# TMIV — Raport z treningu",
            f"- Zbiór: **{dataset_name}**",
            f"- Target: **{target_name}**",
            f"- Problem: **{meta.get('problem_type', '-') }**",
            f"- Silnik: **{meta.get('engine', 'auto')}**",
            "",
            "## Metryki",
        ]
        for k, v in (metrics or {}).items():
            if isinstance(v, (int, float)):
                lines.append(f"- **{k}**: {v:.6g}")
            else:
                lines.append(f"- **{k}**: {v}")
        readme.write_text("\n".join(lines), encoding="utf-8")
        created.setdefault("readme", []).append(str(readme))
    except Exception:
        pass

    return created

# ==============================
# 🏋️‍♂️ Trening — UI
# ==============================
st.markdown("## 🏋️‍♂️ Trening")
st.caption("**Silnik ML:** Auto")
with st.form("train_form", clear_on_submit=False):
    train_btn = st.form_submit_button("🚀 Wytrenuj model", type="primary")
    # Pierwszy rząd
    row1 = st.columns([1, 1, 1])
    with row1[0]:
        sample_mode = st.selectbox(
            "Rozmiar próbki",
            ["Cały zbiór", "Próbka 5k", "Próbka 1k"],
            index=0,
            help="Kontroluje tylko dane do trenowania modelu. "
                 "EDA (analiza wstępna) korzysta z Trybu szybkiego w sidebarze."
        )
    with row1[1]:
        run_cv = st.checkbox(
            "3-fold CV (stabilność)",
            value=False,
            help="Walidacja krzyżowa (3 podziały). Raportujemy średnią i odchylenie metryki, "
                 "aby ocenić stabilność i wariancję wyników."
        )
    with row1[2]:
        gen_shap_on_demand = st.checkbox(
            "SHAP na żądanie",
            value=False,
            help="Policzy wartości SHAP (wpływ cech na predykcję) na małej próbce. "
                 "Może spowolnić działanie, dlatego liczony tylko gdy zaznaczysz."
        )

    # Drugi rząd
    row2 = st.columns([1, 1, 1])
    with row2[0]:
        rm_outliers = st.checkbox(
            "Usuń outliery (>3σ)",
            value=False,
            help="Filtruje obserwacje odstające (|z-score|>3) dla kolumn numerycznych. "
                 "Może ustabilizować model, ale usuwa też rzadkie przypadki."
        )
    with row2[1]:
        show_learning_curve = st.checkbox(
            "Krzywa uczenia",
            value=False,
            help="Pokazuje jakość modelu w funkcji rozmiaru próby (train vs CV). "
                 "Pomaga wykryć underfitting (zbyt prosty) lub overfitting (zbyt skomplikowany)."
        )
    with row2[2]:
        show_pca_preview = st.checkbox(
            "PCA 2D (podgląd)",
            value=False,
            help="Szybki rzut PCA do 2D. Daje pogląd na strukturę danych i ewentualne klastry."
        )

    st.markdown("---")
    train_btn = st.form_submit_button("🚀 Wytrenuj model", type="primary")


def _sample_df(df: pd.DataFrame, mode: str, seed: int = 42) -> pd.DataFrame:
    if mode == "Próbka 5k" and len(df) > 5000:
        return df.sample(5000, random_state=seed)
    if mode == "Próbka 1k" and len(df) > 1000:
        return df.sample(1000, random_state=seed)
    return df


def auto_prepare_data(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "dropped_rows_target": 0,
        "dropped_constant_cols": [],
        "date_features": {},
        "high_cardinality": {},
        "target_transform": None,
        "target_winsorized": False,
    }
    df2 = df.copy()

    if target in df2.columns:
        before = len(df2)
        df2 = df2.replace([np.inf, -np.inf], np.nan)
        df2 = df2.dropna(subset=[target])
        info["dropped_rows_target"] = before - len(df2)

    for col in list(df2.columns):
        if col == target:
            continue
        s = df2[col]
        if pd.api.types.is_datetime64_any_dtype(s) or s.dtype == object:
            try:
                parsed = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
                if parsed.notna().mean() >= 0.9:
                    ycol = f"{col}__year"
                    mcol = f"{col}__month"
                    dcol = f"{col}__dow"
                    df2[ycol] = parsed.dt.year
                    df2[mcol] = parsed.dt.month
                    df2[dcol] = parsed.dt.dayofweek
                    info["date_features"][col] = [ycol, mcol, dcol]
            except Exception:
                pass

    const_cols = [c for c in df2.columns if c != target and df2[c].nunique(dropna=False) <= 1]
    if const_cols:
        df2.drop(columns=const_cols, inplace=True, errors="ignore")
        info["dropped_constant_cols"] = const_cols

    n = len(df2)
    for col in list(df2.columns):
        if col == target:
            continue
        if (df2[col].dtype == "object") or str(df2[col].dtype).startswith("category"):
            nunq = df2[col].nunique(dropna=True)
            if (nunq > 200) or (n > 0 and nunq / max(n, 1) > 0.30):
                top_vals = df2[col].value_counts(dropna=True).head(50).index
                df2[col] = np.where(df2[col].isin(top_vals), df2[col], "OTHER")
                info["high_cardinality"][col] = {"n_unique": int(nunq), "kept": 50, "other": True}

    try:
        y = pd.to_numeric(df2[target], errors="coerce")
        if y.notna().mean() > 0 and (y > 0).mean() > 0.98 and abs(y.skew()) > 1.0:
            df2[target] = np.log1p(y)
            info["target_transform"] = "log1p"
        if y.notna().any():
            p1, p99 = y.quantile([0.01, 0.99])
            out_ratio = ((y < p1) | (y > p99)).mean()
            if out_ratio > 0.05:
                df2[target] = y.clip(p1, p99)
                info["target_winsorized"] = True
    except Exception:
        pass

    return df2, info

def _should_remove_outliers(df: pd.DataFrame, target: str) -> bool:
    """Włącza filtr >3σ gdy rozkłady mocno skośne / ciężkie ogony i mamy wystarczająco danych."""
    try:
        if len(df) < 800: 
            return False
        num_cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            return False
        ratios = []
        for c in num_cols:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if s.empty: 
                continue
            mu, sd = float(s.mean()), float(s.std())
            if not np.isfinite(sd) or sd == 0: 
                continue
            z = np.abs((s - mu) / sd)
            ratios.append(float((z > 3).mean()))
        return (len(ratios) > 0) and (np.mean(ratios) > 0.03)
    except Exception:
        return False

def _auto_training_policy(df: pd.DataFrame, target: str) -> dict:
    """Zwraca politykę: sample_size, cv_folds, compute_shap, remove_outliers, make_lc, make_pca."""
    n = len(df)
    # Próbka do treningu dla bardzo dużych zbiorów (EDA ma własny automat)
    sample_mode = "all"
    if n > 200_000:
        sample_mode = "1k"
    elif n > 50_000:
        sample_mode = "5k"

    # CV: stabilizacja tylko gdy danych jest sensownie dużo
    cv_folds = 3 if n >= 2_000 else 0

    # SHAP tylko dla mniejszych / średnich problemów
    xcols = [c for c in df.columns if c != target]
    compute_shap = (n <= 5_000) and (len(xcols) <= 50)

    # Outliery wg heurystyki
    remove_outliers = _should_remove_outliers(df, target)

    # Dodatkowe wykresy po treningu – licz zawsze, ale na próbkach
    make_lc = True
    make_pca = True

    return dict(
        sample_mode=sample_mode,
        cv_folds=cv_folds,
        compute_shap=compute_shap,
        remove_outliers=remove_outliers,
        make_lc=make_lc,
        make_pca=make_pca,
    )

def _apply_sample_mode(df: pd.DataFrame, mode: str, seed: int = 42) -> pd.DataFrame:
    if mode == "5k" and len(df) > 5000:
        return df.sample(5000, random_state=seed)
    if mode == "1k" and len(df) > 1000:
        return df.sample(1000, random_state=seed)
    return df


if train_btn and tgt_auto:
    target_name = tgt_auto

    # Polityka doboru opcji
    policy = _auto_training_policy(df, target_name)

    # Próbkowanie danych do trenowania (jeśli trzeba)
    df_train = _apply_sample_mode(df, {"all": "all", "5k": "5k", "1k": "1k"}[policy["sample_mode"]])

    # (opcjonalne) outliery >3σ
    if policy["remove_outliers"]:
        try:
            df_train = _remove_outliers_sigma(df_train, target=target_name, sigma=3.0)
            st.caption(f"Usunięto outliery (>3σ); nowe rozmiary: {df_train.shape[0]}×{df_train.shape[1]}")
        except Exception as e:
            st.warning(f"Nie udało się zastosować filtra outlierów: {e}")

    # Auto-przygotowanie danych
    df_train, _prep_info = auto_prepare_data(df_train, target_name)
    st.session_state["_prep_info"] = _prep_info

    try:
        ptype = detect_problem_type(df_train[target_name]) if target_name in df_train.columns else None
    except Exception:
        ptype = None

    engine_key = "auto"
    cv_folds = int(policy["cv_folds"])

    with st.status("Trwa trenowanie...", expanded=False) as s:
        with st.spinner("Trening w toku…"):
            model, metrics, fi_df, meta = train_sklearn(
                df_train,
                target=target_name,
                problem_type=ptype if ptype else None,
                engine=engine_key,
                cv_folds=cv_folds,
                out_dir="tmiv_out",
                random_state=42,
                compute_shap=bool(policy["compute_shap"]),
            )
            # zapis modelu
            try:
                from joblib import dump
                out_dir = Path("tmiv_out"); out_dir.mkdir(parents=True, exist_ok=True)
                dump(model, out_dir / "model.joblib")
                try:
                    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
                except Exception:
                    pass
                st.toast("Model zapisany: tmiv_out/model.joblib", icon="💾")
            except Exception as e:
                st.warning(f"Nie udało się zapisać modelu: {e}")

            st.session_state["model"] = model
            st.session_state["metrics"] = metrics
            st.session_state["fi_df"] = fi_df
            st.session_state["meta"] = meta
            st.session_state["X_last"] = df_train.drop(columns=[target_name], errors="ignore")
            st.session_state["y_last"] = df_train[target_name] if target_name in df_train.columns else None

        # Raport + dodatkowe wykresy (zawsze generujemy; LC/PCA trafiły też do EDA)
        try:
            report_fig = make_model_report_figure(
                dataset_name=dataset_name or "dataset",
                target_name=target_name or "target",
                problem_type=meta.get("problem_type", ""),
                metrics=metrics,
                fi_df=fi_df,
                prep_info=st.session_state.get("_prep_info") or {},
                engine_name=meta.get("engine", "auto"),
            )
            png_path, html_path = save_model_report(fig=report_fig, out_dir=Path("tmiv_out"))
            st.session_state["model_report_fig"] = report_fig
            if png_path: st.toast(f"Zapisano raport: {png_path.name}", icon="🖼️")
            elif html_path: st.toast(f"Zapisano raport HTML: {html_path.name}", icon="🖼️")
        except Exception as e:
            st.warning(f"Nie udało się zbudować raportu modelu: {e}")

        # extra figs
        extra_figs = {}
        try:
            task_flag = (meta.get("problem_type") or "")
            extra_figs["lc"] = _plot_learning_curve(model, st.session_state["X_last"], st.session_state["y_last"], task=task_flag)
            extra_figs["pca"] = _pca_preview_2d(st.session_state["X_last"], st.session_state["y_last"], task=task_flag)
        except Exception as e:
            st.warning(f"Nie udało się wygenerować dodatkowych wykresów: {e}")
        st.session_state["extra_figs"] = extra_figs

        # MAPE/SMAPE gdy regresja
        try:
            if isinstance(st.session_state.get("X_last"), pd.DataFrame) and st.session_state.get("y_last") is not None:
                problem = (meta.get("problem_type") or "").lower()
                if ("reg" in problem) or ("R2" in metrics):
                    y_true = np.asarray(st.session_state["y_last"])
                    y_pred = np.asarray(model.predict(st.session_state["X_last"]))
                    _mape = safe_mape(y_true, y_pred, zero_policy="skip")
                    metrics["MAPE"] = None if (isinstance(_mape, float) and np.isnan(_mape)) else float(_mape)
                    metrics["SMAPE"] = float(smape(y_true, y_pred))
                    st.session_state["metrics"] = metrics
        except Exception:
            pass

        # Log historii
        try:
            log_run(
                conn,
                dataset=dataset_name or "dataset",
                target=target_name or "target",
                problem_type=meta.get("problem_type", ""),
                engine_name=meta.get("engine", ""),
                metrics=metrics,
                run_id=meta.get("run_name", ""),
            )
            st.toast("Run zapisany w historii.", icon="✅")
        except Exception as e:
            st.warning(f"Nie udało się zapisać historii: {e}")

        s.update(label="Gotowe ✅", state="complete")

# ==============================
# 📊 Wyniki + wizualizacje + eksport
# ==============================
if st.session_state.get("model") is not None:
    st.markdown("## 📊 Wyniki")

prep_info_show = st.session_state.get("_prep_info") or {}
if prep_info_show:
    with st.expander("📌 Automatyczne przygotowanie danych (log)"):
        st.json(to_native(prep_info_show))

met = st.session_state.get("metrics") or {}
if met:
    # Opisy metryk (krótkie, zawijane dzięki CSS)
    titles = {
        "RMSE": "RMSE (błąd średniokwadratowy — duże błędy karane mocniej)",
        "R2": "R² (udział wariancji wyjaśnionej przez model)",
        "MAE": "MAE (średni błąd bezwzględny w jednostkach celu)",
        "MAPE": "MAPE (średni błąd procentowy vs wartość rzeczywista)",
        "SMAPE": "SMAPE (symetryczny % błąd — stabilniejszy blisko zera)",
        "Accuracy": "Accuracy (odsetek poprawnych klasyfikacji)",
        "F1_weighted": "F1_weighted (średnia ważona precyzji i czułości)",
        "ROC_AUC": "ROC_AUC (AUC — rozdzielczość klasyfikatora, binarnie)",
    }

    # metryki z deltakami
    c1m, c2m, c3m, c4m, c5m = st.columns(5)
    if "RMSE" in met:
        render_metric(c1m, label=titles["RMSE"], value=float(met["RMSE"]), key="rmse", lower_is_better=True)
    if "R2" in met:
        render_metric(c2m, label=titles["R2"], value=float(met["R2"]), key="r2", lower_is_better=False)
    if "MAE" in met:
        render_metric(c3m, label=titles["MAE"], value=float(met["MAE"]), key="mae", lower_is_better=True)
    if "MAPE" in met:
        v = met["MAPE"]
        render_metric(c4m, label=titles["MAPE"], value=float(v) if (isinstance(v, (int, float)) and np.isfinite(v)) else float("nan"),
                      key="mape", lower_is_better=True, is_pct=True, dec=2)
    if "SMAPE" in met:
        render_metric(c5m, label=titles["SMAPE"], value=float(met["SMAPE"]), key="smape", lower_is_better=True, is_pct=True, dec=2)

    # klasyfikacja — opcjonalnie
    cols_cls = st.columns(4)
    i = 0
    for key in ["Accuracy", "F1_weighted", "ROC_AUC"]:
        if key in met:
            val = met[key]
            txt = f"{float(val):.5f}" if isinstance(val, (int, float)) else str(val)
            cols_cls[i % 4].metric(titles.get(key, key), txt)
            i += 1

    # Cross-Validation info
    if all(k in met for k in ("cv_metric", "cv_mean", "cv_std")):
        st.write(
            f"**3-fold Cross-Validation** — {met['cv_metric']}: {met['cv_mean']:.4f} ± {met['cv_std']:.4f} (na {met.get('cv_folds',3)} foldach)"
        )
        if "cv_explanation" in met:
            st.caption(met["cv_explanation"])

    # Feature importance
    fi = st.session_state.get("fi_df")
    fi = fi if isinstance(fi, pd.DataFrame) else pd.DataFrame()
    if not fi.empty:
        max_val = max(1, min(50, len(fi)))
        default_val = max(1, min(20, len(fi)))
        topk = st.slider("Ile najważniejszych cech pokazać", 1, max_val, default_val)
        st.dataframe(fi.head(topk), use_container_width=True)
        st.bar_chart(fi.head(topk).set_index("feature")["importance"], use_container_width=True)

    # Podgląd raportu modelu + przyciski pobrania
    report_fig = st.session_state.get("model_report_fig")
    if report_fig is not None:
        with st.expander("🖼️ Raport o modelu (podgląd)", expanded=False):
            st.plotly_chart(report_fig, use_container_width=True)
            try:
                # spróbuj podać plik PNG jeśli jest
                png_file = Path("tmiv_out/model_report.png")
                html_file = Path("tmiv_out/model_report.html")
                if png_file.exists():
                    st.download_button(
                        "⬇️ Pobierz raport (PNG)",
                        data=png_file.read_bytes(),
                        file_name="model_report.png",
                        mime="image/png"
                    )
                elif html_file.exists():
                    st.download_button(
                        "⬇️ Pobierz raport (HTML)",
                        data=html_file.read_bytes(),
                        file_name="model_report.html",
                        mime="text/html"
                    )
                else:
                    st.caption("Plik raportu nie został znaleziony w tmiv_out/.")
            except Exception:
                pass

    # Dodatkowe wykresy (uczenie / PCA) — z sesji
    if st.session_state.get("model") is not None:
        extra_figs = st.session_state.get("extra_figs", {})
        with st.expander("📈 Dodatkowe wykresy (uczenie / PCA)", expanded=bool(extra_figs)):
            if not extra_figs:
                st.caption("Nie wybrano opcji dodatkowych wykresów przy treningu.")
            else:
                if "lc" in extra_figs:
                    st.plotly_chart(extra_figs["lc"], use_container_width=True)
                if "pca" in extra_figs:
                    st.plotly_chart(extra_figs["pca"], use_container_width=True)

    # ===== Rekomendacje (regułowe) =====
    def _recommendations(metrics: Dict[str, Any], prep_info: Dict[str, Any], fi_df: pd.DataFrame) -> List[str]:
        rec: List[str] = []
        m = metrics or {}
        pi = prep_info or {}

        r2 = m.get("R2")
        mape = m.get("MAPE")
        acc = m.get("Accuracy")
        f1w = m.get("F1_weighted")

        # REGRESJA
        if isinstance(r2, (int, float)):
            if r2 < 0.60:
                rec.append("R² < 0.60 — dodaj nowe cechy (np. interakcje, cechy z dat), sprawdź jakość danych i outliery.")
            elif r2 >= 0.85:
                rec.append("R² ≥ 0.85 — bardzo dobre dopasowanie. Rozważ walidację na innym zbiorze lub prostszy model dla wyjaśnialności.")

        if isinstance(mape, (int, float)):
            if mape > 0.15:
                rec.append("MAPE > 15% — rozważ transformacje (np. log1p), standaryzację i obsługę outlierów.")
            elif mape > 0.08:
                rec.append("MAPE 8–15% — możliwe drobne usprawnienia: inżynieria cech, redukcja szumu, więcej danych.")

        # KLASYFIKACJA
        if isinstance(acc, (int, float)) and acc < 0.75:
            rec.append("Accuracy < 75% — sprawdź niezbalansowanie klas (undersampling/oversampling), regularyzację i cechy informacyjne.")
        if isinstance(f1w, (int, float)) and f1w < 0.80:
            rec.append("F1_weighted < 0.8 — wzmocnij klasy mniejszościowe lub dostrój próg decyzyjny pod F1.")

        # Dane / przygotowanie
        if bool(pi.get("target_winsorized")):
            rec.append("Zastosowano winsoryzację celu — sprawdź rozkład błędów; rozważ usunięcie skrajnych obserwacji.")
        if pi.get("target_transform") == "log1p":
            rec.append("Cel przekształcono log1p — pamiętaj o odlogowaniu prognoz przy interpretacji.")

        hc = pi.get("high_cardinality", {}) or {}
        if isinstance(hc, dict) and len(hc) > 0:
            cols = ", ".join(list(hc.keys())[:5])
            rec.append(f"Wysoka kardynalność w: {cols} — zostaw TOP-k i grupuj rzadkie wartości (częściowo już zastosowane).")

        const = pi.get("dropped_constant_cols", []) or []
        if isinstance(const, list) and len(const) > 0:
            rec.append(f"Usunięto kolumny stałe: {', '.join(map(str, const[:5]))} — były bez informacji.")

        if isinstance(fi_df, pd.DataFrame) and not fi_df.empty and "feature" in fi_df.columns:
            top_feats = [str(x) for x in fi_df.head(3)["feature"].tolist()]
            if top_feats:
                rec.append(f"Największy wpływ: {', '.join(top_feats)}. Warto pozyskać dokładniejsze dane dla tych cech.")

        if not rec:
            rec.append("Wyniki wyglądają stabilnie. Kolejny krok: walidacja na danych z innego okresu/źródła.")
        return rec

    metrics_ss: Dict[str, Any] = st.session_state.get("metrics") or {}
    prep_info_ss: Dict[str, Any] = st.session_state.get("_prep_info") or {}
    fi_obj = st.session_state.get("fi_df")
    if not isinstance(fi_obj, pd.DataFrame):
        fi_obj = pd.DataFrame()

    with st.expander("💡 Rekomendacje", expanded=True):
        recs = _recommendations(metrics_ss, prep_info_ss, fi_obj)
        for r in recs:
            st.markdown(f"- {r}")

    # --- Eksport ZIP ---
    st.markdown("## 📦 Eksport artefaktów")
    meta = st.session_state.get("meta") or {}
    zip_path = Path(meta.get("zip_path", "")) if isinstance(meta, dict) else Path("")
    if zip_path and zip_path.exists():
        data_bytes = zip_path.read_bytes()
        st.download_button("📦 Zapisz wszystko (ZIP)", data=data_bytes, file_name=zip_path.name, mime="application/zip")
    else:
        st.caption("ZIP nie jest dostępny (spróbuj przyciskiem w sidebarze „Zrób ZIP z tmiv_out”).")


# ==============================
# ⚖️ Próg decyzyjny (tylko klasyfikacja) + 🔮 Szybkie predykcje
# ==============================
if st.session_state.get("model") is not None and st.session_state.get("X_last") is not None:
    model = st.session_state["model"]
    X_last: pd.DataFrame = st.session_state["X_last"]
    y_last = st.session_state.get("y_last")
    meta = st.session_state.get("meta", {}) or {}
    problem = (meta.get("problem_type") or "").lower()

    # ---------- Sekcja: Próg decyzyjny ----------
    if "class" in problem and y_last is not None:
        st.markdown("## ⚖️ Próg decyzyjny (klasyfikacja)")

        # Spróbuj pozyskać "score" (proba lub decision_function)
        y_true = pd.Series(y_last).values
        proba = None
        scores = None

        try:
            if hasattr(model, "predict_proba"):
                P = model.predict_proba(X_last)
                if P.ndim == 2 and P.shape[1] >= 2:
                    proba = P[:, 1]
                else:
                    proba = P.ravel()
            elif hasattr(model, "decision_function"):
                s = model.decision_function(X_last)
                scores = np.asarray(s).ravel()
        except Exception:
            pass

        # Suwak progu
        cth1, cth2, cth3 = st.columns([2, 1, 1])
        with cth1:
            threshold = st.slider("Próg klasy pozytywnej", 0.01, 0.99, 0.50, 0.01,
                                  help="Stosowany tylko do metryk poniżej (model pozostaje bez zmian).")
        with cth2:
            sample_rows = st.number_input("Ile wierszy do podglądu (Tabela)", min_value=5, max_value=200, value=20, step=1)
        with cth3:
            show_cm = st.checkbox("Pokaż macierz pomyłek", value=False)

        # Predykcja wg progu (binarny scenariusz)
        y_pred_thr = None
        if proba is not None:
            y_pred_thr = (proba >= threshold).astype(int)
        elif scores is not None:
            sig = 1 / (1 + np.exp(-scores))
            y_pred_thr = (sig >= threshold).astype(int)
        else:
            try:
                y_pred_thr = model.predict(X_last)
            except Exception:
                y_pred_thr = None

        # Metryki vs próg (binarne)
        if y_pred_thr is not None and len(np.unique(y_true)) == 2:
            acc = accuracy_score(y_true, y_pred_thr)
            prec = precision_score(y_true, y_pred_thr, zero_division=0)
            rec = recall_score(y_true, y_pred_thr, zero_division=0)
            f1v = f1_score(y_true, y_pred_thr, zero_division=0)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy (próg)", f"{acc:.4f}")
            m2.metric("Precision (próg)", f"{prec:.4f}")
            m3.metric("Recall (próg)", f"{rec:.4f}")
            m4.metric("F1 (próg)", f"{f1v:.4f}")

            if show_cm:
                try:
                    cm = confusion_matrix(y_true, y_pred_thr)
                    cm_df = pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"])
                    st.dataframe(cm_df, use_container_width=True)
                except Exception:
                    pass
        else:
            st.caption("Brak predict_proba/decision_function lub problem wieloklasowy — suwak progu ma charakter poglądowy.")

        # Mini-tabela podglądowa (proba/scores + pred)
        try:
            head_idx = X_last.index[: int(sample_rows)]
            tbl = pd.DataFrame(index=head_idx)
            if proba is not None:
                tbl["proba_pos"] = pd.Series(proba, index=X_last.index).loc[head_idx].round(6)
            elif scores is not None:
                tbl["score"] = pd.Series(scores, index=X_last.index).loc[head_idx].round(6)
            if y_pred_thr is not None:
                tbl["pred_thr"] = pd.Series(y_pred_thr, index=X_last.index).loc[head_idx]
            if y_last is not None:
                tbl["y_true"] = pd.Series(y_true, index=X_last.index).loc[head_idx]
            st.dataframe(tbl, use_container_width=True)
        except Exception:
            pass

    # ---------- Sekcja: Szybkie predykcje ----------
    st.markdown("## 🔮 Szybkie predykcje")

    with st.expander("ℹ️ Jak korzystać z szybkich predykcji?", expanded=False):
        st.markdown("""
    - **Domyślnie** pokazujemy predykcje dla pierwszych *n* wierszy danych treningowych (X_last).
    - Możesz **wkleić próbkę CSV** z nagłówkiem **identycznym jak w X_last** (kolumny i ich typy).
    - Braki są dozwolone — pipeline ma wbudowaną **imputację**.
    - Jeśli w treningu tworzono cechy z dat, podawaj **oryginalną kolumnę daty**, a nie kolumny pochodne (`__year`, `__month`).
    - Użyj **„⬇️ Pobierz predykcje (CSV)”**, aby zapisać wyniki.
    """)

    cqp1, cqp2 = st.columns([1, 1])
    with cqp1:
        n_preview = st.number_input("Podgląd na head(n)", min_value=5, max_value=200, value=20, step=1)
    with cqp2:
        sample_csv = st.text_area(
            "Wklej krótką próbkę CSV (opcjonalnie; kolumny jak w X_last)",
            placeholder="col1,col2,...\n1,foo,...\n2,bar,...",
            height=120
        )

    # Zbuduj ramkę do predykcji
    X_pred = X_last.head(int(n_preview)).copy()
    if sample_csv.strip():
        try:
            from io import StringIO
            X_custom = pd.read_csv(StringIO(sample_csv))
            X_pred = pd.concat([X_pred, X_custom], axis=0, ignore_index=True, sort=False)
            X_pred = X_pred[X_last.columns]  # tylko kolumny, które model widział
        except Exception as e:
            st.warning(f"Nie udało się wczytać próbki CSV: {e}")

    # Predykcje
    try:
        y_hat = model.predict(X_pred)
    except Exception as e:
        st.error(f"Nie udało się policzyć predykcji: {e}")
        y_hat = None

    if y_hat is not None:
        out = pd.DataFrame(index=X_pred.index)
        out["prediction"] = y_hat

        # Rezydua dla regresji
        if "reg" in problem and y_last is not None:
            y_true_map = pd.Series(y_last, index=X_last.index)
            common = out.index.intersection(y_true_map.index)
            out.loc[common, "y_true"] = y_true_map.loc[common]
            out["residual"] = out["y_true"] - out["prediction"]
            st.dataframe(out, use_container_width=True)

            try:
                res_show = out.dropna(subset=["residual"]).reset_index(drop=True)
                if not res_show.empty:
                    res_fig = px.histogram(res_show, x="residual", nbins=30, title="Rozkład residuów (podgląd)")
                    st.plotly_chart(res_fig, use_container_width=True)
            except Exception:
                pass
        else:
            st.dataframe(out, use_container_width=True)

        # Pobierz CSV z predykcjami
        try:
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Pobierz predykcje (CSV)", data=csv_bytes, file_name="predictions_preview.csv", mime="text/csv")
        except Exception:
            pass


# ==============================
# 🗂️ Historia + Porównanie
# ==============================
st.markdown("## 🗂️ Historia uruchomień")

try:
    if not st.session_state["history_cleared"]:
        df_hist = get_history(conn, limit=200)
    else:
        df_hist = pd.DataFrame()
except Exception:
    df_hist = pd.DataFrame()

# Po pobraniu df_hist
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    tz_pl = ZoneInfo("Europe/Warsaw")
except Exception:
    tz_pl = None

def _fix_tz(dfh: pd.DataFrame) -> pd.DataFrame:
    if dfh is None or dfh.empty:
        return dfh
    # znajdź kolumny czasu
    cand_cols = [c for c in dfh.columns if any(k in c.lower() for k in ["time", "date", "created", "updated", "timestamp"])]
    for c in cand_cols:
        try:
            s = pd.to_datetime(dfh[c], errors="coerce", utc=True)  # zakładamy, że w DB jest UTC
            if tz_pl is not None:
                s = s.dt.tz_convert(tz_pl).dt.tz_localize(None)
            dfh[c] = s
        except Exception:
            pass
    return dfh

df_hist = _fix_tz(df_hist)

if df_hist is None or df_hist.empty:
    if st.session_state["history_cleared"]:
        st.success("Historia wyczyszczona.")
    else:
        st.info("Brak historii w bazie.")
else:
    st.dataframe(df_hist, use_container_width=True)
    c1h, c2h = st.columns(2)
    with c1h:
        if st.button("🗑️ Wyczyść historię", key="clear_hist_btn"):
            try:
                clear_history(conn)
                st.session_state["history_cleared"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Nie udało się wyczyścić: {e}")
    with c2h:
        try:
            csv_bytes = export_history_csv(conn)
            st.download_button("⬇️ Pobierz historię (CSV)", data=csv_bytes,
                               file_name="tmiv_history.csv", mime="text/csv")
        except Exception:
            pass

    st.subheader("🔁 Porównanie dwóch ostatnich")
    if len(df_hist) >= 2:
        a = df_hist.iloc[-2]
        b = df_hist.iloc[-1]
        try:
            ma = json.loads(a.get("metrics_json", "{}"))
        except Exception:
            ma = {}
        try:
            mb = json.loads(b.get("metrics_json", "{}"))
        except Exception:
            mb = {}
        keys = sorted(
            {k for k, v in ma.items() if isinstance(v, (int, float))}
            | {k for k, v in mb.items() if isinstance(v, (int, float))}
        )
        rows = []
        for k in keys:
            va = float(ma.get(k)) if isinstance(ma.get(k), (int, float)) else np.nan
            vb = float(mb.get(k)) if isinstance(mb.get(k), (int, float)) else np.nan
            delta = (vb - va) if np.isfinite(va) and np.isfinite(vb) else np.nan
            rows.append({"metryka": k, "poprzedni": va, "ostatni": vb, "Δ (ostatni - poprzedni)": delta})
        cmp_df = pd.DataFrame(rows)
        st.dataframe(cmp_df, use_container_width=True)
        st.caption("Porównanie używa unii kluczy i ignoruje nieliczbowe wartości.")
    else:
        st.caption("Potrzeba co najmniej dwóch wpisów w historii.")
