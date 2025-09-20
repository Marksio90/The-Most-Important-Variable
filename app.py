# app.py — ZMODERNIZOWANA APLIKACJA TMIV z pełną integracją wszystkich modułów
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import time
import io
import mimetypes
import os

# === ENV & OpenAI key helpers (override) ===
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from pathlib import Path as _Path

def _load_env_files_override():
    if load_dotenv is None:
        return
    for p in (_Path(".env"), _Path("config/.env")):
        if p.exists():
            load_dotenv(dotenv_path=p, override=True)

def _pull_key_from_secrets_override():
    for k in ("OPENAI_API_KEY", "openai_api_key", "openai", "openaiKey"):
        try:
            v = st.secrets.get(k)
        except Exception:
            v = None
        if v:
            return str(v).strip()
    return None

def _pull_key_from_environ_override():
    for k in ("OPENAI_API_KEY", "openai_api_key"):
        v = os.environ.get(k)
        if v:
            return v.strip()
    return None

def ensure_openai_api_key_override() -> bool:
    _load_env_files_override()
    key = _pull_key_from_secrets_override() or _pull_key_from_environ_override()
    if key:
        os.environ["OPENAI_API_KEY"] = key
        st.session_state["openai_api_key"] = key
        return True
    return False


# Minimal sidebar
def _minimal_render_sidebar():
    st.header("⚙️ Ustawienia")

    has_key = ensure_openai_api_key_override()
    if has_key:
        st.success("✅ Klucz OpenAI: ustawiony")
    else:
        st.error("❌ Brak klucza OpenAI")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Wczytaj .env", use_container_width=True):
            ok = ensure_openai_api_key_override()
            st.success("Wczytano .env / secrets — klucz dostępny." if ok else "Nie znaleziono klucza.")
            st.rerun()
    with c2:
        if st.button("Wyczyść cache", use_container_width=True):
            for fn in (getattr(st.cache_data, "clear", None), getattr(st.cache_resource, "clear", None)):
                try:
                    fn()
                except Exception:
                    pass
            st.success("Cache wyczyszczony.")
            st.rerun()

    with st.expander("Wklej klucz OpenAI (opcjonalnie)"):
        typed = st.text_input("OPENAI_API_KEY", type="password", value="")
        if st.button("Ustaw klucz tymczasowo"):
            if typed.strip():
                os.environ["OPENAI_API_KEY"] = typed.strip()
                st.session_state["openai_api_key"] = typed.strip()
                st.success("Klucz ustawiony (do końca sesji).")
                st.rerun()
            else:
                st.warning("Wpisz klucz.")

    with st.expander("🛠 Diagnostyka klucza"):
        env_paths = [Path(".env"), Path("config/.env")]
        st.write("**Sprawdzane ścieżki .env:**")
        for p in env_paths:
            st.write(f"- `{p}` — **{'ISTNIEJE' if p.exists() else 'brak'}**")

        secrets_candidates = ("OPENAI_API_KEY", "openai_api_key", "openai", "openaiKey")
        secrets_found = []
        for k in secrets_candidates:
            try:
                v = st.secrets.get(k)
            except Exception:
                v = None
            if v:
                secrets_found.append(k)
        st.write("**st.secrets:**", ", ".join(secrets_found) if secrets_found else "— nic —")

        env_key = os.environ.get("OPENAI_API_KEY", "")
        masked = (env_key[:4] + "..." + env_key[-4:]) if env_key else "(pusty)"
        st.write("**os.environ['OPENAI_API_KEY']**:", masked)

        ss_key = st.session_state.get("openai_api_key", "")
        masked_ss = (ss_key[:4] + "..." + ss_key[-4:]) if ss_key else "(pusty)"
        st.write("**st.session_state['openai_api_key']**:", masked_ss)

    if st.button("Reset ustawień", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.success("Ustawienia zresetowane.")
        st.rerun()


# --- Prosty EDA ---
def render_eda_section(df: pd.DataFrame) -> None:
    st.subheader("Podstawowe statystyki")
    st.write("**Kształt:**", f"{df.shape[0]} wierszy × {df.shape[1]} kolumn")
    st.write("**Typy kolumn:**")
    st.write(pd.DataFrame({"dtype": df.dtypes.astype(str)}))

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        st.write("**Opis statystyczny (kolumny liczbowe):**")
        st.dataframe(df[num_cols].describe().transpose(), use_container_width=True)
    else:
        st.info("Brak kolumn liczbowych do statystyk opisowych.")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        with st.expander("Podgląd unikatowych wartości (do 20 na kolumnę)"):
            for c in cat_cols[:30]:
                uniq = df[c].dropna().unique()
                st.write(f"**{c}** — unikatowe: {min(len(uniq), 20)} / {len(uniq)}")
                st.write(uniq[:20])


def render_download_buttons(export_files: dict) -> None:
    if not export_files:
        st.info("Brak plików do pobrania.")
        return

    st.subheader("📥 Pobierz artefakty")
    for label, obj in export_files.items():
        file_name = None
        data_bytes = None
        mime = "application/octet-stream"

        if isinstance(obj, (str, Path)):
            p = Path(obj)
            if p.exists() and p.is_file():
                file_name = p.name
                mime_guess, _ = mimetypes.guess_type(str(p))
                if mime_guess:
                    mime = mime_guess
                with open(p, "rb") as f:
                    data_bytes = f.read()
            else:
                st.warning(f"Plik nie istnieje: {obj}")
                continue
        elif isinstance(obj, (bytes, bytearray)):
            data_bytes = bytes(obj)
            file_name = f"{label}.bin"
        elif hasattr(obj, "read"):
            try:
                pos = obj.tell() if hasattr(obj, "tell") else None
                data_bytes = obj.read()
                if pos is not None and hasattr(obj, "seek"):
                    obj.seek(pos)
            except Exception:
                st.warning(f"Nie udało się odczytać obiektu pliku dla: {label}")
                continue
            file_name = f"{label}.bin"
        elif isinstance(obj, pd.DataFrame):
            buf = io.StringIO()
            obj.to_csv(buf, index=False)
            data_bytes = buf.getvalue().encode("utf-8")
            file_name = f"{label}.csv"
            mime = "text/csv"
        else:
            st.warning(f"Nieobsługiwany typ dla '{label}': {type(obj)}")
            continue

        st.download_button(
            label=f"⬇️ Pobierz: {label}",
            data=data_bytes,
            file_name=file_name or f"{label}",
            mime=mime,
            use_container_width=True,
        )

# === NORMALIZACJA WYJŚCIA z render_upload_section() ===
def _coerce_uploaded_to_df(uploaded_data):
    """
    Zwraca krotkę: (df: DataFrame | None, dataset_name: str, status_msg: str).
    Obsługuje: DF, (df,name,msg), {df:..., dataset_name:...}, None.
    """
    if uploaded_data is None:
        return None, "", ""
    if isinstance(uploaded_data, pd.DataFrame):
        return uploaded_data, "uploaded_dataframe", "ok"
    if isinstance(uploaded_data, (tuple, list)):
        if len(uploaded_data) >= 1 and isinstance(uploaded_data[0], pd.DataFrame):
            df = uploaded_data[0]
            name = uploaded_data[1] if len(uploaded_data) >= 2 and isinstance(uploaded_data[1], str) else "dataset"
            msg  = uploaded_data[2] if len(uploaded_data) >= 3 and isinstance(uploaded_data[2], str) else ""
            return df, name, msg
    if isinstance(uploaded_data, dict):
        df = uploaded_data.get("df") or uploaded_data.get("dataframe")
        if isinstance(df, pd.DataFrame):
            name = uploaded_data.get("dataset_name") or uploaded_data.get("name") or "dataset"
            msg  = uploaded_data.get("status") or ""
            return df, name, msg
    return None, "", ""

# === Wbudowane przykłady (bez słowa „demo”) ===
def _load_builtin_example(example_key: str):
    """
    example_key in {'avocado','diabetes','california_housing'}
    - avocado: próbuje czytać 'data/avocado.csv'
    - diabetes: sklearn load_diabetes(as_frame=True)
    - california_housing: sklearn fetch_california_housing(as_frame=True)
    """
    key = (example_key or "").lower().strip()

    if key == "avocado":
        csv_path = Path("data/avocado.csv")
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                return df, "avocado", f"Wczytano {csv_path}."
            except Exception as e:
                st.error(f"Nie udało się wczytać {csv_path}: {e}")
                return None, "", ""
        else:
            st.warning("Nie znaleziono pliku **data/avocado.csv**. Umieść go w repo/kontenerze.")
            return None, "", ""

    if key == "diabetes":
        try:
            from sklearn.datasets import load_diabetes
            data = load_diabetes(as_frame=True)
            df = data.frame.copy()
            return df, "diabetes", "Dane z sklearn.load_diabetes"
        except Exception as e:
            st.error(f"Nie udało się wczytać diabetes: {e}")
            return None, "", ""

    if key == "california_housing":
        try:
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing(as_frame=True)
            df = data.frame.copy()
            return df, "california_housing", "Dane z sklearn.fetch_california_housing"
        except Exception as e:
            st.error(f"Nie udało się wczytać California Housing: {e}")
            return None, "", ""

    st.warning("Nieznany przykład.")
    return None, "", ""


# ====== NASZE MODUŁY ======
from config.settings import get_settings
from frontend.ui_components import (
    render_sidebar, render_footer, render_upload_section,
    render_model_config_section, render_training_results,
    render_data_preview_enhanced
)
render_sidebar = _minimal_render_sidebar  # lokalny override

from backend.smart_target import SmartTargetSelector, format_target_explanation
from backend.smart_target_llm import (
    LLMTargetSelector, render_openai_config, 
    render_smart_target_section_with_llm
)
from backend.ml_integration import (
    ModelConfig, train_model_comprehensive, TrainingResult
)
from backend.utils import (
    infer_problem_type, validate_dataframe, seed_everything,
)
from backend.report_generator import (
    export_model_comprehensive, generate_quick_report
)
from db.db_utils import (
    DatabaseManager, create_training_record,
    save_training_record, get_training_history
)

# ====== KONFIGURACJA STRONY ======
st.set_page_config(
    page_title="TMIV - The Most Important Variables",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/tmiv',
        'Report a bug': "https://github.com/your-repo/tmiv/issues",
        'About': "TMIV – inteligentna analiza najważniejszych zmiennych"
    }
)

CUSTOM_CSS = """
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 { font-size: 2.8rem; font-weight: 700; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    .main-header p { font-size: 1.2rem; margin: 0.5rem 0 0 0; opacity: 0.95; }
    .target-recommendation { background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border: 2px solid #2196f3; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 2px 8px rgba(33,150,243,0.1); }
    .warning-box { background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%); border: 2px solid #ff9800; border-radius: 12px; padding: 1.2rem; margin-top: 1rem; }
    .success-box { background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); border: 2px solid #4caf50; border-radius: 12px; padding: 1.2rem; margin-top: 1rem; }
    .error-box { background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); border: 2px solid #f44336; border-radius: 12px; padding: 1.2rem; margin-top: 1rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ====== KLASA APLIKACJI ======
class TMIVApp:
    def __init__(self):
        self.settings = get_settings()
        self.db_manager = DatabaseManager(self.settings.database_url)
        self.smart_target = SmartTargetSelector()
        self.llm_target = LLMTargetSelector()
        self._init_session_state()
        seed_everything(self.settings.random_seed)
    
    def _init_session_state(self):
        defaults = {
            'df': None, 'dataset_name': '',
            'target_recommendations': [], 'selected_target': None,
            'training_result': None, 'last_training_id': None,
            'openai_key_set': False, 'data_processed': False,
            'model_trained': False, 'export_files': {},
            'current_tab': 'upload', 'settings_changed': False
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v
    
    def run(self):
        self._render_header()
        with st.sidebar:
            render_sidebar()
        tabs = st.tabs(["📤 Wczytywanie", "📊 Dane", "🔍 EDA", "🎯 Target", "⚙️ Model", "📈 Wyniki", "📚 Historia"])
        with tabs[0]: self._render_upload_tab()
        with tabs[1]: self._render_data_tab()
        with tabs[2]: self._render_eda_tab()
        with tabs[3]: self._render_target_tab()
        with tabs[4]: self._render_model_tab()
        with tabs[5]: self._render_results_tab()
        with tabs[6]: self._render_history_tab()
    
    def _render_upload_tab(self):
        st.header("📤 Wczytywanie danych")

        # 1) Sekcja komponentu (plik/URL itp.)
        uploaded_raw = render_upload_section()
        df_new, dataset_name, status_msg = _coerce_uploaded_to_df(uploaded_raw)

        # 2) Jeśli nic nowego nie przyszło, ale coś już mamy w sesji – pokaż sukces i nie blokuj
        if df_new is None and isinstance(st.session_state.get("df"), pd.DataFrame) and not st.session_state.df.empty:
            st.success(
                f"✅ Dane są już wczytane: **{st.session_state.dataset_name or 'dataset'}** "
                f"({len(st.session_state.df):,} × {st.session_state.df.shape[1]:,})"
            )

        # 3) Przyszły nowe dane z komponentu – zapisujemy
        elif isinstance(df_new, pd.DataFrame) and not df_new.empty:
            st.session_state.df = df_new
            st.session_state.dataset_name = dataset_name or "dataset"
            st.session_state.data_processed = True
            validation_result = validate_dataframe(df_new)
            if validation_result.get('valid', True):
                st.markdown(f"""
                <div class="success-box">
                    ✅ <strong>Dane wczytane pomyślnie!</strong><br>
                    📊 Dataset: <code>{st.session_state.dataset_name}</code><br>
                    📏 Rozmiar: {len(df_new):,} wierszy × {len(df_new.columns):,} kolumn<br>
                    {status_msg or ''}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ <strong>Wykryto potencjalne problemy w danych:</strong><br>
                    {validation_result.get('message','')}
                </div>""", unsafe_allow_html=True)

        # 4) Wbudowane „Dane przykładowe” — bez słowa „demo”
        st.subheader("🎲 Dane przykładowe")
        example = st.selectbox(
            "Wybierz przykład do szybkiego startu:",
            options=["(wybierz)", "avocado", "diabetes", "california_housing"],
            index=0,
            help="Możesz wczytać lokalny plik data/avocado.csv lub zbiory z sklearn."
        )
        if st.button("Wczytaj wybrany przykład", use_container_width=True, disabled=(example == "(wybierz)")):
            df_ex, name_ex, msg_ex = _load_builtin_example(example if example != "(wybierz)" else "")
            if isinstance(df_ex, pd.DataFrame) and not df_ex.empty:
                st.session_state.df = df_ex
                st.session_state.dataset_name = name_ex          # <= zapis dokładnie 'avocado' / 'diabetes' / 'california_housing'
                st.session_state.data_processed = True
                st.success(f"✅ Wczytano przykład **{name_ex}** ({len(df_ex):,} × {df_ex.shape[1]:,}). {msg_ex}")
                st.rerun()

        # 5) Miękkie ostrzeżenia (nie blokują przejścia dalej)
        ds_name = (st.session_state.get("dataset_name") or "").lower()
        if ds_name in {"boston_housing", "boston"}:
            st.warning("Dataset **Boston Housing** jest przestarzały (deprecated). Użyj własnych danych lub np. **california_housing**.")

        # 6) Informacja końcowa tylko, gdy naprawdę nie mamy DF
        if not (isinstance(st.session_state.get("df"), pd.DataFrame) and not st.session_state.df.empty):
            st.info("Wgraj plik **lub** wybierz dane przykładowe/URL, aby przejść dalej.")
    
    def _render_data_tab(self):
        st.header("📊 Analiza danych")
        df = st.session_state.get("df")
        if not isinstance(df, pd.DataFrame) or df.empty:
            st.info("🔼 Najpierw wczytaj dane w zakładce **Wczytywanie** (plik / URL / dane przykładowe).")
            return
        render_data_preview_enhanced(df, st.session_state.dataset_name)
    
    def _render_eda_tab(self):
        st.header("🔍 Eksploracyjna Analiza Danych (EDA)")
        df = st.session_state.get("df")
        if not isinstance(df, pd.DataFrame) or df.empty:
            st.info("🔼 Najpierw wczytaj dane w zakładce **Wczytywanie**.")
            return
        render_eda_section(df)
    
    def _render_target_tab(self):
        st.header("🎯 Wybór zmiennej docelowej (Target)")
        df = st.session_state.get("df")
        if not isinstance(df, pd.DataFrame) or df.empty:
            st.info("🔼 Najpierw wczytaj dane w zakładce **Wczytywanie**.")
            return

        with st.spinner("Analizuję kolumny pod kątem potencjalnego celu..."):
            recommendations = self.smart_target.recommend_targets(df)
            st.session_state.target_recommendations = recommendations
        
        if recommendations:
            for i, rec in enumerate(recommendations[:3]):
                st.markdown(f"""
                    <div class="target-recommendation">
                        <h4>🥇 Rekomendacja #{i+1}: {rec['column']}</h4>
                        <p><strong>Typ problemu:</strong> {rec['problem_type']}</p>
                        <p><strong>Wynik:</strong> {rec['score']:.3f}</p>
                        <p><strong>Uzasadnienie:</strong> {rec['explanation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            st.subheader("📋 Wybór ręczny")
            available_columns = [c for c in df.columns if df[c].dtype in ['int64', 'float64', 'object', 'category']]
            selected_target = st.selectbox("Wybierz zmienną docelową:", options=available_columns, index=0)
        else:
            st.warning("Nie udało się wygenerować rekomendacji. Wybierz cel ręcznie.")
            selected_target = st.selectbox("Wybierz zmienną docelową:", options=list(df.columns))
        
        if selected_target:
            st.session_state.selected_target = selected_target
            target_series = df[selected_target]
            problem_type = infer_problem_type(target_series)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Wybrana zmienna", selected_target)
                st.metric("Typ problemu", problem_type.title())
            with col2:
                st.write(" ")
                st.info(format_target_explanation(selected_target, problem_type))

            render_openai_config()
            with st.expander("🤖 Wsparcie LLM dla wyboru celu (opcjonalnie)", expanded=False):
                if os.environ.get("OPENAI_API_KEY"):
                    render_smart_target_section_with_llm(df)
                else:
                    st.warning("Dodaj klucz OpenAI w sidebarze, aby skorzystać z LLM.")
    
    def _render_model_tab(self):
        st.header("⚙️ Konfiguracja i trening modelu")
        df = st.session_state.get("df")
        target = st.session_state.get("selected_target")
        if not isinstance(df, pd.DataFrame) or df.empty:
            st.info("🔼 Najpierw wczytaj dane w zakładce **Wczytywanie**.")
            return
        if not target:
            st.info("🎯 Wybierz zmienną docelową w zakładce **Target**.")
            return

        problem_type = infer_problem_type(df[target])
        model_config = render_model_config_section(df, target, problem_type)
        if model_config and st.button("🚀 Rozpocznij trening modelu", type="primary", use_container_width=True):
            self._train_model(df, model_config)
    
    def _train_model(self, df: pd.DataFrame, config: 'ModelConfig'):
        with st.container():
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                status_text.text("🔄 Przygotowywanie danych...")
                progress_bar.progress(10); time.sleep(0.3)
                status_text.text("🤖 Trenowanie modelu...")
                progress_bar.progress(30)

                result = train_model_comprehensive(df, config)
                progress_bar.progress(70)

                status_text.text("💾 Zapisywanie wyników...")
                training_record = create_training_record(
                    dataset_name=st.session_state.dataset_name,
                    target=config.target,
                    problem_type=result.problem_type,
                    engine=result.metadata.get('engine', 'auto'),
                    metrics=result.metrics,
                    n_features=result.metadata.get('n_features', None),
                    training_time_seconds=result.metadata.get('training_time_seconds', 0.0),
                    notes=result.metadata.get('notes', '')
                )
                save_training_record(self.db_manager, training_record)
                st.session_state.last_training_id = training_record.id

                status_text.text("📦 Generowanie artefaktów i raportów...")
                export_files = export_model_comprehensive(result, df, st.session_state.dataset_name)
                st.session_state.export_files = export_files
                progress_bar.progress(90)

                status_text.empty(); progress_bar.empty()
                st.markdown(f"""
                <div class="success-box">
                    🎉 <strong>Model wytrenowany pomyślnie!</strong><br>
                    📊 R²/Accuracy: {result.metrics.get('r2', result.metrics.get('accuracy', 0)):.4f}<br>
                    ⏱️ Czas treningu: {result.metadata.get('training_time_seconds', 0):.2f}s<br>
                    📁 Wygenerowano {len(export_files)} plików eksportowych
                </div>""", unsafe_allow_html=True)

                quick_report = generate_quick_report(result, config, df, st.session_state.dataset_name)
                with st.expander("📋 Szybki raport", expanded=True):
                    st.markdown(quick_report)
            except Exception as e:
                progress_bar.empty(); status_text.empty()
                st.markdown(f"""<div class="error-box">❌ <strong>Błąd treningu:</strong> {str(e)}</div>""", unsafe_allow_html=True)
                if getattr(self.settings, "debug", False):
                    st.exception(e)
    
    def _render_results_tab(self):
        st.header("📈 Wyniki i eksporty")
        if st.session_state.export_files:
            render_training_results(st.session_state.export_files)
            render_download_buttons(st.session_state.export_files)
        else:
            st.info("Brak wyników do wyświetlenia. Wytrenuj model w zakładce 'Model'.")
    
    def _render_history_tab(self):
        st.header("📚 Historia treningów")
        history = get_training_history(self.db_manager, limit=10)
        if not history:
            st.info("Brak zapisanych treningów.")
            return
        st.subheader(f"📋 Ostatnie {len(history)} treningów")
        for record in history:
            with st.expander(f"🎯 {record.dataset_name} → {record.target} ({record.run_id[:8]}...)", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Problem", record.problem_type.title())
                    st.metric("Silnik", record.engine)
                with col2:
                    st.metric("Data", record.created_at.strftime("%Y-%m-%d %H:%M"))
                    st.metric("Czas treningu", f"{record.training_time_seconds:.2f}s")
                with col3:
                    primary = record.metrics.get('r2') or record.metrics.get('accuracy')
                    if primary is not None:
                        st.metric("Główna metryka", f"{primary:.4f}")
                    st.metric("Cechy", record.n_features)

    def _render_header(self):
        st.markdown("""
        <div class="main-header">
            <h1>TMIV — The Most Important Variables</h1>
            <p>Automatyczna analiza najważniejszych cech, EDA i trening modeli.</p>
        </div>""", unsafe_allow_html=True)

    def _render_footer(self):
        render_footer()


def main():
    try:
        app = TMIVApp()
        app.run()
    except Exception as e:
        st.error("❌ Krytyczny błąd aplikacji")
        st.exception(e)

if __name__ == "__main__":
    main()