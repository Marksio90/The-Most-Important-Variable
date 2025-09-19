# app.py — ZMODERNIZOWANA APLIKACJA TMIV z pełną integracją wszystkich modułów
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import traceback
import time
import io
import mimetypes
import os  # <-- potrzebne dla env

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

# Minimal, pomocny sidebar (bez bajerów wizualnych)
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
            if ok:
                st.success("Wczytano .env / secrets — klucz dostępny.")
            else:
                st.error("Nie znaleziono klucza.")
            st.experimental_rerun()
    with c2:
        if st.button("Wyczyść cache", use_container_width=True):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            try:
                st.cache_resource.clear()
            except Exception:
                pass
            st.success("Cache wyczyszczony.")
            st.experimental_rerun()

    with st.expander("Wklej klucz OpenAI (opcjonalnie)"):
        typed = st.text_input("OPENAI_API_KEY", type="password", value="")
        if st.button("Ustaw klucz tymczasowo"):
            if typed.strip():
                os.environ["OPENAI_API_KEY"] = typed.strip()
                st.session_state["openai_api_key"] = typed.strip()
                st.success("Klucz ustawiony (do końca sesji).")
                st.experimental_rerun()
            else:
                st.warning("Wpisz klucz.")

    if st.button("Reset ustawień", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.success("Ustawienia zresetowane.")
        st.experimental_rerun()

# --- Lokalny fallback EDA (bez interaktywnych wykresów) ---
def render_eda_section(df: pd.DataFrame) -> None:
    st.subheader("Podstawowe statystyki")
    st.write("**Kształt:**", f"{df.shape[0]} wierszy × {df.shape[1]} kolumn")
    st.write("**Typy kolumn:**")
    st.write(pd.DataFrame({"dtype": df.dtypes.astype(str)}))

    # Podsumowania liczbowe
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        st.write("**Opis statystyczny (kolumny liczbowe):**")
        st.dataframe(df[num_cols].describe().transpose(), use_container_width=True)
    else:
        st.info("Brak kolumn liczbowych do statystyk opisowych.")

    # Unikatowe wartości dla krótkich kolumn kategorycznych/tekstowych
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        with st.expander("Podgląd unikatowych wartości (do 20 na kolumnę)"):
            for c in cat_cols[:30]:
                uniq = df[c].dropna().unique()
                st.write(f"**{c}** — unikatowe: {min(len(uniq), 20)} / {len(uniq)}")
                st.write(uniq[:20])

def render_download_buttons(export_files: dict) -> None:
    """
    Oczekuje słownika: nazwa -> ścieżka (str/Path) lub bytes / file-like.
    Dla ścieżek odczytuje plik i tworzy st.download_button.
    """
    if not export_files:
        st.info("Brak plików do pobrania.")
        return

    st.subheader("📥 Pobierz artefakty")
    for label, obj in export_files.items():
        file_name = None
        data_bytes = None
        mime = "application/octet-stream"

        # 1) Ścieżka (str/Path)
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

        # 2) Bytes
        elif isinstance(obj, (bytes, bytearray)):
            data_bytes = bytes(obj)
            file_name = f"{label}.bin"

        # 3) Plik w pamięci (np. io.BytesIO)
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

        # 4) DataFrame -> CSV
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

# ====== NASZE MODUŁY (z paczek 1-8) ======
from config.settings import get_settings
from frontend.ui_components import (
    render_sidebar, render_footer, render_upload_section,
    render_model_config_section, render_training_results,
    render_data_preview_enhanced
)

# Nadpisujemy importowany render_sidebar lokalną wersją:
render_sidebar = _minimal_render_sidebar

from backend.smart_target import SmartTargetSelector, format_target_explanation, format_alternatives_list
from backend.smart_target_llm import (
    LLMTargetSelector, render_openai_config, 
    render_smart_target_section_with_llm
)
from backend.ml_integration import (
    ModelConfig, train_model_comprehensive, save_model_artifacts, 
    load_model_artifacts, TrainingResult
)
from backend.utils import (
    infer_problem_type, validate_dataframe, seed_everything,
    hash_dataframe_signature, get_openai_key_from_envs
)
from backend.report_generator import (
    export_model_comprehensive, generate_quick_report, ModelReportGenerator
)
from db.db_utils import (
    DatabaseManager, TrainingRecord, create_training_record,
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

# ====== STYLE (bez palet użytkownika — czysty wygląd) ======
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
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
    }
    .target-recommendation {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: 2px solid #2196f3;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border: 2px solid #ff9800;
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 2px solid #4caf50;
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .error-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border: 2px solid #f44336;
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ====== KLASA APLIKACJI ======
class TMIVApp:
    """Główna klasa aplikacji TMIV z pełną integracją wszystkich modułów."""
    
    def __init__(self):
        self.settings = get_settings()
        self.db_manager = DatabaseManager(self.settings.database_url)
        self.smart_target = SmartTargetSelector()
        self.llm_target = LLMTargetSelector()
        
        # Inicjalizacja session state
        self._init_session_state()
        
        # Seeding dla reproducibility
        seed_everything(self.settings.random_seed)
    
    def _init_session_state(self):
        """Inicjalizuje wszystkie zmienne session state."""
        defaults = {
            'df': None,
            'dataset_name': '',
            'target_recommendations': [],
            'selected_target': None,
            'training_result': None,
            'last_training_id': None,
            'openai_key_set': False,
            'data_processed': False,
            'model_trained': False,
            'export_files': {},
            'current_tab': 'upload',
            'settings_changed': False
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v
    
    def run(self):
        """Uruchamia aplikację."""
        self._render_header()
        
        with st.sidebar:
            render_sidebar()  # <- nasz minimalistyczny sidebar
        
        # Prosty układ bez zbędnych “palet/szczegółowości”
        tab_upload, tab_data, tab_eda, tab_target, tab_model, tab_results, tab_history = st.tabs([
            "📤 Wczytywanie", "📊 Dane", "🔍 EDA", "🎯 Target", "⚙️ Model", "📈 Wyniki", "📚 Historia"
        ])
        
        with tab_upload:
            self._render_upload_tab()
        
        with tab_data:
            self._render_data_tab()
        
        with tab_eda:
            self._render_eda_tab()
        
        with tab_target:
            self._render_target_tab()
        
        with tab_model:
            self._render_model_tab()
        
        with tab_results:
            self._render_results_tab()
        
        with tab_history:
            self._render_history_tab()
    
    def _render_upload_tab(self):
        """Renderuje zakładkę wczytywania danych."""
        st.header("📤 Wczytywanie danych")
        
        # Upload section
        uploaded_data = render_upload_section()
        
        # >>> FIX: NIE używamy "if uploaded_data:" dla obiektu tuple/DF!
        if uploaded_data is not None and isinstance(uploaded_data, (list, tuple)) and len(uploaded_data) == 3 and uploaded_data[0] is not None and isinstance(uploaded_data[0], pd.DataFrame) and not uploaded_data[0].empty:
            df, dataset_name, status_msg = uploaded_data
            
            if df is not None:
                st.session_state.df = df
                st.session_state.dataset_name = dataset_name
                st.session_state.data_processed = True
                
                # Walidacja danych
                validation_result = validate_dataframe(df)
                if validation_result['valid']:
                    st.markdown(f"""
                    <div class="success-box">
                        ✅ <strong>Dane wczytane pomyślnie!</strong><br>
                        📊 Dataset: <code>{dataset_name}</code><br>
                        📏 Rozmiar: {len(df):,} wierszy × {len(df.columns):,} kolumn<br>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                        ⚠️ <strong>Wykryto potencjalne problemy w danych:</strong><br>
                        {validation_result['message']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Nie udało się wczytać danych.")
        else:
            st.info("Wgraj plik, aby przejść dalej.")
    
    def _render_data_tab(self):
        """Renderuje zakładkę danych z rozbudowanym podglądem."""
        st.header("📊 Analiza danych")
        
        if st.session_state.df is None:
            st.info("🔼 Najpierw wczytaj dane w zakładce 'Wczytywanie'")
            return
        
        df = st.session_state.df
        
        # Podgląd danych (bez dodatkowych suwaków/siatek itd.)
        render_data_preview_enhanced(df, st.session_state.dataset_name)
    
    def _render_eda_tab(self):
        """Renderuje zakładkę EDA."""
        st.header("🔍 Eksploracyjna Analiza Danych (EDA)")
        
        if st.session_state.df is None:
            st.info("🔼 Najpierw wczytaj dane w zakładce 'Wczytywanie'")
            return
        
        df = st.session_state.df
        
        # Rozbudowane EDA (jeśli w komponentach są zbędne kontrolki — można je tam wyciąć)
        render_eda_section(df)
    
    def _render_target_tab(self):
        """Renderuje zakładkę wyboru targetu."""
        st.header("🎯 Wybór zmiennej docelowej (Target)")
        
        if st.session_state.df is None:
            st.info("🔼 Najpierw wczytaj dane w zakładce 'Wczytywanie'")
            return
        
        df = st.session_state.df
        
        # Rekomendacje celu (SmartTarget)
        with st.spinner("Analizuję kolumny pod kątem potencjalnego celu..."):
            recommendations = self.smart_target.recommend_targets(df)
            st.session_state.target_recommendations = recommendations
        
        # Wyświetlenie rekomendacji
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
            
            # Manual target selection
            st.subheader("📋 Wybór ręczny")
            available_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64', 'object', 'category']]
            selected_target = st.selectbox(
                "Wybierz zmienną docelową:",
                options=available_columns,
                index=0 if recommendations else 0,
                help="Wybierz kolumnę, którą chcesz przewidywać"
            )
        else:
            st.warning("Nie udało się wygenerować rekomendacji. Wybierz cel ręcznie.")
            selected_target = st.selectbox("Wybierz zmienną docelową:", options=list(df.columns))
        
        if selected_target:
            st.session_state.selected_target = selected_target
            
            # Analiza wybranego targetu
            target_series = df[selected_target]
            problem_type = infer_problem_type(target_series)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Wybrana zmienna", selected_target)
                st.metric("Typ problemu", problem_type.title())
            
            with col2:
                st.write(" ")
                st.info(format_target_explanation(selected_target, problem_type))
            
            # Sekcja LLM (konfiguracja OpenAI)
            render_openai_config()  # ten komponent bazuje teraz na naszym kluczu z env/secrets
            
            # Dodatkowe wsparcie LLM przy wyborze celu
            with st.expander("🤖 Wsparcie LLM dla wyboru celu (opcjonalnie)", expanded=False):
                if os.environ.get("OPENAI_API_KEY"):
                    render_smart_target_section_with_llm(df)
                else:
                    st.warning("Dodaj klucz OpenAI w sidebarze, aby skorzystać z LLM.")
    
    def _render_model_tab(self):
        """Konfiguracja modelu i start treningu."""
        st.header("⚙️ Konfiguracja i trening modelu")
        
        if st.session_state.df is None:
            st.info("🔼 Najpierw wczytaj dane w zakładce 'Wczytywanie'")
            return
        
        if st.session_state.selected_target is None:
            st.info("🎯 Wybierz zmienną docelową w zakładce 'Target'")
            return
        
        df = st.session_state.df
        target = st.session_state.selected_target
        problem_type = infer_problem_type(df[target])
        
        # Pobierz konfigurację (tu możesz okroić komponent z “bajerów”, jeśli chcesz)
        model_config = render_model_config_section(df, target, problem_type)
        
        if model_config and st.button("🚀 Rozpocznij trening modelu", type="primary", use_container_width=True):
            self._train_model(df, model_config)
    
    def _train_model(self, df: pd.DataFrame, config: ModelConfig):
        """Trenuje model z progress barem."""
        with st.container():
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Przygotowanie danych
                status_text.text("🔄 Przygotowywanie danych...")
                progress_bar.progress(10)
                time.sleep(0.5)
                
                # Trening
                status_text.text("🤖 Trenowanie modelu...")
                progress_bar.progress(30)
                
                # Główny trening
                result = train_model_comprehensive(df, config)
                progress_bar.progress(70)
                
                # Zapis do bazy
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
                
                # Eksporty
                status_text.text("📦 Generowanie artefaktów i raportów...")
                export_files = export_model_comprehensive(result, df, st.session_state.dataset_name)
                st.session_state.export_files = export_files
                progress_bar.progress(90)
                
                # Sukces
                status_text.empty()
                progress_bar.empty()
                
                st.markdown(f"""
                <div class="success-box">
                    🎉 <strong>Model wytrenowany pomyślnie!</strong><br>
                    📊 R² Score: {result.metrics.get('r2', result.metrics.get('accuracy', 0)):.4f}<br>
                    ⏱️ Czas treningu: {result.metadata.get('training_time_seconds', 0):.2f}s<br>
                    📁 Wygenerowano {len(export_files)} plików eksportowych
                </div>
                """, unsafe_allow_html=True)
                
                # Szybki raport
                quick_report = generate_quick_report(result, config, df, st.session_state.dataset_name)
                with st.expander("📋 Szybki raport", expanded=True):
                    st.markdown(quick_report)
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                
                st.markdown(f"""
                <div class="error-box">
                    ❌ <strong>Błąd treningu:</strong> {str(e)}
                </div>
                """, unsafe_allow_html=True)
                
                if self.settings.debug:
                    st.exception(e)
    
    def _render_results_tab(self):
        """Wyniki i eksporty."""
        st.header("📈 Wyniki i eksporty")
        
        if st.session_state.export_files:
            render_training_results(st.session_state.export_files)
            render_download_buttons(st.session_state.export_files)
        else:
            st.info("Brak wyników do wyświetlenia. Wytrenuj model w zakładce 'Model'.")
    
    def _render_history_tab(self):
        """Historia treningów."""
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
                    primary_metric = record.metrics.get('r2') or record.metrics.get('accuracy')
                    if primary_metric:
                        st.metric("Główna metryka", f"{primary_metric:.4f}")
                    
                    st.metric("Cechy", record.n_features)
                
                # Metryki szczegółowe
                if record.metrics:
                    st.write("**Wszystkie metryki:**")
                    nums = []
                    for k, v in record.metrics.items():
                        if isinstance(v, (int, float)):
                            nums.append(f"{k}: {v:.4f}")
                    if nums:
                        st.text(" | ".join(nums))
                
                # Pobranie modelu (jeśli wspierane)
                # if st.button(f"📥 Pobierz model {record.run_id[:8]}", key=f"download_{record.id}"):
                #     st.info("Funkcja pobierania może wymagać dodatkowej konfiguracji.")
    
    def _render_header(self):
        """Renderuje nagłówek aplikacji."""
        st.markdown("""
        <div class="main-header">
            <h1>TMIV — The Most Important Variables</h1>
            <p>Automatyczna analiza najważniejszych cech, EDA i trening modeli.</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_footer(self):
        """Renderuje stopkę aplikacji."""
        render_footer()

def main():
    """Główna funkcja aplikacji."""
    try:
        app = TMIVApp()
        app.run()
    except Exception as e:
        st.error("❌ Krytyczny błąd aplikacji")
        st.exception(e)

if __name__ == "__main__":
    main()
