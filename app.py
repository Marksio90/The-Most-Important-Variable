# app.py — TMIV (Uruchomienia, Model Registry, Replay, lokalny czas, Plotly-only)
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.figure_factory as ff

# ====== MODUŁY PROJEKTU (po patchach) ======
from config.settings import get_settings, MLEngine
from frontend.ui_components import TMIVApp, DataConfig, UIConfig
from backend.utils import (
    SmartTargetDetector, auto_pick_target, get_openai_key_from_envs,
    to_local, utc_now_iso_z
)
from backend.ml_integration import (
    ModelConfig, train_model_comprehensive,
    save_model_artifacts, load_model_artifacts
)
from db.db_utils import (
    MLExperimentTracker, RunRecord,
    ProblemType, RunStatus, QueryFilter
)

# (opcjonalnie) lekki preproces
try:
    from backend.eda_integration import SmartDataPreprocessor
except Exception:
    SmartDataPreprocessor = None  # type: ignore


# ================== STAN ==================
@dataclass
class AppState:
    dataset: Optional[pd.DataFrame] = None
    dataset_name: str = ""
    target_column: Optional[str] = None
    model: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    feature_importance: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["feature", "importance"]))
    metadata: Dict[str, Any] = field(default_factory=dict)
    training_completed: bool = False
    last_run_id: Optional[str] = None
    loaded_model_info: Optional[Dict[str, Any]] = None  # info o wczytanym modelu (replay)


def _state() -> AppState:
    if "tmiv_state" not in st.session_state:
        st.session_state.tmiv_state = AppState()
    return st.session_state.tmiv_state


# ================== POMOCNICZE WYKRESY ==================
def _plot_regression(y_true, y_pred, title="Predykcje vs Rzeczywistość"):
    if y_true is None or y_pred is None:
        return
    if len(y_true) == 0 or len(y_pred) == 0:
        return
    dfp = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    fig = px.scatter(dfp, x="y_true", y="y_pred", title=title)
    mn = float(np.nanmin([dfp["y_true"].min(), dfp["y_pred"].min()]))
    mx = float(np.nanmax([dfp["y_true"].max(), dfp["y_pred"].max()]))
    fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx)
    st.plotly_chart(fig, use_container_width=True)


def _plot_confusion_matrix(y_true, y_pred, title="Macierz pomyłek"):
    if y_true is None or y_pred is None:
        return
    if len(y_true) == 0 or len(y_pred) == 0:
        return
    from sklearn.metrics import confusion_matrix
    labels = sorted(list(set(y_true) | set(y_pred)))
    if len(labels) == 0:
        return
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    z = cm.astype(float)
    fig = ff.create_annotated_heatmap(
        z=z,
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        showscale=True
    )
    fig.update_layout(title=title, xaxis_title="Predykcja", yaxis_title="Prawda")
    st.plotly_chart(fig, use_container_width=True)


def _show_metrics(metrics: Dict[str, Any], metadata: Dict[str, Any]):
    st.subheader("📈 Metryki modelu")
    if not metrics:
        st.info("Brak metryk do wyświetlenia.")
        return

    # kafelki (max 4)
    items = list(metrics.items())
    cols = st.columns(min(4, max(1, len(items))))
    for i, (k, v) in enumerate(items[:4]):
        try:
            if isinstance(v, (int, float)) and not pd.isna(v):
                cols[i % 4].metric(k.upper().replace("_", " "), f"{v:.4f}")
            else:
                cols[i % 4].metric(k.upper().replace("_", " "), str(v))
        except Exception:
            pass

    # ostrzeżenia z metadata
    warnings_list = (metadata or {}).get("warnings") or []
    if isinstance(warnings_list, list) and warnings_list:
        for w in warnings_list:
            st.warning(str(w))

    with st.expander("Szczegóły metryk i metadane"):
        c1, c2 = st.columns(2)
        with c1:
            st.json(metrics)
        with c2:
            st.json(metadata or {})


# ================== HISTORIA (log) ==================
def _log_run(tracker: MLExperimentTracker, *, dataset_name: str, target: str,
             problem_type: str, engine: str, metrics: Dict[str, Any],
             duration_s: Optional[float] = None, notes: str = "") -> str:
    pt = ProblemType.REGRESSION if problem_type == "regression" else (
        ProblemType.CLASSIFICATION if problem_type == "classification" else ProblemType.OTHER
    )
    run_id = f"run_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%SZ')}"
    record = RunRecord(
        dataset=dataset_name,
        target=target,
        run_id=run_id,
        problem_type=pt,
        engine=engine,
        status=RunStatus.COMPLETED,
        metrics=metrics,
        notes=notes,
        duration_seconds=duration_s,
        tags=["ui", "tmiv", "uruchomienie"]
    )
    tracker.log_run(record)
    return run_id


# ================== MODEL REGISTRY HELPERS ==================
def _run_dir(models_dir: Path, dataset: str, target: str, run_id: str) -> Path:
    safe_ds = (dataset or "dataset").replace("/", "_")
    safe_tg = (target or "target").replace("/", "_")
    return models_dir / f"{safe_ds}__{safe_tg}__{run_id}"


def _list_model_dirs(models_dir: Path, dataset: str, target: str) -> List[Path]:
    safe_ds = (dataset or "dataset").replace("/", "_")
    safe_tg = (target or "target").replace("/", "_")
    pattern = f"{safe_ds}__{safe_tg}__"
    if not models_dir.exists():
        return []
    return sorted([p for p in models_dir.iterdir() if p.is_dir() and p.name.startswith(pattern)], reverse=True)


# ================== APLIKACJA ==================
def main():
    settings = get_settings()
    out_dir = Path(getattr(settings, "output_dir", "tmiv_out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(getattr(settings, "models_dir", str(out_dir / "models")))
    models_dir.mkdir(parents=True, exist_ok=True)

    state = _state()
    tracker = MLExperimentTracker(db_path=getattr(settings, "history_db_path", str(out_dir / "history.sqlite")))

    st.set_page_config(page_title="TMIV", layout="wide")
    st.title("TMIV — AutoML")
    st.caption("Analiza danych • Trening • **Historia uruchomień** • Rejestr modeli")

    # ===== Konfiguracja UI =====
    data_cfg = DataConfig(
        max_file_size_mb=getattr(settings, "data_max_file_size_mb", 200),
        supported_formats=getattr(settings, "data_supported_formats", [".csv", ".xlsx"]),
        auto_detect_encoding=True,
        max_preview_rows=50,
    )
    ui_cfg = UIConfig(
        app_title=getattr(settings, "app_name", "TMIV"),
        app_subtitle="AutoML • EDA • Historia uruchomień",
        enable_llm=bool(get_openai_key_from_envs()),
        show_advanced_options=True,
    )
    tmiv = TMIVApp(data_cfg, ui_cfg)

    # ===== 1) Dane =====
    st.header("1) Dane")
    df, dataset_name = tmiv.render_data_selection()
    if df is not None:
        state.dataset, state.dataset_name = df, dataset_name

        with st.expander("Jakość danych / informacje", expanded=False):
            try:
                st.write(f"Wiersze: {len(df):,} • Kolumny: {len(df.columns)}")
                st.write("Podgląd:", df.head(min(5, len(df))).reset_index(drop=True))
            except Exception:
                pass
    else:
        st.info("Wczytaj dane, aby przejść dalej.")
        st.stop()

    # ===== 2) Cel (target) =====
    st.header("2) Cel (target)")
    detector = SmartTargetDetector()
    auto_target = None
    try:
        auto_target = detector.detect_target(state.dataset) or auto_pick_target(state.dataset)
    except Exception:
        auto_target = auto_pick_target(state.dataset)

    all_cols = list(state.dataset.columns)
    if not all_cols:
        st.error("Brak kolumn w danych.")
        st.stop()

    default_idx = 0
    if auto_target in all_cols:
        default_idx = all_cols.index(auto_target)

    target = st.selectbox(
        "Wybierz kolumnę celu",
        options=all_cols,
        index=default_idx,
        help="To kolumna, którą przewiduje model."
    )
    state.target_column = target

    # mała ściągawka klas (jeśli klasyfikacja wyjdzie z heurystyki)
    with st.expander("Informacje o kolumnie celu", expanded=False):
        vc = state.dataset[target].value_counts(dropna=False)
        st.write(vc.to_frame("count"))

    # ===== 3) Trening =====
    st.header("3) Trenuj i oceń")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        engine = st.selectbox("Silnik", ["auto", "sklearn", "lightgbm", "xgboost", "catboost", "pycaret"], index=0)
    with c2:
        cv_folds = st.slider("CV folds", min_value=1, max_value=10, value=int(getattr(settings, "default_cv_folds", 3)))
    with c3:
        test_size = st.slider("Test size (%)", 1, 40, int(100 * float(getattr(settings, "default_test_size", 0.2)))) / 100.0

    train_btn = st.button("🚀 Trenuj i oceń", type="primary", use_container_width=True)

    if train_btn:
        with st.spinner("Trenuję i liczę metryki…"):
            df_in = state.dataset.copy()

            if SmartDataPreprocessor:
                try:
                    pre = SmartDataPreprocessor(dataset_name=state.dataset_name)
                    df_in, _report = pre.preprocess(df_in, target=state.target_column)
                except Exception:
                    # cicho — preproces jest opcjonalny
                    pass

            cfg = ModelConfig(
                target=state.target_column,
                engine=engine if engine != "auto" else MLEngine.AUTO.value,
                cv_folds=cv_folds,
                test_size=float(test_size),
                random_state=int(getattr(settings, "default_random_state", 42))
            )

            t0 = time.time()
            try:
                result = train_model_comprehensive(df_in, cfg, use_advanced=True)
            except Exception as e:
                st.error(f"Trening nie powiódł się: {e}")
                st.stop()
            dt = time.time() - t0

            # --- bezpieczne przypisania (bez 'truthiness' DataFrame) ---
            state.model = result.model
            state.metrics = result.metrics or {}

            fi_df = result.feature_importance if isinstance(result.feature_importance, pd.DataFrame) else None
            if fi_df is not None and not fi_df.empty:
                state.feature_importance = fi_df
            else:
                state.feature_importance = pd.DataFrame(columns=["feature", "importance"])

            meta = result.metadata if isinstance(result.metadata, dict) else {}
            state.metadata = (meta or {}) | {"training_time_s": dt}
            state.training_completed = (state.model is not None) and bool(state.metrics)

            # zapis do historii
            engine_name = state.metadata.get("engine", engine)
            problem_type = (state.metadata.get("problem_type") or "unknown").lower()
            try:
                state.last_run_id = _log_run(
                    tracker,
                    dataset_name=state.dataset_name or "dataset",
                    target=state.target_column or "target",
                    problem_type=problem_type,
                    engine=engine_name,
                    metrics=state.metrics,
                    duration_s=dt,
                    notes=f"Uruchomienie TMIV — {len(df_in):,}×{len(df_in.columns)}; ts={utc_now_iso_z()}"
                )
            except Exception as e:
                st.warning(f"Nie udało się zapisać do historii: {e}")

        st.success("Gotowe! Poniżej wyniki.")

    # ===== 4) Wyniki =====
    if state.training_completed:
        st.header("Wyniki")
        _show_metrics(state.metrics, state.metadata)

        vinfo = {}
        if isinstance(state.metadata, dict):
            vinfo = state.metadata.get("validation_info", {}) or {}

        ptype = (state.metadata.get("problem_type") or "").lower()

        if ptype == "regression" and "y_true" in vinfo and "y_pred" in vinfo:
            _plot_regression(vinfo.get("y_true"), vinfo.get("y_pred"))
        elif ptype == "classification" and "y_true" in vinfo and "y_pred" in vinfo:
            _plot_confusion_matrix(vinfo.get("y_true"), vinfo.get("y_pred"))

        # FI
        if isinstance(state.feature_importance, pd.DataFrame) and not state.feature_importance.empty:
            st.subheader("🏆 Najważniejsze cechy")
            max_show = int(min(30, len(state.feature_importance)))
            topn = st.slider("Ile cech pokazać", min_value=1, max_value=max_show, value=min(10, max_show))
            df_fi = state.feature_importance.head(topn)
            if {"feature", "importance"}.issubset(df_fi.columns):
                fig = px.bar(df_fi, x="importance", y="feature", orientation="h", title=f"Top {topn}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(df_fi, use_container_width=True)
        else:
            st.info("Brak obliczalnej ważności cech dla wybranego modelu.")

    # ===== 5) Rejestr modeli (Eksport/Wczytaj/Default) =====
    st.header("💾 Rejestr modeli")
    colL, colR = st.columns([2, 2])

    with colL:
        st.subheader("Eksport bieżącego modelu")
        disabled = not state.training_completed or state.last_run_id is None
        if st.button("Zapisz artefakty (model + meta)", disabled=disabled, use_container_width=True):
            try:
                run_dir = _run_dir(models_dir, state.dataset_name or "dataset", state.target_column or "target", state.last_run_id or "run")
                save_model_artifacts(run_dir, state.model, state.metadata)
                st.success(f"Zapisano: {run_dir}")
            except Exception as e:
                st.error(f"Nie udało się zapisać artefaktów: {e}")

        if st.button("Ustaw jako domyślny dla tego dataset/target", disabled=disabled, use_container_width=True):
            try:
                ok = tracker.set_default_model(state.dataset_name or "dataset", state.target_column or "target", state.last_run_id or "")
                if ok:
                    st.success("Ustawiono domyślny model.")
                else:
                    st.warning("Nie udało się ustawić domyślnego modelu.")
            except Exception as e:
                st.error(f"Błąd przy ustawianiu domyślnego modelu: {e}")

    with colR:
        st.subheader("Wczytaj model (Replay)")
        ds = state.dataset_name or "dataset"
        tg = state.target_column or "target"
        dirs = _list_model_dirs(models_dir, ds, tg)
        if not dirs:
            st.info("Brak zapisanych modeli dla tego dataset/target. Zapisz najpierw artefakty.")
        else:
            pick = st.selectbox("Wybierz artefakt (katalog)", options=[str(p) for p in dirs], index=0)
            colA, colB = st.columns(2)
            with colA:
                if st.button("Wczytaj model", use_container_width=True):
                    try:
                        model, meta = load_model_artifacts(Path(pick))
                        state.loaded_model_info = {"path": pick, "metadata": meta}
                        st.success("Model wczytany.")
                    except Exception as e:
                        st.error(f"Nie udało się wczytać modelu: {e}")
            with colB:
                if st.button("Predykcja na bieżących danych", use_container_width=True):
                    try:
                        model, meta = load_model_artifacts(Path(pick))
                        if state.target_column in state.dataset.columns:
                            X = state.dataset.drop(columns=[state.target_column])
                        else:
                            X = state.dataset
                        preds = model.predict(X)
                        df_out = state.dataset.copy()
                        df_out["prediction"] = preds
                        st.dataframe(df_out.head(min(50, len(df_out))), use_container_width=True)
                        st.download_button(
                            "Pobierz predykcje CSV",
                            df_out.to_csv(index=False).encode("utf-8"),
                            file_name=f"predictions__{ds}__{tg}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Nie udało się wykonać predykcji: {e}")

        st.caption("Domyślny model można ustawić po eksporcie. Zostanie skojarzony z parą (dataset, target).")

    # ===== 6) Historia uruchomień =====
    st.header("📚 Historia uruchomień")
    try:
        hist = tracker.get_history(QueryFilter(limit=500))
    except Exception as e:
        st.error(f"Nie udało się pobrać historii: {e}")
        hist = pd.DataFrame()

    if hist.empty:
        st.info("Brak historii. Uruchom trening, aby zapisać pierwsze uruchomienie.")
    else:
        # konwersja do lokalnej strefy (Europe/Warsaw) tylko do wyświetlenia
        if "created_at" in hist.columns:
            try:
                hist["_created_at_local"] = hist["created_at"].apply(
                    lambda d: to_local(pd.to_datetime(d, utc=True).to_pydatetime())
                )
                latest_local = max(hist["_created_at_local"])
                st.metric("Ostatnie uruchomienie", latest_local.strftime("%d.%m %H:%M"))
            except Exception:
                st.metric("Ostatnie uruchomienie", "-")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Łącznie uruchomień", len(hist))
        with c2:
            st.metric("Datasety", int(hist["dataset"].nunique()) if "dataset" in hist.columns else 1)
        with c3:
            st.metric("Targety", int(hist["target"].nunique()) if "target" in hist.columns else 1)
        with c4:
            try:
                st.metric("Zakończone", int((hist["status"] == "completed").sum()) if "status" in hist.columns else len(hist))
            except Exception:
                st.metric("Zakończone", "-")

        # Widok tabeli
        cols = [c for c in ["run_id", "dataset", "target", "engine", "status", "_created_at_local"] if (c in hist.columns or c == "_created_at_local")]
        view = hist.copy()
        if "_created_at_local" not in view.columns and "created_at" in view.columns:
            view["_created_at_local"] = view["created_at"]
        if "_created_at_local" in view.columns:
            view = view.sort_values("_created_at_local", ascending=False)
        st.dataframe(view[cols], use_container_width=True, hide_index=True)

        # Operacje na historii
        with st.expander("Operacje na historii"):
            run_ids = view["run_id"].tolist() if "run_id" in view.columns else []
            if run_ids:
                rid = st.selectbox("Wybierz run_id", options=run_ids, index=0)
                cA, cB, cC = st.columns(3)
                with cA:
                    if st.button("Ustaw jako domyślny (dla bieżącego dataset/target)"):
                        try:
                            ok = tracker.set_default_model(state.dataset_name or "dataset", state.target_column or "target", rid)
                            st.success("Ustawiono.") if ok else st.warning("Nie udało się ustawić.")
                        except Exception as e:
                            st.error(f"Błąd: {e}")
                with cB:
                    # spróbuj znaleźć katalog artefaktów dla tego run_id
                    rd = _run_dir(models_dir, state.dataset_name or "dataset", state.target_column or "target", rid)
                    exists = rd.exists()
                    st.write(f"Artefakty: {'✅' if exists else '—'} {rd.name}")
                with cC:
                    if st.button("Usuń wpis (tylko z historii)"):
                        try:
                            ok = tracker.delete_run(rid)
                            st.success("Usunięto wpis historii.") if ok else st.warning("Nie znaleziono.")
                        except Exception as e:
                            st.error(f"Błąd: {e}")

        # Domyślny model (podgląd)
        with st.expander("Domyślny model (dla bieżącego dataset/target)"):
            try:
                current_default = tracker.get_default_model(state.dataset_name or "dataset", state.target_column or "target")
                if current_default:
                    st.success(f"Domyślny run_id: {current_default}")
                    rd = _run_dir(models_dir, state.dataset_name or "dataset", state.target_column or "target", current_default)
                    st.write(f"Ścieżka artefaktów: {rd}")
                else:
                    st.info("Brak domyślnego modelu.")
            except Exception as e:
                st.error(f"Błąd: {e}")

    # ===== Panel debug (opcjonalnie) =====
    try:
        if bool(getattr(settings, "show_debug_panel", False)):
            with st.expander("🛠️ Panel debug"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write("ENV:", getattr(settings, "app_env", "DEV"))
                    st.write("Debug:", bool(getattr(settings, "debug", True)))
                    st.write("LLM key:", "✅" if get_openai_key_from_envs() else "—")
                with c2:
                    st.write("Output dir:", str(out_dir))
                    st.write("Models dir:", str(models_dir))
                    st.write("History DB:", getattr(settings, "history_db_path", "tmiv_out/history.sqlite"))
                with c3:
                    if state.dataset is not None:
                        st.write("Dataset shape:", state.dataset.shape)
                        st.write("Target:", state.target_column)
                    if state.metadata:
                        st.write("Engine:", state.metadata.get("engine"))
                        st.write("Problem:", state.metadata.get("problem_type"))
                        st.write("Train time [s]:", round(state.metadata.get("training_time_s", 0), 3))
    except Exception:
        pass

    # ===== Stopka =====
    st.caption("TMIV • Uruchomienia • Model Registry • ©")


if __name__ == "__main__":
    main()
