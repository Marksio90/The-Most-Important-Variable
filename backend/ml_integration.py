from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import glob
import joblib
import os
import shutil
import time

import numpy as np
import pandas as pd
from zipfile import ZipFile, ZIP_DEFLATED

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    mean_absolute_percentage_error,   # <— brakujący import
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

# ===== LightGBM (opcjonalnie) =====
_HAS_LGBM = False
try:
    import lightgbm as lgb  # type: ignore
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False


# ===== helpers: bezpieczna stratyfikacja i CV =====
def _safe_stratify(y: pd.Series | np.ndarray, test_size: float, min_per_split: int = 1):
    """
    Zwraca `y` do stratify tylko jeśli każda klasa ma dość próbek,
    aby trafić zarazem do train i do test. W przeciwnym razie zwraca None.
    """
    if y is None:
        return None
    try:
        ys = pd.Series(y)
        counts = ys.value_counts(dropna=False)
        if counts.min() < 2:
            return None  # jakaś klasa ma tylko 1 próbkę
        # Czy w teście zmieści się min. 1 próbka z każdej klasy?
        enough_in_test = (np.floor(counts * float(test_size)) >= min_per_split).all()
        if not enough_in_test:
            return None
        return ys
    except Exception:
        return None


def _cap_cv_folds_for_classification(y: pd.Series | np.ndarray, requested_folds: int) -> int:
    """
    Dla klasyfikacji: liczba foldów nie może przekraczać najmniejszej liczności klasy.
    Jeżeli wyjdzie < 2, zwracamy 0 (wyłączenie CV).
    """
    try:
        ys = pd.Series(y)
        k = int(min(ys.value_counts(dropna=False)))
        folds = min(int(requested_folds), k)
        return folds if folds >= 2 else 0
    except Exception:
        return max(0, int(requested_folds))


def _smape(y_true, y_pred):
    import numpy as _np
    y_true = _np.asarray(y_true, dtype="float64")
    y_pred = _np.asarray(y_pred, dtype="float64")
    denom = (abs(y_true) + abs(y_pred))
    mask = denom > 0
    if not _np.any(mask):
        return float("nan")
    return float(_np.mean(2.0 * _np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))


# ===== Public API =====
def get_available_ml_engines() -> List[str]:
    engines = ["rf", "gb", "tree"]
    if _HAS_LGBM:
        engines.insert(0, "lgbm")
    engines.insert(0, "auto")
    return engines


def detect_problem_type(y: pd.Series | np.ndarray) -> str:
    """
    - datetime -> regresja
    - obiekty/kategorie/bool -> klasyfikacja
    - numeryczne: jeśli integer-like i mało klas (<=20) -> klasyfikacja, inaczej regresja
    """
    y_series = pd.Series(y)
    y_nona = y_series.dropna()
    if y_nona.empty:
        return "regression"

    if pd.api.types.is_datetime64_any_dtype(y_nona):
        return "regression"

    if pd.api.types.is_bool_dtype(y_nona) or y_nona.dtype.name in {"object", "category"}:
        return "classification"

    if pd.api.types.is_numeric_dtype(y_nona):
        nunique = int(pd.Series(y_nona).nunique())
        try:
            arr = pd.to_numeric(y_nona, errors="coerce").astype(float).to_numpy()
            is_int_like = np.allclose(arr, np.round(arr), equal_nan=True)
        except Exception:
            is_int_like = False
        if nunique <= 20 and is_int_like:
            return "classification"
        return "regression"

    nunique = int(y_nona.nunique())
    return "classification" if nunique <= 20 else "regression"


@dataclass
class TrainResult:
    model: BaseEstimator
    metrics: Dict[str, Any]
    fi_df: pd.DataFrame
    meta: Dict[str, Any]


def train_sklearn(
    X_or_df: pd.DataFrame | np.ndarray,
    y: Optional[pd.Series | np.ndarray] = None,
    *,
    target: Optional[str] = None,
    target_col: Optional[str] = None,
    problem_type: Optional[str] = None,
    engine: str = "auto",
    cv_folds: int = 3,
    out_dir: str = "tmiv_out",
    run_name: Optional[str] = None,
    random_state: int = 42,
    compute_shap: bool = False,
    shap_max_samples: int = 500,
    topk_fi: int = 50,
    log_run_cb: Optional[Any] = None,
) -> Tuple[BaseEstimator, Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """
    Wspiera:
    - (X, y, ...)
    - (df, target='col') / (df, target_col='col')
    - (df, 'col', ...)
    Zwraca: model, metrics, fi_df, meta
    """
    # --- Normalizacja wejścia ---
    if isinstance(y, str) and target is None and target_col is None:
        target = y
        y = None

    if y is None:
        if not isinstance(X_or_df, pd.DataFrame):
            raise ValueError("Gdy 'y' jest None, pierwszy argument musi być DataFrame z kolumną celu.")
        tgt = target or target_col
        if not tgt:
            raise ValueError("Podaj 'y' lub nazwę kolumny celu przez 'target' / 'target_col'.")
        if tgt not in X_or_df.columns:
            raise ValueError(f"Kolumna celu '{tgt}' nie występuje w DataFrame.")
        y_series = X_or_df[tgt].copy()
        X_df = X_or_df.drop(columns=[tgt]).copy()
        target_name = tgt
    else:
        X_df = X_or_df if isinstance(X_or_df, pd.DataFrame) else pd.DataFrame(X_or_df)
        y_series = pd.Series(y)
        target_name = target or target_col or "target"

    # --- Typ zadania ---
    task = (problem_type or detect_problem_type(y_series)).lower()

    # --- Kolumny num/kateg ---
    num_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    # --- Preprocessing ---
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    # OneHotEncoder – zgodność wersji
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01)
    except TypeError:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, min_frequency=0.01)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)])

    pre = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # --- Estymator + pipeline ---
    est = _build_estimator(task, engine, random_state)
    pipe = Pipeline(steps=[("pre", pre), ("model", est)])

    # --- Train/test split (bezpieczne) ---
    ts = 0.2  # sensowny default
    strat = _safe_stratify(y_series, test_size=ts) if task == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=ts, random_state=random_state, stratify=strat
        )
    except ValueError:
        # fallback: bez stratify (np. klasa z 1 rekordem)
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=ts, random_state=random_state, stratify=None
        )

    # --- Cross-Validation (opcjonalne) ---
    cv_mean = cv_std = None
    cv_metric_name = None
    folds = int(cv_folds) if isinstance(cv_folds, int) else 0
    if folds >= 2:
        if task == "classification":
            folds = _cap_cv_folds_for_classification(y_train, folds)
        if folds >= 2:
            try:
                cv_mean, cv_std, cv_metric_name = _do_cv(pipe, X_train, y_train, task, folds, random_state)
            except Exception:
                cv_mean = cv_std = None
                cv_metric_name = None

    # --- Trening ---
    pipe.fit(X_train, y_train)

    # --- Ewaluacja holdout ---
    metrics = _eval_metrics(pipe, X_test, y_test, task)
    if cv_metric_name is not None:
        metrics["cv_folds"] = int(folds)
        metrics["cv_metric"] = cv_metric_name
        metrics["cv_mean"] = float(cv_mean)
        metrics["cv_std"] = float(cv_std)
        metrics["cv_explanation"] = _cv_short_explanation(task)

    # --- Permutation Importance (lekko) ---
    X_pi, y_pi = X_test.copy(), y_test.copy()
    if len(X_pi) > 2000:
        X_pi = X_pi.sample(2000, random_state=random_state)
        y_pi = y_pi.loc[X_pi.index]

    y_pi_in = _to_numeric_target(y_pi) if task == "regression" else y_pi
    pi = permutation_importance(
        pipe,
        X_pi,
        y_pi_in,
        scoring=_primary_scorer(task),
        n_repeats=5,
        random_state=random_state,
    )
    imp_mean = pi.importances_mean

    # Nazwy cech do PI
    feat_names: List[str] = []
    if isinstance(X_pi, pd.DataFrame) and len(imp_mean) == X_pi.shape[1]:
        feat_names = list(map(str, X_pi.columns))
    else:
        try:
            transformed_names = list(map(str, pipe[:-1].get_feature_names_out()))
            if len(imp_mean) == len(transformed_names):
                feat_names = transformed_names
        except Exception:
            feat_names = []
    if not feat_names or len(feat_names) != len(imp_mean):
        feat_names = [f"feat_{i}" for i in range(len(imp_mean))]

    fi_df = (
        pd.DataFrame({"feature": feat_names, "importance": imp_mean})
        .sort_values("importance", ascending=False)
        .head(topk_fi)
        .reset_index(drop=True)
    )

    # --- Katalogi i nazwy artefaktów ---
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    engine_name = _engine_tag(est)
    run_name = run_name or _make_run_name(engine_name)
    run_dir = out_base / f"run_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Zapis modelu i artefaktów bazowych ---
    model_basename = f"model_{target_name}_{engine_name}_{run_name}.joblib"
    model_path = run_dir / model_basename
    joblib.dump(pipe, model_path)

    report = {
        "task": task,
        "engine": engine_name,
        "target": target_name,
        "n_rows": int(X_df.shape[0]),
        "n_cols": int(X_df.shape[1]),
        "metrics": metrics,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_path": str(model_path),
        "features_topk": fi_df.to_dict(orient="records"),
        "dataset_assessment": assess_dataset(X_df, y_series),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fi_df.to_csv(run_dir / "feature_importances.csv", index=False, encoding="utf-8")
    (run_dir / "model_card.md").write_text(
        _render_model_card(task, engine_name, target_name, metrics, X_df, fi_df),
        encoding="utf-8",
    )

    # --- Opcjonalny SHAP ---
    shap_info = None
    if compute_shap:
        try:
            shap_info = _compute_shap_summary(pipe, X_test, max_rows=shap_max_samples, random_state=random_state)
            (run_dir / "shap_summary.json").write_text(json.dumps(shap_info, indent=2), encoding="utf-8")
        except Exception as e:
            shap_info = {"error": repr(e)}

    # --- ZIP z całością ---
    zip_path = make_artifacts_zip(run_dir)

    # --- Meta ---
    meta: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "zip_path": str(zip_path),
        "model_path": str(model_path),
        "engine": engine_name,
        "target": target_name,
        "run_name": run_name,
        "n_rows": int(X_df.shape[0]),
        "n_cols": int(X_df.shape[1]),
        "features": list(X_df.columns),
        "problem_type": task,
        "shap_summary": shap_info,
    }

    # --- Log historii (opcjonalny callback) ---
    if log_run_cb:
        try:
            log_run_cb(
                run_id=run_name,
                engine=engine_name,
                target=target_name,
                problem=task,
                metrics=metrics,
                model_path=str(model_path),
                created_at=datetime.utcnow().isoformat() + "Z",
            )
        except Exception:
            pass

    return pipe, metrics, fi_df, meta


def _export_pycaret_plots(model, task: str, run_dir: Path) -> Dict[str, str]:
    "Generuje i zapisuje podstawowe wykresy PyCaret do run_dir."
    out: Dict[str, str] = {}
    try:
        if task == "regression":
            try:
                from pycaret.regression import plot_model as _plot
            except Exception:
                return out
            plots = ["residuals", "error", "feature", "learning"]
        else:
            try:
                from pycaret.classification import plot_model as _plot
            except Exception:
                return out
            plots = ["auc", "pr", "confusion_matrix", "feature", "learning"]

        run_dir.mkdir(parents=True, exist_ok=True)
        for p in plots:
            try:
                before = {x: os.path.getmtime(x) for x in glob.glob("*.png")}
                _plot(model, plot=p, save=True)
                time.sleep(0.2)
                after = [x for x in glob.glob("*.png") if x not in before or os.path.getmtime(x) > max(before.values() or [0])]
                if after:
                    src = sorted(after, key=os.path.getmtime)[-1]
                    dst = run_dir / f"pycaret_{p}.png"
                    try:
                        shutil.move(src, dst)
                    except Exception:
                        shutil.copy(src, dst)
                    out[p] = str(dst)
            except Exception:
                pass
    except Exception:
        pass
    return out


# ===== Wizualizacje na żądanie =====
def export_visualizations(
    model: BaseEstimator,
    X: pd.DataFrame,
    *,
    feature: Optional[str] = None,
    out_dir: str,
    run_name: str,
    max_tree_depth: int = 3,
    pdp_grid_resolution: int = 20,
    random_state: int = 42,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    run_dir = Path(out_dir) / f"run_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Drzewo 0 (jeśli możliwe)
    try:
        fig_path = run_dir / "model_graph_tree0.png"
        _plot_any_tree(model, X, fig_path, max_depth=max_tree_depth, random_state=random_state)
        if fig_path.exists() and fig_path.stat().st_size > 0:
            out["tree"] = str(fig_path)
            try:
                out["tree_bytes"] = fig_path.read_bytes()
            except Exception:
                pass
    except Exception:
        pass

    # PDP dla cechy
    if feature and isinstance(X, pd.DataFrame) and feature in X.columns:
        try:
            Xp = X.copy()
            if len(Xp) > 2000:
                Xp = Xp.sample(2000, random_state=random_state)
            pdp_path = run_dir / f"pdp_{feature}.png"
            _plot_pdp(model, Xp, feature, pdp_path, grid_resolution=pdp_grid_resolution)
            if pdp_path.exists() and pdp_path.stat().st_size > 0:
                out["pdp"] = str(pdp_path)
                try:
                    out["pdp_bytes"] = pdp_path.read_bytes()
                except Exception:
                    pass
        except Exception:
            pass

    return out


def make_artifacts_zip(run_dir: Path | str) -> Path:
    run_dir = Path(run_dir)
    zip_path = run_dir.with_suffix(".zip")
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
        for p in run_dir.rglob("*"):
            if p.is_file():
                arc = p.relative_to(run_dir.parent)
                zf.write(p, arcname=str(arc))
    return zip_path


# ===== ZGODNOŚĆ WSTECZNA =====
def save_model_artifacts(*args, **kwargs) -> str:
    """
    Zachowana zgodność ze starszymi wywołaniami.
    Zwraca ścieżkę do ZIP-a z artefaktami.
    """
    meta = None
    for a in args:
        if isinstance(a, dict) and any(k in a for k in ("run_dir", "zip_path", "run_name")):
            meta = a
            break
    if meta is None:
        cand = kwargs.get("meta")
        if isinstance(cand, dict):
            meta = cand

    zip_path = kwargs.get("zip_path")
    if zip_path:
        return str(zip_path)

    if meta and "zip_path" in meta:
        return str(meta["zip_path"])

    run_dir = kwargs.get("run_dir") or (meta.get("run_dir") if meta else None)
    if run_dir:
        return str(make_artifacts_zip(run_dir))

    out_dir = kwargs.get("out_dir") or (meta.get("out_dir") if meta else None) or "tmiv_out"
    run_name = kwargs.get("run_name") or (meta.get("run_name") if meta else None)
    if run_name:
        return str(make_artifacts_zip(Path(out_dir) / f"run_{run_name}"))

    raise ValueError(
        "save_model_artifacts: nie udało się ustalić lokalizacji runu. "
        "Przekaż meta={'run_dir': ...} lub out_dir + run_name."
    )


def assess_dataset(X: pd.DataFrame, y: Optional[pd.Series | np.ndarray] = None) -> Dict[str, Any]:
    X = X.copy()
    info: Dict[str, Any] = {}
    info["shape"] = {"rows": int(X.shape[0]), "cols": int(X.shape[1])}
    info["memory_mb"] = float(X.memory_usage(deep=True).sum() / 1e6)

    types = {"numeric": [], "categorical": [], "boolean": [], "datetime": [], "other": []}
    for c in X.columns:
        dt = X[c].dtype
        if pd.api.types.is_bool_dtype(dt):
            types["boolean"].append(c)
        elif pd.api.types.is_numeric_dtype(dt):
            types["numeric"].append(c)
        elif pd.api.types.is_datetime64_any_dtype(dt):
            types["datetime"].append(c)
        elif pd.api.types.is_string_dtype(dt) or pd.api.types.is_categorical_dtype(dt):
            types["categorical"].append(c)
        else:
            types["other"].append(c)
    info["types"] = {k: {"count": len(v), "columns": v[:50]} for k, v in types.items()}

    miss = X.isna().mean().sort_values(ascending=False)
    top_miss = miss[miss > 0].head(20)
    info["missing_top"] = [{"column": c, "pct": float(v)} for c, v in top_miss.items()]
    info["missing_total_pct"] = float(X.isna().sum().sum() / (X.shape[0] * max(1, X.shape[1])))

    info["duplicate_rows"] = int(X.duplicated().sum())
    nunq = X.nunique(dropna=False)
    const_cols = nunq[nunq <= 1].index.tolist()
    info["constant_cols"] = const_cols[:50]

    high_card = []
    for c in types["categorical"]:
        u = X[c].nunique(dropna=True)
        if u > min(100, 0.5 * len(X)):
            high_card.append({"column": c, "unique": int(u)})
    info["high_cardinality"] = high_card[:50]

    if y is not None:
        y_ser = pd.Series(y)
        info["target_nunique"] = int(y_ser.nunique(dropna=True))
        try:
            dist = (y_ser.value_counts(normalize=True)).to_dict()
            info["target_distribution"] = {str(k): float(v) for k, v in dist.items()}
            if len(dist) > 1:
                info["class_imbalance_ratio"] = float(min(dist.values()) / max(dist.values()))
        except Exception:
            pass

    return info


# ===== Helpers =====
def _build_estimator(task: str, engine: str, random_state: int) -> BaseEstimator:
    eng = (engine or "auto").lower()
    if eng == "auto":
        eng = "lgbm" if _HAS_LGBM else "rf"

    if task == "regression":
        if eng == "lgbm" and _HAS_LGBM:
            return lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=random_state
            )
        if eng == "gb":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(random_state=random_state)
        if eng == "tree":
            return DecisionTreeRegressor(max_depth=12, random_state=random_state)
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    else:
        if eng == "lgbm" and _HAS_LGBM:
            return lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=random_state
            )
        if eng == "gb":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(random_state=random_state)
        if eng == "tree":
            return DecisionTreeClassifier(max_depth=12, random_state=random_state)
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)


def _primary_scorer(task: str) -> str:
    return "neg_root_mean_squared_error" if task == "regression" else "f1_weighted"


def _to_numeric_target(y) -> np.ndarray:
    """
    Dla regresji (także datetime) konwertuje cel do float64.
    """
    ser = pd.Series(y)
    if pd.api.types.is_datetime64_any_dtype(ser):
        ser = pd.to_datetime(ser)
        arr = (ser.view("int64").astype("float64")) / 86_400_000_000_000.0  # ns -> dni
        return np.asarray(arr, dtype="float64")
    arr = pd.to_numeric(ser, errors="coerce").astype("float64")
    return np.asarray(arr, dtype="float64")


def _do_cv(pipe: Pipeline, X, y, task: str, cv_folds: int, random_state: int) -> Tuple[float, float, str]:
    scoring = _primary_scorer(task)
    if task == "regression":
        cv = KFold(n_splits=int(cv_folds), shuffle=True, random_state=random_state)
        y_in = _to_numeric_target(y)
        scores = cross_val_score(pipe, X, y_in, scoring=scoring, cv=cv, n_jobs=-1)
        rmse_vals = -scores  # neg_root_mean_squared_error -> dodatni RMSE
        return float(np.mean(rmse_vals)), float(np.std(rmse_vals)), "RMSE"
    else:
        # Klasyfikacja — stratyfikacja
        cv = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe, X, y, scoring=scoring, cv=cv, n_jobs=-1)
        return float(np.mean(scores)), float(np.std(scores)), "F1_weighted"


def _eval_metrics(model: Pipeline, X, y, task: str) -> Dict[str, Any]:
    pred = model.predict(X)
    out: Dict[str, Any] = {}

    if task == "regression":
        y_arr = _to_numeric_target(y)
        pred_arr = np.asarray(pd.to_numeric(pd.Series(pred), errors="coerce"), dtype="float64")

        mask = np.isfinite(y_arr) & np.isfinite(pred_arr)
        if not np.any(mask):
            return {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}

        y_a, p_a = y_arr[mask], pred_arr[mask]
        rmse = math.sqrt(mean_squared_error(y_a, p_a))
        mae = mean_absolute_error(y_a, p_a)
        try:
            r2 = r2_score(y_a, p_a) if np.unique(y_a).size > 1 else float("nan")
        except Exception:
            r2 = float("nan")

        try:
            mape = float(mean_absolute_percentage_error(y_a, p_a))
        except Exception:
            mape = float("nan")
        try:
            smape = float(_smape(y_a, p_a))
        except Exception:
            smape = float("nan")
        out.update({"RMSE": float(rmse), "MAE": float(mae), "MAPE": float(mape), "SMAPE": float(smape), "R2": float(r2)})
        return out

    # --- klasyfikacja ---
    pred_cls = pred
    proba = None
    try:
        proba = model.predict_proba(X)
    except Exception:
        pass

    out.update(
        {
            "Accuracy": float(accuracy_score(y, pred_cls)),
            "F1_weighted": float(f1_score(y, pred_cls, average="weighted", zero_division=0)),
        }
    )
    y_ser = pd.Series(y)
    if proba is not None and y_ser.nunique() == 2:
        try:
            out["ROC_AUC"] = float(roc_auc_score(y, proba[:, 1]))
        except Exception:
            pass
    return out


def _engine_tag(est: BaseEstimator) -> str:
    name = est.__class__.__name__.lower()
    if "lgbm" in name or "lightgbm" in name:
        return "lgbm"
    if "randomforest" in name:
        return "rf"
    if "gradientboosting" in name or "gb" in name:
        return "gb"
    if "decisiontree" in name or "tree" in name:
        return "tree"
    return name


def _make_run_name(engine_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{engine_name}"


def _cv_short_explanation(task: str) -> str:
    if task == "regression":
        return (
            "3-fold CV = podział danych na 3 części i rotacyjne testy; "
            "RMSE (pierwiastek z MSE – mocniej karze duże błędy). "
            "Podajemy średnią ± odchylenie, co pokazuje stabilność modelu."
        )
    else:
        return (
            "3-fold CV = podział danych na 3 części i rotacyjne testy; "
            "F1-weighted (średnia ważona precyzji i czułości; wagi = liczność klas). "
            "Średnia ± odchylenie ocenia stabilność modelu."
        )


def _render_model_card(
    task: str, engine: str, target_name: str, metrics: Dict[str, Any], X: pd.DataFrame, fi_df: pd.DataFrame
) -> str:
    lines = [
        f"# Model Card",
        "",
        f"**Zadanie:** {task}",
        f"**Silnik:** {engine}",
        f"**Cel (target):** `{target_name}`",
        "",
        "## Metryki",
    ]
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            lines.append(f"- **{k}:** {v:.5f}")
        elif isinstance(v, str):
            lines.append(f"- **{k}:** {v}")
    lines += [
        "",
        "*(RMSE = pierwiastek z błędu średniokwadratowego – mocniej karze duże błędy; "
        "MAE = średni błąd bezwzględny; R2 = dopasowanie; "
        "F1_weighted = średnia ważona precyzji i czułości; "
        "ROC_AUC = pole pod krzywą ROC [binarne]).*",
        "",
        "## Dane",
        f"- Wiersze: {X.shape[0]}",
        f"- Cechy: {X.shape[1]}",
        "",
        "## Najważniejsze cechy (Permutation Importance)",
        "",
    ]
    head = fi_df.head(20)
    for _, r in head.iterrows():
        lines.append(f"- {r['feature']}: {r['importance']:.6f}")
    lines.append("")
    lines.append("*(Permutation Importance = wpływ cechy mierzony spadkiem jakości po losowej permutacji kolumny).*")
    return "\n".join(lines)


def _plot_any_tree(model: BaseEstimator, X: pd.DataFrame, out_path: Path, max_depth: int, random_state: int = 42):
    import matplotlib.pyplot as plt

    clf = None
    if hasattr(model, "named_steps"):
        final_est = model.named_steps.get("model", model)
        if hasattr(final_est, "estimators_"):
            clf = final_est.estimators_[0]
        elif isinstance(final_est, (DecisionTreeClassifier, DecisionTreeRegressor)):
            clf = final_est
    else:
        est = model
        if hasattr(est, "estimators_"):
            clf = est.estimators_[0]
        elif isinstance(est, (DecisionTreeClassifier, DecisionTreeRegressor)):
            clf = est

    if clf is None:
        return

    import matplotlib
    matplotlib.use("Agg")  # bezpieczeństwo w środowiskach bez X

    plt.figure(figsize=(10, 7))
    try:
        plot_tree(clf, max_depth=max_depth, filled=False, fontsize=8)
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=160)
    finally:
        plt.close()


def _plot_pdp(model: BaseEstimator, X: pd.DataFrame, feature: str, out_path: Path, grid_resolution: int = 20):
    import matplotlib.pyplot as plt

    fig = PartialDependenceDisplay.from_estimator(
        model, X, [feature], grid_resolution=grid_resolution, kind="average"
    )
    fig.figure_.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.figure_.savefig(out_path, dpi=160)
    plt.close(fig.figure_)


def _compute_shap_summary(
    model: Pipeline, X: pd.DataFrame, max_rows: int = 500, random_state: int = 42
) -> Dict[str, Any]:
    try:
        import shap  # type: ignore
    except Exception as e:
        return {"available": False, "reason": f"SHAP niedostępny: {e!r}"}

    Xs = X.copy()
    if len(Xs) > max_rows:
        Xs = Xs.sample(max_rows, random_state=random_state)

    explainer = shap.Explainer(model.predict, Xs)
    vals = explainer(Xs)
    mean_abs = np.abs(vals.values).mean(axis=0)

    try:
        feature_names = model[:-1].get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(len(mean_abs))]

    order = np.argsort(-mean_abs)
    top = [{"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])} for i in order[: min(50, len(order))]]
    return {"available": True, "n_rows": int(len(Xs)), "top": top}
