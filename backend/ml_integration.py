# ml_integration.py — TMIV: trenowanie, metryki, FI, artefakty (sklearn + opcjonalnie LGBM/XGB/CB)
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import json
import numpy as np
import pandas as pd

# Bazowe ML
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)

from sklearn.ensemble import (
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)

# Opcjonalne silniki
def _soft_import(name: str):
    try:
        module = __import__(name)
        return module
    except Exception:
        return None

XGB = _soft_import("xgboost")
LGBM = _soft_import("lightgbm")
CATB = _soft_import("catboost")

# Utils (nasze)
from backend.utils import (
    seed_everything, infer_problem_type,
    hash_dataframe_signature
)

# ---------------------------
# Konfiguracja / kontrakty
# ---------------------------
EngineName = Literal["auto", "sklearn", "lightgbm", "xgboost", "catboost", "pycaret"]

@dataclass
class ModelConfig:
    target: str
    engine: EngineName = "auto"
    test_size: float = 0.2
    cv_folds: int = 3  # (na przyszłość; obecnie holdout)
    random_state: int = 42
    stratify: bool = True
    # ewentualne flagi
    enable_probabilities: bool = True
    max_categories: int = 200  # odcięcie rzadkich kategorii (na przyszłość)

@dataclass
class TrainingResult:
    model: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    feature_importance: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["feature", "importance"]))
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------
# API publiczne
# ---------------------------
def train_model_comprehensive(
    df: pd.DataFrame,
    cfg: ModelConfig,
    use_advanced: bool = True,
) -> TrainingResult:
    """
    Jeden punkt wejścia:
    - automatyczna detekcja typu problemu,
    - preprocessing (imputacja num/kat + one-hot),
    - wybór silnika (AUTO preferuje LGBM/XGB/CB gdy dostępny),
    - split, trening, metryki, FI,
    - metadata.validation_info: y_true, y_pred, labels (gdy ma sens).
    """
    # Seed
    seed_everything(cfg.random_state)

    if cfg.target not in df.columns:
        raise ValueError(f"Brak kolumny celu '{cfg.target}' w danych.")

    # Oddziel X/y
    y = df[cfg.target]
    X = df.drop(columns=[cfg.target])

    if y.isna().any():
        raise ValueError("Kolumna celu zawiera wartości puste (NaN). Uzupełnij lub odfiltruj przed treningiem.")

    # Detekcja problemu (heurystyka)
    problem_type = infer_problem_type(df, cfg.target).lower()
    if problem_type not in ("regression", "classification"):
        # fallback: numeric & >=3 unikatowe -> regression, inaczej classification
        if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) >= 3:
            problem_type = "regression"
        else:
            problem_type = "classification"

    # Podział cech wg dtype
    num_cols = X.select_dtypes(include=["number", "float", "int", "float64", "int64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # OneHotEncoder kompatybilny ze starszym sklearn (sparse_output vs sparse)
    def _ohe_safe() -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # starsze sklearn
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    # Preprocessing
    num_tr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_tr = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", _ohe_safe())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_tr, num_cols),
            ("cat", cat_tr, cat_cols)
        ],
        remainder="drop",
        n_jobs=None
    )

    # Wybór silnika
    selected_engine = _select_engine(cfg.engine, problem_type)

    # Model bazowy w zależności od typu
    model = _build_model(selected_engine, problem_type, cfg.random_state)

    # Pipeline
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    # ---------- Split (bezpieczny dla rzadkich klas) ----------
    warnings_local: List[str] = []
    stratify_param = None
    if problem_type == "classification" and cfg.stratify:
        class_counts = y.value_counts()
        min_class = int(class_counts.min()) if len(class_counts) > 0 else 0
        if y.nunique() > 1 and min_class >= 2:
            stratify_param = y
        else:
            warnings_local.append(
                f"Wyłączono stratify przy train/test — najmniej liczna klasa ma {min_class} próbek."
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify_param
    )

    # Jeśli w klasyfikacji train ma tylko jedną klasę → fallback na Dummy
    from sklearn.dummy import DummyClassifier, DummyRegressor
    if problem_type == "classification" and pd.Series(y_train).nunique() <= 1:
        warnings_local.append(
            "Zbiór treningowy ma jedną klasę — użyto DummyClassifier(strategy='most_frequent')."
        )
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", DummyClassifier(strategy="most_frequent"))
        ])

    # ---------- Fit (z zabezpieczeniem) ----------
    try:
        pipe.fit(X_train, y_train)
    except Exception as fit_err:
        # Miękki fallback (np. gdy model nie radzi sobie z danymi)
        warnings_local.append(f"Fallback na Dummy z powodu błędu treningu: {type(fit_err).__name__}.")
        if problem_type == "classification":
            pipe = Pipeline([
                ("prep", preprocessor),
                ("model", DummyClassifier(strategy="most_frequent"))
            ])
        else:
            pipe = Pipeline([
                ("prep", preprocessor),
                ("model", DummyRegressor(strategy="mean"))
            ])
        pipe.fit(X_train, y_train)

    # Predict
    y_pred = pipe.predict(X_test)

    # Metryki
    metrics = _compute_metrics(problem_type, y_test, y_pred, pipe, X_test, cfg)

    # Feature importance (z modelu po preprocesie)
    fi_df = _extract_feature_importance(pipe, preprocessor, num_cols, cat_cols)

    # Validation info (dla wykresów w app)
    validation_info = _build_validation_info(problem_type, y_test, y_pred, pipe, X_test, cfg)

    # Metadata
    meta: Dict[str, Any] = {
        "engine": selected_engine,
        "problem_type": problem_type,
        "n_rows": int(len(df)),
        "n_features_raw": int(X.shape[1]),
        "n_features_after_preproc": int(fi_df.shape[0]) if fi_df is not None and not fi_df.empty else None,
        "feature_names": fi_df["feature"].tolist() if fi_df is not None and not fi_df.empty else list(X.columns),
        "validation_info": validation_info,
        "data_signature": hash_dataframe_signature(df)
    }
    if warnings_local:
        meta["warnings"] = warnings_local

    return TrainingResult(
        model=pipe,
        metrics=metrics,
        feature_importance=fi_df if fi_df is not None else pd.DataFrame(columns=["feature", "importance"]),
        metadata=meta
    )


# ---------------------------
# Artefakty modelu (replay)
# ---------------------------
def save_model_artifacts(run_dir: Path, model: Any, meta: Dict[str, Any]) -> None:
    """
    Zapisuje pipeline (z preprocessingiem) + metadane JSON.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
    except Exception as e:
        raise RuntimeError(f"Brak joblib do zapisu modelu: {e}")

    joblib.dump(model, run_dir / "model.joblib")
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def load_model_artifacts(run_dir: Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Ładuje pipeline i metadane.
    """
    run_dir = Path(run_dir)
    try:
        import joblib
    except Exception as e:
        raise RuntimeError(f"Brak joblib do odczytu modelu: {e}")

    model = joblib.load(run_dir / "model.joblib")
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    return model, meta


# ---------------------------
# Selekcja silnika i budowa
# ---------------------------
def _select_engine(engine: EngineName, problem_type: str) -> str:
    """
    AUTO: preferencja LGBM > XGB > CAT > sklearn
    """
    if engine != "auto":
        return engine

    if LGBM is not None:
        return "lightgbm"
    if XGB is not None:
        return "xgboost"
    if CATB is not None:
        return "catboost"
    return "sklearn"


def _build_model(engine: str, problem_type: str, random_state: int):
    """
    Tworzy model bazowy pod dany silnik i typ problemu.
    Minimalny zestaw parametrów, ale deterministyczny (seed).
    """
    if engine == "lightgbm" and LGBM is not None:
        if problem_type == "regression":
            return LGBM.LGBMRegressor(random_state=random_state, n_estimators=300)
        else:
            return LGBM.LGBMClassifier(random_state=random_state, n_estimators=300)
    if engine == "xgboost" and XGB is not None:
        if problem_type == "regression":
            return XGB.XGBRegressor(random_state=random_state, n_estimators=400, tree_method="hist", n_jobs=0)
        else:
            return XGB.XGBClassifier(random_state=random_state, n_estimators=400, tree_method="hist", n_jobs=0, use_label_encoder=False, eval_metric="logloss")
    if engine == "catboost" and CATB is not None:
        # CatBoost ma własne encodery, ale tu używamy naszego preprocesu; ustawiamy prosty model
        if problem_type == "regression":
            return CATB.CatBoostRegressor(random_state=random_state, iterations=300, verbose=False)
        else:
            return CATB.CatBoostClassifier(random_state=random_state, iterations=300, verbose=False)

    # sklearn (domyślnie)
    if problem_type == "regression":
        # szybki i dość mocny baseline
        return HistGradientBoostingRegressor(random_state=random_state)
    else:
        # baseline klasyfikacyjny
        return HistGradientBoostingClassifier(random_state=random_state)


# ---------------------------
# Metryki, FI, walidacje
# ---------------------------
def _compute_metrics(
    problem_type: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    pipe: Pipeline,
    X_test: pd.DataFrame,
    cfg: ModelConfig
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    if problem_type == "regression":
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        metrics.update({
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })
    else:
        # klasyfikacja
        acc = float(accuracy_score(y_true, y_pred))
        # Macro-F1 do niebalansów
        f1 = float(f1_score(y_true, y_pred, average="macro"))
        metrics.update({
            "accuracy": acc,
            "f1_macro": f1
        })

        # ROC-AUC (tylko dla przypadków z predict_proba i sensowną liczbą klas)
        try:
            if cfg.enable_probabilities and hasattr(pipe.named_steps["model"], "predict_proba"):
                proba = pipe.predict_proba(X_test)
                if proba is not None:
                    if proba.ndim == 1 or proba.shape[1] == 2:
                        # AUC dla klasy 1 (binarna)
                        pos = proba[:, 1] if proba.ndim > 1 else proba
                        # może się wywalić, jeśli y_true ma jedną klasę → try/except wyżej
                        auc = float(roc_auc_score(y_true, pos))
                        metrics["roc_auc"] = auc
                    else:
                        # Wieloklasowa (macro OVR)
                        auc = float(roc_auc_score(y_true, proba, multi_class="ovr"))
                        metrics["roc_auc_ovr_macro"] = auc
        except Exception:
            pass

    return metrics


def _expanded_feature_names(pre: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    """
    Zwraca nazwy cech po transformacji (uwzględnia one-hot).
    """
    feature_names: List[str] = []

    # num
    if len(num_cols) > 0:
        feature_names.extend(num_cols)

    # cat (OneHotEncoder)
    try:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        cats = ohe.get_feature_names_out(cat_cols)
        feature_names.extend(cats.tolist())
    except Exception:
        # jeśli nie ma OHE (brak kat), nic nie dodawaj
        pass

    return feature_names


def _extract_feature_importance(
    pipe: Pipeline,
    pre: ColumnTransformer,
    num_cols: List[str],
    cat_cols: List[str]
) -> Optional[pd.DataFrame]:
    """
    Wyciąga FI z modelu, gdy to możliwe (feature_importances_ / coef_).
    Zwraca DataFrame: feature, importance (posortowane malejąco).
    """
    try:
        model = pipe.named_steps["model"]
    except Exception:
        return None

    feat_names = _expanded_feature_names(pre, num_cols, cat_cols)
    importances: Optional[np.ndarray] = None

    # Drzewa
    for attr in ("feature_importances_",):
        if hasattr(model, attr):
            importances = getattr(model, attr)
            break

    # Liniowe
    if importances is None and hasattr(model, "coef_"):
        coef = getattr(model, "coef_")
        if isinstance(coef, np.ndarray):
            # multiclass -> średnia wartości bezwzględnych po klasach
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            importances = np.abs(coef)

    if importances is None:
        return None

    # Bezpieczeństwo długości
    importances = np.ravel(importances)
    n = min(len(importances), len(feat_names))
    data = pd.DataFrame({
        "feature": feat_names[:n],
        "importance": importances[:n]
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return data


def _build_validation_info(
    problem_type: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    pipe: Pipeline,
    X_test: pd.DataFrame,
    cfg: ModelConfig
) -> Dict[str, Any]:
    """
    Zbiera dane do wykresów w UI: y_true/y_pred (regresja),
    confusion (klasyfikacja) oraz etykiety.
    Zwraca mały wektor (pierwsze 20k), by nie zalewać UI.
    """
    out: Dict[str, Any] = {}
    MAX_N = 20000

    try:
        yt = np.asarray(y_true)[:MAX_N]
        yp = np.asarray(y_pred)[:MAX_N]
    except Exception:
        yt, yp = None, None

    if problem_type == "regression":
        if yt is not None and yp is not None:
            out["y_true"] = yt.tolist()
            out["y_pred"] = yp.tolist()
    else:
        # klasyfikacja
        if yt is not None and yp is not None:
            out["y_true"] = yt.tolist()
            out["y_pred"] = yp.tolist()
            try:
                labels = sorted(list(set(list(yt) + list(yp))))
                cm = confusion_matrix(yt, yp, labels=labels)
                out["labels"] = [str(l) for l in labels]
                out["confusion_matrix"] = cm.astype(int).tolist()
            except Exception:
                pass

    return out
