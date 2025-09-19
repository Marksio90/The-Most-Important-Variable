# backend/ml_integration.py — NAPRAWIONE: bezpieczna stratyfikacja, stabilny preprocessing, czytelne metadane
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

# Wersja sklearn (do metadanych/diagnoz)
import sklearn
SKLEARN_VERSION = sklearn.__version__

# Opcjonalne silniki — import miękki
def _soft_import(name: str):
    try:
        module = __import__(name)
        return module
    except Exception:
        return None

XGB = _soft_import("xgboost")
LGBM = _soft_import("lightgbm")
CATB = _soft_import("catboost")

from backend.utils import (
    seed_everything, infer_problem_type,
    hash_dataframe_signature
)

# ---------------------------
# Typy/kontrakty
# ---------------------------
EngineName = Literal["auto", "sklearn", "lightgbm", "xgboost", "catboost", "pycaret"]

@dataclass
class ModelConfig:
    target: str
    engine: EngineName = "auto"
    test_size: float = 0.2
    cv_folds: int = 3
    random_state: int = 42
    stratify: bool = True
    enable_probabilities: bool = True
    max_categories: int = 200  # do ewentualnych walidacji w UI

@dataclass
class TrainingResult:
    model: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    feature_importance: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["feature", "importance"]))
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------
# Pomocnicze: stratyfikacja/OHE/kolumny
# ---------------------------
def _safe_stratify_check(y: pd.Series, min_samples_per_class: int = 2) -> bool:
    """
    True tylko jeśli każda klasa ma >= min_samples_per_class.
    Zapobiega ValueError: least populated class in y has only 1 member.
    """
    try:
        vc = y.value_counts(dropna=True)
        if len(vc) < 2:
            return False
        return (vc >= min_samples_per_class).all()
    except Exception:
        return False


def _create_ohe_compatible() -> OneHotEncoder:
    """Kompatybilny OneHotEncoder dla różnych wersji sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
        except Exception:
            return OneHotEncoder(handle_unknown="ignore")


def _safe_categorical_columns(df: pd.DataFrame, exclude_cols: List[str]) -> List[str]:
    """
    Wybiera kolumny kategoryczne: object/category/bool oraz wąskie integerowe.
    Pomija całkowicie puste.
    """
    cat_cols: List[str] = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        s = df[col]
        if s.isna().all():
            continue

        dt = s.dtype
        if dt in ["object", "category", "bool"]:
            cat_cols.append(col)
        elif dt in ["int8", "int16", "int32", "int64"]:
            nunq = s.nunique(dropna=True)
            if nunq <= 50:
                cat_cols.append(col)
    return cat_cols


# ---------------------------
# API publiczne: główny trening
# ---------------------------
def train_model_comprehensive(
    df: pd.DataFrame,
    cfg: ModelConfig,
    use_advanced: bool = True,
) -> TrainingResult:
    """
    Główny punkt wejścia. Buduje pipeline (imputacja + OHE + model),
    dzieli dane z bezpieczną stratyfikacją (jeśli możliwa) i zwraca
    model + metryki + ważności + metadane.
    """
    seed_everything(cfg.random_state)

    if cfg.target not in df.columns:
        raise ValueError(f"Brak kolumny celu '{cfg.target}' w danych.")

    # X / y
    y = df[cfg.target]
    X = df.drop(columns=[cfg.target])

    if y.isna().any():
        raise ValueError("Kolumna celu zawiera wartości puste (NaN). Uzupełnij lub odfiltruj przed treningiem.")

    # Typ problemu (heurystyka)
    problem_type = infer_problem_type(df, cfg.target).lower()
    if problem_type not in ("regression", "classification"):
        if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) >= 3:
            problem_type = "regression"
        else:
            problem_type = "classification"

    # Bezpieczna stratyfikacja
    can_stratify = False
    stratify_param = None
    if problem_type == "classification" and cfg.stratify:
        can_stratify = _safe_stratify_check(y, min_samples_per_class=2)
        stratify_param = y if can_stratify else None

    # Podział cech wg dtype
    num_cols = X.select_dtypes(include=["number", "float", "int", "float64", "int64"]).columns.tolist()
    cat_cols = _safe_categorical_columns(X, exclude_cols=num_cols)

    if not num_cols and not cat_cols:
        raise ValueError("Brak kolumn do przetworzenia — ani numerycznych, ani kategorycznych!")

    # Preprocessing
    transformers = []
    if num_cols:
        num_tr = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        transformers.append(("num", num_tr, num_cols))
    if cat_cols:
        cat_tr = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", _create_ohe_compatible())
        ])
        transformers.append(("cat", cat_tr, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", n_jobs=None)

    # Wybór i budowa modelu
    selected_engine = _select_engine(cfg.engine, problem_type)
    model = _build_model(selected_engine, problem_type, cfg.random_state)

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    # Podział danych (fallback na brak stratyfikacji)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=stratify_param
        )
    except ValueError as e:
        if "least populated class" in str(e) or "too few" in str(e):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=None
            )
            can_stratify = False
        else:
            raise

    # Fit
    pipe.fit(X_train, y_train)

    # Predict
    y_pred = pipe.predict(X_test)

    # Metryki
    metrics = _compute_metrics(problem_type, y_test, y_pred, pipe, X_test, cfg)

    # Feature importance
    fi_df = _extract_feature_importance(pipe, preprocessor, num_cols, cat_cols)

    # Validation info
    validation_info = _build_validation_info(problem_type, y_test, y_pred, pipe, X_test, cfg)

    # Ostrzeżenia
    warnings: List[str] = []
    if problem_type == "classification" and not can_stratify:
        warnings.append("Nie można użyć stratyfikacji — niektóre klasy mają za mało przykładów.")

    if problem_type == "classification":
        vc = y.value_counts()
        if len(vc) > 1 and vc.min() > 0:
            imbalance_ratio = vc.max() / vc.min()
            if imbalance_ratio > 10:
                warnings.append(f"Silny niebalans klas (ratio: {imbalance_ratio:.1f}:1)")

    meta: Dict[str, Any] = {
        "engine": selected_engine,
        "problem_type": problem_type,
        "n_rows": int(len(df)),
        "n_features_raw": int(X.shape[1]),
        "n_features_after_preproc": int(fi_df.shape[0]) if fi_df is not None and not fi_df.empty else None,
        "feature_names": fi_df["feature"].tolist() if fi_df is not None and not fi_df.empty else list(X.columns),
        "validation_info": validation_info,
        "data_signature": hash_dataframe_signature(df),
        "sklearn_version": SKLEARN_VERSION,
        "num_cols_count": len(num_cols),
        "cat_cols_count": len(cat_cols),
        "warnings": warnings,
        "stratified": can_stratify,
        "class_distribution": y.value_counts().to_dict() if problem_type == "classification" else None,
    }

    return TrainingResult(
        model=pipe,
        metrics=metrics,
        feature_importance=fi_df if fi_df is not None else pd.DataFrame(columns=["feature", "importance"]),
        metadata=meta
    )


# ---------------------------
# Zapis/odczyt artefaktów
# ---------------------------
def save_model_artifacts(run_dir: Path, model: Any, meta: Dict[str, Any]) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
    except Exception as e:
        raise RuntimeError(f"Brak joblib do zapisu modelu: {e}")
    joblib.dump(model, run_dir / "model.joblib")
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def load_model_artifacts(run_dir: Path) -> Tuple[Any, Dict[str, Any]]:
    run_dir = Path(run_dir)
    try:
        import joblib
    except Exception as e:
        raise RuntimeError(f"Brak joblib do odczytu modelu: {e}")
    model = joblib.load(run_dir / "model.joblib")
    meta = json.loads((run_dir / "meta.json").read_text())
    return model, meta


# ---------------------------
# Wybór i konstrukcja modelu
# ---------------------------
def _select_engine(engine: EngineName, problem_type: str) -> str:
    if engine != "auto":
        return engine
    # heurystyka: preferuj gradient boosting zewnętrzny, jeśli dostępny
    if LGBM is not None:
        return "lightgbm"
    if XGB is not None:
        return "xgboost"
    if CATB is not None:
        return "catboost"
    return "sklearn"


def _build_model(engine: str, problem_type: str, random_state: int):
    if engine == "lightgbm" and LGBM is not None:
        if problem_type == "regression":
            return LGBM.LGBMRegressor(random_state=random_state, n_estimators=300, verbosity=-1)
        else:
            return LGBM.LGBMClassifier(random_state=random_state, n_estimators=300, verbosity=-1)

    if engine == "xgboost" and XGB is not None:
        if problem_type == "regression":
            return XGB.XGBRegressor(random_state=random_state, n_estimators=400, tree_method="hist", n_jobs=1)
        else:
            return XGB.XGBClassifier(
                random_state=random_state,
                n_estimators=400,
                tree_method="hist",
                n_jobs=1,
                use_label_encoder=False,
                eval_metric="logloss",
            )

    if engine == "catboost" and CATB is not None:
        if problem_type == "regression":
            return CATB.CatBoostRegressor(random_state=random_state, iterations=300, verbose=False)
        else:
            return CATB.CatBoostClassifier(random_state=random_state, iterations=300, verbose=False)

    # fallback: szybkie i sensowne modele sklearn
    if problem_type == "regression":
        return HistGradientBoostingRegressor(random_state=random_state)
    else:
        return HistGradientBoostingClassifier(random_state=random_state)


# ---------------------------
# Metryki / importance / walidacja
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
        metrics.update({"mae": mae, "rmse": rmse, "r2": r2})

    else:
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro"))
        metrics.update({"accuracy": acc, "f1_macro": f1})

        # AUC (binarny / wieloklasowy, jeśli dostępne proby)
        try:
            if cfg.enable_probabilities and hasattr(pipe["model"], "predict_proba"):
                proba = pipe.predict_proba(X_test)
                if proba is not None:
                    if proba.ndim == 1 or proba.shape[1] == 2:
                        pos = proba[:, 1] if proba.ndim > 1 else proba
                        auc = float(roc_auc_score(y_true, pos))
                        metrics["roc_auc"] = auc
                    else:
                        auc = float(roc_auc_score(y_true, proba, multi_class="ovr"))
                        metrics["roc_auc_ovr_macro"] = auc
        except Exception:
            pass

    return metrics


def _expanded_feature_names(pre: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    feat_names: List[str] = []
    if num_cols:
        feat_names.extend(num_cols)
    if cat_cols:
        try:
            ohe = pre.named_transformers_["cat"].named_steps["ohe"]
            cats = ohe.get_feature_names_out(cat_cols)
            feat_names.extend(cats.tolist())
        except Exception:
            feat_names.extend(cat_cols)
    return feat_names


def _extract_feature_importance(
    pipe: Pipeline,
    pre: ColumnTransformer,
    num_cols: List[str],
    cat_cols: List[str]
) -> Optional[pd.DataFrame]:
    try:
        model = pipe.named_steps["model"]
    except Exception:
        return None

    feat_names = _expanded_feature_names(pre, num_cols, cat_cols)
    importances: Optional[np.ndarray] = None

    # Modele drzewiaste
    if hasattr(model, "feature_importances_"):
        importances = getattr(model, "feature_importances_", None)

    # Modele liniowe (coef_)
    if importances is None and hasattr(model, "coef_"):
        coef = getattr(model, "coef_")
        if isinstance(coef, np.ndarray):
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            importances = np.abs(coef)

    if importances is None:
        return None

    importances = np.ravel(importances)
    n = min(len(importances), len(feat_names))
    df_imp = pd.DataFrame({
        "feature": feat_names[:n],
        "importance": importances[:n]
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df_imp


def _build_validation_info(
    problem_type: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    pipe: Pipeline,
    X_test: pd.DataFrame,
    cfg: ModelConfig
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    MAX_N = 20000

    try:
        yt = np.asarray(y_true)[:MAX_N]
        yp = np.asarray(y_pred)[:MAX_N]
    except Exception:
        yt, yp = None, None

    if yt is not None and yp is not None:
        out["y_true"] = yt.tolist()
        out["y_pred"] = yp.tolist()

    if problem_type == "classification" and yt is not None and yp is not None:
        try:
            labels = sorted(list(set(list(yt) + list(yp))))
            cm = confusion_matrix(yt, yp, labels=labels)
            out["labels"] = [str(l) for l in labels]
            out["confusion_matrix"] = cm.astype(int).tolist()
        except Exception:
            pass

    return out
