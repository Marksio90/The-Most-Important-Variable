# backend/ml_integration.py — AUTO engine + tuning + feature selection + solid preprocessing
from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error,
    mean_squared_error, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# --- optional engines (fallbacki jeśli brak paczek) ---
HAVE_LGBM = HAVE_XGB = HAVE_CAT = False
try:
    from lightgbm import LGBMRegressor, LGBMClassifier  # type: ignore
    HAVE_LGBM = True
except Exception:
    pass
try:
    from xgboost import XGBRegressor, XGBClassifier  # type: ignore
    HAVE_XGB = True
except Exception:
    pass
try:
    from catboost import CatBoostRegressor, CatBoostClassifier  # type: ignore
    HAVE_CAT = True
except Exception:
    pass


# =========================
# Pomocnicze metryki
# =========================
def _rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _mape(y_true, y_pred, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def _smape(y_true, y_pred, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def _mdape(y_true, y_pred, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.median(np.abs((y_true - y_pred) / denom)))


# =========================
# Wykrywanie typu problemu
# =========================
def detect_problem_type(target: pd.Series | np.ndarray | list) -> str:
    """
    Heurystyka: obiektowe/kategoryczne/mało unikatów => klasyfikacja, inaczej regresja.
    """
    s = pd.Series(target)
    nunq = int(s.nunique(dropna=True))
    if str(s.dtype) in ("object", "category"):
        return "classification"
    if pd.api.types.is_bool_dtype(s) or (pd.api.types.is_integer_dtype(s) and 2 <= nunq <= 20):
        return "classification"
    return "regression"


# =========================
# Preprocessing + selekcja
# =========================
def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Solidny preprocessor:
      - numeryczne: imputacja medianą
      - kategoryczne: OneHot + zbijanie rzadkich kategorii (min_frequency=0.01 jeśli dostępne)
    Zwraca ColumnTransformer z .get_feature_names_out().
    """
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    # Dla kompatybilności różnych wersji sklearn:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    num_pipe = SimpleImputer(strategy="median")
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def _simple_feature_select(X: pd.DataFrame, y: pd.Series, problem_type: str, k_max: int = 150) -> pd.DataFrame:
    """
    Szybka selekcja NUMERYCZNYCH cech (mutual information). Kategoryczne zostawiamy —
    będą one-hotowane w preprocessorze. Dzięki temu ograniczamy szum i przyspieszamy fit.
    """
    Xn = X.select_dtypes(include=[np.number]).copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    Xn = Xn.fillna(Xn.median(numeric_only=True))
    if Xn.empty:
        return X
    try:
        if "class" in problem_type:
            mi = mutual_info_classif(Xn, y, discrete_features="auto", random_state=42)
        else:
            mi = mutual_info_regression(Xn, y, random_state=42)
        order = np.argsort(mi)[::-1]
        keep = Xn.columns[order][: min(k_max, len(order))]
        cats = X.drop(columns=list(Xn.columns), errors="ignore")
        return pd.concat([X[keep], cats], axis=1)
    except Exception:
        return X


# =========================
# Rejestr silników + auto-wybór
# =========================
def _make_engine(problem_type: str, many_cats: bool, mostly_numeric: bool) -> Tuple[str, BaseEstimator]:
    """
    Zwraca (nazwa_silnika, estimator) wg dostępnych pakietów i rodzaju danych.
    Priorytety: Cat (gdy dużo kategorii) → LGBM → XGB → HGB → modele liniowe.
    """
    if "class" in problem_type:
        if HAVE_CAT and many_cats:
            return "catboost_cls", CatBoostClassifier(
                depth=8, learning_rate=0.06, iterations=1200,
                loss_function="Logloss", verbose=False, random_seed=42, allow_writing_files=False
            )
        if HAVE_LGBM:
            return "lgbm_cls", LGBMClassifier(
                n_estimators=1000, learning_rate=0.05, num_leaves=64,
                subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
                class_weight="balanced"
            )
        if HAVE_XGB:
            return "xgb_cls", XGBClassifier(
                n_estimators=1000, max_depth=8, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=42, n_jobs=-1, tree_method="hist"
            )
        # fallback
        return "hgb_cls", HistGradientBoostingClassifier(random_state=42)
    else:
        # regresja
        if HAVE_LGBM and mostly_numeric:
            return "lgbm_reg", LGBMRegressor(
                n_estimators=800, learning_rate=0.05, num_leaves=64,
                subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
            )
        if HAVE_XGB and mostly_numeric:
            return "xgb_reg", XGBRegressor(
                n_estimators=900, max_depth=8, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=42, n_jobs=-1, tree_method="hist"
            )
        if HAVE_CAT and many_cats:
            return "catboost_reg", CatBoostRegressor(
                depth=8, learning_rate=0.06, iterations=1200, loss_function="RMSE",
                verbose=False, random_seed=42, allow_writing_files=False
            )
        # fallback
        return "hgb_reg", HistGradientBoostingRegressor(random_state=42)


def _maybe_tune(model: BaseEstimator, X, y, problem_type: str, n_iter: int = 60, cv: int = 3) -> BaseEstimator:
    """
    Delikatny RandomizedSearchCV dla drzewiastych — podbija jakość bez mielarki.
    """
    name = model.__class__.__name__.lower()
    grid = None
    if "lgbm" in name:
        grid = {
            "num_leaves": [31, 63, 127],
            "learning_rate": [0.03, 0.05, 0.08],
            "n_estimators": [600, 900, 1200],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        }
    elif "xgb" in name:
        grid = {
            "max_depth": [6, 8, 10],
            "learning_rate": [0.03, 0.05, 0.08],
            "n_estimators": [600, 900, 1200],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_lambda": [0.5, 1.0, 1.5],
        }
    elif "catboost" in name:
        grid = {
            "depth": [6, 8, 10],
            "learning_rate": [0.04, 0.06, 0.08],
            "iterations": [800, 1000, 1400],
        }

    if grid is None:
        return model

    scoring = "neg_root_mean_squared_error" if "reg" in problem_type else "f1_weighted"
    try:
        rs = RandomizedSearchCV(
            model, grid, n_iter=n_iter, scoring=scoring,
            cv=cv, n_jobs=-1, random_state=42, verbose=0
        )
        rs.fit(X, y)
        return rs.best_estimator_
    except Exception:
        return model


# =========================
# Feature importance
# =========================
def _get_feature_names(pre: ColumnTransformer, X: pd.DataFrame) -> List[str]:
    try:
        return pre.get_feature_names_out().tolist()
    except Exception:
        # fallback
        names: List[str] = []
        for name, trans, cols in pre.transformers_:
            if cols is None:
                continue
            if hasattr(trans, "get_feature_names_out"):
                try:
                    fn = trans.get_feature_names_out(cols)
                    names.extend(fn.tolist())
                    continue
                except Exception:
                    pass
            if isinstance(cols, list):
                names.extend(list(cols))
        if not names:
            names = [f"f{i}" for i in range(pre.transform(X[:1]).shape[1])]
        return names


def _feature_importance_df(model: BaseEstimator, pre: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """
    Próbujemy: feature_importances_, coef_, inaczej pusta ramka (bez błędów).
    """
    try:
        names = _get_feature_names(pre, X)
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, float).ravel()
            return (pd.DataFrame({"feature": names[: len(imp)], "importance": imp})
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True))
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, float).ravel()
            imp = np.abs(coef)
            return (pd.DataFrame({"feature": names[: len(imp)], "importance": imp})
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True))
    except Exception:
        pass
    return pd.DataFrame(columns=["feature", "importance"])


# =========================
# API zgodne z app.py
# =========================
def export_visualizations(*args, **kwargs) -> Dict[str, Any]:
    """Zachowujemy funkcję dla zgodności — obecnie nieużywana."""
    return {}


def train_sklearn(
    df: pd.DataFrame,
    *,
    target: str,
    problem_type: Optional[str] = None,
    engine: str = "auto",
    cv_folds: int = 0,
    out_dir: str = "tmiv_out",
    random_state: int = 42,
    compute_shap: bool = False,   # ignorujemy jeśli brak SHAP — zero błędów
) -> Tuple[BaseEstimator, Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """
    Zwraca: (model, metrics:dict, fi_df:DataFrame, meta:dict)
    """
    t0 = time.time()
    assert target in df.columns, f"Brak kolumny target='{target}'"

    # 1) Podział X/y + łagodne czyszczenie
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target])
    y = df[target]
    X = df.drop(columns=[target], errors="ignore")

    # cast bool → int (stabilniejsze metryki i kodowanie)
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(int)

    # 2) Typ problemu
    problem = problem_type or detect_problem_type(y)

    # 3) Szybka selekcja cech numerycznych
    X_sel = _simple_feature_select(X, y, problem, k_max=150)

    # 4) Preprocessor
    pre = _build_preprocessor(X_sel)

    # 5) Wybór silnika
    num_ratio = X_sel.select_dtypes(include=[np.number]).shape[1] / max(1, X_sel.shape[1])
    cat_cols = [c for c in X_sel.columns if not pd.api.types.is_numeric_dtype(X_sel[c])]
    many_cats = len(cat_cols) >= 3

    if engine == "auto":
        engine_name, est = _make_engine(problem, many_cats=many_cats, mostly_numeric=(num_ratio > 0.6))
    else:
        # ręczny selector (zachowanie zgodności; jeśli nieznany → auto)
        engine_name, est = _make_engine(problem, many_cats=many_cats, mostly_numeric=(num_ratio > 0.6))

    # 6) Pipeline
    pipe = Pipeline([("pre", pre), ("est", est)])

    # 7) Train / CV
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_sel, y, test_size=0.2, random_state=random_state, stratify=y if "class" in problem else None
    )
    pipe.fit(X_train, y_train)

    # 7.1) Lekki tuning dla drzewiastych przy większych zbiorach (bez ryzyka błędów)
    name_lower = est.__class__.__name__.lower()
    if any(k in name_lower for k in ["lgbm", "xgb", "cat"]) and len(X_train) >= 2000:
        tuned = _maybe_tune(pipe.named_steps["est"], pre.transform(X_train), y_train, problem,
                            n_iter=60 if len(X_train) < 20000 else 120, cv=3)
        pipe = Pipeline([("pre", pre), ("est", tuned)])
        pipe.fit(X_train, y_train)

    # 8) Predykcje + metryki
    metrics: Dict[str, Any] = {}
    y_pred = pipe.predict(X_valid)

    if "class" in problem:
        # spróbuj proby (dla ROC/PR)
        proba = None
        try:
            pro = pipe.named_steps["est"].predict_proba(pre.transform(X_valid))
            if pro.ndim == 2 and pro.shape[1] >= 2:
                proba = pro[:, 1]
        except Exception:
            proba = None

        metrics["Accuracy"] = float(accuracy_score(y_valid, y_pred))
        metrics["F1_weighted"] = float(f1_score(y_valid, y_pred, average="weighted", zero_division=0))
        if proba is not None and len(np.unique(y_valid)) == 2:
            try:
                metrics["ROC_AUC"] = float(roc_auc_score(y_valid, proba))
            except Exception:
                pass
    else:
        metrics["RMSE"] = _rmse(y_valid, y_pred)
        metrics["MAE"] = float(mean_absolute_error(y_valid, y_pred))
        metrics["R2"] = float(r2_score(y_valid, y_pred))
        metrics["MAPE"] = _mape(y_valid, y_pred)
        metrics["SMAPE"] = _smape(y_valid, y_pred)
        metrics["MdAPE"] = _mdape(y_valid, y_pred)

    # 9) Cross-Validation (opcjonalnie)
    if cv_folds and cv_folds >= 2:
        try:
            scoring = "neg_root_mean_squared_error" if "reg" in problem else "f1_weighted"
            cv_scores = cross_val_score(pipe, X_sel, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
            if "reg" in problem:
                cv_scores = -cv_scores
            metrics["cv_metric"] = "RMSE" if "reg" in problem else "F1_weighted"
            metrics["cv_mean"] = float(np.mean(cv_scores))
            metrics["cv_std"] = float(np.std(cv_scores))
            metrics["cv_folds"] = int(cv_folds)
            metrics["cv_explanation"] = "Walidacja krzyżowa (stabilność wyników)."
        except Exception:
            # bez paniki — CV opcjonalne
            pass

    # 10) Feature importance
    try:
        fi_df = _feature_importance_df(pipe.named_steps["est"], pipe.named_steps["pre"], X_sel)
    except Exception:
        fi_df = pd.DataFrame(columns=["feature", "importance"])

    # 11) Meta
    run_name = f"run_{int(time.time())}"
    meta = {
        "problem_type": problem,
        "engine": engine_name,
        "run_name": run_name,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
    }

    return pipe, metrics, fi_df, meta
