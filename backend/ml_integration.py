# backend/ml_integration.py — ULEPSZONY: więcej opcji treningu, lepsze metryki, zaawansowana optymalizacja
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import json
import numpy as np
import pandas as pd
import time

# Bazowe ML
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.decomposition import PCA

from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score,
    classification_report, explained_variance_score, max_error
)

from sklearn.ensemble import (
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier,
    RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

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

# Imbalanced-learn (opcjonalne)
IMBLEARN = _soft_import("imblearn")

from backend.utils import (
    seed_everything, infer_problem_type,
    hash_dataframe_signature
)

# ---------------------------
# Typy/kontrakty - ROZSZERZONE
# ---------------------------
EngineName = Literal["auto", "sklearn", "lightgbm", "xgboost", "catboost", "pycaret"]

@dataclass
class ModelConfig:
    """Rozszerzona konfiguracja modelu z dodatkowymi opcjami."""
    target: str
    engine: EngineName = "auto"
    test_size: float = 0.2
    cv_folds: int = 3
    random_state: int = 42
    stratify: bool = True
    enable_probabilities: bool = True
    max_categories: int = 200
    
    # NOWE OPCJE ZAAWANSOWANE
    feature_engineering: bool = False
    feature_selection: bool = False
    handle_imbalance: bool = False
    hyperparameter_tuning: bool = False
    early_stopping: bool = False
    ensemble_methods: bool = False
    
    # Preprocessing opcje
    scaler_type: str = "standard"  # standard, robust, power, none
    imputer_type: str = "simple"   # simple, knn, iterative
    
    # Feature selection opcje
    selection_k: int = 10
    selection_method: str = "f_test"  # f_test, rfe, pca
    
    # Ensemble opcje
    ensemble_type: str = "voting"  # voting, stacking, bagging


@dataclass
class TrainingResult:
    """Rozszerzony wynik treningu z dodatkowymi metrykami."""
    model: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    feature_importance: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["feature", "importance"]))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # NOWE POLA
    cross_val_scores: Dict[str, List[float]] = field(default_factory=dict)
    training_history: Dict[str, List[float]] = field(default_factory=dict)
    best_params: Dict[str, Any] = field(default_factory=dict)
    model_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------
# Pomocnicze: preprocessing - ROZBUDOWANE
# ---------------------------
def _create_advanced_preprocessor(
    X: pd.DataFrame, 
    config: ModelConfig
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Tworzy zaawansowany preprocessor z opcjami skalowania i imputacji."""
    
    # Identyfikacja kolumn
    num_cols = X.select_dtypes(include=["number", "float", "int"]).columns.tolist()
    cat_cols = _safe_categorical_columns(X, exclude_cols=num_cols)
    
    transformers = []
    
    # Numerical preprocessing - ROZBUDOWANY
    if num_cols:
        num_steps = []
        
        # Imputacja
        if config.imputer_type == "knn":
            try:
                num_steps.append(("imputer", KNNImputer(n_neighbors=5)))
            except:
                num_steps.append(("imputer", SimpleImputer(strategy="median")))
        else:
            num_steps.append(("imputer", SimpleImputer(strategy="median")))
        
        # Skalowanie
        if config.scaler_type == "robust":
            num_steps.append(("scaler", RobustScaler()))
        elif config.scaler_type == "power":
            try:
                num_steps.append(("scaler", PowerTransformer(method='yeo-johnson')))
            except:
                num_steps.append(("scaler", StandardScaler()))
        elif config.scaler_type == "standard":
            num_steps.append(("scaler", StandardScaler()))
        # "none" - brak skalowania
        
        num_pipeline = Pipeline(num_steps)
        transformers.append(("num", num_pipeline, num_cols))
    
    # Categorical preprocessing - bez zmian
    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", _create_ohe_compatible())
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))
    
    preprocessor = ColumnTransformer(
        transformers=transformers, 
        remainder="drop", 
        n_jobs=None
    )
    
    return preprocessor, num_cols, cat_cols


def _create_feature_selector(config: ModelConfig, problem_type: str) -> Optional[Any]:
    """Tworzy selektor cech jeśli włączony."""
    if not config.feature_selection:
        return None
    
    if config.selection_method == "f_test":
        if problem_type == "regression":
            return SelectKBest(score_func=f_regression, k=config.selection_k)
        else:
            return SelectKBest(score_func=f_classif, k=config.selection_k)
    
    elif config.selection_method == "pca":
        return PCA(n_components=min(config.selection_k, 50))
    
    # RFE będzie dodane dynamicznie po utworzeniu modelu
    return None


def _handle_class_imbalance(X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """Obsługuje niebalans klas jeśli włączony."""
    if not config.handle_imbalance or not IMBLEARN:
        return X, y
    
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.combine import SMOTETomek
        
        # Sprawdź czy to klasyfikacja
        problem_type = infer_problem_type(pd.concat([X, y], axis=1), y.name)
        if problem_type.lower() != "classification":
            return X, y
        
        # Sprawdź czy jest niebalans
        value_counts = y.value_counts()
        if len(value_counts) < 2:
            return X, y
        
        imbalance_ratio = value_counts.max() / value_counts.min()
        if imbalance_ratio <= 2:
            return X, y
        
        # Zastosuj SMOTE
        smote = SMOTE(random_state=config.random_state, k_neighbors=min(5, value_counts.min()-1))
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        return X_resampled, y_resampled
        
    except Exception as e:
        print(f"[ML] Błąd balansowania klas: {e}")
        return X, y


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
    """Wybiera kolumny kategoryczne bezpiecznie."""
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
# Bezpieczna stratyfikacja
# ---------------------------
def _safe_stratify_check(y: pd.Series, min_samples_per_class: int = 2) -> bool:
    """Sprawdza czy stratyfikacja jest możliwa."""
    try:
        vc = y.value_counts(dropna=True)
        if len(vc) < 2:
            return False
        return (vc >= min_samples_per_class).all()
    except Exception:
        return False


# ---------------------------
# Wybór i konstrukcja modelu - ROZBUDOWANY
# ---------------------------
def _select_engine(engine: EngineName, problem_type: str, config: ModelConfig) -> str:
    """Inteligentny wybór silnika na podstawie danych i konfiguracji."""
    if engine != "auto":
        return engine
    
    # Heurystyka wyboru na podstawie rozmiaru danych i opcji
    if config.ensemble_methods:
        # Dla ensembles preferuj sklearn
        return "sklearn"
    elif config.hyperparameter_tuning:
        # Dla tuning preferuj szybkie silniki
        if LGBM is not None:
            return "lightgbm"
        elif XGB is not None:
            return "xgboost"
    
    # Standardowa heurystyka
    if LGBM is not None:
        return "lightgbm"
    if XGB is not None:
        return "xgboost"
    if CATB is not None:
        return "catboost"
    return "sklearn"


def _build_base_model(engine: str, problem_type: str, random_state: int) -> Any:
    """Buduje podstawowy model bez tuningu."""
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
                random_state=random_state, n_estimators=400, tree_method="hist", n_jobs=1,
                use_label_encoder=False, eval_metric="logloss"
            )

    if engine == "catboost" and CATB is not None:
        if problem_type == "regression":
            return CATB.CatBoostRegressor(random_state=random_state, iterations=300, verbose=False)
        else:
            return CATB.CatBoostClassifier(random_state=random_state, iterations=300, verbose=False)

    # sklearn models
    if problem_type == "regression":
        return HistGradientBoostingRegressor(random_state=random_state)
    else:
        return HistGradientBoostingClassifier(random_state=random_state)


def _build_ensemble_model(base_model: Any, problem_type: str, config: ModelConfig) -> Any:
    """Buduje model ensemble jeśli włączony."""
    if not config.ensemble_methods:
        return base_model
    
    try:
        # Różne modele do ensemble
        models = []
        
        # Model podstawowy
        models.append(("main", base_model))
        
        # Dodatkowe modele
        if problem_type == "regression":
            models.append(("rf", RandomForestRegressor(n_estimators=100, random_state=config.random_state, n_jobs=1)))
            models.append(("ridge", Ridge(random_state=config.random_state)))
            
            if len(models) >= 2:
                if config.ensemble_type == "stacking":
                    return StackingRegressor(
                        estimators=models,
                        final_estimator=Ridge(),
                        cv=3
                    )
                else:  # voting
                    return VotingRegressor(estimators=models)
        else:
            models.append(("rf", RandomForestClassifier(n_estimators=100, random_state=config.random_state, n_jobs=1)))
            models.append(("lr", LogisticRegression(random_state=config.random_state, max_iter=1000)))
            
            if len(models) >= 2:
                if config.ensemble_type == "stacking":
                    return StackingClassifier(
                        estimators=models,
                        final_estimator=LogisticRegression(max_iter=1000),
                        cv=3
                    )
                else:  # voting
                    return VotingClassifier(estimators=models, voting='soft')
        
        return base_model
        
    except Exception as e:
        print(f"[ML] Błąd tworzenia ensemble: {e}")
        return base_model


def _tune_hyperparameters(model: Any, X: pd.DataFrame, y: pd.Series, config: ModelConfig, problem_type: str) -> Tuple[Any, Dict[str, Any]]:
    """Optymalizacja hiperparametrów."""
    if not config.hyperparameter_tuning:
        return model, {}
    
    try:
        # Wybierz strategie na podstawie typu modelu
        param_distributions = _get_param_distributions(model, problem_type)
        
        if not param_distributions:
            return model, {}
        
        # Wybierz metodę optymalizacji
        if len(X) > 10000:
            # RandomizedSearch dla większych zbiorów
            search = RandomizedSearchCV(
                model,
                param_distributions,
                n_iter=20,
                cv=min(3, config.cv_folds),
                scoring=_get_scoring_metric(problem_type),
                random_state=config.random_state,
                n_jobs=1
            )
        else:
            # GridSearch dla mniejszych zbiorów
            search = GridSearchCV(
                model,
                param_distributions,
                cv=min(3, config.cv_folds),
                scoring=_get_scoring_metric(problem_type),
                n_jobs=1
            )
        
        # Fit z timeout
        search.fit(X, y)
        
        return search.best_estimator_, search.best_params_
        
    except Exception as e:
        print(f"[ML] Błąd tuningu hiperparametrów: {e}")
        return model, {}


def _get_param_distributions(model: Any, problem_type: str) -> Dict[str, Any]:
    """Zwraca rozkłady parametrów dla różnych modeli."""
    model_name = type(model).__name__.lower()
    
    if "lgbm" in model_name or "lightgbm" in model_name:
        return {
            'num_leaves': [20, 31, 50, 100],
            'learning_rate': [0.05, 0.1, 0.15],
            'feature_fraction': [0.8, 0.9, 1.0],
            'bagging_fraction': [0.8, 0.9, 1.0],
            'min_data_in_leaf': [5, 10, 20]
        }
    
    elif "xgb" in model_name or "xgboost" in model_name:
        return {
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        }
    
    elif "catboost" in model_name:
        return {
            'depth': [4, 5, 6],
            'learning_rate': [0.05, 0.1, 0.15],
            'l2_leaf_reg': [1, 3, 5, 7, 9]
        }
    
    elif "histgradientboosting" in model_name:
        return {
            'max_iter': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7],
            'min_samples_leaf': [10, 20, 30]
        }
    
    elif "randomforest" in model_name:
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    return {}


def _get_scoring_metric(problem_type: str) -> str:
    """Zwraca metrykę scoringu dla optymalizacji."""
    if problem_type == "regression":
        return "r2"
    else:
        return "f1_macro"


# ---------------------------
# Metryki - ROZBUDOWANE
# ---------------------------
def _compute_comprehensive_metrics(
    problem_type: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    pipe: Pipeline,
    X_test: pd.DataFrame,
    config: ModelConfig
) -> Dict[str, Any]:
    """Oblicza komprehensywny zestaw metryk."""
    metrics: Dict[str, Any] = {}

    if problem_type == "regression":
        # Podstawowe metryki regresji
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        
        # Dodatkowe metryki regresji
        try:
            mape = float(mean_absolute_percentage_error(y_true, y_pred))
            explained_var = float(explained_variance_score(y_true, y_pred))
            max_err = float(max_error(y_true, y_pred))
            
            metrics.update({
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "mape": mape,
                "explained_variance": explained_var,
                "max_error": max_err
            })
        except Exception:
            metrics.update({"mae": mae, "rmse": rmse, "r2": r2})
        
        # Dodatkowe statystyki
        residuals = y_true - y_pred
        metrics.update({
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "residual_skewness": float(pd.Series(residuals).skew()) if len(residuals) > 3 else 0.0
        })

    else:  # classification
        # Podstawowe metryki klasyfikacji
        acc = float(accuracy_score(y_true, y_pred))
        
        try:
            f1_macro = float(f1_score(y_true, y_pred, average="macro"))
            f1_micro = float(f1_score(y_true, y_pred, average="micro"))
            f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))
            
            precision_macro = float(precision_score(y_true, y_pred, average="macro"))
            recall_macro = float(recall_score(y_true, y_pred, average="macro"))
            
            metrics.update({
                "accuracy": acc,
                "f1_macro": f1_macro,
                "f1_micro": f1_micro,
                "f1_weighted": f1_weighted,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro
            })
        except Exception:
            metrics.update({"accuracy": acc})

        # AUC scores
        try:
            if config.enable_probabilities and hasattr(pipe["model"], "predict_proba"):
                proba = pipe.predict_proba(X_test)
                if proba is not None:
                    n_classes = len(np.unique(y_true))
                    
                    if n_classes == 2:
                        # Binary classification
                        pos_proba = proba[:, 1] if proba.ndim > 1 else proba
                        auc = float(roc_auc_score(y_true, pos_proba))
                        metrics["roc_auc"] = auc
                    elif n_classes > 2:
                        # Multi-class classification
                        try:
                            auc_ovr = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
                            auc_ovo = float(roc_auc_score(y_true, proba, multi_class="ovo", average="macro"))
                            metrics["roc_auc_ovr_macro"] = auc_ovr
                            metrics["roc_auc_ovo_macro"] = auc_ovo
                        except Exception:
                            pass
        except Exception:
            pass

    return metrics


def _cross_validate_model(model: Any, X: pd.DataFrame, y: pd.Series, config: ModelConfig, problem_type: str) -> Dict[str, List[float]]:
    """Walidacja krzyżowa z wieloma metrykami."""
    from sklearn.model_selection import cross_validate
    
    # Wybierz metryki do walidacji krzyżowej
    if problem_type == "regression":
        scoring = ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
    else:
        scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    
    try:
        # Walidacja krzyżowa
        cv_results = cross_validate(
            model, X, y,
            cv=config.cv_folds,
            scoring=scoring,
            return_train_score=True,
            n_jobs=1
        )
        
        # Przetwórz wyniki
        cv_scores = {}
        for metric in scoring:
            test_key = f"test_{metric}"
            train_key = f"train_{metric}"
            
            if test_key in cv_results:
                cv_scores[f"{metric}_test"] = cv_results[test_key].tolist()
            if train_key in cv_results:
                cv_scores[f"{metric}_train"] = cv_results[train_key].tolist()
        
        return cv_scores
        
    except Exception as e:
        print(f"[ML] Błąd walidacji krzyżowej: {e}")
        return {}


# ---------------------------
# Feature importance - ROZBUDOWANE
# ---------------------------
def _extract_comprehensive_feature_importance(
    pipe: Pipeline,
    preprocessor: ColumnTransformer,
    num_cols: List[str],
    cat_cols: List[str],
    config: ModelConfig
) -> Optional[pd.DataFrame]:
    """Rozbudowane wyciąganie ważności cech z różnych modeli."""
    try:
        model = pipe.named_steps["model"]
    except Exception:
        return None

    # Nazwy cech po preprocessing
    try:
        feature_names = _get_feature_names_out(preprocessor, num_cols, cat_cols)
    except:
        feature_names = num_cols + cat_cols

    importances: Optional[np.ndarray] = None

    # 1. Tree-based models (najlepsze)
    if hasattr(model, "feature_importances_"):
        importances = getattr(model, "feature_importances_", None)
    
    # 2. Linear models (coefficients)
    elif hasattr(model, "coef_"):
        coef = getattr(model, "coef_")
        if isinstance(coef, np.ndarray):
            if coef.ndim > 1:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
    
    # 3. Ensemble models
    elif hasattr(model, "estimators_"):
        try:
            # Dla VotingClassifier/Regressor
            if hasattr(model, "estimators_") and hasattr(model, "named_estimators_"):
                ensemble_importances = []
                for name, estimator in model.named_estimators_.items():
                    if hasattr(estimator, "feature_importances_"):
                        ensemble_importances.append(estimator.feature_importances_)
                    elif hasattr(estimator, "coef_"):
                        coef = estimator.coef_
                        if coef.ndim > 1:
                            coef = np.mean(np.abs(coef), axis=0)
                        ensemble_importances.append(np.abs(coef))
                
                if ensemble_importances:
                    importances = np.mean(ensemble_importances, axis=0)
        except:
            pass
    
    # 4. Permutation importance jako fallback (kosztowne)
    if importances is None and config.feature_selection:
        try:
            from sklearn.inspection import permutation_importance
            # Tylko dla mniejszych zbiorów danych
            if len(pipe.named_steps["prep"].transform(pipe[:-1].transform(feature_names[:100]))) < 5000:
                X_sample = pipe[:-1].transform(pd.DataFrame(dict(zip(feature_names, np.random.randn(100, len(feature_names))))))
                perm_importance = permutation_importance(
                    model, X_sample, np.random.randn(100), 
                    n_repeats=5, random_state=42
                )
                importances = perm_importance.importances_mean
        except:
            pass

    if importances is None:
        return None

    # Przygotuj DataFrame
    importances = np.ravel(importances)
    n_features = min(len(importances), len(feature_names))
    
    if n_features == 0:
        return None
    
    df_imp = pd.DataFrame({
        "feature": feature_names[:n_features],
        "importance": importances[:n_features]
    })
    
    # Znormalizuj ważności
    if df_imp['importance'].sum() > 0:
        df_imp['importance'] = df_imp['importance'] / df_imp['importance'].sum()
    
    # Sortuj i dodaj rangi
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    df_imp['rank'] = range(1, len(df_imp) + 1)
    df_imp['cumulative_importance'] = df_imp['importance'].cumsum()
    
    return df_imp


def _get_feature_names_out(preprocessor: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    """Pobiera nazwy cech po preprocessing."""
    feature_names = []
    
    # Numerical features
    if num_cols:
        feature_names.extend(num_cols)
    
    # Categorical features (po OHE)
    if cat_cols:
        try:
            if hasattr(preprocessor, 'named_transformers_') and "cat" in preprocessor.named_transformers_:
                cat_transformer = preprocessor.named_transformers_["cat"]
                if hasattr(cat_transformer, "named_steps") and "ohe" in cat_transformer.named_steps:
                    ohe = cat_transformer.named_steps["ohe"]
                    if hasattr(ohe, "get_feature_names_out"):
                        cat_feature_names = ohe.get_feature_names_out(cat_cols)
                        feature_names.extend(cat_feature_names.tolist())
                    else:
                        # Fallback
                        feature_names.extend([f"{col}_encoded" for col in cat_cols])
                else:
                    feature_names.extend(cat_cols)
            else:
                feature_names.extend(cat_cols)
        except Exception:
            feature_names.extend(cat_cols)
    
    return feature_names


# ---------------------------
# Główna funkcja treningu - ROZBUDOWANA
# ---------------------------
def train_model_comprehensive(
    df: pd.DataFrame,
    cfg: ModelConfig,
    use_advanced: bool = True,
) -> TrainingResult:
    """
    ROZBUDOWANA funkcja treningu z zaawansowanymi opcjami:
    - Feature engineering i selekcja cech
    - Balansowanie klas
    - Hyperparameter tuning
    - Ensemble methods
    - Rozbudowane metryki
    - Cross-validation
    """
    seed_everything(cfg.random_state)
    start_time = time.time()

    if cfg.target not in df.columns:
        raise ValueError(f"Brak kolumny celu '{cfg.target}' w danych.")

    # Przygotowanie danych
    y = df[cfg.target]
    X = df.drop(columns=[cfg.target])

    if y.isna().any():
        raise ValueError("Kolumna celu zawiera wartości puste (NaN). Uzupełnij lub odfiltruj przed treningiem.")

    # Określ typ problemu
    problem_type = infer_problem_type(df, cfg.target).lower()
    if problem_type not in ("regression", "classification"):
        if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) >= 3:
            problem_type = "regression"
        else:
            problem_type = "classification"

    # Feature engineering (podstawowy)
    if cfg.feature_engineering:
        X = _apply_feature_engineering(X)

    # Bezpieczna stratyfikacja
    can_stratify = False
    stratify_param = None
    if problem_type == "classification" and cfg.stratify:
        can_stratify = _safe_stratify_check(y, min_samples_per_class=2)
        stratify_param = y if can_stratify else None

    # Podział danych
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

    # Balansowanie klas
    if cfg.handle_imbalance:
        X_train, y_train = _handle_class_imbalance(X_train, y_train, cfg)

    # Preprocessor
    preprocessor, num_cols, cat_cols = _create_advanced_preprocessor(X_train, cfg)

    # Feature selector
    feature_selector = _create_feature_selector(cfg, problem_type)

    # Wybór silnika i budowa modelu
    selected_engine = _select_engine(cfg.engine, problem_type, cfg)
    base_model = _build_base_model(selected_engine, problem_type, cfg.random_state)
    model = _build_ensemble_model(base_model, problem_type, cfg)

    # Budowa pipeline
    pipeline_steps = [("prep", preprocessor)]
    
    if feature_selector:
        pipeline_steps.append(("selector", feature_selector))
    
    pipeline_steps.append(("model", model))
    
    pipe = Pipeline(pipeline_steps)

    # Hyperparameter tuning
    best_params = {}
    if cfg.hyperparameter_tuning:
        pipe, best_params = _tune_hyperparameters(pipe, X_train, y_train, cfg, problem_type)

    # Trening modelu
    pipe.fit(X_train, y_train)

    # Predykcje
    y_pred = pipe.predict(X_test)

    # Cross-validation
    cv_scores = _cross_validate_model(pipe, X_train, y_train, cfg, problem_type)

    # Metryki
    metrics = _compute_comprehensive_metrics(problem_type, y_test, y_pred, pipe, X_test, cfg)

    # Feature importance
    fi_df = _extract_comprehensive_feature_importance(pipe, preprocessor, num_cols, cat_cols, cfg)

    # Validation info
    validation_info = _build_validation_info(problem_type, y_test, y_pred, pipe, X_test, cfg)

    # Ostrzeżenia i notatki
    warnings, notes = _generate_training_warnings_and_notes(
        df, cfg, problem_type, metrics, can_stratify, selected_engine, best_params
    )

    # Metadane
    training_time = time.time() - start_time
    metadata = {
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
        "notes": notes,
        "stratified": can_stratify,
        "class_distribution": y.value_counts().to_dict() if problem_type == "classification" else None,
        "training_time_seconds": training_time,
        "config_used": _config_to_dict(cfg),
    }

    # Zwróć rozbudowany wynik
    return TrainingResult(
        model=pipe,
        metrics=metrics,
        feature_importance=fi_df if fi_df is not None else pd.DataFrame(columns=["feature", "importance"]),
        metadata=metadata,
        cross_val_scores=cv_scores,
        best_params=best_params
    )


def _apply_feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """Podstawowa inżynieria cech."""
    X_eng = X.copy()
    
    try:
        # Interakcje między numerycznymi cechami (top 5)
        numeric_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()[:5]
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Mnożenie
                X_eng[f'{col1}_x_{col2}'] = X_eng[col1] * X_eng[col2]
                # Dzielenie (bezpieczne)
                X_eng[f'{col1}_div_{col2}'] = X_eng[col1] / (X_eng[col2] + 1e-8)
        
        # Transformacje numeryczne
        for col in numeric_cols[:3]:
            # Logarytm (dla dodatnich wartości)
            if (X_eng[col] > 0).all():
                X_eng[f'{col}_log'] = np.log1p(X_eng[col])
            
            # Kwadrat
            X_eng[f'{col}_squared'] = X_eng[col] ** 2
        
        # Binning (dyskretyzacja) dla numerycznych
        for col in numeric_cols[:3]:
            try:
                X_eng[f'{col}_binned'] = pd.qcut(X_eng[col], q=5, labels=False, duplicates='drop')
            except:
                pass
        
    except Exception as e:
        print(f"[ML] Błąd feature engineering: {e}")
    
    return X_eng


def _build_validation_info(
    problem_type: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    pipe: Pipeline,
    X_test: pd.DataFrame,
    cfg: ModelConfig
) -> Dict[str, Any]:
    """Buduje szczegółowe info walidacyjne."""
    out: Dict[str, Any] = {}
    MAX_SAMPLES = 10000  # Zwiększony limit

    try:
        yt = np.asarray(y_true)[:MAX_SAMPLES]
        yp = np.asarray(y_pred)[:MAX_SAMPLES]
    except Exception:
        yt, yp = None, None

    if yt is not None and yp is not None:
        out["y_true"] = yt.tolist()
        out["y_pred"] = yp.tolist()

    # Confusion matrix dla klasyfikacji
    if problem_type == "classification" and yt is not None and yp is not None:
        try:
            labels = sorted(list(set(list(yt) + list(yp))))
            cm = confusion_matrix(yt, yp, labels=labels)
            out["labels"] = [str(l) for l in labels]
            out["confusion_matrix"] = cm.astype(int).tolist()
            
            # Detailed classification report
            try:
                class_report = classification_report(yt, yp, output_dict=True)
                out["classification_report"] = class_report
            except:
                pass
                
        except Exception:
            pass

    # Residuals dla regresji
    if problem_type == "regression" and yt is not None and yp is not None:
        try:
            residuals = yt - yp
            out["residuals"] = residuals.tolist()
            out["residual_stats"] = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals))
            }
        except:
            pass

    # Probabilitities dla klasyfikacji
    if problem_type == "classification" and cfg.enable_probabilities:
        try:
            if hasattr(pipe["model"], "predict_proba"):
                proba = pipe.predict_proba(X_test.head(MAX_SAMPLES))
                if proba is not None:
                    out["prediction_probabilities"] = proba.tolist()
        except:
            pass

    return out


def _generate_training_warnings_and_notes(
    df: pd.DataFrame,
    cfg: ModelConfig,
    problem_type: str,
    metrics: Dict[str, Any],
    can_stratify: bool,
    selected_engine: str,
    best_params: Dict[str, Any]
) -> Tuple[List[str], List[str]]:
    """Generuje ostrzeżenia i notatki o treningu."""
    warnings: List[str] = []
    notes: List[str] = []

    # Ostrzeżenia
    if problem_type == "classification" and not can_stratify:
        warnings.append("Nie można użyć stratyfikacji — niektóre klasy mają za mało przykładów.")

    if problem_type == "classification":
        y = df[cfg.target]
        vc = y.value_counts()
        if len(vc) > 1 and vc.min() > 0:
            imbalance_ratio = vc.max() / vc.min()
            if imbalance_ratio > 10:
                warnings.append(f"Silny niebalans klas (ratio: {imbalance_ratio:.1f}:1)")
            elif imbalance_ratio > 3:
                warnings.append(f"Niebalans klas (ratio: {imbalance_ratio:.1f}:1)")

    # Performance warnings
    if problem_type == "regression":
        r2 = metrics.get("r2", 0)
        if r2 < 0.5:
            warnings.append("Niska jakość modelu regresji (R² < 0.5)")
    else:
        accuracy = metrics.get("accuracy", 0)
        if accuracy < 0.7:
            warnings.append("Niska dokładność modelu klasyfikacji (< 70%)")

    # Notatki
    notes.append(f"Użyto silnika: {selected_engine}")
    
    if cfg.feature_engineering:
        notes.append("Zastosowano podstawową inżynierię cech")
    
    if cfg.feature_selection:
        notes.append(f"Zastosowano selekcję cech (metoda: {cfg.selection_method})")
    
    if cfg.handle_imbalance:
        notes.append("Zastosowano balansowanie klas")
    
    if cfg.hyperparameter_tuning and best_params:
        notes.append(f"Zoptymalizowano {len(best_params)} hiperparametrów")
    
    if cfg.ensemble_methods:
        notes.append(f"Użyto metod zespołowych ({cfg.ensemble_type})")

    return warnings, notes


def _config_to_dict(cfg: ModelConfig) -> Dict[str, Any]:
    """Konwertuje konfigurację do słownika."""
    return {
        "target": cfg.target,
        "engine": cfg.engine,
        "test_size": cfg.test_size,
        "cv_folds": cfg.cv_folds,
        "random_state": cfg.random_state,
        "stratify": cfg.stratify,
        "enable_probabilities": cfg.enable_probabilities,
        "feature_engineering": cfg.feature_engineering,
        "feature_selection": cfg.feature_selection,
        "handle_imbalance": cfg.handle_imbalance,
        "hyperparameter_tuning": cfg.hyperparameter_tuning,
        "early_stopping": cfg.early_stopping,
        "ensemble_methods": cfg.ensemble_methods,
        "scaler_type": cfg.scaler_type,
        "imputer_type": cfg.imputer_type,
        "selection_method": cfg.selection_method,
        "ensemble_type": cfg.ensemble_type
    }


# ---------------------------
# Zapis/odczyt artefaktów - BEZ ZMIAN
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