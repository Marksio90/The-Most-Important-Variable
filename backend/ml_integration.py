# backend/ml_integration.py — Refactored with clear separation of responsibilities
from __future__ import annotations
import json
import math
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error,
    mean_squared_error, precision_recall_curve, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, RandomizedSearchCV, 
    StratifiedKFold, KFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, RobustScaler, 
    LabelEncoder, PowerTransformer
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression, 
    SelectKBest, f_classif, f_regression
)

# Dodaj po istniejących importach
try:
    import pycaret
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare, finalize_model as reg_finalize
    from pycaret.classification import setup as clf_setup, compare_models as clf_compare, finalize_model as clf_finalize
    from pycaret.regression import pull as reg_pull
    from pycaret.classification import pull as clf_pull
    HAVE_PYCARET = True
except ImportError:
    HAVE_PYCARET = False

# Dodaj nową klasę PyCaret trainer
class PyCaretTrainer(BaseModelTrainer):
    """Trainer using PyCaret for automated ML."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def create_model(self, problem_type: str, **kwargs) -> BaseEstimator:
        """PyCaret doesn't use single model creation - handled in train method."""
        pass
    
    def get_param_grid(self) -> Dict[str, Any]:
        """PyCaret handles hyperparameter tuning internally."""
        return {}
    
    def train_with_pycaret(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> Tuple[Any, Dict[str, float], pd.DataFrame]:
        """Train model using PyCaret's automated pipeline."""
        
        # Combine X and y for PyCaret
        df_combined = X.copy()
        df_combined['target'] = y
        
        try:
            if problem_type == "classification":
                # Setup PyCaret classification
                clf_exp = clf_setup(
                    data=df_combined,
                    target='target',
                    session_id=self.random_state,
                    train_size=0.8,
                    silent=True,
                    verbose=False
                )
                
                # Compare models and get best
                best_models = clf_compare(
                    include=['rf', 'lr', 'xgboost', 'lightgbm', 'dt'],
                    sort='Accuracy',
                    n_select=1,
                    verbose=False
                )
                
                # Get metrics
                results_df = clf_pull()
                best_model = clf_finalize(best_models)
                
                # Extract metrics
                metrics = {
                    'accuracy': float(results_df['Accuracy'].iloc[0]),
                    'auc': float(results_df.get('AUC', [0]).iloc[0]),
                    'recall': float(results_df.get('Recall', [0]).iloc[0]),
                    'precision': float(results_df.get('Prec.', [0]).iloc[0]),
                    'f1': float(results_df.get('F1', [0]).iloc[0])
                }
                
            else:  # regression
                # Setup PyCaret regression
                reg_exp = reg_setup(
                    data=df_combined,
                    target='target',
                    session_id=self.random_state,
                    train_size=0.8,
                    silent=True,
                    verbose=False
                )
                
                # Compare models and get best
                best_models = reg_compare(
                    include=['rf', 'lr', 'xgboost', 'lightgbm', 'dt'],
                    sort='R2',
                    n_select=1,
                    verbose=False
                )
                
                # Get metrics
                results_df = reg_pull()
                best_model = reg_finalize(best_models)
                
                # Extract metrics
                metrics = {
                    'r2': float(results_df['R2'].iloc[0]),
                    'mae': float(results_df['MAE'].iloc[0]),
                    'mse': float(results_df['MSE'].iloc[0]),
                    'rmse': float(results_df['RMSE'].iloc[0]),
                    'mape': float(results_df.get('MAPE', [0]).iloc[0])
                }
            
            # Extract feature importance
            try:
                if hasattr(best_model, 'feature_importances_'):
                    importance_values = best_model.feature_importances_
                elif hasattr(best_model, 'coef_'):
                    importance_values = np.abs(best_model.coef_).flatten()
                else:
                    importance_values = np.ones(len(X.columns))  # Fallback
                
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importance_values
                }).sort_values('importance', ascending=False)
                
            except Exception:
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': np.ones(len(X.columns))
                }).sort_values('importance', ascending=False)
            
            return best_model, metrics, feature_importance
            
        except Exception as e:
            raise RuntimeError(f"PyCaret training failed: {str(e)}")

# Optional engines with fallbacks
HAVE_LGBM = HAVE_XGB = HAVE_CAT = HAVE_OPTUNA = False
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAVE_LGBM = True
except ImportError:
    pass
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAVE_XGB = True
except ImportError:
    pass
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAVE_CAT = True
except ImportError:
    pass
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAVE_OPTUNA = True
except ImportError:
    pass

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class ModelConfig:
    """Configuration for model training with sensible defaults."""
    target: str
    problem_type: Optional[str] = None
    engine: str = "auto"
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    feature_selection_k: int = 150
    hyperopt_trials: int = 100
    early_stopping_rounds: int = 100
    imbalance_threshold: float = 0.1
    outlier_detection: bool = True
    feature_engineering: bool = True
    use_optuna: bool = True

@dataclass
class TrainingResult:
    """Complete result of model training."""
    model: BaseEstimator
    metrics: Dict[str, Any]
    feature_importance: pd.DataFrame
    metadata: Dict[str, Any]
    preprocessing_info: Dict[str, Any]
    training_time: float
    validation_scores: Optional[List[float]] = None

# ============================================================================
# PROBLEM TYPE DETECTION
# ============================================================================

class ProblemTypeDetector:
    """Detects ML problem type from target variable."""
    
    @staticmethod
    def detect_problem_type(target_series: pd.Series) -> str:
        """Detects problem type with improved heuristics."""
        if target_series.empty:
            return "regression"
        
        # Clean the series
        clean_series = target_series.dropna()
        unique_values = clean_series.nunique()
        total_samples = len(clean_series)
        
        # Check data types
        is_numeric = pd.api.types.is_numeric_dtype(clean_series)
        is_bool = pd.api.types.is_bool_dtype(clean_series)
        is_categorical = clean_series.dtype.name in ['object', 'category']
        
        # Clear classification cases
        if is_bool or is_categorical:
            return "classification"
        
        # Binary numeric (0/1, 1/2, etc.)
        if unique_values == 2:
            return "classification"
        
        # Few unique values relative to sample size
        if unique_values <= 20 and unique_values / total_samples < 0.05:
            return "classification"
        
        # Integer with reasonable number of classes
        if is_numeric and pd.api.types.is_integer_dtype(clean_series):
            if unique_values <= min(50, total_samples * 0.1):
                return "classification"
        
        # Default to regression for continuous numeric data
        return "regression"

# ============================================================================
# DATA VALIDATION
# ============================================================================

class DataValidator:
    """Validates input data quality and compatibility."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Comprehensive data validation."""
        issues = []
        recommendations = []
        warnings = []
        
        # Basic checks
        if df.empty:
            issues.append("DataFrame is empty")
            return {
                "valid": False, 
                "issues": issues, 
                "recommendations": recommendations,
                "warnings": warnings
            }
        
        if target not in df.columns:
            issues.append(f"Target column '{target}' not found")
            return {
                "valid": False, 
                "issues": issues, 
                "recommendations": recommendations,
                "warnings": warnings
            }
        
        # Target validation
        y = df[target]
        target_missing = y.isna().sum()
        if target_missing > len(df) * 0.5:
            issues.append(f"Target has {target_missing} missing values (>{len(df)*0.5:.0f} threshold)")
        elif target_missing > 0:
            warnings.append(f"Target has {target_missing} missing values")
        
        if y.nunique(dropna=True) < 2:
            issues.append("Target has less than 2 unique values")
        
        # Features validation
        X = df.drop(columns=[target])
        if X.empty:
            issues.append("No feature columns available")
        
        # Missing values analysis
        missing_pct = (X.isna().sum() / len(X)) * 100
        high_missing = missing_pct[missing_pct > 80]
        if not high_missing.empty:
            recommendations.append(f"Consider removing features with >80% missing: {list(high_missing.index)}")
        
        # Constant features
        constant_features = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]
        if constant_features:
            recommendations.append(f"Constant features to remove: {constant_features}")
        
        # High cardinality check
        high_card_features = []
        for col in X.select_dtypes(include=['object']).columns:
            if X[col].nunique() > len(X) * 0.8:
                high_card_features.append(col)
        if high_card_features:
            warnings.append(f"High cardinality features: {high_card_features}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
            "warnings": warnings,
            "n_rows": len(df),
            "n_features": len(X.columns),
            "target_info": {
                "nunique": y.nunique(),
                "missing_count": target_missing,
                "dtype": str(y.dtype)
            }
        }

# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

class PreprocessingStep(ABC):
    """Abstract base class for preprocessing steps."""
    
    @abstractmethod
    def transform(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Transform the dataframe and return info about changes."""
        pass

class RemoveConstantFeaturesStep(PreprocessingStep):
    """Removes constant and quasi-constant features."""
    
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
    
    def transform(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        X = df.drop(columns=[target], errors='ignore')
        constant_features = []
        
        for col in X.columns:
            if X[col].nunique(dropna=True) <= 1:
                constant_features.append(col)
            elif X[col].dtype in ['int64', 'float64'] and X[col].std() < self.threshold:
                constant_features.append(col)
        
        if constant_features:
            df = df.drop(columns=constant_features)
        
        return df, {"removed_constant_features": constant_features}

class HandleMissingValuesStep(PreprocessingStep):
    """Handles missing values with different strategies."""
    
    def __init__(self, numeric_strategy: str = "median", categorical_strategy: str = "most_frequent"):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
    
    def transform(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        info = {"imputed_columns": []}
        
        # Handle missing values in features only
        features = [col for col in df.columns if col != target]
        
        for col in features:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    if self.numeric_strategy == "median":
                        fill_value = df[col].median()
                    elif self.numeric_strategy == "mean":
                        fill_value = df[col].mean()
                    else:
                        fill_value = 0
                    df[col] = df[col].fillna(fill_value)
                else:
                    if self.categorical_strategy == "most_frequent":
                        fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else "MISSING"
                    else:
                        fill_value = "MISSING"
                    df[col] = df[col].fillna(fill_value)
                
                info["imputed_columns"].append(col)
        
        # Handle target missing values by removal
        if target in df.columns:
            initial_rows = len(df)
            df = df.dropna(subset=[target])
            info["dropped_target_missing"] = initial_rows - len(df)
        
        return df, info

class CreateDateFeaturesStep(PreprocessingStep):
    """Creates features from datetime columns."""
    
    def __init__(self, date_features: List[str] = None):
        self.date_features = date_features or ["year", "month", "dayofweek", "is_weekend"]
    
    def transform(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        created_features = []
        
        for col in df.columns:
            if col == target:
                continue
                
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self._create_date_features(df, col, created_features)
            elif df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    if parsed.notna().mean() > 0.8:  # If most values are parseable
                        df[col] = parsed
                        self._create_date_features(df, col, created_features)
                except Exception:
                    continue
        
        return df, {"created_date_features": created_features}
    
    def _create_date_features(self, df: pd.DataFrame, col: str, created_features: List[str]):
        """Helper to create date features."""
        base_name = col.replace('_date', '').replace('_time', '')
        
        if 'year' in self.date_features:
            new_col = f"{base_name}__year"
            df[new_col] = df[col].dt.year
            created_features.append(new_col)
        
        if 'month' in self.date_features:
            new_col = f"{base_name}__month"
            df[new_col] = df[col].dt.month
            created_features.append(new_col)
        
        if 'dayofweek' in self.date_features:
            new_col = f"{base_name}__dayofweek"
            df[new_col] = df[col].dt.dayofweek
            created_features.append(new_col)
        
        if 'is_weekend' in self.date_features:
            new_col = f"{base_name}__is_weekend"
            df[new_col] = (df[col].dt.dayofweek >= 5).astype(int)
            created_features.append(new_col)

class DataPreprocessor:
    """Orchestrates preprocessing pipeline."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.steps = [
            RemoveConstantFeaturesStep(),
            HandleMissingValuesStep(),
            CreateDateFeaturesStep() if config.feature_engineering else None
        ]
        self.steps = [step for step in self.steps if step is not None]
    
    def preprocess(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply all preprocessing steps."""
        preprocessing_info = {}
        
        for i, step in enumerate(self.steps):
            df, step_info = step.transform(df, target)
            preprocessing_info[f"step_{i}_{step.__class__.__name__}"] = step_info
        
        return df, preprocessing_info

# ============================================================================
# MODEL TRAINERS
# ============================================================================

class BaseModelTrainer(ABC):
    """Abstract base class for model trainers."""
    
    @abstractmethod
    def create_model(self, problem_type: str, **kwargs) -> BaseEstimator:
        """Create and configure the model."""
        pass
    
    @abstractmethod
    def get_param_grid(self) -> Dict[str, Any]:
        """Get hyperparameter search space."""
        pass

class SklearnTrainer(BaseModelTrainer):
    """Trainer for scikit-learn models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def create_model(self, problem_type: str, **kwargs) -> BaseEstimator:
        """Create sklearn model based on problem type."""
        if problem_type == "classification":
            return HistGradientBoostingClassifier(
                random_state=self.random_state,
                **kwargs
            )
        else:
            return HistGradientBoostingRegressor(
                random_state=self.random_state,
                **kwargs
            )
    
    def get_param_grid(self) -> Dict[str, Any]:
        """Hyperparameter grid for sklearn models."""
        return {
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_iter': [100, 200]
        }

class LightGBMTrainer(BaseModelTrainer):
    """Trainer for LightGBM models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def create_model(self, problem_type: str, **kwargs) -> BaseEstimator:
        """Create LightGBM model."""
        common_params = {
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': -1
        }
        common_params.update(kwargs)
        
        if problem_type == "classification":
            return LGBMClassifier(**common_params)
        else:
            return LGBMRegressor(**common_params)
    
    def get_param_grid(self) -> Dict[str, Any]:
        return {
            'n_estimators': [100, 200, 500],
            'num_leaves': [20, 50, 100],
            'learning_rate': [0.01, 0.1, 0.2],
            'feature_fraction': [0.8, 0.9, 1.0]
        }

class XGBoostTrainer(BaseModelTrainer):
    """Trainer for XGBoost models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def create_model(self, problem_type: str, **kwargs) -> BaseEstimator:
        """Create XGBoost model."""
        common_params = {
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        common_params.update(kwargs)
        
        if problem_type == "classification":
            return XGBClassifier(**common_params)
        else:
            return XGBRegressor(**common_params)
    
    def get_param_grid(self) -> Dict[str, Any]:
        return {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }

class CatBoostTrainer(BaseModelTrainer):
    """Trainer for CatBoost models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def create_model(self, problem_type: str, **kwargs) -> BaseEstimator:
        """Create CatBoost model."""
        common_params = {
            'random_seed': self.random_state,
            'verbose': False,
            'allow_writing_files': False
        }
        common_params.update(kwargs)
        
        if problem_type == "classification":
            return CatBoostClassifier(**common_params)
        else:
            return CatBoostRegressor(**common_params)
    
    def get_param_grid(self) -> Dict[str, Any]:
        return {
            'iterations': [100, 200, 500],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
        }

# ============================================================================
# MODEL FACTORY
# ============================================================================

class ModelFactory:
    """Factory for creating appropriate model trainers."""
    
    def __init__(self):
        self.trainers = {
            'sklearn': SklearnTrainer,
            'lightgbm': LightGBMTrainer if HAVE_LGBM else SklearnTrainer,
            'xgboost': XGBoostTrainer if HAVE_XGB else SklearnTrainer,
            'catboost': CatBoostTrainer if HAVE_CAT else SklearnTrainer,
            'pycaret': PyCaretTrainer if HAVE_PYCARET else SklearnTrainer,  # DODANE
            'auto_pycaret': PyCaretTrainer if HAVE_PYCARET else SklearnTrainer,  # DODANE
        }

    def _select_auto_engine(self) -> str:
        """Automatically select best available engine - prefer PyCaret."""
        if HAVE_PYCARET:
            return 'auto_pycaret'  # ZMIENIONE: Preferuj PyCaret
        elif HAVE_LGBM:
            return 'lightgbm'
        elif HAVE_XGB:
            return 'xgboost'
        elif HAVE_CAT:
            return 'catboost'
        else:
            return 'sklearn'

# ============================================================================
# METRICS CALCULATION
# ============================================================================

class MetricsCalculator:
    """Calculates comprehensive metrics for model evaluation."""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_proba, problem_type: str) -> Dict[str, float]:
        """Calculate metrics based on problem type."""
        metrics = {}
        
        try:
            if problem_type == "classification":
                metrics.update(MetricsCalculator._calculate_classification_metrics(y_true, y_pred, y_proba))
            else:
                metrics.update(MetricsCalculator._calculate_regression_metrics(y_true, y_pred))
        except Exception as e:
            metrics["calculation_error"] = str(e)
        
        return metrics
    
    @staticmethod
    def _calculate_classification_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {}
        
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        
        if len(np.unique(y_true)) == 2 and y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
                metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
                metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
            except Exception:
                pass
        
        return metrics
    
    @staticmethod
    def _calculate_regression_metrics(y_true, y_pred) -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {}
        
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))
        
        # Additional regression metrics with error handling
        try:
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)))
            metrics["mape"] = float(mape)
        except Exception:
            pass
        
        try:
            smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
            metrics["smape"] = float(smape)
        except Exception:
            pass
        
        return metrics

# ============================================================================
# FEATURE IMPORTANCE EXTRACTION
# ============================================================================

class FeatureImportanceExtractor:
    """Extracts feature importance from trained models."""
    
    @staticmethod
    def extract_importance(model: BaseEstimator, feature_names: List[str]) -> pd.DataFrame:
        """Extract feature importance with multiple fallback methods."""
        try:
            # Method 1: feature_importances_ (tree-based models)
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                return FeatureImportanceExtractor._create_importance_df(importances, feature_names)
            
            # Method 2: coef_ (linear models)
            if hasattr(model, "coef_"):
                importances = np.abs(model.coef_).flatten()
                return FeatureImportanceExtractor._create_importance_df(importances, feature_names)
            
            # Method 3: Try to get from pipeline
            if hasattr(model, 'named_steps') and 'estimator' in model.named_steps:
                return FeatureImportanceExtractor.extract_importance(
                    model.named_steps['estimator'], feature_names
                )
            
            # Fallback: return empty dataframe
            return pd.DataFrame(columns=["feature", "importance"])
            
        except Exception:
            return pd.DataFrame(columns=["feature", "importance"])
    
    @staticmethod
    def _create_importance_df(importances: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """Create feature importance dataframe."""
        n_features = min(len(importances), len(feature_names))
        
        df = pd.DataFrame({
            "feature": feature_names[:n_features],
            "importance": importances[:n_features]
        })
        
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

# ============================================================================
# MAIN TRAINING ORCHESTRATOR
# ============================================================================

class MLTrainingOrchestrator:
    """Main class that orchestrates the entire ML training process."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.validator = DataValidator()
        self.problem_detector = ProblemTypeDetector()
        self.preprocessor = DataPreprocessor(config)
        self.model_factory = ModelFactory()
        self.metrics_calculator = MetricsCalculator()
        self.importance_extractor = FeatureImportanceExtractor()
    
    def train(self, df: pd.DataFrame) -> TrainingResult:
        """Main training pipeline."""
        start_time = time.time()
        
        try:
            # Step 1: Validation
            validation_result = self.validator.validate_dataframe(df, self.config.target)
            if not validation_result["valid"]:
                raise ValueError(f"Data validation failed: {validation_result['issues']}")
            
            # Step 2: Problem type detection
            target_series = df[self.config.target]
            problem_type = self.config.problem_type or self.problem_detector.detect_problem_type(target_series)
            
            # Step 3: Preprocessing
            df_processed, preprocessing_info = self.preprocessor.preprocess(df, self.config.target)
            
            # Step 4: Prepare data for training
            y = df_processed[self.config.target]
            X = df_processed.drop(columns=[self.config.target])
            
            # Step 5: Train/test split
            stratify = y if problem_type == "classification" and y.nunique() > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify
            )
            
            # Step 6: Create and train model
            trainer = self.model_factory.get_trainer(self.config.engine, self.config.random_state)

            # DODANE: Specjalna obsługa dla PyCaret
            if isinstance(trainer, PyCaretTrainer) and HAVE_PYCARET:
                st.write("🚀 Użycie PyCaret - porównanie wielu algorytmów...")
                
                try:
                    model, metrics, feature_importance = trainer.train_with_pycaret(
                        X_train, y_train, problem_type
                    )
                    
                    # Skip regular pipeline creation for PyCaret
                    full_pipeline = model
                    
                except Exception as e:
                    st.warning(f"PyCaret failed: {e}. Fallback to sklearn...")
                    # Fallback to regular training
                    model = trainer.create_model(problem_type)
                    preprocessor_pipeline = self._create_sklearn_preprocessor(X_train)
                    full_pipeline = Pipeline([
                        ('preprocessor', preprocessor_pipeline),
                        ('estimator', model)
                    ])
                    full_pipeline.fit(X_train, y_train)
                    
                    # Generate predictions for fallback
                    y_pred = full_pipeline.predict(X_test)
                    y_proba = None
                    
                    if problem_type == "classification" and hasattr(full_pipeline, "predict_proba"):
                        try:
                            proba_output = full_pipeline.predict_proba(X_test)
                            if proba_output.shape[1] == 2:
                                y_proba = proba_output[:, 1]
                        except Exception:
                            pass
                    
                    metrics = self.metrics_calculator.calculate_metrics(y_test, y_pred, y_proba, problem_type)
                    feature_names = list(X.columns)
                    feature_importance = self.importance_extractor.extract_importance(full_pipeline, feature_names)

            else:
                # Regular training for non-PyCaret engines
                model = trainer.create_model(problem_type)
                preprocessor_pipeline = self._create_sklearn_preprocessor(X_train)
                full_pipeline = Pipeline([
                    ('preprocessor', preprocessor_pipeline),
                    ('estimator', model)
                ])
                full_pipeline.fit(X_train, y_train)
                        
            # Step 7: Create preprocessing pipeline
            preprocessor_pipeline = self._create_sklearn_preprocessor(X_train)
            full_pipeline = Pipeline([
                ('preprocessor', preprocessor_pipeline),
                ('estimator', model)
            ])
            
            # Step 8: Train the model
            full_pipeline.fit(X_train, y_train)
            
            # Step 9: Generate predictions
            y_pred = full_pipeline.predict(X_test)
            y_proba = None
            
            if problem_type == "classification" and hasattr(full_pipeline, "predict_proba"):
                try:
                    proba_output = full_pipeline.predict_proba(X_test)
                    if proba_output.shape[1] == 2:
                        y_proba = proba_output[:, 1]
                except Exception:
                    pass
            
            # Step 10: Calculate metrics
            metrics = self.metrics_calculator.calculate_metrics(y_test, y_pred, y_proba, problem_type)
            
            # Step 11: Cross-validation
            if self.config.cv_folds > 1:
                cv_scores = self._perform_cross_validation(full_pipeline, X, y, problem_type)
                metrics.update({
                    "cv_mean": float(np.mean(cv_scores)),
                    "cv_std": float(np.std(cv_scores)),
                    "cv_scores": cv_scores.tolist()
                })
            
            # Step 12: Feature importance
            feature_names = list(X.columns)
            feature_importance = self.importance_extractor.extract_importance(full_pipeline, feature_names)
            
            # Step 13: Create metadata
            metadata = {
                "problem_type": problem_type,
                "engine": self.config.engine,
                "n_samples": len(df_processed),
                "n_features": len(X.columns),
                "validation_info": validation_result,
                "model_params": model.get_params() if hasattr(model, 'get_params') else {}
            }
            
            training_time = time.time() - start_time
            
            return TrainingResult(
                model=full_pipeline,
                metrics=metrics,
                feature_importance=feature_importance,
                metadata=metadata,
                preprocessing_info=preprocessing_info,
                training_time=training_time,
                validation_scores=cv_scores.tolist() if self.config.cv_folds > 1 else None
            )
            
        except Exception as e:
            # Return error result
            return TrainingResult(
                model=None,
                metrics={"error": str(e)},
                feature_importance=pd.DataFrame(),
                metadata={"error": str(e), "problem_type": "unknown"},
                preprocessing_info={"error": str(e)},
                training_time=time.time() - start_time
            )
    
    def _create_sklearn_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create sklearn preprocessing pipeline."""
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        transformers = []
        
        if numeric_features:
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])
            transformers.append(('num', numeric_transformer, numeric_features))
        
        if categorical_features:
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        return ColumnTransformer(transformers=transformers, remainder='drop')
    
    def _perform_cross_validation(self, model: BaseEstimator, X: pd.DataFrame, 
                                y: pd.Series, problem_type: str) -> np.ndarray:
        """Perform cross-validation."""
        try:
            if problem_type == "classification":
                cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
                scoring = "f1_weighted"
            else:
                cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
                scoring = "neg_root_mean_squared_error"
            
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            
            # Convert negative scores back to positive for RMSE
            if scoring == "neg_root_mean_squared_error":
                scores = -scores
                
            return scores
            
        except Exception:
            # Return dummy scores if CV fails
            return np.array([0.0] * self.config.cv_folds)

# ============================================================================
# CONVENIENCE FUNCTIONS (maintaining backward compatibility)
# ============================================================================

def detect_problem_type(target) -> str:
    """Legacy function for problem type detection."""
    detector = ProblemTypeDetector()
    return detector.detect_problem_type(pd.Series(target))

def train_sklearn_enhanced(df: pd.DataFrame, config: ModelConfig) -> Tuple[BaseEstimator, Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """Enhanced training function with new architecture."""
    orchestrator = MLTrainingOrchestrator(config)
    result = orchestrator.train(df)
    
    return (
        result.model,
        result.metrics,
        result.feature_importance,
        result.metadata
    )

def train_sklearn(df: pd.DataFrame, **kwargs) -> Tuple[BaseEstimator, Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """Legacy compatibility wrapper maintaining original signature."""
    
    # Convert old parameters to new ModelConfig
    config_params = {}
    
    # Map common parameters
    param_mapping = {
        'target': 'target',
        'problem_type': 'problem_type', 
        'engine': 'engine',
        'cv_folds': 'cv_folds',
        'test_size': 'test_size',
        'random_state': 'random_state'
    }
    
    for old_param, new_param in param_mapping.items():
        if old_param in kwargs:
            config_params[new_param] = kwargs[old_param]
    
    # Handle special cases
    if 'compute_shap' in kwargs:
        # SHAP computation would be handled separately in the new architecture
        pass
    
    if 'out_dir' in kwargs:
        # Output directory handling would be in a separate export component
        pass
    
    # Create config with defaults for missing required fields
    if 'target' not in config_params:
        raise ValueError("target parameter is required")
    
    config = ModelConfig(**config_params)
    
    # Use new training pipeline
    return train_sklearn_enhanced(df, config)

def export_visualizations(*args, **kwargs) -> Dict[str, Any]:
    """Placeholder for visualization export - maintained for compatibility."""
    return {}

# ============================================================================
# HYPERPARAMETER OPTIMIZATION (Optuna integration)
# ============================================================================

class HyperparameterOptimizer:
    """Handles hyperparameter optimization using Optuna or RandomizedSearchCV."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def optimize(self, trainer: BaseModelTrainer, X: pd.DataFrame, y: pd.Series, problem_type: str) -> BaseEstimator:
        """Optimize hyperparameters for the given trainer."""
        if HAVE_OPTUNA and self.config.use_optuna:
            return self._optimize_with_optuna(trainer, X, y, problem_type)
        else:
            return self._optimize_with_sklearn(trainer, X, y, problem_type)
    
    def _optimize_with_optuna(self, trainer: BaseModelTrainer, X: pd.DataFrame, y: pd.Series, problem_type: str) -> BaseEstimator:
        """Optimize using Optuna."""
        def objective(trial):
            # Get parameter suggestions from the trainer
            param_grid = trainer.get_param_grid()
            params = {}
            
            for param_name, param_values in param_grid.items():
                if isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
            
            # Create model with suggested parameters
            model = trainer.create_model(problem_type, **params)
            
            # Evaluate with cross-validation
            if problem_type == "classification":
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
                scoring = "f1_weighted"
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
                scoring = "neg_mean_squared_error"
            
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()
        
        try:
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.config.hyperopt_trials, timeout=self.config.hyperopt_trials * 5)
            
            # Create final model with best parameters
            best_params = study.best_params
            return trainer.create_model(problem_type, **best_params)
            
        except Exception:
            # Fallback to default model
            return trainer.create_model(problem_type)
    
    def _optimize_with_sklearn(self, trainer: BaseModelTrainer, X: pd.DataFrame, y: pd.Series, problem_type: str) -> BaseEstimator:
        """Optimize using sklearn RandomizedSearchCV."""
        try:
            base_model = trainer.create_model(problem_type)
            param_grid = trainer.get_param_grid()
            
            if problem_type == "classification":
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
                scoring = "f1_weighted"
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
                scoring = "neg_mean_squared_error"
            
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=min(50, self.config.hyperopt_trials),
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=self.config.random_state
            )
            
            search.fit(X, y)
            return search.best_estimator_
            
        except Exception:
            # Fallback to default model
            return trainer.create_model(problem_type)

# ============================================================================
# ADVANCED TRAINING ORCHESTRATOR WITH HYPEROPT
# ============================================================================

class AdvancedMLTrainingOrchestrator(MLTrainingOrchestrator):
    """Extended orchestrator with hyperparameter optimization."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.hyperopt = HyperparameterOptimizer(config)
    
    def train(self, df: pd.DataFrame) -> TrainingResult:
        """Enhanced training pipeline with hyperparameter optimization."""
        start_time = time.time()
        
        try:
            # Steps 1-5: Same as base class
            validation_result = self.validator.validate_dataframe(df, self.config.target)
            if not validation_result["valid"]:
                raise ValueError(f"Data validation failed: {validation_result['issues']}")
            
            target_series = df[self.config.target]
            problem_type = self.config.problem_type or self.problem_detector.detect_problem_type(target_series)
            
            df_processed, preprocessing_info = self.preprocessor.preprocess(df, self.config.target)
            
            y = df_processed[self.config.target]
            X = df_processed.drop(columns=[self.config.target])
            
            stratify = y if problem_type == "classification" and y.nunique() > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify
            )
            
            # Step 6: Create trainer and preprocessing
            trainer = self.model_factory.get_trainer(self.config.engine, self.config.random_state)
            preprocessor_pipeline = self._create_sklearn_preprocessor(X_train)
            
            # Apply preprocessing to training data for hyperopt
            X_train_processed = preprocessor_pipeline.fit_transform(X_train)
            
            # Step 7: Hyperparameter optimization (if dataset is large enough)
            if len(X_train) > 1000 and self.config.hyperopt_trials > 10:
                optimized_model = self.hyperopt.optimize(trainer, X_train_processed, y_train, problem_type)
            else:
                optimized_model = trainer.create_model(problem_type)
            
            # Step 8: Create final pipeline
            full_pipeline = Pipeline([
                ('preprocessor', preprocessor_pipeline),
                ('estimator', optimized_model)
            ])
            
            # Step 9: Train final model
            full_pipeline.fit(X_train, y_train)
            
            # Steps 10-13: Same as base class (predictions, metrics, etc.)
            y_pred = full_pipeline.predict(X_test)
            y_proba = None
            
            if problem_type == "classification" and hasattr(full_pipeline, "predict_proba"):
                try:
                    proba_output = full_pipeline.predict_proba(X_test)
                    if proba_output.shape[1] == 2:
                        y_proba = proba_output[:, 1]
                except Exception:
                    pass
            
            metrics = self.metrics_calculator.calculate_metrics(y_test, y_pred, y_proba, problem_type)
            
            if self.config.cv_folds > 1:
                cv_scores = self._perform_cross_validation(full_pipeline, X, y, problem_type)
                metrics.update({
                    "cv_mean": float(np.mean(cv_scores)),
                    "cv_std": float(np.std(cv_scores)),
                    "cv_scores": cv_scores.tolist()
                })
            
            feature_names = list(X.columns)
            feature_importance = self.importance_extractor.extract_importance(full_pipeline, feature_names)
            
            metadata = {
                "problem_type": problem_type,
                "engine": self.config.engine,
                "n_samples": len(df_processed),
                "n_features": len(X.columns),
                "validation_info": validation_result,
                "model_params": optimized_model.get_params() if hasattr(optimized_model, 'get_params') else {},
                "hyperopt_applied": len(X_train) > 1000 and self.config.hyperopt_trials > 10
            }
            
            training_time = time.time() - start_time
            
            return TrainingResult(
                model=full_pipeline,
                metrics=metrics,
                feature_importance=feature_importance,
                metadata=metadata,
                preprocessing_info=preprocessing_info,
                training_time=training_time,
                validation_scores=cv_scores.tolist() if self.config.cv_folds > 1 else None
            )
            
        except Exception as e:
            return TrainingResult(
                model=None,
                metrics={"error": str(e)},
                feature_importance=pd.DataFrame(),
                metadata={"error": str(e), "problem_type": "unknown"},
                preprocessing_info={"error": str(e)},
                training_time=time.time() - start_time
            )

# ============================================================================
# MAIN FUNCTIONS FOR EXTERNAL USE
# ============================================================================

def train_model_comprehensive(df: pd.DataFrame, config: ModelConfig, use_advanced: bool = True) -> TrainingResult:
    """
    Main function for comprehensive model training.
    
    Args:
        df: Input dataframe
        config: Training configuration
        use_advanced: Whether to use advanced orchestrator with hyperopt
        
    Returns:
        Complete training result
    """
    if use_advanced:
        orchestrator = AdvancedMLTrainingOrchestrator(config)
    else:
        orchestrator = MLTrainingOrchestrator(config)
    
    return orchestrator.train(df)

# Export main classes and functions for backward compatibility
__all__ = [
    "ModelConfig",
    "TrainingResult", 
    "train_sklearn",
    "train_sklearn_enhanced",
    "train_model_comprehensive",
    "detect_problem_type",
    "export_visualizations",
    "MLTrainingOrchestrator",
    "AdvancedMLTrainingOrchestrator"
]