from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

class ProblemType(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    UNKNOWN = "unknown"

@dataclass
class TargetAnalysis:
    """Analiza kolumny celu."""
    name: str
    problem_type: ProblemType
    unique_values: int
    missing_ratio: float
    target_distribution: Dict[str, Any]
    is_balanced: bool = True
    needs_transformation: bool = False
    
class SmartTargetDetector:
    """Inteligentny detektor kolumn celu."""
    
    def __init__(self, 
                 target_keywords: List[str] = None,
                 exclude_patterns: List[str] = None,
                 min_unique_ratio: float = 0.01):
        
        self.target_keywords = target_keywords or [
            "target", "y", "label", "class", "outcome",
            "price", "amount", "value", "revenue", "sales", 
            "score", "rating", "prediction", "result"
        ]
        
        self.exclude_patterns = exclude_patterns or [
            "id", "key", "index", "unnamed", "date", "time", "timestamp"
        ]
        
        self.min_unique_ratio = min_unique_ratio
    
    def detect_target(self, df: pd.DataFrame, 
                     preferred_target: Optional[str] = None) -> Optional[str]:
        """Wykrywa najlepszą kolumnę celu."""
        
        if preferred_target and preferred_target in df.columns:
            return preferred_target
        
        # 1. Sprawdź po nazwach (case-insensitive)
        column_lower = {col.lower(): col for col in df.columns}
        
        for keyword in self.target_keywords:
            if keyword.lower() in column_lower:
                candidate = column_lower[keyword.lower()]
                if self._is_valid_target(df[candidate]):
                    return candidate
        
        # 2. Fallback: numeryczne kolumny z wystarczającą różnorodnością
        numeric_candidates = []
        
        for col in df.columns:
            if not self._should_exclude_column(col) and pd.api.types.is_numeric_dtype(df[col]):
                if self._is_valid_target(df[col]):
                    numeric_candidates.append(col)
        
        # Sortuj po pozycji (ostatnie kolumny często to target)
        numeric_candidates.sort(key=lambda x: list(df.columns).index(x), reverse=True)
        
        return numeric_candidates[0] if numeric_candidates else None
    
    def _should_exclude_column(self, column_name: str) -> bool:
        """Sprawdza czy kolumnę należy wykluczyć."""
        name_lower = column_name.lower()
        return any(pattern in name_lower for pattern in self.exclude_patterns)
    
    def _is_valid_target(self, series: pd.Series) -> bool:
        """Waliduje czy seria może być targetem."""
        if len(series.dropna()) == 0:
            return False
        
        unique_ratio = series.nunique() / len(series)
        return unique_ratio >= self.min_unique_ratio
    
    def analyze_target(self, df: pd.DataFrame, target_col: str) -> TargetAnalysis:
        """Analizuje kolumnę celu."""
        
        if target_col not in df.columns:
            raise ValueError(f"Column {target_col} not found")
        
        series = df[target_col]
        unique_vals = series.nunique()
        missing_ratio = series.isna().sum() / len(series)
        
        # Określ typ problemu
        if pd.api.types.is_numeric_dtype(series):
            if unique_vals == 2:
                problem_type = ProblemType.BINARY_CLASSIFICATION
            elif unique_vals <= 20:  # Heurystyka
                problem_type = ProblemType.MULTICLASS_CLASSIFICATION
            else:
                problem_type = ProblemType.REGRESSION
        else:
            if unique_vals == 2:
                problem_type = ProblemType.BINARY_CLASSIFICATION
            else:
                problem_type = ProblemType.MULTICLASS_CLASSIFICATION
        
        # Analiza dystrybucji
        if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            value_counts = series.value_counts(normalize=True)
            is_balanced = (value_counts.min() / value_counts.max()) > 0.3  # Heurystyka
            distribution = {"class_distribution": value_counts.to_dict()}
        else:
            is_balanced = True
            distribution = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "skewness": float(series.skew()) if hasattr(series, 'skew') else 0.0
            }
        
        # Sprawdź potrzebę transformacji
        needs_transformation = False
        if problem_type == ProblemType.REGRESSION:
            skewness = abs(distribution.get("skewness", 0))
            needs_transformation = skewness > 2.0  # Bardzo skośny rozkład
        
        return TargetAnalysis(
            name=target_col,
            problem_type=problem_type,
            unique_values=unique_vals,
            missing_ratio=missing_ratio,
            target_distribution=distribution,
            is_balanced=is_balanced,
            needs_transformation=needs_transformation
        )

class MLRecommendationEngine:
    """Generator rekomendacji ML."""
    
    def __init__(self, language: str = "pl"):
        self.language = language
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Ładuje szablony rekomendacji."""
        return {
            "pl": {
                "regression": [
                    "Sprawdź rozkład targetu - jeśli skośny, zastosuj log/sqrt transform",
                    "Usuń outliers lub zastosuj winsorization - poprawi RMSE/MAE", 
                    "Dodaj cechy kalendarzowe jeśli masz daty - model lepiej uchwyci sezonowość",
                    "Przetestuj feature engineering - interakcje ważnych cech mogą dać boost",
                    "Sprawdź residuals vs predicted - nieliniowości mogą wymagać innych modeli"
                ],
                "classification": [
                    "Zbalansuj klasy przez class_weight lub oversampling - poprawi F1/ROC-AUC",
                    "Dostosuj próg decyzyjny pod cel biznesowy - minimalizuj FP lub FN",
                    "Przeanalizuj confusion matrix - gdzie model się najczęściej myli?",
                    "Zbadaj feature importance - czy cechy są stabilne i sensowne biznesowo?",
                    "Rozważ ensemble methods dla lepszej generalizacji"
                ],
                "imbalanced": [
                    "Użyj stratified sampling przy podziale train/test",
                    "Zastosuj SMOTE lub ADASYN do generacji syntetycznych próbek",
                    "Wypróbuj cost-sensitive learning algorithms",
                    "Skup się na precision/recall zamiast accuracy"
                ]
            }
        }
    
    def generate_recommendations(self, 
                               analysis: TargetAnalysis,
                               top_features: List[str] = None,
                               dataset_size: int = None) -> str:
        """Generuje spersonalizowane rekomendacje."""
        
        recommendations = []
        
        # Feature-based recommendations
        if top_features:
            top_features_text = ", ".join(f"`{f}`" for f in top_features[:5])
            recommendations.append(
                f"🎯 **Najważniejsze cechy**: {top_features_text}. "
                "Skup się na ich jakości, stabilności i sensowności biznesowej."
            )
        
        # Problem-specific recommendations
        if analysis.problem_type == ProblemType.REGRESSION:
            recommendations.extend(self.templates[self.language]["regression"])
            
            if analysis.needs_transformation:
                recommendations.insert(-2, 
                    f"⚠️ Target `{analysis.name}` ma skośny rozkład - "
                    "zastosuj log1p() lub Box-Cox transformation"
                )
                
        elif analysis.problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            recommendations.extend(self.templates[self.language]["classification"])
            
            if not analysis.is_balanced:
                recommendations.extend(self.templates[self.language]["imbalanced"])
        
        # Dataset-specific recommendations  
        if dataset_size:
            if dataset_size < 1000:
                recommendations.append(
                    "📊 Mały dataset - rozważ simpler models (linear/tree) i cross-validation"
                )
            elif dataset_size > 100000:
                recommendations.append(
                    "🚀 Duży dataset - możesz użyć complex models (deep learning, ensembles)"
                )
        
        # Format output
        formatted = "\n".join(f"• {rec}" for rec in recommendations)
        
        return f"""
## 🎯 Rekomendacje dla `{analysis.name}` ({analysis.problem_type.value})

{formatted}

---
💡 **Następne kroki**: Zacznij od najważniejszych cech, popraw jakość danych, następnie eksperymentuj z modelami.
"""

# Kompatybilność wsteczna
def auto_pick_target(df: pd.DataFrame) -> str | None:
    """Kompatybilność wsteczna - wykrywa target."""
    detector = SmartTargetDetector()
    return detector.detect_target(df)

def recommendations_text(target: str, problem: str, top_features: list[str]) -> str:
    """Kompatybilność wsteczna - generuje rekomendacje."""
    
    # Map old problem format to new
    problem_type_map = {
        "regression": ProblemType.REGRESSION,
        "classification": ProblemType.BINARY_CLASSIFICATION,
        "clf": ProblemType.BINARY_CLASSIFICATION
    }
    
    # Create dummy analysis
    analysis = TargetAnalysis(
        name=target,
        problem_type=problem_type_map.get(problem, ProblemType.UNKNOWN),
        unique_values=0,
        missing_ratio=0.0,
        target_distribution={}
    )
    
    engine = MLRecommendationEngine()
    return engine.generate_recommendations(analysis, top_features)