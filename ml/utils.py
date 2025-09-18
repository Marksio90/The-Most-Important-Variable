from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging

# Konfiguracja logowania
logger = logging.getLogger(__name__)

# ============================================================================
# WSPÓLNE DEFINICJE TYPÓW (zsynchronizowane z całą aplikacją)
# ============================================================================

class ProblemType(Enum):
    """Typy problemów ML - spójne z resztą aplikacji."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    UNKNOWN = "unknown"

class TargetQuality(Enum):
    """Jakość wykrytego targetu."""
    EXCELLENT = "excellent"  # Jasno identyfikowalny target
    GOOD = "good"           # Prawdopodobny target
    POOR = "poor"           # Słaby kandydat
    UNKNOWN = "unknown"     # Nie można określić

# ============================================================================
# KONFIGURACJA I WYNIKI ANALIZY
# ============================================================================

@dataclass
class TargetAnalysis:
    """Kompleksowa analiza kolumny celu."""
    name: str
    problem_type: ProblemType
    quality: TargetQuality
    unique_values: int
    missing_ratio: float
    target_distribution: Dict[str, Any]
    is_balanced: bool = True
    needs_transformation: bool = False
    confidence_score: float = 0.0  # 0-1, pewność że to dobry target
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje analizę do słownika."""
        return {
            "name": self.name,
            "problem_type": self.problem_type.value,
            "quality": self.quality.value,
            "unique_values": self.unique_values,
            "missing_ratio": self.missing_ratio,
            "target_distribution": self.target_distribution,
            "is_balanced": self.is_balanced,
            "needs_transformation": self.needs_transformation,
            "confidence_score": self.confidence_score,
            "recommendations": self.recommendations
        }

@dataclass
class DetectionConfig:
    """Konfiguracja dla detektora targetu."""
    target_keywords: List[str] = field(default_factory=lambda: [
        "target", "y", "label", "class", "outcome",
        "price", "amount", "value", "revenue", "sales", 
        "score", "rating", "prediction", "result", "AveragePrice"
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "id", "key", "index", "unnamed", "date", "time", "timestamp", "url", "link"
    ])
    min_unique_ratio: float = 0.01  # Minimum unikatów względem wielkości datasetu
    max_unique_ratio: float = 0.95  # Maximum unikatów (powyżej = prawdopodobnie ID)
    min_samples_classification: int = 5  # Minimum próbek na klasę
    max_classes_classification: int = 50  # Maximum klas do uznania za klasyfikację

# ============================================================================
# INTERFEJSY I ABSTRAKCJE
# ============================================================================

class TargetDetector(ABC):
    """Abstrakcyjny detektor targetu."""
    
    @abstractmethod
    def detect_target(self, df: pd.DataFrame, preferred_target: Optional[str] = None) -> Optional[str]:
        """Wykrywa najlepszą kolumnę celu."""
        pass
    
    @abstractmethod
    def analyze_target(self, df: pd.DataFrame, target_col: str) -> TargetAnalysis:
        """Analizuje konkretną kolumnę celu."""
        pass

class RecommendationEngine(ABC):
    """Abstrakcyjny generator rekomendacji."""
    
    @abstractmethod
    def generate_recommendations(self, analysis: TargetAnalysis, **kwargs) -> str:
        """Generuje rekomendacje na podstawie analizy."""
        pass

# ============================================================================
# IMPLEMENTACJE DETEKTORÓW
# ============================================================================

class HeuristicTargetDetector(TargetDetector):
    """Detektor targetu oparty na heurystykach."""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
    
    def detect_target(self, df: pd.DataFrame, preferred_target: Optional[str] = None) -> Optional[str]:
        """Wykrywa najlepszą kolumnę celu używając heurystyk."""
        
        if df.empty:
            return None
        
        # 1. Sprawdź preferowany target
        if preferred_target and preferred_target in df.columns:
            if self._is_valid_target(df[preferred_target]):
                return preferred_target
        
        # 2. Sprawdź po nazwach (case-insensitive, priorytet)
        column_lower = {col.lower(): col for col in df.columns}
        
        for keyword in self.config.target_keywords:
            if keyword.lower() in column_lower:
                candidate = column_lower[keyword.lower()]
                if self._is_valid_target(df[candidate]):
                    return candidate
        
        # 3. Heurystyki na podstawie danych
        candidates = self._get_target_candidates(df)
        
        if candidates:
            # Sortuj według confidence score
            best_candidate = max(candidates, key=lambda x: x[1])
            return best_candidate[0]
        
        return None
    
    def analyze_target(self, df: pd.DataFrame, target_col: str) -> TargetAnalysis:
        """Analizuje kolumnę celu."""
        
        if target_col not in df.columns:
            raise ValueError(f"Column {target_col} not found")
        
        series = df[target_col]
        unique_vals = series.nunique(dropna=True)
        missing_ratio = series.isna().sum() / len(series)
        
        # Określ typ problemu i jakość
        problem_type, confidence = self._determine_problem_type(series)
        quality = self._assess_target_quality(series, missing_ratio, unique_vals, len(df))
        
        # Analiza dystrybucji
        distribution, is_balanced = self._analyze_distribution(series, problem_type)
        
        # Sprawdź potrzebę transformacji
        needs_transformation = self._needs_transformation(series, problem_type)
        
        # Generuj rekomendacje
        recommendations = self._generate_target_recommendations(
            series, problem_type, quality, is_balanced, needs_transformation
        )
        
        return TargetAnalysis(
            name=target_col,
            problem_type=problem_type,
            quality=quality,
            unique_values=unique_vals,
            missing_ratio=missing_ratio,
            target_distribution=distribution,
            is_balanced=is_balanced,
            needs_transformation=needs_transformation,
            confidence_score=confidence,
            recommendations=recommendations
        )
    
    def _is_valid_target(self, series: pd.Series) -> bool:
        """Waliduje czy seria może być targetem."""
        if len(series.dropna()) == 0:
            return False
        
        unique_ratio = series.nunique() / len(series)
        return (self.config.min_unique_ratio <= unique_ratio <= self.config.max_unique_ratio)
    
    def _should_exclude_column(self, column_name: str) -> bool:
        """Sprawdza czy kolumnę należy wykluczyć."""
        name_lower = column_name.lower()
        return any(pattern in name_lower for pattern in self.config.exclude_patterns)
    
    def _get_target_candidates(self, df: pd.DataFrame) -> List[Tuple[str, float]]:
        """Zwraca listę kandydatów na target z confidence score."""
        candidates = []
        
        for col in df.columns:
            if self._should_exclude_column(col):
                continue
            
            if not self._is_valid_target(df[col]):
                continue
            
            confidence = self._calculate_target_confidence(df[col], col)
            candidates.append((col, confidence))
        
        return candidates
    
    def _calculate_target_confidence(self, series: pd.Series, column_name: str) -> float:
        """Oblicza confidence score dla kandydata na target."""
        score = 0.0
        
        # Bonus za nazwę
        name_lower = column_name.lower()
        for keyword in self.config.target_keywords:
            if keyword.lower() in name_lower:
                score += 0.4
                break
        
        # Bonus za typ danych
        if pd.api.types.is_numeric_dtype(series):
            score += 0.2
        elif series.dtype == 'object':
            score += 0.1
        
        # Bonus za odpowiednią liczbę unikalnych wartości
        unique_ratio = series.nunique() / len(series)
        if 0.01 <= unique_ratio <= 0.5:  # Sweet spot for targets
            score += 0.2
        
        # Penalty za braki danych
        missing_ratio = series.isna().sum() / len(series)
        score -= missing_ratio * 0.3
        
        # Bonus za pozycję w DataFrame (ostatnie kolumny często to target)
        position_bonus = 0.1 * (1 - (list(series.index).index(series.name) / len(series)))
        score += position_bonus
        
        return min(1.0, max(0.0, score))
    
    def _determine_problem_type(self, series: pd.Series) -> Tuple[ProblemType, float]:
        """Określa typ problemu i confidence."""
        clean_series = series.dropna()
        if clean_series.empty:
            return ProblemType.UNKNOWN, 0.0
        
        unique_values = clean_series.nunique()
        total_samples = len(clean_series)
        
        # Jasne przypadki klasyfikacji
        if series.dtype == 'object' or series.dtype.name == 'category':
            return ProblemType.MULTICLASS_CLASSIFICATION if unique_values > 2 else ProblemType.BINARY_CLASSIFICATION, 0.9
        
        if pd.api.types.is_bool_dtype(series):
            return ProblemType.BINARY_CLASSIFICATION, 0.95
        
        # Binarna numeryczna
        if unique_values == 2:
            return ProblemType.BINARY_CLASSIFICATION, 0.8
        
        # Klasyfikacja wieloklasowa
        if unique_values <= self.config.max_classes_classification and unique_values / total_samples < 0.1:
            return ProblemType.MULTICLASS_CLASSIFICATION, 0.7
        
        # Integer z małą liczbą klas
        if pd.api.types.is_integer_dtype(series) and unique_values <= 20:
            return ProblemType.MULTICLASS_CLASSIFICATION, 0.6
        
        # Regresja
        if pd.api.types.is_numeric_dtype(series):
            return ProblemType.REGRESSION, 0.8
        
        return ProblemType.UNKNOWN, 0.0
    
    def _assess_target_quality(self, series: pd.Series, missing_ratio: float, 
                              unique_vals: int, total_samples: int) -> TargetQuality:
        """Ocenia jakość targetu."""
        
        # Kryteria jakości
        if missing_ratio > 0.5:
            return TargetQuality.POOR
        
        if unique_vals < 2:
            return TargetQuality.POOR
        
        # Sprawdź czy nie jest to prawdopodobnie ID
        if unique_vals / total_samples > 0.95:
            return TargetQuality.POOR
        
        # Dobra jakość
        if missing_ratio < 0.1 and unique_vals >= 2:
            return TargetQuality.EXCELLENT if missing_ratio < 0.05 else TargetQuality.GOOD
        
        return TargetQuality.POOR
    
    def _analyze_distribution(self, series: pd.Series, problem_type: ProblemType) -> Tuple[Dict[str, Any], bool]:
        """Analizuje rozkład targetu."""
        
        if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            value_counts = series.value_counts(normalize=True)
            is_balanced = (value_counts.min() / value_counts.max()) > 0.3 if len(value_counts) > 1 else True
            
            distribution = {
                "type": "categorical",
                "class_distribution": value_counts.to_dict(),
                "n_classes": len(value_counts),
                "most_frequent_class": value_counts.index[0] if len(value_counts) > 0 else None,
                "least_frequent_class": value_counts.index[-1] if len(value_counts) > 0 else None
            }
        else:
            # Regresja
            is_balanced = True
            try:
                distribution = {
                    "type": "continuous",
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "skewness": float(series.skew()) if hasattr(series, 'skew') else 0.0,
                    "kurtosis": float(series.kurtosis()) if hasattr(series, 'kurtosis') else 0.0
                }
            except Exception:
                distribution = {"type": "continuous", "error": "Could not calculate statistics"}
        
        return distribution, is_balanced
    
    def _needs_transformation(self, series: pd.Series, problem_type: ProblemType) -> bool:
        """Sprawdza czy target potrzebuje transformacji."""
        
        if problem_type != ProblemType.REGRESSION:
            return False
        
        try:
            # Sprawdź skośność
            skewness = abs(series.skew())
            if skewness > 2.0:
                return True
            
            # Sprawdź czy wszystkie wartości dodatnie (kandydat do log transform)
            if (series > 0).all() and skewness > 1.0:
                return True
                
        except Exception:
            pass
        
        return False
    
    def _generate_target_recommendations(self, series: pd.Series, problem_type: ProblemType,
                                       quality: TargetQuality, is_balanced: bool,
                                       needs_transformation: bool) -> List[str]:
        """Generuje rekomendacje dla targetu."""
        recommendations = []
        
        # Rekomendacje jakości
        if quality == TargetQuality.POOR:
            missing_ratio = series.isna().sum() / len(series)
            if missing_ratio > 0.3:
                recommendations.append("Rozważ usunięcie wierszy z brakującymi wartościami targetu")
            
            if series.nunique() < 2:
                recommendations.append("Target ma za mało unikalnych wartości - sprawdź czy to właściwa kolumna")
        
        # Rekomendacje dla klasyfikacji
        if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            if not is_balanced:
                recommendations.append("Klasy są niezbalansowane - rozważ techniki balansowania (SMOTE, class_weight)")
            
            value_counts = series.value_counts()
            min_class_size = value_counts.min()
            if min_class_size < self.config.min_samples_classification:
                recommendations.append(f"Niektóre klasy mają za mało próbek (<{self.config.min_samples_classification}) - rozważ grupowanie rzadkich klas")
        
        # Rekomendacje dla regresji
        if problem_type == ProblemType.REGRESSION:
            if needs_transformation:
                recommendations.append("Rozkład targetu jest skośny - rozważ transformację log1p() lub Box-Cox")
            
            # Sprawdź outliery
            try:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
                if outliers > len(series) * 0.05:
                    recommendations.append("Wykryto dużo outlierów - rozważ winsoryzację lub usunięcie")
            except Exception:
                pass
        
        if not recommendations:
            recommendations.append("Target wygląda dobrze przygotowany do trenowania")
        
        return recommendations

# ============================================================================
# GENERATOR REKOMENDACJI ML
# ============================================================================

class MLRecommendationEngine(RecommendationEngine):
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
                    "Usuń outliery lub zastosuj winsorization - poprawi RMSE/MAE", 
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
                ],
                "data_quality": [
                    "Sprawdź korelacje między cechami - usuń silnie skorelowane",
                    "Zidentyfikuj i obsłuż outliery w kluczowych cechach",
                    "Rozważ normalizację/standaryzację cech numerycznych",
                    "Grupuj rzadkie kategorie w cechach kategorycznych"
                ]
            }
        }
    
    def generate_recommendations(self, 
                               analysis: TargetAnalysis,
                               top_features: List[str] = None,
                               dataset_size: int = None,
                               **kwargs) -> str:
        """Generuje spersonalizowane rekomendacje."""
        
        recommendations = []
        
        # Rekomendacje na podstawie analizy targetu
        if analysis.recommendations:
            recommendations.append("🎯 **Rekomendacje dla targetu:**")
            for rec in analysis.recommendations:
                recommendations.append(f"• {rec}")
            recommendations.append("")
        
        # Feature-based recommendations
        if top_features:
            top_features_text = ", ".join(f"`{f}`" for f in top_features[:5])
            recommendations.append(
                f"🏆 **Najważniejsze cechy**: {top_features_text}. "
                "Skup się na ich jakości, stabilności i sensowności biznesowej."
            )
            recommendations.append("")
        
        # Problem-specific recommendations
        if analysis.problem_type == ProblemType.REGRESSION:
            recommendations.append("📈 **Rekomendacje dla regresji:**")
            recommendations.extend(f"• {rec}" for rec in self.templates[self.language]["regression"])
            
            if analysis.needs_transformation:
                recommendations.append(
                    f"• ⚠️ Target `{analysis.name}` ma skośny rozkład - "
                    "zastosuj log1p() lub Box-Cox transformation"
                )
                
        elif analysis.problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            recommendations.append("🎯 **Rekomendacje dla klasyfikacji:**")
            recommendations.extend(f"• {rec}" for rec in self.templates[self.language]["classification"])
            
            if not analysis.is_balanced:
                recommendations.append("")
                recommendations.append("⚖️ **Niezbalansowane klasy:**")
                recommendations.extend(f"• {rec}" for rec in self.templates[self.language]["imbalanced"])
        
        # Dataset-specific recommendations  
        if dataset_size:
            recommendations.append("")
            recommendations.append("📊 **Rekomendacje dla datasetu:**")
            if dataset_size < 1000:
                recommendations.append(
                    "• Mały dataset - rozważ simpler models (linear/tree) i cross-validation"
                )
            elif dataset_size > 100000:
                recommendations.append(
                    "• Duży dataset - możesz użyć complex models (deep learning, ensembles)"
                )
        
        # Quality recommendations
        if analysis.quality in [TargetQuality.POOR, TargetQuality.UNKNOWN]:
            recommendations.append("")
            recommendations.append("⚠️ **Jakość danych:**")
            recommendations.extend(f"• {rec}" for rec in self.templates[self.language]["data_quality"])
        
        # Format output
        formatted = "\n".join(recommendations)
        
        confidence_emoji = "🟢" if analysis.confidence_score > 0.7 else "🟡" if analysis.confidence_score > 0.4 else "🔴"
        
        return f"""
## {confidence_emoji} Rekomendacje dla `{analysis.name}` ({analysis.problem_type.value})

{formatted}

---
💡 **Następne kroki**: Zacznij od najważniejszych cech, popraw jakość danych, następnie eksperymentuj z modelami.

**Confidence score**: {analysis.confidence_score:.2f}/1.0
"""

# ============================================================================
# FACTORY PATTERN
# ============================================================================

class TargetAnalysisFactory:
    """Factory do tworzenia komponentów analizy targetu."""
    
    @staticmethod
    def create_detector(config: Optional[DetectionConfig] = None) -> TargetDetector:
        """Tworzy detektor targetu."""
        return HeuristicTargetDetector(config)
    
    @staticmethod
    def create_recommendation_engine(language: str = "pl") -> RecommendationEngine:
        """Tworzy generator rekomendacji."""
        return MLRecommendationEngine(language)
    
    @staticmethod
    def create_smart_target_detector(config: Optional[DetectionConfig] = None) -> 'SmartTargetDetector':
        """Tworzy kompletny smart target detector."""
        return SmartTargetDetector(config)

# ============================================================================
# GŁÓWNA KLASA ORCHESTRATORA
# ============================================================================

class SmartTargetDetector:
    """Główny orchestrator do analizy i detekcji targetu."""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.detector = HeuristicTargetDetector(self.config)
        self.recommendation_engine = MLRecommendationEngine()
    
    def detect_target(self, df: pd.DataFrame, preferred_target: Optional[str] = None) -> Optional[str]:
        """Wykrywa najlepszą kolumnę celu."""
        return self.detector.detect_target(df, preferred_target)
    
    def analyze_target(self, df: pd.DataFrame, target_col: str) -> TargetAnalysis:
        """Analizuje kolumnę celu."""
        return self.detector.analyze_target(df, target_col)
    
    def get_recommendations(self, analysis: TargetAnalysis, **kwargs) -> str:
        """Generuje rekomendacje na podstawie analizy."""
        return self.recommendation_engine.generate_recommendations(analysis, **kwargs)
    
    def full_target_analysis(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Kompleksowa analiza targetu - wykrywanie + analiza + rekomendacje."""
        
        # Wykryj target jeśli nie podano
        if not target_col:
            target_col = self.detect_target(df)
            if not target_col:
                return {
                    "success": False,
                    "error": "Could not detect suitable target column",
                    "suggestions": [
                        "Sprawdź czy dataset ma kolumnę do przewidywania",
                        "Upewnij się że target ma sensowną liczbę unikalnych wartości",
                        "Usuń kolumny ID lub timestamps które mogą mylić detektor"
                    ]
                }
        
        # Analizuj target
        analysis = self.analyze_target(df, target_col)
        
        # Generuj rekomendacje
        recommendations = self.get_recommendations(
            analysis, 
            dataset_size=len(df)
        )
        
        return {
            "success": True,
            "target_column": target_col,
            "analysis": analysis.to_dict(),
            "recommendations": recommendations,
            "summary": {
                "problem_type": analysis.problem_type.value,
                "quality": analysis.quality.value,
                "confidence": analysis.confidence_score,
                "needs_attention": analysis.quality == TargetQuality.POOR or not analysis.is_balanced
            }
        }

# ============================================================================
# KOMPATYBILNOŚĆ WSTECZNA
# ============================================================================

def auto_pick_target(df: pd.DataFrame) -> Optional[str]:
    """Kompatybilność wsteczna - wykrywa target."""
    detector = SmartTargetDetector()
    return detector.detect_target(df)

def recommendations_text(target: str, problem: str, top_features: List[str]) -> str:
    """Kompatybilność wsteczna - generuje rekomendacje."""
    
    # Map old problem format to new
    problem_type_map = {
        "regression": ProblemType.REGRESSION,
        "classification": ProblemType.BINARY_CLASSIFICATION,
        "clf": ProblemType.BINARY_CLASSIFICATION
    }
    
    # Create dummy analysis for compatibility
    analysis = TargetAnalysis(
        name=target,
        problem_type=problem_type_map.get(problem, ProblemType.UNKNOWN),
        quality=TargetQuality.GOOD,
        unique_values=0,
        missing_ratio=0.0,
        target_distribution={},
        confidence_score=0.8
    )
    
    engine = MLRecommendationEngine()
    return engine.generate_recommendations(analysis, top_features)

# ============================================================================
# PRZYKŁAD UŻYCIA
# ============================================================================

if __name__ == "__main__":
    # Przykład użycia nowego API
    
    # Stwórz konfigurację
    config = DetectionConfig(
        target_keywords=["price", "revenue", "target", "y"],
        exclude_patterns=["id", "timestamp"],
        min_unique_ratio=0.02
    )
    
    # Stwórz analyzer
    analyzer = SmartTargetDetector(config)
    
    # Przykład użycia (gdyby był DataFrame)
    # result = analyzer.full_target_analysis(df)
    # print("Success:", result["success"])
    # if result["success"]:
    #     print("Target:", result["target_column"])
    #     print("Quality:", result["summary"]["quality"])
    #     print("Recommendations:")
    #     print(result["recommendations"])
    
    # Lub przez factory
    detector = TargetAnalysisFactory.create_smart_target_detector()
    # target = detector.detect_target(df)
    # analysis = detector.analyze_target(df, target)
    # recommendations = detector.get_recommendations(analysis)