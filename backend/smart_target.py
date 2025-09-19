# backend/smart_target.py — Inteligentny wybór targetu z wyjaśnieniami i rekomendacjami
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import re

from backend.utils import SmartTargetDetector, infer_problem_type, is_id_like


@dataclass
class TargetRecommendation:
    """Struktura rekomendacji wyboru targetu."""
    column: str
    confidence: float  # 0-1
    reason: str
    evidence: Dict[str, Any]
    problem_type: str
    quality_score: float  # 0-1
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class SmartTargetSelector:
    """
    Zaawansowany selektor targetu z wyjaśnieniami i rekomendacjami.
    Automatycznie analizuje dane i proponuje najlepszy target z uzasadnieniem.
    """
    
    def __init__(self):
        self.price_patterns = [
            "averageprice", "avgprice", "avg_price", "price", "cost", "value",
            "amount", "sum", "total", "revenue", "sales", "income", "profit"
        ]
        self.target_patterns = [
            "target", "label", "y", "outcome", "result", "prediction", "class"
        ]
        self.id_patterns = [
            "id", "uuid", "guid", "index", "idx", "key", "code", "number"
        ]
    
    def analyze_and_recommend(self, df: pd.DataFrame) -> List[TargetRecommendation]:
        """
        Analizuje DataFrame i zwraca listę rekomendacji targetu.
        Sortowane według confidence (najlepsze pierwsze).
        """
        if df is None or df.empty:
            return []
        
        recommendations = []
        
        # Analiza każdej kolumny
        for col in df.columns:
            try:
                rec = self._analyze_column(df, col)
                if rec and rec.confidence > 0.1:  # min threshold
                    recommendations.append(rec)
            except Exception:
                continue
        
        # Sortuj według confidence
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        # Dodaj pozycyjne bonusy
        self._apply_positional_bonuses(recommendations, df.columns)
        
        # Ponowne sortowanie po bonusach
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        return recommendations[:5]  # Top 5
    
    def get_best_target(self, df: pd.DataFrame) -> Optional[TargetRecommendation]:
        """Zwraca najlepszą rekomendację targetu."""
        recommendations = self.analyze_and_recommend(df)
        return recommendations[0] if recommendations else None
    
    def _analyze_column(self, df: pd.DataFrame, col: str) -> Optional[TargetRecommendation]:
        """Analizuje pojedynczą kolumnę jako potencjalny target."""
        series = df[col]
        
        # Podstawowe sprawdzenia
        if self._is_invalid_target(series, col):
            return None
        
        confidence = 0.0
        reasons = []
        evidence = {}
        warnings = []
        recommendations = []
        
        # 1) NAZWA - analiza wzorców w nazwie
        name_score, name_reason = self._analyze_column_name(col)
        confidence += name_score
        if name_reason:
            reasons.append(name_reason)
        
        # 2) TYP DANYCH I ROZKŁAD
        data_score, data_reason, data_evidence = self._analyze_data_distribution(series)
        confidence += data_score
        if data_reason:
            reasons.append(data_reason)
        evidence.update(data_evidence)
        
        # 3) POZYCJA W DATAFRAME
        position_score, position_reason = self._analyze_position(col, df.columns)
        confidence += position_score
        if position_reason:
            reasons.append(position_reason)
        
        # 4) KORELACJE Z INNYMI KOLUMNAMI
        correlation_score, correlation_reason = self._analyze_correlations(df, col)
        confidence += correlation_score
        if correlation_reason:
            reasons.append(correlation_reason)
        
        # 5) WYKRYJ TYP PROBLEMU
        problem_type = infer_problem_type(df, col)
        
        # 6) OCEŃ JAKOŚĆ JAKO TARGET
        quality_score, quality_warnings, quality_recs = self._assess_target_quality(series, problem_type)
        warnings.extend(quality_warnings)
        recommendations.extend(quality_recs)
        
        # Normalizuj confidence do 0-1
        confidence = min(1.0, confidence / 4.0)  # 4 komponenty
        
        # Główny powód (najważniejszy)
        main_reason = reasons[0] if reasons else "Analiza statystyczna"
        
        return TargetRecommendation(
            column=col,
            confidence=confidence,
            reason=main_reason,
            evidence=evidence,
            problem_type=problem_type,
            quality_score=quality_score,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _is_invalid_target(self, series: pd.Series, col: str) -> bool:
        """Sprawdza czy kolumna nie nadaje się na target."""
        # ID-like kolumny
        if is_id_like(series, col):
            return True
        
        # Całe puste
        if series.isna().all():
            return True
        
        # Za mało różnych wartości
        if series.nunique(dropna=True) <= 1:
            return True
        
        return False
    
    def _analyze_column_name(self, col: str) -> Tuple[float, str]:
        """Analizuje nazwę kolumny pod kątem wzorców targetowych."""
        col_norm = re.sub(r'[^a-z0-9]', '', col.lower())
        
        # Najwyższy priorytet: ceny
        for pattern in self.price_patterns:
            if pattern in col_norm:
                return 1.0, f"Nazwa '{col}' sugeruje zmienną cenową/finansową"
        
        # Średni priorytet: klasyczne targety
        for pattern in self.target_patterns:
            if pattern in col_norm:
                return 0.8, f"Nazwa '{col}' sugeruje zmienną docelową"
        
        # Sprawdź końcówki sugerujące wynik
        if col_norm.endswith(('rate', 'ratio', 'score', 'level', 'grade')):
            return 0.6, f"Nazwa '{col}' sugeruje zmienną wynikową"
        
        return 0.0, ""
    
    def _analyze_data_distribution(self, series: pd.Series) -> Tuple[float, str, Dict[str, Any]]:
        """Analizuje rozkład danych w kolumnie."""
        evidence = {}
        
        nunique = series.nunique(dropna=True)
        total_count = len(series)
        non_null_count = series.count()
        
        evidence.update({
            'nunique': nunique,
            'total_count': total_count,
            'non_null_count': non_null_count,
            'null_ratio': 1 - (non_null_count / total_count) if total_count > 0 else 1
        })
        
        # Numeryczne - może być regresja
        if pd.api.types.is_numeric_dtype(series):
            # Wysoką różnorodność = prawdopodobnie target regresyjny
            diversity_ratio = nunique / non_null_count if non_null_count > 0 else 0
            evidence['diversity_ratio'] = diversity_ratio
            
            if diversity_ratio > 0.7:
                return 0.8, f"Wysoka różnorodność wartości ({nunique}/{non_null_count}) sugeruje target regresyjny", evidence
            elif diversity_ratio > 0.3:
                return 0.6, f"Średnia różnorodność wartości może wskazywać na target", evidence
        
        # Kategoryczne - może być klasyfikacja
        else:
            if 2 <= nunique <= 20:  # dobra liczba klas
                evidence['class_distribution'] = series.value_counts().to_dict()
                return 0.7, f"Optymalna liczba klas ({nunique}) dla klasyfikacji", evidence
            elif nunique == 2:
                return 0.9, f"Binarna klasyfikacja (2 klasy) - idealny target", evidence
        
        return 0.1, "Podstawowa analiza rozkładu", evidence
    
    def _analyze_position(self, col: str, all_columns: List[str]) -> Tuple[float, str]:
        """Analizuje pozycję kolumny w DataFrame."""
        if not all_columns:
            return 0.0, ""
        
        col_idx = all_columns.index(col)
        
        # Ostatnia kolumna = często target
        if col_idx == len(all_columns) - 1:
            return 0.3, "Ostatnia kolumna w danych (częsta konwencja dla targetu)"
        
        # Pierwsza kolumna = rzadko target (często ID)
        if col_idx == 0:
            return -0.2, ""
        
        return 0.0, ""
    
    def _analyze_correlations(self, df: pd.DataFrame, col: str) -> Tuple[float, str]:
        """Analizuje korelacje z innymi kolumnami."""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if col not in numeric_df.columns or len(numeric_df.columns) < 2:
                return 0.0, ""
            
            corr_matrix = numeric_df.corr()
            col_correlations = corr_matrix[col].drop(col).abs()
            
            if len(col_correlations) == 0:
                return 0.0, ""
            
            max_corr = col_correlations.max()
            mean_corr = col_correlations.mean()
            
            # Umiarkowane korelacje = dobry target (nie za wysokie, nie za niskie)
            if 0.3 <= max_corr <= 0.8:
                return 0.4, f"Umiarkowane korelacje z innymi cechami (max: {max_corr:.2f})"
            elif mean_corr > 0.2:
                return 0.2, f"Średnie korelacje z cechami (średnia: {mean_corr:.2f})"
        
        except Exception:
            pass
        
        return 0.0, ""
    
    def _assess_target_quality(self, series: pd.Series, problem_type: str) -> Tuple[float, List[str], List[str]]:
        """Ocenia jakość kolumny jako target i daje rekomendacje."""
        warnings = []
        recommendations = []
        quality = 1.0  # start z max
        
        # Sprawdź braki
        null_ratio = series.isna().mean()
        if null_ratio > 0.1:
            quality -= 0.3
            warnings.append(f"Wysoki % braków danych ({null_ratio:.1%})")
            recommendations.append("Rozważ uzupełnienie braków przed treningiem")
        
        # Dla klasyfikacji
        if problem_type == "classification":
            value_counts = series.value_counts()
            
            # Sprawdź balans klas
            if len(value_counts) > 1:
                min_class = value_counts.min()
                max_class = value_counts.max()
                imbalance_ratio = max_class / min_class
                
                if imbalance_ratio > 10:
                    quality -= 0.2
                    warnings.append(f"Silny niebalans klas (ratio {imbalance_ratio:.1f}:1)")
                    recommendations.append("Rozważ techniki balansowania klas")
                
                # Sprawdź minimalne liczności
                if min_class < 5:
                    quality -= 0.4
                    warnings.append(f"Niektóre klasy mają za mało przykładów (min: {min_class})")
                    recommendations.append("Zbierz więcej danych lub połącz rzadkie klasy")
        
        # Dla regresji
        elif problem_type == "regression":
            if pd.api.types.is_numeric_dtype(series):
                # Sprawdź rozkład
                try:
                    skewness = series.skew()
                    if abs(skewness) > 2:
                        quality -= 0.1
                        recommendations.append(f"Silnie skośny rozkład (skewness: {skewness:.2f}) - rozważ transformację")
                except Exception:
                    pass
        
        quality = max(0.0, min(1.0, quality))
        return quality, warnings, recommendations
    
    def _apply_positional_bonuses(self, recommendations: List[TargetRecommendation], columns: List[str]) -> None:
        """Aplikuje bonusy pozycyjne do rekomendacji."""
        for rec in recommendations:
            try:
                col_idx = columns.index(rec.column)
                # Bonus za ostatnią pozycję
                if col_idx == len(columns) - 1:
                    rec.confidence += 0.1
                # Malus za pierwszą pozycję (często ID)
                elif col_idx == 0:
                    rec.confidence -= 0.1
                
                rec.confidence = max(0.0, min(1.0, rec.confidence))
            except ValueError:
                continue


def format_target_explanation(recommendation: TargetRecommendation) -> str:
    """Formatuje wyjaśnienie wyboru targetu dla UI."""
    if not recommendation:
        return "Brak rekomendacji targetu."
    
    explanation = f"""
🎯 **Rekomendowany target: `{recommendation.column}`**

**Powód wyboru:** {recommendation.reason}

**Confidence:** {recommendation.confidence:.1%} | **Jakość:** {recommendation.quality_score:.1%} | **Typ problemu:** {recommendation.problem_type}

**Szczegóły analizy:**
"""
    
    # Dodaj evidence
    if 'nunique' in recommendation.evidence:
        explanation += f"• Unikalnych wartości: {recommendation.evidence['nunique']}\n"
    
    if 'diversity_ratio' in recommendation.evidence:
        explanation += f"• Różnorodność: {recommendation.evidence['diversity_ratio']:.1%}\n"
    
    if 'class_distribution' in recommendation.evidence:
        dist = recommendation.evidence['class_distribution']
        explanation += f"• Rozkład klas: {dict(list(dist.items())[:3])}\n"
    
    # Ostrzeżenia
    if recommendation.warnings:
        explanation += f"\n⚠️ **Ostrzeżenia:**\n"
        for warning in recommendation.warnings:
            explanation += f"• {warning}\n"
    
    # Rekomendacje
    if recommendation.recommendations:
        explanation += f"\n💡 **Rekomendacje:**\n"
        for rec in recommendation.recommendations:
            explanation += f"• {rec}\n"
    
    return explanation


def format_alternatives_list(recommendations: List[TargetRecommendation]) -> str:
    """Formatuje listę alternatywnych targetów."""
    if not recommendations or len(recommendations) <= 1:
        return ""
    
    alternatives = "\n🔄 **Alternatywne opcje:**\n"
    for i, rec in enumerate(recommendations[1:4], 2):  # Top 2-4
        alternatives += f"{i}. `{rec.column}` ({rec.confidence:.1%}) - {rec.reason}\n"
    
    return alternatives