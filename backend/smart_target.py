# backend/smart_target.py — Podstawowy inteligentny wybór targetu (bez LLM)
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re
from collections import Counter

from backend.utils import infer_problem_type


class SmartTargetSelector:
    """
    Inteligentny selektor zmiennej docelowej bez użycia LLM.
    Używa heurystyk i algorytmów scoringowych do rekomendacji targetu.
    """
    
    def __init__(self):
        # Słowa kluczowe sugerujące target (w kolejności priorytetu)
        self.target_keywords = {
            'price': 10.0,      # Największy priorytet
            'cost': 9.0,
            'value': 8.5,
            'amount': 8.0,
            'total': 7.5,
            'sum': 7.0,
            'revenue': 9.5,
            'income': 9.0,
            'profit': 9.5,
            'loss': 8.0,
            'sales': 8.5,
            'target': 10.0,
            'goal': 8.0,
            'outcome': 7.5,
            'result': 7.0,
            'score': 6.5,
            'rating': 6.0,
            'rank': 6.0,
            'class': 5.5,
            'category': 5.0,
            'type': 4.5,
            'label': 5.5,
            'status': 5.0,
            'success': 6.5,
            'failure': 6.0,
            'win': 5.5,
            'lose': 5.5,
        }
        
        # Wzorce regexowe dla nazw kolumn
        self.regex_patterns = {
            r'.*price.*': 10.0,
            r'.*cost.*': 9.0,
            r'.*value.*': 8.5,
            r'.*amount.*': 8.0,
            r'.*target.*': 10.0,
            r'.*label.*': 5.5,
            r'.*class.*': 5.5,
            r'.*category.*': 5.0,
            r'.*outcome.*': 7.5,
            r'.*result.*': 7.0,
            r'.*score.*': 6.5,
            r'.*rating.*': 6.0,
            r'.*total.*': 7.5,
            r'.*sum.*': 7.0,
            r'.*revenue.*': 9.5,
            r'.*profit.*': 9.5,
            r'.*sales.*': 8.5,
            r'.*income.*': 9.0,
        }
    
    def recommend_targets(self, df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Rekomenduje najlepsze kolumny jako potencjalne targety.
        
        Args:
            df: DataFrame do analizy
            top_n: Liczba rekomendacji do zwrócenia
        
        Returns:
            Lista słowników z rekomendacjami posortowane według score
        """
        recommendations = []
        
        for column in df.columns:
            score_info = self._score_column(df, column)
            if score_info['score'] > 0:
                recommendations.append({
                    'column': column,
                    'score': score_info['score'],
                    'problem_type': score_info['problem_type'],
                    'explanation': score_info['explanation'],
                    'details': score_info['details']
                })
        
        # Sortuj według score (malejąco)
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:top_n]
    
    def _score_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Ocenia kolumnę jako potencjalny target.
        
        Returns:
            Dict z score, problem_type, explanation i details
        """
        series = df[column]
        total_score = 0.0
        explanations = []
        details = {}
        
        # 1. KEYWORD SCORING - najważniejszy czynnik
        keyword_score = self._score_by_keywords(column)
        total_score += keyword_score * 3.0  # Waga x3
        if keyword_score > 0:
            explanations.append(f"nazwa zawiera kluczowe słowa (+{keyword_score:.1f})")
        details['keyword_score'] = keyword_score
        
        # 2. REGEX PATTERN SCORING
        regex_score = self._score_by_regex(column)
        total_score += regex_score * 2.0  # Waga x2
        if regex_score > 0:
            explanations.append(f"pasuje do wzorców target (+{regex_score:.1f})")
        details['regex_score'] = regex_score
        
        # 3. DATA TYPE SCORING
        dtype_score = self._score_by_data_type(series)
        total_score += dtype_score
        if dtype_score > 0:
            explanations.append(f"odpowiedni typ danych (+{dtype_score:.1f})")
        details['dtype_score'] = dtype_score
        
        # 4. DISTRIBUTION SCORING
        dist_score = self._score_by_distribution(series)
        total_score += dist_score
        if dist_score > 0:
            explanations.append(f"dobry rozkład danych (+{dist_score:.1f})")
        details['distribution_score'] = dist_score
        
        # 5. CARDINALITY SCORING
        card_score = self._score_by_cardinality(series)
        total_score += card_score
        if card_score > 0:
            explanations.append(f"optymalna kardynalność (+{card_score:.1f})")
        details['cardinality_score'] = card_score
        
        # 6. MISSING VALUES PENALTY
        missing_penalty = self._penalty_missing_values(series)
        total_score -= missing_penalty
        if missing_penalty > 0:
            explanations.append(f"kara za braki danych (-{missing_penalty:.1f})")
        details['missing_penalty'] = missing_penalty
        
        # 7. POSITION BONUS (kolumny na końcu często to targety)
        position_bonus = self._bonus_column_position(df, column)
        total_score += position_bonus
        if position_bonus > 0:
            explanations.append(f"korzystna pozycja w tabeli (+{position_bonus:.1f})")
        details['position_bonus'] = position_bonus
        
        # Określ typ problemu
        problem_type = infer_problem_type(series)
        
        # Finalne dostosowanie score
        total_score = max(0.0, total_score)  # Nie może być ujemny
        
        # Główne wyjaśnienie
        main_explanation = self._generate_main_explanation(column, series, problem_type, total_score)
        
        return {
            'score': total_score,
            'problem_type': problem_type,
            'explanation': main_explanation,
            'details': details
        }
    
    def _score_by_keywords(self, column: str) -> float:
        """Ocenia kolumnę na podstawie słów kluczowych w nazwie."""
        column_lower = column.lower()
        max_score = 0.0
        
        for keyword, score in self.target_keywords.items():
            if keyword in column_lower:
                max_score = max(max_score, score)
        
        return max_score
    
    def _score_by_regex(self, column: str) -> float:
        """Ocenia kolumnę na podstawie wzorców regex."""
        column_lower = column.lower()
        max_score = 0.0
        
        for pattern, score in self.regex_patterns.items():
            if re.match(pattern, column_lower, re.IGNORECASE):
                max_score = max(max_score, score)
        
        return max_score
    
    def _score_by_data_type(self, series: pd.Series) -> float:
        """Ocenia typ danych kolumny."""
        if pd.api.types.is_numeric_dtype(series):
            # Numeryczne dane są dobre dla regresji
            if pd.api.types.is_integer_dtype(series):
                # Integer może być klasyfikacją lub regresją
                unique_ratio = series.nunique() / len(series)
                if unique_ratio < 0.1:  # Mało unikalnych wartości = klasyfikacja
                    return 4.0
                else:  # Dużo unikalnych = regresja
                    return 5.0
            else:  # Float - prawdopodobnie regresja
                return 6.0
        elif pd.api.types.is_categorical_dtype(series) or series.dtype == 'object':
            # Kategoryczne dobre dla klasyfikacji
            return 4.0
        elif pd.api.types.is_bool_dtype(series):
            # Boolean perfect dla binarnej klasyfikacji
            return 5.0
        else:
            # Inne typy (datetime, etc.) - mniej prawdopodobne
            return 1.0
    
    def _score_by_distribution(self, series: pd.Series) -> float:
        """Ocenia rozkład wartości w kolumnie."""
        if len(series) == 0:
            return 0.0
        
        # Usuń NaN dla analizy
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return 0.0
        
        score = 0.0
        
        if pd.api.types.is_numeric_dtype(clean_series):
            # Dla danych numerycznych
            
            # Bonus za zróżnicowanie (nie wszystkie wartości takie same)
            if clean_series.nunique() > 1:
                score += 1.0
            
            # Bonus za rozsądny zakres wartości
            if not (clean_series.var() == 0):  # Nie wszystkie takie same
                score += 1.0
            
            # Sprawdź czy są outlier (może wskazywać na ważną zmienną)
            if len(clean_series) > 10:  # Tylko dla większych próbek
                q1 = clean_series.quantile(0.25)
                q3 = clean_series.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:  # Są quartile
                    outliers = clean_series[(clean_series < q1 - 1.5 * iqr) | 
                                          (clean_series > q3 + 1.5 * iqr)]
                    outlier_ratio = len(outliers) / len(clean_series)
                    if 0.01 < outlier_ratio < 0.1:  # Umiarkowana liczba outlierów
                        score += 0.5
        
        else:
            # Dla danych kategorycznych
            
            # Bonus za zbalansowane klasy
            value_counts = clean_series.value_counts()
            if len(value_counts) > 1:
                # Sprawdź balans klas
                class_ratios = value_counts / len(clean_series)
                min_ratio = class_ratios.min()
                max_ratio = class_ratios.max()
                
                # Im bardziej zbalansowane, tym lepiej
                balance_score = 1.0 - (max_ratio - min_ratio)
                score += max(0.0, balance_score * 2.0)
        
        return score
    
    def _score_by_cardinality(self, series: pd.Series) -> float:
        """Ocenia kardynalność (liczbę unikalnych wartości)."""
        if len(series) == 0:
            return 0.0
        
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return 0.0
        
        unique_count = clean_series.nunique()
        total_count = len(clean_series)
        unique_ratio = unique_count / total_count
        
        score = 0.0
        
        if pd.api.types.is_numeric_dtype(clean_series):
            # Dla regresji - im więcej unikalnych wartości, tym lepiej (do pewnego stopnia)
            if unique_ratio > 0.5:  # Dużo unikalnych wartości
                score += 2.0
            elif unique_ratio > 0.1:  # Umiarkowanie unikalnych
                score += 1.0
        
        else:
            # Dla klasyfikacji - optymalna liczba klas
            if 2 <= unique_count <= 10:  # Idealna liczba klas
                score += 2.0
            elif unique_count == 2:  # Binarna klasyfikacja
                score += 2.5
            elif 11 <= unique_count <= 50:  # Akceptowalna liczba klas
                score += 1.0
            elif unique_count > 100:  # Za dużo klas
                score -= 1.0
        
        return score
    
    def _penalty_missing_values(self, series: pd.Series) -> float:
        """Kara za braki danych."""
        if len(series) == 0:
            return 0.0
        
        missing_ratio = series.isna().sum() / len(series)
        
        if missing_ratio == 0:
            return 0.0  # Brak kary
        elif missing_ratio < 0.05:  # < 5% braków
            return 0.5
        elif missing_ratio < 0.1:   # 5-10% braków
            return 1.0
        elif missing_ratio < 0.2:   # 10-20% braków
            return 2.0
        else:  # > 20% braków
            return 3.0
    
    def _bonus_column_position(self, df: pd.DataFrame, column: str) -> float:
        """Bonus za pozycję kolumny (ostatnie kolumny często to targety)."""
        try:
            column_index = df.columns.get_loc(column)
            total_columns = len(df.columns)
            
            # Ostatnie 25% kolumn dostaje bonus
            if column_index >= total_columns * 0.75:
                return 1.0
            # Środkowe kolumny - neutralne
            elif column_index >= total_columns * 0.25:
                return 0.0
            # Pierwsze kolumny - lekka kara
            else:
                return -0.5
        except Exception:
            return 0.0
    
    def _generate_main_explanation(self, column: str, series: pd.Series, problem_type: str, score: float) -> str:
        """Generuje główne wyjaśnienie dla rekomendacji."""
        explanations = []
        
        # Nazwa kolumny
        column_lower = column.lower()
        if any(keyword in column_lower for keyword in ['price', 'cost', 'value', 'amount']):
            explanations.append("nazwa sugeruje wartość cenową")
        elif any(keyword in column_lower for keyword in ['target', 'label', 'class']):
            explanations.append("nazwa bezpośrednio wskazuje na zmienną docelową")
        elif any(keyword in column_lower for keyword in ['outcome', 'result', 'score']):
            explanations.append("nazwa sugeruje wynik/rezultat")
        
        # Typ problemu
        if problem_type == 'regression':
            explanations.append("odpowiednia dla regresji")
        elif problem_type == 'classification':
            explanations.append("odpowiednia dla klasyfikacji")
        
        # Jakość danych
        missing_ratio = series.isna().sum() / len(series) if len(series) > 0 else 0
        if missing_ratio < 0.05:
            explanations.append("wysoka jakość danych")
        elif missing_ratio < 0.2:
            explanations.append("akceptowalna jakość danych")
        
        # Kardynalność
        if len(series) > 0:
            unique_ratio = series.nunique() / len(series.dropna()) if len(series.dropna()) > 0 else 0
            if problem_type == 'regression' and unique_ratio > 0.5:
                explanations.append("bogaty zakres wartości")
            elif problem_type == 'classification' and 2 <= series.nunique() <= 10:
                explanations.append("optymalna liczba klas")
        
        if not explanations:
            explanations.append("podstawowe kryteria spełnione")
        
        return f"Kolumna '{column}' - " + ", ".join(explanations) + f" (score: {score:.1f})"


def format_target_explanation(recommendation: Dict[str, Any]) -> str:
    """
    Formatuje wyjaśnienie rekomendacji targetu dla UI.
    
    Args:
        recommendation: Dict z rekomendacją z recommend_targets()
    
    Returns:
        Sformatowany string z wyjaśnieniem
    """
    column = recommendation['column']
    score = recommendation['score']
    problem_type = recommendation['problem_type']
    explanation = recommendation['explanation']
    details = recommendation.get('details', {})
    
    # Główny opis
    formatted = f"**{column}** (score: {score:.2f})\n\n"
    formatted += f"📊 **Typ problemu:** {problem_type.title()}\n"
    formatted += f"💡 **Wyjaśnienie:** {explanation}\n\n"
    
    # Szczegóły scoringu
    if details:
        formatted += "**Szczegóły oceny:**\n"
        if details.get('keyword_score', 0) > 0:
            formatted += f"• Słowa kluczowe: +{details['keyword_score']:.1f}\n"
        if details.get('regex_score', 0) > 0:
            formatted += f"• Wzorce nazwy: +{details['regex_score']:.1f}\n"
        if details.get('dtype_score', 0) > 0:
            formatted += f"• Typ danych: +{details['dtype_score']:.1f}\n"
        if details.get('distribution_score', 0) > 0:
            formatted += f"• Rozkład: +{details['distribution_score']:.1f}\n"
        if details.get('cardinality_score', 0) > 0:
            formatted += f"• Kardynalność: +{details['cardinality_score']:.1f}\n"
        if details.get('missing_penalty', 0) > 0:
            formatted += f"• Kara za braki: -{details['missing_penalty']:.1f}\n"
        if details.get('position_bonus', 0) > 0:
            formatted += f"• Pozycja: +{details['position_bonus']:.1f}\n"
    
    return formatted


def format_alternatives_list(recommendations: List[Dict[str, Any]], exclude_top: bool = True) -> str:
    """
    Formatuje listę alternatywnych rekomendacji.
    
    Args:
        recommendations: Lista rekomendacji z recommend_targets()
        exclude_top: Czy pominąć pierwszą (najlepszą) rekomendację
    
    Returns:
        Sformatowany string z alternatywami
    """
    if not recommendations:
        return "Brak alternatywnych rekomendacji."
    
    start_idx = 1 if exclude_top else 0
    alternatives = recommendations[start_idx:start_idx+4]  # Max 4 alternatywy
    
    if not alternatives:
        return "Brak alternatywnych rekomendacji."
    
    formatted = "**Alternatywne opcje:**\n\n"
    
    for i, rec in enumerate(alternatives, 1):
        formatted += f"{i}. **{rec['column']}** "
        formatted += f"(score: {rec['score']:.2f}, {rec['problem_type']})\n"
        # Skrócone wyjaśnienie
        explanation = rec['explanation'].split('(score:')[0].strip()
        if len(explanation) > 80:
            explanation = explanation[:77] + "..."
        formatted += f"   {explanation}\n\n"
    
    return formatted