# backend/smart_target_llm.py — Prawdziwy inteligentny wybór targetu z LLM
from __future__ import annotations

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
import streamlit as st

# NAPRAWIONE: Usunięto problematyczny import powodujący circular import
# from backend.smart_target import SmartTargetSelector, TargetRecommendation
from backend.utils import get_openai_key_from_envs


@dataclass
class LLMTargetAnalysis:
    """Analiza targetu przez LLM."""
    recommended_target: str
    confidence: float  # 0-1
    reasoning: str
    problem_type: str  # "regression" | "classification"
    business_context: str
    alternative_targets: List[str]
    warnings: List[str]
    data_insights: str


@dataclass 
class SimpleTargetRecommendation:
    """Prosta rekomendacja targetu dla fallback."""
    column: str
    confidence: float
    reason: str
    problem_type: str


class SimpleFallbackSelector:
    """Uproszczony selektor jako fallback - bez circular import."""
    
    def analyze_and_recommend(self, df: pd.DataFrame) -> List[SimpleTargetRecommendation]:
        """Prosta analiza kolumn."""
        recommendations = []
        
        price_keywords = ['price', 'cost', 'value', 'amount']
        target_keywords = ['target', 'label', 'y', 'outcome']
        
        for col in df.columns:
            score = 0.0
            reason = "Analiza statystyczna"
            
            col_lower = col.lower()
            
            # Sprawdź keywords
            for keyword in price_keywords:
                if keyword in col_lower:
                    score = 0.9
                    reason = f"Nazwa zawiera '{keyword}' (prawdopodobnie wartość do predykcji)"
                    break
            
            for keyword in target_keywords:
                if keyword in col_lower:
                    score = 0.8
                    reason = f"Nazwa zawiera '{keyword}' (klasyczny target)"
                    break
            
            # Sprawdź pozycję (ostatnia kolumna)
            if list(df.columns).index(col) == len(df.columns) - 1:
                score += 0.2
                if not reason.startswith("Nazwa"):
                    reason = "Ostatnia kolumna (częsta konwencja)"
            
            # Sprawdź dane
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                nunique = series.nunique()
                if 2 <= nunique <= 20:
                    score += 0.3
                    problem_type = "classification"
                elif nunique > 20:
                    score += 0.2
                    problem_type = "regression"
                else:
                    problem_type = "classification"
            else:
                problem_type = "classification"
            
            if score > 0.1:
                recommendations.append(SimpleTargetRecommendation(
                    column=col,
                    confidence=min(1.0, score),
                    reason=reason,
                    problem_type=problem_type
                ))
        
        return sorted(recommendations, key=lambda x: x.confidence, reverse=True)


class LLMTargetSelector:
    """Inteligentny selektor targetu wykorzystujący LLM."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or get_openai_key_from_envs()
        self.fallback_selector = SimpleFallbackSelector()  # NAPRAWIONE: Używa lokalnej klasy
    
    def is_available(self) -> bool:
        """Sprawdza czy LLM jest dostępny."""
        return bool(self.api_key)
    
    def analyze_dataset_with_llm(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Optional[LLMTargetAnalysis]:
        """Analizuje dataset przy pomocy LLM i rekomenduje target."""
        if not self.is_available():
            return None
        
        try:
            import openai
            
            # Przygotuj opis danych dla LLM
            data_summary = self._prepare_data_summary(df, dataset_name)
            
            # Wywołaj OpenAI API
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # tańszy model
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": f"Przeanalizuj ten dataset i zarekomenduj najlepszy target:\n\n{data_summary}"
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parsuj odpowiedź
            response_text = response.choices[0].message.content
            return self._parse_llm_response(response_text, df)
            
        except Exception as e:
            st.error(f"Błąd wywołania LLM: {str(e)}")
            return None
    
    def get_hybrid_recommendations(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, Any]:
        """Łączy rekomendacje LLM z heurystyką."""
        results = {
            "llm_available": self.is_available(),
            "llm_analysis": None,
            "heuristic_recommendations": [],
            "final_recommendation": None,
            "confidence_source": "heuristic"
        }
        
        # Zawsze rób analizę heurystyczną jako fallback
        heuristic_recs = self.fallback_selector.analyze_and_recommend(df)
        results["heuristic_recommendations"] = heuristic_recs
        
        # Spróbuj analizy LLM
        if self.is_available():
            llm_analysis = self.analyze_dataset_with_llm(df, dataset_name)
            results["llm_analysis"] = llm_analysis
            
            if llm_analysis:
                # LLM ma priorytet
                results["final_recommendation"] = llm_analysis.recommended_target
                results["confidence_source"] = "llm"
            else:
                # Fallback na heurystykę
                results["final_recommendation"] = heuristic_recs[0].column if heuristic_recs else None
        else:
            # Tylko heurystyka
            results["final_recommendation"] = heuristic_recs[0].column if heuristic_recs else None
        
        return results
    
    def _prepare_data_summary(self, df: pd.DataFrame, dataset_name: str) -> str:
        """Przygotowuje opis danych dla LLM."""
        summary_parts = []
        
        # Podstawowe info
        summary_parts.append(f"DATASET: {dataset_name}")
        summary_parts.append(f"ROZMIAR: {len(df)} wierszy × {len(df.columns)} kolumn")
        summary_parts.append("")
        
        # Kolumny z podstawowymi statystykami
        summary_parts.append("KOLUMNY:")
        for col in df.columns:
            series = df[col]
            col_info = f"- {col} ({series.dtype})"
            
            if pd.api.types.is_numeric_dtype(series):
                col_info += f" | Zakres: {series.min():.2f}-{series.max():.2f}"
                col_info += f" | Średnia: {series.mean():.2f}"
            else:
                col_info += f" | Unikalne: {series.nunique()}"
                if series.nunique() <= 10:
                    unique_vals = list(series.unique()[:10])
                    col_info += f" | Wartości: {unique_vals}"
            
            # Braki
            null_pct = series.isna().mean() * 100
            if null_pct > 0:
                col_info += f" | Braki: {null_pct:.1f}%"
            
            summary_parts.append(col_info)
        
        # Próbka danych
        summary_parts.append("\nPRÓBKA DANYCH (pierwsze 3 wiersze):")
        sample_data = df.head(3).to_string(index=False)
        summary_parts.append(sample_data)
        
        return "\n".join(summary_parts)
    
    def _get_system_prompt(self) -> str:
        """Zwraca system prompt dla LLM."""
        return """Jesteś ekspertem machine learning, który analizuje datasety i rekomenduje najlepszy target (zmienną docelową) do przewidywania.

ZADANIE:
1. Przeanalizuj podany dataset
2. Zarekomenduj najlepszą kolumnę jako target
3. Określ typ problemu (regression/classification)
4. Wyjaśnij swoje uzasadnienie
5. Zaproponuj alternatywne opcje

ODPOWIEDŹ W FORMACIE JSON:
{
    "recommended_target": "nazwa_kolumny",
    "confidence": 0.95,
    "reasoning": "Szczegółowe wyjaśnienie dlaczego ta kolumna to najlepszy target",
    "problem_type": "regression",
    "business_context": "Kontekst biznesowy - co będziemy przewidywać i dlaczego to wartościowe",
    "alternative_targets": ["kolumna1", "kolumna2"],
    "warnings": ["Ostrzeżenie jeśli jakieś są"],
    "data_insights": "Dodatkowe spostrzeżenia o danych"
}

KRYTERIA WYBORU TARGET:
- Wartość biznesowa (co ma sens przewidywać?)
- Jakość danych (brak braków, odpowiednia dystrybucja)
- Przewidywalność (czy inne cechy mogą to przewidzieć?)
- Typ problemu (regression dla ciągłych, classification dla kategorycznych)

UNIKAJ:
- ID, index, timestamp jako target
- Kolumn z bardzo wysoką liczbą unikalnych wartości (dla classification)
- Kolumn z brakami >20%"""
    
    def _parse_llm_response(self, response_text: str, df: pd.DataFrame) -> Optional[LLMTargetAnalysis]:
        """Parsuje odpowiedź LLM do struktury."""
        try:
            # Wyciągnij JSON z odpowiedzi
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end <= json_start:
                return None
            
            json_text = response_text[json_start:json_end]
            parsed = json.loads(json_text)
            
            # Waliduj czy target istnieje
            recommended_target = parsed.get("recommended_target", "")
            if recommended_target not in df.columns:
                return None
            
            return LLMTargetAnalysis(
                recommended_target=recommended_target,
                confidence=float(parsed.get("confidence", 0.5)),
                reasoning=parsed.get("reasoning", ""),
                problem_type=parsed.get("problem_type", "unknown"),
                business_context=parsed.get("business_context", ""),
                alternative_targets=parsed.get("alternative_targets", []),
                warnings=parsed.get("warnings", []),
                data_insights=parsed.get("data_insights", "")
            )
            
        except Exception as e:
            st.error(f"Błąd parsowania odpowiedzi LLM: {str(e)}")
            return None


def render_openai_config():
    """Renderuje konfigurację klucza OpenAI."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Konfiguracja AI")
    
    # Sprawdź obecny klucz
    current_key = get_openai_key_from_envs()
    has_key = bool(current_key)
    
    # Status
    if has_key:
        st.sidebar.success("✅ Klucz OpenAI: aktywny")
        masked_key = current_key[:8] + "..." + current_key[-4:] if len(current_key) > 12 else "***"
        st.sidebar.caption(f"Klucz: {masked_key}")
    else:
        st.sidebar.error("❌ Brak klucza OpenAI")
        st.sidebar.caption("Tylko analiza heurystyczna")
    
    # Opcje konfiguracji
    with st.sidebar.expander("⚙️ Konfiguracja klucza", expanded=not has_key):
        st.write("**Opcje ustawienia klucza OpenAI:**")
        
        # Opcja 1: Zmienna środowiskowa
        st.write("**1. Zmienna środowiskowa:**")
        st.code("""
export OPENAI_API_KEY="sk-..."
# lub w .env:
OPENAI_API_KEY=sk-...
        """)
        
        # Opcja 2: Session state (tymczasowe) - NAPRAWIONE: bez zapętlenia
        st.write("**2. Tymczasowy klucz (tylko ta sesja):**")
        temp_key = st.text_input(
            "Klucz OpenAI:",
            type="password",
            placeholder="sk-...",
            help="Klucz jest używany tylko w tej sesji",
            key="temp_openai_input"
        )
        
        # NAPRAWIONE: Przycisk zamiast automatycznego ustawiania
        if st.button("🔑 Ustaw klucz", disabled=not (temp_key and temp_key.startswith("sk-"))):
            if temp_key and temp_key.startswith("sk-"):
                # Ustaw w session state i environment
                st.session_state.temp_openai_key = temp_key
                os.environ["OPENAI_API_KEY"] = temp_key
                st.success("✅ Klucz ustawiony tymczasowo")
                # USUNIĘTO st.rerun() - nie potrzebne, sidebar się odświeży automatycznie
        
        # Przycisk wyczyść klucz
        if has_key and st.button("🗑️ Wyczyść klucz"):
            if "temp_openai_key" in st.session_state:
                del st.session_state.temp_openai_key
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            st.info("Klucz wyczyszczony")
        
        # Opcja 3: Info o kluczu
        st.write("**3. Jak zdobyć klucz:**")
        st.markdown("""
        1. Idź na [platform.openai.com](https://platform.openai.com)
        2. Zarejestruj się / zaloguj
        3. API Keys → Create new secret key
        4. Skopiuj klucz (sk-...)
        """)
    
    return has_key


def render_smart_target_section_with_llm(df: pd.DataFrame, dataset_name: str = "dataset"):
    """Renderuje sekcję wyboru targetu z LLM."""
    st.header("🎯 Inteligentny wybór targetu")
    
    # Sprawdź dostępność LLM
    llm_selector = LLMTargetSelector()
    has_llm = llm_selector.is_available()
    
    # Info o trybie
    if has_llm:
        st.info("🤖 **Tryb AI**: Używam GPT-4 do analizy danych i rekomendacji targetu")
    else:
        st.warning("🔧 **Tryb heurystyczny**: Brak klucza OpenAI - używam analizy statystycznej")
        st.caption("💡 Skonfiguruj klucz OpenAI w sidebar dla prawdziwej inteligencji AI")
    
    # Przycisk analizy
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analyze_btn = st.button(
            f"🔍 {'Analizuj z AI' if has_llm else 'Analizuj heurystycznie'}", 
            type="primary"
        )
    
    with col2:
        manual_mode = st.checkbox("✋ Wybór ręczny", help="Pomiń automatyczną analizę")
    
    # Analiza lub wybór ręczny
    if manual_mode:
        st.subheader("✋ Ręczny wybór targetu")
        st.info("Wybierz target ręcznie bez automatycznych rekomendacji")
        
        selected_target = st.selectbox(
            "Wybierz kolumnę targetu:",
            df.columns,
            help="Kolumna którą model będzie przewidywał"
        )
        
        if st.button("✅ Zatwierdź wybór"):
            return selected_target
            
    elif analyze_btn or 'target_analysis_done' in st.session_state:
        # Wykonaj analizę
        if analyze_btn:
            with st.spinner(f"{'🤖 Analizuję dane z AI...' if has_llm else '🔧 Analizuję dane...'}"):
                recommendations = llm_selector.get_hybrid_recommendations(df, dataset_name)
                st.session_state.target_analysis = recommendations
                st.session_state.target_analysis_done = True
        
        # Pokaż wyniki
        if 'target_analysis' in st.session_state:
            analysis = st.session_state.target_analysis
            
            # LLM analiza (jeśli dostępna)
            if analysis.get("llm_analysis"):
                llm_result = analysis["llm_analysis"]
                
                st.success(f"🤖 **Rekomendacja AI**: `{llm_result.recommended_target}`")
                
                # Szczegóły analizy
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Uzasadnienie AI:**")
                    st.write(llm_result.reasoning)
                    
                    if llm_result.business_context:
                        st.write("**Kontekst biznesowy:**")
                        st.write(llm_result.business_context)
                
                with col2:
                    st.metric("Confidence", f"{llm_result.confidence:.1%}")
                    st.metric("Typ problemu", llm_result.problem_type)
                
                # Dodatkowe insights
                if llm_result.data_insights:
                    with st.expander("💡 Dodatkowe spostrzeżenia AI"):
                        st.write(llm_result.data_insights)
                
                # Ostrzeżenia
                if llm_result.warnings:
                    st.subheader("⚠️ Ostrzeżenia AI")
                    for warning in llm_result.warnings:
                        st.warning(warning)
                
                # Alternatywy
                if llm_result.alternative_targets:
                    with st.expander("🔄 Alternatywne opcje od AI"):
                        for alt in llm_result.alternative_targets:
                            if alt in df.columns:
                                st.write(f"• {alt}")
                
                recommended_target = llm_result.recommended_target
            
            else:
                # Fallback na heurystykę
                heuristic_recs = analysis.get("heuristic_recommendations", [])
                if heuristic_recs:
                    best_rec = heuristic_recs[0]
                    st.success(f"🔧 **Rekomendacja heurystyczna**: `{best_rec.column}`")
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Powód**: {best_rec.reason}")
                    with col2:
                        st.metric("Confidence", f"{best_rec.confidence:.1%}")
                    
                    recommended_target = best_rec.column
                else:
                    st.error("Nie udało się znaleźć rekomendacji targetu")
                    recommended_target = df.columns[0]
            
            # Finalna sekcja wyboru
            st.subheader("📋 Finalizuj wybór targetu")
            
            # Domyślnie rekomendowany target
            try:
                default_idx = list(df.columns).index(recommended_target)
            except ValueError:
                default_idx = 0
            
            final_target = st.selectbox(
                "Zatwierdź lub zmień target:",
                df.columns,
                index=default_idx,
                help="Możesz wybrać inny target niż rekomendowany"
            )
            
            if st.button("✅ Zatwierdź target", type="primary"):
                # Wyczyść cache analizy
                if 'target_analysis' in st.session_state:
                    del st.session_state.target_analysis
                if 'target_analysis_done' in st.session_state:
                    del st.session_state.target_analysis_done
                
                return final_target
    
    else:
        st.info("👆 Kliknij przycisk analizy aby otrzymać rekomendacje targetu")
    
    return None