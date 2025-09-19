# backend/smart_target_llm.py — NAPRAWIONY: automatyczne odświeżanie po ustawieniu klucza
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import pandas as pd
import streamlit as st

from backend.utils import (
    get_openai_key_from_envs,
    set_openai_key_temp,
    clear_openai_key,
    infer_problem_type,
)


# ==============================
# Modele danych
# ==============================
@dataclass
class LLMTargetAnalysis:
    """Analiza targetu zwrócona przez LLM."""
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
    """Uproszczona rekomendacja targetu w trybie fallback (bez LLM)."""
    column: str
    confidence: float
    reason: str
    problem_type: str


# ==============================
# Heurystyczny fallback (bez LLM)
# ==============================
class SimpleFallbackSelector:
    """Uproszczony selektor jako fallback – szybki i bez zależności."""

    def analyze_and_recommend(self, df: pd.DataFrame, top_k: int = 5) -> List[SimpleTargetRecommendation]:
        recs: List[SimpleTargetRecommendation] = []

        price_keywords = ["price", "cost", "value", "amount", "revenue", "sales", "total", "avg", "mean", "average"]
        target_keywords = ["target", "label", "y", "outcome", "result", "class", "category", "response", "output"]

        for col in df.columns:
            s = df[col]
            name = str(col).lower()
            score = 0.0
            why = "Analiza statystyczna i nazwy kolumny"
            problem = "classification"

            if any(k in name for k in price_keywords):
                score += 0.6
                why = f"Nazwa zawiera słowo typu wartość/cena ('{[k for k in price_keywords if k in name][0]}')."

            if any(k in name for k in target_keywords):
                score += 0.5
                why = f"Nazwa sugeruje klasyczny target ('{[k for k in target_keywords if k in name][0]}')."

            # pozycja kolumny – ostatnia kolumna często jest targetem
            if list(df.columns).index(col) == len(df.columns) - 1:
                score += 0.25
                if "Nazwa" not in why:
                    why = "Ostatnia kolumna (częsta konwencja)."

            # charakterystyka danych
            if pd.api.types.is_numeric_dtype(s):
                nunq = s.nunique(dropna=True)
                if 2 <= nunq <= 20:
                    score += 0.2
                    problem = "classification"
                elif nunq > 20:
                    score += 0.3
                    problem = "regression"
            else:
                nunq = s.nunique(dropna=True)
                if 2 <= nunq <= 50:
                    score += 0.15
                    problem = "classification"

            # penalty za braki danych
            missing_ratio = s.isna().mean()
            if missing_ratio > 0.3:
                score *= 0.7
                why += f" (Uwaga: {missing_ratio:.1%} braków danych)"

            if score > 0.15:
                recs.append(SimpleTargetRecommendation(
                    column=col,
                    confidence=min(1.0, score),
                    reason=why,
                    problem_type=problem
                ))

        return sorted(recs, key=lambda r: r.confidence, reverse=True)[:top_k]


# ==============================
# LLM Target Selector - NAPRAWIONY
# ==============================
class LLMTargetSelector:
    """Inteligentny selektor targetu z opcjonalnym wsparciem LLM (OpenAI)."""

    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or get_openai_key_from_envs()
        self._openai_available: Optional[bool] = None
        self.fallback_selector = SimpleFallbackSelector()

    # --- dostępność LLM ---
    def is_available(self) -> bool:
        """NAPRAWIONA: sprawdza na bieżąco dostępność klucza i biblioteki."""
        # Zawsze sprawdzaj klucz na nowo (może się zmienić w trakcie sesji)
        current_key = get_openai_key_from_envs()
        
        if not current_key or not current_key.startswith("sk-"):
            self._openai_available = False
            return False
            
        try:
            import openai  # noqa: F401
            self._openai_available = True
            self.api_key = current_key  # aktualizuj klucz
            return True
        except Exception:
            self._openai_available = False
            return False

    # --- wywołanie LLM ---
    def analyze_dataset_with_llm(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Optional[LLMTargetAnalysis]:
        """Analiza datasetu przez LLM. Zwraca LLMTargetAnalysis albo None (gdy błąd/brak klucza)."""
        if not self.is_available():
            return None

        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)

            data_summary = self._prepare_data_summary(df, dataset_name)

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": f"Przeanalizuj dataset i zarekomenduj target.\n\n{data_summary}"}
                ],
                temperature=0.1,
                max_tokens=1200,
            )

            text = resp.choices[0].message.content
            return self._parse_llm_response(text, df)

        except ImportError:
            # ciche wyłączenie LLM
            return None
        except Exception as e:
            # tylko błędy API/parsingu pokazujemy
            if "No module named" not in str(e):
                st.error(f"Błąd wywołania OpenAI: {e}")
            return None

    # --- strategia hybrydowa ---
    def get_hybrid_recommendations(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "llm_available": self.is_available(),
            "llm_analysis": None,
            "heuristic_recommendations": [],
            "final_recommendation": None,
            "confidence_source": "heuristic",
        }

        # heurystyka zawsze
        heur = self.fallback_selector.analyze_and_recommend(df)
        results["heuristic_recommendations"] = heur

        # LLM – jeśli dostępny
        if results["llm_available"]:
            with st.spinner("🤖 Analizuję dane z pomocą AI..."):
                llm = self.analyze_dataset_with_llm(df, dataset_name)
            
            results["llm_analysis"] = llm
            if llm:
                results["final_recommendation"] = llm.recommended_target
                results["confidence_source"] = "llm"
            else:
                results["final_recommendation"] = heur[0].column if heur else None
        else:
            results["final_recommendation"] = heur[0].column if heur else None

        return results

    # --- pomocnicze ---
    def _prepare_data_summary(self, df: pd.DataFrame, dataset_name: str) -> str:
        lines: List[str] = []
        lines.append(f"DATASET: {dataset_name}")
        lines.append(f"ROZMIAR: {len(df)} wierszy × {len(df.columns)} kolumn\n")
        lines.append("KOLUMNY:")

        for col in df.columns:
            s = df[col]
            info = f"- {col} ({s.dtype})"
            if pd.api.types.is_numeric_dtype(s):
                try:
                    non_null = s.dropna()
                    if len(non_null) > 0:
                        info += f" | Zakres: {non_null.min():.3f}-{non_null.max():.3f}"
                        info += f" | Średnia: {non_null.mean():.3f}"
                except Exception:
                    pass
            else:
                info += f" | Unikalne: {int(s.nunique(dropna=True))}"
                if s.nunique(dropna=True) <= 10:
                    try:
                        vals = [str(v) for v in s.dropna().unique()[:10]]
                        info += f" | Wartości: {vals}"
                    except Exception:
                        pass
            null_pct = float(s.isna().mean() * 100.0)
            if null_pct > 0:
                info += f" | Braki: {null_pct:.1f}%"
            lines.append(info)

        lines.append("\nPRÓBKA DANYCH (pierwsze 3 wiersze):")
        try:
            lines.append(df.head(3).to_string(index=False))
        except Exception:
            pass

        return "\n".join(lines)

    def _system_prompt(self) -> str:
        return """Jesteś ekspertem ML. Masz wskazać najlepszą kolumnę jako target (zmienną docelową).
Oceń: wartość biznesową, jakość danych, przewidywalność oraz typ problemu (regression/classification).

ZASADY:
- Preferuj kolumny o wartości biznesowej (ceny, sprzedaż, wyniki)
- Unikaj ID/index/timestamp/date jako target
- Dla klasyfikacji: 2-20 klas to optymalnie
- Dla regresji: wysoka zmienność wartości
- Zwracaj TYLKO poprawny JSON bez dodatkowych komentarzy

Zwróć **wyłącznie** JSON w tym schemacie:
{
  "recommended_target": "kolumna",
  "confidence": 0.90,
  "reasoning": "konkretne, praktyczne uzasadnienie dlaczego ta kolumna",
  "problem_type": "regression",
  "business_context": "co przewidujemy i jaka korzyść biznesowa",
  "alternative_targets": ["kolumna1", "kolumna2"],
  "warnings": ["opcjonalne ostrzeżenia o jakości danych"],
  "data_insights": "kluczowe spostrzeżenia o strukturze danych"
}"""

    def _parse_llm_response(self, text: str, df: pd.DataFrame) -> Optional[LLMTargetAnalysis]:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end <= start:
                return None
            
            data = json.loads(text[start:end])

            tgt = data.get("recommended_target", "")
            if tgt not in df.columns:
                return None

            return LLMTargetAnalysis(
                recommended_target=tgt,
                confidence=float(data.get("confidence", 0.6)),
                reasoning=str(data.get("reasoning", "")),
                problem_type=str(data.get("problem_type", infer_problem_type(df, tgt))),
                business_context=str(data.get("business_context", "")),
                alternative_targets=list(data.get("alternative_targets", [])),
                warnings=list(data.get("warnings", [])),
                data_insights=str(data.get("data_insights", "")),
            )
        except Exception as e:
            st.error(f"Błąd parsowania odpowiedzi AI: {e}")
            return None


# ==============================
# UI: konfiguracja OpenAI - NAPRAWIONA
# ==============================
def render_openai_config():
    """NAPRAWIONA wersja z automatycznym odświeżaniem statusu."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Konfiguracja AI")

    # Status klucza - sprawdzany na bieżąco
    current_key = get_openai_key_from_envs()
    
    # Kontener na status (będzie się odświeżał)
    status_container = st.sidebar.container()
    
    with status_container:
        if current_key and current_key.startswith("sk-"):
            masked = current_key[:8] + "..." + current_key[-4:] if len(current_key) > 12 else "****"
            st.success("✅ Klucz OpenAI aktywny")
            st.caption(f"🔑 {masked}")
            
            # Test dostępności biblioteki
            try:
                import openai  # noqa: F401
                st.caption("📚 Biblioteka: openai ✅")
            except ImportError:
                st.warning("📚 Biblioteka: openai ❌")
                st.caption("Zainstaluj: `pip install openai`")
        else:
            st.error("❌ Brak klucza OpenAI")
            st.caption("Ustaw klucz aby używać funkcji AI")

    # Sekcja ustawiania klucza
    with st.sidebar.expander("🔑 Zarządzanie kluczem", expanded=not bool(current_key)):
        # Input dla klucza
        temp_key = st.text_input(
            "Wklej klucz OpenAI (sk-...):",
            type="password",
            key="openai_key_input",
            help="Klucz używany tylko w tej sesji (nie zapisujemy na stałe)",
            placeholder="sk-proj-..."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔑 Ustaw klucz", type="primary", disabled=not temp_key):
                if temp_key and temp_key.strip():
                    if set_openai_key_temp(temp_key.strip()):
                        st.success("✅ Klucz ustawiony!")
                        # Wymuś ponowne renderowanie przez rerun
                        st.rerun()
                    else:
                        st.error("❌ Nieprawidłowy format klucza")
                else:
                    st.warning("⚠️ Wprowadź prawidłowy klucz")
        
        with col2:
            if st.button("🗑️ Wyczyść"):
                clear_openai_key()
                # Wyczyść też input
                if "openai_key_input" in st.session_state:
                    st.session_state["openai_key_input"] = ""
                st.info("🧹 Klucz wyczyszczony")
                st.rerun()

    # Informacje pomocnicze
    with st.sidebar.expander("ℹ️ Informacje o kluczu", expanded=False):
        st.write("**Kolejność sprawdzania:**")
        st.write("1. Klucz z tej sesji")
        st.write("2. Streamlit secrets")  
        st.write("3. Plik .env")
        st.write("4. Zmienne środowiskowe")
        
        st.write("**Wymagany format:**")
        st.code("sk-proj-... (OpenAI API key)")
        
        st.write("**Gdzie uzyskać klucz:**")
        st.write("[platform.openai.com/api-keys](https://platform.openai.com/api-keys)")


# ==============================
# UI: sekcja wyboru targetu - NAPRAWIONA
# ==============================
def render_smart_target_section_with_llm(df: pd.DataFrame, dataset_name: str = "dataset") -> Optional[str]:
    """NAPRAWIONA wersja z lepszą obsługą statusów i przejrzystym workflow."""
    st.header("🎯 Inteligentny wybór targetu")
    
    # Krótki opis
    st.markdown("""
    **Wykorzystaj AI do analizy danych i automatycznego wyboru najlepszego targetu do treningu modelu.**
    System analizuje strukturę danych, nazwy kolumn i charakterystyki biznesowe.
    """)

    selector = LLMTargetSelector()
    has_llm = selector.is_available()

    # Status AI/heurystyki - zawsze na górze
    status_col1, status_col2 = st.columns([3, 1])
    
    with status_col1:
        if has_llm:
            st.success("🤖 **Tryb AI aktywny** - Używam GPT-4o-mini do inteligentnej analizy danych")
        else:
            st.warning("🔧 **Tryb heurystyczny** - Analiza oparta na regułach statystycznych")
            st.caption("💡 Skonfiguruj klucz OpenAI w sidebarze, aby aktywować funkcje AI")
    
    with status_col2:
        if st.button("🔄 Odśwież status"):
            st.rerun()

    # Główne opcje analizy
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analyze_btn = st.button(
            f"🔍 {'Analizuj z AI' if has_llm else 'Analizuj heurystycznie'}", 
            type="primary",
            use_container_width=True
        )
    
    with col2:
        manual_mode = st.checkbox(
            "✋ Wybór ręczny", 
            help="Pomiń automatyczną analizę i wybierz target samodzielnie"
        )

    # Ręczny wybór
    if manual_mode:
        st.subheader("✋ Ręczny wybór targetu")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            chosen = st.selectbox(
                "Wybierz kolumnę targetu:",
                df.columns,
                help="Wybierz kolumnę, którą chcesz przewidywać"
            )
        with col2:
            if chosen:
                problem_type = infer_problem_type(df, chosen)
                st.info(f"**Typ:** {problem_type}")
        
        if st.button("✅ Zatwierdź wybór", type="primary"):
            return chosen
        return None

    # Analiza automatyczna
    if analyze_btn:
        # Resetuj poprzednie analizy
        for key in list(st.session_state.keys()):
            if key.startswith("target_analysis"):
                del st.session_state[key]
        
        with st.spinner("🔬 Analizuję strukture danych..."):
            result = selector.get_hybrid_recommendations(df, dataset_name)
            st.session_state["target_analysis"] = result
            st.session_state["target_analysis_done"] = True

    # Wyświetlenie wyników analizy
    if st.session_state.get("target_analysis_done"):
        analysis: Dict[str, Any] = st.session_state.get("target_analysis", {})
        
        st.markdown("---")
        st.subheader("📋 Wyniki analizy")

        # Preferuj LLM jeśli dostępne
        recommended_target = None
        
        if analysis.get("llm_analysis"):
            llm: LLMTargetAnalysis = analysis["llm_analysis"]
            
            # Główna rekomendacja AI
            st.success(f"🤖 **Rekomendacja AI:** `{llm.recommended_target}`")
            
            col_info, col_metrics = st.columns([2, 1])
            
            with col_info:
                st.write("**🧠 Uzasadnienie AI:**")
                st.write(llm.reasoning)
                
                if llm.business_context:
                    st.write("**💼 Kontekst biznesowy:**")
                    st.write(llm.business_context)
            
            with col_metrics:
                st.metric("Pewność AI", f"{llm.confidence:.0%}")
                st.metric("Typ problemu", llm.problem_type.title())
                
                # Wizualna ocena confidence
                if llm.confidence >= 0.8:
                    st.success("🎯 Bardzo wysoka pewność")
                elif llm.confidence >= 0.6:
                    st.info("✅ Dobra pewność")
                else:
                    st.warning("⚠️ Niska pewność")
            
            # Dodatkowe analizy AI
            if llm.data_insights:
                with st.expander("💡 Spostrzeżenia AI o danych", expanded=False):
                    st.write(llm.data_insights)
            
            if llm.warnings:
                with st.expander("⚠️ Ostrzeżenia AI", expanded=True):
                    for warning in llm.warnings:
                        st.warning(f"⚠️ {warning}")
            
            if llm.alternative_targets:
                with st.expander("🔄 Alternatywne opcje od AI", expanded=False):
                    alternatives = [target for target in llm.alternative_targets if target in df.columns]
                    if alternatives:
                        for i, alt in enumerate(alternatives, 1):
                            problem_type = infer_problem_type(df, alt)
                            st.write(f"{i}. **{alt}** ({problem_type})")
                    else:
                        st.info("Brak prawidłowych alternatyw")
            
            recommended_target = llm.recommended_target

        # Fallback heurystyczny
        if not recommended_target:
            heur = analysis.get("heuristic_recommendations", [])
            if heur:
                top = heur[0]
                st.info(f"🔧 **Rekomendacja heurystyczna:** `{top.column}`")
                
                col_info, col_metrics = st.columns([2, 1])
                
                with col_info:
                    st.write(f"**Powód:** {top.reason}")
                
                with col_metrics:
                    st.metric("Pewność", f"{top.confidence:.0%}")
                    st.metric("Typ problemu", top.problem_type.title())
                
                # Dodatkowe heurystyczne opcje
                if len(heur) > 1:
                    with st.expander("🔄 Inne opcje heurystyczne", expanded=False):
                        for i, rec in enumerate(heur[1:4], 2):
                            st.write(f"{i}. **{rec.column}** ({rec.confidence:.0%}) - {rec.reason}")
                
                recommended_target = top.column
            else:
                st.error("❌ Nie udało się wyznaczyć żadnej rekomendacji")
                recommended_target = df.columns[0] if len(df.columns) > 0 else None

        # Finalizacja wyboru
        if recommended_target:
            st.markdown("---")
            st.subheader("🎯 Finalizuj wybór targetu")
            
            try:
                default_idx = list(df.columns).index(recommended_target)
            except ValueError:
                default_idx = 0

            # Podgląd wybranego targetu
            preview_col1, preview_col2 = st.columns([1, 1])
            
            final_target = st.selectbox(
                "Zatwierdź lub zmień target:",
                df.columns,
                index=default_idx,
                help="Możesz wybrać inny target niż rekomendowany"
            )
            
            if final_target:
                target_series = df[final_target]
                problem_type = infer_problem_type(df, final_target)
                
                with preview_col1:
                    st.info(f"**Wybrano:** {final_target}")
                    st.info(f"**Typ:** {problem_type}")
                    
                with preview_col2:
                    st.metric("Unikalne wartości", int(target_series.nunique()))
                    missing_pct = target_series.isna().mean() * 100
                    st.metric("Braki danych", f"{missing_pct:.1f}%")
            
            # Przycisk finalizacji
            if st.button("✅ Zatwierdź target", type="primary", use_container_width=True):
                # Czyść cache analizy
                for key in list(st.session_state.keys()):
                    if key.startswith("target_analysis"):
                        del st.session_state[key]
                return final_target

    else:
        st.info("👆 **Kliknij przycisk analizy**, aby otrzymać inteligentne rekomendacje targetu")
        
        # Krótka instrukcja
        with st.expander("ℹ️ Jak to działa?", expanded=False):
            st.write("""
            **Tryb AI (z kluczem OpenAI):**
            - 🤖 Analiza przez GPT-4o-mini
            - 💼 Ocena wartości biznesowej
            - 🎯 Inteligentne rekomendacje
            - ⚠️ Ostrzeżenia o jakości danych
            
            **Tryb heurystyczny (bez klucza):**
            - 📊 Analiza statystyczna
            - 🏷️ Rozpoznawanie wzorców nazw
            - 📍 Analiza pozycji kolumn
            - 🔢 Ocena rozkładów wartości
            """)

    return None