# backend/smart_target_llm.py — Inteligentny wybór targetu (LLM + fallback)
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

        price_keywords = ["price", "cost", "value", "amount", "revenue", "sales", "total", "avg", "mean"]
        target_keywords = ["target", "label", "y", "outcome", "result", "class", "category"]

        for col in df.columns:
            s = df[col]
            name = str(col).lower()
            score = 0.0
            why = "Analiza statystyczna i nazwy kolumny"
            problem = "classification"

            if any(k in name for k in price_keywords):
                score += 0.55
                why = f"Nazwa zawiera słowo typu wartość/liczba ('{[k for k in price_keywords if k in name][0]}')."

            if any(k in name for k in target_keywords):
                score += 0.45
                why = f"Nazwa sugeruje klasyczny target ('{[k for k in target_keywords if k in name][0]}')."

            # pozycja kolumny – ostatnia kolumna często jest targetem
            if list(df.columns).index(col) == len(df.columns) - 1:
                score += 0.2
                if "Nazwa" not in why:
                    why = "Ostatnia kolumna (częsta konwencja)."

            # charakterystyka danych
            if pd.api.types.is_numeric_dtype(s):
                nunq = s.nunique(dropna=True)
                if 2 <= nunq <= 20:
                    score += 0.2
                    problem = "classification"
                elif nunq > 20:
                    score += 0.25
                    problem = "regression"
            else:
                nunq = s.nunique(dropna=True)
                if 2 <= nunq <= 50:
                    score += 0.15
                    problem = "classification"

            if score > 0.15:
                recs.append(SimpleTargetRecommendation(
                    column=col,
                    confidence=min(1.0, score),
                    reason=why,
                    problem_type=problem
                ))

        return sorted(recs, key=lambda r: r.confidence, reverse=True)[:top_k]


# ==============================
# LLM Target Selector
# ==============================
class LLMTargetSelector:
    """Inteligentny selektor targetu z opcjonalnym wsparciem LLM (OpenAI)."""

    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or get_openai_key_from_envs()
        self._openai_available: Optional[bool] = None
        self.fallback_selector = SimpleFallbackSelector()

    # --- dostępność LLM ---
    def is_available(self) -> bool:
        if not self.api_key:
            self._openai_available = False
            return False
        if self._openai_available is not None:
            return self._openai_available
        try:
            import openai  # noqa: F401
            self._openai_available = True
        except Exception:
            self._openai_available = False
        return self._openai_available

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
                max_tokens=1100,
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
                    info += f" | Zakres: {pd.to_numeric(s, errors='coerce').min():.3f}-{pd.to_numeric(s, errors='coerce').max():.3f}"
                    info += f" | Średnia: {pd.to_numeric(s, errors='coerce').mean():.3f}"
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
Zwróć **wyłącznie** JSON w tym schemacie:

{
  "recommended_target": "kolumna",
  "confidence": 0.90,
  "reasoning": "krótkie, konkretne uzasadnienie",
  "problem_type": "regression",
  "business_context": "co przewidujemy i po co",
  "alternative_targets": ["kolumna1", "kolumna2"],
  "warnings": ["opcjonalne ostrzeżenia"],
  "data_insights": "krótkie spostrzeżenia o danych"
}

Unikaj ID/index/timestamp jako targetu. Dla classification unikaj kolumn o ogromnej liczbie unikalnych wartości."""

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
            st.error(f"Błąd parsowania odpowiedzi LLM: {e}")
            return None


# ==============================
# UI: konfiguracja OpenAI
# ==============================
def render_openai_config():
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Konfiguracja AI")

    current_key = get_openai_key_from_envs()
    if current_key:
        masked = current_key[:8] + "..." + current_key[-4:] if len(current_key) > 12 else "****"
        st.sidebar.success("✅ Klucz OpenAI aktywny")
        st.sidebar.caption(f"Klucz: {masked}")
    else:
        st.sidebar.error("❌ Brak klucza OpenAI")

    with st.sidebar.expander("Ustaw klucz ręcznie", expanded=not bool(current_key)):
        temp = st.text_input(
            "Wklej klucz (sk-…)",
            type="password",
            key="temp_openai_input",
            help="Klucz użyty tylko w tej sesji (nie zapisujemy do pliku)."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔑 Ustaw klucz", type="primary", disabled=not temp):
                if set_openai_key_temp(temp):
                    st.success("Ustawiono klucz tymczasowo")
                else:
                    st.error("Nieprawidłowy format klucza (powinien zaczynać się od sk-)")
        with c2:
            if st.button("🗑️ Wyczyść"):
                try:
                    if "temp_openai_key" in st.session_state:
                        del st.session_state["temp_openai_key"]
                    if "OPENAI_API_KEY" in os.environ:
                        del os.environ["OPENAI_API_KEY"]
                    st.info("Wyczyszczono klucz w sesji")
                except Exception:
                    pass

    st.sidebar.caption("Źródła klucza: st.session_state → st.secrets → .env/ENV")


# ==============================
# UI: sekcja wyboru targetu
# ==============================
def render_smart_target_section_with_llm(df: pd.DataFrame, dataset_name: str = "dataset") -> Optional[str]:
    """Renderuje sekcję wyboru targetu z (opcjonalną) analizą LLM i bezpiecznym fallbackiem."""
    st.header("🎯 Inteligentny wybór targetu")

    selector = LLMTargetSelector()
    has_llm = selector.is_available()

    if has_llm:
        st.info("🤖 **Tryb AI**: Użyję GPT-4o-mini do analizy danych i rekomendacji targetu.")
    else:
        st.warning("🔧 **Tryb heurystyczny**: Brak klucza OpenAI – używam analizy statystycznej.")
        st.caption("💡 Skonfiguruj klucz w sidebarze, aby włączyć rekomendacje AI.")

    c1, c2 = st.columns([2, 1])
    with c1:
        analyze_btn = st.button(f"🔍 {'Analizuj z AI' if has_llm else 'Analizuj heurystycznie'}", type="primary")
    with c2:
        manual_mode = st.checkbox("✋ Wybór ręczny", help="Pomiń automatyczną analizę")

    # Ręczny wybór
    if manual_mode:
        st.subheader("✋ Ręczny wybór targetu")
        chosen = st.selectbox("Wybierz kolumnę targetu:", df.columns)
        if st.button("✅ Zatwierdź wybór"):
            return chosen
        return None

    # Analiza (po kliknięciu) – zapamiętujemy wynik w session_state
    if analyze_btn or st.session_state.get("target_analysis_done"):
        if analyze_btn:
            with st.spinner("Analizuję dane..."):
                result = selector.get_hybrid_recommendations(df, dataset_name)
                st.session_state["target_analysis"] = result
                st.session_state["target_analysis_done"] = True

        analysis: Dict[str, Any] = st.session_state.get("target_analysis", {})

        # Preferuj LLM
        recommended_target = None
        if analysis.get("llm_analysis"):
            llm: LLMTargetAnalysis = analysis["llm_analysis"]
            st.success(f"🤖 **Rekomendacja AI**: `{llm.recommended_target}`")
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write("**Uzasadnienie AI:**")
                st.write(llm.reasoning or "_brak uzasadnienia_")
                if llm.business_context:
                    st.write("**Kontekst biznesowy:**")
                    st.write(llm.business_context)
            with col_b:
                st.metric("Pewność", f"{llm.confidence:.0%}")
                st.metric("Typ problemu", llm.problem_type)

            if llm.data_insights:
                with st.expander("💡 Dodatkowe spostrzeżenia AI"):
                    st.write(llm.data_insights)
            if llm.warnings:
                st.subheader("⚠️ Ostrzeżenia AI")
                for w in llm.warnings:
                    st.warning(w)
            if llm.alternative_targets:
                with st.expander("🔄 Alternatywy od AI"):
                    st.write(", ".join([f"`{x}`" for x in llm.alternative_targets if x in df.columns]))
            recommended_target = llm.recommended_target

        # Fallback heurystyczny
        if not recommended_target:
            heur = analysis.get("heuristic_recommendations", [])
            if heur:
                top = heur[0]
                st.success(f"🔧 **Rekomendacja heurystyczna**: `{top.column}`")
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**Powód:** {top.reason}")
                with col_b:
                    st.metric("Pewność", f"{top.confidence:.0%}")
                    st.metric("Typ problemu", top.problem_type)
                recommended_target = top.column
            else:
                st.error("Nie udało się wyznaczyć rekomendacji targetu.")
                recommended_target = df.columns[0]

        # Finalizacja wyboru
        st.subheader("📋 Finalizuj wybór targetu")
        try:
            default_idx = list(df.columns).index(recommended_target)
        except ValueError:
            default_idx = 0

        final_target = st.selectbox(
            "Zatwierdź lub zmień target:",
            df.columns,
            index=default_idx,
            help="Możesz wybrać inny target niż rekomendowany."
        )
        if st.button("✅ Zatwierdź target", type="primary"):
            # czyścimy cache analizy, żeby nie mieszać przy kolejnym dataset
            st.session_state.pop("target_analysis", None)
            st.session_state.pop("target_analysis_done", None)
            return final_target

    else:
        st.info("👆 Kliknij przycisk analizy, aby otrzymać rekomendacje targetu.")

    return None
