# demo_avocado.py — Demo TMIV na zbiorze avocado
"""
Szybkie demo TMIV z użyciem datasetu avocado.csv
Testuje wszystkie główne funkcje: Smart Target, EDA, ML Training
"""

import pandas as pd
import sys
from pathlib import Path

# Dodaj ścieżki do modułów TMIV
sys.path.append('.')

from backend.smart_target import SmartTargetSelector, format_target_explanation
from backend.ml_integration import ModelConfig, train_model_comprehensive
from backend.utils import infer_problem_type, validate_dataframe
from frontend.advanced_eda import AdvancedEDAComponents
from db.db_utils import DatabaseManager, create_training_record


def load_avocado_data():
    """Wczytuje i czyści dataset avocado."""
    print("📁 Wczytywanie avocado.csv...")
    
    data_path = Path("data/avocado.csv")
    if not data_path.exists():
        print(f"❌ Nie znaleziono pliku: {data_path}")
        print("   Upewnij się, że avocado.csv jest w folderze data/")
        return None
    
    df = pd.read_csv(data_path)
    print(f"✅ Wczytano {len(df)} wierszy, {len(df.columns)} kolumn")
    
    # Podstawowe czyszczenie
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        print("🧹 Usunięto kolumnę Unnamed: 0")
    
    # Informacje o danych
    print(f"\n📊 Przegląd danych:")
    print(f"   Zakres dat: {df['Date'].min()} - {df['Date'].max()}")
    print(f"   Cena avg: ${df['AveragePrice'].mean():.2f} (min: ${df['AveragePrice'].min():.2f}, max: ${df['AveragePrice'].max():.2f})")
    print(f"   Typy avocado: {df['type'].unique().tolist()}")
    print(f"   Regiony: {df['region'].nunique()} unikalnych")
    
    return df


def demo_smart_target_selection(df):
    """Demo inteligentnego wyboru targetu."""
    print("\n🎯 DEMO: Smart Target Selection")
    print("=" * 50)
    
    selector = SmartTargetSelector()
    recommendations = selector.analyze_and_recommend(df)
    
    if recommendations:
        print(f"🏆 Najlepsza rekomendacja:")
        best = recommendations[0]
        print(f"   Kolumna: {best.column}")
        print(f"   Confidence: {best.confidence:.1%}")
        print(f"   Powód: {best.reason}")
        print(f"   Typ problemu: {best.problem_type}")
        
        if len(recommendations) > 1:
            print(f"\n🔄 Alternatywne opcje:")
            for i, rec in enumerate(recommendations[1:4], 2):
                print(f"   {i}. {rec.column} ({rec.confidence:.1%}) - {rec.reason}")
        
        return best.column
    else:
        print("❌ Brak rekomendacji targetu")
        return "AveragePrice"  # fallback


def demo_data_validation(df):
    """Demo walidacji danych."""
    print("\n🔍 DEMO: Walidacja danych")
    print("=" * 40)
    
    validation = validate_dataframe(df)
    
    print(f"Status: {'✅ Poprawne' if validation['valid'] else '❌ Problemy'}")
    print(f"Wiersze: {validation['stats']['n_rows']:,}")
    print(f"Kolumny: {validation['stats']['n_cols']}")
    print(f"Pamięć: {validation['stats']['memory_mb']:.1f} MB")
    print(f"Braki: {validation['stats']['null_cells']:,} komórek")
    print(f"Duplikaty: {validation['stats']['duplicate_rows']:,} wierszy")
    
    if validation['warnings']:
        print(f"\n⚠️ Ostrzeżenia:")
        for warning in validation['warnings']:
            print(f"   • {warning}")
    
    if validation['suggestions']:
        print(f"\n💡 Sugestie:")
        for suggestion in validation['suggestions']:
            print(f"   • {suggestion}")


def demo_training(df, target_column):
    """Demo treningu modelu."""
    print(f"\n🤖 DEMO: Trening modelu na target '{target_column}'")
    print("=" * 60)
    
    # Sprawdź typ problemu
    problem_type = infer_problem_type(df, target_column)
    print(f"🔍 Wykryty typ problemu: {problem_type}")
    
    # Konfiguracja modelu
    config = ModelConfig(
        target=target_column,
        engine="auto",
        test_size=0.2,
        cv_folds=3,
        random_state=42,
        stratify=True if problem_type.lower() == "classification" else False
    )
    
    print(f"⚙️ Konfiguracja:")
    print(f"   Engine: {config.engine}")
    print(f"   Test size: {config.test_size}")
    print(f"   CV folds: {config.cv_folds}")
    print(f"   Stratify: {config.stratify}")
    
    # Trening
    print(f"\n🚀 Rozpoczynam trening...")
    try:
        import time
        start_time = time.time()
        
        result = train_model_comprehensive(df, config, use_advanced=True)
        
        training_time = time.time() - start_time
        print(f"✅ Trening zakończony w {training_time:.1f}s")
        
        # Wyniki
        print(f"\n📈 Metryki:")
        for metric_name, metric_value in result.metrics.items():
            if isinstance(metric_value, (int, float)):
                print(f"   {metric_name}: {metric_value:.4f}")
            else:
                print(f"   {metric_name}: {metric_value}")
        
        # Feature importance
        if not result.feature_importance.empty:
            print(f"\n🏆 Top 5 najważniejszych cech:")
            top_features = result.feature_importance.head(5)
            for _, row in top_features.iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Ostrzeżenia
        if result.metadata and result.metadata.get('warnings'):
            print(f"\n⚠️ Ostrzeżenia:")
            for warning in result.metadata['warnings']:
                print(f"   • {warning}")
        
        return result
        
    except Exception as e:
        print(f"❌ Błąd treningu: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def demo_save_to_history(result, target_column):
    """Demo zapisywania do historii."""
    print(f"\n💾 DEMO: Zapisywanie do historii")
    print("=" * 40)
    
    try:
        # Inicjalizacja bazy danych
        db_manager = DatabaseManager("./demo_history.db")
        
        # Stwórz rekord (uproszczony)
        from backend.ml_integration import ModelConfig
        dummy_config = ModelConfig(target=target_column, engine="auto")
        
        record = create_training_record(
            model_config=dummy_config,
            result=result,
            df=pd.DataFrame()  # dummy
        )
        record.dataset_name = "avocado_demo"
        
        # Zapisz
        success = db_manager.save_training_record(record)
        
        if success:
            print(f"✅ Zapisano do historii: {record.run_id}")
            
            # Pokaż statystyki
            stats = db_manager.get_statistics()
            print(f"📊 Statystyki bazy:")
            print(f"   Łącznie uruchomień: {stats.get('total_runs', 0)}")
            print(f"   Unikalne datasety: {stats.get('unique_datasets', 0)}")
            print(f"   Zakończone: {stats.get('completed_runs', 0)}")
        else:
            print("❌ Błąd zapisu do historii")
            
    except Exception as e:
        print(f"❌ Błąd historii: {str(e)}")


def main():
    """Główna funkcja demo."""
    print("🥑 TMIV AVOCADO DEMO")
    print("=" * 60)
    print("Testuje wszystkie główne funkcje TMIV na zbiorze avocado")
    
    # 1. Wczytaj dane
    df = load_avocado_data()
    if df is None:
        return
    
    # 2. Smart Target Selection
    target_column = demo_smart_target_selection(df)
    
    # 3. Walidacja danych
    demo_data_validation(df)
    
    # 4. Trening modelu
    result = demo_training(df, target_column)
    
    # 5. Zapis do historii
    if result:
        demo_save_to_history(result, target_column)
    
    print(f"\n🎉 DEMO ZAKOŃCZONE!")
    print(f"💡 Teraz uruchom: streamlit run app.py")
    print(f"   i wczytaj avocado.csv przez interfejs")


if __name__ == "__main__":
    main()