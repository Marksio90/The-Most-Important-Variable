https://the-most-important-variable-production.up.railway.app/

# 🎯 TMIV - The Most Important Variables

Zaawansowana platforma AutoML z inteligentnym wyborem targetu i automatyczną optymalizacją modeli uczenia maszynowego.

## 📋 Opis

TMIV to kompleksowa aplikacja AutoML zbudowana w Streamlit, która automatyzuje cały proces uczenia maszynowego - od wczytywania danych, przez eksploracyjną analizę danych (EDA), inteligentny wybór zmiennej docelowej, po trening modeli i generowanie szczegółowych raportów.

### Główne funkcjonalności:

- **🔍 Inteligentny wybór targetu** - AI-powered rekomendacje zmiennej docelowej z użyciem OpenAI GPT
- **📊 Zaawansowana EDA** - Komprehensywna analiza eksploracyjna z interaktywnymi wykresami
- **🤖 Multi-engine ML** - Wsparcie dla sklearn, XGBoost, LightGBM, CatBoost
- **⚙️ Smart preprocessing** - Automatyczne czyszczenie danych i feature engineering
- **📈 Comprehensive reporting** - Szczegółowe raporty HTML, PNG, CSV, JSON
- **💾 Model registry** - Pełne zarządzanie cyklem życia modeli
- **🔄 Training history** - Śledzenie wszystkich eksperymentów z metadanymi

## 🚀 Szybki start

### Wymagania

- Python 3.8+
- 4GB RAM (zalecane 8GB)
- Klucz API OpenAI (opcjonalny, ale zalecany)

### Instalacja

1. Sklonuj repozytorium:
```bash
git clone https://github.com/your-repo/tmiv.git
cd tmiv
```

2. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

3. Skonfiguruj zmienne środowiskowe:
```bash
cp .env.template .env
# Edytuj .env i ustaw OPENAI_API_KEY
```

4. Uruchom aplikację:
```bash
streamlit run app.py
```

5. Otwórz przeglądarkę na `http://localhost:8501`

## 🏗️ Architektura

```
tmiv/
├── app.py                          # Główna aplikacja Streamlit
├── config/
│   └── settings.py                 # Konfiguracja aplikacji
├── frontend/
│   ├── ui_components.py           # Komponenty interfejsu użytkownika
│   └── advanced_eda.py            # Zaawansowane komponenty EDA
├── backend/
│   ├── smart_target.py            # Inteligentny wybór targetu
│   ├── smart_target_llm.py        # LLM-based target selection
│   ├── ml_integration.py          # Integracja z bibliotekami ML
│   ├── utils.py                   # Funkcje pomocnicze
│   └── report_generator.py        # Generator raportów i eksportów
├── db/
│   └── db_utils.py                # Zarządzanie bazą danych
├── requirements.txt               # Zależności Python
├── .env.template                  # Szablon zmiennych środowiskowych
└── README.md                      # Ten plik
```

## 🎯 Workflow

### 1. Wczytywanie danych
- Obsługa CSV, Excel (XLS/XLSX)
- Automatyczna detekcja kodowania i separatorów
- Walidacja jakości danych
- Obsługa do 100MB / 100k wierszy

### 2. Eksploracyjna Analiza Danych (EDA)
- **Profil danych** - Statystyki opisowe, braki danych, duplikaty
- **Rozkłady** - Histogramy, box plots, density plots
- **Korelacje** - Interaktywne macierze korelacji
- **Analiza kategorii** - Rozkłady klas, frequency analysis
- **Analiza targetu** - Szczegółowa analiza zmiennej docelowej
- **Interakcje cech** - Pairwise relationships, scatter matrices
- **Anomalie** - Wykrywanie outlierów i wartości odstających
- **Feature importance preview** - Wstępna analiza ważności
- **Data drift** - Wykrywanie zmian w dystrybucji
- **Multicollinearity** - Analiza współliniowości
- **Temporal patterns** - Analiza trendów czasowych

### 3. Wybór targetu
- **AI-powered rekomendacje** - GPT-4 analizuje dane i sugeruje optymalne targety
- **Smart scoring** - Algorytm punktowania kolumn na podstawie nazw, typów, rozkładów
- **Automatyczna detekcja** - Rozpoznawanie problemów regresji vs klasyfikacji
- **Uzasadnienia** - Szczegółowe wyjaśnienia dlaczego dana kolumna to dobry target

### 4. Konfiguracja modelu
- **Multi-engine support** - sklearn, XGBoost, LightGBM, CatBoost
- **Smart recommendations** - Inteligentne sugestie ustawień dla konkretnego datasetu
- **Advanced preprocessing** - Feature engineering, selekcja cech, handle imbalance
- **Hyperparameter tuning** - GridSearch, RandomSearch, Bayesian optimization
- **Ensemble methods** - Voting, stacking, blending

### 5. Trening
- **Progress tracking** - Real-time progress bars
- **Cross-validation** - Stratified K-fold validation
- **Comprehensive metrics** - Wszystkie istotne metryki dla regresji/klasyfikacji
- **Early stopping** - Automatyczne zatrzymywanie przy overfittingu

### 6. Wyniki i eksport
- **Interactive visualizations** - Plotly charts, confusion matrices, ROC curves
- **Model artifacts** - .joblib model ready for production
- **Comprehensive reports** - HTML, TXT, PNG, CSV, JSON, README
- **Metadata** - Kompletne informacje o konfiguracji i wynikach

## 📊 Obsługiwane metryki

### Regresja
- R² (coefficient of determination)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- Explained Variance
- Max Error
- Mean/Std Residuals

### Klasyfikacja
- Accuracy
- Precision (macro/micro/weighted)
- Recall (macro/micro/weighted)
- F1-score (macro/micro/weighted)
- ROC AUC (binary/multiclass)
- Confusion Matrix
- Classification Report
- Log Loss

## 🔧 Konfiguracja zaawansowana

### Model Recommendations

TMIV automatycznie analizuje Twój dataset i rekomenduje optymalne ustawienia:

- **Test size** - Optymalna wielkość zbioru testowego na podstawie rozmiaru danych
- **CV folds** - Liczba foldów cross-validation dostosowana do liczby sampli
- **Feature engineering** - Czy włączyć feature engineering dla Twojego typu danych
- **Stratification** - Czy stratyfikować split dla nierównoważnych klas
- **Hyperparameter tuning** - Czy uruchomić tuning (na podstawie czasu treningu)

### Export Artifacts

Po treningu generowane są następujące pliki:

```
tmiv_out/exports/{run_id}/
├── model.joblib              # Gotowy model do produkcji
├── metadata.json             # Kompletne metadane
├── model_report.html         # Interaktywny raport HTML
├── feature_importance.png    # Wykres ważności cech
├── model_performance.png     # Wykresy wydajności
├── validation_results.csv    # Dane walidacyjne
├── model_report.txt          # Szczegółowy raport tekstowy
├── README.md                 # Dokumentacja modelu
└── export_summary.json       # Podsumowanie eksportu
```

## 🔑 Konfiguracja OpenAI

Dla pełnej funkcjonalności zalecane jest skonfigurowanie klucza OpenAI:

1. Uzyskaj klucz API na https://platform.openai.com/api-keys
2. Ustaw w pliku `.env`:
```bash
OPENAI_API_KEY=sk-your-key-here
```
3. Lub ustaw bezpośrednio w aplikacji przez sidebar

### Funkcje wymagające OpenAI:
- Inteligentne rekomendacje targetu z uzasadnieniami
- AI-powered data insights
- Smart feature naming suggestions
- Automated model interpretation

## 📈 Performance

### Limits
- Maksymalny rozmiar pliku: 100MB (konfigurowalny)
- Maksymalna liczba wierszy: 100,000 (konfigurowalny)
- Maksymalna liczba kolumn: 1,000 (konfigurowalny)
- Maksymalny czas treningu: 5 minut (konfigurowalny)

### Optymalizacje
- Lazy loading dużych datasetów
- Chunk-based processing dla EDA
- Memory-efficient preprocessing
- Parallel model training (gdy dostępne)
- SQLite dla szybkiego dostępu do historii

## 🐛 Troubleshooting

### Najczęstsze problemy:

**1. Błąd importu modułów**
```bash
pip install -r requirements.txt
```

**2. Brak klucza OpenAI**
- Sprawdź plik `.env`
- Upewnij się, że klucz zaczyna się od `sk-`

**3. Błędy bazy danych**
```bash
rm tmiv_data.db  # Usuń i pozwól aplikacji odtworzyć
```

**4. Problemy z wykresami**
```bash
pip install --upgrade plotly kaleido
```

**5. Out of memory errors**
- Zmniejsz `MAX_ROWS_LIMIT` w `.env`
- Zwiększ dostępną RAM
- Użyj próbkowania danych

### Debug mode

Włącz szczegółowe logowanie:
```bash
DEBUG=true streamlit run app.py
```

## 🤝 Contributing

1. Fork repozytorium
2. Stwórz branch feature (`git checkout -b feature/AmazingFeature`)
3. Commit zmiany (`git commit -m 'Add AmazingFeature'`)
4. Push do brancha (`git push origin feature/AmazingFeature`)
5. Otwórz Pull Request

## 📄 Licencja

Ten projekt jest licencjonowany na licencji MIT - zobacz plik `LICENSE` dla szczegółów.

## 🙋‍♂️ Wsparcie

- **GitHub Issues**: https://github.com/your-repo/tmiv/issues
- **Documentation**: https://your-docs-site.com
- **Email**: support@tmiv.com

## 🔄 Changelog

### v2.0.0 (2024-09-19)
- Dodano LLM-powered target selection
- Rozbudowana EDA z 10+ nowymi sekcjami
- Comprehensive export system
- Smart model recommendations
- Enhanced UI/UX
- Multi-engine ML support
- Improved performance and error handling

### v1.0.0 (2024-08-15)
- Pierwsza stabilna wersja
- Podstawowe funkcje AutoML
- SQLite integration
- Basic export functionality

---

**TMIV** - Automatyzacja uczenia maszynowego na wyższym poziomie 🚀