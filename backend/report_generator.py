# backend/report_generator.py — NOWY: Komprehensywny generator raportów i eksportów
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from backend.ml_integration import TrainingResult, ModelConfig
from backend.utils import format_datetime_for_display, local_now_iso
from db.db_utils import TrainingRecord, DatabaseManager


class ModelReportGenerator:
    """
    Komprehensywny generator raportów i artefaktów modelu ML.
    Tworzy różne typy eksportów: raporty HTML, wykresy PNG, pliki JSON, README, itp.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "tmiv_out"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Konfiguracja Plotly dla eksportów
        pio.kaleido.scope.mathjax = None  # Szybsze renderowanie
    
    def generate_comprehensive_export(
        self,
        model: Any,
        result: TrainingResult,
        config: ModelConfig,
        df: pd.DataFrame,
        dataset_name: str,
        run_id: str,
        db_manager: Optional[DatabaseManager] = None
    ) -> Dict[str, str]:
        """
        Generuje kompletny eksport modelu ze wszystkimi artefaktami.
        
        Returns:
            Dict mapujący typ artefaktu do ścieżki pliku
        """
        
        # Utwórz katalog dla tego run_id
        run_dir = self.output_dir / "exports" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files: Dict[str, str] = {}
        file_sizes: Dict[str, float] = {}
        
        try:
            # 1. ZAPISZ WYTRENOWANY MODEL (.joblib)
            model_path = run_dir / "model.joblib"
            self._save_model(model, model_path)
            exported_files["model"] = str(model_path)
            file_sizes["model"] = self._get_file_size_mb(model_path)
            
            # 2. METADANE MODELU (.json)
            metadata_path = run_dir / "metadata.json"
            self._save_metadata(result, config, df, dataset_name, metadata_path)
            exported_files["metadata"] = str(metadata_path)
            file_sizes["metadata"] = self._get_file_size_mb(metadata_path)
            
            # 3. RAPORT HTML
            html_path = run_dir / "model_report.html"
            self._generate_html_report(result, config, df, dataset_name, run_id, html_path)
            exported_files["html_report"] = str(html_path)
            file_sizes["html_report"] = self._get_file_size_mb(html_path)
            
            # 4. WYKRES FEATURE IMPORTANCE (.png)
            if not result.feature_importance.empty:
                importance_path = run_dir / "feature_importance.png"
                self._generate_importance_chart(result.feature_importance, importance_path)
                exported_files["feature_importance"] = str(importance_path)
                file_sizes["feature_importance"] = self._get_file_size_mb(importance_path)
            
            # 5. WYKRESY WYNIKÓW (.png)
            results_chart_path = run_dir / "model_performance.png"
            self._generate_performance_charts(result, results_chart_path)
            exported_files["performance_charts"] = str(results_chart_path)
            file_sizes["performance_charts"] = self._get_file_size_mb(results_chart_path)
            
            # 6. README.md
            readme_path = run_dir / "README.md"
            self._generate_readme(result, config, df, dataset_name, run_id, readme_path)
            exported_files["readme"] = str(readme_path)
            file_sizes["readme"] = self._get_file_size_mb(readme_path)
            
            # 7. MODEL REPORT (szczegółowy raport tekstowy)
            report_path = run_dir / "model_report.txt"
            self._generate_detailed_report(result, config, df, dataset_name, run_id, report_path)
            exported_files["detailed_report"] = str(report_path)
            file_sizes["detailed_report"] = self._get_file_size_mb(report_path)
            
            # 8. DANE WALIDACYJNE (.csv)
            validation_path = run_dir / "validation_results.csv"
            self._save_validation_data(result, validation_path)
            exported_files["validation_data"] = str(validation_path)
            file_sizes["validation_data"] = self._get_file_size_mb(validation_path)
            
            # 9. PODSUMOWANIE EKSPORTU (.json)
            export_summary = {
                "run_id": run_id,
                "dataset_name": dataset_name,
                "target": config.target,
                "engine": result.metadata.get("engine", config.engine),
                "problem_type": result.metadata.get("problem_type", "unknown"),
                "export_timestamp": local_now_iso(),
                "exported_files": exported_files,
                "file_sizes_mb": file_sizes,
                "total_size_mb": sum(file_sizes.values())
            }
            
            summary_path = run_dir / "export_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(export_summary, f, ensure_ascii=False, indent=2)
            
            exported_files["export_summary"] = str(summary_path)
            
            # Zapisz informacje o eksporcie w bazie danych
            if db_manager:
                for export_type, file_path in exported_files.items():
                    file_size = file_sizes.get(export_type, 0)
                    db_manager.save_export_record(run_id, export_type, file_path, file_size)
            
            return exported_files
            
        except Exception as e:
            print(f"[EXPORT] Błąd podczas eksportu: {e}")
            return exported_files  # Zwróć to co się udało
    
    def _save_model(self, model: Any, model_path: Path) -> None:
        """Zapisuje model do pliku joblib."""
        try:
            import joblib
            joblib.dump(model, model_path, compress=3)
        except Exception as e:
            print(f"[EXPORT] Błąd zapisu modelu: {e}")
            raise
    
    def _save_metadata(self, result: TrainingResult, config: ModelConfig, df: pd.DataFrame, dataset_name: str, metadata_path: Path) -> None:
        """Zapisuje rozbudowane metadane modelu."""
        metadata = {
            # Podstawowe informacje
            "model_info": {
                "dataset_name": dataset_name,
                "target_column": config.target,
                "problem_type": result.metadata.get("problem_type", "unknown"),
                "engine": result.metadata.get("engine", config.engine),
                "training_timestamp": local_now_iso(),
                "sklearn_version": result.metadata.get("sklearn_version", "unknown")
            },
            
            # Konfiguracja treningu
            "training_config": {
                "test_size": config.test_size,
                "cv_folds": config.cv_folds,
                "random_state": config.random_state,
                "stratify": config.stratify,
                "enable_probabilities": config.enable_probabilities,
                "feature_engineering": getattr(config, "feature_engineering", False),
                "feature_selection": getattr(config, "feature_selection", False),
                "handle_imbalance": getattr(config, "handle_imbalance", False),
                "hyperparameter_tuning": getattr(config, "hyperparameter_tuning", False),
                "ensemble_methods": getattr(config, "ensemble_methods", False)
            },
            
            # Metryki wydajności
            "performance_metrics": result.metrics,
            
            # Informacje o danych
            "data_info": {
                "n_rows": len(df),
                "n_features": len(df.columns) - 1,  # -1 dla targetu
                "feature_names": [col for col in df.columns if col != config.target],
                "target_distribution": self._get_target_distribution(df, config.target),
                "missing_values": df.isna().sum().to_dict(),
                "data_types": df.dtypes.astype(str).to_dict()
            },
            
            # Ważność cech
            "feature_importance": result.feature_importance.to_dict('records') if not result.feature_importance.empty else [],
            
            # Cross-validation scores
            "cross_validation": result.cross_val_scores,
            
            # Najlepsze parametry (jeśli tuning był używany)
            "best_parameters": result.best_params,
            
            # Ostrzeżenia i notatki
            "warnings": result.metadata.get("warnings", []),
            "notes": result.metadata.get("notes", []),
            
            # Dodatkowe metadane
            "additional_metadata": {
                "training_time_seconds": result.metadata.get("training_time_seconds"),
                "stratified": result.metadata.get("stratified", False),
                "class_distribution": result.metadata.get("class_distribution"),
                "data_signature": result.metadata.get("data_signature")
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    
    def _generate_html_report(self, result: TrainingResult, config: ModelConfig, df: pd.DataFrame, dataset_name: str, run_id: str, html_path: Path) -> None:
        """Generuje komprehensywny raport HTML."""
        
        # Przygotuj dane do raportu
        problem_type = result.metadata.get("problem_type", "unknown")
        engine = result.metadata.get("engine", config.engine)
        training_time = result.metadata.get("training_time_seconds", 0)
        
        # Template HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TMIV Model Report - {run_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        .metric-card h3 {{
            margin: 0;
            color: #667eea;
            font-size: 1.8em;
        }}
        .metric-card p {{
            margin: 5px 0 0 0;
            color: #666;
        }}
        .info-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        .info-table th, .info-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .info-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #333;
        }}
        .info-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .feature-importance {{
            max-height: 500px;
            overflow-y: auto;
        }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            color: #856404;
        }}
        .success {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            color: #155724;
        }}
        .footer {{
            background-color: #333;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 TMIV Model Report</h1>
            <p>Raport dla modelu {run_id}</p>
            <p>Wygenerowano: {format_datetime_for_display(datetime.now())}</p>
        </div>
        
        <div class="content">
            <!-- PODSTAWOWE INFORMACJE -->
            <div class="section">
                <h2>📋 Podstawowe informacje</h2>
                <table class="info-table">
                    <tr><th>Dataset</th><td>{dataset_name}</td></tr>
                    <tr><th>Target</th><td>{config.target}</td></tr>
                    <tr><th>Typ problemu</th><td>{problem_type.title()}</td></tr>
                    <tr><th>Silnik ML</th><td>{engine}</td></tr>
                    <tr><th>Czas treningu</th><td>{training_time:.2f} sekund</td></tr>
                    <tr><th>Liczba wierszy</th><td>{len(df):,}</td></tr>
                    <tr><th>Liczba cech</th><td>{len(df.columns) - 1:,}</td></tr>
                </table>
            </div>
            
            <!-- METRYKI WYDAJNOŚCI -->
            <div class="section">
                <h2>📊 Metryki wydajności</h2>
                <div class="metrics-grid">
                    {self._generate_metrics_html(result.metrics, problem_type)}
                </div>
            </div>
            
            <!-- KONFIGURACJA TRENINGU -->
            <div class="section">
                <h2>⚙️ Konfiguracja treningu</h2>
                <table class="info-table">
                    <tr><th>Rozmiar zbioru testowego</th><td>{config.test_size}</td></tr>
                    <tr><th>CV folds</th><td>{config.cv_folds}</td></tr>
                    <tr><th>Random state</th><td>{config.random_state}</td></tr>
                    <tr><th>Stratyfikacja</th><td>{'✅ Tak' if config.stratify else '❌ Nie'}</td></tr>
                    <tr><th>Feature engineering</th><td>{'✅ Tak' if getattr(config, 'feature_engineering', False) else '❌ Nie'}</td></tr>
                    <tr><th>Feature selection</th><td>{'✅ Tak' if getattr(config, 'feature_selection', False) else '❌ Nie'}</td></tr>
                    <tr><th>Hyperparameter tuning</th><td>{'✅ Tak' if getattr(config, 'hyperparameter_tuning', False) else '❌ Nie'}</td></tr>
                    <tr><th>Ensemble methods</th><td>{'✅ Tak' if getattr(config, 'ensemble_methods', False) else '❌ Nie'}</td></tr>
                </table>
            </div>
            
            <!-- WAŻNOŚĆ CECH -->
            {self._generate_feature_importance_html(result.feature_importance)}
            
            <!-- CROSS VALIDATION -->
            {self._generate_cv_html(result.cross_val_scores)}
            
            <!-- OSTRZEŻENIA -->
            {self._generate_warnings_html(result.metadata.get('warnings', []))}
            
            <!-- INFORMACJE O DANYCH -->
            <div class="section">
                <h2>📈 Informacje o danych</h2>
                <table class="info-table">
                    <tr><th>Rozkład targetu</th><td>{self._get_target_distribution_text(df, config.target)}</td></tr>
                    <tr><th>Braki danych</th><td>{df.isna().sum().sum():,} komórek ({(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%)</td></tr>
                    <tr><th>Duplikaty</th><td>{df.duplicated().sum():,} wierszy</td></tr>
                    <tr><th>Kolumny numeryczne</th><td>{len(df.select_dtypes(include=[np.number]).columns):,}</td></tr>
                    <tr><th>Kolumny kategoryczne</th><td>{len(df.select_dtypes(include=['object', 'category']).columns):,}</td></tr>
                </table>
            </div>
        </div>
        
        <div class="footer">
            <p>📊 Wygenerowano przez TMIV (The Most Important Variables) Platform</p>
            <p>Zaawansowana platforma AutoML z inteligentnym wyborem targetu i automatyczną optymalizacją modeli</p>
        </div>
    </div>
</body>
</html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_metrics_html(self, metrics: Dict[str, Any], problem_type: str) -> str:
        """Generuje HTML dla metryk."""
        html_parts = []
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)) and not pd.isna(metric_value):
                # Formatuj nazwę metryki
                display_name = self._format_metric_name(metric_name)
                
                # Formatuj wartość
                if 0 <= abs(metric_value) <= 1:
                    formatted_value = f"{metric_value:.4f}"
                else:
                    formatted_value = f"{metric_value:.3f}"
                
                html_parts.append(f"""
                    <div class="metric-card">
                        <h3>{formatted_value}</h3>
                        <p>{display_name}</p>
                    </div>
                """)
        
        return "".join(html_parts)
    
    def _generate_feature_importance_html(self, fi_df: pd.DataFrame) -> str:
        """Generuje HTML dla ważności cech."""
        if fi_df.empty:
            return ""
        
        top_features = fi_df.head(20)
        
        rows_html = []
        for _, row in top_features.iterrows():
            importance_pct = row['importance'] * 100
            rows_html.append(f"""
                <tr>
                    <td>{row['feature']}</td>
                    <td>{row['importance']:.6f}</td>
                    <td>{importance_pct:.2f}%</td>
                    <td><div style="background-color: #667eea; height: 20px; width: {importance_pct * 3:.0f}px; border-radius: 3px;"></div></td>
                </tr>
            """)
        
        return f"""
            <div class="section">
                <h2>🏆 Ważność cech (Top 20)</h2>
                <div class="feature-importance">
                    <table class="info-table">
                        <thead>
                            <tr>
                                <th>Cecha</th>
                                <th>Ważność</th>
                                <th>Procent</th>
                                <th>Wizualizacja</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(rows_html)}
                        </tbody>
                    </table>
                </div>
            </div>
        """
    
    def _generate_cv_html(self, cv_scores: Dict[str, List[float]]) -> str:
        """Generuje HTML dla wyników cross-validation."""
        if not cv_scores:
            return ""
        
        rows_html = []
        for metric_name, scores in cv_scores.items():
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                rows_html.append(f"""
                    <tr>
                        <td>{metric_name}</td>
                        <td>{mean_score:.4f} ± {std_score:.4f}</td>
                        <td>{min(scores):.4f}</td>
                        <td>{max(scores):.4f}</td>
                    </tr>
                """)
        
        if not rows_html:
            return ""
        
        return f"""
            <div class="section">
                <h2>🔄 Cross-Validation</h2>
                <table class="info-table">
                    <thead>
                        <tr>
                            <th>Metryka</th>
                            <th>Średnia ± Std</th>
                            <th>Minimum</th>
                            <th>Maximum</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(rows_html)}
                    </tbody>
                </table>
            </div>
        """
    
    def _generate_warnings_html(self, warnings: List[str]) -> str:
        """Generuje HTML dla ostrzeżeń."""
        if not warnings:
            return '<div class="section"><div class="success">✅ Brak ostrzeżeń - model wytrenowany pomyślnie!</div></div>'
        
        warnings_html = []
        for warning in warnings:
            warnings_html.append(f'<div class="warning">⚠️ {warning}</div>')
        
        return f"""
            <div class="section">
                <h2>⚠️ Ostrzeżenia</h2>
                {"".join(warnings_html)}
            </div>
        """
    
    def _generate_importance_chart(self, fi_df: pd.DataFrame, chart_path: Path) -> None:
        """Generuje wykres ważności cech jako PNG."""
        if fi_df.empty:
            return
        
        # Top 15 cech
        top_features = fi_df.head(15)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 15 najważniejszych cech',
            labels={'importance': 'Ważność względna', 'feature': 'Cechy'},
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=max(500, len(top_features) * 35),
            title_x=0.5,
            title_font_size=20,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            font=dict(size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_yaxis(categoryorder='total ascending')
        
        # Zapisz jako PNG z wysoką rozdzielczością
        fig.write_image(chart_path, width=1200, height=max(600, len(top_features) * 35), scale=2)
    
    def _generate_performance_charts(self, result: TrainingResult, chart_path: Path) -> None:
        """Generuje wykresy wydajności modelu."""
        validation_info = result.metadata.get('validation_info', {})
        problem_type = result.metadata.get('problem_type', '').lower()
        
        if 'y_true' not in validation_info or 'y_pred' not in validation_info:
            # Stwórz wykres tylko z metryk
            self._generate_metrics_chart(result.metrics, problem_type, chart_path)
            return
        
        y_true = np.array(validation_info['y_true'])
        y_pred = np.array(validation_info['y_pred'])
        
        if problem_type == 'regression':
            self._generate_regression_charts(y_true, y_pred, result.metrics, chart_path)
        else:
            self._generate_classification_charts(y_true, y_pred, validation_info, result.metrics, chart_path)
    
    def _generate_regression_charts(self, y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict[str, Any], chart_path: Path) -> None:
        """Generuje wykresy dla regresji."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Predykcje vs Rzeczywistość', 
                'Rozkład residuów',
                'Residua vs Predykcje',
                'Metryki modelu'
            ],
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Scatter plot predykcji
        fig.add_trace(
            go.Scatter(
                x=y_true, y=y_pred,
                mode='markers',
                name='Predykcje',
                marker=dict(color='blue', opacity=0.6)
            ),
            row=1, col=1
        )
        
        # Linia idealna
        min_val, max_val = y_true.min(), y_true.max()
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Idealna predykcja',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # 2. Histogram residuów
        residuals = y_true - y_pred
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Residua',
                marker_color='lightblue',
                nbinsx=30
            ),
            row=1, col=2
        )
        
        # 3. Residua vs predykcje
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals,
                mode='markers',
                name='Residua vs Pred',
                marker=dict(color='green', opacity=0.6)
            ),
            row=2, col=1
        )
        
        # Linia zero
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        # 4. Wykres metryk
        metric_names = list(metrics.keys())[:6]  # Top 6 metryk
        metric_values = [metrics[name] for name in metric_names if isinstance(metrics[name], (int, float))]
        metric_names = [self._format_metric_name(name) for name in metric_names if isinstance(metrics[name], (int, float))]
        
        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=metric_values,
                name='Metryki',
                marker_color='orange'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Analiza wydajności modelu regresji",
            title_x=0.5,
            showlegend=False
        )
        
        fig.write_image(chart_path, width=1400, height=800, scale=2)
    
    def _generate_classification_charts(self, y_true: np.ndarray, y_pred: np.ndarray, validation_info: Dict[str, Any], metrics: Dict[str, Any], chart_path: Path) -> None:
        """Generuje wykresy dla klasyfikacji."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Macierz pomyłek', 
                'Rozkład klas',
                'Metryki modelu',
                'Dokładność vs Klasy'
            ],
            specs=[[{"type": "heatmap"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Macierz pomyłek
        if 'confusion_matrix' in validation_info:
            cm = np.array(validation_info['confusion_matrix'])
            labels = validation_info.get('labels', [f'Klasa {i}' for i in range(len(cm))])
            
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=[f"Pred: {label}" for label in labels],
                    y=[f"True: {label}" for label in labels],
                    colorscale='Blues',
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 12}
                ),
                row=1, col=1
            )
        
        # 2. Rozkład klas
        unique_labels, counts = np.unique(y_true, return_counts=True)
        fig.add_trace(
            go.Bar(
                x=[str(label) for label in unique_labels],
                y=counts,
                name='Rozkład klas',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        # 3. Wykres metryk
        metric_names = list(metrics.keys())[:6]
        metric_values = [metrics[name] for name in metric_names if isinstance(metrics[name], (int, float))]
        metric_names = [self._format_metric_name(name) for name in metric_names if isinstance(metrics[name], (int, float))]
        
        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=metric_values,
                name='Metryki',
                marker_color='orange'
            ),
            row=2, col=1
        )
        
        # 4. Per-class accuracy (jeśli dostępne)
        if 'classification_report' in validation_info:
            class_report = validation_info['classification_report']
            class_names = []
            class_f1_scores = []
            
            for class_name, class_metrics in class_report.items():
                if isinstance(class_metrics, dict) and 'f1-score' in class_metrics:
                    class_names.append(str(class_name))
                    class_f1_scores.append(class_metrics['f1-score'])
            
            if class_names:
                fig.add_trace(
                    go.Bar(
                        x=class_names,
                        y=class_f1_scores,
                        name='F1-Score per Class',
                        marker_color='purple'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=800,
            title_text="Analiza wydajności modelu klasyfikacji",
            title_x=0.5,
            showlegend=False
        )
        
        fig.write_image(chart_path, width=1400, height=800, scale=2)
    
    def _generate_metrics_chart(self, metrics: Dict[str, Any], problem_type: str, chart_path: Path) -> None:
        """Generuje prosty wykres metryk."""
        metric_names = []
        metric_values = []
        
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                metric_names.append(self._format_metric_name(name))
                metric_values.append(value)
        
        if not metric_names:
            return
        
        fig = px.bar(
            x=metric_names,
            y=metric_values,
            title=f"Metryki modelu ({problem_type})",
            labels={'x': 'Metryki', 'y': 'Wartość'},
            color=metric_values,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=600,
            title_x=0.5,
            xaxis_tickangle=45
        )
        
        fig.write_image(chart_path, width=1200, height=600, scale=2)
    
    def _generate_readme(self, result: TrainingResult, config: ModelConfig, df: pd.DataFrame, dataset_name: str, run_id: str, readme_path: Path) -> None:
        """Generuje plik README.md."""
        
        problem_type = result.metadata.get("problem_type", "unknown")
        engine = result.metadata.get("engine", config.engine)
        training_time = result.metadata.get("training_time_seconds", 0)
        
        # Główne metryki
        main_metrics = []
        if problem_type == "regression":
            for metric in ["r2", "rmse", "mae"]:
                if metric in result.metrics:
                    main_metrics.append(f"- **{self._format_metric_name(metric)}**: {result.metrics[metric]:.4f}")
        else:
            for metric in ["accuracy", "f1_macro", "precision_macro"]:
                if metric in result.metrics:
                    main_metrics.append(f"- **{self._format_metric_name(metric)}**: {result.metrics[metric]:.4f}")
        
        readme_content = f"""# 🎯 TMIV Model Report: {run_id}

## 📋 Podstawowe informacje

- **Dataset**: {dataset_name}
- **Target**: `{config.target}`
- **Problem**: {problem_type.title()}
- **Silnik ML**: {engine}
- **Data treningu**: {format_datetime_for_display(datetime.now())}
- **Czas treningu**: {training_time:.2f} sekund

## 📊 Rozmiar danych

- **Wiersze**: {len(df):,}
- **Cechy**: {len(df.columns) - 1:,} (+ 1 target)
- **Braki danych**: {df.isna().sum().sum():,} komórek ({(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%)

## 🎯 Wydajność modelu

### Główne metryki:
{chr(10).join(main_metrics) if main_metrics else "- Brak dostępnych metryk"}

### Konfiguracja treningu:
- **Test size**: {config.test_size}
- **CV folds**: {config.cv_folds}
- **Random state**: {config.random_state}
- **Stratyfikacja**: {'✅' if config.stratify else '❌'}
- **Feature engineering**: {'✅' if getattr(config, 'feature_engineering', False) else '❌'}
- **Feature selection**: {'✅' if getattr(config, 'feature_selection', False) else '❌'}
- **Hyperparameter tuning**: {'✅' if getattr(config, 'hyperparameter_tuning', False) else '❌'}
- **Ensemble methods**: {'✅' if getattr(config, 'ensemble_methods', False) else '❌'}

## 🏆 Top cechy (ważność)

{self._generate_top_features_markdown(result.feature_importance)}

## 📁 Zawartość eksportu

- `model.joblib` - Wytrenowany model (gotowy do użycia)
- `metadata.json` - Kompletne metadane modelu
- `model_report.html` - Interaktywny raport HTML
- `feature_importance.png` - Wykres ważności cech
- `model_performance.png` - Wykresy wydajności
- `validation_results.csv` - Dane walidacyjne
- `model_report.txt` - Szczegółowy raport tekstowy
- `README.md` - Ten plik

## 🚀 Użycie modelu

```python
import joblib
import pandas as pd

# Wczytaj model
model = joblib.load('model.joblib')

# Przygotuj dane (te same kolumny co w treningu, bez targetu)
new_data = pd.DataFrame({{
    # ... twoje dane ...
}})

# Predykcje
predictions = model.predict(new_data)

# Prawdopodobieństwa (dla klasyfikacji)
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(new_data)
```

## ⚠️ Ostrzeżenia i uwagi

{self._generate_warnings_markdown(result.metadata.get('warnings', []))}

## 📞 Kontakt

Model wygenerowany przez **TMIV (The Most Important Variables)** Platform.

Zaawansowana platforma AutoML z inteligentnym wyborem targetu i automatyczną optymalizacją modeli uczenia maszynowego.

---
*Raport wygenerowany automatycznie: {local_now_iso()}*
"""

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def _generate_detailed_report(self, result: TrainingResult, config: ModelConfig, df: pd.DataFrame, dataset_name: str, run_id: str, report_path: Path) -> None:
        """Generuje szczegółowy raport tekstowy."""
        
        report_content = f"""
================================================================================================
                            TMIV MODEL REPORT - SZCZEGÓŁOWA ANALIZA
================================================================================================

RUN ID: {run_id}
TIMESTAMP: {local_now_iso()}
DATASET: {dataset_name}
TARGET: {config.target}

================================================================================================
1. INFORMACJE O MODELU
================================================================================================

Problem Type: {result.metadata.get('problem_type', 'unknown').title()}
ML Engine: {result.metadata.get('engine', config.engine)}
Training Time: {result.metadata.get('training_time_seconds', 0):.2f} seconds
Sklearn Version: {result.metadata.get('sklearn_version', 'unknown')}

Model Configuration:
- Test Size: {config.test_size}
- CV Folds: {config.cv_folds}
- Random State: {config.random_state}
- Stratified: {config.stratify}
- Feature Engineering: {getattr(config, 'feature_engineering', False)}
- Feature Selection: {getattr(config, 'feature_selection', False)}
- Handle Imbalance: {getattr(config, 'handle_imbalance', False)}
- Hyperparameter Tuning: {getattr(config, 'hyperparameter_tuning', False)}
- Ensemble Methods: {getattr(config, 'ensemble_methods', False)}

================================================================================================
2. DANE TRENINGOWE
================================================================================================

Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns
Target Column: {config.target}
Features Count: {df.shape[1] - 1:,}

Data Quality:
- Missing Values: {df.isna().sum().sum():,} cells ({(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%)
- Duplicate Rows: {df.duplicated().sum():,}
- Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

Column Types:
- Numeric: {len(df.select_dtypes(include=[np.number]).columns):,}
- Categorical: {len(df.select_dtypes(include=['object', 'category']).columns):,}
- Other: {len(df.columns) - len(df.select_dtypes(include=[np.number, 'object', 'category']).columns):,}

Target Distribution:
{self._get_detailed_target_distribution(df, config.target)}

================================================================================================
3. METRYKI WYDAJNOŚCI
================================================================================================

{self._format_metrics_detailed(result.metrics)}

================================================================================================
4. CROSS-VALIDATION WYNIKI
================================================================================================

{self._format_cv_scores_detailed(result.cross_val_scores)}

================================================================================================
5. WAŻNOŚĆ CECH
================================================================================================

{self._format_feature_importance_detailed(result.feature_importance)}

================================================================================================
6. BEST PARAMETERS (Hyperparameter Tuning)
================================================================================================

{self._format_best_params_detailed(result.best_params)}

================================================================================================
7. OSTRZEŻENIA I NOTATKI
================================================================================================

Warnings:
{chr(10).join(f"- {warning}" for warning in result.metadata.get('warnings', [])) or "- Brak ostrzeżeń"}

Notes:
{chr(10).join(f"- {note}" for note in result.metadata.get('notes', [])) or "- Brak dodatkowych notatek"}

================================================================================================
8. DODATKOWE METADANE
================================================================================================

Data Signature: {result.metadata.get('data_signature', 'N/A')[:16]}...
Stratified Split: {result.metadata.get('stratified', False)}
Number of Features (Raw): {result.metadata.get('n_features_raw', 'N/A')}
Number of Features (After Preprocessing): {result.metadata.get('n_features_after_preproc', 'N/A')}
Numeric Columns: {result.metadata.get('num_cols_count', 'N/A')}
Categorical Columns: {result.metadata.get('cat_cols_count', 'N/A')}

Class Distribution (for Classification):
{self._format_class_distribution_detailed(result.metadata.get('class_distribution'))}

================================================================================================
END OF REPORT
================================================================================================

Raport wygenerowany przez TMIV Platform
https://github.com/your-repo/tmiv
"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def _save_validation_data(self, result: TrainingResult, validation_path: Path) -> None:
        """Zapisuje dane walidacyjne do CSV."""
        validation_info = result.metadata.get('validation_info', {})
        
        if 'y_true' not in validation_info or 'y_pred' not in validation_info:
            # Stwórz pusty plik
            pd.DataFrame({"note": ["Brak danych walidacyjnych"]}).to_csv(validation_path, index=False)
            return
        
        y_true = validation_info['y_true']
        y_pred = validation_info['y_pred']
        
        # Przygotuj DataFrame
        validation_df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
        })
        
        # Dodaj residua dla regresji
        problem_type = result.metadata.get('problem_type', '').lower()
        if problem_type == 'regression':
            validation_df['residual'] = np.array(y_true) - np.array(y_pred)
            validation_df['abs_residual'] = np.abs(validation_df['residual'])
        
        # Dodaj prawdopodobieństwa dla klasyfikacji
        if 'prediction_probabilities' in validation_info:
            proba = validation_info['prediction_probabilities']
            if proba and len(proba) == len(y_true):
                proba_array = np.array(proba)
                if proba_array.ndim == 2:
                    for i in range(proba_array.shape[1]):
                        validation_df[f'proba_class_{i}'] = proba_array[:, i]
                elif proba_array.ndim == 1:
                    validation_df['probability'] = proba_array
        
        validation_df.to_csv(validation_path, index=False, float_format='%.6f')
    
    # POMOCNICZE METODY FORMATOWANIA
    def _format_metric_name(self, metric_name: str) -> str:
        """Formatuje nazwę metryki."""
        formatting_map = {
            'accuracy': 'Dokładność',
            'f1_macro': 'F1 Score (Macro)',
            'f1_micro': 'F1 Score (Micro)',
            'f1_weighted': 'F1 Score (Weighted)',
            'precision_macro': 'Precision (Macro)',
            'recall_macro': 'Recall (Macro)',
            'roc_auc': 'ROC AUC',
            'roc_auc_ovr_macro': 'ROC AUC (OvR Macro)',
            'roc_auc_ovo_macro': 'ROC AUC (OvO Macro)',
            'mae': 'MAE',
            'rmse': 'RMSE',
            'r2': 'R²',
            'mape': 'MAPE (%)',
            'explained_variance': 'Explained Variance',
            'max_error': 'Max Error',
            'mean_residual': 'Mean Residual',
            'std_residual': 'Std Residual'
        }
        return formatting_map.get(metric_name, metric_name.replace('_', ' ').title())
    
    def _get_target_distribution(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Pobiera rozkład targetu."""
        target_series = df[target_col]
        
        if pd.api.types.is_numeric_dtype(target_series):
            return {
                "type": "numeric",
                "min": float(target_series.min()),
                "max": float(target_series.max()),
                "mean": float(target_series.mean()),
                "std": float(target_series.std()),
                "median": float(target_series.median())
            }
        else:
            value_counts = target_series.value_counts()
            return {
                "type": "categorical",
                "unique_values": int(target_series.nunique()),
                "distribution": value_counts.head(10).to_dict()
            }
    
    def _get_target_distribution_text(self, df: pd.DataFrame, target_col: str) -> str:
        """Zwraca tekstowy opis rozkładu targetu."""
        target_series = df[target_col]
        
        if pd.api.types.is_numeric_dtype(target_series):
            return f"Numeryczny: min={target_series.min():.3f}, max={target_series.max():.3f}, mean={target_series.mean():.3f}"
        else:
            unique_count = target_series.nunique()
            most_common = target_series.mode().iloc[0] if len(target_series.mode()) > 0 else "N/A"
            return f"Kategoryczny: {unique_count} klas, najczęstsza: {most_common}"
    
    def _get_detailed_target_distribution(self, df: pd.DataFrame, target_col: str) -> str:
        """Szczegółowy opis rozkładu targetu."""
        target_series = df[target_col]
        
        if pd.api.types.is_numeric_dtype(target_series):
            stats = target_series.describe()
            return f"""
Min: {stats['min']:.6f}
25%: {stats['25%']:.6f}
50%: {stats['50%']:.6f}
75%: {stats['75%']:.6f}
Max: {stats['max']:.6f}
Mean: {stats['mean']:.6f}
Std: {stats['std']:.6f}
"""
        else:
            value_counts = target_series.value_counts().head(10)
            distribution_text = "\n".join([f"{label}: {count} ({count/len(target_series)*100:.2f}%)" 
                                         for label, count in value_counts.items()])
            return f"""
Unique Classes: {target_series.nunique()}
Total Samples: {len(target_series)}

Distribution:
{distribution_text}
"""
    
    def _format_metrics_detailed(self, metrics: Dict[str, Any]) -> str:
        """Formatuje metryki szczegółowo."""
        if not metrics:
            return "Brak dostępnych metryk."
        
        lines = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                formatted_name = self._format_metric_name(name)
                lines.append(f"{formatted_name:30} : {value:.6f}")
        
        return "\n".join(lines) if lines else "Brak metryk numerycznych."
    
    def _format_cv_scores_detailed(self, cv_scores: Dict[str, List[float]]) -> str:
        """Formatuje wyniki CV szczegółowo."""
        if not cv_scores:
            return "Brak wyników cross-validation."
        
        lines = []
        for metric_name, scores in cv_scores.items():
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                min_score = np.min(scores)
                max_score = np.max(scores)
                
                lines.append(f"{metric_name:30} : {mean_score:.6f} ± {std_score:.6f} (min: {min_score:.6f}, max: {max_score:.6f})")
        
        return "\n".join(lines) if lines else "Brak wyników CV."
    
    def _format_feature_importance_detailed(self, fi_df: pd.DataFrame) -> str:
        """Formatuje ważność cech szczegółowo."""
        if fi_df.empty:
            return "Brak danych o ważności cech."
        
        lines = []
        for i, (_, row) in enumerate(fi_df.head(20).iterrows(), 1):
            feature_name = row['feature']
            importance = row['importance']
            lines.append(f"{i:2d}. {feature_name:40} : {importance:.8f} ({importance*100:.4f}%)")
        
        return "\n".join(lines)
    
    def _format_best_params_detailed(self, best_params: Dict[str, Any]) -> str:
        """Formatuje najlepsze parametry szczegółowo."""
        if not best_params:
            return "Brak danych o optymalizacji hiperparametrów."
        
        lines = []
        for param_name, param_value in best_params.items():
            lines.append(f"{param_name:30} : {param_value}")
        
        return "\n".join(lines)
    
    def _format_class_distribution_detailed(self, class_dist: Optional[Dict[str, Any]]) -> str:
        """Formatuje rozkład klas szczegółowo."""
        if not class_dist:
            return "N/A (nie dotyczy lub brak danych)"
        
        lines = []
        total = sum(class_dist.values())
        for class_label, count in class_dist.items():
            percentage = (count / total * 100) if total > 0 else 0
            lines.append(f"{str(class_label):20} : {count:8d} ({percentage:6.2f}%)")
        
        return "\n".join(lines)
    
    def _generate_top_features_markdown(self, fi_df: pd.DataFrame) -> str:
        """Generuje markdown dla top cech."""
        if fi_df.empty:
            return "Brak danych o ważności cech."
        
        lines = []
        for i, (_, row) in enumerate(fi_df.head(10).iterrows(), 1):
            feature_name = row['feature']
            importance = row['importance']
            lines.append(f"{i}. **{feature_name}** - {importance:.6f} ({importance*100:.3f}%)")
        
        return "\n".join(lines)
    
    def _generate_warnings_markdown(self, warnings: List[str]) -> str:
        """Generuje markdown dla ostrzeżeń."""
        if not warnings:
            return "✅ Brak ostrzeżeń - model wytrenowany pomyślnie!"
        
        lines = []
        for warning in warnings:
            lines.append(f"- ⚠️ {warning}")
        
        return "\n".join(lines)
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """Pobiera rozmiar pliku w MB."""
        try:
            if file_path.exists():
                return file_path.stat().st_size / 1024 / 1024
            return 0.0
        except Exception:
            return 0.0


# FUNKCJE POMOCNICZE DO UŻYCIA W APP.PY
def export_model_comprehensive(
    model: Any,
    result: TrainingResult,
    config: ModelConfig,
    df: pd.DataFrame,
    dataset_name: str,
    run_id: str,
    output_dir: Union[str, Path] = "tmiv_out",
    db_manager: Optional[DatabaseManager] = None
) -> Dict[str, str]:
    """
    Funkcja wrapper do eksportu modelu ze wszystkimi artefaktami.
    
    Returns:
        Dict mapujący typ artefaktu do ścieżki pliku
    """
    generator = ModelReportGenerator(output_dir)
    return generator.generate_comprehensive_export(
        model, result, config, df, dataset_name, run_id, db_manager
    )


def generate_quick_report(
    result: TrainingResult,
    config: ModelConfig,
    df: pd.DataFrame,
    dataset_name: str
) -> str:
    """Generuje szybki raport tekstowy dla wyświetlenia w UI."""
    
    problem_type = result.metadata.get("problem_type", "unknown")
    engine = result.metadata.get("engine", config.engine)
    training_time = result.metadata.get("training_time_seconds", 0)
    
    # Główne metryki
    main_metrics = []
    if problem_type == "regression":
        for metric in ["r2", "rmse", "mae"]:
            if metric in result.metrics:
                metric_name = ModelReportGenerator()._format_metric_name(metric)
                main_metrics.append(f"• **{metric_name}**: {result.metrics[metric]:.4f}")
    else:
        for metric in ["accuracy", "f1_macro", "precision_macro"]:
            if metric in result.metrics:
                metric_name = ModelReportGenerator()._format_metric_name(metric)
                main_metrics.append(f"• **{metric_name}**: {result.metrics[metric]:.4f}")
    
    report = f"""
## 🎯 Podsumowanie treningu modelu

**Dataset:** {dataset_name} | **Target:** `{config.target}` | **Problem:** {problem_type.title()} | **Silnik:** {engine}

**Czas treningu:** {training_time:.2f}s | **Dane:** {len(df):,} wierszy × {len(df.columns)-1:,} cech

### 📊 Kluczowe metryki:
{chr(10).join(main_metrics) if main_metrics else "• Brak dostępnych metryk"}

### ⚙️ Konfiguracja:
• **Test size:** {config.test_size} | **CV folds:** {config.cv_folds} | **Random state:** {config.random_state}
• **Advanced options:** Feature eng: {'✅' if getattr(config, 'feature_engineering', False) else '❌'} | Feature sel: {'✅' if getattr(config, 'feature_selection', False) else '❌'} | Tuning: {'✅' if getattr(config, 'hyperparameter_tuning', False) else '❌'} | Ensemble: {'✅' if getattr(config, 'ensemble_methods', False) else '❌'}

### 🏆 Top 5 najważniejszych cech:
{chr(10).join([f"{i+1}. **{row['feature']}** ({row['importance']:.4f})" for i, (_, row) in enumerate(result.feature_importance.head(5).iterrows())]) if not result.feature_importance.empty else "• Brak danych o ważności cech"}
"""
    
    return report.strip()