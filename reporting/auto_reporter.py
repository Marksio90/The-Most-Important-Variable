# reporting/auto_reporter.py
def generate_ml_report(model_results, df_profile):
    """Automatyczny raport ML z kluczowymi insights"""
    
    report = f"""
    # 📊 TMIV AutoML Report
    
    ## Dataset Overview
    - **Rozmiar**: {df_profile['n_rows']:,} wierszy × {df_profile['n_cols']} kolumn
    - **Braki**: {df_profile['missing_pct']:.1f}%
    - **Target**: {df_profile['target']} ({df_profile['problem_type']})
    
    ## Model Performance
    - **Najlepszy model**: {model_results['best_engine']}
    - **Główna metryka**: {model_results['primary_metric']:.4f}
    - **Czas treningu**: {model_results['training_time']:.1f}s
    
    ## Top Features
    {model_results['top_features_markdown']}
    
    ## Recommendations
    {generate_recommendations(model_results, df_profile)}
    """
    
    return report

def generate_recommendations(model_results, df_profile):
    """AI-powered recommendations"""
    recommendations = []
    
    if df_profile['missing_pct'] > 20:
        recommendations.append("⚠️ Wysoki % braków - rozważ zaawansowaną imputację")
    
    if model_results['primary_metric'] < 0.8:
        recommendations.append("📈 Niska jakość modelu - spróbuj feature engineering")
    
    if df_profile['n_features'] > 100:
        recommendations.append("🎯 Dużo cech - rozważ selekcję cech")
    
    return "\n".join(f"- {rec}" for rec in recommendations)