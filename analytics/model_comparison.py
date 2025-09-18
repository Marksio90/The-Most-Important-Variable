# analytics/model_comparison.py
def create_model_comparison_dashboard(results_history):
    """Interaktywny dashboard porównujący modele"""
    
    # Metrics over time
    fig1 = px.line(
        results_history, 
        x='created_at', 
        y='r2_score',
        color='engine',
        title='Model Performance Over Time'
    )
    
    # Performance vs Training Time scatter
    fig2 = px.scatter(
        results_history,
        x='training_time_seconds',
        y='r2_score',
        size='n_features',
        color='engine',
        hover_data=['dataset', 'target'],
        title='Performance vs Training Time'
    )
    
    # Feature importance comparison
    importance_data = []
    for _, row in results_history.iterrows():
        if 'feature_importance' in row and row['feature_importance']:
            for feature, importance in row['feature_importance'].items():
                importance_data.append({
                    'run_id': row['run_id'],
                    'feature': feature,
                    'importance': importance,
                    'engine': row['engine']
                })
    
    if importance_data:
        fig3 = px.box(
            pd.DataFrame(importance_data),
            x='feature',
            y='importance',
            color='engine',
            title='Feature Importance Distribution Across Models'
        )
        fig3.update_xaxis(tickangle=45)
    
    return fig1, fig2, fig3