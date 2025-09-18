# ml/explanations.py
import shap

class ModelExplainer:
    def __init__(self, model, X_train):
        self.model = model
        self.explainer = shap.Explainer(model, X_train[:100])  # Sample dla wydajności
    
    def get_feature_importance(self, X_test):
        shap_values = self.explainer(X_test[:50])
        
        # Global importance
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(shap_values.values).mean(0)
        }).sort_values('importance', ascending=False)
        
        return importance_df, shap_values
    
    def plot_waterfall(self, instance_idx=0):
        """Waterfall plot dla pojedynczej predykcji"""
        return shap.waterfall_plot(self.shap_values[instance_idx])