# monitoring/drift_detector.py
class DataDriftDetector:
    def detect_drift(self, X_train, X_new, threshold=0.05):
        """Wykrywa drift używając testów statystycznych"""
        drift_results = {}
        
        for col in X_train.columns:
            if X_train[col].dtype in ['float64', 'int64']:
                # KS test dla numerycznych
                statistic, p_value = ks_2samp(X_train[col], X_new[col])
                drift_results[col] = {
                    'test': 'KS',
                    'statistic': statistic,
                    'p_value': p_value,
                    'drift_detected': p_value < threshold
                }
            else:
                # Chi-square test dla kategorycznych
                try:
                    chi2, p_value, _, _ = chi2_contingency([
                        X_train[col].value_counts().values,
                        X_new[col].value_counts().values
                    ])
                    drift_results[col] = {
                        'test': 'Chi2',
                        'statistic': chi2,
                        'p_value': p_value,
                        'drift_detected': p_value < threshold
                    }
                except:
                    drift_results[col] = {'drift_detected': False}
        
        return drift_results