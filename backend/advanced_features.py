# backend/advanced_features.py
class AdvancedFeatureEngineer:
    def create_features(self, df, target):
        # Interakcje między cechami
        numerical_cols = df.select_dtypes(include=['number']).columns
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
        
        # Cechy temporalne (jeśli są daty)
        date_cols = df.select_dtypes(include=['datetime']).columns
        for col in date_cols:
            df[f'{col}_quarter'] = df[col].dt.quarter
            df[f'{col}_week'] = df[col].dt.isocalendar().week
            df[f'{col}_is_weekend'] = df[col].dt.dayofweek >= 5
        
        # Binning numerycznych
        for col in numerical_cols[:5]:  # Top 5 numerical
            df[f'{col}_binned'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
        
        return df