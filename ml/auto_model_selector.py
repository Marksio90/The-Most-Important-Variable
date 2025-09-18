# ml/auto_model_selector.py
class IntelligentModelSelector:
    def select_best_model(self, X, y, problem_type, time_budget=300):
        """Automatycznie wybiera najlepszy model w czasie budget"""
        if problem_type == 'regression':
            models = {
                'lgbm': LGBMRegressor(),
                'xgb': XGBRegressor(),
                'rf': RandomForestRegressor(),
                'catboost': CatBoostRegressor(verbose=False)
            }
            metric = 'neg_mean_squared_error'
        else:
            models = {
                'lgbm': LGBMClassifier(),
                'xgb': XGBClassifier(),
                'rf': RandomForestClassifier(),
                'catboost': CatBoostClassifier(verbose=False)
            }
            metric = 'f1_macro'
        
        # Szybkie CV dla każdego modelu
        results = {}
        for name, model in models.items():
            start_time = time.time()
            cv_scores = cross_val_score(model, X, y, cv=3, scoring=metric, n_jobs=-1)
            results[name] = {
                'score': cv_scores.mean(),
                'std': cv_scores.std(),
                'time': time.time() - start_time
            }
        
        # Wybierz najlepszy z balance score/time
        best_model = max(results.keys(), 
                        key=lambda x: results[x]['score'] - 0.1 * results[x]['time'])
        return models[best_model], results