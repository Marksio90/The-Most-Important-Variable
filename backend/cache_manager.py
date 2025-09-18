# backend/cache_manager.py
import hashlib
import pickle
from pathlib import Path

class ResultCache:
    def __init__(self, cache_dir="tmiv_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, df, config):
        # Hash z danych + konfiguracji
        df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
        config_hash = hashlib.md5(str(config).encode()).hexdigest()
        return f"{df_hash}_{config_hash}"
    
    def get(self, key):
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            return pickle.load(open(cache_file, 'rb'))
        return None
    
    def set(self, key, value):
        cache_file = self.cache_dir / f"{key}.pkl"
        pickle.dump(value, open(cache_file, 'wb'))