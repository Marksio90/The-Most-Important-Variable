# tests/test_basic.py — sanity & środowisko

import os
import sys
import time
import json
import tempfile
import pytest

# Poziom 1: Absolutne minimum
def test_sanity_level1():
    assert 2 + 2 == 4

# Poziom 2: Basic environment check
def test_sanity_level2():
    # Math + stdlib
    assert 2 + 2 == 4
    data = {"test": True}
    assert json.dumps(data) == '{"test": true}'

# Poziom 3: Comprehensive sanity (imports + FS + pamięć)
def test_sanity_level3_comprehensive():
    # Core Python
    assert "hello".upper() == "HELLO"
    assert [1, 2, 3][1] == 2

    # Krytyczne importy (opcjonalny streamlit nie może blokować)
    try:
        import pandas as pd  # noqa: F401
        import sqlite3       # noqa: F401
        import json as _j    # noqa: F401
    except ImportError as e:
        pytest.fail(f"Critical import failed: {e}")

    # File system
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test")
        tmp = f.name
    assert os.path.exists(tmp)
    os.unlink(tmp)

    # Pamięć / wydajność prostych operacji
    test_data = list(range(1000))
    assert len(test_data) == 1000
    assert sum(test_data) == 499500

# Poziom 4: Application-specific sanity
def test_app_sanity_level4():
    # SQLite in-memory
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE test (id INTEGER)")
    conn.execute("INSERT INTO test VALUES (1)")
    row = conn.execute("SELECT * FROM test").fetchone()
    assert row[0] == 1
    conn.close()

    # DataFrame basics
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert len(df) == 3
    assert list(df.columns) == ["a", "b"]

    # ML basics (opcjonalne)
    try:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=4, random_state=0)
        assert X.shape == (100, 4)
        assert len(y) == 100
    except Exception:
        pytest.skip("scikit-learn not available")

# Poziom 5: Performance sanity
def test_performance_level5():
    start = time.time()
    result = sum(range(10000))
    duration = time.time() - start

    assert result == 49995000
    assert duration < 1.0  # szybciej niż sekunda

    data = [i for i in range(1000)]
    size_kb = sys.getsizeof(data) / 1024  # rozmiar listy (bez elementów)
    assert size_kb < 100
