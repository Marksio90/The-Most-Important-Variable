# Poziom 1: Absolutne minimum (obecny kod)
def test_sanity():
    assert 2+2 == 4

# Poziom 2: Basic environment check  
def test_sanity():
    # Math works
    assert 2+2 == 4
    
    # Basic imports work
    import json, datetime, pathlib
    
    # Basic operations work
    data = {"test": True}
    assert json.dumps(data) == '{"test": true}'

# Poziom 3: Comprehensive sanity
def test_sanity():
    """Comprehensive sanity check for environment and dependencies."""
    
    # Core Python functionality
    assert 2+2 == 4
    assert "hello".upper() == "HELLO"
    assert [1,2,3][1] == 2
    
    # Critical imports for the app
    try:
        import pandas as pd
        import streamlit as st
        import sqlite3
        import json
    except ImportError as e:
        pytest.fail(f"Critical import failed: {e}")
    
    # File system access
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test")
        temp_path = f.name
    
    assert os.path.exists(temp_path)
    os.unlink(temp_path)
    
    # Memory and basic operations
    test_data = list(range(1000))
    assert len(test_data) == 1000
    assert sum(test_data) == 499500

# Poziom 4: Application-specific sanity  
def test_app_sanity():
    """Application-specific sanity checks."""
    
    # Database connectivity
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE test (id INTEGER)")
    conn.execute("INSERT INTO test VALUES (1)")
    result = conn.execute("SELECT * FROM test").fetchone()
    assert result[0] == 1
    conn.close()
    
    # DataFrame operations
    import pandas as pd
    df = pd.DataFrame({"a": [1,2,3], "b": [4,5,6]})
    assert len(df) == 3
    assert list(df.columns) == ["a", "b"]
    
    # ML basics (jeśli używasz)
    try:
        import sklearn
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=4)
        assert X.shape == (100, 4)
        assert len(y) == 100
    except ImportError:
        pytest.skip("scikit-learn not available")

# Poziom 5: Performance sanity
def test_performance_sanity():
    """Basic performance sanity checks."""
    import time
    
    # Simple operation should be fast
    start = time.time()
    result = sum(range(10000))
    duration = time.time() - start
    
    assert result == 49995000
    assert duration < 1.0  # Should complete in under 1 second
    
    # Memory usage reasonable
    import sys
    data = [i for i in range(1000)]
    size_kb = sys.getsizeof(data) / 1024
    assert size_kb < 100  # Should be under 100KB