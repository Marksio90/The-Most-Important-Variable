import pytest
import pandas as pd
from ml.utils import auto_pick_target

class TestAutoPickTarget:
    """Comprehensive test suite for auto_pick_target function."""
    
    def test_prefers_average_price(self):
        """Should prefer AveragePrice when available."""
        df = pd.DataFrame({
            'AveragePrice': [1, 2, 3], 
            'volume': [10, 20, 30]
        })
        assert auto_pick_target(df) == 'AveragePrice'
    
    def test_prefers_price_over_others(self):
        """Should prefer 'price' when AveragePrice not available."""
        df = pd.DataFrame({
            'price': [1, 2, 3],
            'volume': [10, 20, 30], 
            'category': ['A', 'B', 'C']
        })
        assert auto_pick_target(df) == 'price'
    
    def test_target_keyword_priority(self):
        """Should follow priority order: AveragePrice > price > target > y > label."""
        df = pd.DataFrame({
            'label': [0, 1, 0],
            'target': [1, 2, 3],
            'price': [10, 20, 30],
            'volume': [100, 200, 300]
        })
        assert auto_pick_target(df) == 'price'  # price beats target and label
    
    def test_falls_back_to_last_numeric(self):
        """Should fall back to last numeric column when no keywords match."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'volume': [10, 20, 30],
            'revenue': [100, 200, 300]
        })
        assert auto_pick_target(df) == 'revenue'  # Last numeric column
    
    def test_ignores_unnamed_columns(self):
        """Should ignore columns starting with 'unnamed'."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'Unnamed: 0': [1, 2, 3],
            'volume': [10, 20, 30]
        })
        assert auto_pick_target(df) == 'volume'
    
    def test_empty_dataframe(self):
        """Should return None for empty DataFrame."""
        df = pd.DataFrame()
        assert auto_pick_target(df) is None
    
    def test_no_numeric_columns(self):
        """Should return None when no numeric columns available."""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'category': ['A', 'B', 'C'],
            'status': ['active', 'inactive', 'pending']
        })
        assert auto_pick_target(df) is None
    
    def test_single_column_dataframe(self):
        """Should handle single column DataFrame."""
        df = pd.DataFrame({'price': [1, 2, 3]})
        assert auto_pick_target(df) == 'price'
    
    def test_case_sensitivity(self):
        """Should be case-sensitive in current implementation."""
        df = pd.DataFrame({
            'AVERAGEPRICE': [1, 2, 3],  # Different case
            'volume': [10, 20, 30]
        })
        # Current implementation is case-sensitive
        assert auto_pick_target(df) == 'volume'  # Falls back to numeric
    
    def test_with_missing_values(self):
        """Should work with columns containing NaN values."""
        df = pd.DataFrame({
            'price': [1, None, 3],
            'volume': [10, 20, 30]
        })
        assert auto_pick_target(df) == 'price'
    
    def test_mixed_data_types(self):
        """Should handle mixed data types correctly."""
        df = pd.DataFrame({
            'id': ['A1', 'B2', 'C3'],           # object
            'date': pd.date_range('2024-01-01', periods=3),  # datetime
            'price': [1.5, 2.7, 3.9],          # float
            'count': [10, 20, 30],             # int
            'active': [True, False, True]       # bool
        })
        assert auto_pick_target(df) == 'price'  # Should prefer price keyword
    
    @pytest.mark.parametrize("target_col", [
        "AveragePrice", "price", "target", "y", "label"
    ])
    def test_keyword_detection(self, target_col):
        """Should detect any of the target keywords."""
        df = pd.DataFrame({
            target_col: [1, 2, 3],
            'other_col': [10, 20, 30]
        })
        assert auto_pick_target(df) == target_col

# Integration/property-based tests
class TestAutoPickTargetProperties:
    """Property-based tests for auto_pick_target."""
    
    def test_always_returns_existing_column_or_none(self):
        """Result should always be None or existing column name."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z'],
            'c': [10, 20, 30]
        })
        result = auto_pick_target(df)
        assert result is None or result in df.columns
    
    def test_deterministic_behavior(self):
        """Should always return same result for same input."""
        df = pd.DataFrame({
            'price': [1, 2, 3],
            'volume': [10, 20, 30]
        })
        
        result1 = auto_pick_target(df)
        result2 = auto_pick_target(df)
        result3 = auto_pick_target(df)
        
        assert result1 == result2 == result3
    
    def test_preserves_dataframe(self):
        """Should not modify the input DataFrame."""
        original_data = {'price': [1, 2, 3], 'volume': [10, 20, 30]}
        df = pd.DataFrame(original_data)
        df_copy = df.copy()
        
        auto_pick_target(df)
        
        pd.testing.assert_frame_equal(df, df_copy)