import pandas as pd
from ml.utils import auto_pick_target

def test_auto_pick_target_prefers_price():
    df = pd.DataFrame({'AveragePrice':[1,2,3], 'volume':[10,20,30]})
    assert auto_pick_target(df) == 'AveragePrice'
