# tests/test_utils.py — testy helperów: auto-pick target + typ problemu

import pandas as pd
from backend.utils import auto_pick_target, infer_problem_type

def test_prefers_average_price_over_others():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "AveragePrice": [10.0, 11.5, 9.9, 10.7],
        "target": [0, 1, 0, 1],
        "label": ["a", "b", "a", "b"],
    })
    assert auto_pick_target(df) == "AveragePrice"

def test_prefers_price_over_target_and_label():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],               # ID-like – powinno być ominięte
        "price": [100, 101, 99, 102],
        "target": [0, 1, 0, 1],
        "label": ["x", "y", "x", "y"],
    })
    assert auto_pick_target(df) == "price"

def test_skips_id_like_and_picks_y():
    df = pd.DataFrame({
        "UUID": ["a1", "a2", "a3", "a4"],  # ID-like
        "y": [0, 1, 1, 0],
        "f": [3.2, 1.1, 0.4, 2.2],
    })
    tgt = auto_pick_target(df)
    assert tgt == "y"
    assert tgt != "UUID"

def test_infer_problem_type_classification_and_regression():
    # klasyfikacja (niewiele klas, tekst/kategoria)
    dfc = pd.DataFrame({"f": [1, 2, 3, 4, 5], "target": ["a", "a", "b", "a", "b"]})
    assert infer_problem_type(dfc, "target") == "classification"

    # regresja (dużo unikatów, liczby)
    dfr = pd.DataFrame({"f": [1, 2, 3, 4, 5], "price": [10.0, 10.1, 9.9, 10.4, 10.7]})
    # tu „price” byłby targetem; sprawdzamy samą inferencję typu
    dfr["target"] = dfr["price"]
    assert infer_problem_type(dfr, "target") == "regression"
