import numpy as np
import pandas as pd

from spklu.analysis import compute_demand_weights


def test_population_and_activity_are_normalized_before_blending():
    demand = pd.DataFrame(
        {
            "population": [90, 10],
            "retail": [0, 3],
            "office": [0, 0],
            "leisure": [0, 0],
            "public": [0, 0],
        }
    )
    weights = compute_demand_weights(demand, population_fraction=0.5)
    np.testing.assert_allclose(weights, [0.45, 0.55])


def test_missing_population_uses_activity_even_when_slider_is_one():
    demand = pd.DataFrame(
        {
            "population": [0, 0],
            "retail": [1, 3],
            "office": [0, 0],
            "leisure": [0, 0],
            "public": [0, 0],
        }
    )
    weights = compute_demand_weights(demand, population_fraction=1.0)
    np.testing.assert_allclose(weights, [0.25, 0.75])
