"""Shared demand assumptions for the app and command-line analysis."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

ACTIVITY_COLUMNS = ("retail", "office", "leisure", "public")


def compute_demand_weights(
    demand: pd.DataFrame,
    population_fraction: float = 0.5,
    activity_weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Blend residential population and mapped activity as *proxies* for EV demand.

    Each component is normalized to sum to one before blending so that the
    slider expresses a share rather than depending on the components' units.
    A missing component is ignored and the available component gets full weight.
    If both are zero, each demand point receives equal weight.
    """
    if not 0 <= population_fraction <= 1:
        raise ValueError("population_fraction must be between 0 and 1")
    if demand.empty:
        raise ValueError("demand must contain at least one point")

    supplied = dict(activity_weights or {})
    unknown = set(supplied) - set(ACTIVITY_COLUMNS)
    if unknown:
        raise ValueError(f"unknown activity categories: {sorted(unknown)}")
    category_weights = {key: float(supplied.get(key, 1.0)) for key in ACTIVITY_COLUMNS}
    if any(not np.isfinite(value) or value < 0 for value in category_weights.values()):
        raise ValueError("activity category weights must be finite and nonnegative")

    population = _column(demand, "population")
    activity = np.zeros(len(demand), dtype=float)
    for key in ACTIVITY_COLUMNS:
        activity += category_weights[key] * _column(demand, key)

    population_sum = float(population.sum())
    activity_sum = float(activity.sum())
    if population_sum == 0 and activity_sum == 0:
        return np.full(len(demand), 1.0 / len(demand))
    if population_sum == 0:
        return activity / activity_sum
    if activity_sum == 0:
        return population / population_sum

    weights = population_fraction * population / population_sum
    weights += (1 - population_fraction) * activity / activity_sum
    return weights / weights.sum()


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        raise ValueError(f"demand is missing the '{name}' column")
    try:
        values = frame[name].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"demand '{name}' must be numeric") from exc
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"demand '{name}' must be finite and nonnegative")
    return values
