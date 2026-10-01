"""Facility-location objectives and optimizers for SPKLU siting.

Rows in the distance matrices represent demand points. Columns represent
candidate or existing charging stations. Distances are directed road-network
distances from demand points to stations, in metres; ``np.inf`` means that no
route is available.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class Problem:
    distances_m: np.ndarray
    existing_distances_m: np.ndarray
    demand_weights: np.ndarray
    candidate_areas: np.ndarray
    k: int
    radius_m: float = 3000
    distance_weight: float = 0.5
    per_area_counts: dict[str, int] | None = None
    _unreachable_penalty_m: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        distances = np.array(self.distances_m, dtype=float, copy=True)
        existing = np.array(self.existing_distances_m, dtype=float, copy=True)
        weights = np.array(self.demand_weights, dtype=float, copy=True)
        areas_input = np.asarray(self.candidate_areas, dtype=object)

        if distances.ndim != 2 or existing.ndim != 2:
            raise ValueError("distance matrices must be two-dimensional")
        demand_count, candidate_count = distances.shape
        if demand_count == 0:
            raise ValueError("at least one demand point is required")
        if existing.shape[0] != demand_count:
            raise ValueError("existing distances must have one row per demand point")
        if weights.ndim != 1 or weights.shape[0] != demand_count:
            raise ValueError("demand_weights must have one value per demand point")
        if areas_input.ndim != 1 or areas_input.shape[0] != candidate_count:
            raise ValueError("candidate_areas must have one value per candidate")
        if any(not isinstance(area, str) or not area.strip() for area in areas_input):
            raise ValueError("candidate areas must be nonempty strings")
        areas = np.array(areas_input, dtype=str, copy=True)

        for name, matrix in (("distances_m", distances), ("existing_distances_m", existing)):
            if np.isnan(matrix).any() or np.isneginf(matrix).any() or (matrix < 0).any():
                raise ValueError(f"{name} must contain nonnegative distances or np.inf")
        if (
            not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.isfinite(weights.sum())
            or weights.sum() <= 0
        ):
            raise ValueError("demand_weights must be finite, nonnegative, and sum to more than zero")
        if isinstance(self.k, bool) or not isinstance(self.k, Integral) or not 0 <= self.k <= candidate_count:
            raise ValueError("k must be an integer between zero and the candidate count")
        if not np.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("radius_m must be positive and finite")
        if not np.isfinite(self.distance_weight) or not 0 <= self.distance_weight <= 1:
            raise ValueError("distance_weight must be between zero and one")

        quotas = None
        if self.per_area_counts is not None:
            if not isinstance(self.per_area_counts, dict):
                raise ValueError("per_area_counts must be a dictionary")
            quotas = dict(self.per_area_counts)
            for area, count in quotas.items():
                if not isinstance(area, str) or not area.strip():
                    raise ValueError("per_area_counts keys must be nonempty area names")
                if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
                    raise ValueError("per_area_counts values must be nonnegative integers")
                if count > int(np.count_nonzero(areas == area)):
                    raise ValueError(f"not enough candidates in area {area!r}")
            if sum(quotas.values()) != self.k:
                raise ValueError("per_area_counts must sum to k")

        finite = np.concatenate((distances[np.isfinite(distances)], existing[np.isfinite(existing)]))
        largest_finite = float(finite.max()) if finite.size else 0.0
        penalty = max(2.0 * float(self.radius_m), largest_finite + float(self.radius_m))
        if not np.isfinite(penalty):
            raise ValueError("distances are too large to define a finite unreachable penalty")

        for array in (distances, existing, weights, areas):
            array.setflags(write=False)
        object.__setattr__(self, "distances_m", distances)
        object.__setattr__(self, "existing_distances_m", existing)
        object.__setattr__(self, "demand_weights", weights)
        object.__setattr__(self, "candidate_areas", areas)
        object.__setattr__(self, "per_area_counts", quotas)
        object.__setattr__(self, "_unreachable_penalty_m", penalty)


@dataclass(frozen=True)
class OptimizationResult:
    indices: tuple[int, ...]
    score: float
    metrics: dict[str, float]
    history: list[float]
    evaluations: int
    seed: int
    algorithm: str


def _validate_indices(problem: Problem, indices: Iterable[int]) -> tuple[int, ...]:
    try:
        values = tuple(indices)
    except TypeError as exc:
        raise ValueError("indices must be an iterable of candidate indices") from exc
    if len(values) != problem.k:
        raise ValueError(f"exactly {problem.k} candidates must be selected")
    if any(isinstance(i, bool) or not isinstance(i, Integral) for i in values):
        raise ValueError("candidate indices must be integers")
    selected = tuple(sorted(int(i) for i in values))
    if len(set(selected)) != len(selected):
        raise ValueError("candidate indices must be unique")
    if any(i < 0 or i >= problem.distances_m.shape[1] for i in selected):
        raise ValueError("candidate index is out of range")
    if problem.per_area_counts is not None:
        selected_areas = problem.candidate_areas[list(selected)]
        for area in set(problem.candidate_areas):
            if int(np.count_nonzero(selected_areas == area)) != problem.per_area_counts.get(str(area), 0):
                raise ValueError("candidate selection violates per-area counts")
    return selected


def _weighted_p90(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    at = int(np.searchsorted(cumulative, 0.9 * float(cumulative[-1]), side="left"))
    return float(values[order[min(at, len(order) - 1)]])


def _score_selection(problem: Problem, selected: tuple[int, ...]) -> tuple[float, dict[str, float]]:
    demand_count = problem.distances_m.shape[0]
    nearest = np.full(demand_count, np.inf, dtype=float)
    if problem.existing_distances_m.shape[1]:
        nearest = np.minimum(nearest, problem.existing_distances_m.min(axis=1))
    if selected:
        nearest = np.minimum(nearest, problem.distances_m[:, selected].min(axis=1))

    weights = problem.demand_weights
    total_weight = float(weights.sum())
    unreachable = ~np.isfinite(nearest)
    covered = (~unreachable) & (nearest <= problem.radius_m)
    effective = np.where(unreachable, problem._unreachable_penalty_m, nearest)
    mean_distance = float(np.dot(weights, effective) / total_weight)
    coverage_share = float(np.dot(weights, covered.astype(float)) / total_weight)
    uncovered_share = 1.0 - coverage_share
    score = float(
        problem.distance_weight * mean_distance / problem.radius_m
        + (1.0 - problem.distance_weight) * uncovered_share
    )
    metrics = {
        "mean_distance_m": mean_distance,
        "p90_distance_m": _weighted_p90(effective, weights),
        "coverage_share": coverage_share,
        "uncovered_share": uncovered_share,
        "unreachable_share": float(np.dot(weights, unreachable.astype(float)) / total_weight),
        "score": score,
    }
    return score, metrics


def evaluate(problem: Problem, indices: Iterable[int]) -> tuple[float, dict[str, float]]:
    """Score an exact-k selection; smaller scores are better.

    Unreachable demand uses a finite penalty greater than any road distance in
    the supplied matrices and is always counted as uncovered.
    """
    return _score_selection(problem, _validate_indices(problem, indices))


def _groups(problem: Problem) -> list[tuple[np.ndarray, int]]:
    count = problem.distances_m.shape[1]
    if problem.per_area_counts is None:
        return [(np.arange(count, dtype=int), problem.k)]
    return [
        (np.flatnonzero(problem.candidate_areas == area), problem.per_area_counts.get(str(area), 0))
        for area in sorted(set(problem.candidate_areas))
    ]


def _random_mask(problem: Problem, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(problem.distances_m.shape[1], dtype=bool)
    for indices, target in _groups(problem):
        if target:
            mask[rng.choice(indices, size=target, replace=False)] = True
    return mask


def _repair_mask(mask: np.ndarray, groups: list[tuple[np.ndarray, int]], rng: np.random.Generator) -> np.ndarray:
    repaired = mask.copy()
    for indices, target in groups:
        selected = indices[repaired[indices]]
        if len(selected) > target:
            repaired[rng.choice(selected, size=len(selected) - target, replace=False)] = False
        elif len(selected) < target:
            available = indices[~repaired[indices]]
            repaired[rng.choice(available, size=target - len(selected), replace=False)] = True
    return repaired


def _mutate_mask(mask: np.ndarray, groups: list[tuple[np.ndarray, int]], rng: np.random.Generator) -> np.ndarray:
    mutable = [(indices, target) for indices, target in groups if 0 < target < len(indices)]
    if not mutable:
        return mask
    indices, _ = mutable[int(rng.integers(len(mutable)))]
    selected = indices[mask[indices]]
    available = indices[~mask[indices]]
    changed = mask.copy()
    changed[int(rng.choice(selected))] = False
    changed[int(rng.choice(available))] = True
    return changed


def _decode_keys(keys: np.ndarray, groups: list[tuple[np.ndarray, int]]) -> tuple[int, ...]:
    chosen: list[int] = []
    for indices, target in groups:
        if target:
            order = np.lexsort((indices, -keys[indices]))
            chosen.extend(int(i) for i in indices[order[:target]])
    return tuple(sorted(chosen))


def _seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _probability(value: float, name: str) -> float:
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return float(value)


def _better(score: float, indices: tuple[int, ...], best_score: float, best_indices: tuple[int, ...]) -> bool:
    return (score, indices) < (best_score, best_indices)


def _baseline_result(problem: Problem, algorithm: str, seed: int) -> OptimizationResult:
    score, metrics = _score_selection(problem, ())
    return OptimizationResult((), score, metrics, [score], 1, seed, algorithm)


def optimize_ga(
    problem: Problem,
    population_size: int = 40,
    max_evaluations: int = 4000,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.8,
    seed: int = 42,
    on_epoch: Callable[[int, int, tuple[int, ...], float], None] | None = None,
) -> OptimizationResult:
    """Run a fixed-cardinality genetic algorithm with uniform crossover."""
    population_size = _positive_integer(population_size, "population_size")
    max_evaluations = _positive_integer(max_evaluations, "max_evaluations")
    mutation_rate = _probability(mutation_rate, "mutation_rate")
    crossover_rate = _probability(crossover_rate, "crossover_rate")
    seed = _seed(seed)
    if problem.k == 0:
        return _baseline_result(problem, "ga", seed)
    rng = np.random.default_rng(seed)
    groups = _groups(problem)
    population: list[tuple[np.ndarray, float, tuple[int, ...]]] = []
    history: list[float] = []
    best_score = float("inf")
    best_indices: tuple[int, ...] = ()
    best_metrics: dict[str, float] = {}

    def assess(mask: np.ndarray) -> tuple[np.ndarray, float, tuple[int, ...]]:
        nonlocal best_score, best_indices, best_metrics
        indices = tuple(int(i) for i in np.flatnonzero(mask))
        score, metrics = _score_selection(problem, indices)
        if _better(score, indices, best_score, best_indices):
            best_score, best_indices, best_metrics = score, indices, metrics
        history.append(best_score)
        return mask, score, indices

    for _ in range(min(population_size, max_evaluations)):
        population.append(assess(_random_mask(problem, rng)))

    initial_count = min(population_size, max_evaluations)
    total_epochs = 1 + int(np.ceil(max(0, max_evaluations - initial_count) / max(1, population_size - 1)))
    epoch = 1
    if on_epoch is not None:
        on_epoch(epoch, total_epochs, best_indices, best_score)

    def select_parent() -> np.ndarray:
        contestants = rng.integers(len(population), size=min(3, len(population)))
        winner = min((population[int(i)] for i in contestants), key=lambda item: (item[1], item[2]))
        return winner[0]

    while len(history) < max_evaluations:
        elite = min(population, key=lambda item: (item[1], item[2]))
        next_population = [elite] if population_size > 1 else []
        children = min(population_size - len(next_population), max_evaluations - len(history))
        for _ in range(children):
            first = select_parent()
            second = select_parent()
            if rng.random() < crossover_rate:
                child = np.where(rng.random(first.size) < 0.5, first, second)
                child = _repair_mask(child, groups, rng)
            else:
                child = first.copy()
            if rng.random() < mutation_rate:
                child = _mutate_mask(child, groups, rng)
            next_population.append(assess(child))
        population = next_population
        epoch += 1
        if on_epoch is not None:
            on_epoch(epoch, total_epochs, best_indices, best_score)

    return OptimizationResult(best_indices, best_score, best_metrics, history, len(history), seed, "ga")


def optimize_pso(
    problem: Problem,
    swarm_size: int = 40,
    max_evaluations: int = 4000,
    inertia: float = 0.7,
    cognitive: float = 1.5,
    social: float = 1.5,
    seed: int = 42,
    on_epoch: Callable[[int, int, tuple[int, ...], float], None] | None = None,
) -> OptimizationResult:
    """Run random-key particle swarm optimization with exact-k decoding."""
    swarm_size = _positive_integer(swarm_size, "swarm_size")
    max_evaluations = _positive_integer(max_evaluations, "max_evaluations")
    for name, value in (("inertia", inertia), ("cognitive", cognitive), ("social", social)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    seed = _seed(seed)
    if problem.k == 0:
        return _baseline_result(problem, "pso", seed)
    rng = np.random.default_rng(seed)
    groups = _groups(problem)
    active = min(swarm_size, max_evaluations)
    candidate_count = problem.distances_m.shape[1]
    positions = rng.random((active, candidate_count))
    velocities = rng.uniform(-0.25, 0.25, (active, candidate_count))
    personal_positions = positions.copy()
    personal_scores = np.full(active, np.inf)
    personal_indices: list[tuple[int, ...]] = [() for _ in range(active)]
    history: list[float] = []
    best_score = float("inf")
    best_indices: tuple[int, ...] = ()
    best_metrics: dict[str, float] = {}
    best_position = np.zeros(candidate_count)

    def assess(particle: int) -> None:
        nonlocal best_score, best_indices, best_metrics, best_position
        indices = _decode_keys(positions[particle], groups)
        score, metrics = _score_selection(problem, indices)
        if _better(score, indices, float(personal_scores[particle]), personal_indices[particle]):
            personal_scores[particle] = score
            personal_indices[particle] = indices
            personal_positions[particle] = positions[particle].copy()
        if _better(score, indices, best_score, best_indices):
            best_score, best_indices, best_metrics = score, indices, metrics
            best_position = positions[particle].copy()
        history.append(best_score)

    for particle in range(active):
        assess(particle)
    total_epochs = max(1, int(np.ceil(max_evaluations / active)))
    epoch = 1
    if on_epoch is not None:
        on_epoch(epoch, total_epochs, best_indices, best_score)
    while len(history) < max_evaluations:
        for particle in range(active):
            if len(history) == max_evaluations:
                break
            random_personal = rng.random(candidate_count)
            random_global = rng.random(candidate_count)
            velocities[particle] = (
                inertia * velocities[particle]
                + cognitive * random_personal * (personal_positions[particle] - positions[particle])
                + social * random_global * (best_position - positions[particle])
            )
            np.clip(velocities[particle], -1.0, 1.0, out=velocities[particle])
            positions[particle] += velocities[particle]
            assess(particle)
        epoch += 1
        if on_epoch is not None:
            on_epoch(epoch, total_epochs, best_indices, best_score)

    return OptimizationResult(best_indices, best_score, best_metrics, history, len(history), seed, "pso")


def optimize_greedy(problem: Problem) -> OptimizationResult:
    """Add the feasible candidate with the greatest immediate score reduction."""
    selected: tuple[int, ...] = ()
    history: list[float] = []
    evaluations = 0
    score, metrics = _score_selection(problem, selected)
    if problem.k == 0:
        return _baseline_result(problem, "greedy", 0)

    for _ in range(problem.k):
        winner: tuple[int, ...] | None = None
        winner_score = float("inf")
        winner_metrics: dict[str, float] = {}
        for candidate in range(problem.distances_m.shape[1]):
            if candidate in selected:
                continue
            if problem.per_area_counts is not None:
                area = str(problem.candidate_areas[candidate])
                already = sum(problem.candidate_areas[i] == area for i in selected)
                if already >= problem.per_area_counts.get(area, 0):
                    continue
            trial = tuple(sorted((*selected, candidate)))
            trial_score, trial_metrics = _score_selection(problem, trial)
            evaluations += 1
            if winner is None or _better(trial_score, trial, winner_score, winner):
                winner, winner_score, winner_metrics = trial, trial_score, trial_metrics
        assert winner is not None  # Problem validation guarantees a feasible completion.
        selected, score, metrics = winner, winner_score, winner_metrics
        history.append(score)
    return OptimizationResult(selected, score, metrics, history, evaluations, 0, "greedy")


def optimize_random(problem: Problem, seed: int = 42, samples: int = 1000) -> OptimizationResult:
    """Sample feasible exact-k selections as a reproducible baseline."""
    seed = _seed(seed)
    samples = _positive_integer(samples, "samples")
    if problem.k == 0:
        return _baseline_result(problem, "random", seed)
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_indices: tuple[int, ...] = ()
    best_metrics: dict[str, float] = {}
    history: list[float] = []
    for _ in range(samples):
        indices = tuple(int(i) for i in np.flatnonzero(_random_mask(problem, rng)))
        score, metrics = _score_selection(problem, indices)
        if _better(score, indices, best_score, best_indices):
            best_score, best_indices, best_metrics = score, indices, metrics
        history.append(best_score)
    return OptimizationResult(best_indices, best_score, best_metrics, history, samples, seed, "random")
