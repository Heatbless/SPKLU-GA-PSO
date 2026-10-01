"""Behavioral tests for the SPKLU facility-location optimizers."""

import math
import unittest

import numpy as np

from spklu.optimization import (
    Problem,
    evaluate,
    optimize_ga,
    optimize_greedy,
    optimize_pso,
    optimize_random,
)


def small_problem(*, k: int = 1) -> Problem:
    return Problem(
        distances_m=np.array([[100, np.inf], [2000, 500], [np.inf, 100]]),
        existing_distances_m=np.array([[800], [np.inf], [np.inf]]),
        demand_weights=np.array([2, 1, 1]),
        candidate_areas=np.array(["Gading Serpong", "BSD City"]),
        k=k,
        radius_m=1000,
        distance_weight=0.5,
    )


def constrained_problem() -> Problem:
    return Problem(
        distances_m=np.array([[100, 500, 300, 900], [500, 100, 900, 300]]),
        existing_distances_m=np.empty((2, 0)),
        demand_weights=np.array([1, 2]),
        candidate_areas=np.array(["Gading Serpong", "Gading Serpong", "BSD City", "BSD City"]),
        k=2,
        radius_m=400,
        per_area_counts={"Gading Serpong": 1, "BSD City": 1},
    )


class EvaluateTests(unittest.TestCase):
    def test_weighted_objective_uses_existing_stations_and_coverage(self) -> None:
        problem = small_problem()
        score, metrics = evaluate(problem, [1])
        self.assertAlmostEqual(score, 0.275)
        self.assertAlmostEqual(metrics["mean_distance_m"], 550)
        self.assertAlmostEqual(metrics["p90_distance_m"], 800)
        self.assertAlmostEqual(metrics["coverage_share"], 1)
        self.assertAlmostEqual(metrics["uncovered_share"], 0)
        self.assertAlmostEqual(metrics["unreachable_share"], 0)
        self.assertAlmostEqual(metrics["score"], score)

    def test_unreachable_is_finite_but_uncovered(self) -> None:
        score, metrics = evaluate(small_problem(), [0])
        self.assertTrue(math.isfinite(score))
        self.assertAlmostEqual(metrics["mean_distance_m"], 1300)
        self.assertAlmostEqual(metrics["p90_distance_m"], 3000)
        self.assertAlmostEqual(metrics["coverage_share"], 0.5)
        self.assertAlmostEqual(metrics["unreachable_share"], 0.25)
        self.assertAlmostEqual(score, 0.9)

    def test_exact_selection_and_area_counts_are_enforced(self) -> None:
        problem = constrained_problem()
        for indices in ([], [0], [0, 1], [0, 0], [0, 4], [0, 1, 2]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                evaluate(problem, indices)
        score, _ = evaluate(problem, [3, 0])
        self.assertTrue(math.isfinite(score))

    def test_problem_rejects_bad_inputs(self) -> None:
        base = {
            "distances_m": np.array([[100, 200], [300, 400]]),
            "existing_distances_m": np.empty((2, 0)),
            "demand_weights": np.array([1, 1]),
            "candidate_areas": np.array(["A", "B"]),
            "k": 1,
        }
        invalid = (
            {"existing_distances_m": np.empty((1, 0))},
            {"distances_m": np.array([[1, 2], [3, np.nan]])},
            {"distances_m": np.array([[1, 2], [3, -1]])},
            {"demand_weights": np.array([0, 0])},
            {"candidate_areas": np.array(["A"])},
            {"k": 3},
            {"radius_m": 0},
            {"distance_weight": 1.1},
            {"per_area_counts": {"A": 2}},
            {"per_area_counts": {"A": 1, "B": 1}},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Problem(**(base | changes))


class OptimizerTests(unittest.TestCase):
    def test_all_methods_find_tiny_exact_solution(self) -> None:
        problem = small_problem()
        results = (
            optimize_ga(problem, population_size=8, max_evaluations=20, seed=7),
            optimize_pso(problem, swarm_size=8, max_evaluations=20, seed=7),
            optimize_greedy(problem),
            optimize_random(problem, samples=20, seed=7),
        )
        for result in results:
            with self.subTest(algorithm=result.algorithm):
                self.assertEqual(result.indices, (1,))
                self.assertAlmostEqual(result.score, 0.275)
                self.assertAlmostEqual(result.history[-1], result.score)

    def test_quota_feasibility_and_exact_budgets(self) -> None:
        problem = constrained_problem()
        results = (
            optimize_ga(problem, population_size=6, max_evaluations=17, seed=11),
            optimize_pso(problem, swarm_size=5, max_evaluations=17, seed=11),
            optimize_random(problem, samples=17, seed=11),
            optimize_greedy(problem),
        )
        for result in results:
            with self.subTest(algorithm=result.algorithm):
                self.assertEqual(len(result.indices), 2)
                self.assertEqual(sum(i < 2 for i in result.indices), 1)
                self.assertEqual(sum(i >= 2 for i in result.indices), 1)
                self.assertAlmostEqual(evaluate(problem, result.indices)[0], result.score)
                self.assertTrue(all(a >= b for a, b in zip(result.history, result.history[1:])))
                if result.algorithm != "greedy":
                    self.assertEqual(result.evaluations, 17)
                    self.assertEqual(len(result.history), 17)

    def test_seeded_runs_are_reproducible(self) -> None:
        problem = constrained_problem()
        for optimizer, kwargs in (
            (optimize_ga, {"population_size": 4, "max_evaluations": 13, "seed": 83}),
            (optimize_pso, {"swarm_size": 4, "max_evaluations": 13, "seed": 83}),
            (optimize_random, {"samples": 13, "seed": 83}),
        ):
            with self.subTest(optimizer=optimizer.__name__):
                self.assertEqual(optimizer(problem, **kwargs), optimizer(problem, **kwargs))

    def test_zero_new_stations_returns_one_baseline_evaluation(self) -> None:
        problem = Problem(
            distances_m=np.empty((2, 0)),
            existing_distances_m=np.empty((2, 0)),
            demand_weights=np.array([1, 3]),
            candidate_areas=np.array([], dtype=str),
            k=0,
            radius_m=1000,
            per_area_counts={},
        )
        for result in (
            optimize_ga(problem, max_evaluations=30),
            optimize_pso(problem, max_evaluations=30),
            optimize_greedy(problem),
            optimize_random(problem, samples=30),
        ):
            with self.subTest(algorithm=result.algorithm):
                self.assertEqual(result.indices, ())
                self.assertEqual(result.evaluations, 1)
                self.assertEqual(result.history, [result.score])
                self.assertAlmostEqual(result.score, 1.5)
                self.assertAlmostEqual(result.metrics["unreachable_share"], 1)

    def test_invalid_optimizer_parameters_fail_early(self) -> None:
        problem = small_problem()
        for call in (
            lambda: optimize_ga(problem, population_size=0),
            lambda: optimize_ga(problem, mutation_rate=2),
            lambda: optimize_pso(problem, swarm_size=0),
            lambda: optimize_pso(problem, inertia=-1),
            lambda: optimize_random(problem, samples=0),
            lambda: optimize_random(problem, seed=-1),
        ):
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
