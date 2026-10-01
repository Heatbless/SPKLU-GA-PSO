"""Reproducible command-line preparation and GA/PSO comparison."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from spklu.analysis import ACTIVITY_COLUMNS, compute_demand_weights
from spklu.data import demo_scenario, prepare_scenario
from spklu.optimization import (
    Problem,
    evaluate,
    optimize_ga,
    optimize_greedy,
    optimize_pso,
    optimize_random,
)
from spklu.scenario import load_scenario, save_scenario

AREAS = ("Gading Serpong", "BSD City")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spklu", description="Screen public car-charging sites with custom GA and PSO"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="download and cache a real OSM/WorldPop scenario")
    prepare.add_argument("--output", type=Path, default=Path("data/real"))
    prepare.add_argument("--boundaries", type=Path, help="GeoJSON with two named area polygons")
    prepare.add_argument("--refresh", action="store_true")

    demo = sub.add_parser("demo", help="create an explicitly synthetic local scenario")
    demo.add_argument("--output", type=Path, default=Path("data/demo"))
    demo.add_argument("--seed", type=int, default=42)

    info = sub.add_parser("info", help="show cached scenario metadata")
    info.add_argument("--scenario", type=Path, default=Path("data/real"))

    compare = sub.add_parser("compare", help="run matched GA and PSO experiments")
    compare.add_argument("--scenario", type=Path, default=Path("data/real"))
    compare.add_argument("--output", type=Path, default=Path("data/results"))
    compare.add_argument(
        "--mode", choices=("combined", "separate"), default="separate",
        help="combined: free allocation; separate: exact Gading Serpong and BSD City counts",
    )
    compare.add_argument("--count", type=int, default=6, help="combined new-site count")
    compare.add_argument("--gading-count", type=int, default=3)
    compare.add_argument("--bsd-count", type=int, default=3)
    compare.add_argument("--site-types", nargs="*", default=["parking", "fuel", "mall"])
    compare.add_argument("--radius-km", type=float, default=3.0)
    compare.add_argument("--distance-weight", type=float, default=0.5)
    compare.add_argument("--population-fraction", type=float, default=0.5)
    for category in ACTIVITY_COLUMNS:
        compare.add_argument(f"--{category}-weight", type=float, default=1.0)
    compare.add_argument("--population-size", type=int, default=40)
    compare.add_argument("--swarm-size", type=int, default=40)
    compare.add_argument("--evaluations", type=int, default=4000)
    compare.add_argument("--repeats", type=int, default=5)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--mutation-rate", type=float, default=0.2)
    compare.add_argument("--crossover-rate", type=float, default=0.8)
    compare.add_argument("--inertia", type=float, default=0.7)
    compare.add_argument("--cognitive", type=float, default=1.5)
    compare.add_argument("--social", type=float, default=1.5)
    return parser


def _filtered_problem(scenario, args):
    demand_mask = np.ones(len(scenario.demand), dtype=bool)
    candidate_mask = scenario.candidates["site_type"].isin(args.site_types).to_numpy()

    demand = scenario.demand.loc[demand_mask].reset_index(drop=True)
    candidates = scenario.candidates.loc[candidate_mask].reset_index(drop=True)
    if demand.empty:
        raise ValueError("No demand points in the study area")
    per_area_counts = (
        {AREAS[0]: args.gading_count, AREAS[1]: args.bsd_count}
        if args.mode == "separate"
        else None
    )
    count = sum(per_area_counts.values()) if per_area_counts is not None else args.count
    if candidates.empty and count > 0:
        raise ValueError("No eligible candidates in the study area")
    weights = compute_demand_weights(
        demand,
        population_fraction=args.population_fraction,
        activity_weights={
            category: getattr(args, f"{category}_weight") for category in ACTIVITY_COLUMNS
        },
    )

    if "include_default" in scenario.existing:
        existing_mask = scenario.existing["include_default"].fillna(False).astype(bool).to_numpy()
    else:
        existing_mask = np.ones(len(scenario.existing), dtype=bool)
    existing_distances = scenario.existing_distances_m[np.ix_(demand_mask, existing_mask)]
    problem = Problem(
        distances_m=scenario.distances_m[np.ix_(demand_mask, candidate_mask)],
        existing_distances_m=existing_distances,
        demand_weights=weights,
        candidate_areas=candidates["area"].to_numpy(),
        k=count,
        radius_m=args.radius_km * 1000,
        distance_weight=args.distance_weight,
        per_area_counts=per_area_counts,
    )
    return problem, candidates


def _result_row(area: str, repeat: int, result, candidates: pd.DataFrame) -> dict:
    row = {
        "area": area,
        "repeat": repeat,
        "algorithm": result.algorithm,
        "seed": result.seed,
        "evaluations": result.evaluations,
        "score": result.score,
        "selected_ids": ";".join(str(candidates.iloc[i]["id"]) for i in result.indices),
    }
    row.update(result.metrics)
    return row


def _site_features(area: str, problem: Problem, result, candidates: pd.DataFrame) -> list[dict]:
    ranked = []
    for index in result.indices:
        site = candidates.iloc[index]
        quotas = None
        if problem.per_area_counts is not None:
            quotas = dict(problem.per_area_counts)
            quotas[str(site["area"])] -= 1
        reduced = replace(problem, k=problem.k - 1, per_area_counts=quotas)
        remaining = tuple(other for other in result.indices if other != index)
        without_score, _ = evaluate(reduced, remaining)
        ranked.append((without_score - result.score, index))
    ranked.sort(
        key=lambda item: (
            -item[0],
            str(candidates.iloc[item[1]]["area"]),
            str(candidates.iloc[item[1]]["name"]),
            str(candidates.iloc[item[1]]["id"]),
        )
    )
    features = []
    for rank, (contribution, index) in enumerate(ranked, start=1):
        site = candidates.iloc[index]
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(site["lon"]), float(site["lat"])],
                },
                "properties": {
                    "area": str(site["area"]),
                    "experiment": area,
                    "algorithm": result.algorithm,
                    "rank": rank,
                    "id": str(site["id"]),
                    "name": str(site["name"]),
                    "site_type": str(site["site_type"]),
                    "score": result.score,
                    "marginal_score_contribution": contribution,
                },
            }
        )
    return features


def _compare(args) -> None:
    if args.repeats < 1:
        raise ValueError("repeats must be at least one")
    scenario = load_scenario(args.scenario)
    area = "Combined" if args.mode == "combined" else "Exact area counts"
    rows: list[dict] = []
    history: list[dict] = []
    features: list[dict] = []

    problem, candidates = _filtered_problem(scenario, args)
    best = {}
    greedy = optimize_greedy(problem)
    rows.append(_result_row(area, 0, greedy, candidates))
    best[greedy.algorithm] = greedy

    for repeat in range(args.repeats):
        seed = args.seed + repeat
        results = (
                optimize_ga(
                    problem,
                    population_size=args.population_size,
                    max_evaluations=args.evaluations,
                    mutation_rate=args.mutation_rate,
                    crossover_rate=args.crossover_rate,
                    seed=seed,
                ),
                optimize_pso(
                    problem,
                    swarm_size=args.swarm_size,
                    max_evaluations=args.evaluations,
                    inertia=args.inertia,
                    cognitive=args.cognitive,
                    social=args.social,
                    seed=seed,
            ),
            optimize_random(problem, samples=args.evaluations, seed=seed),
        )
        for result in results:
            rows.append(_result_row(area, repeat + 1, result, candidates))
            if result.algorithm not in best or result.score < best[result.algorithm].score:
                best[result.algorithm] = result
            for step, score in enumerate(result.history, start=1):
                history.append(
                    {
                        "area": area,
                        "repeat": repeat + 1,
                        "algorithm": result.algorithm,
                        "seed": seed,
                        "evaluation_step": step,
                        "best_score": score,
                    }
                )

    for result in best.values():
        features.extend(_site_features(area, problem, result, candidates))

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.output / "convergence.csv", index=False)
    with (args.output / "selected_sites.geojson").open("w", encoding="utf-8") as handle:
        json.dump({"type": "FeatureCollection", "features": features}, handle, indent=2)
    summary = {
        "scenario": str(args.scenario),
        "scenario_metadata": scenario.metadata,
        "mode": args.mode,
        "objective": "distance_weight * weighted_mean_distance / radius + "
        "(1-distance_weight) * uncovered_demand_share",
        "parameters": {
            "count": args.count,
            "gading_count": args.gading_count,
            "bsd_count": args.bsd_count,
            "site_types": args.site_types,
            "radius_km": args.radius_km,
            "distance_weight": args.distance_weight,
            "population_fraction": args.population_fraction,
            "activity_weights": {
                category: getattr(args, f"{category}_weight") for category in ACTIVITY_COLUMNS
            },
            "population_size": args.population_size,
            "swarm_size": args.swarm_size,
            "evaluations": args.evaluations,
            "repeats": args.repeats,
            "seed": args.seed,
            "mutation_rate": args.mutation_rate,
            "crossover_rate": args.crossover_rate,
            "inertia": args.inertia,
            "cognitive": args.cognitive,
            "social": args.social,
        },
        "aggregate": (
            pd.DataFrame(rows)
            .groupby(["area", "algorithm"])["score"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
            .fillna(0)
            .to_dict(orient="records")
        ),
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"Saved comparison results to {args.output.resolve()}")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        boundaries = None
        if args.boundaries:
            boundaries = json.loads(args.boundaries.read_text(encoding="utf-8"))
        scenario = prepare_scenario(
            args.output, boundaries_geojson=boundaries, refresh=args.refresh
        )
        save_scenario(scenario, args.output)
        print(f"Saved real scenario to {args.output.resolve()}")
    elif args.command == "demo":
        save_scenario(demo_scenario(seed=args.seed), args.output)
        print(f"Saved synthetic demo to {args.output.resolve()}")
    elif args.command == "info":
        scenario = load_scenario(args.scenario)
        print(
            json.dumps(
                {
                    "metadata": scenario.metadata,
                    "candidates": len(scenario.candidates),
                    "demand_cells": len(scenario.demand),
                    "existing_stations": len(scenario.existing),
                },
                indent=2,
                default=str,
            )
        )
    else:
        _compare(args)


if __name__ == "__main__":
    main()
