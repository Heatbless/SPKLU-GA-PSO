"""End-to-end checks for command-line comparison exports."""

import json

import numpy as np
import pandas as pd

from spklu.cli import main
from spklu.scenario import Scenario, save_scenario


def _synthetic_scenario(path):
    candidates = pd.DataFrame(
        [
            ("a1", "A1", -6.25, 106.62, "Gading Serpong", "parking"),
            ("a2", "A2", -6.26, 106.63, "Gading Serpong", "parking"),
            ("b1", "B1", -6.30, 106.68, "BSD City", "parking"),
            ("b2", "B2", -6.31, 106.69, "BSD City", "parking"),
        ],
        columns=["id", "name", "lat", "lon", "area", "site_type"],
    )
    demand = pd.DataFrame(
        [
            ("x", -6.25, 106.62, "Gading Serpong", 2, 0, 0, 0, 0),
            ("y", -6.26, 106.63, "Gading Serpong", 1, 0, 0, 0, 0),
            ("z", -6.30, 106.68, "BSD City", 1, 0, 0, 0, 0),
        ],
        columns=[
            "id", "lat", "lon", "area", "population", "retail", "office", "leisure", "public",
        ],
    )
    existing = pd.DataFrame(
        columns=["id", "name", "lat", "lon", "area", "access", "include_default"]
    )
    scenario = Scenario(
        candidates=candidates,
        demand=demand,
        existing=existing,
        boundaries={"type": "FeatureCollection", "features": []},
        graph=None,
        metadata={"mode": "synthetic", "distance_method": "test_matrix"},
        distances_m=np.array(
            [
                [100, 900, 100, 900],
                [900, 100, 900, 900],
                [900, 900, 100, 200],
            ],
            dtype=float,
        ),
        existing_distances_m=np.empty((3, 0)),
    )
    save_scenario(scenario, path)


def test_separate_mode_uses_one_quota_constrained_objective_and_exports_rank(tmp_path):
    scenario_path = tmp_path / "scenario"
    output_path = tmp_path / "results"
    _synthetic_scenario(scenario_path)

    main(
        [
            "compare", "--scenario", str(scenario_path), "--output", str(output_path),
            "--mode", "separate", "--gading-count", "1", "--bsd-count", "1",
            "--population-fraction", "1", "--distance-weight", "1", "--radius-km", "1",
            "--retail-weight", "2.5", "--office-weight", "0.5",
            "--population-size", "4", "--evaluations", "20", "--repeats", "1",
        ]
    )

    metrics = pd.read_csv(output_path / "metrics.csv")
    convergence = pd.read_csv(output_path / "convergence.csv")
    sites = json.loads((output_path / "selected_sites.geojson").read_text(encoding="utf-8"))
    summary = json.loads((output_path / "summary.json").read_text(encoding="utf-8"))

    assert set(metrics["area"]) == {"Exact area counts"}
    assert set(metrics["algorithm"]) == {"ga", "pso", "greedy", "random"}
    greedy = metrics.loc[metrics["algorithm"] == "greedy"].iloc[0]
    # An independently optimized Gading problem would choose a1; the shared
    # objective chooses a2 because the BSD site b1 already serves demand x.
    assert set(greedy["selected_ids"].split(";")) == {"a2", "b1"}
    assert np.isclose(greedy["score"], 0.1)
    assert len(convergence) == 3 * 20
    assert set(convergence["evaluation_step"]) == set(range(1, 21))

    for algorithm in ("ga", "pso", "greedy", "random"):
        chosen = [
            feature["properties"] for feature in sites["features"]
            if feature["properties"]["algorithm"] == algorithm
        ]
        assert len(chosen) == 2
        assert {item["area"] for item in chosen} == {"Gading Serpong", "BSD City"}
        assert [item["rank"] for item in chosen] == [1, 2]
        assert chosen[0]["marginal_score_contribution"] >= chosen[1]["marginal_score_contribution"]

    greedy_sites = [
        feature["properties"] for feature in sites["features"]
        if feature["properties"]["algorithm"] == "greedy"
    ]
    assert [site["id"] for site in greedy_sites] == ["b1", "a2"]
    assert np.isclose(greedy_sites[0]["marginal_score_contribution"], 0.6)
    assert np.isclose(greedy_sites[1]["marginal_score_contribution"], 0.2)
    assert summary["parameters"]["activity_weights"] == {
        "retail": 2.5, "office": 0.5, "leisure": 1.0, "public": 1.0,
    }


def test_combined_zero_count_allows_empty_candidate_filter(tmp_path):
    scenario_path = tmp_path / "scenario"
    output_path = tmp_path / "results"
    _synthetic_scenario(scenario_path)

    main(
        [
            "compare", "--scenario", str(scenario_path), "--output", str(output_path),
            "--mode", "combined", "--count", "0", "--site-types", "custom",
            "--population-size", "4", "--evaluations", "20", "--repeats", "1",
        ]
    )

    metrics = pd.read_csv(output_path / "metrics.csv")
    sites = json.loads((output_path / "selected_sites.geojson").read_text(encoding="utf-8"))
    assert len(metrics) == 4
    assert set(metrics["area"]) == {"Combined"}
    assert set(metrics["evaluations"]) == {1}
    assert metrics["selected_ids"].isna().all()
    assert sites["features"] == []
