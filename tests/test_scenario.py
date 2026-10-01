"""Network-free checks of scenario tables, distances, and snapshots."""

import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from spklu.data import _poi_tables, _population_cells, default_boundaries, demo_scenario
from spklu.scenario import Scenario, load_scenario, rebuild_distances, save_scenario


def _boundaries():
    def feature(area, west, south, east, north):
        return {
            "type": "Feature",
            "properties": {"area": area},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [west, south],
                        [east, south],
                        [east, north],
                        [west, north],
                        [west, south],
                    ]
                ],
            },
        }

    return {
        "type": "FeatureCollection",
        "features": [
            feature("Gading Serpong", 106.60, -6.28, 106.65, -6.22),
            feature("BSD City", 106.65, -6.34, 106.70, -6.28),
        ],
    }


def _directed_scenario():
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(1, x=106.610, y=-6.250)
    graph.add_node(2, x=106.615, y=-6.250)
    graph.add_node(3, x=106.620, y=-6.250)
    graph.add_edge(1, 2, length=100.0)
    graph.add_edge(2, 3, length=200.0)
    return Scenario(
        candidates=pd.DataFrame(
            [
                {
                    "id": "c1",
                    "name": "Test parking",
                    "lat": -6.250,
                    "lon": 106.615,
                    "area": "Gading Serpong",
                    "site_type": "parking",
                }
            ]
        ),
        demand=pd.DataFrame(
            [
                {
                    "id": "d1",
                    "lat": -6.250,
                    "lon": 106.610,
                    "area": "Gading Serpong",
                    "population": 100,
                    "retail": 0,
                    "office": 0,
                    "leisure": 0,
                    "public": 0,
                },
                {
                    "id": "d2",
                    "lat": -6.250,
                    "lon": 106.620,
                    "area": "Gading Serpong",
                    "population": 100,
                    "retail": 0,
                    "office": 0,
                    "leisure": 0,
                    "public": 0,
                },
            ]
        ),
        existing=pd.DataFrame(
            [
                {
                    "id": "e1",
                    "name": "Test station",
                    "lat": -6.250,
                    "lon": 106.610,
                    "area": "Gading Serpong",
                    "access": "yes",
                    "include_default": True,
                }
            ]
        ),
        boundaries=_boundaries(),
        graph=graph,
        metadata={"mode": "real"},
        distances_m=np.empty((2, 1)),
        existing_distances_m=np.empty((2, 1)),
    )


def test_directed_road_distance_and_unreachable():
    scenario = rebuild_distances(_directed_scenario())
    assert scenario.distances_m.shape == (2, 1)
    assert scenario.distances_m[0, 0] == pytest.approx(100.0, abs=0.001)
    assert np.isinf(scenario.distances_m[1, 0])
    assert scenario.existing_distances_m[0, 0] == pytest.approx(0.0, abs=0.001)
    assert np.isinf(scenario.existing_distances_m[1, 0])
    assert scenario.metadata["distance_method"].startswith("directed_road")


def test_rebuild_accepts_concise_csv_overrides_and_zero_existing():
    source = demo_scenario(seed=7)
    candidates = pd.DataFrame([{"lat": -6.25, "lon": 106.62}])
    existing = pd.DataFrame(columns=["lat", "lon"])
    rebuilt = rebuild_distances(source, candidates=candidates, existing=existing)
    assert rebuilt.candidates.loc[0, "id"] == "candidates_0001"
    assert rebuilt.candidates.loc[0, "site_type"] == "custom"
    assert rebuilt.candidates.loc[0, "area"] == "Gading Serpong"
    assert rebuilt.distances_m.shape == (len(source.demand), 1)
    assert rebuilt.existing_distances_m.shape == (len(source.demand), 0)
    assert np.isfinite(rebuilt.distances_m).all()


def test_real_scenario_without_road_graph_is_rejected():
    source = _directed_scenario()
    source.graph = None
    with pytest.raises(ValueError, match="real scenario needs a road graph"):
        rebuild_distances(source)


def test_snapshot_round_trip_and_checksum(tmp_path):
    scenario = demo_scenario(seed=11)
    folder = tmp_path / "snapshot"
    save_scenario(scenario, folder)
    restored = load_scenario(folder)
    assert restored.metadata["mode"] == "synthetic"
    pd.testing.assert_frame_equal(restored.candidates, scenario.candidates, check_dtype=False)
    np.testing.assert_allclose(restored.distances_m, scenario.distances_m)
    assert restored.existing_distances_m.shape == (len(restored.demand), len(restored.existing))
    assert restored.graph is None
    (folder / "candidates.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="modified scenario component"):
        load_scenario(folder)


def test_snapshot_graph_round_trip(tmp_path):
    scenario = rebuild_distances(_directed_scenario())
    folder = tmp_path / "road_snapshot"
    save_scenario(scenario, folder)
    restored = load_scenario(folder)
    assert isinstance(restored.graph, nx.MultiDiGraph)
    assert restored.graph.number_of_edges() == 2
    np.testing.assert_allclose(restored.distances_m[0, 0], 100.0, atol=0.001)


def test_boundaries_use_validated_cached_geojson_without_network(tmp_path, monkeypatch):
    path = tmp_path / "boundaries.geojson"
    path.write_text(json.dumps(_boundaries()), encoding="utf-8")
    monkeypatch.setattr(
        "spklu.data.requests.get", lambda *args, **kwargs: pytest.fail("network called")
    )
    assert [f["properties"]["area"] for f in default_boundaries(tmp_path)["features"]] == [
        "Gading Serpong",
        "BSD City",
    ]
    wrong = _boundaries()
    wrong["features"][0]["geometry"]["type"] = "Point"
    path.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid geometry|Polygon"):
        default_boundaries(tmp_path)


def test_missing_snapshot_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="run prepare_scenario first"):
        load_scenario(tmp_path / "absent")


def test_worldpop_pixels_aggregate_into_500m_area_cells(tmp_path):
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin

    to_metric = Transformer.from_crs("EPSG:4326", "EPSG:32748", always_xy=True)
    to_geo = Transformer.from_crs("EPSG:32748", "EPSG:4326", always_xy=True)
    x, y = to_metric.transform(106.63, -6.25)
    west = int(x // 500) * 500
    south = int(y // 500) * 500
    raster_path = tmp_path / "population.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:32748",
        transform=from_origin(west, south + 400, 100, 100),
        nodata=-9999,
    ) as target:
        target.write(np.ones((4, 4), dtype="float32"), 1)

    def feature(area, left, right):
        corners = [
            to_geo.transform(left, south),
            to_geo.transform(right, south),
            to_geo.transform(right, south + 400),
            to_geo.transform(left, south + 400),
            to_geo.transform(left, south),
        ]
        return {
            "type": "Feature",
            "properties": {"area": area},
            "geometry": {"type": "Polygon", "coordinates": [[list(pair) for pair in corners]]},
        }

    boundaries = {
        "type": "FeatureCollection",
        "features": [
            feature("Gading Serpong", west, west + 200),
            feature("BSD City", west + 200, west + 400),
        ],
    }
    cells = _population_cells(raster_path, boundaries)
    assert cells.set_index("area")["population"].to_dict() == {
        "Gading Serpong": 8.0,
        "BSD City": 8.0,
    }
    assert cells[["grid_x", "grid_y"]].drop_duplicates().shape[0] == 1


def test_nearby_existing_station_outside_polygons_is_retained():
    import geopandas as gpd
    from shapely.geometry import Point

    features = gpd.GeoDataFrame(
        {
            "amenity": ["parking", "charging_station"],
            "access": ["yes", "yes"],
            "name": ["Inside parking", "Nearby charger"],
        },
        geometry=[Point(106.62, -6.25), Point(106.655, -6.25)],
        crs="EPSG:4326",
        index=pd.MultiIndex.from_tuples([("node", 1), ("node", 2)]),
    )
    candidates, existing, _ = _poi_tables(features, _boundaries())
    assert candidates["id"].tolist() == ["osm_node_1"]
    assert existing["id"].tolist() == ["osm_node_2"]
    assert existing.loc[0, "area"] == "Outside study area"
