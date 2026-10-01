"""Scenario container, persistence, and distance recomputation.

Coordinates in the public tables are WGS84 latitude/longitude. Distances are
metres, with ``inf`` representing an unreachable directed road journey.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

CANDIDATE_COLUMNS = ("id", "name", "lat", "lon", "area", "site_type")
DEMAND_COLUMNS = (
    "id",
    "lat",
    "lon",
    "area",
    "population",
    "retail",
    "office",
    "leisure",
    "public",
)
EXISTING_COLUMNS = ("id", "name", "lat", "lon", "area", "access", "include_default")
AREA_NAMES = ("Gading Serpong", "BSD City")
FORMAT_VERSION = 1
_EARTH_RADIUS_M = 6_371_008.8


@dataclass
class Scenario:
    """Input tables and route matrices for the station-location optimizers."""

    candidates: pd.DataFrame
    demand: pd.DataFrame
    existing: pd.DataFrame
    boundaries: dict[str, Any]
    graph: nx.MultiDiGraph | None
    metadata: dict[str, Any]
    distances_m: np.ndarray
    existing_distances_m: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _area_for_point(lat: float, lon: float, boundaries: dict[str, Any]) -> str:
    # Loaded lazily so importing the optimizer/demo does not import GIS packages.
    from shapely.geometry import Point, shape

    point = Point(lon, lat)
    for feature in boundaries.get("features", []):
        if shape(feature["geometry"]).covers(point):
            return str(feature["properties"]["area"])
    return "Outside study area"


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value) if pd.notna(value) else False


def _normalize_table(
    table: pd.DataFrame,
    kind: str,
    boundaries: dict[str, Any],
) -> pd.DataFrame:
    """Accept concise CSV overrides while preserving the public table schema."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{kind} must be a pandas DataFrame")
    result = table.copy().reset_index(drop=True)
    if "lat" not in result or "lon" not in result:
        raise ValueError(f"{kind} needs lat and lon columns")
    for column in ("lat", "lon"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["lat", "lon"]].isna().any().any():
        raise ValueError(f"{kind} has missing or nonnumeric coordinates")
    if not result["lat"].between(-90, 90).all() or not result["lon"].between(-180, 180).all():
        raise ValueError(f"{kind} coordinates must be WGS84 latitude/longitude")

    if "id" not in result:
        result["id"] = [f"{kind}_{i + 1:04d}" for i in range(len(result))]
    result["id"] = result["id"].astype("string").str.strip()
    if result["id"].isna().any() or result["id"].eq("").any() or result["id"].duplicated().any():
        raise ValueError(f"{kind} IDs must be nonempty and unique")

    if "area" not in result:
        result["area"] = [
            _area_for_point(float(lat), float(lon), boundaries)
            for lat, lon in zip(result["lat"], result["lon"])
        ]
    else:
        missing = result["area"].isna() | result["area"].astype("string").str.strip().eq("")
        if missing.any():
            result.loc[missing, "area"] = [
                _area_for_point(float(lat), float(lon), boundaries)
                for lat, lon in zip(result.loc[missing, "lat"], result.loc[missing, "lon"])
            ]
    result["area"] = result["area"].astype(str)

    if kind == "candidates":
        if "name" not in result:
            result["name"] = result["id"]
        if "site_type" not in result:
            result["site_type"] = "custom"
        required = CANDIDATE_COLUMNS
    elif kind == "existing":
        if "name" not in result:
            result["name"] = result["id"]
        if "access" not in result:
            result["access"] = "unknown"
        if "include_default" not in result:
            access = result["access"].fillna("unknown").astype(str).str.lower()
            result["include_default"] = ~access.isin({"private", "no", "customers", "permit"})
        result["include_default"] = result["include_default"].map(_as_bool).astype(bool)
        required = EXISTING_COLUMNS
    elif kind == "demand":
        for column in ("population", "retail", "office", "leisure", "public"):
            if column not in result:
                result[column] = 0.0
            result[column] = pd.to_numeric(result[column], errors="coerce")
            if result[column].isna().any() or (result[column] < 0).any():
                raise ValueError(f"{kind}.{column} must be finite and nonnegative")
        required = DEMAND_COLUMNS
    else:
        raise ValueError(f"unknown table kind: {kind}")

    return result.loc[:, list(required) + [c for c in result if c not in required]]


def _haversine_matrix(origins: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
    shape = (len(origins), len(targets))
    if not all(shape):
        return np.empty(shape, dtype=float)
    lat1 = np.radians(origins["lat"].to_numpy(dtype=float))[:, None]
    lon1 = np.radians(origins["lon"].to_numpy(dtype=float))[:, None]
    lat2 = np.radians(targets["lat"].to_numpy(dtype=float))[None, :]
    lon2 = np.radians(targets["lon"].to_numpy(dtype=float))[None, :]
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _nearest_nodes(
    graph: nx.MultiDiGraph,
    points: pd.DataFrame,
) -> tuple[list[Any], np.ndarray]:
    if points.empty:
        return [], np.empty(0, dtype=float)
    if not graph:
        raise ValueError("road graph has no nodes")
    from pyproj import Transformer

    metric_crs = "EPSG:32748"
    graph_crs = graph.graph.get("crs", "EPSG:4326")
    node_ids = list(graph.nodes)
    try:
        source_xy = np.array(
            [(float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])) for node in node_ids],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("road graph nodes need numeric x and y coordinates") from exc
    graph_to_metric = Transformer.from_crs(graph_crs, metric_crs, always_xy=True)
    point_to_metric = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    node_x, node_y = graph_to_metric.transform(source_xy[:, 0], source_xy[:, 1])
    point_x, point_y = point_to_metric.transform(
        points["lon"].to_numpy(dtype=float),
        points["lat"].to_numpy(dtype=float),
    )
    nodes_xy = np.column_stack((node_x, node_y))
    points_xy = np.column_stack((point_x, point_y))
    if not np.isfinite(nodes_xy).all() or not np.isfinite(points_xy).all():
        raise ValueError("road graph/point coordinates could not be projected")
    try:
        from scipy.spatial import cKDTree

        offsets, indexes = cKDTree(nodes_xy).query(points_xy)
    except ImportError:
        indexes = np.empty(len(points_xy), dtype=int)
        offsets = np.empty(len(points_xy), dtype=float)
        for start in range(0, len(points_xy), 256):
            block = points_xy[start : start + 256]
            squared = np.sum((block[:, None, :] - nodes_xy[None, :, :]) ** 2, axis=2)
            indexes[start : start + len(block)] = np.argmin(squared, axis=1)
            offsets[start : start + len(block)] = np.sqrt(np.min(squared, axis=1))
    return [node_ids[int(i)] for i in indexes], np.asarray(offsets, dtype=float)


def _road_matrices(
    graph: nx.MultiDiGraph,
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    existing: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    for _, _, edge in graph.edges(data=True):
        try:
            length = float(edge["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("every road edge needs a numeric length in metres") from exc
        if not math.isfinite(length) or length < 0:
            raise ValueError("road edge lengths must be finite and nonnegative")

    origins, origin_offsets = _nearest_nodes(graph, demand)
    destinations, dest_offsets = _nearest_nodes(graph, candidates)
    existing_nodes, existing_offsets = _nearest_nodes(graph, existing)
    candidate_distances = np.full((len(demand), len(candidates)), np.inf, dtype=float)
    existing_distances = np.full((len(demand), len(existing)), np.inf, dtype=float)
    candidate_columns: dict[Any, list[int]] = {}
    existing_columns: dict[Any, list[int]] = {}
    for column, node in enumerate(destinations):
        candidate_columns.setdefault(node, []).append(column)
    for column, node in enumerate(existing_nodes):
        existing_columns.setdefault(node, []).append(column)
    # On the reversed graph, a path from a site to a demand node has exactly
    # the original demand-to-site length. Route once per snapped site node.
    reversed_graph = graph.reverse(copy=False)
    for destination in candidate_columns.keys() | existing_columns.keys():
        paths = nx.single_source_dijkstra_path_length(
            reversed_graph, destination, weight="length"
        )
        road_lengths = np.array([paths.get(origin, np.inf) for origin in origins], dtype=float)
        for column in candidate_columns.get(destination, []):
            candidate_distances[:, column] = origin_offsets + road_lengths + dest_offsets[column]
        for column in existing_columns.get(destination, []):
            existing_distances[:, column] = (
                origin_offsets + road_lengths + existing_offsets[column]
            )
    return candidate_distances, existing_distances


def rebuild_distances(
    scenario: Scenario,
    candidates: pd.DataFrame | None = None,
    demand: pd.DataFrame | None = None,
    existing: pd.DataFrame | None = None,
) -> Scenario:
    """Return a scenario with table overrides and freshly aligned distance matrices.

    A graph-less scenario is permitted only for an explicitly synthetic demo.
    Its matrices use great-circle distance and must not be interpreted as roads.
    """
    candidate_table = _normalize_table(
        scenario.candidates if candidates is None else candidates,
        "candidates",
        scenario.boundaries,
    )
    demand_table = _normalize_table(
        scenario.demand if demand is None else demand,
        "demand",
        scenario.boundaries,
    )
    existing_table = _normalize_table(
        scenario.existing if existing is None else existing,
        "existing",
        scenario.boundaries,
    )
    metadata = dict(scenario.metadata)
    if scenario.graph is None:
        if metadata.get("mode") != "synthetic":
            raise ValueError(
                "a real scenario needs a road graph; use demo_scenario for synthetic distances"
            )
        candidate_distances = _haversine_matrix(demand_table, candidate_table)
        existing_distances = _haversine_matrix(demand_table, existing_table)
        metadata["distance_method"] = "haversine_synthetic"
    else:
        candidate_distances, existing_distances = _road_matrices(
            scenario.graph,
            demand_table,
            candidate_table,
            existing_table,
        )
        metadata["distance_method"] = "directed_road_shortest_path_with_snap_offsets"
    return replace(
        scenario,
        candidates=candidate_table,
        demand=demand_table,
        existing=existing_table,
        distances_m=candidate_distances,
        existing_distances_m=existing_distances,
        metadata=metadata,
    )


def _check_scenario(scenario: Scenario) -> None:
    for name, columns in (
        ("candidates", CANDIDATE_COLUMNS),
        ("demand", DEMAND_COLUMNS),
        ("existing", EXISTING_COLUMNS),
    ):
        missing = set(columns) - set(getattr(scenario, name).columns)
        if missing:
            raise ValueError(f"{name} lacks columns: {', '.join(sorted(missing))}")
    expected = (len(scenario.demand), len(scenario.candidates))
    existing_expected = (len(scenario.demand), len(scenario.existing))
    if np.shape(scenario.distances_m) != expected:
        raise ValueError(f"candidate distances shape must be {expected}")
    if np.shape(scenario.existing_distances_m) != existing_expected:
        raise ValueError(f"existing distances shape must be {existing_expected}")
    if np.isnan(scenario.distances_m).any() or np.isnan(scenario.existing_distances_m).any():
        raise ValueError("distance matrices cannot contain NaN")


def save_scenario(scenario: Scenario, path: str | Path) -> None:
    """Write a portable snapshot folder with checksums for all components."""
    _check_scenario(scenario)
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        "candidates.csv": lambda p: scenario.candidates.to_csv(p, index=False),
        "demand.csv": lambda p: scenario.demand.to_csv(p, index=False),
        "existing.csv": lambda p: scenario.existing.to_csv(p, index=False),
        "boundaries.geojson": lambda p: p.write_text(
            json.dumps(scenario.boundaries, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        ),
    }
    for filename, writer in files.items():
        writer(folder / filename)
    np.save(folder / "distances_m.npy", np.asarray(scenario.distances_m, dtype=float))
    np.save(
        folder / "existing_distances_m.npy", np.asarray(scenario.existing_distances_m, dtype=float)
    )
    graph_path = folder / "graph.graphml"
    if scenario.graph is not None:
        import osmnx as ox

        ox.save_graphml(scenario.graph, filepath=graph_path)
    elif graph_path.exists():
        graph_path.unlink()
    component_names = list(files) + ["distances_m.npy", "existing_distances_m.npy"]
    if scenario.graph is not None:
        component_names.append("graph.graphml")
    metadata = dict(scenario.metadata)
    metadata["format_version"] = FORMAT_VERSION
    metadata["graph_present"] = scenario.graph is not None
    metadata["component_sha256"] = {name: _sha256(folder / name) for name in component_names}
    (folder / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def load_scenario(path: str | Path) -> Scenario:
    """Read and verify a snapshot folder; report a missing cache clearly."""
    folder = Path(path)
    metadata_path = folder / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"scenario snapshot not found at {folder}; run prepare_scenario first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported scenario snapshot format: {metadata.get('format_version')}")
    for filename, expected in metadata.get("component_sha256", {}).items():
        component = folder / filename
        if not component.is_file() or _sha256(component) != expected:
            raise ValueError(f"missing or modified scenario component: {component}")
    boundaries = json.loads((folder / "boundaries.geojson").read_text(encoding="utf-8"))
    graph = None
    if metadata.get("graph_present"):
        import osmnx as ox

        graph = ox.load_graphml(filepath=folder / "graph.graphml")
    scenario = Scenario(
        candidates=pd.read_csv(folder / "candidates.csv"),
        demand=pd.read_csv(folder / "demand.csv"),
        existing=pd.read_csv(folder / "existing.csv"),
        boundaries=boundaries,
        graph=graph,
        metadata=metadata,
        distances_m=np.load(folder / "distances_m.npy", allow_pickle=False),
        existing_distances_m=np.load(folder / "existing_distances_m.npy", allow_pickle=False),
    )
    if "include_default" in scenario.existing:
        scenario.existing["include_default"] = (
            scenario.existing["include_default"].map(_as_bool).astype(bool)
        )
    _check_scenario(scenario)
    return scenario
