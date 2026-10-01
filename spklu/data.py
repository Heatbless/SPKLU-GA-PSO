"""Build real SPKLU siting data for Gading Serpong and BSD City.

Network requests occur only when ``prepare_scenario`` or
``default_boundaries`` is called. ``demo_scenario`` is fully synthetic and
offline. Raw WorldPop and OSMnx responses are cached under ``cache_dir``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .scenario import (
    AREA_NAMES,
    CANDIDATE_COLUMNS,
    DEMAND_COLUMNS,
    EXISTING_COLUMNS,
    Scenario,
)
from .scenario import (
    load_scenario as _load_scenario,
)
from .scenario import (
    rebuild_distances as _rebuild_distances,
)
from .scenario import (
    save_scenario as _save_scenario,
)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/lookup"
OSM_RELATIONS = {"Gading Serpong": "R6543247", "BSD City": "R10759331"}
WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2015_2030/"
    "R2025A/2025/IDN/v1/100m/constrained/idn_pop_2025_CN_100m_R2025A_v1.tif"
)
WORLDPOP_FILENAME = "idn_pop_2025_CN_100m_R2025A_v1.tif"
GRID_SIZE_M = 500
ROAD_BUFFER_M = 2_000
METRIC_CRS = "EPSG:32748"
_USER_AGENT = "SPKLU-GA-PSO/0.1 (research scenario builder; OpenStreetMap attribution included)"
_FEATURE_TAGS = {
    "amenity": [
        "parking",
        "fuel",
        "charging_station",
        "school",
        "university",
        "college",
        "hospital",
        "clinic",
        "community_centre",
        "townhall",
        "library",
        "cinema",
        "marketplace",
        "bus_station",
        "place_of_worship",
    ],
    "shop": True,
    "office": True,
    "leisure": True,
}
_PUBLIC_AMENITIES = {
    "school",
    "university",
    "college",
    "hospital",
    "clinic",
    "community_centre",
    "townhall",
    "library",
    "cinema",
    "marketplace",
    "bus_station",
    "place_of_worship",
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_boundaries(collection: dict[str, Any]) -> dict[str, Any]:
    from shapely.geometry import shape

    if not isinstance(collection, dict) or collection.get("type") != "FeatureCollection":
        raise ValueError("boundaries must be a GeoJSON FeatureCollection")
    features = collection.get("features")
    if not isinstance(features, list) or len(features) != len(AREA_NAMES):
        raise ValueError("boundaries need one polygon each for Gading Serpong and BSD City")
    by_area = {}
    for feature in features:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError("every boundary must be a GeoJSON Feature")
        area = feature.get("properties", {}).get("area")
        if area not in AREA_NAMES or area in by_area:
            raise ValueError(f"boundary area must be unique and in {AREA_NAMES}")
        try:
            polygon = shape(feature["geometry"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid geometry for {area}") from exc
        if (
            polygon.geom_type not in {"Polygon", "MultiPolygon"}
            or polygon.is_empty
            or not polygon.is_valid
        ):
            raise ValueError(f"{area} must have a nonempty valid Polygon or MultiPolygon")
        west, south, east, north = polygon.bounds
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError(f"{area} boundary coordinates must be WGS84 lon/lat")
        if polygon.area <= 0:
            raise ValueError(f"{area} polygon has zero area")
        by_area[area] = feature
    if set(by_area) != set(AREA_NAMES):
        raise ValueError("both study-area boundaries are required")
    return {"type": "FeatureCollection", "features": [by_area[name] for name in AREA_NAMES]}


def _fetch_boundaries(cache_dir: Path) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "osm_ids": ",".join(OSM_RELATIONS[name] for name in AREA_NAMES),
        "format": "jsonv2",
        "polygon_geojson": 1,
    }
    response = requests.get(
        NOMINATIM_URL,
        params=params,
        headers={"User-Agent": _USER_AGENT},
        timeout=45,
    )
    response.raise_for_status()
    records = response.json()
    if not isinstance(records, list):
        raise TypeError("Nominatim returned an unexpected boundary response")
    record_by_id = {f"R{record.get('osm_id')}": record for record in records}
    features = []
    for area in AREA_NAMES:
        osmid = OSM_RELATIONS[area]
        record = record_by_id.get(osmid)
        if not record or "geojson" not in record:
            raise ValueError(f"Nominatim did not return a polygon for {area} ({osmid})")
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "area": area,
                    "osm_id": osmid,
                    "source": "OpenStreetMap/Nominatim",
                    "fetched_utc": _now(),
                },
                "geometry": record["geojson"],
            }
        )
    collection = _validate_boundaries({"type": "FeatureCollection", "features": features})
    (cache_dir / "boundaries.geojson").write_text(
        json.dumps(collection, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return collection


def default_boundaries(cache_dir: str | Path) -> dict[str, Any]:
    """Load cached OSM relation polygons or fetch them once from Nominatim."""
    folder = Path(cache_dir)
    path = folder / "boundaries.geojson"
    if path.is_file():
        return _validate_boundaries(json.loads(path.read_text(encoding="utf-8")))
    return _fetch_boundaries(folder)


def _custom_boundaries(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return _validate_boundaries(value)
    raw = str(value)
    if raw.lstrip().startswith("{"):
        return _validate_boundaries(json.loads(raw))
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"boundary GeoJSON not found: {path}")
    return _validate_boundaries(json.loads(path.read_text(encoding="utf-8")))


def _worldpop_raster(cache_dir: Path, *, refresh: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / WORLDPOP_FILENAME
    if target.is_file() and target.stat().st_size > 0 and not refresh:
        return target
    temporary = target.with_suffix(".tif.partial")
    try:
        with requests.get(
            WORLDPOP_URL,
            headers={"User-Agent": _USER_AGENT},
            stream=True,
            timeout=(30, 120),
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            expected = response.headers.get("Content-Length")
        if expected and temporary.stat().st_size != int(expected):
            raise OSError("WorldPop download was incomplete")
        if temporary.stat().st_size == 0:
            raise OSError("WorldPop download was empty")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _projected_areas(boundaries: dict[str, Any]) -> dict[str, Any]:
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform

    project = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True).transform
    return {
        feature["properties"]["area"]: transform(project, shape(feature["geometry"]))
        for feature in boundaries["features"]
    }


def _population_cells(raster_path: Path, boundaries: dict[str, Any]) -> pd.DataFrame:
    """Assign positive 100 m pixels to 500 m UTM grid cells and sum counts."""
    import rasterio
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.mask import mask
    from rasterio.warp import transform_geom

    with rasterio.open(raster_path) as raster:
        if raster.crs is None:
            raise ValueError("WorldPop raster has no coordinate reference system")
        geometries = [
            transform_geom("EPSG:4326", raster.crs, feature["geometry"])
            for feature in boundaries["features"]
        ]
        clipped, affine = mask(raster, geometries, crop=True, filled=False)
        population = clipped[0]
        area_codes = rasterize(
            [(geometry, i + 1) for i, geometry in enumerate(geometries)],
            out_shape=population.shape,
            transform=affine,
            fill=0,
            dtype="uint8",
        )
        values = np.ma.filled(population, 0).astype(float)
        valid = np.isfinite(values) & (values > 0) & (area_codes > 0)
        rows, columns = np.nonzero(valid)
        if not len(rows):
            return pd.DataFrame(columns=["area", "grid_x", "grid_y", "population"])
        x, y = rasterio.transform.xy(affine, rows, columns, offset="center")
        to_metric = Transformer.from_crs(raster.crs, METRIC_CRS, always_xy=True)
        metric_x, metric_y = to_metric.transform(x, y)
        frame = pd.DataFrame(
            {
                "area": [AREA_NAMES[int(code) - 1] for code in area_codes[rows, columns]],
                "grid_x": np.floor(np.asarray(metric_x) / GRID_SIZE_M).astype(int),
                "grid_y": np.floor(np.asarray(metric_y) / GRID_SIZE_M).astype(int),
                "population": values[rows, columns],
            }
        )
    return frame.groupby(["area", "grid_x", "grid_y"], as_index=False, sort=True)[
        "population"
    ].sum()


def _tag(row: pd.Series, key: str) -> str:
    value = row.get(key)
    if isinstance(value, str):
        return value.strip().lower()
    return ""


def _feature_rows(features: pd.DataFrame, boundaries: dict[str, Any]):
    from pyproj import Transformer
    from shapely.geometry import shape

    polygons = [(f["properties"]["area"], shape(f["geometry"])) for f in boundaries["features"]]
    to_metric = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
    if getattr(features, "crs", None) is not None and str(features.crs).upper() != "EPSG:4326":
        features = features.to_crs("EPSG:4326")
    for index, row in features.iterrows():
        geometry = row.get("geometry")
        if geometry is None or geometry.is_empty:
            continue
        point = geometry.representative_point()
        area = next(
            (name for name, polygon in polygons if polygon.covers(point)), "Outside study area"
        )
        if isinstance(index, tuple) and len(index) >= 2:
            element_type, osmid = index[:2]
        else:
            element_type, osmid = "feature", index
        projected_x, projected_y = to_metric.transform(point.x, point.y)
        yield {
            "id": f"osm_{element_type}_{osmid}",
            "name": str(row.get("name")) if isinstance(row.get("name"), str) else f"OSM {osmid}",
            "lat": float(point.y),
            "lon": float(point.x),
            "area": area,
            "grid_x": int(np.floor(projected_x / GRID_SIZE_M)),
            "grid_y": int(np.floor(projected_y / GRID_SIZE_M)),
            "amenity": _tag(row, "amenity"),
            "shop": _tag(row, "shop"),
            "office": _tag(row, "office"),
            "leisure": _tag(row, "leisure"),
            "access": _tag(row, "access") or "unknown",
        }


def _poi_tables(
    features: pd.DataFrame, boundaries: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = []
    existing = []
    activity = []
    for item in _feature_rows(features, boundaries):
        amenity, shop, access = item["amenity"], item["shop"], item["access"]
        public_access = access not in {"private", "no", "customers", "permit"}
        site_type = (
            "parking"
            if item["area"] in AREA_NAMES and amenity == "parking" and public_access
            else "fuel"
            if item["area"] in AREA_NAMES and amenity == "fuel" and public_access
            else "mall"
            if item["area"] in AREA_NAMES and shop == "mall" and public_access
            else None
        )
        if site_type:
            candidates.append(
                {key: item[key] for key in ("id", "name", "lat", "lon", "area")}
                | {"site_type": site_type}
            )
        # Keep nearby chargers visible as baseline context, except for the
        # specifically reviewed OSM feature that the user identified as out of scope.
        if amenity == "charging_station" and item["id"] != "osm_node_13872408261":
            existing.append(
                {key: item[key] for key in ("id", "name", "lat", "lon", "area", "access")}
                | {"include_default": public_access}
            )
        if item["area"] in AREA_NAMES:
            activity.append(
                {
                    "area": item["area"],
                    "grid_x": item["grid_x"],
                    "grid_y": item["grid_y"],
                    "retail": int(bool(shop)),
                    "office": int(bool(item["office"])),
                    "leisure": int(bool(item["leisure"])),
                    "public": int(amenity in _PUBLIC_AMENITIES),
                }
            )
    candidate_frame = pd.DataFrame(candidates, columns=CANDIDATE_COLUMNS)
    existing_frame = pd.DataFrame(existing, columns=EXISTING_COLUMNS)
    activity_frame = pd.DataFrame(
        activity,
        columns=["area", "grid_x", "grid_y", "retail", "office", "leisure", "public"],
    )
    if not activity_frame.empty:
        activity_frame = activity_frame.groupby(
            ["area", "grid_x", "grid_y"],
            as_index=False,
            sort=True,
        )[["retail", "office", "leisure", "public"]].sum()
    return candidate_frame, existing_frame, activity_frame


def _demand_table(
    population: pd.DataFrame,
    activities: pd.DataFrame,
    boundaries: dict[str, Any],
) -> pd.DataFrame:
    from pyproj import Transformer
    from shapely.geometry import box

    cells = population.merge(activities, on=["area", "grid_x", "grid_y"], how="outer")
    if cells.empty:
        return pd.DataFrame(columns=DEMAND_COLUMNS)
    for column in ("population", "retail", "office", "leisure", "public"):
        cells[column] = cells[column].fillna(0.0)
    cells = cells.loc[
        (cells[["population", "retail", "office", "leisure", "public"]] > 0).any(axis=1)
    ].sort_values(["area", "grid_x", "grid_y"], kind="stable")
    projected = _projected_areas(boundaries)
    to_geo = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)
    records = []
    for row in cells.itertuples(index=False):
        cell = box(
            row.grid_x * GRID_SIZE_M,
            row.grid_y * GRID_SIZE_M,
            (row.grid_x + 1) * GRID_SIZE_M,
            (row.grid_y + 1) * GRID_SIZE_M,
        )
        inside = cell.intersection(projected[row.area])
        centre = inside.representative_point() if not inside.is_empty else cell.centroid
        lon, lat = to_geo.transform(centre.x, centre.y)
        slug = "gading_serpong" if row.area == "Gading Serpong" else "bsd_city"
        records.append(
            {
                "id": f"grid_{slug}_{row.grid_x}_{row.grid_y}",
                "lat": lat,
                "lon": lon,
                "area": row.area,
                "population": float(row.population),
                "retail": int(row.retail),
                "office": int(row.office),
                "leisure": int(row.leisure),
                "public": int(row.public),
            }
        )
    return pd.DataFrame.from_records(records, columns=DEMAND_COLUMNS)


def _query_osm(boundaries: dict[str, Any], cache_dir: Path, *, refresh: bool):
    import osmnx as ox
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform, unary_union

    ox.settings.use_cache = not refresh
    ox.settings.cache_folder = str(cache_dir / "osmnx")
    polygon = unary_union([shape(f["geometry"]) for f in boundaries["features"]])
    to_metric = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True).transform
    to_geo = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True).transform
    buffered = transform(to_geo, transform(to_metric, polygon).buffer(ROAD_BUFFER_M))
    graph = ox.graph_from_polygon(buffered, network_type="drive", retain_all=True)
    feature_parts = []
    for feature in boundaries["features"]:
        try:
            feature_parts.append(
                ox.features_from_polygon(shape(feature["geometry"]), tags=_FEATURE_TAGS)
            )
        except ox._errors.InsufficientResponseError:
            # An area with zero matching POIs should not erase results from
            # the other area or prevent population demand from being used.
            pass
    # Nearby SPKLU just outside either study polygon can still serve demand.
    # This narrow query uses the same 2 km road buffer as the routing graph.
    try:
        feature_parts.append(
            ox.features_from_polygon(buffered, tags={"amenity": "charging_station"})
        )
    except ox._errors.InsufficientResponseError:
        pass
    features = pd.concat(feature_parts) if feature_parts else pd.DataFrame()
    features = features.loc[~features.index.duplicated(keep="first")]
    return graph, features, ox.__version__


def prepare_scenario(
    cache_dir: str | Path,
    boundaries_geojson: str | Path | dict[str, Any] | None = None,
    refresh: bool = False,
) -> Scenario:
    """Build or load a real cached OSM + WorldPop scenario.

    This can download a 169 MB national raster on first use. Source failures
    are raised; callers may explicitly choose ``demo_scenario`` for offline use.
    """
    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    if boundaries_geojson is None:
        boundaries = _fetch_boundaries(folder) if refresh else default_boundaries(folder)
        snapshot_dir = folder / "scenario"
    else:
        boundaries = _custom_boundaries(boundaries_geojson)
        snapshot_dir = folder / f"scenario_{_json_hash(boundaries)[:12]}"
    if (snapshot_dir / "metadata.json").is_file() and not refresh:
        return _load_scenario(snapshot_dir)

    raster_path = _worldpop_raster(folder, refresh=refresh)
    graph, features, osmnx_version = _query_osm(boundaries, folder, refresh=refresh)
    candidates, existing, activities = _poi_tables(features, boundaries)
    if candidates.empty:
        raise ValueError(
            "OSM returned no public parking, fuel, or mall candidate sites in the study areas"
        )
    population = _population_cells(raster_path, boundaries)
    demand = _demand_table(population, activities, boundaries)
    if demand.empty:
        raise ValueError("WorldPop and OSM produced no demand cells within the study areas")
    metadata = {
        "mode": "real",
        "created_utc": _now(),
        "study_areas": list(AREA_NAMES),
        "grid_size_m": GRID_SIZE_M,
        "road_buffer_m": ROAD_BUFFER_M,
        "projected_crs": METRIC_CRS,
        "sources": {
            "boundaries": {
                "url": NOMINATIM_URL,
                "osm_relations": OSM_RELATIONS,
                "sha256": _json_hash(boundaries),
                "fetched_utc": [f["properties"].get("fetched_utc") for f in boundaries["features"]],
                "note": "OpenStreetMap community polygons are approximate and editable.",
            },
            "worldpop": {
                "url": WORLDPOP_URL,
                "sha256": _sha256(raster_path),
                "edition": "Global2 R2025A, 2025, constrained, 100 m, Indonesia",
            },
            "osm": {
                "url": "https://www.openstreetmap.org/copyright",
                "retrieved_utc": _now(),
                "osmnx_version": osmnx_version,
                "feature_tags": _FEATURE_TAGS,
                "network_type": "drive",
                "candidate_sha256": _json_hash(candidates.to_dict(orient="records")),
                "existing_sha256": _json_hash(existing.to_dict(orient="records")),
                "activity_sha256": _json_hash(activities.to_dict(orient="records")),
            },
        },
        "attribution": "© OpenStreetMap contributors; WorldPop (www.worldpop.org).",
        "notes": [
            "OSM parking/fuel/mall POIs are candidate proxies, not verified land or grid availability.",
            "OSM charging stations with unknown access are included by default; verify on site.",
            "WorldPop is modeled population, not measured EV ownership or charging demand.",
        ],
    }
    empty_candidates = np.empty((len(demand), len(candidates)), dtype=float)
    empty_existing = np.empty((len(demand), len(existing)), dtype=float)
    scenario = Scenario(
        candidates=candidates,
        demand=demand,
        existing=existing,
        boundaries=boundaries,
        graph=graph,
        metadata=metadata,
        distances_m=empty_candidates,
        existing_distances_m=empty_existing,
    )
    scenario = _rebuild_distances(scenario)
    _save_scenario(scenario, snapshot_dir)
    return scenario


def demo_scenario(seed: int = 42) -> Scenario:
    """Create a small, reproducible, unmistakably synthetic offline example."""
    rng = np.random.default_rng(seed)
    areas = [
        ("Gading Serpong", 106.600, -6.282, 106.652, -6.224),
        ("BSD City", 106.593, -6.342, 106.704, -6.283),
    ]
    boundaries = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"area": name, "source": "synthetic_demo_rectangle"},
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
            for name, west, south, east, north in areas
        ],
    }
    candidate_rows = []
    demand_rows = []
    existing_rows = []
    for area_index, (name, west, south, east, north) in enumerate(areas):
        for site_index in range(9):
            candidate_rows.append(
                {
                    "id": f"demo_c_{area_index}_{site_index}",
                    "name": f"Synthetic {('parking', 'fuel', 'mall')[site_index % 3]} {site_index + 1}",
                    "lat": rng.uniform(south + 0.003, north - 0.003),
                    "lon": rng.uniform(west + 0.003, east - 0.003),
                    "area": name,
                    "site_type": ("parking", "fuel", "mall")[site_index % 3],
                }
            )
        for demand_index in range(28):
            demand_rows.append(
                {
                    "id": f"demo_d_{area_index}_{demand_index}",
                    "lat": rng.uniform(south + 0.003, north - 0.003),
                    "lon": rng.uniform(west + 0.003, east - 0.003),
                    "area": name,
                    "population": float(rng.integers(250, 2200)),
                    "retail": int(rng.poisson(2.0)),
                    "office": int(rng.poisson(1.5)),
                    "leisure": int(rng.poisson(0.7)),
                    "public": int(rng.poisson(0.8)),
                }
            )
        existing_rows.append(
            {
                "id": f"demo_e_{area_index}",
                "name": f"Synthetic existing station {area_index + 1}",
                "lat": (south + north) / 2,
                "lon": (west + east) / 2,
                "area": name,
                "access": "unknown",
                "include_default": True,
            }
        )
    candidates = pd.DataFrame.from_records(candidate_rows, columns=CANDIDATE_COLUMNS)
    demand = pd.DataFrame.from_records(demand_rows, columns=DEMAND_COLUMNS)
    existing = pd.DataFrame.from_records(existing_rows, columns=EXISTING_COLUMNS)
    scenario = Scenario(
        candidates=candidates,
        demand=demand,
        existing=existing,
        boundaries=boundaries,
        graph=None,
        metadata={
            "mode": "synthetic",
            "created_utc": _now(),
            "seed": int(seed),
            "study_areas": list(AREA_NAMES),
            "source": "Procedurally generated demo; no real sites, demand, roads, or boundaries.",
            "attribution": "Synthetic example generated by this application.",
        },
        distances_m=np.empty((len(demand), len(candidates))),
        existing_distances_m=np.empty((len(demand), len(existing))),
    )
    return _rebuild_distances(scenario)
