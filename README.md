# SPKLU site screening: Gading Serpong and BSD City

This project compares a custom genetic algorithm (GA) and particle swarm optimization (PSO) for selecting **new public passenger-car charging sites** in Gading Serpong and BSD City, Tangerang. It includes a native Python desktop application and a command-line workflow.

The selected points are a **preliminary shortlist**. Mapped parking or retail sites are not proof of available land, permission, charger compatibility, spare grid capacity, safe access, or financial viability. Check those items on site before using a recommendation for deployment.

## Quick start

Install [uv](https://docs.astral.sh/uv/) and run from the project directory:

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
$env:UV_PROJECT_ENVIRONMENT = "$PWD\.venv-desktop"
uv sync --extra dev
uv run spklu-gui
```

The desktop window opens the cached real scenario at `data/real/scenario` when available. Otherwise it opens an **explicitly synthetic demo**. From the window you can edit either area directly on the map, review current SPKLU and candidate locations, import custom CSV tables, change model and optimizer settings, animate GA and PSO by generation/iteration, compare results, and export selected sites or boundaries. Use **Save scenario…** and **Open scenario…** to keep working with edited scenario snapshots. On Windows, the same app can be started with `uv run python app.py`.

To build or refresh a real public-data scenario, run:

```powershell
uv run spklu prepare --output data/real
uv run spklu info --scenario data/real
uv run spklu compare --scenario data/real --mode separate --gading-count 3 --bsd-count 3 --output data/results
```

The first real preparation downloads the [WorldPop 2025 Indonesia population raster](https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/IDN/v1/100m/constrained/idn_pop_2025_CN_100m_R2025A_v1.tif) (about 169 MB) and queries OpenStreetMap through Nominatim and Overpass. This may take several minutes. The inputs and road-distance matrix are cached under `data/real`; the desktop app and GA/PSO runs use the cache. To update public sources, prepare again with `--refresh` and record the new snapshot date.

For a combined allocation across the two areas:

```powershell
uv run spklu compare --scenario data/real --mode combined --count 6 --output data/results_combined
```

Run checks with `uv run pytest` and `uv run ruff check .`.

## Model

- Initial, editable study polygons come from [OSM relation 6543247 (Gading Serpong)](https://www.openstreetmap.org/relation/6543247) and [OSM relation 10759331 (BSD City)](https://www.openstreetmap.org/relation/10759331). These are community mapped township extents, not legal boundaries. You can replace them with reviewed GeoJSON.
- New-location search points default to a configurable grid of nearby OSM road-network nodes sampled inside both study areas. This lets GA/PSO explore beyond mapped parking and mall POIs. A second mode restricts the search to mapped parking, fuel, and mall host candidates. Generated points are screening locations only; the model does not verify land ownership, site access, permits, or grid capacity. Existing mapped car-charging stations form fixed baseline supply. The road graph preserves one-way travel when computing **demand-to-site driving distance**.
- Demand is a tunable blend of [WorldPop Global 2](https://www.worldpop.org/datacatalog/) residential population and selected OSM activity destinations, aggregated to local grid cells. Both are proxies: neither is a measured count of EVs or charging sessions. The population and activity components each sum to one before blending; if one is absent, the available component is used.
- For exactly `k` new sites, the shared score is `distance_weight × (weighted mean driving distance / coverage_radius) + (1 − distance_weight) × uncovered demand share`. Demand is covered when a selected or included existing site is within the driving-distance radius. Lower is better. The app also reports coverage, mean and 90th-percentile distance, area summaries, and repeated-run statistics.
- GA selects an exact-size candidate subset through genetic operators. PSO uses continuous random keys whose highest-ranked candidates form the same exact-size subset. Both evaluate the same score and data with matched evaluation budgets and seeds. Greedy and random selections provide reference baselines.

Adjusting the score weights or radius changes the planning question. Keep those settings equal when comparing GA and PSO. Combined mode permits all new sites to fall in either area; separate mode sets a count for each area.

## Inputs and outputs

The desktop app exposes map-based study-boundary editing, candidate and existing-site review, demand blend and activity category weights, site counts and coverage controls, and GA/PSO parameters. Boundary edits are converted to GeoJSON and applied to the active scenario; the edited scenario can be saved as a portable snapshot. Candidate locations form the algorithms' search pool; the selected GA and PSO recommendations are displayed separately on the map. The synthetic demo uses approximate straight-line distances and must not be treated as a real shortlist.

CSV columns:

| Table | Required columns | Notes |
| --- | --- | --- |
| Candidates | `id,name,lat,lon,area,site_type` | `site_type`: `parking`, `fuel`, or `mall`; `area`: `Gading Serpong` or `BSD City`. |
| Demand | `id,lat,lon,area,population,retail,office,leisure,public` | Numeric demand columns must be nonnegative; units are proxy counts. |
| Existing | `id,name,lat,lon,area,access,include_default` | Review public access and operational status before including a station. |

The CLI comparison writes `metrics.csv`, `convergence.csv`, `selected_sites.geojson`, and `summary.json`. Keep scenario metadata and parameter settings with any map or ranking you share.

## Data and use conditions

Map and POI data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) (ODbL). Population estimates © [WorldPop](https://www.worldpop.org/faq/) (check the specific dataset metadata for attribution and license conditions). The public map can be incomplete, especially for charger access and operating status. Indonesia's [ESDM Regulation 1/2023](https://jdih.esdm.go.id/common/dokumen-external/Permen%20ESDM%20Nomor%201%20Tahun%202023.pdf) includes requirements for accessible SPKLU sites, dedicated parking, safety, and traffic; the model does not certify compliance.
