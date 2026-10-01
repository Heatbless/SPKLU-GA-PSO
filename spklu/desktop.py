"""Native desktop GUI for interactive SPKLU GA/PSO site planning."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union

from spklu.analysis import compute_demand_weights
from spklu.data import demo_scenario
from spklu.optimization import (
    Problem,
    optimize_ga,
    optimize_greedy,
    optimize_pso,
    optimize_random,
)
from spklu.scenario import Scenario, load_scenario, rebuild_distances, save_scenario

ROOT = Path(__file__).resolve().parents[1]
REAL_SCENARIO = ROOT / "data" / "real" / "scenario"
AREAS = ("Gading Serpong", "BSD City")
AREA_COLORS = {"Gading Serpong": "#2563eb", "BSD City": "#e11d48"}


def _feature_area(feature: dict[str, Any]) -> str:
    return str((feature.get("properties") or {}).get("area", ""))


def _filter_points(frame: pd.DataFrame, boundaries: dict[str, Any]) -> pd.DataFrame:
    polygons = {
        _feature_area(feature): shape(feature["geometry"])
        for feature in boundaries.get("features", [])
    }
    inside = [
        bool(polygon is not None and polygon.covers(Point(float(row.lon), float(row.lat))))
        for row in frame.itertuples(index=False)
        for polygon in [polygons.get(str(row.area))]
    ]
    return frame.loc[inside].reset_index(drop=True)


def _road_grid_candidates(scenario: Scenario, spacing_m: int = 500) -> pd.DataFrame:
    """Sample road-network nodes inside each area at a configurable grid spacing."""
    if scenario.graph is None:
        raise ValueError("Road-grid search points require a real road network.")
    metric_crs = (scenario.metadata or {}).get("projected_crs", "EPSG:32748")
    graph_crs = scenario.graph.graph.get("crs", "EPSG:4326")
    to_metric = Transformer.from_crs(graph_crs, metric_crs, always_xy=True)
    node_ids = list(scenario.graph.nodes)
    lons = np.array([float(scenario.graph.nodes[node]["x"]) for node in node_ids])
    lats = np.array([float(scenario.graph.nodes[node]["y"]) for node in node_ids])
    eastings, northings = to_metric.transform(lons, lats)
    road_points = shapely.points(eastings, northings)
    area_names = ("Gading Serpong", "BSD City")
    records: list[dict[str, Any]] = []
    existing_rows = scenario.existing
    if not existing_rows.empty:
        ex, ey = to_metric.transform(
            existing_rows["lon"].to_numpy(dtype=float),
            existing_rows["lat"].to_numpy(dtype=float),
        )
        existing_points = shapely.points(ex, ey)
    else:
        existing_points = np.array([], dtype=object)

    for feature in scenario.boundaries.get("features", []):
        area = _feature_area(feature)
        if area not in area_names:
            continue
        projected_area = shapely.transform(
            shape(feature["geometry"]),
            lambda coords: np.column_stack(to_metric.transform(coords[:, 0], coords[:, 1])),
            include_z=False,
        )
        in_area = np.flatnonzero(shapely.covers(projected_area, road_points))
        if not len(in_area):
            continue
        west, south, _, _ = projected_area.bounds
        cells: dict[tuple[int, int], list[int]] = {}
        for node_index in in_area:
            cell = (
                int((eastings[node_index] - west) // spacing_m),
                int((northings[node_index] - south) // spacing_m),
            )
            cells.setdefault(cell, []).append(int(node_index))
        area_slug = "gading_serpong" if area == "Gading Serpong" else "bsd_city"
        for (cell_x, cell_y), candidates in cells.items():
            center_x = west + (cell_x + 0.5) * spacing_m
            center_y = south + (cell_y + 0.5) * spacing_m
            node_index = min(
                candidates,
                key=lambda index: (eastings[index] - center_x) ** 2 + (northings[index] - center_y) ** 2,
            )
            point = road_points[node_index]
            if len(existing_points) and np.min(shapely.distance(existing_points, point)) < 120:
                continue
            node_x = round(float(eastings[node_index]))
            node_y = round(float(northings[node_index]))
            records.append({
                "id": f"road_{area_slug}_{node_x}_{node_y}",
                "name": f"{area} road-grid point {len(records) + 1:03d}",
                "lat": lats[node_index],
                "lon": lons[node_index],
                "area": area,
                "site_type": "road_grid_node",
            })
    if not records:
        raise ValueError("No road-network search points were found inside the selected study areas.")
    return pd.DataFrame(records, columns=["id", "name", "lat", "lon", "area", "site_type"])


class MapBridge(QObject):
    boundaryChanged = Signal(str)

    @Slot(str)
    def saveBoundaries(self, value: str) -> None:
        self.boundaryChanged.emit(value)


class OptimizationWorker(QThread):
    progress = Signal(str, int, int, int, float, object)
    finishedRuns = Signal(object, object, object)
    failed = Signal(str)

    def __init__(self, problem_args: dict[str, Any], settings: dict[str, Any], parent=None):
        super().__init__(parent)
        self.problem_args = problem_args
        self.settings = settings

    def run(self) -> None:
        try:
            a = self.problem_args
            s = self.settings
            problem = Problem(
                distances_m=a["distances_m"],
                existing_distances_m=a["existing_distances_m"],
                demand_weights=a["demand_weights"],
                candidate_areas=a["candidate_areas"],
                k=s["k"],
                radius_m=s["radius_m"],
                distance_weight=s["distance_weight"],
                per_area_counts=s["per_area_counts"],
            )
            runs: dict[str, list[Any]] = {"GA": [], "PSO": [], "Random": [], "Greedy": []}
            for repeat in range(s["repetitions"]):
                seed = s["seed"] + repeat
                for label, function in (("GA", optimize_ga), ("PSO", optimize_pso)):
                    def on_epoch(epoch: int, total: int, indices: tuple[int, ...], score: float,
                                 algorithm=label, repeat_number=repeat + 1) -> None:
                        self.progress.emit(
                            algorithm, repeat_number, epoch, total,
                            float(score), tuple(int(i) for i in indices),
                        )
                        if s["animation_delay"]:
                            time.sleep(s["animation_delay"])

                    if label == "GA":
                        result = function(
                            problem,
                            population_size=s["population_size"],
                            max_evaluations=s["max_evaluations"],
                            mutation_rate=s["mutation_rate"],
                            crossover_rate=s["crossover_rate"],
                            seed=seed,
                            on_epoch=on_epoch,
                        )
                    else:
                        result = function(
                            problem,
                            swarm_size=s["swarm_size"],
                            max_evaluations=s["max_evaluations"],
                            inertia=s["inertia"],
                            cognitive=s["cognitive"],
                            social=s["social"],
                            seed=seed,
                            on_epoch=on_epoch,
                        )
                    runs[label].append(result)
                runs["Random"].append(
                    optimize_random(problem, seed=seed, samples=s["max_evaluations"])
                )
            runs["Greedy"] = [optimize_greedy(problem)]
            self.finishedRuns.emit(runs, problem, a["candidates"])
        except Exception as exc:  # noqa: BLE001 - report worker errors in the window
            self.failed.emit(str(exc))


class CandidatePoolWorker(QThread):
    ready = Signal(object, object, str)
    failed = Signal(str)

    def __init__(self, scenario: Scenario, candidates: pd.DataFrame, mode: str, parent=None):
        super().__init__(parent)
        self.scenario = scenario
        self.candidates = candidates
        self.mode = mode

    def run(self) -> None:
        try:
            scenario = rebuild_distances(self.scenario, candidates=self.candidates)
            self.ready.emit(scenario, self.candidates, self.mode)
        except Exception as exc:  # noqa: BLE001 - return matrix errors to the window
            self.failed.emit(str(exc))


def _map_html(scenario: Scenario) -> str:
    center = [
        float(scenario.demand["lat"].median()),
        float(scenario.demand["lon"].median()),
    ]
    boundaries = json.dumps(scenario.boundaries, ensure_ascii=False)
    existing = json.dumps(scenario.existing.to_dict("records"), default=str, ensure_ascii=False)
    candidates = json.dumps(scenario.candidates.to_dict("records"), default=str, ensure_ascii=False)
    area_colors = json.dumps(AREA_COLORS)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css">
<style>html,body,#map{{height:100%;margin:0;font:14px sans-serif}}.leaflet-popup-content{{line-height:1.6}}.legend{{background:#fff;padding:8px 12px;border-radius:5px;box-shadow:0 1px 5px #888}}.legend i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}</style>
</head><body><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.js"></script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script>
const boundaries={boundaries}, existing={existing}, candidates={candidates}, colors={area_colors};
const map=L.map('map').setView({json.dumps(center)},12);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19,attribution:'&copy; OpenStreetMap contributors'}}).addTo(map);
const editGroup=new L.FeatureGroup().addTo(map),candidateLayer=L.layerGroup().addTo(map),existingLayer=L.layerGroup().addTo(map),resultLayer=L.layerGroup().addTo(map);
function boundaryStyle(feature){{return {{color:colors[feature.properties.area]||'#7c3aed',weight:3,fillOpacity:.06}}}}
function drawBoundaries(data){{editGroup.clearLayers();data.features.forEach(f=>{{const parts=f.geometry.type==='MultiPolygon'?f.geometry.coordinates.map(coords=>({{type:'Polygon',coordinates:coords}})):[f.geometry];parts.forEach(g=>{{const piece={{type:'Feature',properties:f.properties,geometry:g}};L.geoJSON(piece,{{style:boundaryStyle,onEachFeature:(feature,layer)=>{{layer.feature=feature;layer.bindTooltip(feature.properties.area);editGroup.addLayer(layer)}}}})}})}})}}
drawBoundaries(boundaries);
existing.forEach(p=>{{const c=p.include_default===false?'#6b7280':'#15803d';L.circleMarker([+p.lat,+p.lon],{{radius:9,color:'#fff',weight:2,fillColor:c,fillOpacity:1}}).bindPopup('<b>Existing SPKLU</b><br>'+p.name+'<br>'+p.area).addTo(existingLayer)}});
candidates.forEach(p=>L.circleMarker([+p.lat,+p.lon],{{radius:4,color:colors[p.area]||'#8b5cf6',fillOpacity:.75}}).bindTooltip('Candidate: '+p.name+' · '+p.area).addTo(candidateLayer));
const bounds=editGroup.getBounds();if(bounds.isValid())map.fitBounds(bounds.pad(.05));
const drawControl=new L.Control.Draw({{position:'topleft',draw:{{polyline:false,rectangle:false,circle:false,circlemarker:false,marker:false,polygon:{{allowIntersection:false,showArea:true}}}},edit:{{featureGroup:editGroup,selectedPathOptions:{{maintainColor:true}},remove:false}}}});map.addControl(drawControl);
function emitBoundaries(){{const geo=editGroup.toGeoJSON();window.bridge.saveBoundaries(JSON.stringify(geo))}}
map.on(L.Draw.Event.CREATED,e=>{{const area=window.selectedArea||'BSD City';const f=e.layer.toGeoJSON();f.properties={{...(f.properties||{{}}),area}};const remove=[];editGroup.eachLayer(l=>{{if(l.feature&&l.feature.properties&&l.feature.properties.area===area)remove.push(l)}});remove.forEach(l=>editGroup.removeLayer(l));e.layer.feature=f;e.layer.bindTooltip(area);editGroup.addLayer(e.layer);emitBoundaries()}});
map.on(L.Draw.Event.EDITED,emitBoundaries);
window.setSelectedArea=a=>window.selectedArea=a;
window.updateOptimizerSites=(algorithm,sites)=>{{resultLayer.clearLayers();sites.forEach((p,i)=>{{const color=algorithm==='GA'?'#2563eb':'#f97316';L.circleMarker([+p.lat,+p.lon],{{radius:10,color:'#fff',weight:3,fillColor:color,fillOpacity:.95}}).bindPopup('<b>'+algorithm+' site '+(i+1)+'</b><br>'+p.name+'<br>'+p.area).addTo(resultLayer)}})}};
window.clearOptimizerSites=()=>resultLayer.clearLayers();
window.setCandidatePool=sites=>{{candidateLayer.clearLayers();sites.forEach(p=>L.circleMarker([+p.lat,+p.lon],{{radius:4,color:colors[p.area]||'#8b5cf6',fillOpacity:.75}}).bindTooltip('Candidate: '+p.name+' · '+p.area).addTo(candidateLayer))}};
L.control.layers(null,{{'Editable study boundaries':editGroup,'Current SPKLU':existingLayer,'Candidate search pool':candidateLayer,'Live GA/PSO selection':resultLayer}},{{collapsed:false}}).addTo(map);
new QWebChannel(qt.webChannelTransport,channel=>{{window.bridge=channel.objects.bridge}});
</script></body></html>"""


class SPKLUMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SPKLU Site Optimizer · Gading Serpong + BSD City")
        self.resize(1500, 960)
        self.setMinimumSize(1100, 720)
        self.scenario: Scenario = self._load_default_scenario()
        self.source_label = "Cached real scenario" if self.scenario.metadata.get("mode") == "real" else "Synthetic demonstration"
        self.road_grid_spacing_m = 1000
        self.osm_candidates: pd.DataFrame | None = None
        self.road_grid_pool: pd.DataFrame | None = None
        self.search_mode = "Mapped candidate locations"
        self._configure_candidate_pools()
        self.worker: OptimizationWorker | None = None
        self.pool_worker: CandidatePoolWorker | None = None
        self.results: dict[str, list[Any]] | None = None
        self.result_candidates: pd.DataFrame | None = None
        self.result_problem: Problem | None = None
        self.map_view = QWebEngineView()
        self.bridge = MapBridge(self)
        self.channel = QWebChannel(self.map_view.page())
        self.channel.registerObject("bridge", self.bridge)
        self.map_view.page().setWebChannel(self.channel)
        self.bridge.boundaryChanged.connect(self._apply_map_boundaries)
        self._build_ui()
        self._refresh_map()
        self._populate_candidate_table()
        self._populate_existing_table()
        if getattr(self, "pool_build_pending", False):
            self.run_button.setEnabled(False)
            self.status.setText("Preparing area-wide road-grid recommendations…")
            QTimer.singleShot(150, self._start_initial_pool_build)

    @staticmethod
    def _load_default_scenario() -> Scenario:
        try:
            return load_scenario(REAL_SCENARIO)
        except Exception:  # noqa: BLE001 - missing or invalid cache uses an explicitly labelled demo
            return demo_scenario(seed=42)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        bar = QHBoxLayout()
        title = QLabel("SPKLU Site Optimizer")
        title.setStyleSheet("font-size:22px;font-weight:700")
        bar.addWidget(title)
        bar.addStretch(1)
        self.source_label_widget = QLabel(self.source_label)
        self.source_label_widget.setStyleSheet("color:#536173;font-weight:600")
        bar.addWidget(self.source_label_widget)
        open_button = QPushButton("Open scenario…")
        open_button.clicked.connect(self._open_scenario)
        bar.addWidget(open_button)
        self.open_button = open_button
        self.load_real_button = QPushButton("Load cached real data")
        self.load_real_button.clicked.connect(self._load_real)
        bar.addWidget(self.load_real_button)
        demo_button = QPushButton("Demo data")
        demo_button.clicked.connect(self._load_demo)
        bar.addWidget(demo_button)
        self.demo_button = demo_button
        save_button = QPushButton("Save scenario…")
        save_button.clicked.connect(self._save_scenario)
        bar.addWidget(save_button)
        export_boundary_button = QPushButton("Export boundaries…")
        export_boundary_button.clicked.connect(self._export_boundaries)
        bar.addWidget(export_boundary_button)
        root_layout.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(350)
        left_scroll.setMaximumWidth(470)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)

        map_edit = QGroupBox("Edit study areas on the map")
        map_form = QFormLayout(map_edit)
        self.edit_area = QComboBox()
        self.edit_area.addItems(AREAS)
        self.edit_area.currentTextChanged.connect(self._select_map_area)
        map_form.addRow("Area to edit", self.edit_area)
        map_form.addRow(QLabel("Draw a replacement polygon or click the edit tool to reshape existing boundaries. Edits update the scenario and map immediately."))
        controls_layout.addWidget(map_edit)

        demand_group = QGroupBox("Demand model")
        demand_form = QFormLayout(demand_group)
        self.population_share = QDoubleSpinBox()
        self.population_share.setRange(0, 1)
        self.population_share.setSingleStep(.05)
        self.population_share.setValue(.5)
        demand_form.addRow("WorldPop share", self.population_share)
        self.activity_spins: dict[str, QDoubleSpinBox] = {}
        for category in ("retail", "office", "leisure", "public"):
            spin = QDoubleSpinBox()
            spin.setRange(0, 3)
            spin.setSingleStep(.1)
            spin.setValue(1)
            demand_form.addRow(f"{category.title()} activity", spin)
            self.activity_spins[category] = spin
        controls_layout.addWidget(demand_group)

        candidates_group = QGroupBox("New SPKLU location search points")
        candidates_layout = QVBoxLayout(candidates_group)
        candidates_layout.addWidget(QLabel("GA and PSO select sites from this pool. Road-grid points explore the area instead of limiting results to currently mapped host locations."))
        self.search_mode_combo = QComboBox()
        self.search_mode_combo.addItem("Generated road-grid points", "road_grid")
        if self.osm_candidates is not None:
            self.search_mode_combo.addItem("Mapped parking / fuel / mall locations", "mapped")
        initial_mode = "road_grid" if self.search_mode == "Generated road-grid points" else "mapped"
        self.search_mode_combo.setCurrentIndex(max(0, self.search_mode_combo.findData(initial_mode)))
        candidates_layout.addWidget(self.search_mode_combo)
        grid_controls = QHBoxLayout()
        grid_controls.addWidget(QLabel("Grid spacing (m)"))
        self.grid_spacing = self._int_spin(250, 2000, self.road_grid_spacing_m, 50)
        grid_controls.addWidget(self.grid_spacing)
        self.regenerate_grid_button = QPushButton("Generate points")
        self.regenerate_grid_button.clicked.connect(self._regenerate_road_grid)
        grid_controls.addWidget(self.regenerate_grid_button)
        candidates_layout.addLayout(grid_controls)
        self.candidate_table = QTableWidget()
        self.candidate_table.setMinimumHeight(200)
        self.candidate_table.itemChanged.connect(self._candidate_changed)
        candidates_layout.addWidget(self.candidate_table)
        controls_layout.addWidget(candidates_group)
        self.search_mode_combo.currentIndexChanged.connect(self._candidate_mode_changed)
        self.regenerate_grid_button.setEnabled(self.scenario.graph is not None)

        existing_group = QGroupBox("Current SPKLU baseline")
        existing_layout = QVBoxLayout(existing_group)
        self.existing_table = QTableWidget()
        self.existing_table.setMinimumHeight(140)
        existing_layout.addWidget(self.existing_table)
        controls_layout.addWidget(existing_group)

        import_group = QGroupBox("Customize input data")
        import_layout = QHBoxLayout(import_group)
        self.import_kind = QComboBox()
        self.import_kind.addItems(["Candidate sites", "Current SPKLU", "Demand points"])
        import_layout.addWidget(self.import_kind)
        self.import_button = QPushButton("Import CSV…")
        self.import_button.clicked.connect(self._import_csv)
        import_layout.addWidget(self.import_button)
        controls_layout.addWidget(import_group)

        settings_group = QGroupBox("GA + PSO settings")
        settings_form = QFormLayout(settings_group)
        self.allocation = QComboBox()
        self.allocation.addItems(["Combined total", "Exact counts by area"])
        settings_form.addRow("Site allocation", self.allocation)
        self.site_count = self._int_spin(1, 500, 6)
        settings_form.addRow("New sites (total)", self.site_count)
        self.gading_count = self._int_spin(0, 500, 3)
        self.bsd_count = self._int_spin(0, 500, 3)
        settings_form.addRow("Gading Serpong count", self.gading_count)
        settings_form.addRow("BSD City count", self.bsd_count)
        self.radius = self._int_spin(100, 30000, 3000, 100)
        settings_form.addRow("Service radius (m)", self.radius)
        self.distance_weight = QDoubleSpinBox()
        self.distance_weight.setRange(0, 1)
        self.distance_weight.setSingleStep(.05)
        self.distance_weight.setValue(.5)
        settings_form.addRow("Distance weight", self.distance_weight)
        self.evaluations = self._int_spin(50, 100000, 2000, 50)
        settings_form.addRow("Evaluations per method", self.evaluations)
        self.ga_population = self._int_spin(4, 500, 40, 2)
        settings_form.addRow("GA population", self.ga_population)
        self.pso_swarm = self._int_spin(4, 500, 40, 2)
        settings_form.addRow("PSO swarm", self.pso_swarm)
        self.repetitions = self._int_spin(1, 30, 3)
        settings_form.addRow("Repeated runs", self.repetitions)
        self.seed = self._int_spin(0, 2_147_483_647, 42)
        settings_form.addRow("First seed", self.seed)
        self.mutation = self._double_spin(0, 1, .2, .05)
        settings_form.addRow("GA mutation", self.mutation)
        self.crossover = self._double_spin(0, 1, .8, .05)
        settings_form.addRow("GA crossover", self.crossover)
        self.inertia = self._double_spin(0, 1.5, .7, .05)
        settings_form.addRow("PSO inertia", self.inertia)
        self.cognitive = self._double_spin(0, 4, 1.5, .1)
        settings_form.addRow("PSO cognitive", self.cognitive)
        self.social = self._double_spin(0, 4, 1.5, .1)
        settings_form.addRow("PSO social", self.social)
        self.animation_delay = self._double_spin(0, 1, .1, .05, 2)
        settings_form.addRow("Animation delay / epoch (s)", self.animation_delay)
        controls_layout.addWidget(settings_group)
        self.run_button = QPushButton("Run GA + PSO")
        self.run_button.setStyleSheet("font-weight:bold;padding:9px;background:#166534;color:white")
        self.run_button.clicked.connect(self._run_optimization)
        controls_layout.addWidget(self.run_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        controls_layout.addWidget(self.progress)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        controls_layout.addWidget(self.status)
        controls_layout.addStretch(1)
        left_scroll.setWidget(controls)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.map_view, 3)
        self.output_tabs = QTabWidget()
        self.summary_table = QTableWidget()
        self.output_tabs.addTab(self.summary_table, "Comparison")
        self.selected_table = QTableWidget()
        self.output_tabs.addTab(self.selected_table, "Selected sites")
        right_layout.addWidget(self.output_tabs, 1)
        output_buttons = QHBoxLayout()
        self.result_method = QComboBox()
        self.result_method.addItems(["GA", "PSO", "Greedy", "Random"])
        self.result_method.currentTextChanged.connect(self._show_final_result)
        output_buttons.addWidget(QLabel("Display result"))
        output_buttons.addWidget(self.result_method)
        export_button = QPushButton("Export selected sites…")
        export_button.clicked.connect(self._export_sites)
        output_buttons.addWidget(export_button)
        output_buttons.addStretch(1)
        right_layout.addLayout(output_buttons)
        split.addWidget(left_scroll)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        root_layout.addWidget(split)
        self.setCentralWidget(root)
        self._sync_count_bounds()

    @staticmethod
    def _int_spin(minimum: int, maximum: int, value: int, step: int = 1) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    @staticmethod
    def _double_spin(minimum: float, maximum: float, value: float, step: float, decimals: int = 2) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _sync_count_bounds(self) -> None:
        count = len(self.scenario.candidates)
        self.site_count.setMaximum(max(1, count))
        self.gading_count.setMaximum(count)
        self.bsd_count.setMaximum(count)
        self.site_count.setValue(min(self.site_count.value(), max(1, count)))

    def _configure_candidate_pools(self) -> None:
        current = self.scenario.candidates.copy().reset_index(drop=True)
        if not current.empty and current["site_type"].astype(str).eq("road_grid_node").all():
            self.osm_candidates = None
            self.road_grid_pool = current
            self.search_mode = "Generated road-grid points"
            self.pool_build_pending = False
        else:
            self.osm_candidates = current
            self.road_grid_pool = None
            self.pool_build_pending = False
        if self.scenario.graph is not None and self.road_grid_pool is None:
            self.road_grid_pool = _road_grid_candidates(self.scenario, self.road_grid_spacing_m)
            self.search_mode = "Generated road-grid points"
            self.pool_build_pending = True
        else:
            if self.scenario.graph is None:
                self.search_mode = "Mapped candidate locations"

    def _apply_candidate_pool(self, pool: pd.DataFrame, message: str) -> None:
        mode = "road_grid" if self.search_mode == "Generated road-grid points" else "mapped"
        self._start_candidate_pool_build(self.scenario, pool, mode, message)

    def _start_initial_pool_build(self) -> None:
        if self.road_grid_pool is not None:
            self._start_candidate_pool_build(
                self.scenario,
                self.road_grid_pool,
                "road_grid",
                f"Building road-grid search points at {self.road_grid_spacing_m} m spacing",
            )

    def _start_candidate_pool_build(
        self, base: Scenario, pool: pd.DataFrame, mode: str, message: str,
        osm_candidates: pd.DataFrame | None = None,
    ) -> None:
        self.run_button.setEnabled(False)
        self.candidate_table.setEnabled(False)
        self.search_mode_combo.setEnabled(False)
        self.grid_spacing.setEnabled(False)
        self.regenerate_grid_button.setEnabled(False)
        self.map_view.setEnabled(False)
        for button in (getattr(self, "open_button", None), getattr(self, "load_real_button", None),
                       getattr(self, "demo_button", None), getattr(self, "import_button", None)):
            if button is not None:
                button.setEnabled(False)
        self.status.setText(f"{message} · calculating driving distances; the window remains responsive")
        self.pending_osm_candidates = osm_candidates
        self.pool_worker = CandidatePoolWorker(base, pool, mode, self)
        self.pool_worker.ready.connect(self._candidate_pool_ready)
        self.pool_worker.failed.connect(self._candidate_pool_failed)
        self.pool_worker.start()

    @Slot(object, object, str)
    def _candidate_pool_ready(self, scenario: Scenario, pool: pd.DataFrame, mode: str) -> None:
        self.scenario = scenario
        self.search_mode = "Generated road-grid points" if mode == "road_grid" else "Mapped candidate locations"
        if mode == "road_grid":
            self.road_grid_pool = pool.copy()
        if self.pending_osm_candidates is not None:
            self.osm_candidates = self.pending_osm_candidates
        self.pending_osm_candidates = None
        self.pool_build_pending = False
        self.results = None
        self._populate_candidate_table()
        self._sync_count_bounds()
        self._refresh_map()
        self.run_button.setEnabled(True)
        self.candidate_table.setEnabled(True)
        self.search_mode_combo.setEnabled(True)
        self.grid_spacing.setEnabled(True)
        self.regenerate_grid_button.setEnabled(self.scenario.graph is not None)
        self.map_view.setEnabled(True)
        for button in (getattr(self, "open_button", None), getattr(self, "load_real_button", None),
                       getattr(self, "demo_button", None), getattr(self, "import_button", None)):
            if button is not None:
                button.setEnabled(True)
        self.status.setText(f"Ready to optimize · {len(pool)} {self.search_mode.lower()}")

    @Slot(str)
    def _candidate_pool_failed(self, message: str) -> None:
        self.pending_osm_candidates = None
        self.pool_build_pending = False
        self.run_button.setEnabled(True)
        self.candidate_table.setEnabled(True)
        self.search_mode_combo.setEnabled(True)
        self.grid_spacing.setEnabled(True)
        self.regenerate_grid_button.setEnabled(self.scenario.graph is not None)
        self.map_view.setEnabled(True)
        for button in (getattr(self, "open_button", None), getattr(self, "load_real_button", None),
                       getattr(self, "demo_button", None), getattr(self, "import_button", None)):
            if button is not None:
                button.setEnabled(True)
        self.status.setText(f"Could not build this search pool: {message}")
        if self.osm_candidates is not None:
            self.search_mode = "Mapped candidate locations"
            index = self.search_mode_combo.findData("mapped")
            if index >= 0:
                self.search_mode_combo.blockSignals(True)
                self.search_mode_combo.setCurrentIndex(index)
                self.search_mode_combo.blockSignals(False)
        self._refresh_map()
        QMessageBox.critical(self, "Cannot build search locations", message)

    def _candidate_mode_changed(self) -> None:
        mode = self.search_mode_combo.currentData()
        if mode == "road_grid":
            self._regenerate_road_grid()
        elif self.osm_candidates is not None:
            self.search_mode = "Mapped candidate locations"
            self._apply_candidate_pool(self.osm_candidates, "Using mapped host locations")

    def _regenerate_road_grid(self) -> None:
        if self.scenario.graph is None:
            return
        try:
            self.road_grid_spacing_m = self.grid_spacing.value()
            self.road_grid_pool = _road_grid_candidates(self.scenario, self.road_grid_spacing_m)
            self.search_mode = "Generated road-grid points"
            self._apply_candidate_pool(
                self.road_grid_pool,
                f"Generated road-network search points at {self.road_grid_spacing_m} m spacing",
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Cannot generate road-grid locations", str(exc))

    def _refresh_map(self) -> None:
        self.map_view.setHtml(_map_html(self.scenario), QUrl("https://spklu.local/"))

    def _select_map_area(self, area: str) -> None:
        self.map_view.page().runJavaScript(f"window.setSelectedArea({json.dumps(area)})")

    @Slot(str)
    def _apply_map_boundaries(self, value: str) -> None:
        try:
            boundaries = json.loads(value)
            if boundaries.get("type") != "FeatureCollection":
                raise ValueError("The map did not return a GeoJSON FeatureCollection.")
            geometries: dict[str, list[Any]] = {area: [] for area in AREAS}
            for feature in boundaries.get("features", []):
                area = _feature_area(feature)
                geometry = shape(feature["geometry"])
                if area not in AREAS or geometry.is_empty or not geometry.is_valid:
                    raise ValueError("Keep both named study areas as valid polygons.")
                if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
                    raise ValueError("Study areas must remain polygons.")
                geometries[area].append(geometry)
            if any(not pieces for pieces in geometries.values()):
                raise ValueError("Both Gading Serpong and BSD City boundaries must remain on the map.")
            cleaned_features = []
            for area in AREAS:
                merged = unary_union(geometries[area])
                if not merged.is_valid or merged.is_empty or merged.geom_type not in {"Polygon", "MultiPolygon"}:
                    raise ValueError(f"{area} must remain a valid polygon or multipolygon.")
                cleaned_features.append({
                    "type": "Feature",
                    "properties": {"area": area},
                    "geometry": mapping(merged),
                })
            cleaned = {"type": "FeatureCollection", "features": cleaned_features}
            demand = _filter_points(self.scenario.demand, cleaned)
            if demand.empty:
                raise ValueError("This boundary edit removes all demand points.")
            base = replace(self.scenario, boundaries=cleaned, demand=demand)
            if self.search_mode == "Generated road-grid points" and base.graph is not None:
                pool = _road_grid_candidates(base, self.road_grid_spacing_m)
            elif self.osm_candidates is not None:
                pool = _filter_points(self.osm_candidates, cleaned)
            else:
                pool = _filter_points(self.scenario.candidates, cleaned)
            if pool.empty:
                raise ValueError("This boundary edit leaves no new-location search points.")
            updated_osm_candidates = (
                _filter_points(self.osm_candidates, cleaned)
                if self.osm_candidates is not None
                else None
            )
            mode = "road_grid" if self.search_mode == "Generated road-grid points" else "mapped"
            self._start_candidate_pool_build(
                base,
                pool,
                mode,
                "Boundary updated; applying edited GeoJSON and recalculating road distances",
                osm_candidates=updated_osm_candidates,
            )
        except Exception as exc:  # noqa: BLE001 - avoid silently accepting broken map geometry
            self.status.setText(f"Boundary edit was not applied: {exc}")
            self._refresh_map()

    def _load_real(self) -> None:
        try:
            self.scenario = load_scenario(REAL_SCENARIO)
            self.source_label = "Cached real scenario · OSM / WorldPop / Open Charge Map"
            self._scenario_replaced()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Cannot load real scenario", str(exc))

    def _open_scenario(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open a saved SPKLU scenario", str(ROOT / "data"))
        if not folder:
            return
        try:
            self.scenario = load_scenario(folder)
            self.source_label = f"Scenario: {folder}"
            self._scenario_replaced()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Cannot open scenario", str(exc))

    def _load_demo(self) -> None:
        self.scenario = demo_scenario(seed=42)
        self.source_label = "Synthetic demonstration"
        self._scenario_replaced()

    def _scenario_replaced(self) -> None:
        self._configure_candidate_pools()
        self.search_mode_combo.blockSignals(True)
        self.search_mode_combo.clear()
        if self.scenario.graph is not None:
            self.search_mode_combo.addItem("Generated road-grid points", "road_grid")
        if self.osm_candidates is not None:
            self.search_mode_combo.addItem("Mapped parking / fuel / mall locations", "mapped")
        mode_data = "road_grid" if self.search_mode == "Generated road-grid points" else "mapped"
        mode_index = self.search_mode_combo.findData(mode_data)
        if mode_index >= 0:
            self.search_mode_combo.setCurrentIndex(mode_index)
        self.search_mode_combo.blockSignals(False)
        self.regenerate_grid_button.setEnabled(self.scenario.graph is not None)
        self.source_label_widget.setText(self.source_label)
        self.results = None
        self._refresh_map()
        self._populate_candidate_table()
        self._populate_existing_table()
        self._sync_count_bounds()
        self.status.setText(f"Loaded {self.source_label}.")
        if self.pool_build_pending:
            self.run_button.setEnabled(False)
            self.status.setText("Preparing area-wide road-grid recommendations…")
            QTimer.singleShot(150, self._start_initial_pool_build)

    def _save_scenario(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder for the scenario snapshot", str(ROOT / "data"))
        if folder:
            try:
                save_scenario(self.scenario, folder)
                self.status.setText(f"Scenario saved to {folder}")
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Cannot save scenario", str(exc))

    def _export_boundaries(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export edited study areas", "spklu_boundaries.geojson", "GeoJSON (*.geojson *.json)"
        )
        if filename:
            Path(filename).write_text(
                json.dumps(self.scenario.boundaries, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.status.setText(f"Exported edited area GeoJSON to {filename}")

    def _import_csv(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Import scenario table", "", "CSV files (*.csv)")
        if not filename:
            return
        kind = {
            "Candidate sites": "candidates",
            "Current SPKLU": "existing",
            "Demand points": "demand",
        }[self.import_kind.currentText()]
        try:
            frame = pd.read_csv(filename)
            updated = rebuild_distances(self.scenario, **{kind: frame})
            self.scenario = updated
            self.results = None
            if kind == "candidates":
                self.osm_candidates = updated.candidates.copy()
                self.search_mode = "Mapped candidate locations"
                self.road_grid_pool = (
                    _road_grid_candidates(updated, self.road_grid_spacing_m)
                    if updated.graph is not None
                    else None
                )
                mode_index = self.search_mode_combo.findData("mapped")
                if mode_index >= 0:
                    self.search_mode_combo.blockSignals(True)
                    self.search_mode_combo.setCurrentIndex(mode_index)
                    self.search_mode_combo.blockSignals(False)
            self.source_label += " · custom CSV"
            self.source_label_widget.setText(self.source_label)
            self._refresh_map()
            self._populate_candidate_table()
            self._populate_existing_table()
            self._sync_count_bounds()
            self.status.setText(f"Imported {len(frame)} {kind} rows from {filename}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Cannot import CSV", str(exc))

    def _populate_candidate_table(self) -> None:
        table = self.candidate_table
        table.blockSignals(True)
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Use", "Search point", "Area", "Type", "ID"])
        table.setRowCount(len(self.scenario.candidates))
        for i, row in enumerate(self.scenario.candidates.itertuples(index=False)):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
            check.setCheckState(Qt.CheckState.Checked)
            table.setItem(i, 0, check)
            for col, value in enumerate((row.name, row.area, row.site_type, row.id), start=1):
                table.setItem(i, col, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()
        table.blockSignals(False)

    def _candidate_changed(self, _item: QTableWidgetItem) -> None:
        self.results = None

    def _populate_existing_table(self) -> None:
        table = self.existing_table
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Baseline", "Current station", "Area"])
        table.setRowCount(len(self.scenario.existing))
        for i, row in enumerate(self.scenario.existing.itertuples(index=False)):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
            check.setCheckState(Qt.CheckState.Checked if getattr(row, "include_default", True) else Qt.CheckState.Unchecked)
            table.setItem(i, 0, check)
            table.setItem(i, 1, QTableWidgetItem(str(row.name)))
            table.setItem(i, 2, QTableWidgetItem(str(row.area)))
        table.resizeColumnsToContents()

    def _run_optimization(self) -> None:
        selected_rows = [
            i for i in range(self.candidate_table.rowCount())
            if self.candidate_table.item(i, 0).checkState() == Qt.CheckState.Checked
        ]
        existing_rows = [
            i for i in range(self.existing_table.rowCount())
            if self.existing_table.item(i, 0).checkState() == Qt.CheckState.Checked
        ]
        candidates = self.scenario.candidates.iloc[selected_rows].reset_index(drop=True)
        if candidates.empty:
            QMessageBox.warning(self, "No candidate sites", "Select at least one candidate site in the search-pool table.")
            return
        allocation_mode = self.allocation.currentText()
        quotas = None
        if allocation_mode == "Exact counts by area":
            quotas = {"Gading Serpong": self.gading_count.value(), "BSD City": self.bsd_count.value()}
            k = sum(quotas.values())
            for area, count in quotas.items():
                if count > int((candidates.area == area).sum()):
                    QMessageBox.warning(self, "Quota exceeds candidates", f"There are not enough selected candidates in {area}.")
                    return
        else:
            k = self.site_count.value()
        if k < 1 or k > len(candidates):
            QMessageBox.warning(self, "Invalid site count", "Choose at least one new site and no more than the selected candidates.")
            return
        demand_weights = compute_demand_weights(
            self.scenario.demand,
            population_fraction=self.population_share.value(),
            activity_weights={name: spin.value() for name, spin in self.activity_spins.items()},
        )
        candidate_ids = candidates["id"].tolist()
        candidate_indices = self.scenario.candidates.index[self.scenario.candidates.id.isin(candidate_ids)].to_numpy()
        settings = {
            "k": k,
            "per_area_counts": quotas,
            "radius_m": self.radius.value(),
            "distance_weight": self.distance_weight.value(),
            "max_evaluations": self.evaluations.value(),
            "population_size": self.ga_population.value(),
            "swarm_size": self.pso_swarm.value(),
            "repetitions": self.repetitions.value(),
            "seed": self.seed.value(),
            "mutation_rate": self.mutation.value(),
            "crossover_rate": self.crossover.value(),
            "inertia": self.inertia.value(),
            "cognitive": self.cognitive.value(),
            "social": self.social.value(),
            "animation_delay": self.animation_delay.value(),
        }
        args = {
            "distances_m": np.asarray(self.scenario.distances_m)[:, candidate_indices],
            "existing_distances_m": np.asarray(self.scenario.existing_distances_m)[:, existing_rows],
            "demand_weights": np.asarray(demand_weights),
            "candidate_areas": candidates.area.astype(str).to_numpy(),
            "candidates": candidates,
        }
        self.results = None
        self.run_button.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Starting GA and PSO…")
        self.map_view.page().runJavaScript("window.clearOptimizerSites()")
        self.map_view.page().runJavaScript(
            f"window.setCandidatePool({json.dumps(candidates.to_dict('records'), default=str, ensure_ascii=False)})"
        )
        self.worker = OptimizationWorker(args, settings, self)
        self.worker.progress.connect(self._optimization_progress)
        self.worker.finishedRuns.connect(self._optimization_finished)
        self.worker.failed.connect(self._optimization_failed)
        self.worker.start()

    @Slot(str, int, int, int, float, object)
    def _optimization_progress(self, algorithm: str, run_number: int, epoch: int,
                               total_epochs: int, score: float, indices: tuple[int, ...]) -> None:
        # Each run has both algorithms; give each method half of the run progress.
        algo_fraction = 0.5 if algorithm == "GA" else 1.0
        current_fraction = ((run_number - 1) + ((epoch / max(1, total_epochs)) * algo_fraction)) / max(1, self.repetitions.value())
        self.progress.setValue(int(current_fraction * 100))
        self.status.setText(f"{algorithm} run {run_number} · epoch {epoch}/{total_epochs} · best score {score:.5f}")
        candidates = self._current_worker_candidates
        rows = candidates.iloc[list(indices)].to_dict("records") if indices else []
        self.map_view.page().runJavaScript(
            f"window.updateOptimizerSites({json.dumps(algorithm)},{json.dumps(rows, default=str, ensure_ascii=False)})"
        )

    @property
    def _current_worker_candidates(self) -> pd.DataFrame:
        return self.worker.problem_args["candidates"] if self.worker else self.scenario.candidates

    @Slot(object, object, object)
    def _optimization_finished(self, runs: dict[str, list[Any]], problem: Problem,
                                candidates: pd.DataFrame) -> None:
        self.results = runs
        self.result_problem = problem
        self.result_candidates = candidates
        self.run_button.setEnabled(True)
        self.progress.setValue(100)
        self.status.setText("GA + PSO comparison complete.")
        self._populate_results()
        self._show_final_result(self.result_method.currentText())

    @Slot(str)
    def _optimization_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.status.setText(f"Optimization failed: {message}")
        QMessageBox.critical(self, "Optimization failed", message)

    def _populate_results(self) -> None:
        if not self.results:
            return
        rows = []
        for method, results in self.results.items():
            for index, result in enumerate(results, start=1):
                rows.append([method, index, f"{result.score:.5f}", result.evaluations, result.seed])
        table = self.summary_table
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Method", "Run", "Score (lower is better)", "Evaluations", "Seed"])
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()
        self.output_tabs.setCurrentIndex(0)

    def _show_final_result(self, method: str) -> None:
        if not self.results or not self.result_candidates:
            return
        result = min(self.results[method], key=lambda item: item.score)
        selected = self.result_candidates.iloc[list(result.indices)].copy()
        table = self.selected_table
        columns = ["name", "area", "site_type", "lat", "lon"]
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels([c.replace("_", " ").title() for c in columns])
        table.setRowCount(len(selected))
        for i, (_, row) in enumerate(selected.iterrows()):
            for j, column in enumerate(columns):
                table.setItem(i, j, QTableWidgetItem(str(row[column])))
        table.resizeColumnsToContents()
        color_method = "PSO" if method == "PSO" else "GA"
        sites = selected.to_dict("records")
        self.map_view.page().runJavaScript(
            f"window.updateOptimizerSites({json.dumps(color_method)},{json.dumps(sites, default=str, ensure_ascii=False)})"
        )
        self.output_tabs.setCurrentIndex(1)

    def _export_sites(self) -> None:
        if not self.results:
            QMessageBox.information(self, "No results", "Run GA and PSO first.")
            return
        method = self.result_method.currentText()
        result = min(self.results[method], key=lambda item: item.score)
        selected = self.result_candidates.iloc[list(result.indices)].copy()
        filename, _ = QFileDialog.getSaveFileName(self, "Export selected sites", f"spklu_{method.lower()}_sites.csv", "CSV (*.csv);;GeoJSON (*.geojson)")
        if not filename:
            return
        try:
            if filename.lower().endswith(".geojson"):
                geo = {"type": "FeatureCollection", "features": [
                    {"type": "Feature", "properties": {k: v for k, v in row.items() if k not in {"lat", "lon"}},
                     "geometry": {"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]}}
                    for row in selected.to_dict("records")
                ]}
                Path(filename).write_text(json.dumps(geo, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            else:
                selected.to_csv(filename, index=False)
            self.status.setText(f"Exported {method} sites to {filename}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SPKLU Site Optimizer")
    app.setStyle("Fusion")
    window = SPKLUMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
