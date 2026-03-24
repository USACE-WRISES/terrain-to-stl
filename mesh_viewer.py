from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk
from matplotlib.colors import LinearSegmentedColormap


FLOAT_TOLERANCE = 1e-6
PREVIEW_FILE_SIZE_BYTES = 100 * 1024 * 1024
PREVIEW_CELL_THRESHOLD = 300_000
PREVIEW_EMPTY_FRACTION = 0.005
PREVIEW_BASE_XY_DIVISIONS = 128
PREVIEW_MIN_XY_DIVISIONS = 48
PREVIEW_Z_DIVISIONS = 32
VIEWER_WINDOW_SIZE = (1600, 1000)
DEFAULT_SURFACE_OPACITY = 0.8
PROFILE_MODE_BOTTOM = "bottom"
PROFILE_MODE_FULL_SHELL = "full_shell"

PROFILE_BOTTOM_COLOR = "#ff3b30"
PROFILE_SCENE_COLOR = "#00e5ff"
PROFILE_TOP_COLOR = "#30638e"
TEXT_COLOR = "#1f2933"
TEXT_COLOR_DISABLED = "#8f99a3"
BUTTON_ON_COLOR = "#516b7b"
BUTTON_OFF_COLOR = "#d1d9e0"
BUTTON_ACTION_COLOR = "#6e8ba8"
BACKGROUND_BOTTOM = "#dbe5eb"
BACKGROUND_TOP = "#f4f7fa"
TERRAIN_COLORMAP = LinearSegmentedColormap.from_list(
    "hec_ras_terrain_like",
    [
        "#2b78c2",
        "#2ea34a",
        "#9fca3c",
        "#e6df2a",
        "#f1bb1e",
        "#df7d12",
        "#bb2d22",
        "#7b7b7b",
        "#b4b4b4",
        "#dedede",
    ],
)

pv.global_theme.allow_empty_mesh = True


class MeshViewerError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ProfileSummary:
    segment_count: int
    bottom_min_elevation: float | None
    bottom_max_elevation: float | None
    top_min_elevation: float | None
    top_max_elevation: float | None
    horizontal_length: float
    spatial_length: float


@dataclass(frozen=True, slots=True)
class ProfileResult:
    section_mesh: pv.PolyData
    bottom_segments: list[tuple[np.ndarray, np.ndarray]]
    top_segments: list[tuple[np.ndarray, np.ndarray]]
    summary: ProfileSummary


@dataclass(frozen=True, slots=True)
class MeshLoadResult:
    render_mesh: pv.PolyData
    exact_mesh: pv.PolyData | None
    render_mode: str
    file_size_bytes: int
    exact_cell_count: int | None

    @property
    def needs_lazy_exact_load(self) -> bool:
        return self.render_mode == "Preview" and self.exact_mesh is None


def prompt_stl_path() -> Path:
    while True:
        raw = input("Input STL path: ").strip().strip('"')
        if not raw:
            print("Please enter a path to an STL file.")
            continue

        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()

        validate_stl_path(path)
        return path


def validate_stl_path(path: Path) -> None:
    if not path.exists():
        raise MeshViewerError(f"File not found: {path}")
    if not path.is_file():
        raise MeshViewerError(f"Path is not a file: {path}")
    if path.suffix.lower() != ".stl":
        raise MeshViewerError("The viewer only supports .stl files.")


def resolve_stl_path(argv_path: str | None) -> Path:
    if argv_path is None:
        return prompt_stl_path()

    path = Path(argv_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    validate_stl_path(path)
    return path


def format_file_size(size_bytes: int) -> str:
    size_mb = size_bytes / (1024 * 1024)
    if size_mb < 1024:
        return f"{size_mb:.1f} MB"
    size_gb = size_mb / 1024
    return f"{size_gb:.2f} GB"


def to_rgb(color: str) -> tuple[float, float, float]:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {color!r}.")
    return tuple(int(value[index : index + 2], 16) / 255.0 for index in (0, 2, 4))


def prepare_polydata(mesh: pv.PolyData, source: str) -> pv.PolyData:
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    mesh = mesh.triangulate()
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise MeshViewerError(f"{source} does not contain any displayable mesh geometry.")

    mesh.point_data["Elevation"] = mesh.points[:, 2].astype(np.float32, copy=False)
    mesh.set_active_scalars("Elevation")
    return mesh


def load_stl_mesh(path: Path) -> pv.PolyData:
    return prepare_polydata(pv.read(path), str(path))


def compute_preview_divisions(bounds: tuple[float, float, float, float, float, float]) -> tuple[int, int, int]:
    xmin, xmax, ymin, ymax, _, _ = bounds
    x_range = max(float(xmax - xmin), 1.0)
    y_range = max(float(ymax - ymin), 1.0)
    longest_range = max(x_range, y_range, 1.0)

    x_divisions = int(round(PREVIEW_BASE_XY_DIVISIONS * (x_range / longest_range)))
    y_divisions = int(round(PREVIEW_BASE_XY_DIVISIONS * (y_range / longest_range)))

    x_divisions = max(PREVIEW_MIN_XY_DIVISIONS, min(PREVIEW_BASE_XY_DIVISIONS, x_divisions))
    y_divisions = max(PREVIEW_MIN_XY_DIVISIONS, min(PREVIEW_BASE_XY_DIVISIONS, y_divisions))
    return x_divisions, y_divisions, PREVIEW_Z_DIVISIONS


def cluster_mesh(source: pv.PolyData | vtk.vtkDataSet) -> pv.PolyData:
    bounds = tuple(float(value) for value in source.GetBounds())
    x_divisions, y_divisions, z_divisions = compute_preview_divisions(bounds)

    clustering = vtk.vtkQuadricClustering()
    clustering.SetInputData(source)
    clustering.SetNumberOfXDivisions(x_divisions)
    clustering.SetNumberOfYDivisions(y_divisions)
    clustering.SetNumberOfZDivisions(z_divisions)
    clustering.CopyCellDataOn()
    clustering.Update()
    return prepare_polydata(pv.wrap(clustering.GetOutput()), "Preview mesh")


def build_preview_mesh_from_path(path: Path) -> pv.PolyData:
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()
    return cluster_mesh(reader.GetOutput())


def load_mesh_state(path: Path) -> MeshLoadResult:
    file_size_bytes = path.stat().st_size
    if file_size_bytes > PREVIEW_FILE_SIZE_BYTES:
        print(
            f"Large STL detected ({format_file_size(file_size_bytes)}). "
            "Building a lighter preview mesh for interactive viewing..."
        )
        render_mesh = build_preview_mesh_from_path(path)
        return MeshLoadResult(
            render_mesh=render_mesh,
            exact_mesh=None,
            render_mode="Preview",
            file_size_bytes=file_size_bytes,
            exact_cell_count=None,
        )

    exact_mesh = load_stl_mesh(path)
    if exact_mesh.n_cells > PREVIEW_CELL_THRESHOLD:
        print(
            f"STL contains {exact_mesh.n_cells:,} triangles. "
            "Building a lighter preview mesh for interactive viewing..."
        )
        render_mesh = cluster_mesh(exact_mesh)
        return MeshLoadResult(
            render_mesh=render_mesh,
            exact_mesh=exact_mesh,
            render_mode="Preview",
            file_size_bytes=file_size_bytes,
            exact_cell_count=exact_mesh.n_cells,
        )

    return MeshLoadResult(
        render_mesh=exact_mesh,
        exact_mesh=exact_mesh,
        render_mode="Exact",
        file_size_bytes=file_size_bytes,
        exact_cell_count=exact_mesh.n_cells,
    )


def default_profile_endpoints(
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    ymid = (ymin + ymax) / 2.0
    zmid = (zmin + zmax) / 2.0
    start = np.array([xmin, ymid, zmid], dtype=float)
    end = np.array([xmax, ymid, zmid], dtype=float)
    return start, end


def profile_plane_normal(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, float]:
    horizontal = np.array([end[0] - start[0], end[1] - start[1], 0.0], dtype=float)
    horizontal_length = float(np.linalg.norm(horizontal))
    if horizontal_length <= FLOAT_TOLERANCE:
        raise MeshViewerError("The profile line is too short in plan view. Move the profile endpoints farther apart.")

    horizontal_unit = horizontal / horizontal_length
    normal = np.cross(horizontal_unit, np.array([0.0, 0.0, 1.0], dtype=float))
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= FLOAT_TOLERANCE:
        raise MeshViewerError("Could not build a vertical slice plane from the current profile line.")

    return normal / normal_length, horizontal_length


def profile_mode_label(profile_mode: str) -> str:
    return "Full Shell" if profile_mode == PROFILE_MODE_FULL_SHELL else "Bottom Only"


def estimate_shell_top_elevation(mesh: pv.PolyData) -> float:
    elevations = np.asarray(mesh.points)[:, 2]
    return float(np.quantile(elevations, 0.995))


def top_plane_filter_tolerance(mesh: pv.PolyData) -> float:
    zmin = float(mesh.bounds[4])
    zmax = float(mesh.bounds[5])
    return max(1e-3, (zmax - zmin) * 1e-6)


def build_profile_section(
    mesh: pv.PolyData,
    start: np.ndarray,
    end: np.ndarray,
    profile_mode: str,
) -> ProfileResult:
    horizontal_length = float(np.linalg.norm(np.array([end[0] - start[0], end[1] - start[1], 0.0], dtype=float)))
    spatial_length = float(np.linalg.norm(end - start))

    if mesh.n_points == 0 or mesh.n_cells == 0:
        return ProfileResult(
            section_mesh=pv.PolyData(),
            bottom_segments=[],
            top_segments=[],
            summary=ProfileSummary(
                segment_count=0,
                bottom_min_elevation=None,
                bottom_max_elevation=None,
                top_min_elevation=None,
                top_max_elevation=None,
                horizontal_length=horizontal_length,
                spatial_length=spatial_length,
            ),
        )

    normal, horizontal_length = profile_plane_normal(start, end)
    origin = (start + end) / 2.0
    section = mesh.slice(normal=normal, origin=origin, generate_triangles=False, contour=False)
    if section.n_points == 0 or section.n_cells == 0:
        return ProfileResult(
            section_mesh=pv.PolyData(),
            bottom_segments=[],
            top_segments=[],
            summary=ProfileSummary(
                segment_count=0,
                bottom_min_elevation=None,
                bottom_max_elevation=None,
                top_min_elevation=None,
                top_max_elevation=None,
                horizontal_length=horizontal_length,
                spatial_length=spatial_length,
            ),
        )

    direction_xy = np.array([end[0] - start[0], end[1] - start[1]], dtype=float) / horizontal_length
    shell_top_elevation = estimate_shell_top_elevation(mesh)
    bottom_segments, top_segments = extract_profile_envelopes(
        section,
        start,
        direction_xy,
        horizontal_length,
        bottom_only=(profile_mode == PROFILE_MODE_BOTTOM),
        shell_top_elevation=shell_top_elevation,
        shell_top_tolerance=top_plane_filter_tolerance(mesh),
    )
    if not bottom_segments:
        return ProfileResult(
            section_mesh=pv.PolyData(),
            bottom_segments=[],
            top_segments=[],
            summary=ProfileSummary(
                segment_count=0,
                bottom_min_elevation=None,
                bottom_max_elevation=None,
                top_min_elevation=None,
                top_max_elevation=None,
                horizontal_length=horizontal_length,
                spatial_length=spatial_length,
            ),
        )

    displayed_top_segments = top_segments if profile_mode == PROFILE_MODE_FULL_SHELL else []
    section_mesh = merge_profile_polylines(
        [
            build_profile_polyline(bottom_segments, start, direction_xy),
            build_profile_polyline(displayed_top_segments, start, direction_xy),
        ]
    )
    bottom_min_elevation = min(float(np.min(elevations)) for _, elevations in bottom_segments)
    bottom_max_elevation = max(float(np.max(elevations)) for _, elevations in bottom_segments)
    top_min_elevation = None
    top_max_elevation = None
    if displayed_top_segments:
        top_min_elevation = min(float(np.min(elevations)) for _, elevations in displayed_top_segments)
        top_max_elevation = max(float(np.max(elevations)) for _, elevations in displayed_top_segments)

    return ProfileResult(
        section_mesh=section_mesh,
        bottom_segments=bottom_segments,
        top_segments=displayed_top_segments,
        summary=ProfileSummary(
            segment_count=max(len(bottom_segments), len(displayed_top_segments)),
            bottom_min_elevation=bottom_min_elevation,
            bottom_max_elevation=bottom_max_elevation,
            top_min_elevation=top_min_elevation,
            top_max_elevation=top_max_elevation,
            horizontal_length=horizontal_length,
            spatial_length=spatial_length,
        ),
    )


def extract_profile_envelopes(
    section: pv.PolyData,
    start: np.ndarray,
    direction_xy: np.ndarray,
    horizontal_length: float,
    *,
    bottom_only: bool,
    shell_top_elevation: float,
    shell_top_tolerance: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    points = np.asarray(section.points)
    if points.shape[0] < 2:
        return [], []

    distances = ((points[:, :2] - start[:2]) @ direction_xy).astype(float)
    mask = (distances >= -FLOAT_TOLERANCE) & (distances <= horizontal_length + FLOAT_TOLERANCE)
    filtered_points = points[mask]
    filtered_distances = distances[mask]
    if filtered_points.shape[0] < 2:
        filtered_points = points
        filtered_distances = distances

    if bottom_only:
        keep_mask = filtered_points[:, 2] < (shell_top_elevation - shell_top_tolerance)
        if int(np.count_nonzero(keep_mask)) >= 2:
            filtered_points = filtered_points[keep_mask]
            filtered_distances = filtered_distances[keep_mask]

    order = np.argsort(filtered_distances)
    ordered_distances = filtered_distances[order]
    ordered_elevations = filtered_points[order, 2].astype(float)
    if ordered_distances.shape[0] < 2:
        return [], []

    collapsed_distances, lower_elevations, upper_elevations = collapse_profile_envelopes(
        ordered_distances,
        ordered_elevations,
        horizontal_length,
    )
    if collapsed_distances.shape[0] < 2:
        return [], []

    bottom_segments = split_profile_segments(collapsed_distances, lower_elevations, horizontal_length)
    top_segments = split_profile_segments(collapsed_distances, upper_elevations, horizontal_length)
    return bottom_segments, top_segments


def collapse_profile_envelopes(
    distances: np.ndarray,
    elevations: np.ndarray,
    horizontal_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_tolerance = max(FLOAT_TOLERANCE, horizontal_length * 1e-6)
    collapsed_distances: list[float] = []
    collapsed_min_elevations: list[float] = []
    collapsed_max_elevations: list[float] = []

    index = 0
    point_count = distances.shape[0]
    while index < point_count:
        next_index = index + 1
        while next_index < point_count and (distances[next_index] - distances[index]) <= group_tolerance:
            next_index += 1

        collapsed_distances.append(float(np.mean(distances[index:next_index])))
        collapsed_min_elevations.append(float(np.min(elevations[index:next_index])))
        collapsed_max_elevations.append(float(np.max(elevations[index:next_index])))
        index = next_index

    return (
        np.asarray(collapsed_distances, dtype=float),
        np.asarray(collapsed_min_elevations, dtype=float),
        np.asarray(collapsed_max_elevations, dtype=float),
    )


def split_profile_segments(
    distances: np.ndarray,
    elevations: np.ndarray,
    horizontal_length: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if distances.shape[0] < 2:
        return []

    diffs = np.diff(distances)
    positive_diffs = diffs[diffs > FLOAT_TOLERANCE]
    typical_step = float(np.median(positive_diffs)) if positive_diffs.size else horizontal_length
    gap_threshold = max(typical_step * 8.0, horizontal_length * 0.02)
    split_points = np.where(diffs > gap_threshold)[0] + 1

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start_index = 0
    for end_index in list(split_points) + [distances.shape[0]]:
        segment_distances = distances[start_index:end_index]
        segment_elevations = elevations[start_index:end_index]
        if segment_distances.shape[0] >= 2:
            segments.append((segment_distances, segment_elevations))
        start_index = end_index

    return segments


def build_profile_polyline(
    segments: list[tuple[np.ndarray, np.ndarray]],
    start: np.ndarray,
    direction_xy: np.ndarray,
) -> pv.PolyData:
    polylines: list[pv.PolyData] = []
    for distances, elevations in segments:
        if distances.shape[0] < 2:
            continue

        xy_points = start[:2] + np.outer(distances, direction_xy)
        points = np.column_stack((xy_points, elevations))
        polylines.append(pv.lines_from_points(points, close=False))

    if not polylines:
        return pv.PolyData()
    if len(polylines) == 1:
        return polylines[0]
    return pv.merge(polylines)


def merge_profile_polylines(polylines: list[pv.PolyData]) -> pv.PolyData:
    non_empty_polylines = [polyline for polyline in polylines if polyline.n_cells > 0]
    if not non_empty_polylines:
        return pv.PolyData()
    if len(non_empty_polylines) == 1:
        return non_empty_polylines[0]
    return pv.merge(non_empty_polylines)


class MeshViewerApp:
    def __init__(self, mesh_path: Path) -> None:
        self.mesh_path = mesh_path

        load_result = load_mesh_state(mesh_path)
        self.render_mesh = load_result.render_mesh
        self.exact_mesh = load_result.exact_mesh
        self.render_mode = load_result.render_mode
        self.file_size_bytes = load_result.file_size_bytes
        self.exact_cell_count = load_result.exact_cell_count
        self.lazy_exact_load = load_result.needs_lazy_exact_load

        self.display_mesh = self.render_mesh.copy(deep=True)
        self.profile_section_mesh = pv.PolyData()

        self.default_clip_normal = (0.0, 0.0, 1.0)
        self.default_clip_origin = tuple(float(value) for value in self.render_mesh.center)
        self.clip_normal = self.default_clip_normal
        self.clip_origin = self.default_clip_origin

        self.profile_start, self.profile_end = default_profile_endpoints(self.render_mesh.bounds)
        self.clip_enabled = False
        self.clip_active = False
        self.profile_enabled = False
        self.profile_mode = PROFILE_MODE_BOTTOM
        self.profile_result_mode = "Off"
        self.opacity = DEFAULT_SURFACE_OPACITY
        self.wireframe_visible = False
        self.status_message = ""
        self.profile_message = "Profile: off. Enable the Profile toggle to inspect a section."

        self.plotter = pv.Plotter(window_size=VIEWER_WINDOW_SIZE)
        self.surface_actor = None
        self.wire_actor = None
        self.profile_actor = None
        self.header_actor = None
        self.details_actor = None
        self.chart = pv.Chart2D(size=(0.34, 0.26), loc=(0.61, 0.02))
        self.line_widget = None
        self.plane_widget = None
        self.opacity_slider = None
        self.control_buttons: dict[str, object] = {}
        self.control_labels: dict[str, object] = {}

    def build(self) -> None:
        self.plotter.set_background(BACKGROUND_BOTTOM, top=BACKGROUND_TOP)
        self.plotter.add_axes(interactive=True, line_width=2)

        scalar_bar_args = {
            "title": "Elevation",
            "fmt": "%.2f",
            "vertical": True,
            "position_x": 0.87,
            "position_y": 0.14,
            "height": 0.70,
        }
        self.surface_actor = self.plotter.add_mesh(
            self.display_mesh,
            scalars="Elevation",
            cmap=TERRAIN_COLORMAP,
            show_edges=False,
            opacity=DEFAULT_SURFACE_OPACITY,
            smooth_shading=False,
            lighting=True,
            scalar_bar_args=scalar_bar_args,
            ambient=0.3,
            diffuse=0.7,
            specular=0.02,
            silhouette={"color": "#5f6b77", "line_width": 1.2, "opacity": 0.35},
        )
        self.wire_actor = self.plotter.add_mesh(
            self.display_mesh,
            color="#1f2933",
            style="wireframe",
            line_width=1,
            opacity=0.35,
            render_lines_as_tubes=False,
            pickable=False,
        )
        self.wire_actor.SetVisibility(False)

        self.profile_actor = self.plotter.add_mesh(
            self.profile_section_mesh,
            color=PROFILE_SCENE_COLOR,
            line_width=6,
            render_lines_as_tubes=True,
            pickable=False,
        )
        self.profile_actor.SetVisibility(False)

        self._configure_chart()
        self.chart.visible = False
        self.plotter.add_chart(self.chart)

        self._add_static_text()
        self._add_widgets()
        self._add_control_panel()
        self._bind_keys()
        self._set_button_enabled("full_shell", False)
        self._set_measure_exact_enabled(False)
        self._refresh_header_text()
        self._refresh_details_text()
        self.set_isometric_view(render=False)

    def _configure_chart(self) -> None:
        self.chart.x_axis.label = "Distance"
        self.chart.y_axis.label = "Elevation"
        self.chart.x_axis.label_size = 10
        self.chart.y_axis.label_size = 10
        self.chart.legend_visible = False
        self.chart.background_color = (1.0, 1.0, 1.0, 0.94)
        self.chart.border_color = "#8c8c8c"
        self.chart.visible = True

    def _add_static_text(self) -> None:
        self.header_actor = self.plotter.add_text("", position=(18, 930), viewport=False, font_size=10, color=TEXT_COLOR)
        self.details_actor = self.plotter.add_text("", position=(1570, 930), viewport=False, font_size=10, color=TEXT_COLOR)
        self.details_actor.GetTextProperty().SetJustificationToRight()

    def _add_widgets(self) -> None:
        bounds = self.render_mesh.bounds

        self.plane_widget = self.plotter.add_plane_widget(
            self.on_clip_plane_changed,
            bounds=bounds,
            origin=self.default_clip_origin,
            normal="z",
            interaction_event="end",
            color="#6e8ba8",
            test_callback=False,
        )
        self.plane_widget.SetNormal(*self.default_clip_normal)
        self.plane_widget.SetOrigin(*self.default_clip_origin)
        self.plane_widget.SetEnabled(False)

        self.line_widget = self.plotter.add_line_widget(
            self.on_profile_line_changed,
            bounds=bounds,
            use_vertices=True,
            interaction_event="end",
            color=PROFILE_SCENE_COLOR,
        )
        self.line_widget.SetPoint1(*self.profile_start.tolist())
        self.line_widget.SetPoint2(*self.profile_end.tolist())
        self.line_widget.SetHandleSize(0.02)
        self.line_widget.GetLineProperty().SetColor(*to_rgb(PROFILE_SCENE_COLOR))
        self.line_widget.GetLineProperty().SetLineWidth(4.0)
        self.line_widget.GetSelectedLineProperty().SetColor(*to_rgb("#ffffff"))
        self.line_widget.GetSelectedLineProperty().SetLineWidth(5.0)
        self.line_widget.GetHandleProperty().SetColor(*to_rgb(PROFILE_SCENE_COLOR))
        self.line_widget.GetSelectedHandleProperty().SetColor(*to_rgb("#ffffff"))
        self.line_widget.SetEnabled(False)

        self.opacity_slider = self.plotter.add_slider_widget(
            self.on_opacity_changed,
            rng=(0.1, 1.0),
            value=DEFAULT_SURFACE_OPACITY,
            title="Surface opacity",
            pointa=(0.03, 0.07),
            pointb=(0.26, 0.07),
            fmt="%.2f",
            interaction_event="always",
        )

    def _add_control_panel(self) -> None:
        self.plotter.add_text("View", position=(18, 906), viewport=False, font_size=11, color=TEXT_COLOR)
        self._add_action_button("top_view", "Top", (18.0, 870.0), self.set_top_view)
        self._add_action_button("bottom_view", "Bottom", (18.0, 836.0), self.set_bottom_view)
        self._add_action_button("iso_view", "Isometric", (18.0, 802.0), self.set_isometric_view)
        self._add_action_button("reset_camera", "Reset Camera", (18.0, 768.0), self.reset_camera)

        self.plotter.add_text("Inspect", position=(18, 720), viewport=False, font_size=11, color=TEXT_COLOR)
        self._add_toggle_button("clip", "Clip Tool", (18.0, 684.0), self.on_clip_toggled, initial=False)
        self._add_action_button("reset_clip", "Reset Clip", (18.0, 650.0), self.reset_clip)
        self._add_toggle_button("profile", "Profile Tool", (18.0, 604.0), self.on_profile_toggled, initial=False)
        self._add_toggle_button("full_shell", "Full Shell", (18.0, 570.0), self.on_full_shell_toggled, initial=False)
        self._add_action_button("reset_profile", "Reset Profile", (18.0, 536.0), self.reset_profile)
        self._add_action_button("measure_exact", "Measure Exact", (18.0, 502.0), self.measure_exact_profile)

        self.plotter.add_text("Display", position=(18, 454), viewport=False, font_size=11, color=TEXT_COLOR)
        self._add_toggle_button("wireframe", "Wireframe", (18.0, 418.0), self.on_wireframe_toggled, initial=False)
        self._add_action_button("restore_all", "Restore All", (18.0, 384.0), self.restore_all)

        help_lines = [
            "Shortcuts",
            "T/B/I = view",
            "R = reset camera",
            "X = reset clip",
            "C = clip tool",
            "P = profile tool",
            "F = full shell",
            "M = measure exact",
            "A = restore all",
        ]
        self.plotter.add_text("\n".join(help_lines), position=(18, 120), viewport=False, font_size=9, color=TEXT_COLOR)

    def _bind_keys(self) -> None:
        self.plotter.add_key_event("t", self.set_top_view)
        self.plotter.add_key_event("b", self.set_bottom_view)
        self.plotter.add_key_event("i", self.set_isometric_view)
        self.plotter.add_key_event("r", self.reset_camera)
        self.plotter.add_key_event("x", self.reset_clip)
        self.plotter.add_key_event("c", self.toggle_clip_tool)
        self.plotter.add_key_event("p", self.toggle_profile_tool)
        self.plotter.add_key_event("f", self.toggle_full_shell_mode)
        self.plotter.add_key_event("w", self.toggle_wireframe)
        self.plotter.add_key_event("m", self.measure_exact_profile)
        self.plotter.add_key_event("a", self.restore_all)

    def _add_toggle_button(
        self,
        name: str,
        label: str,
        position: tuple[float, float],
        callback,
        *,
        initial: bool,
    ) -> None:
        widget = self.plotter.add_checkbox_button_widget(
            callback,
            value=initial,
            position=position,
            size=24,
            color_on=BUTTON_ON_COLOR,
            color_off=BUTTON_OFF_COLOR,
            background_color="white",
        )
        label_actor = self.plotter.add_text(
            label,
            position=(int(position[0]) + 34, int(position[1]) + 3),
            viewport=False,
            font_size=10,
            color=TEXT_COLOR,
        )
        self.control_buttons[name] = widget
        self.control_labels[name] = label_actor

    def _add_action_button(
        self,
        name: str,
        label: str,
        position: tuple[float, float],
        action,
    ) -> None:
        widget_box: dict[str, object] = {}

        def on_click(value: bool) -> None:
            if not value:
                return
            action()
            widget = widget_box["widget"]
            self._set_checkbox_state(widget, False)
            self.plotter.render()

        widget = self.plotter.add_checkbox_button_widget(
            on_click,
            value=False,
            position=position,
            size=24,
            color_on=BUTTON_ACTION_COLOR,
            color_off=BUTTON_OFF_COLOR,
            background_color="white",
        )
        widget_box["widget"] = widget
        label_actor = self.plotter.add_text(
            label,
            position=(int(position[0]) + 34, int(position[1]) + 3),
            viewport=False,
            font_size=10,
            color=TEXT_COLOR,
        )
        self.control_buttons[name] = widget
        self.control_labels[name] = label_actor

    def _set_checkbox_state(self, widget: object, value: bool) -> None:
        widget.GetRepresentation().SetState(int(bool(value)))

    def _set_button_enabled(self, name: str, enabled: bool) -> None:
        widget = self.control_buttons[name]
        label = self.control_labels[name]
        widget.SetEnabled(int(enabled))
        label.GetTextProperty().SetColor(*to_rgb(TEXT_COLOR if enabled else TEXT_COLOR_DISABLED))

    def _set_measure_exact_enabled(self, enabled: bool) -> None:
        self._set_button_enabled("measure_exact", enabled)

    def _set_full_shell_button_enabled(self, enabled: bool) -> None:
        self._set_button_enabled("full_shell", enabled)

    def _refresh_header_text(self) -> None:
        lines = [
            f"Mesh Viewer: {self.mesh_path.name}",
            f"Render mesh: {self.render_mode} ({self.render_mesh.n_cells:,} triangles shown)",
            f"STL size: {format_file_size(self.file_size_bytes)}",
        ]
        if self.render_mode == "Preview":
            if self.exact_mesh is None:
                lines.append("Exact profile: loads the full STL on demand")
            else:
                lines.append(f"Exact profile: cached full mesh ({self.exact_mesh.n_cells:,} triangles)")
        else:
            lines.append("Exact profile: using the loaded display mesh")
        self.header_actor.SetInput("\n".join(lines))

    def _refresh_details_text(self) -> None:
        lines = [f"Visible triangles: {self.display_mesh.n_cells:,}"]
        if self.status_message:
            lines.append(self.status_message)
        if self.profile_enabled:
            lines.append(f"Profile mode: {profile_mode_label(self.profile_mode)}")
        if self.profile_message:
            lines.append("")
            lines.extend(self.profile_message.splitlines())
        self.details_actor.SetInput("\n".join(lines).strip())

    def _sync_plane_widget(self) -> None:
        if self.plane_widget is None:
            return
        self.plane_widget.SetNormal(*self.clip_normal)
        self.plane_widget.SetOrigin(*self.clip_origin)

    def _sync_line_widget(self) -> None:
        if self.line_widget is None:
            return
        self.line_widget.SetPoint1(*self.profile_start.tolist())
        self.line_widget.SetPoint2(*self.profile_end.tolist())

    def _set_surface_opacity(self, value: float) -> None:
        self.opacity = float(value)
        self.surface_actor.GetProperty().SetOpacity(self.opacity)

    def on_opacity_changed(self, value: float) -> None:
        self._set_surface_opacity(value)
        self.plotter.render()

    def _set_slider_value(self, value: float) -> None:
        if self.opacity_slider is None:
            return
        self.opacity_slider.GetRepresentation().SetValue(float(value))

    def set_wireframe_visible(self, visible: bool) -> None:
        self.wireframe_visible = bool(visible)
        self.wire_actor.SetVisibility(self.wireframe_visible)
        self._set_checkbox_state(self.control_buttons["wireframe"], self.wireframe_visible)

    def on_wireframe_toggled(self, value: bool) -> None:
        self.set_wireframe_visible(bool(value))
        self.status_message = "Wireframe enabled." if self.wireframe_visible else "Wireframe hidden."
        self._refresh_details_text()
        self.plotter.render()

    def toggle_wireframe(self) -> None:
        next_value = not self.wireframe_visible
        self._set_checkbox_state(self.control_buttons["wireframe"], next_value)
        self.on_wireframe_toggled(next_value)

    def apply_render_mesh(self, mesh: pv.PolyData) -> None:
        self.display_mesh.copy_from(mesh, deep=True)
        self._refresh_details_text()

    def set_clip_enabled(self, enabled: bool) -> None:
        self.clip_enabled = bool(enabled)
        self._set_checkbox_state(self.control_buttons["clip"], self.clip_enabled)
        self.plane_widget.SetEnabled(int(self.clip_enabled))

        if self.clip_enabled:
            self._sync_plane_widget()
            self.status_message = "Clip tool enabled. Drag the plane to open the shell."
        else:
            self.reset_clip(update_toggle=False, render=False)
            self.status_message = "Clip tool disabled. Showing the full shell."
        self._refresh_details_text()

    def on_clip_toggled(self, value: bool) -> None:
        self.set_clip_enabled(bool(value))
        self.plotter.render()

    def toggle_clip_tool(self) -> None:
        self.on_clip_toggled(not self.clip_enabled)

    def on_clip_plane_changed(self, normal: tuple[float, float, float], origin: tuple[float, float, float]) -> None:
        if not self.clip_enabled:
            return

        clipped = self.render_mesh.clip(normal=normal, origin=origin)
        if clipped.n_cells == 0 or (clipped.n_cells / max(self.render_mesh.n_cells, 1)) < PREVIEW_EMPTY_FRACTION:
            self.status_message = "Clip rejected because it would hide almost the entire shell."
            self._sync_plane_widget()
            self._refresh_details_text()
            self.plotter.render()
            return

        self.clip_normal = tuple(float(value) for value in normal)
        self.clip_origin = tuple(float(value) for value in origin)
        self.clip_active = True
        self.apply_render_mesh(clipped)
        self.status_message = f"Clip applied on the {self.render_mode.lower()} mesh."
        if self.profile_enabled:
            self.refresh_profile_preview()
        else:
            self._refresh_details_text()
        self.plotter.render()

    def set_profile_enabled(self, enabled: bool) -> None:
        self.profile_enabled = bool(enabled)
        self._set_checkbox_state(self.control_buttons["profile"], self.profile_enabled)
        self.line_widget.SetEnabled(int(self.profile_enabled))
        self.profile_actor.SetVisibility(self.profile_enabled)
        self.chart.visible = self.profile_enabled
        self._set_full_shell_button_enabled(self.profile_enabled)
        self._set_measure_exact_enabled(self.profile_enabled)

        if self.profile_enabled:
            self.reset_profile(update_toggle=False, render=False)
            self.status_message = f"Profile tool enabled in {profile_mode_label(self.profile_mode)} mode."
        else:
            self.profile_result_mode = "Off"
            self.profile_message = "Profile: off. Enable the Profile toggle to inspect a section."
            self.chart.clear()
            self._configure_chart()
            self.chart.visible = False
            self.profile_section_mesh.copy_from(pv.PolyData(), deep=True)
            self.profile_actor.SetVisibility(False)
            self.status_message = "Profile tool disabled."
        self._refresh_details_text()

    def on_profile_toggled(self, value: bool) -> None:
        self.set_profile_enabled(bool(value))
        self.plotter.render()

    def toggle_profile_tool(self) -> None:
        self.on_profile_toggled(not self.profile_enabled)

    def on_full_shell_toggled(self, value: bool) -> None:
        self.set_full_shell_mode(bool(value))
        self.plotter.render()

    def toggle_full_shell_mode(self) -> None:
        if not self.profile_enabled:
            self.status_message = "Enable the Profile tool before switching profile modes."
            self._refresh_details_text()
            self.plotter.render()
            return
        self.on_full_shell_toggled(self.profile_mode != PROFILE_MODE_FULL_SHELL)

    def set_full_shell_mode(self, enabled: bool) -> None:
        self.profile_mode = PROFILE_MODE_FULL_SHELL if enabled else PROFILE_MODE_BOTTOM
        self._set_checkbox_state(self.control_buttons["full_shell"], enabled)

        if self.profile_enabled:
            self.status_message = f"Profile mode set to {profile_mode_label(self.profile_mode)}."
            self.refresh_profile_preview()
        else:
            self._refresh_details_text()

    def on_profile_line_changed(self, pointa: tuple[float, float, float], pointb: tuple[float, float, float]) -> None:
        if not self.profile_enabled:
            return

        self.profile_start = np.asarray(pointa, dtype=float)
        self.profile_end = np.asarray(pointb, dtype=float)
        self.status_message = f"Preview {profile_mode_label(self.profile_mode).lower()} profile updated."
        self.refresh_profile_preview()
        self.plotter.render()

    def refresh_profile_preview(self) -> None:
        if not self.profile_enabled:
            return

        try:
            result = build_profile_section(
                self.display_mesh,
                self.profile_start,
                self.profile_end,
                self.profile_mode,
            )
        except MeshViewerError as exc:
            self._clear_profile_result("Preview profile unavailable.", str(exc))
            return

        self._apply_profile_result(result, mode="Preview")

    def _clear_profile_result(self, status_message: str, profile_message: str) -> None:
        self.profile_section_mesh.copy_from(pv.PolyData(), deep=True)
        self.profile_actor.SetVisibility(False)
        self.chart.clear()
        self._configure_chart()
        self.chart.visible = self.profile_enabled
        self.profile_result_mode = "Preview" if self.profile_enabled else "Off"
        self.status_message = status_message
        self.profile_message = profile_message
        self._refresh_details_text()

    def _apply_profile_result(
        self,
        result: ProfileResult,
        *,
        mode: str,
    ) -> None:
        summary = result.summary
        self.profile_section_mesh.copy_from(result.section_mesh, deep=True)
        self.profile_actor.SetVisibility(self.profile_enabled and result.section_mesh.n_cells > 0)
        self.chart.clear()
        self._configure_chart()
        self.chart.visible = self.profile_enabled
        self.profile_result_mode = mode

        if result.bottom_segments:
            self._plot_chart_series(
                result.bottom_segments,
                label=f"{mode} bottom",
                color=PROFILE_BOTTOM_COLOR,
            )
            if result.top_segments:
                self._plot_chart_series(
                    result.top_segments,
                    label=f"{mode} top",
                    color=PROFILE_TOP_COLOR,
                )
            self.chart.legend_visible = bool(result.top_segments)
            self.profile_message = "\n".join(
                [
                    f"{mode} terrain profile segments: {summary.segment_count}",
                    f"Start: ({self.profile_start[0]:.2f}, {self.profile_start[1]:.2f}, {self.profile_start[2]:.2f})",
                    f"End: ({self.profile_end[0]:.2f}, {self.profile_end[1]:.2f}, {self.profile_end[2]:.2f})",
                    f"Horizontal distance: {summary.horizontal_length:.2f}",
                    f"3D distance: {summary.spatial_length:.2f}",
                    f"Bottom min elevation: {summary.bottom_min_elevation:.2f}",
                    f"Bottom max elevation: {summary.bottom_max_elevation:.2f}",
                ]
            )
            if result.top_segments:
                self.profile_message += "\n" + "\n".join(
                    [
                        f"Top min elevation: {summary.top_min_elevation:.2f}",
                        f"Top max elevation: {summary.top_max_elevation:.2f}",
                    ]
                )
        else:
            self.profile_message = "\n".join(
                [
                    f"{mode} terrain profile: no intersection with the current clip and line.",
                    f"Start: ({self.profile_start[0]:.2f}, {self.profile_start[1]:.2f}, {self.profile_start[2]:.2f})",
                    f"End: ({self.profile_end[0]:.2f}, {self.profile_end[1]:.2f}, {self.profile_end[2]:.2f})",
                    f"Horizontal distance: {summary.horizontal_length:.2f}",
                    f"3D distance: {summary.spatial_length:.2f}",
                ]
            )
        self._refresh_details_text()

    def _plot_chart_series(
        self,
        segments: list[tuple[np.ndarray, np.ndarray]],
        *,
        label: str,
        color: str,
    ) -> None:
        for index, (distances, elevations) in enumerate(segments):
            self.chart.line(
                distances,
                elevations,
                color=color,
                width=2.5,
                label=label if index == 0 else None,
            )

    def get_exact_mesh(self) -> pv.PolyData:
        if self.exact_mesh is None:
            print(f"Loading full-resolution STL from {self.mesh_path} for an exact profile measurement...")
            self.exact_mesh = load_stl_mesh(self.mesh_path)
            self.exact_cell_count = self.exact_mesh.n_cells
            self.lazy_exact_load = False
            self._refresh_header_text()
        return self.exact_mesh

    def measure_exact_profile(self) -> None:
        if not self.profile_enabled:
            self.status_message = "Enable the Profile tool before running an exact measurement."
            self._refresh_details_text()
            self.plotter.render()
            return

        self.status_message = "Computing exact profile..."
        self._refresh_details_text()
        self.plotter.render()

        try:
            mesh = self.get_exact_mesh()
            if self.clip_active:
                mesh = mesh.clip(normal=self.clip_normal, origin=self.clip_origin)
            result = build_profile_section(mesh, self.profile_start, self.profile_end, self.profile_mode)
        except MeshViewerError as exc:
            self._clear_profile_result("Exact profile failed.", str(exc))
            self.plotter.render()
            return

        self.status_message = (
            f"Showing an exact {profile_mode_label(self.profile_mode).lower()} profile "
            "from the full-resolution STL."
        )
        self._apply_profile_result(result, mode="Exact")
        self.plotter.render()

    def reset_profile(self, *, update_toggle: bool = True, render: bool = True) -> None:
        self.profile_start, self.profile_end = default_profile_endpoints(self.display_mesh.bounds)
        self._sync_line_widget()

        if update_toggle:
            self._set_checkbox_state(self.control_buttons["profile"], self.profile_enabled)

        if self.profile_enabled:
            self.status_message = "Profile reset to the current displayed bounds."
            self.refresh_profile_preview()
        else:
            self.profile_result_mode = "Off"
            self.profile_message = "Profile: off. Enable the Profile toggle to inspect a section."
            self._refresh_details_text()

        if render:
            self.plotter.render()

    def reset_clip(self, *, update_toggle: bool = True, render: bool = True) -> None:
        self.clip_normal = self.default_clip_normal
        self.clip_origin = self.default_clip_origin
        self.clip_active = False
        self.apply_render_mesh(self.render_mesh)
        self._sync_plane_widget()

        if update_toggle:
            self._set_checkbox_state(self.control_buttons["clip"], self.clip_enabled)

        self.status_message = "Clip reset. Showing the full shell."
        if self.profile_enabled:
            self.refresh_profile_preview()
        else:
            self._refresh_details_text()

        if render:
            self.plotter.render()

    def restore_all(self) -> None:
        self.set_profile_enabled(False)
        self.profile_mode = PROFILE_MODE_BOTTOM
        self._set_checkbox_state(self.control_buttons["full_shell"], False)
        self.set_clip_enabled(False)
        self.set_wireframe_visible(False)
        self._set_surface_opacity(DEFAULT_SURFACE_OPACITY)
        self._set_slider_value(DEFAULT_SURFACE_OPACITY)
        self.profile_start, self.profile_end = default_profile_endpoints(self.render_mesh.bounds)
        self._sync_line_widget()
        self.set_isometric_view(render=False)
        self.status_message = "Viewer restored to its default state."
        self.profile_message = "Profile: off. Enable the Profile toggle to inspect a section."
        self._refresh_details_text()
        self.plotter.render()

    def reset_camera(self) -> None:
        self.plotter.reset_camera()
        self.status_message = "Camera reset."
        self._refresh_details_text()
        self.plotter.render()

    def set_top_view(self, *, render: bool = True) -> None:
        self.plotter.enable_parallel_projection()
        self.plotter.view_xy()
        self.status_message = "Top view."
        self._refresh_details_text()
        if render:
            self.plotter.render()

    def set_bottom_view(self, *, render: bool = True) -> None:
        self.plotter.enable_parallel_projection()
        self.plotter.view_vector((0.0, 0.0, -1.0), viewup=(0.0, 1.0, 0.0))
        self.status_message = "Bottom view."
        self._refresh_details_text()
        if render:
            self.plotter.render()

    def set_isometric_view(self, *, render: bool = True) -> None:
        self.plotter.disable_parallel_projection()
        self.plotter.view_isometric()
        self.status_message = "Isometric view."
        self._refresh_details_text()
        if render:
            self.plotter.render()

    def show(self) -> None:
        title = f"STL Mesh Viewer - {self.mesh_path.name}"
        self.plotter.show(title=title)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View and inspect a terrain STL mesh.")
    parser.add_argument("stl_path", nargs="?", help="Path to the STL file to open.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        mesh_path = resolve_stl_path(args.stl_path)
        print(f"Loading STL mesh from {mesh_path}...")
        app = MeshViewerApp(mesh_path)
        app.build()
        print("Opening interactive mesh viewer window...")
        app.show()
        return 0
    except KeyboardInterrupt:
        print("\nViewer cancelled by user.")
        return 1
    except MeshViewerError as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
