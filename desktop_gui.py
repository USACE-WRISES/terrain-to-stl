from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QProcess, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from desktop_gui_support import (
    DEFAULT_VERTICAL_EXAGGERATION,
    VERTICAL_EXAGGERATION_OPTIONS,
    build_associated_file_entries,
    normalize_desktop_path,
    resolve_desktop_terrain_selection,
    scene_bounds_from_world,
    scene_normal_from_world,
    scene_point_from_world,
    world_normal_from_scene,
    world_point_from_scene,
)
from mesh_viewer import (
    BACKGROUND_BOTTOM,
    BACKGROUND_TOP,
    DEFAULT_SURFACE_OPACITY,
    PREVIEW_EMPTY_FRACTION,
    PROFILE_BOTTOM_COLOR,
    PROFILE_MODE_BOTTOM,
    PROFILE_MODE_FULL_SHELL,
    PROFILE_SCENE_COLOR,
    PROFILE_TOP_COLOR,
    TERRAIN_COLORMAP,
    MeshLoadResult,
    MeshViewerError,
    build_profile_section,
    default_profile_endpoints,
    format_file_size,
    load_mesh_state,
    load_stl_mesh,
    profile_mode_label,
    to_rgb,
    validate_stl_path,
)


def format_bytes(byte_count: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(byte_count)
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    precision = 0 if unit_index == 0 else 2
    return f"{value:.{precision}f} {units[unit_index]}"


def format_duration(seconds: float) -> str:
    total_seconds = max(float(seconds), 0.0)
    if total_seconds < 60.0:
        return f"{total_seconds:.1f} sec"

    rounded_seconds = int(round(total_seconds))
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours} hr {minutes} min"
    if secs == 0:
        return f"{minutes} min"
    return f"{minutes} min {secs} sec"


def choose_stl_copy_destination(
    parent: QWidget,
    *,
    source_path: Path,
    title: str,
) -> Path | None:
    selected, _ = QFileDialog.getSaveFileName(
        parent,
        title,
        str(source_path),
        "STL Files (*.stl)",
    )
    if not selected:
        return None

    destination = Path(selected)
    if destination.suffix.lower() != ".stl":
        destination = destination.with_suffix(".stl")
    return destination


def scale_mesh_for_scene(mesh: pv.PolyData, center_z: float, vertical_exaggeration: float) -> pv.PolyData:
    scaled = mesh.copy(deep=True)
    if scaled.n_points == 0:
        return scaled

    points = np.asarray(scaled.points)
    points[:, 2] = center_z + ((points[:, 2] - center_z) * float(vertical_exaggeration))
    scaled.points = points
    return scaled


class ProfileChartCanvas(FigureCanvas):
    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(4.6, 2.6), tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.reset()

    def reset(self) -> None:
        self.axes.clear()
        self.axes.set_xlabel("Distance")
        self.axes.set_ylabel("Elevation")
        self.axes.grid(True, alpha=0.25)
        self.draw_idle()


class ViewerLoadWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    status_changed = Signal(str)

    def __init__(self, mesh_path: Path, *, mode: str) -> None:
        super().__init__()
        self.mesh_path = mesh_path
        self.mode = mode

    @Slot()
    def run(self) -> None:
        try:
            validate_stl_path(self.mesh_path)
            if self.mode == "preview":
                result = load_mesh_state(self.mesh_path, status_callback=self.status_changed.emit)
            elif self.mode == "exact":
                result = load_stl_mesh(self.mesh_path, status_callback=self.status_changed.emit)
            else:
                raise MeshViewerError(f"Unsupported STL load mode: {self.mode}")
        except MemoryError:
            self.failed.emit("The STL load ran out of memory.")
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self.finished.emit(result)


class DesktopViewerWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.mesh_path: Path | None = None
        self.render_mesh: pv.PolyData | None = None
        self.display_mesh_world: pv.PolyData | None = None
        self.display_mesh: pv.PolyData | None = None
        self.exact_mesh: pv.PolyData | None = None
        self.profile_section_world = pv.PolyData()
        self.profile_section_mesh = pv.PolyData()
        self.render_mode = "None"
        self.file_size_bytes = 0
        self.exact_cell_count: int | None = None
        self.lazy_exact_load = False
        self.exact_load_warning_required = False
        self.default_clip_normal = (0.0, 0.0, 1.0)
        self.default_clip_origin = (0.0, 0.0, 0.0)
        self.clip_normal = self.default_clip_normal
        self.clip_origin = self.default_clip_origin
        self.profile_start = None
        self.profile_end = None
        self.clip_enabled = False
        self.clip_active = False
        self.profile_enabled = False
        self.profile_mode = PROFILE_MODE_BOTTOM
        self.opacity = DEFAULT_SURFACE_OPACITY
        self.vertical_exaggeration = DEFAULT_VERTICAL_EXAGGERATION
        self.wireframe_visible = False
        self.status_message = "Open an STL file to inspect it."
        self.profile_message = "Profile: off. Enable the Profile tool to inspect a section."

        self.plotter = QtInteractor(self)
        self.surface_actor = None
        self.wire_actor = None
        self.profile_actor = None
        self.line_widget = None
        self.plane_widget = None
        self.loading_overlay: QWidget | None = None
        self.loading_status_label: QLabel | None = None
        self.loading_progress_bar: QProgressBar | None = None
        self._viewer_busy = False
        self._loading_mode: str | None = None
        self._loading_thread: QThread | None = None
        self._loading_worker: ViewerLoadWorker | None = None
        self._pending_exact_profile = False

        self._build_ui()
        self._apply_empty_viewer_state(render=False)

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter)

        plot_container = QWidget(self)
        plot_layout = QStackedLayout(plot_container)
        plot_layout.setStackingMode(QStackedLayout.StackAll)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.plotter)

        self.loading_overlay = QWidget(plot_container)
        self.loading_overlay.setStyleSheet("background-color: rgba(244, 247, 250, 220);")
        overlay_layout = QVBoxLayout(self.loading_overlay)
        overlay_layout.setContentsMargins(20, 20, 20, 20)
        overlay_layout.addStretch(1)
        loading_panel = QWidget(self.loading_overlay)
        loading_panel.setMinimumWidth(340)
        loading_panel.setMaximumWidth(560)
        loading_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        loading_panel.setStyleSheet(
            "background-color: rgba(255, 255, 255, 232);"
            "border: 1px solid rgba(95, 107, 119, 90);"
            "border-radius: 14px;"
        )
        loading_panel_layout = QVBoxLayout(loading_panel)
        loading_panel_layout.setContentsMargins(28, 24, 28, 24)
        loading_panel_layout.setSpacing(12)
        loading_title = QLabel("Loading STL")
        loading_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_title.setStyleSheet("font-size: 18px; font-weight: 600; color: #1f2933;")
        loading_panel_layout.addWidget(loading_title)
        self.loading_status_label = QLabel("Reading STL data...")
        self.loading_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_status_label.setWordWrap(True)
        self.loading_status_label.setStyleSheet("font-size: 13px; color: #364350;")
        loading_panel_layout.addWidget(self.loading_status_label)
        self.loading_progress_bar = QProgressBar(self.loading_overlay)
        self.loading_progress_bar.setRange(0, 0)
        self.loading_progress_bar.setTextVisible(False)
        self.loading_progress_bar.setFixedHeight(14)
        loading_panel_layout.addWidget(self.loading_progress_bar)
        overlay_layout.addWidget(loading_panel, alignment=Qt.AlignmentFlag.AlignCenter)
        overlay_layout.addStretch(1)
        plot_layout.addWidget(self.loading_overlay)
        splitter.addWidget(plot_container)

        panel = QWidget(self)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(10)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)

        open_section = QVBoxLayout()
        open_section.setSpacing(6)
        open_row = QHBoxLayout()
        self.open_button = QPushButton("Open STL")
        self.open_button.clicked.connect(self.open_stl_dialog)
        open_row.addWidget(self.open_button)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_viewer)
        open_row.addWidget(self.clear_button)
        self.download_button = QPushButton("Download STL")
        self.download_button.clicked.connect(self.download_loaded_stl)
        open_row.addWidget(self.download_button)
        open_row.addStretch(1)
        open_section.addLayout(open_row)
        self.path_label = QLabel("No STL loaded.")
        self.path_label.setWordWrap(True)
        open_section.addWidget(self.path_label)
        panel_layout.addLayout(open_section)

        self.header_label = QLabel("Viewer idle.")
        self.header_label.setWordWrap(True)
        panel_layout.addWidget(self.header_label)

        view_group = QGroupBox("View")
        view_layout = QGridLayout(view_group)
        view_layout.setHorizontalSpacing(8)
        view_layout.setVerticalSpacing(8)
        self.top_view_button = QPushButton("Top")
        self.top_view_button.clicked.connect(self.set_top_view)
        self.bottom_view_button = QPushButton("Bottom")
        self.bottom_view_button.clicked.connect(self.set_bottom_view)
        self.iso_view_button = QPushButton("Isometric")
        self.iso_view_button.clicked.connect(self.set_isometric_view)
        self.reset_camera_button = QPushButton("Reset Camera")
        self.reset_camera_button.clicked.connect(self.reset_camera)
        view_layout.addWidget(self.top_view_button, 0, 0)
        view_layout.addWidget(self.bottom_view_button, 0, 1)
        view_layout.addWidget(self.iso_view_button, 1, 0)
        view_layout.addWidget(self.reset_camera_button, 1, 1)
        panel_layout.addWidget(view_group)

        inspect_group = QGroupBox("Inspect")
        inspect_layout = QVBoxLayout(inspect_group)
        clip_row = QHBoxLayout()
        self.clip_checkbox = QCheckBox("Clip Tool")
        self.clip_checkbox.toggled.connect(self.set_clip_enabled)
        self.reset_clip_button = QPushButton("Reset Clip")
        self.reset_clip_button.clicked.connect(self.reset_clip)
        clip_row.addWidget(self.clip_checkbox)
        clip_row.addWidget(self.reset_clip_button)
        inspect_layout.addLayout(clip_row)

        profile_row = QHBoxLayout()
        self.profile_checkbox = QCheckBox("Profile Tool")
        self.profile_checkbox.toggled.connect(self.set_profile_enabled)
        self.full_shell_checkbox = QCheckBox("Full Shell")
        self.full_shell_checkbox.toggled.connect(self.set_full_shell_mode)
        profile_row.addWidget(self.profile_checkbox)
        profile_row.addWidget(self.full_shell_checkbox)
        inspect_layout.addLayout(profile_row)

        profile_actions = QHBoxLayout()
        self.reset_profile_button = QPushButton("Reset Profile")
        self.reset_profile_button.clicked.connect(self.reset_profile)
        self.measure_exact_button = QPushButton("Measure Exact")
        self.measure_exact_button.clicked.connect(self.measure_exact_profile)
        profile_actions.addWidget(self.reset_profile_button)
        profile_actions.addWidget(self.measure_exact_button)
        inspect_layout.addLayout(profile_actions)
        panel_layout.addWidget(inspect_group)

        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        exaggeration_row = QHBoxLayout()
        exaggeration_row.addWidget(QLabel("Vertical exaggeration"))
        self.vertical_exaggeration_combo = QComboBox()
        for factor in VERTICAL_EXAGGERATION_OPTIONS:
            self.vertical_exaggeration_combo.addItem(f"{factor}x", factor)
        self.vertical_exaggeration_combo.setCurrentText(f"{DEFAULT_VERTICAL_EXAGGERATION}x")
        self.vertical_exaggeration_combo.currentIndexChanged.connect(self.on_vertical_exaggeration_changed)
        exaggeration_row.addWidget(self.vertical_exaggeration_combo, 1)
        display_layout.addLayout(exaggeration_row)

        self.wireframe_checkbox = QCheckBox("Wireframe")
        self.wireframe_checkbox.toggled.connect(self.set_wireframe_visible)
        display_layout.addWidget(self.wireframe_checkbox)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Surface opacity"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(int(DEFAULT_SURFACE_OPACITY * 100))
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        opacity_row.addWidget(self.opacity_slider, 1)
        self.opacity_value_label = QLabel(f"{DEFAULT_SURFACE_OPACITY:.2f}")
        opacity_row.addWidget(self.opacity_value_label)
        display_layout.addLayout(opacity_row)

        self.restore_button = QPushButton("Restore All")
        self.restore_button.clicked.connect(self.restore_all)
        display_layout.addWidget(self.restore_button)
        panel_layout.addWidget(display_group)

        self.details_text = QPlainTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMinimumHeight(130)
        panel_layout.addWidget(self.details_text, 1)

        self.chart_canvas = ProfileChartCanvas(self)
        self.chart_canvas.setMinimumHeight(170)
        self.chart_canvas.hide()
        panel_layout.addWidget(self.chart_canvas, 1)

        self.loading_overlay.hide()
        self._set_controls_enabled(False)

    def _set_checkbox_value(self, widget: QCheckBox, value: bool) -> None:
        widget.blockSignals(True)
        widget.setChecked(bool(value))
        widget.blockSignals(False)

    def _set_controls_enabled(self, enabled: bool) -> None:
        has_mesh = bool(enabled)
        self.open_button.setEnabled(not self._viewer_busy)
        for widget in (
            self.top_view_button,
            self.bottom_view_button,
            self.iso_view_button,
            self.reset_camera_button,
            self.clip_checkbox,
            self.reset_clip_button,
            self.profile_checkbox,
            self.full_shell_checkbox,
            self.reset_profile_button,
            self.measure_exact_button,
            self.vertical_exaggeration_combo,
            self.wireframe_checkbox,
            self.opacity_slider,
            self.restore_button,
            self.clear_button,
            self.download_button,
        ):
            widget.setEnabled(has_mesh and not self._viewer_busy)

        self.full_shell_checkbox.setEnabled(has_mesh and not self._viewer_busy and self.profile_enabled)
        self.measure_exact_button.setEnabled(has_mesh and not self._viewer_busy and self.profile_enabled)

    def _show_loading_overlay(self, visible: bool, message: str | None = None) -> None:
        if self.loading_status_label is not None and message:
            self.loading_status_label.setText(message)
        if self.loading_overlay is not None:
            self.loading_overlay.setVisible(visible)
            if visible:
                self.loading_overlay.raise_()

    def _set_viewer_busy(self, busy: bool, message: str | None = None) -> None:
        self._viewer_busy = bool(busy)
        self._show_loading_overlay(self._viewer_busy, message)
        self._set_controls_enabled(self.render_mesh is not None)

    def _on_load_status_changed(self, message: str) -> None:
        self._show_loading_overlay(True, message)

    def _reset_details(self) -> None:
        self.header_label.setText("Viewer idle.")
        self.details_text.setPlainText(self.status_message)

    def _apply_empty_viewer_state(self, *, render: bool) -> None:
        self.mesh_path = None
        self.render_mesh = None
        self.display_mesh_world = None
        self.display_mesh = None
        self.exact_mesh = None
        self.profile_section_world = pv.PolyData()
        self.profile_section_mesh = pv.PolyData()
        self.render_mode = "None"
        self.file_size_bytes = 0
        self.exact_cell_count = None
        self.lazy_exact_load = False
        self.exact_load_warning_required = False
        self.default_clip_normal = (0.0, 0.0, 1.0)
        self.default_clip_origin = (0.0, 0.0, 0.0)
        self.clip_normal = self.default_clip_normal
        self.clip_origin = self.default_clip_origin
        self.profile_start = None
        self.profile_end = None
        self.clip_enabled = False
        self.clip_active = False
        self.profile_enabled = False
        self.profile_mode = PROFILE_MODE_BOTTOM
        self.opacity = DEFAULT_SURFACE_OPACITY
        self.vertical_exaggeration = DEFAULT_VERTICAL_EXAGGERATION
        self.wireframe_visible = False
        self.status_message = "Open an STL file to inspect it."
        self.profile_message = "Profile: off. Enable the Profile tool to inspect a section."
        self.surface_actor = None
        self.wire_actor = None
        self.profile_actor = None
        self.line_widget = None
        self.plane_widget = None

        self.plotter.clear_plane_widgets()
        self.plotter.clear_line_widgets()
        self.plotter.clear()
        self.plotter.set_background(BACKGROUND_BOTTOM, top=BACKGROUND_TOP)
        self.plotter.hide_axes()
        self.path_label.setText("No STL loaded.")
        self.vertical_exaggeration_combo.blockSignals(True)
        self.vertical_exaggeration_combo.setCurrentText(f"{DEFAULT_VERTICAL_EXAGGERATION}x")
        self.vertical_exaggeration_combo.blockSignals(False)
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(int(DEFAULT_SURFACE_OPACITY * 100))
        self.opacity_slider.blockSignals(False)
        self.opacity_value_label.setText(f"{DEFAULT_SURFACE_OPACITY:.2f}")
        self._set_checkbox_value(self.clip_checkbox, False)
        self._set_checkbox_value(self.profile_checkbox, False)
        self._set_checkbox_value(self.full_shell_checkbox, False)
        self._set_checkbox_value(self.wireframe_checkbox, False)
        self.chart_canvas.reset()
        self.chart_canvas.hide()
        self._pending_exact_profile = False
        self._show_loading_overlay(False)
        self._set_controls_enabled(False)
        self._reset_details()
        if render:
            self.plotter.render()

    def clear_viewer(self, _checked: bool | None = None) -> None:
        self._apply_empty_viewer_state(render=True)

    def download_loaded_stl(self, _checked: bool | None = None) -> None:
        if self.mesh_path is None or not self.mesh_path.exists():
            QMessageBox.warning(self, "Download STL", "The loaded STL file is no longer available on disk.")
            return

        destination = choose_stl_copy_destination(
            self,
            source_path=self.mesh_path,
            title="Save STL Copy",
        )
        if destination is None:
            return

        try:
            if destination.resolve() == self.mesh_path.resolve():
                self.status_message = f"STL is already available at {destination}."
                self._update_details_text()
                if self.render_mesh is not None:
                    self.plotter.render()
                return
            shutil.copy2(self.mesh_path, destination)
        except OSError as exc:
            QMessageBox.critical(self, "Download STL Failed", f"Could not save the STL copy.\n\n{exc}")
            return

        self.status_message = f"Saved STL copy to {destination}."
        self._update_details_text()
        if self.render_mesh is not None:
            self.plotter.render()

    def _update_header_text(self) -> None:
        if self.mesh_path is None or self.render_mesh is None:
            self.header_label.setText("Viewer idle.")
            return

        lines = [
            f"Mesh Viewer: {self.mesh_path.name}",
            f"Render mesh: {self.render_mode} ({self.render_mesh.n_cells:,} triangles shown)",
            f"STL size: {format_file_size(self.file_size_bytes)}",
            f"Vertical exaggeration: {self.vertical_exaggeration}x",
        ]
        if self.exact_cell_count is not None:
            lines.append(f"Source triangles: {self.exact_cell_count:,}")
        if self.render_mode == "Preview":
            if self.exact_mesh is None:
                if self.exact_load_warning_required:
                    lines.append("Exact profile: full-resolution load available on demand after warning")
                else:
                    lines.append("Exact profile: loads the full STL on demand")
            else:
                lines.append(f"Exact profile: cached full mesh ({self.exact_mesh.n_cells:,} triangles)")
        else:
            lines.append("Exact profile: using the loaded display mesh")
        self.header_label.setText("\n".join(lines))

    def _update_details_text(self) -> None:
        if self.display_mesh is None:
            self.details_text.setPlainText(self.status_message)
            return

        lines = [f"Visible triangles: {self.display_mesh.n_cells:,}"]
        if self.status_message:
            lines.append(self.status_message)
        if self.profile_enabled:
            lines.append(f"Profile mode: {profile_mode_label(self.profile_mode)}")
        if self.profile_message:
            lines.append("")
            lines.extend(self.profile_message.splitlines())
        self.details_text.setPlainText("\n".join(lines).strip())

    def _configure_widgets(self) -> None:
        if self.render_mesh is None:
            return

        bounds = scene_bounds_from_world(self.render_mesh.bounds, self._mesh_center_z(), self.vertical_exaggeration)
        self.plane_widget = self.plotter.add_plane_widget(
            self.on_clip_plane_changed,
            bounds=bounds,
            origin=self._scene_point(self.default_clip_origin),
            normal="z",
            interaction_event="end",
            color="#6e8ba8",
            test_callback=False,
        )
        self.plane_widget.SetNormal(*self._scene_normal(self.default_clip_normal))
        self.plane_widget.SetOrigin(*self._scene_point(self.default_clip_origin))
        self.plane_widget.SetEnabled(False)

        self.line_widget = self.plotter.add_line_widget(
            self.on_profile_line_changed,
            bounds=bounds,
            use_vertices=True,
            interaction_event="end",
            color=PROFILE_SCENE_COLOR,
        )
        self.line_widget.SetPoint1(*self._scene_point(self.profile_start))
        self.line_widget.SetPoint2(*self._scene_point(self.profile_end))
        self.line_widget.SetHandleSize(0.02)
        self.line_widget.GetLineProperty().SetColor(*to_rgb(PROFILE_SCENE_COLOR))
        self.line_widget.GetLineProperty().SetLineWidth(4.0)
        self.line_widget.GetSelectedLineProperty().SetColor(*to_rgb("#ffffff"))
        self.line_widget.GetSelectedLineProperty().SetLineWidth(5.0)
        self.line_widget.GetHandleProperty().SetColor(*to_rgb(PROFILE_SCENE_COLOR))
        self.line_widget.GetSelectedHandleProperty().SetColor(*to_rgb("#ffffff"))
        self.line_widget.SetEnabled(False)

    def _mesh_center_z(self) -> float:
        if self.render_mesh is None:
            return 0.0
        return float(self.render_mesh.center[2])

    def _scene_point(self, point: np.ndarray | tuple[float, float, float]) -> tuple[float, float, float]:
        return scene_point_from_world(point, self._mesh_center_z(), self.vertical_exaggeration)

    def _world_point(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        return world_point_from_scene(point, self._mesh_center_z(), self.vertical_exaggeration)

    def _scene_normal(self, normal: tuple[float, float, float]) -> tuple[float, float, float]:
        return scene_normal_from_world(normal, self.vertical_exaggeration)

    def _world_normal(self, normal: tuple[float, float, float]) -> tuple[float, float, float]:
        return world_normal_from_scene(normal, self.vertical_exaggeration)

    def _rebuild_widgets(self) -> None:
        self.plotter.clear_plane_widgets()
        self.plotter.clear_line_widgets()
        self.plane_widget = None
        self.line_widget = None
        self._configure_widgets()
        if self.plane_widget is not None:
            self.plane_widget.SetEnabled(int(self.clip_enabled))
        if self.line_widget is not None:
            self.line_widget.SetEnabled(int(self.profile_enabled))

    def _refresh_scene_meshes(self, *, rebuild_widgets: bool = False, render: bool = True) -> None:
        if self.display_mesh_world is None or self.display_mesh is None:
            return

        self.display_mesh.copy_from(
            scale_mesh_for_scene(self.display_mesh_world, self._mesh_center_z(), self.vertical_exaggeration),
            deep=True,
        )
        self.profile_section_mesh.copy_from(
            scale_mesh_for_scene(self.profile_section_world, self._mesh_center_z(), self.vertical_exaggeration),
            deep=True,
        )
        if rebuild_widgets:
            self._rebuild_widgets()
        else:
            self._sync_plane_widget()
            self._sync_line_widget()

        if self.surface_actor is not None:
            self.surface_actor.GetProperty().SetOpacity(self.opacity)
        if self.wire_actor is not None:
            self.wire_actor.SetVisibility(self.wireframe_visible)
        if self.profile_actor is not None:
            self.profile_actor.SetVisibility(self.profile_enabled and self.profile_section_mesh.n_cells > 0)

        self._update_header_text()
        self._update_details_text()
        if render:
            self.plotter.render()

    def open_stl_dialog(self, _checked: bool | None = None) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open STL File",
            str(Path.home()),
            "STL Files (*.stl)",
        )
        if not selected:
            return
        self.load_stl(Path(selected))

    def _start_async_load(self, mesh_path: Path, *, mode: str, start_message: str, error_title: str) -> bool:
        try:
            validate_stl_path(mesh_path)
        except MeshViewerError as exc:
            QMessageBox.critical(self, error_title, str(exc))
            return False

        if self._loading_thread is not None:
            QMessageBox.warning(self, "Viewer Busy", "The viewer is already loading an STL.")
            return False

        worker = ViewerLoadWorker(mesh_path, mode=mode)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.status_changed.connect(self._on_load_status_changed)
        worker.finished.connect(self._on_async_load_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(self._on_async_load_failed)
        worker.failed.connect(thread.quit)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._on_async_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._loading_thread = thread
        self._loading_worker = worker
        self._loading_mode = mode
        self._set_viewer_busy(True, start_message)
        thread.start()
        return True

    def _on_async_thread_finished(self) -> None:
        self._loading_thread = None
        self._loading_worker = None
        self._loading_mode = None

    def _on_async_load_finished(self, payload: object) -> None:
        mode = self._loading_mode
        if mode == "exact":
            self._set_viewer_busy(False)
            self._show_loading_overlay(False)
            self.exact_mesh = payload if isinstance(payload, pv.PolyData) else None
            if self.exact_mesh is not None:
                self.exact_cell_count = self.exact_mesh.n_cells
                self.lazy_exact_load = False
                self.exact_load_warning_required = False
                self._update_header_text()
            if self._pending_exact_profile:
                self._pending_exact_profile = False
                self._run_exact_profile_measurement()
            else:
                self.status_message = "Full-resolution STL loaded."
                self._update_details_text()
                if self.render_mesh is not None:
                    self.plotter.render()
            return

        if not isinstance(payload, MeshLoadResult):
            self._on_async_load_failed("The viewer did not receive a valid STL load result.")
            return

        self._apply_loaded_mesh_state(payload)
        self._set_viewer_busy(False)
        self._show_loading_overlay(False)
        self.plotter.render()

    def _on_async_load_failed(self, message: str) -> None:
        mode = self._loading_mode
        self._pending_exact_profile = False
        self._set_viewer_busy(False)
        self._show_loading_overlay(False)
        self.status_message = message
        self._update_details_text()
        if self.render_mesh is not None:
            self.plotter.render()
        title = "Exact STL Load Failed" if mode == "exact" else "Open STL Failed"
        QMessageBox.critical(self, title, message)

    def load_stl(self, mesh_path: Path) -> None:
        self._start_async_load(
            mesh_path,
            mode="preview",
            start_message="Reading STL header...",
            error_title="Open STL Failed",
        )

    def _apply_loaded_mesh_state(self, load_result: MeshLoadResult) -> None:
        mesh_path = self._loading_worker.mesh_path if self._loading_worker is not None else self.mesh_path
        if mesh_path is None:
            raise MeshViewerError("No STL path is available for the loaded mesh.")

        self.mesh_path = mesh_path
        self.render_mesh = load_result.render_mesh
        self.exact_mesh = load_result.exact_mesh
        self.render_mode = load_result.render_mode
        self.file_size_bytes = load_result.file_size_bytes
        self.exact_cell_count = load_result.exact_cell_count
        self.lazy_exact_load = load_result.needs_lazy_exact_load
        self.exact_load_warning_required = load_result.exact_load_warning_required
        self.display_mesh_world = self.render_mesh.copy(deep=True)
        self.display_mesh = scale_mesh_for_scene(
            self.display_mesh_world,
            self._mesh_center_z(),
            self.vertical_exaggeration,
        )
        self.profile_section_world = pv.PolyData()
        self.profile_section_mesh = pv.PolyData()
        self.default_clip_origin = tuple(float(value) for value in self.render_mesh.center)
        self.clip_normal = self.default_clip_normal
        self.clip_origin = self.default_clip_origin
        self.profile_start, self.profile_end = default_profile_endpoints(self.render_mesh.bounds)
        self.clip_enabled = False
        self.clip_active = False
        self.profile_enabled = False
        self.profile_mode = PROFILE_MODE_BOTTOM
        self.opacity = DEFAULT_SURFACE_OPACITY
        self.vertical_exaggeration = DEFAULT_VERTICAL_EXAGGERATION
        self.wireframe_visible = False
        if self.render_mode == "Preview" and self.lazy_exact_load:
            if self.exact_load_warning_required:
                self.status_message = (
                    "Preview mesh loaded. Full-resolution loading is available on demand and may take a while."
                )
            else:
                self.status_message = "Preview mesh loaded. Full-resolution data is available on demand."
        else:
            self.status_message = "STL loaded. Use the controls to inspect the shell."
        self.profile_message = "Profile: off. Enable the Profile tool to inspect a section."

        self.plotter.clear()
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
        self._configure_widgets()

        self.path_label.setText(str(mesh_path))
        self.vertical_exaggeration_combo.blockSignals(True)
        self.vertical_exaggeration_combo.setCurrentText(f"{DEFAULT_VERTICAL_EXAGGERATION}x")
        self.vertical_exaggeration_combo.blockSignals(False)
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(int(DEFAULT_SURFACE_OPACITY * 100))
        self.opacity_slider.blockSignals(False)
        self.opacity_value_label.setText(f"{DEFAULT_SURFACE_OPACITY:.2f}")
        self._set_checkbox_value(self.clip_checkbox, False)
        self._set_checkbox_value(self.profile_checkbox, False)
        self._set_checkbox_value(self.full_shell_checkbox, False)
        self._set_checkbox_value(self.wireframe_checkbox, False)
        self.chart_canvas.reset()
        self.chart_canvas.hide()
        self._set_controls_enabled(True)
        self._update_header_text()
        self._update_details_text()
        self.set_isometric_view(render=False)

    def apply_render_mesh(self, mesh: pv.PolyData) -> None:
        if self.display_mesh_world is None:
            return

        self.display_mesh_world.copy_from(mesh, deep=True)
        self._refresh_scene_meshes(render=False)

    def _sync_plane_widget(self) -> None:
        if self.plane_widget is None:
            return
        self.plane_widget.SetNormal(*self._scene_normal(self.clip_normal))
        self.plane_widget.SetOrigin(*self._scene_point(self.clip_origin))

    def _sync_line_widget(self) -> None:
        if self.line_widget is None or self.profile_start is None or self.profile_end is None:
            return
        self.line_widget.SetPoint1(*self._scene_point(self.profile_start))
        self.line_widget.SetPoint2(*self._scene_point(self.profile_end))

    def on_opacity_changed(self, value: int) -> None:
        self.opacity = float(value) / 100.0
        self.opacity_value_label.setText(f"{self.opacity:.2f}")
        if self.surface_actor is not None:
            self.surface_actor.GetProperty().SetOpacity(self.opacity)
            self.plotter.render()

    def on_vertical_exaggeration_changed(self, _index: int) -> None:
        data = self.vertical_exaggeration_combo.currentData()
        if data is None:
            return

        next_factor = int(data)
        if next_factor == self.vertical_exaggeration:
            return

        self.vertical_exaggeration = next_factor
        self.status_message = f"Vertical exaggeration set to {self.vertical_exaggeration}x."
        self._refresh_scene_meshes(rebuild_widgets=True, render=False)
        self.plotter.render()

    def set_wireframe_visible(self, visible: bool) -> None:
        self.wireframe_visible = bool(visible)
        if self.wire_actor is not None:
            self.wire_actor.SetVisibility(self.wireframe_visible)
        self.status_message = "Wireframe enabled." if self.wireframe_visible else "Wireframe hidden."
        self._update_details_text()
        if self.render_mesh is not None:
            self.plotter.render()

    def set_clip_enabled(self, enabled: bool) -> None:
        self.clip_enabled = bool(enabled)
        if self.plane_widget is not None:
            self.plane_widget.SetEnabled(int(self.clip_enabled))

        if self.clip_enabled:
            self._sync_plane_widget()
            self.status_message = "Clip tool enabled. Drag the plane to open the shell."
        else:
            self.reset_clip(update_checkbox=False, render=False)
            self.status_message = "Clip tool disabled. Showing the full shell."
        self._update_details_text()
        if self.render_mesh is not None:
            self.plotter.render()

    def on_clip_plane_changed(self, normal: tuple[float, float, float], origin: tuple[float, float, float]) -> None:
        if not self.clip_enabled or self.render_mesh is None:
            return

        world_normal = self._world_normal(normal)
        world_origin = self._world_point(origin)
        clipped = self.render_mesh.clip(normal=world_normal, origin=world_origin)
        if clipped.n_cells == 0 or (clipped.n_cells / max(self.render_mesh.n_cells, 1)) < PREVIEW_EMPTY_FRACTION:
            self.status_message = "Clip rejected because it would hide almost the entire shell."
            self._sync_plane_widget()
            self._update_details_text()
            self.plotter.render()
            return

        self.clip_normal = tuple(float(value) for value in world_normal)
        self.clip_origin = tuple(float(value) for value in world_origin)
        self.clip_active = True
        self.apply_render_mesh(clipped)
        self.status_message = f"Clip applied on the {self.render_mode.lower()} mesh."
        if self.profile_enabled:
            self.refresh_profile_preview()
        else:
            self._update_details_text()
        self.plotter.render()

    def set_profile_enabled(self, enabled: bool) -> None:
        self.profile_enabled = bool(enabled)
        if self.line_widget is not None:
            self.line_widget.SetEnabled(int(self.profile_enabled))
        if self.profile_actor is not None:
            self.profile_actor.SetVisibility(self.profile_enabled)

        self.full_shell_checkbox.setEnabled(self.profile_enabled)
        self.measure_exact_button.setEnabled(self.profile_enabled)

        if self.profile_enabled:
            self.reset_profile(update_checkbox=False, render=False)
            self.status_message = f"Profile tool enabled in {profile_mode_label(self.profile_mode)} mode."
        else:
            self.profile_message = "Profile: off. Enable the Profile tool to inspect a section."
            self.chart_canvas.reset()
            self.chart_canvas.hide()
            self.profile_section_world.copy_from(pv.PolyData(), deep=True)
            self.profile_section_mesh.copy_from(pv.PolyData(), deep=True)
            if self.profile_actor is not None:
                self.profile_actor.SetVisibility(False)
            self.status_message = "Profile tool disabled."
            self._set_checkbox_value(self.full_shell_checkbox, False)
        self._update_details_text()
        if self.render_mesh is not None:
            self.plotter.render()

    def set_full_shell_mode(self, enabled: bool) -> None:
        self.profile_mode = PROFILE_MODE_FULL_SHELL if enabled else PROFILE_MODE_BOTTOM
        if self.profile_enabled:
            self.status_message = f"Profile mode set to {profile_mode_label(self.profile_mode)}."
            self.refresh_profile_preview()
        else:
            self._update_details_text()

    def on_profile_line_changed(
        self,
        pointa: tuple[float, float, float],
        pointb: tuple[float, float, float],
    ) -> None:
        if not self.profile_enabled:
            return

        self.profile_start = np.asarray(self._world_point(pointa), dtype=float)
        self.profile_end = np.asarray(self._world_point(pointb), dtype=float)
        self.status_message = f"Preview {profile_mode_label(self.profile_mode).lower()} profile updated."
        self.refresh_profile_preview()
        self.plotter.render()

    def _clear_profile_result(self, status_message: str, profile_message: str) -> None:
        self.profile_section_world.copy_from(pv.PolyData(), deep=True)
        self.profile_section_mesh.copy_from(pv.PolyData(), deep=True)
        if self.profile_actor is not None:
            self.profile_actor.SetVisibility(False)
        self.chart_canvas.reset()
        if self.profile_enabled:
            self.chart_canvas.show()
        else:
            self.chart_canvas.hide()
        self.status_message = status_message
        self.profile_message = profile_message
        self._update_details_text()

    def _plot_chart_series(
        self,
        segments: list[tuple[object, object]],
        *,
        label: str,
        color: str,
    ) -> None:
        for index, (distances, elevations) in enumerate(segments):
            self.chart_canvas.axes.plot(
                distances,
                elevations,
                color=color,
                linewidth=2.3,
                label=label if index == 0 else None,
            )

    def _apply_profile_result(self, result, *, mode: str) -> None:
        summary = result.summary
        self.profile_section_world.copy_from(result.section_mesh, deep=True)
        self.profile_section_mesh.copy_from(
            scale_mesh_for_scene(result.section_mesh, self._mesh_center_z(), self.vertical_exaggeration),
            deep=True,
        )
        if self.profile_actor is not None:
            self.profile_actor.SetVisibility(self.profile_enabled and result.section_mesh.n_cells > 0)

        self.chart_canvas.reset()
        if result.bottom_segments:
            self._plot_chart_series(result.bottom_segments, label=f"{mode} bottom", color=PROFILE_BOTTOM_COLOR)
            if result.top_segments:
                self._plot_chart_series(result.top_segments, label=f"{mode} top", color=PROFILE_TOP_COLOR)
                self.chart_canvas.axes.legend(loc="best")
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
        self.chart_canvas.show()
        self.chart_canvas.draw_idle()
        self._update_details_text()

    def refresh_profile_preview(self) -> None:
        if not self.profile_enabled or self.display_mesh_world is None:
            return

        try:
            result = build_profile_section(
                self.display_mesh_world,
                self.profile_start,
                self.profile_end,
                self.profile_mode,
            )
        except MeshViewerError as exc:
            self._clear_profile_result("Preview profile unavailable.", str(exc))
            return

        self._apply_profile_result(result, mode="Preview")

    def _confirm_exact_load(self) -> bool:
        if not self.exact_load_warning_required or self.mesh_path is None:
            return True

        triangle_text = (
            f"{self.exact_cell_count:,} triangles"
            if self.exact_cell_count is not None
            else "an unknown number of triangles"
        )
        message = (
            f"{self.mesh_path.name} is {format_file_size(self.file_size_bytes)} and contains {triangle_text}.\n\n"
            "Loading the full-resolution STL may take a while or run out of memory.\n\n"
            "Do you want to try the full-resolution load anyway?"
        )
        result = QMessageBox.warning(
            self,
            "Load Full-Resolution STL?",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return result == QMessageBox.StandardButton.Yes

    def _run_exact_profile_measurement(self) -> None:
        if self.exact_mesh is None:
            raise MeshViewerError("The full-resolution STL is not loaded.")

        self.status_message = "Computing exact profile..."
        self._update_details_text()
        self.plotter.render()

        try:
            mesh = self.exact_mesh
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

    def measure_exact_profile(self, _checked: bool | None = None) -> None:
        if not self.profile_enabled:
            self.status_message = "Enable the Profile tool before running an exact measurement."
            self._update_details_text()
            self.plotter.render()
            return

        if self.exact_mesh is None:
            if self.mesh_path is None:
                self._clear_profile_result("Exact profile failed.", "No STL file is loaded.")
                self.plotter.render()
                return
            if not self._confirm_exact_load():
                self.status_message = "Kept the preview mesh. Full-resolution loading was not started."
                self._update_details_text()
                self.plotter.render()
                return
            self._pending_exact_profile = True
            self._start_async_load(
                self.mesh_path,
                mode="exact",
                start_message="Loading full-resolution STL for exact profile measurement...",
                error_title="Exact STL Load Failed",
            )
            return

        try:
            self._run_exact_profile_measurement()
        except MeshViewerError as exc:
            self._clear_profile_result("Exact profile failed.", str(exc))
            self.plotter.render()

    def reset_profile(
        self,
        _checked: bool | None = None,
        *,
        update_checkbox: bool = True,
        render: bool = True,
    ) -> None:
        if self.display_mesh_world is None:
            return

        self.profile_start, self.profile_end = default_profile_endpoints(self.display_mesh_world.bounds)
        self._sync_line_widget()

        if update_checkbox:
            self._set_checkbox_value(self.profile_checkbox, self.profile_enabled)

        if self.profile_enabled:
            self.status_message = "Profile reset to the current displayed bounds."
            self.refresh_profile_preview()
        else:
            self.profile_message = "Profile: off. Enable the Profile tool to inspect a section."
            self._update_details_text()

        if render:
            self.plotter.render()

    def reset_clip(
        self,
        _checked: bool | None = None,
        *,
        update_checkbox: bool = True,
        render: bool = True,
    ) -> None:
        if self.render_mesh is None:
            return

        self.clip_normal = self.default_clip_normal
        self.clip_origin = self.default_clip_origin
        self.clip_active = False
        self.apply_render_mesh(self.render_mesh)
        self._sync_plane_widget()

        if update_checkbox:
            self._set_checkbox_value(self.clip_checkbox, self.clip_enabled)

        self.status_message = "Clip reset. Showing the full shell."
        if self.profile_enabled:
            self.refresh_profile_preview()
        else:
            self._update_details_text()

        if render:
            self.plotter.render()

    def restore_all(self, _checked: bool | None = None) -> None:
        if self.render_mesh is None:
            return

        self._set_checkbox_value(self.profile_checkbox, False)
        self.set_profile_enabled(False)
        self.profile_mode = PROFILE_MODE_BOTTOM
        self._set_checkbox_value(self.full_shell_checkbox, False)
        self._set_checkbox_value(self.clip_checkbox, False)
        self.set_clip_enabled(False)
        self._set_checkbox_value(self.wireframe_checkbox, False)
        self.set_wireframe_visible(False)
        self.vertical_exaggeration_combo.blockSignals(True)
        self.vertical_exaggeration_combo.setCurrentText(f"{DEFAULT_VERTICAL_EXAGGERATION}x")
        self.vertical_exaggeration_combo.blockSignals(False)
        self.vertical_exaggeration = DEFAULT_VERTICAL_EXAGGERATION
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(int(DEFAULT_SURFACE_OPACITY * 100))
        self.opacity_slider.blockSignals(False)
        self.on_opacity_changed(int(DEFAULT_SURFACE_OPACITY * 100))
        self.profile_start, self.profile_end = default_profile_endpoints(self.render_mesh.bounds)
        if self.display_mesh_world is not None:
            self.display_mesh_world.copy_from(self.render_mesh, deep=True)
        self._refresh_scene_meshes(rebuild_widgets=True, render=False)
        self._sync_line_widget()
        self.set_isometric_view(render=False)
        self.status_message = "Viewer restored to its default state."
        self.profile_message = "Profile: off. Enable the Profile tool to inspect a section."
        self._update_details_text()
        self.plotter.render()

    def reset_camera(self, _checked: bool | None = None) -> None:
        if self.render_mesh is None:
            return
        self.plotter.reset_camera()
        self.status_message = "Camera reset."
        self._update_details_text()
        self.plotter.render()

    def set_top_view(self, _checked: bool | None = None, *, render: bool = True) -> None:
        if self.render_mesh is None:
            return
        self.plotter.enable_parallel_projection()
        self.plotter.view_xy()
        self.status_message = "Top view."
        self._update_details_text()
        if render:
            self.plotter.render()

    def set_bottom_view(self, _checked: bool | None = None, *, render: bool = True) -> None:
        if self.render_mesh is None:
            return
        self.plotter.enable_parallel_projection()
        self.plotter.view_vector((0.0, 0.0, -1.0), viewup=(0.0, 1.0, 0.0))
        self.status_message = "Bottom view."
        self._update_details_text()
        if render:
            self.plotter.render()

    def set_isometric_view(self, _checked: bool | None = None, *, render: bool = True) -> None:
        if self.render_mesh is None:
            return
        self.plotter.disable_parallel_projection()
        self.plotter.view_isometric()
        self.status_message = "Isometric view."
        self._update_details_text()
        if render:
            self.plotter.render()


class DesktopGuiWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.console_script_path = Path(__file__).resolve().with_name("desktop_console.py")
        self.process: QProcess | None = None
        self.process_mode: str | None = None
        self.stdout_buffer = ""
        self.stderr_buffer = ""
        self.current_inspection: dict[str, object] | None = None
        self.last_result: dict[str, object] | None = None
        self.last_error_message: str | None = None
        self._last_progress_message: str | None = None
        self.selected_input_paths: list[Path] = []
        self.current_sample_step_options: list[dict[str, object]] = []
        self.benchmark_results_by_step: dict[int, dict[str, object]] = {}
        self.pending_benchmark_step: int | None = None

        self.setWindowTitle("Terrain to STL Desktop")
        self.resize(1200, 600)
        self._build_ui()
        self._apply_empty_converter_state()

    def _build_ui(self) -> None:
        tabs = QTabWidget(self)
        self.setCentralWidget(tabs)

        self.viewer_widget = DesktopViewerWidget(self)
        convert_tab = QWidget(self)
        convert_layout = QVBoxLayout(convert_tab)
        convert_layout.setContentsMargins(12, 12, 12, 12)
        convert_layout.setSpacing(10)

        intro_label = QLabel(
            "Use the native desktop workflow to inspect a terrain file set, convert it to STL with streamed progress, "
            "and open the result in the integrated viewer."
        )
        intro_label.setWordWrap(True)
        convert_layout.addWidget(intro_label)

        files_group = QGroupBox("Terrain Files")
        files_layout = QFormLayout(files_group)
        self.input_path_edit = QLineEdit()
        self.input_path_edit.textEdited.connect(self.on_input_path_edited)
        browse_input_button = QPushButton("Browse...")
        browse_input_button.clicked.connect(self.browse_input_paths)
        inspect_button = QPushButton("Inspect")
        inspect_button.clicked.connect(self.start_inspect)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_path_edit, 1)
        input_row.addWidget(browse_input_button)
        input_row.addWidget(inspect_button)
        input_row_widget = QWidget(self)
        input_row_widget.setLayout(input_row)
        files_layout.addRow("Input terrain", input_row_widget)

        self.selected_files_text = QPlainTextEdit()
        self.selected_files_text.setReadOnly(True)
        self.selected_files_text.setMinimumHeight(90)
        self.selected_files_text.setPlainText("No terrain files selected yet.")
        files_layout.addRow("Selected files", self.selected_files_text)

        self.association_status_label = QLabel("Choose one terrain HDF with its raster files, or one DEM GeoTIFF.")
        self.association_status_label.setWordWrap(True)
        files_layout.addRow("Association status", self.association_status_label)

        self.output_path_edit = QLineEdit()
        browse_output_button = QPushButton("Browse...")
        browse_output_button.clicked.connect(self.browse_output_path)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_path_edit, 1)
        output_row.addWidget(browse_output_button)
        output_row_widget = QWidget(self)
        output_row_widget.setLayout(output_row)
        files_layout.addRow("Output STL", output_row_widget)
        convert_layout.addWidget(files_group)

        metadata_group = QGroupBox("Inspection")
        metadata_layout = QFormLayout(metadata_group)
        self.terrain_kind_label = QLabel("-")
        self.raster_name_label = QLabel("-")
        self.raster_size_label = QLabel("-")
        self.terrain_max_label = QLabel("-")
        self.stitch_counts_label = QLabel("-")
        self.sample_step_rule_label = QLabel("-")
        metadata_layout.addRow("Terrain type", self.terrain_kind_label)
        metadata_layout.addRow("Resolved raster", self.raster_name_label)
        metadata_layout.addRow("Raster size", self.raster_size_label)
        metadata_layout.addRow("Terrain max elevation", self.terrain_max_label)
        metadata_layout.addRow("Stitch summary", self.stitch_counts_label)
        metadata_layout.addRow("Sample step rule", self.sample_step_rule_label)
        convert_layout.addWidget(metadata_group)

        settings_group = QGroupBox("Conversion Settings")
        settings_layout = QFormLayout(settings_group)
        self.top_elevation_spin = QDoubleSpinBox()
        self.top_elevation_spin.setDecimals(6)
        self.top_elevation_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.top_elevation_spin.setSingleStep(1.0)
        settings_layout.addRow("Top elevation", self.top_elevation_spin)

        self.sample_step_combo = QComboBox()
        self.sample_step_combo.setEditable(False)
        for step in (1, 2, 4, 8, 16, 32):
            self.sample_step_combo.addItem(str(step), step)
        self.sample_step_combo.setCurrentIndex(0)
        self.sample_step_combo.currentIndexChanged.connect(self.on_sample_step_changed)
        settings_layout.addRow("Sample step", self.sample_step_combo)
        self.sample_step_size_label = QLabel("Inspect the terrain to estimate STL size by sample step.")
        self.sample_step_size_label.setWordWrap(True)
        settings_layout.addRow("Size estimate", self.sample_step_size_label)
        self.sample_step_time_label = QLabel("Inspect the terrain to estimate conversion time.")
        self.sample_step_time_label.setWordWrap(True)
        settings_layout.addRow("Time estimate", self.sample_step_time_label)
        convert_layout.addWidget(settings_group)

        action_layout = QGridLayout()
        action_layout.setHorizontalSpacing(8)
        action_layout.setVerticalSpacing(8)
        self.convert_button = QPushButton("Convert Terrain")
        self.convert_button.clicked.connect(self.start_convert)
        self.clear_converter_button = QPushButton("Clear")
        self.clear_converter_button.clicked.connect(self.clear_converter)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_process)
        self.cancel_button.setEnabled(False)
        self.open_result_button = QPushButton("Open Result In Viewer")
        self.open_result_button.clicked.connect(self.open_result_in_viewer)
        self.open_result_button.setEnabled(False)
        self.download_result_button = QPushButton("Download STL")
        self.download_result_button.clicked.connect(self.download_result_stl)
        self.download_result_button.setEnabled(False)
        action_layout.addWidget(self.convert_button, 0, 0)
        action_layout.addWidget(self.clear_converter_button, 0, 1)
        action_layout.addWidget(self.cancel_button, 0, 2)
        action_layout.addWidget(self.open_result_button, 1, 0, 1, 2)
        action_layout.addWidget(self.download_result_button, 1, 2)
        convert_layout.addLayout(action_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        convert_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Choose terrain files to begin.")
        self.status_label.setWordWrap(True)
        convert_layout.addWidget(self.status_label)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(160)
        convert_layout.addWidget(self.log_text, 1)

        tabs.addTab(convert_tab, "Convert")
        tabs.addTab(self.viewer_widget, "Viewer")
        self.tabs = tabs

    def append_log(self, message: str) -> None:
        if not message:
            return
        self.log_text.appendPlainText(message.rstrip())

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def on_input_path_edited(self, _text: str) -> None:
        self.selected_input_paths = []
        self._clear_inspection_state()
        self._refresh_selection_display()

    def _current_selection(self):
        if self.selected_input_paths:
            selection = resolve_desktop_terrain_selection(self.selected_input_paths)
            manual_text = self.input_path_edit.text().strip()
            if manual_text:
                manual_path = normalize_desktop_path(manual_text)
                if selection.primary_path is None or manual_path != selection.primary_path:
                    return resolve_desktop_terrain_selection([manual_path])
            return selection

        input_path = self.input_path_edit.text().strip()
        if not input_path:
            return resolve_desktop_terrain_selection([])
        return resolve_desktop_terrain_selection([input_path])

    def _refresh_selection_display(self) -> None:
        selection = self._current_selection()
        entries, warning = build_associated_file_entries(selection, self.current_inspection)
        if not entries:
            self.selected_files_text.setPlainText("No terrain files selected yet.")
        else:
            lines = []
            for entry in entries:
                source_label = "selected" if entry.selected else "found automatically"
                lines.append(f"{entry.path} - {entry.role} ({source_label})")
            self.selected_files_text.setPlainText("\n".join(lines))

        if selection.error:
            self.association_status_label.setText(selection.error)
        elif warning:
            self.association_status_label.setText(warning)
        elif selection.primary_path is not None:
            self.association_status_label.setText(
                f"Primary input: {selection.primary_path.name}. Inspect the terrain to confirm the resolved raster."
            )
        else:
            self.association_status_label.setText(
                "Choose one terrain HDF with its raster files, or one DEM GeoTIFF."
            )

    def _sample_step_option_for_value(self, sample_step: int) -> dict[str, object] | None:
        for option in self.current_sample_step_options:
            if int(option.get("value", -1)) == int(sample_step):
                return option
        return None

    def _format_sample_step_option_label(self, option: dict[str, object]) -> str:
        value = int(option["value"])
        estimated_size_bytes = option.get("estimated_size_bytes")
        if estimated_size_bytes is None:
            return str(value)

        return f"{value} - up to {format_bytes(int(estimated_size_bytes))}"

    def _selected_sample_step(self) -> int:
        current_data = self.sample_step_combo.currentData()
        if current_data is not None:
            value = int(current_data)
        else:
            text = self.sample_step_combo.currentText().strip()
            if not text:
                raise ValueError("Choose a raster sample step.")
            value = int(text)
        if value < 1:
            raise ValueError("The sample step must be at least 1.")
        return value

    def _update_sample_step_estimate_labels(self) -> None:
        if self.current_inspection is None:
            self.sample_step_size_label.setText("Inspect the terrain to estimate STL size by sample step.")
            self.sample_step_time_label.setText("Inspect the terrain to estimate conversion time.")
            return

        try:
            sample_step = self._selected_sample_step()
        except ValueError:
            self.sample_step_size_label.setText("Choose a raster sample step.")
            self.sample_step_time_label.setText("Choose a raster sample step.")
            return

        option = self._sample_step_option_for_value(sample_step)
        if option is None or option.get("estimated_size_bytes") is None:
            self.sample_step_size_label.setText("Estimated STL size is unavailable for this sample step.")
        else:
            self.sample_step_size_label.setText(
                f"Sample step {sample_step} STL size estimate: up to {format_bytes(int(option['estimated_size_bytes']))}."
            )

        benchmark = self.benchmark_results_by_step.get(sample_step)
        if benchmark is not None and benchmark.get("estimated_duration_seconds") is not None:
            self.sample_step_time_label.setText(
                f"Sample step {sample_step} estimated time: about {format_duration(float(benchmark['estimated_duration_seconds']))}."
            )
            return

        if self.process_mode == "benchmark" and self.pending_benchmark_step == sample_step:
            self.sample_step_time_label.setText(f"Estimating time for sample step {sample_step}...")
            return

        self.sample_step_time_label.setText(f"Sample step {sample_step} estimated time: pending benchmark.")

    def _maybe_start_pending_benchmark(self) -> bool:
        if self.process is not None or self.current_inspection is None:
            return False
        if self.pending_benchmark_step is None:
            return False
        if self.pending_benchmark_step in self.benchmark_results_by_step:
            self.pending_benchmark_step = None
            self._update_sample_step_estimate_labels()
            return False

        selection = self._current_selection()
        if selection.primary_path is None:
            self.pending_benchmark_step = None
            return False

        sample_step = int(self.pending_benchmark_step)
        self.pending_benchmark_step = sample_step
        self._start_process(
            "benchmark",
            [
                "benchmark",
                "--input",
                str(selection.primary_path),
                "--sample-step",
                str(sample_step),
                "--json-stream",
            ],
        )
        return True

    def _queue_selected_benchmark(self) -> None:
        if self.current_inspection is None:
            return
        try:
            sample_step = self._selected_sample_step()
        except ValueError:
            return

        if sample_step in self.benchmark_results_by_step:
            self.pending_benchmark_step = None
            self._update_sample_step_estimate_labels()
            return

        self.pending_benchmark_step = int(sample_step)
        self._update_sample_step_estimate_labels()
        self._maybe_start_pending_benchmark()

    def on_sample_step_changed(self, _index: int) -> None:
        self._update_sample_step_estimate_labels()
        self._queue_selected_benchmark()

    def _clear_inspection_state(self) -> None:
        self.current_inspection = None
        self.current_sample_step_options = []
        self.benchmark_results_by_step = {}
        self.pending_benchmark_step = None
        self.last_result = None
        self.open_result_button.setEnabled(False)
        self.download_result_button.setEnabled(False)
        self.terrain_kind_label.setText("-")
        self.raster_name_label.setText("-")
        self.raster_size_label.setText("-")
        self.terrain_max_label.setText("-")
        self.stitch_counts_label.setText("-")
        self.sample_step_rule_label.setText("-")
        self.sample_step_combo.blockSignals(True)
        self.sample_step_combo.clear()
        for step in (1, 2, 4, 8, 16, 32):
            self.sample_step_combo.addItem(str(step), step)
        self.sample_step_combo.setCurrentIndex(0)
        self.sample_step_combo.blockSignals(False)
        self.sample_step_size_label.setText("Inspect the terrain to estimate STL size by sample step.")
        self.sample_step_time_label.setText("Inspect the terrain to estimate conversion time.")

    def _apply_empty_converter_state(self) -> None:
        self.selected_input_paths = []
        self.input_path_edit.clear()
        self.output_path_edit.clear()
        self.log_text.clear()
        self.last_error_message = None
        self._last_progress_message = None
        self.top_elevation_spin.setValue(0.0)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._clear_inspection_state()
        self._refresh_selection_display()
        self.set_status("Choose terrain files to begin.")

    def clear_converter(self, _checked: bool | None = None) -> None:
        if self.process is not None:
            return
        self._apply_empty_converter_state()

    def _current_result_path(self) -> Path | None:
        if self.last_result is None:
            return None

        raw_path = str(self.last_result.get("output_path", "")).strip()
        if not raw_path:
            return None

        candidate = Path(raw_path)
        if not candidate.exists():
            return None
        return candidate

    def download_result_stl(self, _checked: bool | None = None) -> None:
        result_path = self._current_result_path()
        if result_path is None:
            QMessageBox.warning(self, "Download STL", "There is no converted STL available to save.")
            return

        destination = choose_stl_copy_destination(
            self,
            source_path=result_path,
            title="Save Converted STL Copy",
        )
        if destination is None:
            return

        try:
            if destination.resolve() == result_path.resolve():
                self.set_status(f"Converted STL is already available at {destination}.")
                self.append_log(f"Converted STL is already available at {destination}")
                return
            shutil.copy2(result_path, destination)
        except OSError as exc:
            QMessageBox.critical(self, "Download STL Failed", f"Could not save the STL copy.\n\n{exc}")
            return

        self.set_status(f"Saved STL copy to {destination}.")
        self.append_log(f"Saved STL copy to {destination}")

    def browse_input_paths(self, _checked: bool | None = None) -> None:
        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Terrain Files",
            str(Path.home()),
            "Terrain Files (*.hdf *.vrt *.tif *.tiff)",
        )
        if not selected:
            return

        self.selected_input_paths = [normalize_desktop_path(path) for path in selected]
        self._clear_inspection_state()
        selection = self._current_selection()
        if selection.primary_path is not None:
            self.input_path_edit.setText(str(selection.primary_path))
        elif selection.selected_paths:
            self.input_path_edit.setText(str(selection.selected_paths[0]))
        else:
            self.input_path_edit.clear()

        self._refresh_selection_display()
        if selection.error:
            self.set_status(selection.error)
            self.append_log(selection.error)
            return

        self.set_status(f"{len(selection.selected_paths)} terrain file(s) selected. Inspect to continue.")
        self.start_inspect()

    def browse_output_path(self, _checked: bool | None = None) -> None:
        initial = self.output_path_edit.text().strip() or str(Path.home() / "terrain.stl")
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Choose Output STL",
            initial,
            "STL Files (*.stl)",
        )
        if not selected:
            return

        path = Path(selected)
        if path.suffix.lower() != ".stl":
            path = path.with_suffix(".stl")
        self.output_path_edit.setText(str(path))

    def _set_busy(self, busy: bool) -> None:
        self.cancel_button.setEnabled(busy)
        self.convert_button.setEnabled(not busy)
        self.clear_converter_button.setEnabled(not busy)
        self.open_result_button.setEnabled((not busy) and self._current_result_path() is not None)
        self.download_result_button.setEnabled((not busy) and self._current_result_path() is not None)

    def _start_process(self, mode: str, arguments: list[str]) -> None:
        if self.process is not None:
            QMessageBox.warning(self, "Busy", "Another desktop operation is already running.")
            return

        self.process = QProcess(self)
        self.process_mode = mode
        self.stdout_buffer = ""
        self.stderr_buffer = ""
        self.last_error_message = None
        self._last_progress_message = None
        self.process.readyReadStandardOutput.connect(self.on_process_stdout)
        self.process.readyReadStandardError.connect(self.on_process_stderr)
        self.process.finished.connect(self.on_process_finished)
        self.process.setProgram(sys.executable)
        self.process.setArguments([str(self.console_script_path)] + arguments)
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parent))
        self._set_busy(True)
        if mode == "inspect":
            self.progress_bar.setRange(0, 0)
            self.set_status("Inspecting terrain...")
        elif mode == "benchmark":
            self.progress_bar.setRange(0, 0)
            if self.pending_benchmark_step is not None:
                self.set_status(f"Estimating time for sample step {self.pending_benchmark_step}...")
            else:
                self.set_status("Estimating conversion time...")
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.set_status("Starting conversion...")
            self.open_result_button.setEnabled(False)
            self.download_result_button.setEnabled(False)
        self.process.start()

    def start_inspect(self, _checked: bool | None = None) -> None:
        selection = self._current_selection()
        self._refresh_selection_display()
        if selection.error:
            QMessageBox.warning(self, "Invalid Terrain Selection", selection.error)
            return

        if selection.primary_path is None:
            QMessageBox.warning(self, "Missing Input", "Select a terrain file first.")
            return

        self._clear_inspection_state()
        self.input_path_edit.setText(str(selection.primary_path))
        self.append_log(f"Inspecting {selection.primary_path}")
        self._start_process("inspect", ["inspect", "--input", str(selection.primary_path), "--json-stream"])

    def start_convert(self, _checked: bool | None = None) -> None:
        selection = self._current_selection()
        self._refresh_selection_display()
        if selection.error:
            QMessageBox.warning(self, "Invalid Terrain Selection", selection.error)
            return

        if selection.primary_path is None:
            QMessageBox.warning(self, "Missing Input", "Select a terrain file first.")
            return

        if self.current_inspection is None:
            QMessageBox.warning(self, "Inspect Terrain", "Inspect the selected terrain before converting it to STL.")
            return

        try:
            sample_step = self._selected_sample_step()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Sample Step", str(exc))
            return

        arguments = [
            "convert",
            "--input",
            str(selection.primary_path),
            "--top-elevation",
            f"{self.top_elevation_spin.value():.6f}",
            "--sample-step",
            str(sample_step),
            "--json-stream",
        ]
        output_path = self.output_path_edit.text().strip()
        if output_path:
            arguments.extend(["--output", output_path])

        self.input_path_edit.setText(str(selection.primary_path))
        self.append_log(f"Starting conversion for {selection.primary_path}")
        self._start_process("convert", arguments)

    def cancel_process(self, _checked: bool | None = None) -> None:
        if self.process is None:
            return
        self.append_log("Cancelling active operation...")
        self.process.kill()

    def _consume_stdout_lines(self, chunk: str) -> None:
        self.stdout_buffer += chunk
        while "\n" in self.stdout_buffer:
            raw_line, self.stdout_buffer = self.stdout_buffer.split("\n", 1)
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                self.append_log(line)
                continue
            self.handle_process_record(record)

    def on_process_stdout(self) -> None:
        if self.process is None:
            return
        chunk = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._consume_stdout_lines(chunk)

    def on_process_stderr(self) -> None:
        if self.process is None:
            return
        chunk = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        self.stderr_buffer += chunk
        for line in chunk.splitlines():
            if line.strip():
                self.append_log(line)

    def handle_process_record(self, record: dict[str, object]) -> None:
        record_type = str(record.get("type", ""))
        if record_type == "inspect":
            inspection = record.get("inspection")
            if isinstance(inspection, dict):
                self.apply_inspection(inspection)
            return

        if record_type == "benchmark":
            benchmark = record.get("benchmark")
            if isinstance(benchmark, dict):
                self.apply_benchmark_result(benchmark)
            return

        if record_type == "progress":
            percent = int(record.get("percent", 0))
            message = str(record.get("message", ""))
            self.progress_bar.setValue(percent)
            self.set_status(message)
            if message and message != self._last_progress_message:
                self.append_log(message)
                self._last_progress_message = message
            return

        if record_type == "result":
            result = record.get("result")
            if isinstance(result, dict):
                self.apply_conversion_result(result)
            return

        if record_type == "error":
            message = str(record.get("message", "Desktop operation failed."))
            self.last_error_message = message
            self.append_log(message)
            self.set_status(message)

    def on_process_finished(self, exit_code: int, exit_status) -> None:
        _ = exit_status
        mode = self.process_mode
        self.process = None
        self.process_mode = None
        self._set_busy(False)

        if mode in {"inspect", "benchmark"}:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)

        if exit_code == 0:
            if mode == "inspect":
                if self._maybe_start_pending_benchmark():
                    return
                self.set_status("Inspection complete.")
            elif mode == "benchmark":
                if self._maybe_start_pending_benchmark():
                    return
                self._update_sample_step_estimate_labels()
                self.set_status("Desktop estimates updated.")
            elif mode == "convert" and self.last_result is None:
                self.set_status("Conversion finished.")
            return

        message = self.last_error_message or "The desktop operation failed."
        if mode == "benchmark":
            self.set_status("Time estimate unavailable for the selected sample step.")
            self.sample_step_time_label.setText("Time estimate unavailable for the selected sample step.")
            self.append_log(message)
            return
        self.set_status(message)
        QMessageBox.critical(self, "Desktop Operation Failed", message)

    def apply_inspection(self, inspection: dict[str, object]) -> None:
        self.current_inspection = inspection
        self.current_sample_step_options = [
            option for option in inspection.get("sample_step_options", [])
            if isinstance(option, dict)
        ]
        self.benchmark_results_by_step = {}
        self.pending_benchmark_step = None
        self.last_result = None
        self.open_result_button.setEnabled(False)

        self.terrain_kind_label.setText(str(inspection["source_kind"]))
        self.raster_name_label.setText(str(inspection["resolved_raster_path"]))
        self.raster_size_label.setText(
            f"{inspection['raster_width']} columns x {inspection['raster_height']} rows"
        )
        self.terrain_max_label.setText(f"{float(inspection['terrain_max_elevation']):.6f}")
        self.stitch_counts_label.setText(
            f"{inspection['stitch_point_count']} points, "
            f"{inspection['stitch_triangle_count']} triangles, "
            f"{inspection['stitch_bridge_triangle_count']} bridge triangles"
        )
        self.sample_step_rule_label.setText(
            "Desktop sample steps use the fixed preset values 1, 2, 4, 8, 16, and 32."
        )

        self.top_elevation_spin.setValue(float(inspection["terrain_max_elevation"]))
        self.output_path_edit.setText(str(inspection["default_output_path"]))
        self.sample_step_combo.blockSignals(True)
        self.sample_step_combo.clear()
        options = self.current_sample_step_options or [
            {"value": step, "estimated_size_bytes": None}
            for step in inspection["suggested_sample_steps"]
        ]
        selected_index = 0
        default_sample_step = int(inspection["default_sample_step"])
        for index, option in enumerate(options):
            self.sample_step_combo.addItem(self._format_sample_step_option_label(option), int(option["value"]))
            if int(option["value"]) == default_sample_step:
                selected_index = index
        self.sample_step_combo.setCurrentIndex(selected_index)
        self.sample_step_combo.blockSignals(False)
        self._refresh_selection_display()
        self._update_sample_step_estimate_labels()
        self.pending_benchmark_step = self._selected_sample_step()

        self.append_log(
            "Inspection complete: "
            f"terrain max {float(inspection['terrain_max_elevation']):.6f}, "
            f"raster {inspection['resolved_raster_name']}"
        )

    def apply_benchmark_result(self, benchmark: dict[str, object]) -> None:
        sample_step = int(benchmark["sample_step"])
        self.benchmark_results_by_step[sample_step] = benchmark
        if self.pending_benchmark_step == sample_step:
            self.pending_benchmark_step = None
        self._update_sample_step_estimate_labels()
        self.append_log(
            f"Estimated time for sample step {sample_step}: "
            f"{format_duration(float(benchmark['estimated_duration_seconds']))}"
        )

    def apply_conversion_result(self, result: dict[str, object]) -> None:
        self.last_result = result
        self.progress_bar.setValue(100)
        self.output_path_edit.setText(str(result["output_path"]))
        has_result_path = self._current_result_path() is not None
        self.open_result_button.setEnabled(has_result_path)
        self.download_result_button.setEnabled(has_result_path)
        self.set_status(f"Finished writing {result['output_filename']}.")
        self.append_log(f"Output STL: {result['output_path']}")
        self.append_log(f"Total STL triangles: {result['triangle_count']}")
        self.append_log(f"Boundary wall triangles: {result['wall_triangle_count']}")
        self.append_log(f"STL size: {format_bytes(int(result['stl_size_bytes']))}")

    def open_result_in_viewer(self, _checked: bool | None = None) -> None:
        if self.last_result is None:
            return
        self.tabs.setCurrentWidget(self.viewer_widget)
        self.viewer_widget.load_stl(Path(str(self.last_result["output_path"])))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open the Terrain to STL native desktop GUI.")
    parser.add_argument(
        "stl_path",
        nargs="?",
        help="Optional STL file to open immediately in the viewer tab.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(sys.argv[1:] if argv is None else argv)
    app = QApplication(sys.argv if argv is None else ["desktop_gui.py", *argv])
    app.setApplicationName("Terrain to STL Desktop")
    window = DesktopGuiWindow()
    if args.stl_path:
        window.tabs.setCurrentWidget(window.viewer_widget)
        window.viewer_widget.load_stl(Path(args.stl_path).expanduser())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
