from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock

QT_IMPORT_ERROR: Exception | None = None

try:
    from desktop_gui import DesktopGuiWindow, DesktopViewerWidget
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    if exc.name != "PySide6":
        raise
    QT_IMPORT_ERROR = exc
    DesktopGuiWindow = None
    DesktopViewerWidget = None


@unittest.skipIf(QT_IMPORT_ERROR is not None, f"PySide6 is not installed: {QT_IMPORT_ERROR}")
class DesktopGuiWindowTests(unittest.TestCase):
    def test_open_result_in_viewer_switches_tabs_before_loading(self) -> None:
        window = DesktopGuiWindow.__new__(DesktopGuiWindow)
        window.last_result = {"output_path": r"D:\terrain\result.stl"}
        window.viewer_widget = Mock()
        window.tabs = Mock()

        events: list[object] = []
        window.tabs.setCurrentWidget.side_effect = lambda widget: events.append(("tab", widget))
        window.viewer_widget.load_stl.side_effect = lambda path: events.append(("load", path))

        DesktopGuiWindow.open_result_in_viewer(window)

        self.assertEqual(events[0], ("tab", window.viewer_widget))
        self.assertEqual(events[1], ("load", Path(r"D:\terrain\result.stl")))


@unittest.skipIf(QT_IMPORT_ERROR is not None, f"PySide6 is not installed: {QT_IMPORT_ERROR}")
class DesktopViewerWidgetTests(unittest.TestCase):
    def _make_measure_widget(self) -> DesktopViewerWidget:
        widget = DesktopViewerWidget.__new__(DesktopViewerWidget)
        widget.profile_enabled = True
        widget.mesh_path = Path(r"D:\terrain\preview.stl")
        widget.exact_mesh = None
        widget.exact_load_warning_required = True
        widget._confirm_exact_load = Mock()
        widget._start_async_load = Mock()
        widget._clear_profile_result = Mock()
        widget._update_details_text = Mock()
        widget.plotter = Mock()
        widget.status_message = ""
        widget._pending_exact_profile = False
        return widget

    def test_measure_exact_profile_keeps_preview_when_warning_declined(self) -> None:
        widget = self._make_measure_widget()
        widget._confirm_exact_load.return_value = False

        DesktopViewerWidget.measure_exact_profile(widget)

        widget._confirm_exact_load.assert_called_once_with()
        widget._start_async_load.assert_not_called()
        self.assertEqual(widget.status_message, "Kept the preview mesh. Full-resolution loading was not started.")
        widget._update_details_text.assert_called_once_with()
        widget.plotter.render.assert_called_once_with()

    def test_measure_exact_profile_starts_async_full_load_after_confirmation(self) -> None:
        widget = self._make_measure_widget()
        widget._confirm_exact_load.return_value = True

        DesktopViewerWidget.measure_exact_profile(widget)

        self.assertTrue(widget._pending_exact_profile)
        widget._start_async_load.assert_called_once_with(
            Path(r"D:\terrain\preview.stl"),
            mode="exact",
            start_message="Loading full-resolution STL for exact profile measurement...",
            error_title="Exact STL Load Failed",
        )


if __name__ == "__main__":
    unittest.main()
