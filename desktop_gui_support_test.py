from __future__ import annotations

import unittest
from pathlib import Path

from desktop_gui_support import (
    DEFAULT_VERTICAL_EXAGGERATION,
    build_associated_file_entries,
    resolve_desktop_terrain_selection,
    scene_normal_from_world,
    scene_point_from_world,
    world_normal_from_scene,
    world_point_from_scene,
)


class TerrainSelectionTests(unittest.TestCase):
    def test_hdf_is_primary_when_associated_rasters_are_selected(self) -> None:
        selection = resolve_desktop_terrain_selection(
            [
                r"D:\terrain\Terrain.hdf",
                r"D:\terrain\Terrain.vrt",
                r"D:\terrain\Terrain.tif",
            ]
        )

        self.assertIsNone(selection.error)
        self.assertEqual(selection.primary_kind, "hdf")
        self.assertEqual(selection.primary_path, Path(r"D:\terrain\Terrain.hdf"))

    def test_single_dem_is_primary_when_no_hdf_is_selected(self) -> None:
        selection = resolve_desktop_terrain_selection([r"D:\terrain\surface.tif"])

        self.assertIsNone(selection.error)
        self.assertEqual(selection.primary_kind, "dem")
        self.assertEqual(selection.primary_path, Path(r"D:\terrain\surface.tif"))

    def test_multiple_hdfs_are_rejected(self) -> None:
        selection = resolve_desktop_terrain_selection(
            [
                r"D:\terrain\A.hdf",
                r"D:\terrain\B.hdf",
            ]
        )

        self.assertIsNone(selection.primary_path)
        self.assertIn("only one terrain HDF", selection.error or "")

    def test_multiple_dems_without_hdf_are_rejected(self) -> None:
        selection = resolve_desktop_terrain_selection(
            [
                r"D:\terrain\A.tif",
                r"D:\terrain\B.tiff",
            ]
        )

        self.assertIsNone(selection.primary_path)
        self.assertIn("exactly one DEM", selection.error or "")

    def test_vrt_only_selection_is_rejected(self) -> None:
        selection = resolve_desktop_terrain_selection([r"D:\terrain\Terrain.vrt"])

        self.assertIsNone(selection.primary_path)
        self.assertIn(".vrt file must be selected with its terrain HDF", selection.error or "")

    def test_extra_selected_rasters_are_marked_unused_after_inspection(self) -> None:
        selection = resolve_desktop_terrain_selection(
            [
                r"D:\terrain\Terrain.hdf",
                r"D:\terrain\Terrain.vrt",
                r"D:\terrain\Extra.tif",
            ]
        )
        entries, warning = build_associated_file_entries(
            selection,
            {
                "resolved_raster_path": r"D:\terrain\Terrain.vrt",
            },
        )

        self.assertEqual(entries[0].role, "Primary terrain")
        self.assertEqual(entries[1].role, "Resolved raster")
        self.assertEqual(entries[2].role, "Selected but unused")
        self.assertIn("resolved sibling raster only", warning or "")


class VerticalExaggerationTransformTests(unittest.TestCase):
    def test_point_round_trip_preserves_world_coordinates(self) -> None:
        world_point = (10.0, 20.0, 55.0)
        center_z = 40.0

        scene_point = scene_point_from_world(world_point, center_z, 5)
        recovered_point = world_point_from_scene(scene_point, center_z, 5)

        self.assertEqual(recovered_point, world_point)

    def test_default_vertical_exaggeration_is_identity_for_point_conversion(self) -> None:
        world_point = (2.0, 4.0, 8.0)

        self.assertEqual(
            scene_point_from_world(world_point, 3.0, DEFAULT_VERTICAL_EXAGGERATION),
            (2.0, 4.0, 18.0),
        )

    def test_normal_round_trip_preserves_orientation(self) -> None:
        world_normal = (0.3, 0.4, 0.8660254038)

        scene_normal = scene_normal_from_world(world_normal, 10)
        recovered_normal = world_normal_from_scene(scene_normal, 10)

        self.assertAlmostEqual(recovered_normal[0], world_normal[0], places=6)
        self.assertAlmostEqual(recovered_normal[1], world_normal[1], places=6)
        self.assertAlmostEqual(recovered_normal[2], world_normal[2], places=6)


if __name__ == "__main__":
    unittest.main()
