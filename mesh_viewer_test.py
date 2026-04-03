from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mesh_viewer import (
    BinaryStlChunk,
    build_preview_mesh_from_binary_stl,
    inspect_binary_stl,
    load_mesh_state,
)


def write_binary_stl(
    path: Path,
    triangles: np.ndarray,
    *,
    triangle_count_override: int | None = None,
) -> None:
    triangle_array = np.asarray(triangles, dtype=np.float32).reshape(-1, 3, 3)
    records = np.zeros(triangle_array.shape[0], dtype=BinaryStlChunk)
    records["vertices"] = triangle_array

    with path.open("wb") as file_handle:
        file_handle.write(b"Terrain to STL test".ljust(80, b"\0"))
        file_handle.write(struct.pack("<I", triangle_array.shape[0] if triangle_count_override is None else triangle_count_override))
        records.tofile(file_handle)


class MeshViewerLoadTests(unittest.TestCase):
    def test_inspect_binary_stl_reads_valid_triangle_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stl_path = Path(temp_dir) / "valid.stl"
            write_binary_stl(
                stl_path,
                np.array(
                    [
                        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                        [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
                    ],
                    dtype=np.float32,
                ),
            )

            metadata = inspect_binary_stl(stl_path)

            self.assertIsNotNone(metadata)
            self.assertEqual(metadata.triangle_count, 2)
            self.assertEqual(metadata.file_size_bytes, 84 + (2 * 50))

    def test_inspect_binary_stl_rejects_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stl_path = Path(temp_dir) / "invalid-size.stl"
            write_binary_stl(
                stl_path,
                np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32),
                triangle_count_override=2,
            )

            self.assertIsNone(inspect_binary_stl(stl_path))

    def test_build_preview_mesh_from_binary_stl_returns_displayable_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stl_path = Path(temp_dir) / "preview.stl"
            write_binary_stl(
                stl_path,
                np.array(
                    [
                        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                        [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    ],
                    dtype=np.float32,
                ),
            )

            metadata = inspect_binary_stl(stl_path)
            preview_mesh = build_preview_mesh_from_binary_stl(stl_path, metadata)

            self.assertGreater(preview_mesh.n_cells, 0)
            self.assertGreater(preview_mesh.n_points, 0)
            self.assertIn("Elevation", preview_mesh.point_data)

    def test_load_mesh_state_uses_streamed_preview_for_dense_binary_stl(self) -> None:
        triangle_count = 320_000
        base_triangles = np.array(
            [
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
                [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
                [[1, 0, 1], [1, 1, 1], [0, 1, 1]],
            ],
            dtype=np.float32,
        )
        repeats = triangle_count // base_triangles.shape[0]
        dense_triangles = np.tile(base_triangles, (repeats, 1, 1))

        with tempfile.TemporaryDirectory() as temp_dir:
            stl_path = Path(temp_dir) / "dense.stl"
            write_binary_stl(stl_path, dense_triangles)

            load_result = load_mesh_state(stl_path)

            self.assertEqual(load_result.render_mode, "Preview")
            self.assertIsNone(load_result.exact_mesh)
            self.assertEqual(load_result.exact_cell_count, triangle_count)
            self.assertLess(load_result.render_mesh.n_cells, triangle_count)
            self.assertFalse(load_result.exact_load_warning_required)


if __name__ == "__main__":
    unittest.main()
