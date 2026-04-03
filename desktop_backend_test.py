from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from desktop_backend import (
    benchmark_source,
    compute_raster_max_elevation_parallel,
    convert_source,
    inspect_source,
    read_raster_metadata,
)
from terrain_to_stl import load_raster_surface


REPO_ROOT = Path(__file__).resolve().parent
EXAMPLE_HDF = REPO_ROOT / "webapp" / "public" / "example" / "example.hdf"
EXAMPLE_TIF = REPO_ROOT / "webapp" / "public" / "example" / "example.tif"


class DesktopBackendTests(unittest.TestCase):
    def test_inspect_source_returns_preset_sample_step_options(self) -> None:
        inspection = inspect_source(EXAMPLE_HDF)

        sample_step_options = inspection["sample_step_options"]
        self.assertEqual([option["value"] for option in sample_step_options], [1, 2, 4, 8, 16, 32])
        step_eight = next(option for option in sample_step_options if option["value"] == 8)
        self.assertGreater(int(step_eight["estimated_size_bytes"]), 0)
        self.assertEqual(step_eight["size_estimate_kind"], "upper-bound")
        self.assertIsNone(step_eight["estimated_duration_seconds"])
        self.assertEqual(step_eight["duration_estimate_kind"], "pending")

    def test_parallel_dem_max_scan_matches_full_surface_max(self) -> None:
        metadata = read_raster_metadata(EXAMPLE_TIF)
        exact_max = compute_raster_max_elevation_parallel(metadata)
        full_surface = load_raster_surface(EXAMPLE_TIF)

        self.assertAlmostEqual(exact_max, full_surface.max_elevation, places=5)

    def test_benchmark_source_returns_duration_estimate(self) -> None:
        benchmark = benchmark_source(EXAMPLE_HDF, 8)

        self.assertEqual(benchmark["sample_step"], 8)
        self.assertGreater(float(benchmark["estimated_duration_seconds"]), 0.0)
        self.assertEqual(benchmark["duration_estimate_kind"], "benchmark")
        self.assertGreater(float(benchmark["benchmark_elapsed_seconds"]), 0.0)

    def test_convert_source_supports_sparse_desktop_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "example_step_8.stl"
            result = convert_source(
                EXAMPLE_HDF,
                100.0,
                8,
                output_path=output_path,
            )

            self.assertEqual(Path(result["output_path"]), output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertGreater(int(result["triangle_count"]), 0)


if __name__ == "__main__":
    unittest.main()
