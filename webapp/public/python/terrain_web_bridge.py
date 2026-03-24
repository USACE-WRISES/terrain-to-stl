from __future__ import annotations
import json
from pathlib import Path
from typing import Callable

import numpy as np
from rasterio.transform import Affine

from terrain_to_stl import (
    CoordinateLookup,
    RasterSurface,
    TerrainConversionError,
    empty_stitch_arrays,
    load_hdf_max_elevation,
    load_raster_surface,
    load_stitch_arrays,
    resolve_raster_path,
    validate_stitch_mesh,
    write_shell_stl,
)

ProgressCallback = Callable[[str, int, int, str], None]


def _terrain_max_elevation(hdf_path: Path, surface_max: float) -> float:
    hdf_max = load_hdf_max_elevation(hdf_path)
    return hdf_max if hdf_max is not None else float(surface_max)


def _load_browser_surface(session_dir: str, surface_meta_name: str) -> RasterSurface:
    session_path = Path(session_dir)
    meta_path = session_path / surface_meta_name
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    width = int(meta["width"])
    height = int(meta["height"])
    transform_values = meta["transform"]
    if len(transform_values) != 6:
        raise TerrainConversionError("The browser raster surface metadata does not contain a 6-value affine transform.")

    elevations_path = session_path / str(meta["elevations_file"])
    valid_mask_path = session_path / str(meta["valid_mask_file"])
    elevations = np.frombuffer(elevations_path.read_bytes(), dtype=np.float32).reshape((height, width))
    valid_mask = np.frombuffer(valid_mask_path.read_bytes(), dtype=np.uint8).reshape((height, width)).astype(bool, copy=False)

    if elevations.shape != (height, width) or valid_mask.shape != (height, width):
        raise TerrainConversionError("The browser raster surface arrays do not match the declared raster dimensions.")
    if not np.any(valid_mask):
        raise TerrainConversionError("The browser raster surface does not contain any valid elevation cells.")

    return RasterSurface(
        path=Path(str(meta["resolved_raster_name"])),
        width=width,
        height=height,
        # Browser raster metadata stores transforms in rasterio affine order: (a, b, c, d, e, f).
        transform=Affine(*transform_values),
        elevations=elevations,
        valid_mask=valid_mask,
        max_elevation=float(meta["max_elevation"]),
    )


def _inspect_surface(
    surface: RasterSurface,
    terrain_max_elevation: float,
    stitch_points: np.ndarray,
    stitch_triangles: np.ndarray,
) -> dict[str, object]:
    return {
        "terrain_max_elevation": float(terrain_max_elevation),
        "resolved_raster_name": surface.path.name,
        "stitch_point_count": int(stitch_points.shape[0]),
        "stitch_triangle_count": int(stitch_triangles.shape[0]),
        "has_populated_stitch_tin": bool(stitch_points.size > 0 and stitch_triangles.size > 0),
    }


def _inspect_hdf_with_surface(hdf_path: Path, surface: RasterSurface) -> dict[str, object]:
    points, triangles = load_stitch_arrays(hdf_path)
    return _inspect_surface(
        surface,
        _terrain_max_elevation(hdf_path, surface.max_elevation),
        points,
        triangles,
    )


def _emit_progress(
    progress_callback: ProgressCallback | None,
    step: str,
    completed: int,
    total: int,
    message: str,
) -> None:
    if progress_callback is None:
        return

    progress_callback(step, int(completed), int(total), message)


def _convert_surface(
    output_stem: str,
    surface: RasterSurface,
    terrain_max: float,
    points: np.ndarray,
    triangles: np.ndarray,
    top_elevation: float,
    sample_step: int,
    output_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    if float(top_elevation) < terrain_max:
        raise TerrainConversionError(
            "The top elevation must be greater than or equal to the max terrain elevation. "
            "Reduce the extent of the terrain surface to just the area you would like "
            "to be made into an STL file, then rerun the converter."
        )

    if points.size > 0 and triangles.size > 0 and int(sample_step) != 1:
        raise TerrainConversionError(
            "This terrain contains populated stitch TIN data. "
            "This converter only supports stitch-aware output at raster sample step 1. "
            "Reduce or crop the terrain externally and rerun with sample step 1."
        )

    _emit_progress(
        progress_callback,
        "validate-terrain",
        0,
        1,
        "Validating terrain metadata and stitch arrays...",
    )
    lookup = CoordinateLookup(surface.transform, surface.width, surface.height)
    stitch_mesh = validate_stitch_mesh(points, triangles, surface, lookup)
    _emit_progress(
        progress_callback,
        "validate-terrain",
        1,
        1,
        "Terrain validation complete.",
    )

    output_name = f"{Path(output_stem).stem}.stl"
    output_path = output_dir / output_name
    triangle_count, wall_triangle_count = write_shell_stl(
        output_path=output_path,
        surface=surface,
        top_elevation=float(top_elevation),
        sample_step=int(sample_step),
        stitch_mesh=stitch_mesh,
        progress_callback=progress_callback,
    )
    _emit_progress(
        progress_callback,
        "finalize",
        0,
        1,
        "Finalizing STL file...",
    )
    stl_size_bytes = output_path.stat().st_size
    _emit_progress(
        progress_callback,
        "finalize",
        1,
        1,
        "Finalizing STL file...",
    )

    return {
        "output_filename": output_name,
        "terrain_max_elevation": terrain_max,
        "resolved_raster_name": surface.path.name,
        "triangle_count": int(triangle_count),
        "wall_triangle_count": int(wall_triangle_count),
        "stitch_point_count": int(stitch_mesh.point_count),
        "stitch_triangle_count": int(stitch_mesh.triangle_count),
        "stitch_bridge_triangle_count": int(len(stitch_mesh.bridge_triangles)),
        "stl_size_bytes": int(stl_size_bytes),
    }


def _convert_hdf_with_surface(
    hdf_path: Path,
    surface: RasterSurface,
    top_elevation: float,
    sample_step: int,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    points, triangles = load_stitch_arrays(hdf_path)
    return _convert_surface(
        hdf_path.stem,
        surface,
        _terrain_max_elevation(hdf_path, surface.max_elevation),
        points,
        triangles,
        top_elevation,
        sample_step,
        hdf_path.parent,
        progress_callback=progress_callback,
    )


def inspect_terrain(session_dir: str, hdf_name: str) -> dict[str, object]:
    session_path = Path(session_dir)
    hdf_path = session_path / hdf_name
    raster_path = resolve_raster_path(hdf_path)
    surface = load_raster_surface(raster_path)
    return _inspect_hdf_with_surface(hdf_path, surface)


def inspect_terrain_from_surface(
    session_dir: str,
    hdf_name: str,
    surface_meta_name: str,
) -> dict[str, object]:
    session_path = Path(session_dir)
    hdf_path = session_path / hdf_name
    surface = _load_browser_surface(session_dir, surface_meta_name)
    return _inspect_hdf_with_surface(hdf_path, surface)


def inspect_surface(
    session_dir: str,
    surface_meta_name: str,
) -> dict[str, object]:
    surface = _load_browser_surface(session_dir, surface_meta_name)
    points, triangles = empty_stitch_arrays()
    return _inspect_surface(surface, surface.max_elevation, points, triangles)


def convert_terrain(
    session_dir: str,
    hdf_name: str,
    top_elevation: float,
    sample_step: int,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    session_path = Path(session_dir)
    hdf_path = session_path / hdf_name
    raster_path = resolve_raster_path(hdf_path)
    surface = load_raster_surface(raster_path)
    return _convert_hdf_with_surface(
        hdf_path,
        surface,
        top_elevation,
        sample_step,
        progress_callback=progress_callback,
    )


def convert_terrain_from_surface(
    session_dir: str,
    hdf_name: str,
    top_elevation: float,
    sample_step: int,
    surface_meta_name: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    session_path = Path(session_dir)
    hdf_path = session_path / hdf_name
    surface = _load_browser_surface(session_dir, surface_meta_name)
    return _convert_hdf_with_surface(
        hdf_path,
        surface,
        top_elevation,
        sample_step,
        progress_callback=progress_callback,
    )


def convert_surface(
    session_dir: str,
    terrain_name: str,
    top_elevation: float,
    sample_step: int,
    surface_meta_name: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    session_path = Path(session_dir)
    surface = _load_browser_surface(session_dir, surface_meta_name)
    points, triangles = empty_stitch_arrays()
    return _convert_surface(
        terrain_name,
        surface,
        surface.max_elevation,
        points,
        triangles,
        top_elevation,
        sample_step,
        session_path,
        progress_callback=progress_callback,
    )
