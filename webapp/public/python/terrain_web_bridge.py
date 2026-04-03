from __future__ import annotations
import json
from pathlib import Path
from typing import Callable

import numpy as np
from rasterio.transform import Affine

from terrain_to_stl import (
    CoordinateLookup,
    RasterSurface,
    SparseRasterSurface,
    SparseRefinementTile,
    SUPPORTED_STITCH_SAMPLE_STEPS,
    TerrainConversionError,
    SurfaceLike,
    build_refinement_regions,
    build_sample_indices,
    build_skipped_coarse_cells,
    build_stitch_components,
    count_transition_strip_triangles,
    empty_stitch_arrays,
    load_hdf_max_elevation,
    load_raster_surface,
    load_stitch_arrays,
    map_stitch_points_to_vertex_ids,
    normalize_stitch_triangle_indices,
    resolve_raster_path,
    validate_sample_step,
    validate_stitch_mesh,
    write_shell_stl,
)

ProgressCallback = Callable[[str, int, int, str], None]
STL_HEADER_BYTES = 84
STL_TRIANGLE_BYTES = 50


def _terrain_max_elevation(hdf_path: Path, surface_max: float) -> float:
    hdf_max = load_hdf_max_elevation(hdf_path)
    return hdf_max if hdf_max is not None else float(surface_max)


def _load_browser_surface(session_dir: str, surface_meta_name: str) -> SurfaceLike:
    session_path = Path(session_dir)
    meta_path = session_path / surface_meta_name
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    kind = str(meta.get("kind", "full"))
    width = int(meta["width"])
    height = int(meta["height"])
    transform_values = meta["transform"]
    if len(transform_values) != 6:
        raise TerrainConversionError("The browser raster surface metadata does not contain a 6-value affine transform.")

    transform = Affine(*transform_values)
    resolved_raster_path = Path(str(meta["resolved_raster_name"]))
    max_elevation = float(meta["max_elevation"])

    if kind == "full":
        elevations_path = session_path / str(meta["elevations_file"])
        valid_mask_path = session_path / str(meta["valid_mask_file"])
        elevations = np.frombuffer(elevations_path.read_bytes(), dtype=np.float32).reshape((height, width))
        valid_mask = np.frombuffer(valid_mask_path.read_bytes(), dtype=np.uint8).reshape((height, width)).astype(bool, copy=False)

        if elevations.shape != (height, width) or valid_mask.shape != (height, width):
            raise TerrainConversionError("The browser raster surface arrays do not match the declared raster dimensions.")
        if not np.any(valid_mask):
            raise TerrainConversionError("The browser raster surface does not contain any valid elevation cells.")

        return RasterSurface(
            path=resolved_raster_path,
            width=width,
            height=height,
            # Browser raster metadata stores transforms in rasterio affine order: (a, b, c, d, e, f).
            transform=transform,
            elevations=elevations,
            valid_mask=valid_mask,
            max_elevation=max_elevation,
        )

    if kind == "sparse":
        sampled_rows_path = session_path / str(meta["sampled_rows_file"])
        sampled_cols_path = session_path / str(meta["sampled_cols_file"])
        coarse_elevations_path = session_path / str(meta["coarse_elevations_file"])
        coarse_valid_mask_path = session_path / str(meta["coarse_valid_mask_file"])
        sampled_rows = np.frombuffer(sampled_rows_path.read_bytes(), dtype=np.int32)
        sampled_cols = np.frombuffer(sampled_cols_path.read_bytes(), dtype=np.int32)
        coarse_shape = (sampled_rows.shape[0], sampled_cols.shape[0])
        coarse_elevations = np.frombuffer(coarse_elevations_path.read_bytes(), dtype=np.float32).reshape(coarse_shape)
        coarse_valid_mask = np.frombuffer(coarse_valid_mask_path.read_bytes(), dtype=np.uint8).reshape(coarse_shape).astype(bool, copy=False)
        raw_tiles = meta.get("refinement_tiles", [])
        if not isinstance(raw_tiles, list):
            raise TerrainConversionError("The browser sparse raster metadata does not contain a valid refinement tile list.")

        refinement_tiles: list[SparseRefinementTile] = []
        has_valid_cells = bool(np.any(coarse_valid_mask))
        for raw_tile in raw_tiles:
            if not isinstance(raw_tile, dict):
                raise TerrainConversionError("The browser sparse raster metadata contains an invalid refinement tile entry.")

            row_start = int(raw_tile["row_start"])
            row_end = int(raw_tile["row_end"])
            col_start = int(raw_tile["col_start"])
            col_end = int(raw_tile["col_end"])
            tile_shape = ((row_end - row_start) + 1, (col_end - col_start) + 1)
            elevations_path = session_path / str(raw_tile["elevations_file"])
            valid_mask_path = session_path / str(raw_tile["valid_mask_file"])
            elevations = np.frombuffer(elevations_path.read_bytes(), dtype=np.float32).reshape(tile_shape)
            valid_mask = np.frombuffer(valid_mask_path.read_bytes(), dtype=np.uint8).reshape(tile_shape).astype(bool, copy=False)
            if np.any(valid_mask):
                has_valid_cells = True

            refinement_tiles.append(
                SparseRefinementTile(
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    elevations=elevations,
                    valid_mask=valid_mask,
                )
            )

        if not has_valid_cells:
            raise TerrainConversionError("The browser sparse raster surface does not contain any valid elevation cells.")

        return SparseRasterSurface(
            path=resolved_raster_path,
            width=width,
            height=height,
            transform=transform,
            sampled_rows=sampled_rows,
            sampled_cols=sampled_cols,
            coarse_elevations=coarse_elevations,
            coarse_valid_mask=coarse_valid_mask,
            refinement_tiles=tuple(refinement_tiles),
            max_elevation=max_elevation,
        )

    raise TerrainConversionError(f"Unsupported browser raster surface kind: {kind}")


def _adaptive_stitch_metrics_by_step(
    width: int,
    height: int,
    transform: Affine,
    stitch_points: np.ndarray,
    stitch_triangles: np.ndarray,
) -> tuple[int, dict[str, dict[str, int]]]:
    if stitch_points.size == 0 or stitch_triangles.size == 0:
        return 0, {}

    point_vertex_ids = map_stitch_points_to_vertex_ids(stitch_points, width, height, transform)
    normalized_triangles = normalize_stitch_triangle_indices(stitch_triangles, stitch_points.shape[0])
    components = build_stitch_components(width, point_vertex_ids, normalized_triangles)
    metrics_by_step: dict[str, dict[str, int]] = {}

    for sample_step in SUPPORTED_STITCH_SAMPLE_STEPS:
        sampled_rows = build_sample_indices(height, sample_step)
        sampled_cols = build_sample_indices(width, sample_step)
        refinement_regions = build_refinement_regions(height, width, sample_step, components)
        skipped_coarse_cells = build_skipped_coarse_cells(sampled_rows, sampled_cols, refinement_regions)
        coarse_cell_count = ((len(sampled_rows) - 1) * (len(sampled_cols) - 1)) - len(skipped_coarse_cells)
        refined_cell_count = sum(
            (region.row_end - region.row_start) * (region.col_end - region.col_start)
            for region in refinement_regions
        )
        transition_triangle_count = sum(
            count_transition_strip_triangles(sampled_rows, sampled_cols, region)
            for region in refinement_regions
        )
        refinement_perimeter_cell_count = sum(
            2 * ((region.row_end - region.row_start) + (region.col_end - region.col_start))
            for region in refinement_regions
        )
        refined_vertex_count = sum(
            ((region.row_end - region.row_start) + 1) * ((region.col_end - region.col_start) + 1)
            for region in refinement_regions
        )
        largest_refinement_vertex_count = max(
            (
                ((region.row_end - region.row_start) + 1) * ((region.col_end - region.col_start) + 1)
                for region in refinement_regions
            ),
            default=0,
        )
        metrics_by_step[str(sample_step)] = {
            "coarse_cell_count": int(coarse_cell_count),
            "refined_cell_count": int(refined_cell_count),
            "transition_triangle_count": int(transition_triangle_count),
            "refinement_perimeter_cell_count": int(refinement_perimeter_cell_count),
            "refined_vertex_count": int(refined_vertex_count),
            "largest_refinement_vertex_count": int(largest_refinement_vertex_count),
            "region_count": int(len(refinement_regions)),
        }

    return len(components), metrics_by_step


def _inspect_surface(
    surface: SurfaceLike,
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
        "stitch_component_count": 0,
        "adaptive_stitch_metrics": {},
    }


def _inspect_hdf_metadata(
    hdf_path: Path,
    resolved_raster_name: str,
    raster_width: int,
    raster_height: int,
    raster_transform: tuple[float, float, float, float, float, float],
    raster_max_elevation: float | None = None,
) -> dict[str, object]:
    stitch_points, stitch_triangles = load_stitch_arrays(hdf_path)
    stitch_point_count = int(stitch_points.shape[0])
    stitch_triangle_count = int(stitch_triangles.shape[0])
    transform = Affine(*raster_transform)
    stitch_component_count, adaptive_stitch_metrics = _adaptive_stitch_metrics_by_step(
        raster_width,
        raster_height,
        transform,
        stitch_points,
        stitch_triangles,
    )

    terrain_max = load_hdf_max_elevation(hdf_path)
    if terrain_max is None:
        terrain_max = None if raster_max_elevation is None else float(raster_max_elevation)

    return {
        "terrain_max_elevation": terrain_max,
        "resolved_raster_name": resolved_raster_name,
        "stitch_point_count": stitch_point_count,
        "stitch_triangle_count": stitch_triangle_count,
        "has_populated_stitch_tin": bool(stitch_point_count > 0 and stitch_triangle_count > 0),
        "stitch_component_count": stitch_component_count,
        "adaptive_stitch_metrics": adaptive_stitch_metrics,
    }


def _inspect_hdf_with_surface(hdf_path: Path, surface: SurfaceLike) -> dict[str, object]:
    points, triangles = load_stitch_arrays(hdf_path)
    return _inspect_surface(
        surface,
        _terrain_max_elevation(hdf_path, surface.max_elevation),
        points,
        triangles,
    )


def _build_sparse_refinement_plan(
    hdf_path: Path,
    raster_width: int,
    raster_height: int,
    raster_transform: tuple[float, float, float, float, float, float],
    sample_step: int,
) -> dict[str, object]:
    stitch_points, stitch_triangles = load_stitch_arrays(hdf_path)
    if stitch_points.size == 0 or stitch_triangles.size == 0 or sample_step <= 1:
        return {"refinement_regions": []}

    transform = Affine(*raster_transform)
    point_vertex_ids = map_stitch_points_to_vertex_ids(stitch_points, raster_width, raster_height, transform)
    normalized_triangles = normalize_stitch_triangle_indices(stitch_triangles, stitch_points.shape[0])
    components = build_stitch_components(raster_width, point_vertex_ids, normalized_triangles)
    refinement_regions = build_refinement_regions(raster_height, raster_width, sample_step, components)
    return {
        "refinement_regions": [
            {
                "row_start": int(region.row_start),
                "row_end": int(region.row_end),
                "col_start": int(region.col_start),
                "col_end": int(region.col_end),
            }
            for region in refinement_regions
        ]
    }


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
    surface: SurfaceLike,
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

    validate_sample_step(
        int(sample_step),
        bool(points.size > 0 and triangles.size > 0),
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
    surface: SurfaceLike,
    top_elevation: float,
    sample_step: int,
    terrain_max_override: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    points, triangles = load_stitch_arrays(hdf_path)
    terrain_max = (
        float(terrain_max_override)
        if terrain_max_override is not None
        else _terrain_max_elevation(hdf_path, surface.max_elevation)
    )
    return _convert_surface(
        hdf_path.stem,
        surface,
        terrain_max,
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


def inspect_hdf_metadata(
    session_dir: str,
    hdf_name: str,
    resolved_raster_name: str,
    raster_width: int,
    raster_height: int,
    raster_transform: tuple[float, float, float, float, float, float],
    raster_max_elevation: float | None = None,
) -> dict[str, object]:
    session_path = Path(session_dir)
    hdf_path = session_path / hdf_name
    return _inspect_hdf_metadata(
        hdf_path,
        resolved_raster_name,
        int(raster_width),
        int(raster_height),
        raster_transform,
        raster_max_elevation=raster_max_elevation,
    )


def inspect_surface(
    session_dir: str,
    surface_meta_name: str,
) -> dict[str, object]:
    surface = _load_browser_surface(session_dir, surface_meta_name)
    points, triangles = empty_stitch_arrays()
    return _inspect_surface(surface, surface.max_elevation, points, triangles)


def build_hdf_sparse_plan(
    session_dir: str,
    hdf_name: str,
    raster_width: int,
    raster_height: int,
    raster_transform: tuple[float, float, float, float, float, float],
    sample_step: int,
) -> dict[str, object]:
    session_path = Path(session_dir)
    hdf_path = session_path / hdf_name
    return _build_sparse_refinement_plan(
        hdf_path,
        int(raster_width),
        int(raster_height),
        raster_transform,
        int(sample_step),
    )


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
    terrain_max_override: float | None = None,
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
        terrain_max_override=terrain_max_override,
        progress_callback=progress_callback,
    )


def convert_surface(
    session_dir: str,
    terrain_name: str,
    top_elevation: float,
    sample_step: int,
    surface_meta_name: str,
    terrain_max_override: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    session_path = Path(session_dir)
    surface = _load_browser_surface(session_dir, surface_meta_name)
    points, triangles = empty_stitch_arrays()
    return _convert_surface(
        terrain_name,
        surface,
        float(terrain_max_override) if terrain_max_override is not None else surface.max_elevation,
        points,
        triangles,
        top_elevation,
        sample_step,
        session_path,
        progress_callback=progress_callback,
    )
