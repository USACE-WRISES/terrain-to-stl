from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

from terrain_to_stl import (
    CoordinateLookup,
    FLOAT_TOLERANCE,
    RasterSurface,
    SUPPORTED_STITCH_SAMPLE_STEPS,
    SparseRasterSurface,
    SparseRefinementTile,
    StitchComponent,
    TerrainConversionError,
    TerrainSource,
    build_bridge_triangles,
    build_refinement_regions,
    build_sample_indices,
    build_skipped_coarse_cells,
    build_stitch_components,
    convert_terrain_source,
    count_transition_strip_triangles,
    emit_shell_triangle,
    empty_stitch_arrays,
    load_hdf_max_elevation,
    load_raster_surface,
    load_stitch_arrays,
    map_stitch_points_to_vertex_ids,
    normalize_stitch_triangle_indices,
    resolve_output_path,
    resolve_raster_path,
    resolve_terrain_source_path,
    validate_sample_step,
    validate_stitch_mesh,
)

DesktopProgressCallback = Callable[[str, int, int, str], None]
DesktopLogCallback = Callable[[str], None]

PROGRESS_STEP_ORDER = (
    "resolve-raster",
    "validate-terrain",
    "write-surfaces",
    "write-stitches",
    "write-walls",
    "finalize",
    "complete",
)
PROGRESS_STEP_WEIGHTS = {
    "resolve-raster": 0.08,
    "validate-terrain": 0.07,
    "write-surfaces": 0.70,
    "write-stitches": 0.05,
    "write-walls": 0.08,
    "finalize": 0.02,
}

STL_HEADER_BYTES = 84
STL_TRIANGLE_BYTES = 50
_MIN_BENCHMARK_ROWS = 4
_MIN_BENCHMARK_TILES = 2
_TRIANGLE_RATE_BENCHMARK_SECONDS = 0.25


@dataclass(frozen=True, slots=True)
class RasterWindowSpec:
    row_off: int
    col_off: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class RasterMetadata:
    path: Path
    width: int
    height: int
    transform: Affine
    block_windows: tuple[RasterWindowSpec, ...]
    largest_block_cell_count: int


@dataclass(frozen=True, slots=True)
class TerrainInspectionContext:
    source_path: Path
    source_kind: str
    raster: RasterMetadata
    terrain_max_elevation: float
    stitch_points: np.ndarray
    stitch_triangles: np.ndarray
    stitch_components: tuple[StitchComponent, ...]
    stitch_bridge_triangle_count: int
    adaptive_stitch_metrics: dict[str, dict[str, int]]

    @property
    def has_populated_stitch_tin(self) -> bool:
        return bool(self.stitch_points.size > 0 and self.stitch_triangles.size > 0)


def _emit_progress(
    progress_callback: DesktopProgressCallback | None,
    step: str,
    completed: int,
    total: int,
    message: str,
) -> None:
    if progress_callback is None:
        return

    safe_total = max(1, int(total))
    safe_completed = max(0, min(int(completed), safe_total))
    progress_callback(step, safe_completed, safe_total, message)


def _emit_log(log_callback: DesktopLogCallback | None, message: str) -> None:
    if log_callback is None:
        return
    log_callback(message)


def progress_percent(step: str, completed: int, total: int) -> int:
    if step == "complete":
        return 100

    total_weight = 0.0
    for ordered_step in PROGRESS_STEP_ORDER:
        if ordered_step == "complete":
            break

        if ordered_step == step:
            fraction = 0.0 if total <= 0 else max(0.0, min(float(completed) / float(total), 1.0))
            total_weight += PROGRESS_STEP_WEIGHTS[ordered_step] * fraction
            break

        total_weight += PROGRESS_STEP_WEIGHTS[ordered_step]

    return int(round(max(0.0, min(total_weight, 1.0)) * 100.0))


def desktop_worker_count() -> int:
    return min(8, max(2, os.cpu_count() or 1))


def _chunk_sequence(items: Sequence[object], chunk_count: int) -> tuple[tuple[object, ...], ...]:
    if not items:
        return ()

    normalized_chunk_count = max(1, min(int(chunk_count), len(items)))
    chunk_size = math.ceil(len(items) / normalized_chunk_count)
    chunks: list[tuple[object, ...]] = []
    for start in range(0, len(items), chunk_size):
        chunks.append(tuple(items[start:start + chunk_size]))
    return tuple(chunks)


def _select_evenly_spaced_items(items: Sequence[object], target_count: int) -> tuple[object, ...]:
    if not items:
        return ()

    if target_count >= len(items):
        return tuple(items)

    selected: list[object] = []
    last_index = len(items) - 1
    for offset in range(target_count):
        fraction = 0.0 if target_count <= 1 else float(offset) / float(target_count - 1)
        index = int(round(fraction * last_index))
        candidate = items[index]
        if not selected or candidate != selected[-1]:
            selected.append(candidate)

    return tuple(selected)


def _window_from_spec(window_spec: RasterWindowSpec) -> Window:
    return Window(
        col_off=int(window_spec.col_off),
        row_off=int(window_spec.row_off),
        width=int(window_spec.width),
        height=int(window_spec.height),
    )


def _window_cell_count(window_spec: RasterWindowSpec) -> int:
    return int(window_spec.height) * int(window_spec.width)


def read_raster_metadata(raster_path: str | Path) -> RasterMetadata:
    resolved = Path(raster_path)
    with rasterio.open(resolved) as dataset:
        if dataset.count < 1:
            raise TerrainConversionError(f"{resolved} does not contain any raster bands.")

        block_windows: list[RasterWindowSpec] = []
        largest_block_cell_count = 0
        for _block_index, window in dataset.block_windows(1):
            spec = RasterWindowSpec(
                row_off=int(window.row_off),
                col_off=int(window.col_off),
                height=int(window.height),
                width=int(window.width),
            )
            block_windows.append(spec)
            largest_block_cell_count = max(largest_block_cell_count, _window_cell_count(spec))

        if not block_windows:
            full_window = RasterWindowSpec(
                row_off=0,
                col_off=0,
                height=int(dataset.height),
                width=int(dataset.width),
            )
            block_windows.append(full_window)
            largest_block_cell_count = _window_cell_count(full_window)

        return RasterMetadata(
            path=resolved,
            width=int(dataset.width),
            height=int(dataset.height),
            transform=dataset.transform,
            block_windows=tuple(block_windows),
            largest_block_cell_count=int(largest_block_cell_count),
        )


def _scan_window_chunk_max(raster_path: Path, window_specs: Sequence[RasterWindowSpec]) -> tuple[float | None, int]:
    local_max: float | None = None
    with rasterio.open(raster_path) as dataset:
        for window_spec in window_specs:
            band = dataset.read(1, window=_window_from_spec(window_spec), masked=True)
            compressed = band.compressed()
            if compressed.size == 0:
                continue
            candidate = float(np.max(compressed))
            if local_max is None or candidate > local_max:
                local_max = candidate
    return local_max, len(window_specs)


def compute_raster_max_elevation_parallel(
    raster: RasterMetadata,
    *,
    progress_callback: DesktopProgressCallback | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
    progress_step: str = "resolve-raster",
    message_prefix: str = "Scanning raster blocks to determine exact terrain max",
) -> float:
    window_specs = raster.block_windows
    if not window_specs:
        raise TerrainConversionError(f"{raster.path} does not contain any readable raster windows.")

    total_windows = len(window_specs)
    callback_total = total_windows if progress_total is None else int(progress_total)
    worker_total = min(desktop_worker_count(), total_windows)
    completed_windows = 0
    local_max: float | None = None
    callback_interval = max(1, total_windows // 20)
    chunks = _chunk_sequence(window_specs, worker_total)

    _emit_progress(
        progress_callback,
        progress_step,
        progress_offset,
        callback_total,
        f"{message_prefix}...",
    )

    with ThreadPoolExecutor(max_workers=worker_total) as executor:
        futures = [executor.submit(_scan_window_chunk_max, raster.path, chunk) for chunk in chunks]
        for future in as_completed(futures):
            chunk_max, processed_count = future.result()
            completed_windows += processed_count
            if chunk_max is not None and (local_max is None or chunk_max > local_max):
                local_max = chunk_max

            overall_completed = progress_offset + completed_windows
            if (
                completed_windows == total_windows
                or completed_windows == processed_count
                or completed_windows % callback_interval == 0
            ):
                _emit_progress(
                    progress_callback,
                    progress_step,
                    overall_completed,
                    callback_total,
                    f"{message_prefix} ({completed_windows}/{total_windows} windows)...",
                )

    if local_max is None:
        raise TerrainConversionError(f"{raster.path} does not contain any valid elevation cells.")

    return float(local_max)


def _adaptive_stitch_metrics_by_step(
    width: int,
    height: int,
    transform: Affine,
    stitch_points: np.ndarray,
    stitch_triangles: np.ndarray,
) -> tuple[tuple[StitchComponent, ...], int, dict[str, dict[str, int]]]:
    if stitch_points.size == 0 or stitch_triangles.size == 0:
        return (), 0, {}

    point_vertex_ids = map_stitch_points_to_vertex_ids(stitch_points, width, height, transform)
    normalized_triangles = normalize_stitch_triangle_indices(stitch_triangles, stitch_points.shape[0])
    components = build_stitch_components(width, point_vertex_ids, normalized_triangles)
    lookup = CoordinateLookup(transform, width, height)
    bridge_triangle_count = len(
        build_bridge_triangles(width, lookup, point_vertex_ids, normalized_triangles)
    )
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
        metrics_by_step[str(int(sample_step))] = {
            "coarse_cell_count": int(coarse_cell_count),
            "refined_cell_count": int(refined_cell_count),
            "transition_triangle_count": int(transition_triangle_count),
            "refinement_perimeter_cell_count": int(refinement_perimeter_cell_count),
            "refined_vertex_count": int(refined_vertex_count),
            "largest_refinement_vertex_count": int(largest_refinement_vertex_count),
            "region_count": int(len(refinement_regions)),
        }

    return components, int(bridge_triangle_count), metrics_by_step


def _inspect_hdf_metadata(
    hdf_path: Path,
    raster: RasterMetadata,
    *,
    require_exact_max: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, tuple[StitchComponent, ...], int, dict[str, dict[str, int]]]:
    stitch_points, stitch_triangles = load_stitch_arrays(hdf_path)
    components, bridge_triangle_count, adaptive_metrics = _adaptive_stitch_metrics_by_step(
        raster.width,
        raster.height,
        raster.transform,
        stitch_points,
        stitch_triangles,
    )

    terrain_max = load_hdf_max_elevation(hdf_path)
    if terrain_max is None and require_exact_max:
        terrain_max = compute_raster_max_elevation_parallel(raster)
    if terrain_max is None:
        terrain_max = 0.0

    return (
        float(terrain_max),
        stitch_points,
        stitch_triangles,
        components,
        int(bridge_triangle_count),
        adaptive_metrics,
    )


def _inspect_dem_metadata(
    source_path: Path,
    raster: RasterMetadata,
    *,
    require_exact_max: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, tuple[StitchComponent, ...], int, dict[str, dict[str, int]]]:
    _ = source_path
    terrain_max = compute_raster_max_elevation_parallel(raster) if require_exact_max else 0.0
    stitch_points, stitch_triangles = empty_stitch_arrays()
    return (
        float(terrain_max),
        stitch_points,
        stitch_triangles,
        (),
        0,
        {},
    )


def _load_inspection_context(input_path: str | Path, *, require_exact_max: bool = True) -> TerrainInspectionContext:
    terrain_path = resolve_terrain_source_path(str(input_path))
    if terrain_path.suffix.lower() == ".hdf":
        raster_path = resolve_raster_path(terrain_path)
        raster = read_raster_metadata(raster_path)
        (
            terrain_max,
            stitch_points,
            stitch_triangles,
            stitch_components,
            stitch_bridge_triangle_count,
            adaptive_stitch_metrics,
        ) = _inspect_hdf_metadata(terrain_path, raster, require_exact_max=require_exact_max)
        return TerrainInspectionContext(
            source_path=terrain_path,
            source_kind="hdf",
            raster=raster,
            terrain_max_elevation=terrain_max,
            stitch_points=stitch_points,
            stitch_triangles=stitch_triangles,
            stitch_components=stitch_components,
            stitch_bridge_triangle_count=stitch_bridge_triangle_count,
            adaptive_stitch_metrics=adaptive_stitch_metrics,
        )

    raster = read_raster_metadata(terrain_path)
    (
        terrain_max,
        stitch_points,
        stitch_triangles,
        stitch_components,
        stitch_bridge_triangle_count,
        adaptive_stitch_metrics,
    ) = _inspect_dem_metadata(terrain_path, raster, require_exact_max=require_exact_max)
    return TerrainInspectionContext(
        source_path=terrain_path,
        source_kind="dem",
        raster=raster,
        terrain_max_elevation=terrain_max,
        stitch_points=stitch_points,
        stitch_triangles=stitch_triangles,
        stitch_components=stitch_components,
        stitch_bridge_triangle_count=stitch_bridge_triangle_count,
        adaptive_stitch_metrics=adaptive_stitch_metrics,
    )


def _estimate_stl_upper_bound_bytes(
    width: int,
    height: int,
    sample_step: int,
    stitch_triangle_upper_bound: int,
) -> int:
    sampled_row_count = len(build_sample_indices(height, sample_step))
    sampled_column_count = len(build_sample_indices(width, sample_step))
    top_bottom_triangles = 4 * (sampled_row_count - 1) * (sampled_column_count - 1)
    wall_triangles = 4 * ((sampled_row_count - 1) + (sampled_column_count - 1))
    total_triangles = top_bottom_triangles + wall_triangles + int(stitch_triangle_upper_bound)
    return int(STL_HEADER_BYTES + (total_triangles * STL_TRIANGLE_BYTES))


def _estimate_adaptive_stl_upper_bound_bytes(
    width: int,
    height: int,
    sample_step: int,
    stitch_triangle_upper_bound: int,
    metrics: dict[str, int],
) -> int:
    sampled_row_count = len(build_sample_indices(height, sample_step))
    sampled_column_count = len(build_sample_indices(width, sample_step))
    top_bottom_triangles = (
        (4 * int(metrics["coarse_cell_count"]))
        + (4 * int(metrics["refined_cell_count"]))
        + (2 * int(metrics["transition_triangle_count"]))
        + (2 * int(stitch_triangle_upper_bound))
    )
    boundary_edge_upper_bound = (
        (2 * ((sampled_row_count - 1) + (sampled_column_count - 1)))
        + int(metrics["refinement_perimeter_cell_count"])
    )
    wall_triangles = boundary_edge_upper_bound * 2
    return int(STL_HEADER_BYTES + ((top_bottom_triangles + wall_triangles) * STL_TRIANGLE_BYTES))


def _build_sample_step_options(context: TerrainInspectionContext) -> list[dict[str, object]]:
    stitch_triangle_upper_bound = int(context.stitch_triangles.shape[0])
    options: list[dict[str, object]] = []
    for sample_step in SUPPORTED_STITCH_SAMPLE_STEPS:
        metrics = context.adaptive_stitch_metrics.get(str(int(sample_step)))
        if context.has_populated_stitch_tin and metrics is not None:
            estimated_size_bytes = _estimate_adaptive_stl_upper_bound_bytes(
                context.raster.width,
                context.raster.height,
                int(sample_step),
                stitch_triangle_upper_bound,
                metrics,
            )
        else:
            estimated_size_bytes = _estimate_stl_upper_bound_bytes(
                context.raster.width,
                context.raster.height,
                int(sample_step),
                stitch_triangle_upper_bound,
            )

        options.append(
            {
                "value": int(sample_step),
                "estimated_size_bytes": int(estimated_size_bytes),
                "size_estimate_kind": "upper-bound",
                "estimated_duration_seconds": None,
                "duration_estimate_kind": "pending",
                "disabled": False,
                "reason": None,
            }
        )
    return options


def inspect_source(input_path: str | Path) -> dict[str, object]:
    context = _load_inspection_context(input_path)
    sample_step_options = _build_sample_step_options(context)
    return {
        "source_path": str(context.source_path),
        "source_kind": context.source_kind,
        "resolved_raster_path": str(context.raster.path),
        "resolved_raster_name": context.raster.path.name,
        "terrain_max_elevation": float(context.terrain_max_elevation),
        "raster_width": int(context.raster.width),
        "raster_height": int(context.raster.height),
        "stitch_point_count": int(context.stitch_points.shape[0]),
        "stitch_triangle_count": int(context.stitch_triangles.shape[0]),
        "stitch_bridge_triangle_count": int(context.stitch_bridge_triangle_count),
        "stitch_component_count": int(len(context.stitch_components)),
        "has_populated_stitch_tin": context.has_populated_stitch_tin,
        "sample_step_requires_preset": context.has_populated_stitch_tin,
        "suggested_sample_steps": [int(step) for step in SUPPORTED_STITCH_SAMPLE_STEPS],
        "default_sample_step": 1,
        "default_output_path": str(resolve_output_path(context.source_path, None)),
        "adaptive_stitch_metrics": context.adaptive_stitch_metrics,
        "sample_step_options": sample_step_options,
    }


def _read_sampled_row_chunk(
    raster_path: Path,
    row_values: Sequence[int],
    sampled_cols: np.ndarray,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    with rasterio.open(raster_path) as dataset:
        for row in row_values:
            band = dataset.read(
                1,
                window=Window(col_off=0, row_off=int(row), width=dataset.width, height=1),
                masked=True,
            )
            elevations = band.filled(np.nan).astype(np.float32, copy=False)[0, sampled_cols].copy()
            valid_mask = (~np.ma.getmaskarray(band))[0, sampled_cols].copy()
            rows.append((int(row), elevations, valid_mask))
    return rows


def _read_refinement_tile_chunk(
    raster_path: Path,
    regions: Sequence[tuple[int, object]],
) -> list[tuple[int, SparseRefinementTile]]:
    tiles: list[tuple[int, SparseRefinementTile]] = []
    with rasterio.open(raster_path) as dataset:
        for region_index, region in regions:
            window = Window(
                col_off=int(region.col_start),
                row_off=int(region.row_start),
                width=int((region.col_end - region.col_start) + 1),
                height=int((region.row_end - region.row_start) + 1),
            )
            band = dataset.read(1, window=window, masked=True)
            elevations = band.filled(np.nan).astype(np.float32, copy=False).copy()
            valid_mask = (~np.ma.getmaskarray(band)).copy()
            tiles.append(
                (
                    int(region_index),
                    SparseRefinementTile(
                        row_start=int(region.row_start),
                        row_end=int(region.row_end),
                        col_start=int(region.col_start),
                        col_end=int(region.col_end),
                        elevations=elevations,
                        valid_mask=valid_mask,
                    ),
                )
            )
    return tiles


def load_sparse_raster_surface_parallel(
    raster: RasterMetadata,
    sample_step: int,
    refinement_regions: Sequence[object],
    *,
    max_elevation: float,
    progress_callback: DesktopProgressCallback | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
    progress_step: str = "resolve-raster",
) -> SparseRasterSurface:
    sampled_rows = np.asarray(build_sample_indices(raster.height, sample_step), dtype=np.int32)
    sampled_cols = np.asarray(build_sample_indices(raster.width, sample_step), dtype=np.int32)
    row_count = int(sampled_rows.shape[0])
    tile_count = int(len(refinement_regions))
    total_units = row_count + tile_count
    callback_total = total_units if progress_total is None else int(progress_total)
    coarse_shape = (row_count, int(sampled_cols.shape[0]))
    coarse_elevations = np.full(coarse_shape, np.nan, dtype=np.float32)
    coarse_valid_mask = np.zeros(coarse_shape, dtype=bool)
    row_index_by_value = {int(row): index for index, row in enumerate(sampled_rows.tolist())}
    refinement_tiles: list[SparseRefinementTile | None] = [None] * tile_count
    completed_units = 0
    worker_total = desktop_worker_count()
    row_progress_interval = max(1, row_count // 20) if row_count > 0 else 1
    tile_progress_interval = max(1, tile_count // 20) if tile_count > 0 else 1

    _emit_progress(
        progress_callback,
        progress_step,
        progress_offset,
        callback_total,
        "Loading sparse raster surface...",
    )

    if row_count > 0:
        row_chunks = _chunk_sequence(sampled_rows.tolist(), min(worker_total, row_count))
        with ThreadPoolExecutor(max_workers=min(worker_total, len(row_chunks))) as executor:
            futures = [
                executor.submit(_read_sampled_row_chunk, raster.path, chunk, sampled_cols)
                for chunk in row_chunks
            ]
            for future in as_completed(futures):
                for row_value, elevations, valid_mask in future.result():
                    row_index = row_index_by_value[int(row_value)]
                    coarse_elevations[row_index, :] = elevations
                    coarse_valid_mask[row_index, :] = valid_mask
                    completed_units += 1
                    if (
                        completed_units == 1
                        or completed_units == row_count
                        or completed_units % row_progress_interval == 0
                    ):
                        _emit_progress(
                            progress_callback,
                            progress_step,
                            progress_offset + completed_units,
                            callback_total,
                            f"Loading sparse raster rows ({completed_units}/{total_units})...",
                        )

    if tile_count > 0:
        indexed_regions = tuple((index, region) for index, region in enumerate(refinement_regions))
        tile_chunks = _chunk_sequence(indexed_regions, min(worker_total, tile_count))
        with ThreadPoolExecutor(max_workers=min(worker_total, len(tile_chunks))) as executor:
            futures = [
                executor.submit(_read_refinement_tile_chunk, raster.path, chunk)
                for chunk in tile_chunks
            ]
            for future in as_completed(futures):
                for region_index, tile in future.result():
                    refinement_tiles[int(region_index)] = tile
                    completed_units += 1
                    tile_completed = completed_units - row_count
                    if (
                        tile_completed == 1
                        or tile_completed == tile_count
                        or tile_completed % tile_progress_interval == 0
                    ):
                        _emit_progress(
                            progress_callback,
                            progress_step,
                            progress_offset + completed_units,
                            callback_total,
                            f"Loading local refinement windows ({completed_units}/{total_units})...",
                        )

    has_valid_cells = bool(np.any(coarse_valid_mask))
    materialized_tiles: list[SparseRefinementTile] = []
    for tile in refinement_tiles:
        if tile is None:
            continue
        if np.any(tile.valid_mask):
            has_valid_cells = True
        materialized_tiles.append(tile)

    if not has_valid_cells:
        raise TerrainConversionError(f"{raster.path} does not contain any valid elevation cells.")

    return SparseRasterSurface(
        path=raster.path,
        width=raster.width,
        height=raster.height,
        transform=raster.transform,
        sampled_rows=sampled_rows,
        sampled_cols=sampled_cols,
        coarse_elevations=coarse_elevations,
        coarse_valid_mask=coarse_valid_mask,
        refinement_tiles=tuple(materialized_tiles),
        max_elevation=float(max_elevation),
    )


def _resolve_conversion_context(
    input_path: str | Path,
    sample_step: int,
    *,
    progress_callback: DesktopProgressCallback | None = None,
    log_callback: DesktopLogCallback | None = None,
) -> tuple[TerrainInspectionContext, TerrainSource]:
    terrain_path = resolve_terrain_source_path(str(input_path))
    _emit_log(log_callback, f"Resolving terrain source from {terrain_path}...")

    if terrain_path.suffix.lower() == ".hdf":
        raster_path = resolve_raster_path(terrain_path)
        source_kind = "hdf"
    else:
        raster_path = terrain_path
        source_kind = "dem"

    _emit_progress(progress_callback, "resolve-raster", 0, 1, "Reading raster metadata...")
    raster = read_raster_metadata(raster_path)
    _emit_log(
        log_callback,
        f"Resolved raster metadata: {raster.path.name} ({raster.width} columns x {raster.height} rows).",
    )

    if source_kind == "hdf":
        stitch_points, stitch_triangles = load_stitch_arrays(terrain_path)
        stitch_components, bridge_triangle_count, adaptive_metrics = _adaptive_stitch_metrics_by_step(
            raster.width,
            raster.height,
            raster.transform,
            stitch_points,
            stitch_triangles,
        )
        hdf_max = load_hdf_max_elevation(terrain_path)
        needs_exact_max_scan = hdf_max is None
        terrain_max = float(hdf_max) if hdf_max is not None else 0.0
    else:
        stitch_points, stitch_triangles = empty_stitch_arrays()
        stitch_components = ()
        bridge_triangle_count = 0
        adaptive_metrics = {}
        needs_exact_max_scan = True
        terrain_max = 0.0

    validate_sample_step(
        int(sample_step),
        bool(stitch_points.size > 0 and stitch_triangles.size > 0),
    )

    if sample_step <= 1:
        surface_units = 1
        refinement_regions: tuple[object, ...] = ()
    else:
        refinement_regions = build_refinement_regions(
            raster.height,
            raster.width,
            int(sample_step),
            stitch_components,
        )
        surface_units = len(build_sample_indices(raster.height, int(sample_step))) + len(refinement_regions)

    total_units = 1 + (len(raster.block_windows) if needs_exact_max_scan else 0) + surface_units
    _emit_progress(
        progress_callback,
        "resolve-raster",
        1,
        total_units,
        f"Resolved terrain raster metadata from {raster.path.name}.",
    )

    progress_offset = 1
    if needs_exact_max_scan:
        terrain_max = compute_raster_max_elevation_parallel(
            raster,
            progress_callback=progress_callback,
            progress_offset=progress_offset,
            progress_total=total_units,
            progress_step="resolve-raster",
            message_prefix="Scanning raster blocks to determine exact terrain max",
        )
        progress_offset += len(raster.block_windows)
    else:
        _emit_log(log_callback, f"Recovered terrain max elevation from HDF metadata: {terrain_max:.6f}.")

    _emit_log(log_callback, f"Using up to {desktop_worker_count()} worker(s) for raster loading.")

    if sample_step <= 1:
        _emit_progress(
            progress_callback,
            "resolve-raster",
            progress_offset,
            total_units,
            "Loading full raster surface...",
        )
        surface = load_raster_surface(raster.path)
        progress_offset += 1
        _emit_progress(
            progress_callback,
            "resolve-raster",
            progress_offset,
            total_units,
            f"Loaded full raster surface from {raster.path.name}.",
        )
    else:
        surface = load_sparse_raster_surface_parallel(
            raster,
            int(sample_step),
            refinement_regions,
            max_elevation=float(terrain_max),
            progress_callback=progress_callback,
            progress_offset=progress_offset,
            progress_total=total_units,
            progress_step="resolve-raster",
        )
        progress_offset = total_units
        _emit_progress(
            progress_callback,
            "resolve-raster",
            progress_offset,
            total_units,
            (
                f"Loaded sparse raster surface from {raster.path.name}: "
                f"{len(surface.sampled_rows)} sampled rows and {len(surface.refinement_tiles)} refinement windows."
            ),
        )

    context = TerrainInspectionContext(
        source_path=terrain_path,
        source_kind=source_kind,
        raster=raster,
        terrain_max_elevation=float(terrain_max),
        stitch_points=stitch_points,
        stitch_triangles=stitch_triangles,
        stitch_components=stitch_components,
        stitch_bridge_triangle_count=int(bridge_triangle_count),
        adaptive_stitch_metrics=adaptive_metrics,
    )
    terrain_source = TerrainSource(
        kind=source_kind,
        source_path=terrain_path,
        surface=surface,  # type: ignore[arg-type]
        terrain_max_elevation=float(terrain_max),
        stitch_points=stitch_points,
        stitch_triangles=stitch_triangles,
    )
    return context, terrain_source


class _NullTriangleWriter:
    def __init__(self) -> None:
        self.triangle_count = 0

    def write_triangle(
        self,
        p0: tuple[float, float, float],
        p1: tuple[float, float, float],
        p2: tuple[float, float, float],
    ) -> bool:
        ux = p1[0] - p0[0]
        uy = p1[1] - p0[1]
        uz = p1[2] - p0[2]
        vx = p2[0] - p0[0]
        vy = p2[1] - p0[1]
        vz = p2[2] - p0[2]

        nx = (uy * vz) - (uz * vy)
        ny = (uz * vx) - (ux * vz)
        nz = (ux * vy) - (uy * vx)
        length = math.sqrt((nx * nx) + (ny * ny) + (nz * nz))
        if length <= FLOAT_TOLERANCE:
            return False

        self.triangle_count += 1
        return True


_triangle_processing_rate_cache: float | None = None


def _benchmark_triangle_processing_rate() -> float:
    global _triangle_processing_rate_cache
    if _triangle_processing_rate_cache is not None:
        return _triangle_processing_rate_cache

    surface = RasterSurface(
        path=Path("triangle-rate-benchmark"),
        width=2,
        height=2,
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
        elevations=np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        valid_mask=np.asarray([[True, True], [True, True]], dtype=bool),
        max_elevation=0.0,
    )
    lookup = CoordinateLookup(surface.transform, surface.width, surface.height)
    triangles = ((0, 2, 3), (0, 3, 1))
    writer = _NullTriangleWriter()
    processed = 0
    started_at = time.perf_counter()
    while (time.perf_counter() - started_at) < _TRIANGLE_RATE_BENCHMARK_SECONDS:
        boundary_edges: dict[tuple[int, int], tuple[int, int]] = {}
        for triangle in triangles:
            emit_shell_triangle(
                writer,
                boundary_edges,
                surface,
                lookup,
                1.0,
                triangle,
            )
        processed += len(triangles)

    elapsed = max(time.perf_counter() - started_at, 1e-6)
    _triangle_processing_rate_cache = max(float(processed) / elapsed, 1.0)
    return _triangle_processing_rate_cache


def _benchmark_window_reads(
    raster_path: Path,
    window_specs: Sequence[RasterWindowSpec],
) -> tuple[float, float, int]:
    if not window_specs:
        return 0.0, 0.0, 0

    worker_total = min(desktop_worker_count(), len(window_specs))
    chunks = _chunk_sequence(window_specs, worker_total)
    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_total) as executor:
        futures = [executor.submit(_scan_window_chunk_max, raster_path, chunk) for chunk in chunks]
        for future in as_completed(futures):
            future.result()

    elapsed = max(time.perf_counter() - started_at, 1e-6)
    total_cells = sum(_window_cell_count(window_spec) for window_spec in window_specs)
    return float(total_cells) / elapsed, elapsed, total_cells


def _total_sparse_tile_cells(refinement_regions: Sequence[object]) -> int:
    return sum(
        int((region.row_end - region.row_start) + 1) * int((region.col_end - region.col_start) + 1)
        for region in refinement_regions
    )


def _benchmark_sparse_raster_reads(
    context: TerrainInspectionContext,
    sample_step: int,
) -> float:
    sampled_rows = build_sample_indices(context.raster.height, int(sample_step))
    sampled_row_specs = [
        RasterWindowSpec(row_off=int(row), col_off=0, height=1, width=context.raster.width)
        for row in sampled_rows
    ]
    target_row_count = max(_MIN_BENCHMARK_ROWS, desktop_worker_count() * 2)
    benchmark_row_specs = _select_evenly_spaced_items(sampled_row_specs, target_row_count)
    row_cells_per_second, _row_elapsed, _row_cells = _benchmark_window_reads(
        context.raster.path,
        benchmark_row_specs,  # type: ignore[arg-type]
    )
    total_row_cells = len(sampled_row_specs) * context.raster.width
    row_seconds = 0.0 if row_cells_per_second <= 0.0 else float(total_row_cells) / row_cells_per_second

    refinement_regions = build_refinement_regions(
        context.raster.height,
        context.raster.width,
        int(sample_step),
        context.stitch_components,
    )
    if not refinement_regions:
        return row_seconds

    tile_window_specs = [
        RasterWindowSpec(
            row_off=int(region.row_start),
            col_off=int(region.col_start),
            height=int((region.row_end - region.row_start) + 1),
            width=int((region.col_end - region.col_start) + 1),
        )
        for region in refinement_regions
    ]
    target_tile_count = max(_MIN_BENCHMARK_TILES, desktop_worker_count())
    benchmark_tile_specs = _select_evenly_spaced_items(tile_window_specs, target_tile_count)
    tile_cells_per_second, _tile_elapsed, _tile_cells = _benchmark_window_reads(
        context.raster.path,
        benchmark_tile_specs,  # type: ignore[arg-type]
    )
    total_tile_cells = _total_sparse_tile_cells(refinement_regions)
    tile_seconds = 0.0 if tile_cells_per_second <= 0.0 else float(total_tile_cells) / tile_cells_per_second
    return row_seconds + tile_seconds


def _benchmark_full_raster_reads(context: TerrainInspectionContext) -> float:
    window_specs = context.raster.block_windows
    target_window_count = max(_MIN_BENCHMARK_ROWS, desktop_worker_count() * 2)
    benchmark_window_specs = _select_evenly_spaced_items(window_specs, target_window_count)
    cells_per_second, _elapsed, _cells = _benchmark_window_reads(
        context.raster.path,
        benchmark_window_specs,  # type: ignore[arg-type]
    )
    total_cells = sum(_window_cell_count(window_spec) for window_spec in window_specs)
    return 0.0 if cells_per_second <= 0.0 else float(total_cells) / cells_per_second


def benchmark_source(input_path: str | Path, sample_step: int) -> dict[str, object]:
    started_at = time.perf_counter()
    context = _load_inspection_context(input_path, require_exact_max=False)
    validate_sample_step(int(sample_step), context.has_populated_stitch_tin)

    if sample_step <= 1:
        estimated_raster_seconds = _benchmark_full_raster_reads(context)
    else:
        estimated_raster_seconds = _benchmark_sparse_raster_reads(context, int(sample_step))

    sample_step_options = _build_sample_step_options(context)
    selected_option = next(
        (option for option in sample_step_options if int(option["value"]) == int(sample_step)),
        None,
    )
    if selected_option is None:
        estimated_size_bytes = _estimate_stl_upper_bound_bytes(
            context.raster.width,
            context.raster.height,
            int(sample_step),
            int(context.stitch_triangles.shape[0]),
        )
    else:
        estimated_size_bytes = int(selected_option["estimated_size_bytes"])

    estimated_total_triangles = max(0, (estimated_size_bytes - STL_HEADER_BYTES) // STL_TRIANGLE_BYTES)
    triangle_processing_rate = _benchmark_triangle_processing_rate()
    estimated_triangle_seconds = float(estimated_total_triangles) / triangle_processing_rate
    estimated_duration_seconds = max(0.1, estimated_raster_seconds + estimated_triangle_seconds)
    benchmark_elapsed_seconds = max(time.perf_counter() - started_at, 1e-6)

    return {
        "sample_step": int(sample_step),
        "estimated_duration_seconds": float(estimated_duration_seconds),
        "duration_estimate_kind": "benchmark",
        "benchmark_elapsed_seconds": float(benchmark_elapsed_seconds),
    }


def convert_source(
    input_path: str | Path,
    top_elevation: float,
    sample_step: int,
    output_path: str | Path | None = None,
    *,
    progress_callback: DesktopProgressCallback | None = None,
    log_callback: DesktopLogCallback | None = None,
) -> dict[str, object]:
    context, terrain_source = _resolve_conversion_context(
        input_path,
        int(sample_step),
        progress_callback=progress_callback,
        log_callback=log_callback,
    )

    if float(top_elevation) < context.terrain_max_elevation:
        raise TerrainConversionError(
            "The top elevation must be greater than or equal to the max terrain elevation. "
            "Reduce the extent of the terrain surface to just the area you would like "
            "to be made into an STL file, then rerun the converter."
        )

    surface = terrain_source.surface
    lookup = CoordinateLookup(surface.transform, surface.width, surface.height)
    _emit_progress(
        progress_callback,
        "validate-terrain",
        0,
        1,
        "Validating terrain metadata and stitch arrays...",
    )
    stitch_mesh = validate_stitch_mesh(
        context.stitch_points,
        context.stitch_triangles,
        surface,
        lookup,
    )
    _emit_progress(
        progress_callback,
        "validate-terrain",
        1,
        1,
        "Terrain validation complete.",
    )

    if stitch_mesh.point_count == 0:
        _emit_log(log_callback, "No populated stitch TIN was found. The raster will be converted by itself.")
    else:
        _emit_log(
            log_callback,
            "Validated stitch TIN: "
            f"{stitch_mesh.point_count} points, "
            f"{stitch_mesh.triangle_count} triangles, "
            f"{len(stitch_mesh.bridge_triangles)} non-grid bridge triangles.",
        )

    destination_path = resolve_output_path(
        terrain_source.source_path,
        None if output_path in (None, "") else str(output_path),
    )
    _emit_log(log_callback, f"Writing STL to {destination_path}...")
    result = convert_terrain_source(
        terrain_source=terrain_source,
        top_elevation=float(top_elevation),
        sample_step=int(sample_step),
        output_path=destination_path,
        stitch_mesh=stitch_mesh,
        progress_callback=progress_callback,
        log_callback=log_callback,
    )
    _emit_progress(progress_callback, "finalize", 0, 1, "Finalizing STL file...")
    stl_size_bytes = result.output_path.stat().st_size
    _emit_progress(progress_callback, "finalize", 1, 1, "Finalizing STL file...")
    _emit_progress(progress_callback, "complete", 1, 1, f"Finished writing {result.output_path.name}.")

    return {
        "output_path": str(result.output_path),
        "output_filename": result.output_path.name,
        "terrain_max_elevation": float(context.terrain_max_elevation),
        "resolved_raster_name": surface.path.name,
        "resolved_raster_path": str(surface.path),
        "triangle_count": int(result.triangle_count),
        "wall_triangle_count": int(result.wall_triangle_count),
        "stitch_point_count": int(stitch_mesh.point_count),
        "stitch_triangle_count": int(stitch_mesh.triangle_count),
        "stitch_bridge_triangle_count": int(len(stitch_mesh.bridge_triangles)),
        "stitch_component_count": int(len(stitch_mesh.components)),
        "stl_size_bytes": int(stl_size_bytes),
        "sample_step": int(sample_step),
        "top_elevation": float(top_elevation),
    }
