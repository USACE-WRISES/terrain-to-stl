from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import h5py
import numpy as np
import rasterio
from rasterio.transform import Affine


FLOAT_TOLERANCE = 1e-6
FACET_STRUCT = struct.Struct("<12fH")
SUPPORTED_STITCH_SAMPLE_STEPS = (1, 2, 4, 8, 16, 32)
ProgressCallback = Callable[[str, int, int, str], None]
LogCallback = Callable[[str], None]


class TerrainConversionError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ConversionRunResult:
    output_path: Path
    triangle_count: int
    wall_triangle_count: int


@dataclass(frozen=True, slots=True)
class RasterSurface:
    path: Path
    width: int
    height: int
    transform: Affine
    elevations: np.ndarray
    valid_mask: np.ndarray
    max_elevation: float

    def is_valid(self, row: int, col: int) -> bool:
        return bool(self.valid_mask[row, col])

    def elevation(self, row: int, col: int) -> float:
        return float(self.elevations[row, col])


@dataclass(frozen=True, slots=True)
class SparseRefinementTile:
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    elevations: np.ndarray
    valid_mask: np.ndarray

    def contains(self, row: int, col: int) -> bool:
        return (
            self.row_start <= row <= self.row_end
            and self.col_start <= col <= self.col_end
        )

    def is_valid(self, row: int, col: int) -> bool:
        local_row = row - self.row_start
        local_col = col - self.col_start
        return bool(self.valid_mask[local_row, local_col])

    def elevation(self, row: int, col: int) -> float:
        local_row = row - self.row_start
        local_col = col - self.col_start
        return float(self.elevations[local_row, local_col])


@dataclass(slots=True)
class SparseRasterSurface:
    path: Path
    width: int
    height: int
    transform: Affine
    sampled_rows: np.ndarray
    sampled_cols: np.ndarray
    coarse_elevations: np.ndarray
    coarse_valid_mask: np.ndarray
    refinement_tiles: tuple[SparseRefinementTile, ...]
    max_elevation: float
    _sampled_row_positions: dict[int, int] = field(init=False, repr=False)
    _sampled_col_positions: dict[int, int] = field(init=False, repr=False)
    _tiles_by_row: dict[int, tuple[SparseRefinementTile, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        expected_shape = (self.sampled_rows.shape[0], self.sampled_cols.shape[0])
        if self.coarse_elevations.shape != expected_shape or self.coarse_valid_mask.shape != expected_shape:
            raise TerrainConversionError(
                "The sparse raster surface coarse arrays do not match the sampled row and column counts."
            )

        self._sampled_row_positions = {
            int(row): index
            for index, row in enumerate(self.sampled_rows.tolist())
        }
        self._sampled_col_positions = {
            int(col): index
            for index, col in enumerate(self.sampled_cols.tolist())
        }

        tiles_by_row: dict[int, list[SparseRefinementTile]] = {}
        for tile in self.refinement_tiles:
            expected_tile_shape = (
                (tile.row_end - tile.row_start) + 1,
                (tile.col_end - tile.col_start) + 1,
            )
            if tile.elevations.shape != expected_tile_shape or tile.valid_mask.shape != expected_tile_shape:
                raise TerrainConversionError(
                    "The sparse raster surface refinement tile arrays do not match the declared tile bounds."
                )

            for row in range(tile.row_start, tile.row_end + 1):
                row_tiles = tiles_by_row.get(row)
                if row_tiles is None:
                    tiles_by_row[row] = [tile]
                else:
                    row_tiles.append(tile)

        self._tiles_by_row = {
            row: tuple(row_tiles)
            for row, row_tiles in tiles_by_row.items()
        }

    def _lookup_tile(self, row: int, col: int) -> SparseRefinementTile | None:
        for tile in self._tiles_by_row.get(row, ()):
            if tile.contains(row, col):
                return tile
        return None

    def is_valid(self, row: int, col: int) -> bool:
        tile = self._lookup_tile(row, col)
        if tile is not None:
            return tile.is_valid(row, col)

        row_index = self._sampled_row_positions.get(int(row))
        col_index = self._sampled_col_positions.get(int(col))
        if row_index is not None and col_index is not None:
            return bool(self.coarse_valid_mask[row_index, col_index])

        raise TerrainConversionError(
            f"Sparse raster surface is missing vertex data for row {row}, col {col}."
        )

    def elevation(self, row: int, col: int) -> float:
        tile = self._lookup_tile(row, col)
        if tile is not None:
            return tile.elevation(row, col)

        row_index = self._sampled_row_positions.get(int(row))
        col_index = self._sampled_col_positions.get(int(col))
        if row_index is not None and col_index is not None:
            return float(self.coarse_elevations[row_index, col_index])

        raise TerrainConversionError(
            f"Sparse raster surface is missing elevation data for row {row}, col {col}."
        )


@dataclass(frozen=True, slots=True)
class StitchMesh:
    point_count: int
    triangle_count: int
    bridge_triangles: tuple[tuple[int, int, int], ...]
    components: tuple["StitchComponent", ...]


@dataclass(frozen=True, slots=True)
class StitchComponent:
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    point_count: int
    triangle_count: int


@dataclass(frozen=True, slots=True)
class RefinementRegion:
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    top_outer_row: int | None
    bottom_outer_row: int | None
    left_outer_col: int | None
    right_outer_col: int | None


@dataclass(frozen=True, slots=True)
class TerrainSource:
    kind: str
    source_path: Path
    surface: RasterSurface
    terrain_max_elevation: float
    stitch_points: np.ndarray
    stitch_triangles: np.ndarray


SurfaceLike = RasterSurface | SparseRasterSurface


class CoordinateLookup:
    def __init__(self, transform: Affine, width: int, height: int) -> None:
        self.transform = transform
        self.width = width
        self.height = height
        self.axis_aligned = abs(transform.b) <= FLOAT_TOLERANCE and abs(transform.d) <= FLOAT_TOLERANCE
        self._x_by_col: np.ndarray | None = None
        self._y_by_row: np.ndarray | None = None

        if self.axis_aligned:
            cols = np.arange(width, dtype=np.float64) + 0.5
            rows = np.arange(height, dtype=np.float64) + 0.5
            self._x_by_col = transform.c + (transform.a * cols)
            self._y_by_row = transform.f + (transform.e * rows)

    def xy(self, row: int, col: int) -> tuple[float, float]:
        if self.axis_aligned:
            return float(self._x_by_col[col]), float(self._y_by_row[row])

        x, y = self.transform * (col + 0.5, row + 0.5)
        return float(x), float(y)


class BinaryStlWriter:
    def __init__(self, path: Path, header_text: str) -> None:
        self.path = path
        self._fh = path.open("wb")
        header = header_text.encode("ascii", "ignore")[:80].ljust(80, b"\0")
        self._fh.write(header)
        self._fh.write(struct.pack("<I", 0))
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

        scale = 1.0 / length
        self._fh.write(
            FACET_STRUCT.pack(
                float(nx * scale),
                float(ny * scale),
                float(nz * scale),
                float(p0[0]),
                float(p0[1]),
                float(p0[2]),
                float(p1[0]),
                float(p1[1]),
                float(p1[2]),
                float(p2[0]),
                float(p2[1]),
                float(p2[2]),
                0,
            )
        )
        self.triangle_count += 1
        return True

    def close(self) -> None:
        self._fh.seek(80)
        self._fh.write(struct.pack("<I", self.triangle_count))
        self._fh.close()


def resolve_existing_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def resolve_terrain_source_path(raw_path: str) -> Path:
    if not raw_path:
        raise TerrainConversionError("Please enter a path to a terrain HDF or DEM GeoTIFF file.")

    path = resolve_existing_path(raw_path)
    if not path.exists():
        raise TerrainConversionError(f"File not found: {path}")
    if not path.is_file():
        raise TerrainConversionError(f"Path is not a file: {path}")
    if path.suffix.lower() not in {".hdf", ".tif", ".tiff"}:
        raise TerrainConversionError("The input file must have a .hdf, .tif, or .tiff extension.")

    return path


def prompt_terrain_source_path() -> Path:
    while True:
        raw = input("Input terrain source path (.hdf, .tif, .tiff): ").strip().strip('"')
        try:
            return resolve_terrain_source_path(raw)
        except TerrainConversionError as exc:
            print(exc)


def prompt_top_elevation(max_elevation: float) -> float:
    prompt = (
        f"Top elevation for the STL shell "
        f"(terrain maximum elevation is {max_elevation:.6f}): "
    )

    while True:
        raw = input(prompt).strip()
        try:
            top_elevation = float(raw)
        except ValueError:
            print("Please enter a numeric elevation value.")
            continue

        if top_elevation < max_elevation:
            raise TerrainConversionError(
                "The top elevation must be greater than or equal to the max terrain elevation. "
                "Reduce the extent of the terrain surface to just the area you would like "
                "to be made into an STL file, then rerun the converter."
            )

        return top_elevation


def prompt_sample_step() -> int:
    while True:
        raw = input("Raster sample step (1, 2, 4, 8, ...): ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer sample step.")
            continue

        if value < 1:
            print("The sample step must be at least 1.")
            continue

        return value


def resolve_output_path(source_path: Path, raw_output_path: str | None) -> Path:
    if raw_output_path is None:
        return next_output_path(source_path)

    output_path = resolve_existing_path(raw_output_path)
    if output_path.suffix.lower() != ".stl":
        raise TerrainConversionError("The output path must use the .stl extension.")
    if not output_path.parent.exists():
        raise TerrainConversionError(f"Output folder does not exist: {output_path.parent}")

    return output_path


def resolve_raster_path(hdf_path: Path) -> Path:
    candidates = [hdf_path.with_suffix(".vrt"), hdf_path.with_suffix(".tif"), hdf_path.with_suffix(".tiff")]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise TerrainConversionError(
        f"Could not find a sibling raster beside {hdf_path.name}. "
        "The HEC terrain HDF does not contain the full terrain surface by itself. "
        f"Expected {hdf_path.with_suffix('.vrt').name}, {hdf_path.with_suffix('.tif').name}, "
        f"or {hdf_path.with_suffix('.tiff').name}."
    )


def next_output_path(source_path: Path) -> Path:
    base = source_path.with_suffix(".stl")
    if not base.exists():
        return base

    counter = 1
    while True:
        candidate = source_path.with_name(f"{source_path.stem}_{counter}.stl")
        if not candidate.exists():
            return candidate
        counter += 1


def load_raster_surface(raster_path: Path) -> RasterSurface:
    with rasterio.open(raster_path) as dataset:
        if dataset.count < 1:
            raise TerrainConversionError(f"{raster_path} does not contain any raster bands.")

        band = dataset.read(1, masked=True)
        mask = np.ma.getmaskarray(band)
        valid_mask = ~mask
        if not np.any(valid_mask):
            raise TerrainConversionError(f"{raster_path} does not contain any valid elevation cells.")

        elevations = band.filled(np.nan).astype(np.float32, copy=False)
        max_elevation = float(np.max(elevations[valid_mask]))

        return RasterSurface(
            path=raster_path,
            width=dataset.width,
            height=dataset.height,
            transform=dataset.transform,
            elevations=elevations,
            valid_mask=valid_mask,
            max_elevation=max_elevation,
        )


def load_hdf_max_elevation(hdf_path: Path) -> float | None:
    maxima: list[float] = []

    with h5py.File(hdf_path, "r") as hdf_file:
        def visit_dataset(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if not isinstance(obj, h5py.Dataset) or not name.endswith("Min-Max"):
                return
            if len(obj.shape) != 2 or obj.shape[1] < 2 or obj.shape[0] == 0:
                return

            maxima.append(float(np.max(obj[:, 1])))

        hdf_file.visititems(visit_dataset)

    if not maxima:
        return None

    return max(maxima)


def empty_stitch_arrays() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.int64),
    )


def load_stitch_arrays(hdf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(hdf_path, "r") as hdf_file:
        try:
            terrain_group = hdf_file["Terrain"]
            points_dataset = terrain_group["Stitch TIN Points"]
            triangles_dataset = terrain_group["Stitch TIN Triangles"]
        except KeyError:
            return empty_stitch_arrays()

        points = points_dataset[:]
        triangles = triangles_dataset[:]

    if points.size == 0 or triangles.size == 0:
        return empty_stitch_arrays()

    if len(points.shape) != 2 or points.shape[1] < 3:
        raise TerrainConversionError("The Stitch TIN Points dataset is not in the expected Nx4 shape.")
    if len(triangles.shape) != 2 or triangles.shape[1] != 3:
        raise TerrainConversionError("The Stitch TIN Triangles dataset is not in the expected Mx3 shape.")

    return points[:, :3].astype(np.float64, copy=False), triangles.astype(np.int64, copy=False)


def load_terrain_source(source_path: Path) -> TerrainSource:
    suffix = source_path.suffix.lower()
    if suffix == ".hdf":
        raster_path = resolve_raster_path(source_path)
        surface = load_raster_surface(raster_path)
        hdf_max_elevation = load_hdf_max_elevation(source_path)
        stitch_points, stitch_triangles = load_stitch_arrays(source_path)
        terrain_max_elevation = hdf_max_elevation if hdf_max_elevation is not None else surface.max_elevation
        return TerrainSource(
            kind="hdf",
            source_path=source_path,
            surface=surface,
            terrain_max_elevation=terrain_max_elevation,
            stitch_points=stitch_points,
            stitch_triangles=stitch_triangles,
        )

    if suffix not in {".tif", ".tiff"}:
        raise TerrainConversionError("Only .hdf, .tif, and .tiff terrain sources are supported.")

    surface = load_raster_surface(source_path)
    stitch_points, stitch_triangles = empty_stitch_arrays()
    return TerrainSource(
        kind="dem",
        source_path=source_path,
        surface=surface,
        terrain_max_elevation=surface.max_elevation,
        stitch_points=stitch_points,
        stitch_triangles=stitch_triangles,
    )


def convert_terrain_source(
    terrain_source: TerrainSource,
    top_elevation: float,
    sample_step: int,
    output_path: Path | None = None,
    stitch_mesh: StitchMesh | None = None,
    progress_callback: ProgressCallback | None = None,
    log_callback: LogCallback | None = print,
) -> ConversionRunResult:
    validate_sample_step(
        sample_step,
        terrain_source.stitch_points.size > 0 and terrain_source.stitch_triangles.size > 0,
    )

    surface = terrain_source.surface
    if stitch_mesh is None:
        lookup = CoordinateLookup(surface.transform, surface.width, surface.height)
        stitch_mesh = validate_stitch_mesh(
            terrain_source.stitch_points,
            terrain_source.stitch_triangles,
            surface,
            lookup,
        )
    destination_path = next_output_path(terrain_source.source_path) if output_path is None else output_path
    triangle_count, wall_triangle_count = write_shell_stl(
        output_path=destination_path,
        surface=surface,
        top_elevation=top_elevation,
        sample_step=sample_step,
        stitch_mesh=stitch_mesh,
        progress_callback=progress_callback,
        log_callback=log_callback,
    )
    return ConversionRunResult(
        output_path=destination_path,
        triangle_count=triangle_count,
        wall_triangle_count=wall_triangle_count,
    )


def build_sample_indices(size: int, step: int) -> list[int]:
    indices = list(range(0, size, step))
    if not indices or indices[-1] != size - 1:
        indices.append(size - 1)
    return indices


def validate_sample_step(sample_step: int, has_populated_stitch_tin: bool) -> None:
    if sample_step < 1:
        raise TerrainConversionError("The sample step must be at least 1.")

    if has_populated_stitch_tin and sample_step not in SUPPORTED_STITCH_SAMPLE_STEPS:
        supported_steps = ", ".join(str(step) for step in SUPPORTED_STITCH_SAMPLE_STEPS)
        raise TerrainConversionError(
            "This terrain contains populated stitch TIN data. "
            f"Supported stitch-aware raster sample steps are {supported_steps}. "
            "Choose one of those preset values."
        )


def encode_vertex(width: int, row: int, col: int) -> int:
    return (row * width) + col


def decode_vertex(width: int, vertex_id: int) -> tuple[int, int]:
    return divmod(vertex_id, width)


def triangle_is_native_grid(width: int, triangle: tuple[int, int, int]) -> bool:
    rows = sorted({vertex_id // width for vertex_id in triangle})
    cols = sorted({vertex_id % width for vertex_id in triangle})

    return (
        len(rows) == 2
        and len(cols) == 2
        and rows[1] - rows[0] == 1
        and cols[1] - cols[0] == 1
    )


def orient_triangle_up(
    width: int,
    lookup: CoordinateLookup,
    triangle: tuple[int, int, int],
) -> tuple[int, int, int] | None:
    a, b, c = triangle
    ar, ac = decode_vertex(width, a)
    br, bc = decode_vertex(width, b)
    cr, cc = decode_vertex(width, c)

    ax, ay = lookup.xy(ar, ac)
    bx, by = lookup.xy(br, bc)
    cx, cy = lookup.xy(cr, cc)

    signed_area = ((bx - ax) * (cy - ay)) - ((by - ay) * (cx - ax))
    if abs(signed_area) <= FLOAT_TOLERANCE:
        return None

    if signed_area < 0.0:
        return a, c, b
    return triangle


def map_stitch_points_to_vertex_ids(
    points: np.ndarray,
    width: int,
    height: int,
    transform: Affine,
) -> np.ndarray:
    inverse_transform = ~transform
    point_vertex_ids = np.empty(points.shape[0], dtype=np.int64)

    for index, (x, y, _z) in enumerate(points):
        col_f, row_f = inverse_transform * (float(x), float(y))
        col = int(round(col_f - 0.5))
        row = int(round(row_f - 0.5))

        if abs(col_f - (col + 0.5)) > FLOAT_TOLERANCE or abs(row_f - (row + 0.5)) > FLOAT_TOLERANCE:
            raise TerrainConversionError(
                "A stitch TIN point does not fall on a raster cell center. "
                "This terrain layout is not supported by this converter."
            )
        if row < 0 or row >= height or col < 0 or col >= width:
            raise TerrainConversionError("A stitch TIN point falls outside the raster extent.")

        point_vertex_ids[index] = encode_vertex(width, row, col)

    return point_vertex_ids


def validate_stitch_point_vertices(
    points: np.ndarray,
    point_vertex_ids: np.ndarray,
    surface: SurfaceLike,
) -> None:
    for index, (_x, _y, z) in enumerate(points):
        row, col = decode_vertex(surface.width, int(point_vertex_ids[index]))
        if not surface.is_valid(row, col):
            raise TerrainConversionError("A stitch TIN point references a raster nodata cell.")

        raster_z = surface.elevation(row, col)
        if abs(raster_z - float(z)) > FLOAT_TOLERANCE:
            raise TerrainConversionError(
                "A stitch TIN point elevation does not match the raster elevation at the same cell center."
            )


def normalize_stitch_triangle_indices(
    triangles: np.ndarray,
    point_count: int,
) -> tuple[tuple[int, int, int], ...]:
    normalized: list[tuple[int, int, int]] = []

    for triangle_index, triangle in enumerate(triangles):
        vertex_indices = tuple(int(idx) for idx in triangle)
        if len(vertex_indices) != 3:
            raise TerrainConversionError(f"Stitch triangle {triangle_index} is not a 3-vertex triangle.")
        if any(idx < 0 or idx >= point_count for idx in vertex_indices):
            raise TerrainConversionError(
                f"Stitch triangle {triangle_index} references a point index outside the Stitch TIN Points array."
            )

        normalized.append(vertex_indices)

    return tuple(normalized)


def build_stitch_components(
    width: int,
    point_vertex_ids: np.ndarray,
    triangles: tuple[tuple[int, int, int], ...],
) -> tuple[StitchComponent, ...]:
    if point_vertex_ids.size == 0 or not triangles:
        return ()

    point_to_triangles: list[list[int]] = [[] for _ in range(point_vertex_ids.shape[0])]
    for triangle_index, triangle in enumerate(triangles):
        for point_index in triangle:
            point_to_triangles[point_index].append(triangle_index)

    visited_points = np.zeros(point_vertex_ids.shape[0], dtype=bool)
    visited_triangles = np.zeros(len(triangles), dtype=bool)
    components: list[StitchComponent] = []

    for point_index, attached_triangles in enumerate(point_to_triangles):
        if visited_points[point_index] or not attached_triangles:
            continue

        stack = [point_index]
        component_points: list[int] = []
        component_triangle_count = 0

        while stack:
            current_point = stack.pop()
            if visited_points[current_point]:
                continue

            visited_points[current_point] = True
            component_points.append(current_point)

            for triangle_index in point_to_triangles[current_point]:
                if not visited_triangles[triangle_index]:
                    visited_triangles[triangle_index] = True
                    component_triangle_count += 1

                for neighbor_point in triangles[triangle_index]:
                    if not visited_points[neighbor_point]:
                        stack.append(neighbor_point)

        rows: list[int] = []
        cols: list[int] = []
        for component_point in component_points:
            row, col = decode_vertex(width, int(point_vertex_ids[component_point]))
            rows.append(row)
            cols.append(col)

        components.append(
            StitchComponent(
                min_row=min(rows),
                max_row=max(rows),
                min_col=min(cols),
                max_col=max(cols),
                point_count=len(component_points),
                triangle_count=component_triangle_count,
            )
        )

    return tuple(
        sorted(
            components,
            key=lambda component: (
                component.min_row,
                component.min_col,
                component.max_row,
                component.max_col,
            ),
        )
    )


def build_bridge_triangles(
    width: int,
    lookup: CoordinateLookup,
    point_vertex_ids: np.ndarray,
    triangles: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    bridge_triangles: list[tuple[int, int, int]] = []

    for triangle_index, triangle in enumerate(triangles):
        vertex_ids = tuple(int(point_vertex_ids[int(idx)]) for idx in triangle)
        if triangle_is_native_grid(width, vertex_ids):
            continue

        oriented = orient_triangle_up(width, lookup, vertex_ids)
        if oriented is None:
            raise TerrainConversionError(f"Stitch triangle {triangle_index} is degenerate.")

        bridge_triangles.append(oriented)

    return tuple(bridge_triangles)


def validate_stitch_mesh(
    points: np.ndarray,
    triangles: np.ndarray,
    surface: SurfaceLike,
    lookup: CoordinateLookup,
) -> StitchMesh:
    if points.size == 0 or triangles.size == 0:
        return StitchMesh(point_count=0, triangle_count=0, bridge_triangles=(), components=())

    point_vertex_ids = map_stitch_points_to_vertex_ids(
        points,
        surface.width,
        surface.height,
        surface.transform,
    )
    validate_stitch_point_vertices(points, point_vertex_ids, surface)
    normalized_triangles = normalize_stitch_triangle_indices(triangles, points.shape[0])
    components = build_stitch_components(surface.width, point_vertex_ids, normalized_triangles)
    bridge_triangles = build_bridge_triangles(
        surface.width,
        lookup,
        point_vertex_ids,
        normalized_triangles,
    )

    return StitchMesh(
        point_count=points.shape[0],
        triangle_count=triangles.shape[0],
        bridge_triangles=bridge_triangles,
        components=components,
    )


def update_boundary_edges(boundary_edges: dict[tuple[int, int], tuple[int, int]], a: int, b: int) -> None:
    key = (a, b) if a < b else (b, a)
    existing = boundary_edges.get(key)
    if existing is None:
        boundary_edges[key] = (a, b)
        return

    if existing == (b, a):
        del boundary_edges[key]
        return

    raise TerrainConversionError("Encountered a non-manifold mesh edge while building the STL shell.")


def iter_sampled_cell_triangles(
    surface: SurfaceLike,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
) -> Iterable[tuple[int, int, int]]:
    if row0 == row1 or col0 == col1:
        return

    v00 = surface.is_valid(row0, col0)
    v10 = surface.is_valid(row1, col0)
    v11 = surface.is_valid(row1, col1)
    v01 = surface.is_valid(row0, col1)

    a = encode_vertex(surface.width, row0, col0)
    b = encode_vertex(surface.width, row1, col0)
    c = encode_vertex(surface.width, row1, col1)
    d = encode_vertex(surface.width, row0, col1)

    if v00 and v10 and v11:
        yield (a, b, c)
    if v00 and v11 and v01:
        yield (a, c, d)


def sampled_floor(sampled_indices: list[int], value: int) -> int:
    index = max(0, bisect_right(sampled_indices, value) - 1)
    return sampled_indices[index]


def sampled_ceil(sampled_indices: list[int], value: int) -> int:
    index = bisect_left(sampled_indices, value)
    if index >= len(sampled_indices):
        index = len(sampled_indices) - 1
    return sampled_indices[index]


def raw_refinement_regions_touch(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    return not (
        left[1] < right[0]
        or right[1] < left[0]
        or left[3] < right[2]
        or right[3] < left[2]
    )


def merge_raw_refinement_regions(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return (
        min(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        max(left[3], right[3]),
    )


def build_refinement_regions(
    height: int,
    width: int,
    sample_step: int,
    components: tuple[StitchComponent, ...],
) -> tuple[RefinementRegion, ...]:
    if sample_step <= 1 or not components:
        return ()

    sampled_rows = build_sample_indices(height, sample_step)
    sampled_cols = build_sample_indices(width, sample_step)
    raw_regions: list[tuple[int, int, int, int]] = []

    for component in components:
        raw_regions.append(
            (
                sampled_floor(sampled_rows, component.min_row),
                sampled_ceil(sampled_rows, component.max_row),
                sampled_floor(sampled_cols, component.min_col),
                sampled_ceil(sampled_cols, component.max_col),
            )
        )

    merged_regions: list[tuple[int, int, int, int]] = []
    for raw_region in sorted(raw_regions):
        current_region = raw_region
        merged_index = 0
        while merged_index < len(merged_regions):
            existing_region = merged_regions[merged_index]
            if raw_refinement_regions_touch(existing_region, current_region):
                current_region = merge_raw_refinement_regions(existing_region, current_region)
                del merged_regions[merged_index]
                merged_index = 0
                continue
            merged_index += 1
        merged_regions.append(current_region)

    row_index_by_value = {value: index for index, value in enumerate(sampled_rows)}
    col_index_by_value = {value: index for index, value in enumerate(sampled_cols)}
    regions: list[RefinementRegion] = []

    for row_start, row_end, col_start, col_end in sorted(merged_regions):
        row_start_index = row_index_by_value[row_start]
        row_end_index = row_index_by_value[row_end]
        col_start_index = col_index_by_value[col_start]
        col_end_index = col_index_by_value[col_end]

        regions.append(
            RefinementRegion(
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                top_outer_row=sampled_rows[row_start_index - 1] if row_start_index > 0 else None,
                bottom_outer_row=sampled_rows[row_end_index + 1] if row_end_index + 1 < len(sampled_rows) else None,
                left_outer_col=sampled_cols[col_start_index - 1] if col_start_index > 0 else None,
                right_outer_col=sampled_cols[col_end_index + 1] if col_end_index + 1 < len(sampled_cols) else None,
            )
        )

    return tuple(regions)


def build_skipped_coarse_cells(
    sampled_rows: list[int],
    sampled_cols: list[int],
    regions: tuple[RefinementRegion, ...],
) -> set[tuple[int, int]]:
    if not regions:
        return set()

    row_index_by_value = {value: index for index, value in enumerate(sampled_rows)}
    col_index_by_value = {value: index for index, value in enumerate(sampled_cols)}
    skipped_cells: set[tuple[int, int]] = set()

    for region in regions:
        row_start_index = row_index_by_value[region.row_start]
        row_end_index = row_index_by_value[region.row_end]
        col_start_index = col_index_by_value[region.col_start]
        col_end_index = col_index_by_value[region.col_end]

        for row_pair_index in range(row_start_index, row_end_index):
            for col_pair_index in range(col_start_index, col_end_index):
                skipped_cells.add((row_pair_index, col_pair_index))

        if region.top_outer_row is not None:
            top_row_index = row_index_by_value[region.top_outer_row]
            for col_pair_index in range(col_start_index, col_end_index):
                skipped_cells.add((top_row_index, col_pair_index))

        if region.bottom_outer_row is not None:
            bottom_row_index = row_index_by_value[region.row_end]
            for col_pair_index in range(col_start_index, col_end_index):
                skipped_cells.add((bottom_row_index, col_pair_index))

        if region.left_outer_col is not None:
            left_col_index = col_index_by_value[region.left_outer_col]
            for row_pair_index in range(row_start_index, row_end_index):
                skipped_cells.add((row_pair_index, left_col_index))

        if region.right_outer_col is not None:
            right_col_index = col_index_by_value[region.col_end]
            for row_pair_index in range(row_start_index, row_end_index):
                skipped_cells.add((row_pair_index, right_col_index))

    return skipped_cells


def count_transition_strip_triangles(
    sampled_rows: list[int],
    sampled_cols: list[int],
    region: RefinementRegion,
) -> int:
    row_index_by_value = {value: index for index, value in enumerate(sampled_rows)}
    col_index_by_value = {value: index for index, value in enumerate(sampled_cols)}
    row_start_index = row_index_by_value[region.row_start]
    row_end_index = row_index_by_value[region.row_end]
    col_start_index = col_index_by_value[region.col_start]
    col_end_index = col_index_by_value[region.col_end]
    triangle_count = 0

    if region.top_outer_row is not None:
        for col_pair_index in range(col_start_index, col_end_index):
            triangle_count += (sampled_cols[col_pair_index + 1] - sampled_cols[col_pair_index]) + 1
    if region.bottom_outer_row is not None:
        for col_pair_index in range(col_start_index, col_end_index):
            triangle_count += (sampled_cols[col_pair_index + 1] - sampled_cols[col_pair_index]) + 1
    if region.left_outer_col is not None:
        for row_pair_index in range(row_start_index, row_end_index):
            triangle_count += (sampled_rows[row_pair_index + 1] - sampled_rows[row_pair_index]) + 1
    if region.right_outer_col is not None:
        for row_pair_index in range(row_start_index, row_end_index):
            triangle_count += (sampled_rows[row_pair_index + 1] - sampled_rows[row_pair_index]) + 1

    return triangle_count


def vertex_is_valid(surface: SurfaceLike, vertex_id: int) -> bool:
    row, col = decode_vertex(surface.width, vertex_id)
    return surface.is_valid(row, col)


def emit_progress(
    progress_callback: ProgressCallback | None,
    step: str,
    completed: int,
    total: int,
    message: str,
) -> None:
    if progress_callback is None:
        return

    safe_total = max(1, int(total))
    safe_completed = min(max(0, int(completed)), safe_total)
    progress_callback(step, safe_completed, safe_total, message)


def emit_log(log_callback: LogCallback | None, message: str) -> None:
    if log_callback is None:
        return

    log_callback(message)


def progress_interval(total: int) -> int:
    if total <= 0:
        return 1
    return max(1, total // 100)


def emit_shell_triangle(
    writer: BinaryStlWriter,
    boundary_edges: dict[tuple[int, int], tuple[int, int]],
    surface: SurfaceLike,
    lookup: CoordinateLookup,
    top_elevation: float,
    triangle: tuple[int, int, int],
) -> bool:
    a, b, c = triangle
    if len({a, b, c}) != 3:
        return False
    if not (
        vertex_is_valid(surface, a)
        and vertex_is_valid(surface, b)
        and vertex_is_valid(surface, c)
    ):
        return False

    oriented = orient_triangle_up(surface.width, lookup, triangle)
    if oriented is None:
        return False

    a, b, c = oriented
    update_boundary_edges(boundary_edges, a, b)
    update_boundary_edges(boundary_edges, b, c)
    update_boundary_edges(boundary_edges, c, a)

    writer.write_triangle(
        make_vertex_xyz(surface, lookup, a),
        make_vertex_xyz(surface, lookup, c),
        make_vertex_xyz(surface, lookup, b),
    )
    writer.write_triangle(
        make_vertex_xyz(surface, lookup, a, top_elevation),
        make_vertex_xyz(surface, lookup, b, top_elevation),
        make_vertex_xyz(surface, lookup, c, top_elevation),
    )
    return True


def triangulate_polygon_fan(polygon_vertices: list[int]) -> Iterable[tuple[int, int, int]]:
    if len(polygon_vertices) < 3:
        return

    anchor = polygon_vertices[0]
    for index in range(1, len(polygon_vertices) - 1):
        triangle = (anchor, polygon_vertices[index], polygon_vertices[index + 1])
        if len({triangle[0], triangle[1], triangle[2]}) == 3:
            yield triangle


def iter_transition_strip_triangles(
    width: int,
    sampled_rows: list[int],
    sampled_cols: list[int],
    region: RefinementRegion,
) -> Iterable[tuple[int, int, int]]:
    row_index_by_value = {value: index for index, value in enumerate(sampled_rows)}
    col_index_by_value = {value: index for index, value in enumerate(sampled_cols)}
    row_start_index = row_index_by_value[region.row_start]
    row_end_index = row_index_by_value[region.row_end]
    col_start_index = col_index_by_value[region.col_start]
    col_end_index = col_index_by_value[region.col_end]

    if region.top_outer_row is not None:
        outer_row = region.top_outer_row
        inner_row = region.row_start
        for col_pair_index in range(col_start_index, col_end_index):
            col0 = sampled_cols[col_pair_index]
            col1 = sampled_cols[col_pair_index + 1]
            inner_vertices = [encode_vertex(width, inner_row, col) for col in range(col0, col1 + 1)]
            polygon = [
                encode_vertex(width, outer_row, col0),
                encode_vertex(width, outer_row, col1),
                *reversed(inner_vertices),
            ]
            yield from triangulate_polygon_fan(polygon)

    if region.bottom_outer_row is not None:
        outer_row = region.bottom_outer_row
        inner_row = region.row_end
        for col_pair_index in range(col_start_index, col_end_index):
            col0 = sampled_cols[col_pair_index]
            col1 = sampled_cols[col_pair_index + 1]
            inner_vertices = [encode_vertex(width, inner_row, col) for col in range(col0, col1 + 1)]
            polygon = [
                encode_vertex(width, outer_row, col0),
                *inner_vertices,
                encode_vertex(width, outer_row, col1),
            ]
            yield from triangulate_polygon_fan(polygon)

    if region.left_outer_col is not None:
        outer_col = region.left_outer_col
        inner_col = region.col_start
        for row_pair_index in range(row_start_index, row_end_index):
            row0 = sampled_rows[row_pair_index]
            row1 = sampled_rows[row_pair_index + 1]
            inner_vertices = [encode_vertex(width, row, inner_col) for row in range(row0, row1 + 1)]
            polygon = [
                encode_vertex(width, row0, outer_col),
                *inner_vertices,
                encode_vertex(width, row1, outer_col),
            ]
            yield from triangulate_polygon_fan(polygon)

    if region.right_outer_col is not None:
        outer_col = region.right_outer_col
        inner_col = region.col_end
        for row_pair_index in range(row_start_index, row_end_index):
            row0 = sampled_rows[row_pair_index]
            row1 = sampled_rows[row_pair_index + 1]
            inner_vertices = [encode_vertex(width, row, inner_col) for row in range(row0, row1 + 1)]
            polygon = [
                encode_vertex(width, row0, outer_col),
                encode_vertex(width, row1, outer_col),
                *reversed(inner_vertices),
            ]
            yield from triangulate_polygon_fan(polygon)


def make_vertex_xyz(
    surface: SurfaceLike,
    lookup: CoordinateLookup,
    vertex_id: int,
    z_override: float | None = None,
) -> tuple[float, float, float]:
    row, col = decode_vertex(surface.width, vertex_id)
    x, y = lookup.xy(row, col)
    z = surface.elevation(row, col) if z_override is None else float(z_override)
    return x, y, z


def write_shell_stl(
    output_path: Path,
    surface: SurfaceLike,
    top_elevation: float,
    sample_step: int,
    stitch_mesh: StitchMesh,
    progress_callback: ProgressCallback | None = None,
    log_callback: LogCallback | None = print,
) -> tuple[int, int]:
    sampled_rows = build_sample_indices(surface.height, sample_step)
    sampled_cols = build_sample_indices(surface.width, sample_step)
    lookup = CoordinateLookup(surface.transform, surface.width, surface.height)
    boundary_edges: dict[tuple[int, int], tuple[int, int]] = {}
    writer = BinaryStlWriter(output_path, "Terrain to STL watertight shell")
    refinement_regions = build_refinement_regions(
        surface.height,
        surface.width,
        sample_step,
        stitch_mesh.components,
    )
    skipped_coarse_cells = build_skipped_coarse_cells(sampled_rows, sampled_cols, refinement_regions)
    coarse_cell_total = ((len(sampled_rows) - 1) * (len(sampled_cols) - 1)) - len(skipped_coarse_cells)
    fine_cell_total = sum(
        (region.row_end - region.row_start) * (region.col_end - region.col_start)
        for region in refinement_regions
    )
    strip_triangle_total = sum(
        count_transition_strip_triangles(sampled_rows, sampled_cols, region)
        for region in refinement_regions
    )
    total_surface_work = max(1, coarse_cell_total + fine_cell_total + strip_triangle_total)

    try:
        total_surface_triangles = 0
        completed_surface_work = 0
        surface_callback_interval = progress_interval(total_surface_work)
        coarse_row_pair_total = len(sampled_rows) - 1
        coarse_print_interval = max(1, coarse_row_pair_total // 20) if coarse_row_pair_total > 0 else 1

        emit_log(log_callback, "Writing bottom and top surfaces...")
        if refinement_regions:
            emit_log(
                log_callback,
                "Using adaptive stitch-aware sampling with "
                f"{len(refinement_regions)} local refinement region(s)."
            )
        emit_progress(
            progress_callback,
            "write-surfaces",
            0,
            total_surface_work,
            "Writing bottom and top STL surfaces...",
        )
        for row_pair_index in range(len(sampled_rows) - 1):
            row0 = sampled_rows[row_pair_index]
            row1 = sampled_rows[row_pair_index + 1]
            for col_pair_index in range(len(sampled_cols) - 1):
                completed_surface_work += 1
                if (row_pair_index, col_pair_index) in skipped_coarse_cells:
                    continue

                col0 = sampled_cols[col_pair_index]
                col1 = sampled_cols[col_pair_index + 1]
                for triangle in iter_sampled_cell_triangles(surface, row0, row1, col0, col1):
                    if emit_shell_triangle(
                        writer,
                        boundary_edges,
                        surface,
                        lookup,
                        top_elevation,
                        triangle,
                    ):
                        total_surface_triangles += 1

                if (
                    completed_surface_work == total_surface_work
                    or completed_surface_work == 1
                    or completed_surface_work % surface_callback_interval == 0
                ):
                    emit_progress(
                        progress_callback,
                        "write-surfaces",
                        completed_surface_work,
                        total_surface_work,
                        (
                            "Writing bottom and top STL surfaces "
                            f"({completed_surface_work}/{total_surface_work} work units)..."
                        ),
                    )

            if (
                coarse_row_pair_total > 0
                and ((row_pair_index + 1) % coarse_print_interval == 0 or row_pair_index + 1 == coarse_row_pair_total)
            ):
                emit_log(
                    log_callback,
                    f"  processed {row_pair_index + 1}/{coarse_row_pair_total} coarse raster row pairs",
                )

        if refinement_regions:
            region_print_interval = max(1, len(refinement_regions) // 20)
            for region_index, region in enumerate(refinement_regions):
                for row0 in range(region.row_start, region.row_end):
                    row1 = row0 + 1
                    for col0 in range(region.col_start, region.col_end):
                        col1 = col0 + 1
                        completed_surface_work += 1
                        for triangle in iter_sampled_cell_triangles(
                            surface,
                            row0,
                            row1,
                            col0,
                            col1,
                        ):
                            if emit_shell_triangle(
                                writer,
                                boundary_edges,
                                surface,
                                lookup,
                                top_elevation,
                                triangle,
                            ):
                                total_surface_triangles += 1

                        if (
                            completed_surface_work == total_surface_work
                            or completed_surface_work == 1
                            or completed_surface_work % surface_callback_interval == 0
                        ):
                            emit_progress(
                                progress_callback,
                                "write-surfaces",
                                completed_surface_work,
                                total_surface_work,
                                (
                                    "Writing bottom and top STL surfaces "
                                    f"({completed_surface_work}/{total_surface_work} work units)..."
                                ),
                            )

                for triangle in iter_transition_strip_triangles(
                    surface.width,
                    sampled_rows,
                    sampled_cols,
                    region,
                ):
                    completed_surface_work += 1
                    if emit_shell_triangle(
                        writer,
                        boundary_edges,
                        surface,
                        lookup,
                        top_elevation,
                        triangle,
                    ):
                        total_surface_triangles += 1

                    if (
                        completed_surface_work == total_surface_work
                        or completed_surface_work == 1
                        or completed_surface_work % surface_callback_interval == 0
                    ):
                        emit_progress(
                            progress_callback,
                            "write-surfaces",
                            completed_surface_work,
                            total_surface_work,
                            (
                                "Writing bottom and top STL surfaces "
                                f"({completed_surface_work}/{total_surface_work} work units)..."
                            ),
                        )

                if (region_index + 1) % region_print_interval == 0 or region_index + 1 == len(refinement_regions):
                    emit_log(
                        log_callback,
                        f"  refined {region_index + 1}/{len(refinement_regions)} stitch region(s)",
                    )

        if stitch_mesh.bridge_triangles:
            emit_log(log_callback, f"Adding {len(stitch_mesh.bridge_triangles)} stitch bridge triangles...")
            total_bridge_triangles = len(stitch_mesh.bridge_triangles)
            bridge_callback_interval = progress_interval(total_bridge_triangles)
            emit_progress(
                progress_callback,
                "write-stitches",
                0,
                total_bridge_triangles,
                f"Adding {total_bridge_triangles} stitch bridge triangles...",
            )
            for triangle_index, triangle in enumerate(stitch_mesh.bridge_triangles):
                if emit_shell_triangle(
                    writer,
                    boundary_edges,
                    surface,
                    lookup,
                    top_elevation,
                    triangle,
                ):
                    total_surface_triangles += 1

                completed_bridge_triangles = triangle_index + 1
                if (
                    completed_bridge_triangles == total_bridge_triangles
                    or completed_bridge_triangles == 1
                    or completed_bridge_triangles % bridge_callback_interval == 0
                ):
                    emit_progress(
                        progress_callback,
                        "write-stitches",
                        completed_bridge_triangles,
                        total_bridge_triangles,
                        (
                            "Adding stitch bridge triangles "
                            f"({completed_bridge_triangles}/{total_bridge_triangles})..."
                        ),
                    )

        emit_log(log_callback, f"Writing {len(boundary_edges)} boundary walls...")
        wall_triangle_count = 0
        total_boundary_edges = len(boundary_edges)
        boundary_callback_interval = progress_interval(total_boundary_edges)
        emit_progress(
            progress_callback,
            "write-walls",
            0,
            max(1, total_boundary_edges),
            f"Writing {total_boundary_edges} boundary walls...",
        )
        for boundary_index, (a, b) in enumerate(boundary_edges.values()):
            a_bottom = make_vertex_xyz(surface, lookup, a)
            b_bottom = make_vertex_xyz(surface, lookup, b)
            b_top = make_vertex_xyz(surface, lookup, b, top_elevation)
            a_top = make_vertex_xyz(surface, lookup, a, top_elevation)

            if writer.write_triangle(a_bottom, b_bottom, b_top):
                wall_triangle_count += 1
            if writer.write_triangle(a_bottom, b_top, a_top):
                wall_triangle_count += 1

            completed_boundary_edges = boundary_index + 1
            if (
                total_boundary_edges > 0
                and (
                    completed_boundary_edges == total_boundary_edges
                    or completed_boundary_edges == 1
                    or completed_boundary_edges % boundary_callback_interval == 0
                )
            ):
                emit_progress(
                    progress_callback,
                    "write-walls",
                    completed_boundary_edges,
                    total_boundary_edges,
                    (
                        "Writing boundary walls "
                        f"({completed_boundary_edges}/{total_boundary_edges})..."
                    ),
                )

        return writer.triangle_count, wall_triangle_count
    finally:
        writer.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an HEC-RAS terrain HDF or DEM GeoTIFF into a watertight STL shell.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        help="Path to the terrain source (.hdf, .tif, or .tiff). Prompts when omitted.",
    )
    parser.add_argument(
        "--top-elevation",
        type=float,
        help="Flat top elevation for the STL shell. Prompts when omitted.",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        help="Raster sample step (1, 2, 4, 8, ...). Prompts when omitted.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Optional output STL path. Defaults to the next sibling .stl beside the input terrain source.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_arg_parser().parse_args(sys.argv[1:] if argv is None else argv)
        source_path = (
            resolve_terrain_source_path(args.input_path)
            if args.input_path is not None
            else prompt_terrain_source_path()
        )
        terrain_source = load_terrain_source(source_path)

        print(f"Loading raster surface from {terrain_source.surface.path}...")
        if terrain_source.kind == "hdf":
            print(f"Reading terrain metadata from {terrain_source.source_path}...")
        else:
            print("No terrain HDF was provided. The DEM raster will be converted by itself.")

        top_elevation = (
            prompt_top_elevation(terrain_source.terrain_max_elevation)
            if args.top_elevation is None
            else args.top_elevation
        )
        if top_elevation < terrain_source.terrain_max_elevation:
            raise TerrainConversionError(
                "The top elevation must be greater than or equal to the max terrain elevation. "
                "Reduce the extent of the terrain surface to just the area you would like "
                "to be made into an STL file, then rerun the converter."
            )

        sample_step = prompt_sample_step() if args.sample_step is None else int(args.sample_step)
        validate_sample_step(
            sample_step,
            terrain_source.stitch_points.size > 0 and terrain_source.stitch_triangles.size > 0,
        )
        output_path = resolve_output_path(terrain_source.source_path, args.output_path)

        surface = terrain_source.surface
        lookup = CoordinateLookup(surface.transform, surface.width, surface.height)
        stitch_mesh = validate_stitch_mesh(
            terrain_source.stitch_points,
            terrain_source.stitch_triangles,
            surface,
            lookup,
        )
        if stitch_mesh.point_count == 0:
            print("No populated stitch TIN was found. The raster will be converted by itself.")
        else:
            print(
                "Validated stitch TIN: "
                f"{stitch_mesh.point_count} points, "
                f"{stitch_mesh.triangle_count} triangles, "
                f"{len(stitch_mesh.bridge_triangles)} non-grid bridge triangles"
            )

        print(f"Writing STL to {output_path}...")
        result = convert_terrain_source(
            terrain_source=terrain_source,
            top_elevation=top_elevation,
            sample_step=sample_step,
            output_path=output_path,
            stitch_mesh=stitch_mesh,
        )

        print(f"Finished writing {result.output_path}")
        print(f"Total STL triangles: {result.triangle_count}")
        print(f"Boundary wall triangles: {result.wall_triangle_count}")
        return 0
    except KeyboardInterrupt:
        print("\nConversion cancelled by user.")
        return 1
    except TerrainConversionError as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
