from __future__ import annotations

import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import rasterio
from rasterio.transform import Affine


FLOAT_TOLERANCE = 1e-6
FACET_STRUCT = struct.Struct("<12fH")


class TerrainConversionError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class RasterSurface:
    path: Path
    width: int
    height: int
    transform: Affine
    elevations: np.ndarray
    valid_mask: np.ndarray
    max_elevation: float


@dataclass(frozen=True, slots=True)
class StitchMesh:
    point_count: int
    triangle_count: int
    bridge_triangles: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class TerrainSource:
    kind: str
    source_path: Path
    surface: RasterSurface
    terrain_max_elevation: float
    stitch_points: np.ndarray
    stitch_triangles: np.ndarray


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


def prompt_terrain_source_path() -> Path:
    while True:
        raw = input("Input terrain source path (.hdf, .tif, .tiff): ").strip().strip('"')
        if not raw:
            print("Please enter a path to a terrain HDF or DEM GeoTIFF file.")
            continue

        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()

        if not path.exists():
            print(f"File not found: {path}")
            continue
        if not path.is_file():
            print(f"Path is not a file: {path}")
            continue
        if path.suffix.lower() not in {".hdf", ".tif", ".tiff"}:
            print("The input file must have a .hdf, .tif, or .tiff extension.")
            continue

        return path


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


def build_sample_indices(size: int, step: int) -> list[int]:
    indices = list(range(0, size, step))
    if not indices or indices[-1] != size - 1:
        indices.append(size - 1)
    return indices


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


def validate_stitch_mesh(
    points: np.ndarray,
    triangles: np.ndarray,
    surface: RasterSurface,
    lookup: CoordinateLookup,
) -> StitchMesh:
    if points.size == 0 or triangles.size == 0:
        return StitchMesh(point_count=0, triangle_count=0, bridge_triangles=())

    inverse_transform = ~surface.transform
    point_vertex_ids = np.empty(points.shape[0], dtype=np.int64)

    for index, (x, y, z) in enumerate(points):
        col_f, row_f = inverse_transform * (float(x), float(y))
        col = int(round(col_f - 0.5))
        row = int(round(row_f - 0.5))

        if abs(col_f - (col + 0.5)) > FLOAT_TOLERANCE or abs(row_f - (row + 0.5)) > FLOAT_TOLERANCE:
            raise TerrainConversionError(
                "A stitch TIN point does not fall on a raster cell center. "
                "This terrain layout is not supported by this converter."
            )
        if row < 0 or row >= surface.height or col < 0 or col >= surface.width:
            raise TerrainConversionError("A stitch TIN point falls outside the raster extent.")
        if not surface.valid_mask[row, col]:
            raise TerrainConversionError("A stitch TIN point references a raster nodata cell.")

        raster_z = float(surface.elevations[row, col])
        if abs(raster_z - float(z)) > FLOAT_TOLERANCE:
            raise TerrainConversionError(
                "A stitch TIN point elevation does not match the raster elevation at the same cell center."
            )

        point_vertex_ids[index] = encode_vertex(surface.width, row, col)

    bridge_triangles: list[tuple[int, int, int]] = []
    for triangle_index, triangle in enumerate(triangles):
        if np.any(triangle < 0) or np.any(triangle >= points.shape[0]):
            raise TerrainConversionError(
                f"Stitch triangle {triangle_index} references a point index outside the Stitch TIN Points array."
            )

        vertex_ids = tuple(int(point_vertex_ids[int(idx)]) for idx in triangle)
        if triangle_is_native_grid(surface.width, vertex_ids):
            continue

        oriented = orient_triangle_up(surface.width, lookup, vertex_ids)
        if oriented is None:
            raise TerrainConversionError(f"Stitch triangle {triangle_index} is degenerate.")

        bridge_triangles.append(oriented)

    return StitchMesh(
        point_count=points.shape[0],
        triangle_count=triangles.shape[0],
        bridge_triangles=tuple(bridge_triangles),
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


def iter_raster_triangles_by_row_pair(
    width: int,
    valid_mask: np.ndarray,
    sampled_rows: list[int],
    sampled_cols: list[int],
) -> Iterable[tuple[int, list[tuple[int, int, int]]]]:
    for row_pair_index in range(len(sampled_rows) - 1):
        row0 = sampled_rows[row_pair_index]
        row1 = sampled_rows[row_pair_index + 1]
        valid0 = valid_mask[row0]
        valid1 = valid_mask[row1]
        row_pair_triangles: list[tuple[int, int, int]] = []

        for col_index in range(len(sampled_cols) - 1):
            col0 = sampled_cols[col_index]
            col1 = sampled_cols[col_index + 1]

            v00 = bool(valid0[col0])
            v10 = bool(valid1[col0])
            v11 = bool(valid1[col1])
            v01 = bool(valid0[col1])

            a = encode_vertex(width, row0, col0)
            b = encode_vertex(width, row1, col0)
            c = encode_vertex(width, row1, col1)
            d = encode_vertex(width, row0, col1)

            if v00 and v10 and v11:
                row_pair_triangles.append((a, b, c))
            if v00 and v11 and v01:
                row_pair_triangles.append((a, c, d))

        yield row_pair_index, row_pair_triangles


def make_vertex_xyz(
    width: int,
    lookup: CoordinateLookup,
    elevations: np.ndarray,
    vertex_id: int,
    z_override: float | None = None,
) -> tuple[float, float, float]:
    row, col = decode_vertex(width, vertex_id)
    x, y = lookup.xy(row, col)
    z = float(elevations[row, col]) if z_override is None else float(z_override)
    return x, y, z


def write_shell_stl(
    output_path: Path,
    surface: RasterSurface,
    top_elevation: float,
    sample_step: int,
    stitch_mesh: StitchMesh,
) -> tuple[int, int]:
    sampled_rows = build_sample_indices(surface.height, sample_step)
    sampled_cols = build_sample_indices(surface.width, sample_step)
    lookup = CoordinateLookup(surface.transform, surface.width, surface.height)
    boundary_edges: dict[tuple[int, int], tuple[int, int]] = {}
    writer = BinaryStlWriter(output_path, "Terrain to STL watertight shell")

    try:
        total_surface_triangles = 0
        total_row_pairs = len(sampled_rows) - 1
        progress_interval = max(1, total_row_pairs // 20) if total_row_pairs > 0 else 1

        print("Writing bottom and top surfaces...")
        for row_pair_index, row_pair_triangles in iter_raster_triangles_by_row_pair(
            surface.width,
            surface.valid_mask,
            sampled_rows,
            sampled_cols,
        ):
            for a, b, c in row_pair_triangles:
                update_boundary_edges(boundary_edges, a, b)
                update_boundary_edges(boundary_edges, b, c)
                update_boundary_edges(boundary_edges, c, a)

                writer.write_triangle(
                    make_vertex_xyz(surface.width, lookup, surface.elevations, a),
                    make_vertex_xyz(surface.width, lookup, surface.elevations, c),
                    make_vertex_xyz(surface.width, lookup, surface.elevations, b),
                )
                writer.write_triangle(
                    make_vertex_xyz(surface.width, lookup, surface.elevations, a, top_elevation),
                    make_vertex_xyz(surface.width, lookup, surface.elevations, b, top_elevation),
                    make_vertex_xyz(surface.width, lookup, surface.elevations, c, top_elevation),
                )
                total_surface_triangles += 1

            if (
                total_row_pairs > 0
                and ((row_pair_index + 1) % progress_interval == 0 or row_pair_index + 1 == total_row_pairs)
            ):
                print(f"  processed {row_pair_index + 1}/{total_row_pairs} raster row pairs")

        if stitch_mesh.bridge_triangles:
            print(f"Adding {len(stitch_mesh.bridge_triangles)} stitch bridge triangles...")
            for triangle in stitch_mesh.bridge_triangles:
                a, b, c = triangle
                update_boundary_edges(boundary_edges, a, b)
                update_boundary_edges(boundary_edges, b, c)
                update_boundary_edges(boundary_edges, c, a)

                writer.write_triangle(
                    make_vertex_xyz(surface.width, lookup, surface.elevations, a),
                    make_vertex_xyz(surface.width, lookup, surface.elevations, c),
                    make_vertex_xyz(surface.width, lookup, surface.elevations, b),
                )
                writer.write_triangle(
                    make_vertex_xyz(surface.width, lookup, surface.elevations, a, top_elevation),
                    make_vertex_xyz(surface.width, lookup, surface.elevations, b, top_elevation),
                    make_vertex_xyz(surface.width, lookup, surface.elevations, c, top_elevation),
                )
                total_surface_triangles += 1

        print(f"Writing {len(boundary_edges)} boundary walls...")
        wall_triangle_count = 0
        for a, b in boundary_edges.values():
            a_bottom = make_vertex_xyz(surface.width, lookup, surface.elevations, a)
            b_bottom = make_vertex_xyz(surface.width, lookup, surface.elevations, b)
            b_top = make_vertex_xyz(surface.width, lookup, surface.elevations, b, top_elevation)
            a_top = make_vertex_xyz(surface.width, lookup, surface.elevations, a, top_elevation)

            if writer.write_triangle(a_bottom, b_bottom, b_top):
                wall_triangle_count += 1
            if writer.write_triangle(a_bottom, b_top, a_top):
                wall_triangle_count += 1

        return writer.triangle_count, wall_triangle_count
    finally:
        writer.close()


def main() -> int:
    try:
        source_path = prompt_terrain_source_path()
        terrain_source = load_terrain_source(source_path)

        print(f"Loading raster surface from {terrain_source.surface.path}...")
        if terrain_source.kind == "hdf":
            print(f"Reading terrain metadata from {terrain_source.source_path}...")
        else:
            print("No terrain HDF was provided. The DEM raster will be converted by itself.")

        top_elevation = prompt_top_elevation(terrain_source.terrain_max_elevation)
        sample_step = prompt_sample_step()

        if (
            terrain_source.stitch_points.size > 0
            and terrain_source.stitch_triangles.size > 0
            and sample_step != 1
        ):
            raise TerrainConversionError(
                "This terrain contains populated stitch TIN data. "
                "This converter only supports stitch-aware output at raster sample step 1. "
                "Reduce or crop the terrain externally and rerun with sample step 1."
            )

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

        output_path = next_output_path(terrain_source.source_path)
        print(f"Writing STL to {output_path}...")
        triangle_count, wall_triangle_count = write_shell_stl(
            output_path=output_path,
            surface=surface,
            top_elevation=top_elevation,
            sample_step=sample_step,
            stitch_mesh=stitch_mesh,
        )

        print(f"Finished writing {output_path}")
        print(f"Total STL triangles: {triangle_count}")
        print(f"Boundary wall triangles: {wall_triangle_count}")
        return 0
    except KeyboardInterrupt:
        print("\nConversion cancelled by user.")
        return 1
    except TerrainConversionError as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
