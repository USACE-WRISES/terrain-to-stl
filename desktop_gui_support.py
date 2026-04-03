from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DEFAULT_VERTICAL_EXAGGERATION = 3
VERTICAL_EXAGGERATION_OPTIONS = (1, 2, 3, 5, 10)
_DEM_SUFFIXES = {".tif", ".tiff"}
_RASTER_SUFFIXES = {".vrt", *_DEM_SUFFIXES}
_SUPPORTED_SELECTION_SUFFIXES = {".hdf", *_RASTER_SUFFIXES}


@dataclass(frozen=True, slots=True)
class TerrainSelectionResult:
    selected_paths: tuple[Path, ...]
    primary_path: Path | None
    primary_kind: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class AssociatedTerrainFile:
    path: Path
    role: str
    selected: bool
    used_for_conversion: bool


def normalize_desktop_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def resolve_desktop_terrain_selection(paths: Sequence[str | Path]) -> TerrainSelectionResult:
    selected_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for raw_path in paths:
        if not str(raw_path).strip():
            continue
        path = normalize_desktop_path(raw_path)
        if path in seen_paths:
            continue
        seen_paths.add(path)
        selected_paths.append(path)

    if not selected_paths:
        return TerrainSelectionResult(tuple(), None, None, None)

    unsupported = [path for path in selected_paths if path.suffix.lower() not in _SUPPORTED_SELECTION_SUFFIXES]
    if unsupported:
        names = ", ".join(path.name for path in unsupported)
        return TerrainSelectionResult(
            tuple(selected_paths),
            None,
            None,
            f"Supported terrain selections are .hdf, .vrt, .tif, and .tiff. Unsupported: {names}",
        )

    hdf_paths = [path for path in selected_paths if path.suffix.lower() == ".hdf"]
    dem_paths = [path for path in selected_paths if path.suffix.lower() in _DEM_SUFFIXES]
    vrt_paths = [path for path in selected_paths if path.suffix.lower() == ".vrt"]

    if len(hdf_paths) > 1:
        names = ", ".join(path.name for path in hdf_paths)
        return TerrainSelectionResult(
            tuple(selected_paths),
            None,
            None,
            f"Select only one terrain HDF at a time. Found: {names}",
        )

    if hdf_paths:
        return TerrainSelectionResult(tuple(selected_paths), hdf_paths[0], "hdf", None)

    if len(dem_paths) > 1:
        return TerrainSelectionResult(
            tuple(selected_paths),
            None,
            None,
            "When no terrain HDF is selected, choose exactly one DEM GeoTIFF (.tif or .tiff).",
        )

    if len(dem_paths) == 1:
        return TerrainSelectionResult(tuple(selected_paths), dem_paths[0], "dem", None)

    if vrt_paths:
        return TerrainSelectionResult(
            tuple(selected_paths),
            None,
            None,
            "A .vrt file must be selected with its terrain HDF. Select one HDF plus its associated raster files, or select one DEM GeoTIFF.",
        )

    return TerrainSelectionResult(
        tuple(selected_paths),
        None,
        None,
        "Select a terrain HDF with its raster files, or a standalone DEM GeoTIFF.",
    )


def build_associated_file_entries(
    selection: TerrainSelectionResult,
    inspection: dict[str, object] | None = None,
) -> tuple[list[AssociatedTerrainFile], str | None]:
    entries: list[AssociatedTerrainFile] = []
    unused_count = 0
    selected_lookup = set(selection.selected_paths)
    resolved_raster_path: Path | None = None
    if inspection is not None and inspection.get("resolved_raster_path"):
        resolved_raster_path = normalize_desktop_path(str(inspection["resolved_raster_path"]))

    for path in selection.selected_paths:
        if selection.primary_path is not None and path == selection.primary_path:
            entries.append(AssociatedTerrainFile(path, "Primary terrain", True, True))
            continue

        if resolved_raster_path is not None and path == resolved_raster_path:
            entries.append(AssociatedTerrainFile(path, "Resolved raster", True, True))
            continue

        if inspection is None:
            entries.append(AssociatedTerrainFile(path, "Selected association candidate", True, False))
            continue

        unused_count += 1
        entries.append(AssociatedTerrainFile(path, "Selected but unused", True, False))

    if (
        inspection is not None
        and resolved_raster_path is not None
        and resolved_raster_path not in selected_lookup
        and selection.primary_kind == "hdf"
    ):
        entries.append(AssociatedTerrainFile(resolved_raster_path, "Resolved raster (found automatically)", False, True))

    warning: str | None = None
    if unused_count > 0 and selection.primary_kind == "hdf":
        warning = (
            "Extra selected raster files are shown below, but conversion uses the resolved sibling raster only."
        )
    elif unused_count > 0:
        warning = "Extra selected files are shown below, but only the primary terrain input will be used for conversion."

    return entries, warning


def scene_point_from_world(
    point: Sequence[float],
    center_z: float,
    vertical_exaggeration: float,
) -> tuple[float, float, float]:
    x, y, z = point
    factor = max(float(vertical_exaggeration), 1e-6)
    return float(x), float(y), center_z + ((float(z) - center_z) * factor)


def world_point_from_scene(
    point: Sequence[float],
    center_z: float,
    vertical_exaggeration: float,
) -> tuple[float, float, float]:
    x, y, z = point
    factor = max(float(vertical_exaggeration), 1e-6)
    return float(x), float(y), center_z + ((float(z) - center_z) / factor)


def scene_bounds_from_world(
    bounds: Sequence[float],
    center_z: float,
    vertical_exaggeration: float,
) -> tuple[float, float, float, float, float, float]:
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    scene_min = scene_point_from_world((min_x, min_y, min_z), center_z, vertical_exaggeration)
    scene_max = scene_point_from_world((max_x, max_y, max_z), center_z, vertical_exaggeration)
    return (
        float(min_x),
        float(max_x),
        float(min_y),
        float(max_y),
        float(scene_min[2]),
        float(scene_max[2]),
    )


def _normalize_vector(vector: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in vector)
    length = math.sqrt((x * x) + (y * y) + (z * z))
    if length <= 1e-12:
        return 0.0, 0.0, 1.0
    return x / length, y / length, z / length


def scene_normal_from_world(normal: Sequence[float], vertical_exaggeration: float) -> tuple[float, float, float]:
    factor = max(float(vertical_exaggeration), 1e-6)
    x, y, z = (float(value) for value in normal)
    return _normalize_vector((x, y, z / factor))


def world_normal_from_scene(normal: Sequence[float], vertical_exaggeration: float) -> tuple[float, float, float]:
    factor = max(float(vertical_exaggeration), 1e-6)
    x, y, z = (float(value) for value in normal)
    return _normalize_vector((x, y, z * factor))
