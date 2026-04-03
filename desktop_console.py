from __future__ import annotations

import argparse
import json
import sys

from desktop_backend import benchmark_source, convert_source, inspect_source, progress_percent
from terrain_to_stl import (
    SUPPORTED_STITCH_SAMPLE_STEPS,
    TerrainConversionError,
    prompt_sample_step,
    prompt_terrain_source_path,
    prompt_top_elevation,
)


def emit_json(record: dict[str, object]) -> None:
    print(json.dumps(record, separators=(",", ":")), flush=True)


class PlainProgressReporter:
    def __init__(self) -> None:
        self._last_message: str | None = None

    def __call__(self, step: str, completed: int, total: int, message: str) -> None:
        percent = progress_percent(step, completed, total)
        rendered = f"[{percent:>3}%] {message}"
        if rendered == self._last_message:
            return

        self._last_message = rendered
        print(rendered, flush=True)


def print_inspection(inspection: dict[str, object]) -> None:
    print(f"Terrain source: {inspection['source_path']}")
    print(f"Terrain type: {inspection['source_kind']}")
    print(f"Resolved raster: {inspection['resolved_raster_path']}")
    print(f"Raster size: {inspection['raster_width']} columns x {inspection['raster_height']} rows")
    print(f"Terrain max elevation: {inspection['terrain_max_elevation']:.6f}")
    print(f"Stitch points: {inspection['stitch_point_count']}")
    print(f"Stitch triangles: {inspection['stitch_triangle_count']}")
    print(f"Stitch bridge triangles: {inspection['stitch_bridge_triangle_count']}")
    print(f"Suggested sample steps: {', '.join(str(step) for step in inspection['suggested_sample_steps'])}")
    sample_step_options = inspection.get("sample_step_options")
    if isinstance(sample_step_options, list) and sample_step_options:
        print("Sample step estimates:")
        for option in sample_step_options:
            if not isinstance(option, dict):
                continue
            estimated_size_bytes = option.get("estimated_size_bytes")
            duration = option.get("estimated_duration_seconds")
            duration_kind = option.get("duration_estimate_kind", "pending")
            size_text = (
                "unknown"
                if estimated_size_bytes is None
                else f"up to {int(estimated_size_bytes):,} bytes"
            )
            if duration is None:
                duration_text = "time estimate pending"
            else:
                duration_text = f"about {float(duration):.1f} seconds ({duration_kind})"
            print(f"  step {option['value']}: {size_text}; {duration_text}")
    if inspection["sample_step_requires_preset"]:
        supported = ", ".join(str(step) for step in SUPPORTED_STITCH_SAMPLE_STEPS)
        print(f"Sample step rule: choose one of {supported}")
    else:
        print("Sample step rule: any integer 1 or greater is allowed")
    print(f"Default output path: {inspection['default_output_path']}")


def print_conversion_result(result: dict[str, object]) -> None:
    print(f"Output STL: {result['output_path']}")
    print(f"Resolved raster: {result['resolved_raster_path']}")
    print(f"Total STL triangles: {result['triangle_count']}")
    print(f"Boundary wall triangles: {result['wall_triangle_count']}")
    print(f"Stitch points: {result['stitch_point_count']}")
    print(f"Stitch triangles: {result['stitch_triangle_count']}")
    print(f"Stitch bridge triangles: {result['stitch_bridge_triangle_count']}")
    print(f"STL size (bytes): {result['stl_size_bytes']}")


def print_benchmark_result(benchmark: dict[str, object]) -> None:
    print(f"Sample step: {benchmark['sample_step']}")
    print(f"Estimated duration (seconds): {float(benchmark['estimated_duration_seconds']):.3f}")
    print(f"Estimate kind: {benchmark['duration_estimate_kind']}")
    print(f"Benchmark elapsed (seconds): {float(benchmark['benchmark_elapsed_seconds']):.3f}")


def prompt_sample_step_for_inspection(inspection: dict[str, object]) -> int:
    while True:
        value = prompt_sample_step()
        if inspection["sample_step_requires_preset"] and value not in SUPPORTED_STITCH_SAMPLE_STEPS:
            supported = ", ".join(str(step) for step in SUPPORTED_STITCH_SAMPLE_STEPS)
            print(f"Supported stitch-aware raster sample steps are {supported}.")
            continue
        return value


def prompt_output_path(default_output_path: str) -> str | None:
    raw = input(f'Output STL path [{default_output_path}]: ').strip().strip('"')
    if not raw:
        return None
    return raw


def run_interactive() -> int:
    try:
        source_path = prompt_terrain_source_path()
        inspection = inspect_source(source_path)
        print_inspection(inspection)

        while True:
            try:
                top_elevation = prompt_top_elevation(float(inspection["terrain_max_elevation"]))
                break
            except TerrainConversionError as exc:
                print(exc)

        sample_step = prompt_sample_step_for_inspection(inspection)
        output_path = prompt_output_path(str(inspection["default_output_path"]))
        result = convert_source(
            source_path,
            top_elevation,
            sample_step,
            output_path=output_path,
            progress_callback=PlainProgressReporter(),
            log_callback=lambda message: print(message, flush=True),
        )
        print_conversion_result(result)
        return 0
    except KeyboardInterrupt:
        print("\nConversion cancelled by user.")
        return 1
    except TerrainConversionError as exc:
        print(f"Error: {exc}")
        return 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Structured console backend and interactive converter for the Terrain to STL desktop workflow.",
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a terrain source and print metadata.")
    inspect_parser.add_argument("--input", dest="input_path", required=True, help="Path to the terrain source.")
    inspect_parser.add_argument(
        "--json-stream",
        action="store_true",
        help="Emit machine-readable JSON output for desktop GUI integration.",
    )

    convert_parser = subparsers.add_parser("convert", help="Convert a terrain source to STL.")
    convert_parser.add_argument("--input", dest="input_path", required=True, help="Path to the terrain source.")
    convert_parser.add_argument("--top-elevation", type=float, required=True, help="Flat top elevation for the STL shell.")
    convert_parser.add_argument("--sample-step", type=int, required=True, help="Raster sample step.")
    convert_parser.add_argument("--output", dest="output_path", help="Optional output STL path.")
    convert_parser.add_argument(
        "--json-stream",
        action="store_true",
        help="Emit machine-readable JSON progress and result records for desktop GUI integration.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Estimate conversion time for a specific sample step.",
    )
    benchmark_parser.add_argument("--input", dest="input_path", required=True, help="Path to the terrain source.")
    benchmark_parser.add_argument("--sample-step", type=int, required=True, help="Raster sample step.")
    benchmark_parser.add_argument(
        "--json-stream",
        action="store_true",
        help="Emit machine-readable JSON benchmark output for desktop GUI integration.",
    )

    return parser


def run_inspect(args: argparse.Namespace) -> int:
    inspection = inspect_source(args.input_path)
    if args.json_stream:
        emit_json({"type": "inspect", "inspection": inspection})
    else:
        print_inspection(inspection)
    return 0


def run_convert(args: argparse.Namespace) -> int:
    if args.json_stream:
        result = convert_source(
            args.input_path,
            args.top_elevation,
            args.sample_step,
            output_path=args.output_path,
            progress_callback=lambda step, completed, total, message: emit_json(
                {
                    "type": "progress",
                    "step": step,
                    "completed": completed,
                    "total": total,
                    "percent": progress_percent(step, completed, total),
                    "message": message,
                }
            ),
            log_callback=None,
        )
        emit_json({"type": "result", "result": result})
        return 0

    result = convert_source(
        args.input_path,
        args.top_elevation,
        args.sample_step,
        output_path=args.output_path,
        progress_callback=PlainProgressReporter(),
        log_callback=lambda message: print(message, flush=True),
    )
    print_conversion_result(result)
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    benchmark = benchmark_source(args.input_path, args.sample_step)
    if args.json_stream:
        emit_json({"type": "benchmark", "benchmark": benchmark})
    else:
        print_benchmark_result(benchmark)
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] not in {"inspect", "convert", "benchmark"} and raw_argv[0].startswith("-"):
        raw_argv = ["convert", *raw_argv]

    parser = build_arg_parser()
    args = parser.parse_args(raw_argv)

    try:
        if args.command is None:
            return run_interactive()
        if args.command == "inspect":
            return run_inspect(args)
        if args.command == "convert":
            return run_convert(args)
        if args.command == "benchmark":
            return run_benchmark(args)
        parser.error(f"Unsupported command: {args.command}")
        return 2
    except KeyboardInterrupt:
        message = "Operation cancelled by user."
        if getattr(args, "json_stream", False):
            emit_json({"type": "error", "message": message})
        else:
            print(f"\n{message}")
        return 1
    except TerrainConversionError as exc:
        if getattr(args, "json_stream", False):
            emit_json({"type": "error", "message": str(exc)})
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
