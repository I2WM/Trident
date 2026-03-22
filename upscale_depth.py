#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.name}_depth_x4")


def _discover_scene_roots(input_path: Path) -> tuple[list[Path], bool]:
    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise SystemExit(f"Input path is not a directory: {input_path}")

    if input_path.name == "depth":
        return [input_path.parent], False

    if (input_path / "depth").is_dir():
        return [input_path], False

    scene_roots = sorted(child for child in input_path.iterdir() if child.is_dir() and (child / "depth").is_dir())
    if not scene_roots:
        raise SystemExit(
            f"No valid scene roots were found under {input_path}. "
            "Expected either a single scene directory with depth/, or a dataset root containing multiple scenes."
        )
    return scene_roots, True


def _ensure_safe_output(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("Input and output directories must be different.")

    try:
        output_path.resolve().relative_to(input_path.resolve())
    except ValueError:
        return
    raise SystemExit("Output directory must not be inside the input directory.")


def _upscale_one(
    *,
    input_file: Path,
    output_file: Path,
    scale: int,
    overwrite: bool,
) -> str:
    if output_file.exists() and not overwrite:
        return "skipped"

    image = cv2.imread(str(input_file), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read: {input_file}")

    height, width = image.shape[:2]
    resized = cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_LINEAR)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_file), resized):
        raise RuntimeError(f"Failed to write: {output_file}")
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Upscale competition depth PNG files with bilinear interpolation. "
            "The input can be either a single scene directory or a dataset root containing multiple scenes."
        )
    )
    parser.add_argument("input", type=Path, help="A scene directory or a dataset root.")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=None,
        help="Optional output root. Defaults to <input>_depth_x4.",
    )
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor. Default: 4.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 8) // 2),
        help="Number of worker threads. Default: half of available CPU cores.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    if args.scale <= 0:
        raise SystemExit("--scale must be a positive integer.")

    input_path = args.input.resolve()
    output_path = (args.output.resolve() if args.output is not None else _default_output_path(input_path)).resolve()

    _ensure_safe_output(input_path, output_path)
    scene_roots, is_multi_scene = _discover_scene_roots(input_path)

    tasks: list[tuple[Path, Path]] = []
    for scene_root in scene_roots:
        input_depth_dir = scene_root / "depth"
        output_depth_dir = output_path / scene_root.name / "depth" if is_multi_scene else output_path / "depth"
        files = sorted(file for file in input_depth_dir.glob("*.png") if file.is_file())
        if not files:
            raise SystemExit(f"No PNG depth files found in {input_depth_dir}.")
        for input_file in files:
            tasks.append((input_file, output_depth_dir / input_file.name))

    total = len(tasks)
    print(f"Found {len(scene_roots)} scene(s) and {total} depth PNG file(s).")
    print(f"Applying cv2.INTER_LINEAR x{args.scale} and writing outputs to {output_path}.")

    ok = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [
            executor.submit(
                _upscale_one,
                input_file=input_file,
                output_file=output_file,
                scale=args.scale,
                overwrite=bool(args.overwrite),
            )
            for input_file, output_file in tasks
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
                if result == "ok":
                    ok += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                print(f"[failed] {exc}")

            if index % 50 == 0 or index == total:
                print(f"Progress {index}/{total} | ok={ok} skipped={skipped} failed={failed}")

    print(f"Done: ok={ok} skipped={skipped} failed={failed}")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
