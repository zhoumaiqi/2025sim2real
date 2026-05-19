#!/usr/bin/env python3
"""
Offline Depth Pro evaluator for Tello videos or image directories.

This tool is intentionally isolated from navigation and flight-control logic.
It only runs offline depth inference and saves evaluation artifacts.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CANDIDATE_RATIOS = [
    (0.10, 0.30),
    (0.30, 0.30),
    (0.50, 0.30),
    (0.70, 0.30),
    (0.90, 0.30),
    (0.10, 0.50),
    (0.30, 0.50),
    (0.50, 0.50),
    (0.70, 0.50),
    (0.90, 0.50),
    (0.10, 0.70),
    (0.30, 0.70),
    (0.50, 0.70),
    (0.70, 0.70),
    (0.90, 0.70),
]
DEFAULT_ROI_CENTER_X_RATIO = 0.50
DEFAULT_ROI_CENTER_Y_RATIO = 0.72
DEFAULT_ROI_WIDTH_RATIO = 0.24
DEFAULT_ROI_HEIGHT_RATIO = 0.18
DEFAULT_CANDIDATE_WINDOW = 9
DEFAULT_MIN_VALID_SAMPLES = 5


@dataclass
class FrameItem:
    name: str
    source_path: str
    image_bgr: np.ndarray
    frame_index: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Apple Depth Pro evaluator for Tello videos or images."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input video file, image file, or image directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for visualizations and metrics.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=10,
        help="Video frame sampling step. Default: 10",
    )
    parser.add_argument(
        "--front-risk-threshold",
        type=float,
        default=1.0,
        help="ROI median depth threshold in meters. Default: 1.0",
    )
    parser.add_argument(
        "--candidate-window",
        type=int,
        default=DEFAULT_CANDIDATE_WINDOW,
        help="Odd-sized candidate sampling window. Default: 9",
    )
    parser.add_argument(
        "--roi-width-ratio",
        type=float,
        default=DEFAULT_ROI_WIDTH_RATIO,
        help="Center-lower ROI width as image-width ratio. Default: 0.24",
    )
    parser.add_argument(
        "--roi-height-ratio",
        type=float,
        default=DEFAULT_ROI_HEIGHT_RATIO,
        help="Center-lower ROI height as image-height ratio. Default: 0.18",
    )
    parser.add_argument(
        "--roi-center-y-ratio",
        type=float,
        default=DEFAULT_ROI_CENTER_Y_RATIO,
        help="Center-lower ROI vertical center ratio. Default: 0.72",
    )
    parser.add_argument(
        "--ml-depth-pro-path",
        default=None,
        help="Optional local path to the cloned ml-depth-pro repository.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to depth_pro.pt checkpoint.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Torch device selection. Default: auto",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of frames/images to process.",
    )
    return parser.parse_args()


def normalize_odd_window(window_size: int) -> int:
    if window_size < 1:
        return 1
    if window_size % 2 == 0:
        return window_size + 1
    return window_size


def find_ml_depth_pro_repo(explicit_path: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())

    script_root = Path(__file__).resolve()
    candidates.extend(
        [
            Path.cwd() / "ml-depth-pro",
            script_root.parents[3] / "ml-depth-pro",
            script_root.parents[2] / "ml-depth-pro",
        ]
    )

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (
            resolved.exists()
            and (resolved / "pyproject.toml").exists()
            and (resolved / "src" / "depth_pro" / "__init__.py").exists()
        ):
            return resolved
    return None


def ensure_depth_pro_import(repo_path: Optional[Path]):
    try:
        import depth_pro  # type: ignore

        return depth_pro
    except ModuleNotFoundError as exc:
        if exc.name and exc.name != "depth_pro":
            raise RuntimeError(
                f"depth_pro import failed because dependency '{exc.name}' is missing. "
                "Please install the missing dependency and retry."
            ) from exc

    if repo_path is None:
        raise RuntimeError(
            "Could not import depth_pro and no local ml-depth-pro repository was found. "
            "Please clone apple/ml-depth-pro or provide --ml-depth-pro-path."
        )

    install_cmd = [sys.executable, "-m", "pip", "install", "-e", str(repo_path), "--no-deps"]
    print(f"[setup] depth_pro not importable, trying local install: {' '.join(install_cmd)}")
    result = subprocess.run(install_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to install local ml-depth-pro repository.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
            "Please install Apple ml-depth-pro manually and retry."
        )

    importlib.invalidate_caches()
    try:
        import depth_pro  # type: ignore

        return depth_pro
    except ModuleNotFoundError as exc:
        if exc.name and exc.name != "depth_pro":
            raise RuntimeError(
                f"Local ml-depth-pro was installed, but dependency '{exc.name}' is still missing. "
                "Please install the missing dependency and retry."
            ) from exc
        raise RuntimeError(
            "Installed local ml-depth-pro, but depth_pro still cannot be imported."
        ) from exc


def resolve_checkpoint_path(args: argparse.Namespace, repo_path: Optional[Path]) -> Path:
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
    elif repo_path is not None:
        checkpoint = (repo_path / "checkpoints" / "depth_pro.pt").resolve()
    else:
        checkpoint = Path("checkpoints/depth_pro.pt").resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(
            "Depth Pro checkpoint not found at "
            f"'{checkpoint}'. Download it first, for example by using "
            f"'{repo_path / 'get_pretrained_models.sh' if repo_path else 'get_pretrained_models.sh'}'."
        )
    return checkpoint


def get_torch_device(device_name: str):
    import torch

    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda:0")
    if device_name == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(depth_pro_module, checkpoint_path: Path, device_name: str):
    import torch
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    device = get_torch_device(device_name)
    precision = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    config = replace(
        DEFAULT_MONODEPTH_CONFIG_DICT,
        checkpoint_uri=str(checkpoint_path),
    )
    model, transform = depth_pro_module.create_model_and_transforms(
        config=config,
        device=device,
        precision=precision,
    )
    model.eval()
    return model, transform, device, precision


def iter_input_frames(input_path: Path, frame_step: int) -> Iterator[FrameItem]:
    suffix = input_path.suffix.lower()
    if input_path.is_dir():
        image_paths = sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found under '{input_path}'.")
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"[warn] skipping unreadable image: {image_path}")
                continue
            yield FrameItem(
                name=image_path.stem,
                source_path=str(image_path.resolve()),
                image_bgr=image,
                frame_index=None,
            )
        return

    if input_path.is_file() and suffix in IMAGE_SUFFIXES:
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image '{input_path}'.")
        yield FrameItem(
            name=input_path.stem,
            source_path=str(input_path.resolve()),
            image_bgr=image,
            frame_index=None,
        )
        return

    if input_path.is_file() and suffix in VIDEO_SUFFIXES:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video '{input_path}'.")
        frame_index = 0
        saved_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % frame_step == 0:
                    yield FrameItem(
                        name=f"frame_{frame_index:06d}",
                        source_path=str(input_path.resolve()),
                        image_bgr=frame,
                        frame_index=frame_index,
                    )
                    saved_index += 1
                frame_index += 1
        finally:
            capture.release()
        if saved_index == 0:
            raise RuntimeError(
                f"No frames were sampled from '{input_path}'. Check --frame-step."
            )
        return

    raise ValueError(
        f"Unsupported input '{input_path}'. Use a video file, image file, or image directory."
    )


def ensure_output_dirs(output_root: Path) -> Dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    dirs = {
        "root": output_root,
        "original": output_root / "original",
        "depth_vis": output_root / "depth_vis",
        "overlay": output_root / "overlay",
        "json": output_root / "json",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def generate_candidate_points(width: int, height: int) -> List[Dict[str, object]]:
    candidates = []
    for idx, (x_ratio, y_ratio) in enumerate(CANDIDATE_RATIOS, start=1):
        x = int(round(x_ratio * (width - 1)))
        y = int(round(y_ratio * (height - 1)))
        candidates.append(
            {
                "id": f"P{idx}",
                "point": [int(round(y_ratio * 1000)), int(round(x_ratio * 1000))],
                "pixel": (x, y),
            }
        )
    return candidates


def compute_roi_bounds(
    width: int,
    height: int,
    width_ratio: float,
    height_ratio: float,
    center_x_ratio: float = DEFAULT_ROI_CENTER_X_RATIO,
    center_y_ratio: float = DEFAULT_ROI_CENTER_Y_RATIO,
) -> Tuple[int, int, int, int]:
    roi_w = max(1, int(round(width * width_ratio)))
    roi_h = max(1, int(round(height * height_ratio)))
    center_x = int(round(center_x_ratio * (width - 1)))
    center_y = int(round(center_y_ratio * (height - 1)))
    x0 = max(0, center_x - roi_w // 2)
    y0 = max(0, center_y - roi_h // 2)
    x1 = min(width, x0 + roi_w)
    y1 = min(height, y0 + roi_h)
    return x0, y0, x1, y1


def valid_depth_values(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values) & (values > 0)]


def sample_window_median(
    depth_map: np.ndarray,
    x: int,
    y: int,
    window_size: int,
    min_valid_samples: int = DEFAULT_MIN_VALID_SAMPLES,
) -> Optional[float]:
    radius = window_size // 2
    h, w = depth_map.shape[:2]
    window = depth_map[
        max(0, y - radius) : min(h, y + radius + 1),
        max(0, x - radius) : min(w, x + radius + 1),
    ]
    valid = valid_depth_values(window)
    if valid.size < min_valid_samples:
        return None
    return float(np.median(valid))


def compute_depth_stats(depth_map: np.ndarray) -> Dict[str, Optional[float]]:
    valid = valid_depth_values(depth_map)
    if valid.size == 0:
        return {
            "min_m": None,
            "max_m": None,
            "median_m": None,
            "valid_depth_ratio": 0.0,
        }
    return {
        "min_m": float(np.min(valid)),
        "max_m": float(np.max(valid)),
        "median_m": float(np.median(valid)),
        "valid_depth_ratio": float(valid.size / depth_map.size),
    }


def compute_candidate_depths(
    depth_map: np.ndarray,
    candidates: Sequence[Dict[str, object]],
    window_size: int,
) -> Dict[str, Optional[float]]:
    candidate_depths: Dict[str, Optional[float]] = {}
    for candidate in candidates:
        x, y = candidate["pixel"]  # type: ignore[index]
        candidate_depths[str(candidate["id"])] = sample_window_median(
            depth_map, x, y, window_size
        )
    return candidate_depths


def compute_roi_median(depth_map: np.ndarray, roi_bounds: Tuple[int, int, int, int]) -> Optional[float]:
    x0, y0, x1, y1 = roi_bounds
    valid = valid_depth_values(depth_map[y0:y1, x0:x1])
    if valid.size == 0:
        return None
    return float(np.median(valid))


def depth_to_color(depth_map: np.ndarray) -> np.ndarray:
    valid = valid_depth_values(depth_map)
    if valid.size == 0:
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)

    inverse_depth = np.zeros_like(depth_map, dtype=np.float32)
    mask = np.isfinite(depth_map) & (depth_map > 0)
    inverse_depth[mask] = 1.0 / depth_map[mask].astype(np.float32)

    max_inv = min(float(np.max(inverse_depth[mask])), 1.0 / 0.1)
    min_inv = max(float(np.min(inverse_depth[mask])), 1.0 / 250.0)
    denom = max(max_inv - min_inv, 1e-6)

    normalized = np.zeros_like(depth_map, dtype=np.float32)
    normalized[mask] = np.clip((inverse_depth[mask] - min_inv) / denom, 0.0, 1.0)
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~mask] = 0
    return color


def format_depth_text(depth_value: Optional[float]) -> str:
    return "n/a" if depth_value is None else f"{depth_value:.2f}m"


def annotate_overlay(
    image_bgr: np.ndarray,
    candidates: Sequence[Dict[str, object]],
    candidate_depths: Dict[str, Optional[float]],
    roi_bounds: Tuple[int, int, int, int],
    roi_median: Optional[float],
    front_risk: bool,
    depth_stats: Dict[str, Optional[float]],
    front_risk_threshold: float,
) -> np.ndarray:
    overlay = image_bgr.copy()
    x0, y0, x1, y1 = roi_bounds
    roi_color = (0, 0, 255) if front_risk else (0, 200, 0)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), roi_color, 2)
    cv2.putText(
        overlay,
        f"ROI median: {format_depth_text(roi_median)}",
        (x0, max(25, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        roi_color,
        2,
    )

    for candidate in candidates:
        candidate_id = str(candidate["id"])
        x, y = candidate["pixel"]  # type: ignore[index]
        depth_value = candidate_depths.get(candidate_id)
        if depth_value is None:
            color = (160, 160, 160)
        elif depth_value < front_risk_threshold:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)
        cv2.circle(overlay, (x, y), 10, color, -1)
        cv2.circle(overlay, (x, y), 10, (0, 0, 0), 2)
        cv2.putText(
            overlay,
            candidate_id,
            (x + 12, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            overlay,
            format_depth_text(depth_value),
            (x + 12, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )

    summary_lines = [
        f"front_risk={front_risk}",
        f"depth median={format_depth_text(depth_stats['median_m'])}",
        f"valid ratio={depth_stats['valid_depth_ratio']:.3f}",
    ]
    for idx, line in enumerate(summary_lines):
        y = 28 + idx * 24
        cv2.putText(
            overlay,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            overlay,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
        )
    return overlay


def to_serializable_float(value: Optional[float]) -> Optional[float]:
    return None if value is None else float(value)


def run_inference(depth_pro_module, model, transform, image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    f_px = None
    transformed = transform(image_rgb)
    prediction = model.infer(transformed, f_px=f_px)
    depth_map = prediction["depth"].detach().cpu().numpy().squeeze()
    return depth_map.astype(np.float32)


def summarize_values(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    valid_values = [float(value) for value in values if value is not None]
    if not valid_values:
        return {"count": 0, "min": None, "max": None, "median": None}
    return {
        "count": len(valid_values),
        "min": float(min(valid_values)),
        "max": float(max(valid_values)),
        "median": float(np.median(valid_values)),
    }


def build_summary(
    results: Sequence[Dict[str, object]],
    args: argparse.Namespace,
    repo_path: Optional[Path],
    checkpoint_path: Path,
    device,
) -> Dict[str, object]:
    candidate_stats: Dict[str, Dict[str, Optional[float]]] = {}
    for idx in range(1, 16):
        candidate_id = f"P{idx}"
        candidate_stats[candidate_id] = summarize_values(
            row["candidate_depth"][candidate_id] for row in results  # type: ignore[index]
        )

    front_risk_count = sum(1 for row in results if row["front_risk"])
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(Path(args.input).resolve()),
        "output": str(Path(args.output).resolve()),
        "ml_depth_pro_path": str(repo_path) if repo_path else None,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "frames_processed": len(results),
        "front_risk_count": front_risk_count,
        "front_risk_ratio": (front_risk_count / len(results)) if results else 0.0,
        "front_risk_threshold": float(args.front_risk_threshold),
        "roi_median_stats": summarize_values(
            row["center_lower_roi"]["median_m"] for row in results  # type: ignore[index]
        ),
        "depth_median_stats": summarize_values(
            row["depth_stats"]["median_m"] for row in results  # type: ignore[index]
        ),
        "valid_depth_ratio_stats": summarize_values(
            row["depth_stats"]["valid_depth_ratio"] for row in results  # type: ignore[index]
        ),
        "candidate_depth_stats": candidate_stats,
    }


def write_results_csv(output_root: Path, results: Sequence[Dict[str, object]]) -> None:
    csv_path = output_root / "results.csv"
    fieldnames = [
        "name",
        "source_path",
        "frame_index",
        "image_width",
        "image_height",
        "front_risk",
        "roi_depth_median",
        "depth_min_m",
        "depth_max_m",
        "depth_median_m",
        "valid_depth_ratio",
    ] + [f"P{idx}_depth_median" for idx in range(1, 16)]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "name": result["name"],
                "source_path": result["source_path"],
                "frame_index": result["frame_index"],
                "image_width": result["image_width"],
                "image_height": result["image_height"],
                "front_risk": result["front_risk"],
                "roi_depth_median": result["center_lower_roi"]["median_m"],  # type: ignore[index]
                "depth_min_m": result["depth_stats"]["min_m"],  # type: ignore[index]
                "depth_max_m": result["depth_stats"]["max_m"],  # type: ignore[index]
                "depth_median_m": result["depth_stats"]["median_m"],  # type: ignore[index]
                "valid_depth_ratio": result["depth_stats"]["valid_depth_ratio"],  # type: ignore[index]
            }
            for idx in range(1, 16):
                row[f"P{idx}_depth_median"] = result["candidate_depth"][f"P{idx}"]  # type: ignore[index]
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    args.frame_step = max(1, args.frame_step)
    args.candidate_window = normalize_odd_window(args.candidate_window)

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        print(f"[error] input path does not exist: {input_path}")
        return 1

    repo_path = find_ml_depth_pro_repo(args.ml_depth_pro_path)
    try:
        depth_pro_module = ensure_depth_pro_import(repo_path)
        checkpoint_path = resolve_checkpoint_path(args, repo_path)
    except Exception as exc:
        print(f"[error] {exc}")
        return 1

    try:
        model, transform, device, precision = build_model(
            depth_pro_module, checkpoint_path, args.device
        )
    except Exception as exc:
        print(f"[error] failed to load Depth Pro model: {exc}")
        return 1

    output_dirs = ensure_output_dirs(output_root)
    print(f"[info] input={input_path}")
    print(f"[info] output={output_root}")
    print(f"[info] repo={repo_path}")
    print(f"[info] checkpoint={checkpoint_path}")
    print(f"[info] device={device}, precision={precision}")

    results: List[Dict[str, object]] = []
    processed = 0
    try:
        for frame in iter_input_frames(input_path, args.frame_step):
            if args.limit is not None and processed >= args.limit:
                break

            print(f"[info] processing {frame.name}")
            depth_map = run_inference(depth_pro_module, model, transform, frame.image_bgr)
            image_h, image_w = frame.image_bgr.shape[:2]
            candidates = generate_candidate_points(image_w, image_h)
            roi_bounds = compute_roi_bounds(
                image_w,
                image_h,
                args.roi_width_ratio,
                args.roi_height_ratio,
                center_y_ratio=args.roi_center_y_ratio,
            )
            depth_stats = compute_depth_stats(depth_map)
            roi_median = compute_roi_median(depth_map, roi_bounds)
            candidate_depths = compute_candidate_depths(
                depth_map, candidates, args.candidate_window
            )
            front_risk = (
                roi_median is not None and roi_median < float(args.front_risk_threshold)
            )

            depth_vis = depth_to_color(depth_map)
            overlay = annotate_overlay(
                frame.image_bgr,
                candidates,
                candidate_depths,
                roi_bounds,
                roi_median,
                front_risk,
                depth_stats,
                float(args.front_risk_threshold),
            )

            cv2.imwrite(str(output_dirs["original"] / f"{frame.name}.jpg"), frame.image_bgr)
            cv2.imwrite(str(output_dirs["depth_vis"] / f"{frame.name}.jpg"), depth_vis)
            cv2.imwrite(str(output_dirs["overlay"] / f"{frame.name}.jpg"), overlay)

            frame_result: Dict[str, object] = {
                "name": frame.name,
                "source_path": frame.source_path,
                "frame_index": frame.frame_index,
                "image_width": image_w,
                "image_height": image_h,
                "candidate_depth": {
                    key: to_serializable_float(value)
                    for key, value in candidate_depths.items()
                },
                "center_lower_roi": {
                    "bounds_xyxy": list(roi_bounds),
                    "median_m": to_serializable_float(roi_median),
                },
                "candidate_window": args.candidate_window,
                "front_risk_threshold": float(args.front_risk_threshold),
                "front_risk": bool(front_risk),
                "depth_stats": {
                    "min_m": to_serializable_float(depth_stats["min_m"]),
                    "max_m": to_serializable_float(depth_stats["max_m"]),
                    "median_m": to_serializable_float(depth_stats["median_m"]),
                    "valid_depth_ratio": float(depth_stats["valid_depth_ratio"] or 0.0),
                },
            }
            with (output_dirs["json"] / f"{frame.name}.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(frame_result, handle, indent=2)

            results.append(frame_result)
            processed += 1
    except Exception as exc:
        print(f"[error] inference stopped: {exc}")
        return 1

    summary = build_summary(results, args, repo_path, checkpoint_path, device)
    write_results_csv(output_root, results)
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[done] processed {len(results)} frame(s)")
    print(f"[done] results.csv -> {output_root / 'results.csv'}")
    print(f"[done] summary.json -> {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
