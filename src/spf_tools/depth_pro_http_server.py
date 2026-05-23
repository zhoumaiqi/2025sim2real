#!/usr/bin/env python3
"""
Lightweight HTTP service for remote Depth Pro inference.

This service loads Apple Depth Pro once and exposes a single JSON endpoint:
POST /infer

Request JSON:
{
  "image_base64": "...",
  "candidate_ratios": [[0.1, 0.3], ...],
  "candidate_window": 9,
  "roi_center_x_ratio": 0.50,
  "roi_center_y_ratio": 0.72,
  "roi_width_ratio": 0.24,
  "roi_height_ratio": 0.18,
  "min_safe_depth_m": 1.0,
  "max_valid_depth_m": 30.0,
  "front_risk_threshold_m": 1.0
}
"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import sys
import time
from dataclasses import replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_CANDIDATE_RATIOS = [
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


APP_STATE: Dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth Pro HTTP service for SPF.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--ml-depth-pro-path",
        required=True,
        help="Path to the cloned apple/ml-depth-pro repository.",
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
    )
    return parser.parse_args()


def normalize_odd_window(window_size: int) -> int:
    if window_size < 1:
        return 1
    if window_size % 2 == 0:
        return window_size + 1
    return window_size


def ensure_depth_pro_import(repo_path: Path):
    try:
        import depth_pro  # type: ignore

        return depth_pro
    except ModuleNotFoundError as exc:
        if exc.name and exc.name != "depth_pro":
            raise RuntimeError(
                f"depth_pro import failed because dependency '{exc.name}' is missing."
            ) from exc

    src_path = repo_path / "src"
    if not src_path.exists():
        raise RuntimeError(f"Depth Pro src path not found: {src_path}")

    sys.path.insert(0, str(src_path))
    importlib.invalidate_caches()
    try:
        import depth_pro  # type: ignore

        return depth_pro
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import depth_pro. Install apple/ml-depth-pro in the active "
            "environment or ensure its src directory is available."
        ) from exc


def resolve_checkpoint_path(repo_path: Path, checkpoint_arg: Optional[str]) -> Path:
    if checkpoint_arg:
        checkpoint = Path(checkpoint_arg).expanduser().resolve()
    else:
        checkpoint = (repo_path / "checkpoints" / "depth_pro.pt").resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(f"Depth Pro checkpoint not found: {checkpoint}")
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


def decode_image_b64_to_bgr(image_base64: str) -> np.ndarray:
    image_bytes = base64.b64decode(image_base64.encode("ascii"))
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode input image")
    return image_bgr


def generate_candidate_points(
    width: int,
    height: int,
    candidate_ratios: Optional[Sequence[Sequence[float]]],
) -> List[Dict[str, object]]:
    ratios = candidate_ratios or DEFAULT_CANDIDATE_RATIOS
    candidates = []
    for idx, ratio_pair in enumerate(ratios, start=1):
        x_ratio = float(ratio_pair[0])
        y_ratio = float(ratio_pair[1])
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
    center_x_ratio: float,
    center_y_ratio: float,
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
            depth_map,
            x,
            y,
            window_size,
        )
    return candidate_depths


def compute_roi_median(
    depth_map: np.ndarray,
    roi_bounds: Tuple[int, int, int, int],
) -> Optional[float]:
    x0, y0, x1, y1 = roi_bounds
    valid = valid_depth_values(depth_map[y0:y1, x0:x1])
    if valid.size == 0:
        return None
    return float(np.median(valid))


def classify_candidate_safety(
    depth_value: Optional[float],
    min_safe_depth_m: float,
    max_valid_depth_m: float,
) -> str:
    if depth_value is None or not np.isfinite(depth_value):
        return "unknown"
    if depth_value <= 0 or depth_value > max_valid_depth_m:
        return "unknown"
    if depth_value < min_safe_depth_m:
        return "blocked"
    return "safe"


def run_inference(image_bgr: np.ndarray) -> np.ndarray:
    depth_pro_module = APP_STATE["depth_pro_module"]
    model = APP_STATE["model"]
    transform = APP_STATE["transform"]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transformed = transform(image_rgb)
    prediction = model.infer(transformed, f_px=None)
    depth_map = prediction["depth"].detach().cpu().numpy().squeeze()
    return depth_map.astype(np.float32)


class DepthProRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/infer":
            self._write_json(
                {"ok": False, "error": f"Unsupported path: {self.path}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
            result = self._handle_infer(payload)
            self._write_json(result)
        except Exception as exc:
            self._write_json(
                {"ok": False, "error": str(exc)},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format, *args):
        print(
            "[DepthProServer] "
            + format % args
        )

    def _handle_infer(self, payload: Dict[str, object]) -> Dict[str, object]:
        started_at = time.time()
        image_base64 = str(payload.get("image_base64", "")).strip()
        if not image_base64:
            raise ValueError("Missing 'image_base64' in request payload")

        image_bgr = decode_image_b64_to_bgr(image_base64)
        image_h, image_w = image_bgr.shape[:2]

        candidate_window = normalize_odd_window(
            int(payload.get("candidate_window", DEFAULT_CANDIDATE_WINDOW))
        )
        roi_center_x_ratio = float(
            payload.get("roi_center_x_ratio", DEFAULT_ROI_CENTER_X_RATIO)
        )
        roi_center_y_ratio = float(
            payload.get("roi_center_y_ratio", DEFAULT_ROI_CENTER_Y_RATIO)
        )
        roi_width_ratio = float(
            payload.get("roi_width_ratio", DEFAULT_ROI_WIDTH_RATIO)
        )
        roi_height_ratio = float(
            payload.get("roi_height_ratio", DEFAULT_ROI_HEIGHT_RATIO)
        )
        min_safe_depth_m = float(payload.get("min_safe_depth_m", 1.0))
        max_valid_depth_m = float(payload.get("max_valid_depth_m", 30.0))
        front_risk_threshold_m = float(payload.get("front_risk_threshold_m", 1.0))

        candidates = generate_candidate_points(
            image_w,
            image_h,
            payload.get("candidate_ratios"),
        )
        roi_bounds = compute_roi_bounds(
            image_w,
            image_h,
            roi_width_ratio,
            roi_height_ratio,
            roi_center_x_ratio,
            roi_center_y_ratio,
        )

        depth_map = run_inference(image_bgr)
        depth_stats = compute_depth_stats(depth_map)
        candidate_depths = compute_candidate_depths(
            depth_map,
            candidates,
            candidate_window,
        )
        candidate_safety = {
            candidate_id: classify_candidate_safety(
                depth_value,
                min_safe_depth_m=min_safe_depth_m,
                max_valid_depth_m=max_valid_depth_m,
            )
            for candidate_id, depth_value in candidate_depths.items()
        }
        roi_median = compute_roi_median(depth_map, roi_bounds)
        front_risk = (
            roi_median is not None and roi_median < front_risk_threshold_m
        )

        return {
            "ok": True,
            "image_width": image_w,
            "image_height": image_h,
            "candidate_depth": candidate_depths,
            "candidate_safety": candidate_safety,
            "center_lower_roi": {
                "bounds_xyxy": list(roi_bounds),
                "median_m": roi_median,
            },
            "front_risk": bool(front_risk),
            "depth_stats": depth_stats,
            "latency_ms": round((time.time() - started_at) * 1000.0, 2),
            "device": str(APP_STATE["device"]),
        }

    def _write_json(self, payload: Dict[str, object], status=HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    args = parse_args()
    repo_path = Path(args.ml_depth_pro_path).expanduser().resolve()
    if not repo_path.exists():
        print(f"[error] ml-depth-pro path does not exist: {repo_path}")
        return 1

    try:
        depth_pro_module = ensure_depth_pro_import(repo_path)
        checkpoint_path = resolve_checkpoint_path(repo_path, args.checkpoint)
        model, transform, device, precision = build_model(
            depth_pro_module,
            checkpoint_path,
            args.device,
        )
    except Exception as exc:
        print(f"[error] failed to initialize Depth Pro service: {exc}")
        return 1

    APP_STATE.update(
        {
            "depth_pro_module": depth_pro_module,
            "model": model,
            "transform": transform,
            "device": device,
            "precision": precision,
            "checkpoint_path": checkpoint_path,
        }
    )

    print(f"[info] repo={repo_path}")
    print(f"[info] checkpoint={checkpoint_path}")
    print(f"[info] device={device}, precision={precision}")
    print(f"[info] serving on http://{args.host}:{args.port}/infer")

    server = ThreadingHTTPServer((args.host, args.port), DepthProRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[info] shutting down Depth Pro service")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
