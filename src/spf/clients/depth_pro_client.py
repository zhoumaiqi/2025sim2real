from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
from urllib import error, request

import cv2
import numpy as np


DEFAULT_ENDPOINT = "http://127.0.0.1:8765/infer"
DEFAULT_ROI_CENTER_X_RATIO = 0.50
DEFAULT_ROI_CENTER_Y_RATIO = 0.72
DEFAULT_ROI_WIDTH_RATIO = 0.24
DEFAULT_ROI_HEIGHT_RATIO = 0.18
DEFAULT_CANDIDATE_WINDOW = 9


@dataclass
class RemoteDepthProClient:
    endpoint: str
    timeout_seconds: float = 5.0
    candidate_window: int = DEFAULT_CANDIDATE_WINDOW
    roi_center_x_ratio: float = DEFAULT_ROI_CENTER_X_RATIO
    roi_center_y_ratio: float = DEFAULT_ROI_CENTER_Y_RATIO
    roi_width_ratio: float = DEFAULT_ROI_WIDTH_RATIO
    roi_height_ratio: float = DEFAULT_ROI_HEIGHT_RATIO
    min_safe_depth_m: float = 1.0
    max_valid_depth_m: float = 30.0
    front_risk_threshold_m: float = 1.0
    inject_depth_safety_into_prompt: bool = False
    reduce_forward_on_front_risk: bool = True
    clamp_downward_on_front_risk: bool = True
    min_motion_depth_m: float = 0.6
    enabled: bool = True

    @classmethod
    def from_config(
        cls, config: Optional[Dict[str, Any]]
    ) -> Optional["RemoteDepthProClient"]:
        if not config or not bool(config.get("enabled", False)):
            return None

        endpoint = str(config.get("endpoint", DEFAULT_ENDPOINT)).strip()
        if not endpoint:
            return None

        return cls(
            endpoint=endpoint,
            timeout_seconds=_coerce_float(config.get("timeout_seconds"), 5.0),
            candidate_window=_coerce_int(
                config.get("candidate_window"), DEFAULT_CANDIDATE_WINDOW
            ),
            roi_center_x_ratio=_coerce_float(
                config.get("roi_center_x_ratio"), DEFAULT_ROI_CENTER_X_RATIO
            ),
            roi_center_y_ratio=_coerce_float(
                config.get("roi_center_y_ratio"), DEFAULT_ROI_CENTER_Y_RATIO
            ),
            roi_width_ratio=_coerce_float(
                config.get("roi_width_ratio"), DEFAULT_ROI_WIDTH_RATIO
            ),
            roi_height_ratio=_coerce_float(
                config.get("roi_height_ratio"), DEFAULT_ROI_HEIGHT_RATIO
            ),
            min_safe_depth_m=_coerce_float(config.get("min_safe_depth_m"), 1.0),
            max_valid_depth_m=_coerce_float(config.get("max_valid_depth_m"), 30.0),
            front_risk_threshold_m=_coerce_float(
                config.get("front_risk_threshold_m"), 1.0
            ),
            inject_depth_safety_into_prompt=bool(
                config.get("inject_depth_safety_into_prompt", False)
            ),
            reduce_forward_on_front_risk=bool(
                config.get("reduce_forward_on_front_risk", True)
            ),
            clamp_downward_on_front_risk=bool(
                config.get("clamp_downward_on_front_risk", True)
            ),
            min_motion_depth_m=_coerce_float(config.get("min_motion_depth_m"), 0.6),
            enabled=True,
        )

    def infer(
        self,
        image_rgb: np.ndarray,
        candidate_ratios: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        encoded_image = _encode_image_rgb_to_base64(image_rgb)
        payload = {
            "image_base64": encoded_image,
            "candidate_ratios": list(candidate_ratios or []),
            "candidate_window": self.candidate_window,
            "roi_center_x_ratio": self.roi_center_x_ratio,
            "roi_center_y_ratio": self.roi_center_y_ratio,
            "roi_width_ratio": self.roi_width_ratio,
            "roi_height_ratio": self.roi_height_ratio,
            "min_safe_depth_m": self.min_safe_depth_m,
            "max_valid_depth_m": self.max_valid_depth_m,
            "front_risk_threshold_m": self.front_risk_threshold_m,
        }

        request_body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            self.endpoint,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                response_text = response.read().decode("utf-8")
        except (error.URLError, TimeoutError, OSError) as exc:
            print(f"[DepthPro] request failed: {exc}")
            return None

        try:
            response_data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            print(f"[DepthPro] invalid JSON response: {exc}")
            return None

        if not isinstance(response_data, dict):
            print("[DepthPro] unexpected response type")
            return None

        if response_data.get("ok") is False:
            print(f"[DepthPro] server error: {response_data.get('error')}")
            return None

        return response_data


def _encode_image_rgb_to_base64(image_rgb: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise ValueError("Failed to encode image for remote depth inference")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
