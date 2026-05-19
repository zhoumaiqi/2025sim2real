from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


class EVEState(str, Enum):
    EXPLORE = "EXPLORE"
    VERIFY = "VERIFY"
    EXPLOIT = "EXPLOIT"


def _extract_xyz(value: Any) -> Optional[Tuple[float, float, float]]:
    """Best-effort extraction for AirSim pose/vector, dicts, or simple sequences."""
    if value is None:
        return None

    if hasattr(value, "position"):
        return _extract_xyz(value.position)

    if isinstance(value, dict):
        if "position" in value:
            return _extract_xyz(value["position"])
        x = value.get("x", value.get("x_val"))
        y = value.get("y", value.get("y_val"))
        z = value.get("z", value.get("z_val", 0.0))
        if x is not None and y is not None:
            try:
                return (float(x), float(y), float(z))
            except (TypeError, ValueError):
                return None

    if all(hasattr(value, attr) for attr in ("x_val", "y_val")):
        try:
            return (
                float(value.x_val),
                float(value.y_val),
                float(getattr(value, "z_val", 0.0)),
            )
        except (TypeError, ValueError):
            return None

    if isinstance(value, Sequence) and len(value) >= 2 and not isinstance(value, str):
        try:
            z = value[2] if len(value) > 2 else 0.0
            return (float(value[0]), float(value[1]), float(z))
        except (TypeError, ValueError):
            return None

    return None


@dataclass
class TargetMemory:
    confidence_threshold: float = 0.5
    visible_conf_threshold: Optional[float] = None
    history_size: int = 5
    last_seen_pixel: Optional[Tuple[int, int]] = None
    last_seen_world_position: Optional[Tuple[float, float, float]] = None
    last_seen_choice: Optional[str] = None
    confidence: float = 0.0
    stable_count: int = 0
    lost_count: int = 0
    selected_history: Deque[str] = field(init=False)

    def __post_init__(self) -> None:
        if self.visible_conf_threshold is None:
            self.visible_conf_threshold = self.confidence_threshold
        self.selected_history = deque(maxlen=self.history_size)

    def update(
        self,
        choice: str,
        target_visible: bool,
        confidence: float,
        pixel: Optional[Tuple[int, int]] = None,
        world_position: Any = None,
    ) -> None:
        choice = str(choice or "").strip().upper()
        confidence = _clamp_confidence(confidence)
        observed_target = target_visible and confidence >= self.visible_conf_threshold

        if choice:
            self.selected_history.append(choice)
            if observed_target:
                if self.last_seen_choice == choice:
                    self.stable_count += 1
                else:
                    self.stable_count = 1

        if observed_target:
            self.lost_count = 0
        else:
            self.lost_count += 1

        self.confidence = confidence
        if observed_target:
            if pixel is not None:
                self.last_seen_pixel = (int(pixel[0]), int(pixel[1]))
            extracted_world = _extract_xyz(world_position)
            if extracted_world is not None:
                self.last_seen_world_position = extracted_world
            self.last_seen_choice = choice or self.last_seen_choice


@dataclass
class EVEStateMachine:
    state: EVEState = EVEState.EXPLORE

    def update(
        self,
        target_visible: bool,
        confidence: float,
        stable_count: int,
        lost_count: int,
    ) -> EVEState:
        confidence = _clamp_confidence(confidence)

        if self.state == EVEState.EXPLORE:
            if target_visible and confidence >= 0.5:
                self.state = EVEState.VERIFY
        elif self.state == EVEState.VERIFY:
            if stable_count >= 3 and confidence >= 0.6:
                self.state = EVEState.EXPLOIT
            elif lost_count >= 3:
                self.state = EVEState.EXPLORE
        elif self.state == EVEState.EXPLOIT:
            if lost_count >= 5:
                self.state = EVEState.EXPLORE

        return self.state


@dataclass
class IEVEDecision:
    selected_candidate: Dict[str, Any]
    eve_state: EVEState
    using_memory_target: bool
    used_center_fallback: bool
    ieve_output_source: str
    selected_point: List[int]
    selected_pixel: Tuple[int, int]


class SPFIEVELite:
    def __init__(
        self,
        memory_confidence_threshold: float = 0.5,
    ) -> None:
        self.target_memory = TargetMemory(
            confidence_threshold=memory_confidence_threshold
        )
        self.state_machine = EVEStateMachine()

    def update_and_select(
        self,
        candidates: List[Dict[str, Any]],
        current_candidate: Dict[str, Any],
        target_visible: bool,
        confidence: float,
        target_world_position: Any = None,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> IEVEDecision:
        confidence = _clamp_confidence(confidence)
        current_pixel = _candidate_pixel(current_candidate)
        current_choice = str(current_candidate.get("id", "")).upper()

        self.target_memory.update(
            choice=current_choice,
            target_visible=target_visible,
            confidence=confidence,
            pixel=current_pixel,
            world_position=target_world_position,
        )

        eve_state = self.state_machine.update(
            target_visible=target_visible,
            confidence=confidence,
            stable_count=self.target_memory.stable_count,
            lost_count=self.target_memory.lost_count,
        )

        selected_candidate = dict(current_candidate)
        using_memory_target = False
        used_center_fallback = False
        ieve_output_source = "vlm"

        if eve_state == EVEState.EXPLOIT and not target_visible:
            memory_pixel = self.target_memory.last_seen_pixel
            if memory_pixel is not None and image_shape is not None:
                selected_candidate = self._candidate_from_memory_pixel(
                    memory_pixel,
                    image_shape=image_shape,
                    candidate_id=self.target_memory.last_seen_choice or "MEMORY",
                )
                using_memory_target = True
                ieve_output_source = "memory"
            else:
                selected_candidate = dict(_center_candidate(candidates))
                used_center_fallback = True
                ieve_output_source = "fallback"

        selected_point = [int(v) for v in selected_candidate["point"]]
        selected_pixel = _candidate_pixel(selected_candidate)

        return IEVEDecision(
            selected_candidate=selected_candidate,
            eve_state=eve_state,
            using_memory_target=using_memory_target,
            used_center_fallback=used_center_fallback,
            ieve_output_source=ieve_output_source,
            selected_point=selected_point,
            selected_pixel=selected_pixel,
        )

    def _candidate_from_memory_pixel(
        self,
        memory_pixel: Tuple[int, int],
        image_shape: Tuple[int, int],
        candidate_id: str,
    ) -> Dict[str, Any]:
        height, width = image_shape[:2]
        pixel_x = int(np.clip(memory_pixel[0], 0, max(width - 1, 0)))
        pixel_y = int(np.clip(memory_pixel[1], 0, max(height - 1, 0)))
        y = int(round((pixel_y / max(height, 1)) * 1000))
        x = int(round((pixel_x / max(width, 1)) * 1000))
        return {
            "id": candidate_id,
            "point": [y, x],
            "pixel": (pixel_x, pixel_y),
        }

    def log_decision(
        self,
        vlm_choice: str,
        target_visible: bool,
        confidence: float,
        decision: IEVEDecision,
    ) -> None:
        print(f"EVE state: {decision.eve_state.value}")
        print(f"raw_vlm_choice: {vlm_choice}")
        print(f"target_visible: {target_visible}")
        print(f"confidence: {_clamp_confidence(confidence):.2f}")
        print(f"memory_last_seen_choice: {self.target_memory.last_seen_choice}")
        print(f"memory_last_seen_pixel: {self.target_memory.last_seen_pixel}")
        print(f"lost_count: {self.target_memory.lost_count}")
        print(f"using_memory_target: {decision.using_memory_target}")
        print(f"ieve_output_source: {decision.ieve_output_source}")
        print(f"ieve_intended_choice: {decision.selected_candidate.get('id')}")
        print(f"stable_count: {self.target_memory.stable_count}")
        print(f"selected point: {decision.selected_point}")


def _candidate_pixel(candidate: Dict[str, Any]) -> Tuple[int, int]:
    pixel = candidate.get("pixel", (0, 0))
    return (int(pixel[0]), int(pixel[1]))


def _center_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    for candidate in candidates:
        if candidate.get("id") == "P8":
            return candidate
    return candidates[len(candidates) // 2]


def _clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    return max(0.0, min(1.0, confidence))


