import cv2
import json
import math
import os
import time
from collections import Counter, deque
from typing import List, Tuple

import numpy as np

from ..base.action_projector import ActionProjector
from ..base.drone_space import ActionPoint
from .drone_space import AirSimDroneActionSpace
from .ieve_lite import SPFIEVELite

MIN_SAFE_DEPTH_M = 1.0
MAX_VALID_DEPTH_M = 30.0
DEPTH_SAMPLE_WINDOW = 9
MIN_VALID_DEPTH_SAMPLES = 5
GROUND_RISK_VZ_THRESHOLD_MPS = 0.2
YAW_DEADBAND_DEG = 10.0
MAX_YAW_STEP_DEG = 20.0


candidate_ratios = [
    (0.10, 0.30), (0.30, 0.30), (0.50, 0.30), (0.70, 0.30), (0.90, 0.30),
    (0.10, 0.50), (0.30, 0.50), (0.50, 0.50), (0.70, 0.50), (0.90, 0.50),
    (0.10, 0.70), (0.30, 0.70), (0.50, 0.70), (0.70, 0.70), (0.90, 0.70),
]


def generate_candidate_points(image):
    height, width = image.shape[:2]
    candidates = []
    for idx, (x_ratio, y_ratio) in enumerate(candidate_ratios, 1):
        candidates.append(
            {
                "id": f"P{idx}",
                "point": [int(round(y_ratio * 1000)), int(round(x_ratio * 1000))],
                "pixel": (int(round(x_ratio * width)), int(round(y_ratio * height))),
            }
        )
    return candidates


def sample_candidate_depth(depth_image, candidate):
    if depth_image is None:
        return None

    try:
        depth_array = np.asarray(depth_image)
        if depth_array.ndim == 3:
            depth_array = depth_array[:, :, 0]

        height, width = depth_array.shape[:2]
        y_norm, x_norm = candidate["point"]
        x = int(round((x_norm / 1000.0) * (width - 1)))
        y = int(round((y_norm / 1000.0) * (height - 1)))
        radius = DEPTH_SAMPLE_WINDOW // 2

        window = depth_array[
            max(0, y - radius): min(height, y + radius + 1),
            max(0, x - radius): min(width, x + radius + 1),
        ]
        valid = window[np.isfinite(window) & (window > 0)]
        if valid.size < MIN_VALID_DEPTH_SAMPLES:
            return None
        return float(np.median(valid))
    except Exception:
        return None


def classify_candidate_safety(depth_value):
    if depth_value is None or not np.isfinite(depth_value):
        return "unknown"
    if depth_value <= 0 or depth_value > MAX_VALID_DEPTH_M:
        return "unknown"
    if depth_value < MIN_SAFE_DEPTH_M:
        return "blocked"
    return "safe"


def attach_candidate_safety(candidates, depth_image):
    if depth_image is None:
        print("Depth image: unavailable; marking all candidates as unknown")
    else:
        _log_depth_range(depth_image)

    print("Candidate safety:")
    for candidate in candidates:
        depth = sample_candidate_depth(depth_image, candidate)
        safety = classify_candidate_safety(depth)
        candidate["depth"] = depth
        candidate["safety"] = safety
        print(f"{candidate['id']} depth={_format_depth(depth)}, safety={safety}")


def draw_candidate_points(image, candidates):
    annotated = image.copy()
    for candidate in candidates:
        x, y = candidate["pixel"]
        safety = candidate.get("safety", "unknown")
        fill_color = {
            "safe": (0, 255, 255),
            "blocked": (80, 80, 255),
            "unknown": (180, 180, 180),
            "ground_risk": (0, 165, 255),
        }.get(safety, (180, 180, 180))
        border_color = {
            "safe": (0, 0, 0),
            "blocked": (0, 0, 255),
            "unknown": (255, 0, 0),
            "ground_risk": (0, 90, 255),
        }.get(safety, (255, 0, 0))

        cv2.circle(annotated, (x, y), 10, fill_color, -1)
        cv2.circle(annotated, (x, y), 10, border_color, 2)
        if safety == "blocked":
            cv2.line(annotated, (x - 12, y - 12), (x + 12, y + 12), (0, 0, 255), 3)
            cv2.line(annotated, (x + 12, y - 12), (x - 12, y + 12), (0, 0, 255), 3)
        elif safety == "ground_risk":
            cv2.line(annotated, (x, y - 13), (x, y + 13), (0, 90, 255), 3)
            cv2.line(annotated, (x - 6, y + 7), (x, y + 13), (0, 90, 255), 3)
            cv2.line(annotated, (x + 6, y + 7), (x, y + 13), (0, 90, 255), 3)
        cv2.putText(
            annotated,
            candidate["id"],
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            annotated,
            candidate["id"],
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            annotated,
            safety,
            (x + 12, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            annotated,
            safety,
            (x + 12, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )
    return annotated


def _draw_status_lines(image, status_lines):
    if not status_lines:
        return image

    for index, line in enumerate(status_lines):
        y = 32 + index * 30
        cv2.putText(
            image,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
        )
        cv2.putText(
            image,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    return image


def draw_selected_candidate(
    image,
    candidates,
    selected_id,
    fallback_used=False,
    status_lines=None,
    selected_pixel=None,
    vlm_selected_id=None,
):
    annotated = draw_candidate_points(image, candidates)
    selected = next((c for c in candidates if c["id"] == selected_id), None)

    if vlm_selected_id:
        vlm_selected = next((c for c in candidates if c["id"] == vlm_selected_id), None)
        if vlm_selected is not None:
            vx, vy = vlm_selected["pixel"]
            cv2.circle(annotated, (vx, vy), 24, (255, 180, 0), 4)
            cv2.putText(
                annotated,
                f"VLM {vlm_selected_id}",
                (vx + 30, vy - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                4,
            )
            cv2.putText(
                annotated,
                f"VLM {vlm_selected_id}",
                (vx + 30, vy - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 180, 0),
                2,
            )

    if selected is not None or selected_pixel is not None:
        if selected_pixel is not None:
            x, y = selected_pixel
        else:
            x, y = selected["pixel"]
        label = f"FINAL {selected_id}"
        if fallback_used:
            label += " (fallback)"

        cv2.circle(annotated, (x, y), 28, (0, 0, 255), 6)
        cv2.circle(annotated, (x, y), 20, (0, 255, 0), 4)
        cv2.putText(
            annotated,
            label,
            (x + 35, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            5,
        )
        cv2.putText(
            annotated,
            label,
            (x + 35, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    return _draw_status_lines(annotated, status_lines)


def _extract_json_object(vlm_response):
    if not isinstance(vlm_response, str):
        return None
    text = vlm_response.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("```")
        ).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def parse_candidate_choice(vlm_response, candidates):
    fallback = candidates[7]
    from ..clients.vlm_client import VLMClient

    parsed = VLMClient.parse_candidate_selection(
        vlm_response,
        valid_choices=[candidate["id"] for candidate in candidates],
        fallback_choice=fallback["id"],
    )
    choice = parsed["choice"]
    candidate = next((c for c in candidates if c["id"] == choice), fallback)
    return candidate, parsed


class SearchMemory:
    def __init__(self, history_size=8, failure_threshold=3, consecutive_failure_threshold=2):
        self.recent_history = deque(maxlen=history_size)
        self.failed_search_count = Counter()
        self.successful_observation_count = Counter()
        self.failure_threshold = failure_threshold
        self.consecutive_failure_threshold = consecutive_failure_threshold
        self.success_count_key_field = "vlm_choice"

    def record(
        self,
        vlm_choice,
        ieve_intended_choice,
        final_action_choice,
        eve_state,
        target_visible,
        confidence,
        stable_count,
        lost_count,
    ):
        vlm_choice = self._normalize_choice(vlm_choice)
        ieve_intended_choice = self._normalize_choice(ieve_intended_choice)
        final_action_choice = self._normalize_choice(final_action_choice)
        eve_state = getattr(eve_state, "value", eve_state)
        eve_state = str(eve_state or "")
        confidence = self._normalize_confidence(confidence)
        target_visible = bool(target_visible)
        observed_target = target_visible and confidence >= 0.5

        entry = {
            "vlm_choice": vlm_choice,
            "ieve_intended_choice": ieve_intended_choice,
            "final_action_choice": final_action_choice,
            "eve_state": eve_state,
            "target_visible": target_visible,
            "confidence": confidence,
            "stable_count": int(stable_count or 0),
            "lost_count": int(lost_count or 0),
        }
        self.recent_history.append(entry)

        if observed_target:
            success_key = vlm_choice
            if not success_key:
                success_key = ieve_intended_choice
            if success_key:
                self.successful_observation_count[success_key] += 1
        elif final_action_choice:
            self.failed_search_count[final_action_choice] += 1

        recent_failure_count = self._recent_failure_count(final_action_choice)
        consecutive_failure_count = self._consecutive_failure_count(final_action_choice)
        repeated_search_detected = (
            eve_state == "EXPLORE"
            and final_action_choice
            and (
                recent_failure_count >= self.failure_threshold
                or consecutive_failure_count >= self.consecutive_failure_threshold
            )
        )
        suggested_alternative = None
        if repeated_search_detected:
            suggested_alternative = self._suggest_adjacent(final_action_choice)

        return {
            "recent_choices": self._format_recent_choices(),
            "failed_top_candidates": self._top_failed_candidates(),
            "failed_search_count": dict(sorted(self.failed_search_count.items())),
            "successful_observation_count": dict(
                sorted(self.successful_observation_count.items())
            ),
            "successful_observation_key": self.success_count_key_field,
            "repeated_search_detected": repeated_search_detected,
            "suggested_alternative": suggested_alternative,
            "repeated_failed_choice": final_action_choice,
            "recent_failure_count": recent_failure_count,
            "consecutive_failure_count": consecutive_failure_count,
            "failure_threshold": self.failure_threshold,
        }

    def log(self, snapshot):
        print("SearchMemory:")
        print(f"recent_choices: {snapshot['recent_choices']}")
        print(f"failed_search_count: {snapshot['failed_search_count']}")
        print(
            "successful_observation_count: "
            f"{snapshot['successful_observation_count']} "
            f"(key={snapshot['successful_observation_key']})"
        )
        print(f"repeated_search_detected: {snapshot['repeated_search_detected']}")
        print(f"suggested_alternative: {snapshot['suggested_alternative']}")
        print(
            "weak_guidance_stats: "
            f"failed_choice={snapshot['repeated_failed_choice']} "
            f"recent_fail={snapshot['recent_failure_count']} "
            f"consecutive_fail={snapshot['consecutive_failure_count']}"
            f" threshold={snapshot['failure_threshold']}"
        )

    def _recent_failure_count(self, choice):
        return sum(
            1
            for item in self.recent_history
            if item["final_action_choice"] == choice
            and (
                not item["target_visible"]
                or item["confidence"] < 0.5
            )
        )

    def _consecutive_failure_count(self, choice):
        count = 0
        for item in reversed(self.recent_history):
            if (
                item["final_action_choice"] == choice
                and (
                    not item["target_visible"]
                    or item["confidence"] < 0.5
                )
            ):
                count += 1
                continue
            break
        return count

    def _suggest_adjacent(self, choice):
        for candidate_id in self._adjacent_choices(choice):
            if (
                self.failed_search_count[candidate_id] < self.failure_threshold
                and self._recent_failure_count(candidate_id) < self.failure_threshold
            ):
                return candidate_id
        return None

    def _top_failed_candidates(self, limit=3):
        return self.failed_search_count.most_common(limit)

    def _adjacent_choices(self, choice):
        try:
            index = int(str(choice).strip().upper().lstrip("P"))
        except ValueError:
            return []
        if index < 1 or index > 15:
            return []

        row = (index - 1) // 5
        col = (index - 1) % 5
        neighbors = []
        for next_row, next_col in (
            (row, col - 1),
            (row, col + 1),
            (row - 1, col),
            (row + 1, col),
        ):
            if 0 <= next_row < 3 and 0 <= next_col < 5:
                neighbors.append(f"P{next_row * 5 + next_col + 1}")
        return neighbors

    def _format_recent_choices(self):
        return [
            (
                f"{item['eve_state']}: "
                f"vlm={item['vlm_choice']} "
                f"ieve={item['ieve_intended_choice']} "
                f"final={item['final_action_choice']} "
                f"vis={item['target_visible']} "
                f"conf={item['confidence']:.2f}"
            )
            for item in self.recent_history
        ]

    @staticmethod
    def _normalize_choice(choice):
        return str(choice or "").strip().upper()

    @staticmethod
    def _normalize_confidence(confidence):
        try:
            return max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            return 0.0


def replace_blocked_choice(intended_candidate, candidates):
    safety_candidate = _candidate_by_id(candidates, intended_candidate.get("id"))
    check_candidate = safety_candidate or intended_candidate
    before_choice = intended_candidate.get("id", "UNKNOWN")
    before_safety = check_candidate.get("safety", "unknown")

    if before_safety != "blocked":
        before = f"{before_choice} ({before_safety})"
        return intended_candidate, before, before, False

    replacement = _nearest_candidate_with_safety(check_candidate, candidates, "safe")
    if replacement is None:
        replacement = _nearest_candidate_with_safety(check_candidate, candidates, "unknown")
    if replacement is None:
        replacement = _center_candidate(candidates)

    after_choice = replacement.get("id", "UNKNOWN")
    before = f"{before_choice} ({before_safety})"
    after = f"{after_choice} ({replacement.get('safety', 'unknown')})"
    return replacement, before, after, after_choice != before_choice


def replace_ground_risk_choice(
    projector, candidate, candidates, depth_for_projection, velocity_multiplier
):
    risk, vector, vz = _candidate_ground_risk(
        projector, candidate, depth_for_projection, velocity_multiplier
    )
    before_id = candidate.get("id", "UNKNOWN")
    if not risk:
        return candidate, before_id, before_id, False, vector, vz

    if candidate.get("safety") != "blocked":
        candidate["safety"] = "ground_risk"
    return candidate, before_id, before_id, False, vector, vz


def _nearest_upper_non_ground_risk(
    projector, reference, candidates, depth_for_projection, velocity_multiplier, safety
):
    ref_y = reference.get("point", [1000, 0])[0]
    ref_x, ref_py = reference.get("pixel", (0, 0))
    matching = []
    for candidate in candidates:
        if candidate.get("id") == reference.get("id"):
            continue
        if candidate.get("safety") != safety:
            continue
        if candidate.get("point", [1000, 0])[0] >= ref_y:
            continue
        risk, _vector, _vz = _candidate_ground_risk(
            projector, candidate, depth_for_projection, velocity_multiplier
        )
        if risk:
            candidate["safety"] = "ground_risk"
            continue
        matching.append(candidate)

    if not matching:
        return None

    return min(
        matching,
        key=lambda candidate: (
            candidate["pixel"][0] - ref_x
        ) ** 2 + (candidate["pixel"][1] - ref_py) ** 2,
    )


def _non_blocked_non_ground_risk_fallback(
    projector, reference, candidates, depth_for_projection, velocity_multiplier
):
    fallback_candidates = [_center_candidate(candidates)] + candidates
    for candidate in fallback_candidates:
        if candidate.get("safety") == "blocked":
            continue
        risk, _vector, _vz = _candidate_ground_risk(
            projector, candidate, depth_for_projection, velocity_multiplier
        )
        if risk:
            if candidate.get("safety") != "blocked":
                candidate["safety"] = "ground_risk"
            continue
        return candidate
    return None


def _candidate_ground_risk(projector, candidate, depth_for_projection, velocity_multiplier):
    vector = _candidate_motion_vector(projector, candidate, depth_for_projection)
    distance_xy = float(np.sqrt(vector["x3d"] ** 2 + vector["y3d"] ** 2))
    if distance_xy <= 0.01 or velocity_multiplier <= 0:
        return False, vector, 0.0

    velocity_scale = getattr(projector.action_space, "base_velocity", 2.0) * velocity_multiplier
    vz = (-vector["z3d"] / distance_xy) * velocity_scale
    return vz > GROUND_RISK_VZ_THRESHOLD_MPS, vector, vz


def _candidate_motion_vector(projector, candidate, depth_for_projection):
    y, x = candidate["point"]
    pixel_x = int((x / 1000.0) * projector.image_width)
    pixel_y = int((y / 1000.0) * projector.image_height)
    x3d, y3d, z3d = projector.reverse_project_point(
        (pixel_x, pixel_y), depth=depth_for_projection
    )
    return {
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "x3d": x3d,
        "y3d": y3d,
        "z3d": z3d,
    }


def _clamp_yaw_motion_vector(x3d, y3d, deadband_deg=YAW_DEADBAND_DEG, max_yaw_step_deg=MAX_YAW_STEP_DEG):
    distance_xy = float(np.sqrt(x3d**2 + y3d**2))
    if distance_xy <= 0.01:
        return x3d, y3d, 0.0, 0.0, False

    raw_target_angle = float(math.degrees(math.atan2(x3d, y3d)))
    if raw_target_angle > 180.0:
        raw_target_angle -= 360.0
    elif raw_target_angle < -180.0:
        raw_target_angle += 360.0

    if abs(raw_target_angle) <= deadband_deg:
        return x3d, y3d, raw_target_angle, raw_target_angle, False

    clamped_yaw_angle = max(-max_yaw_step_deg, min(max_yaw_step_deg, raw_target_angle))
    yaw_was_clamped = not np.isclose(clamped_yaw_angle, raw_target_angle)
    clamped_radians = math.radians(clamped_yaw_angle)
    clamped_x3d = math.sin(clamped_radians) * distance_xy
    clamped_y3d = math.cos(clamped_radians) * distance_xy
    return clamped_x3d, clamped_y3d, raw_target_angle, clamped_yaw_angle, yaw_was_clamped


def _nearest_candidate_with_safety(reference, candidates, safety):
    matching = [candidate for candidate in candidates if candidate.get("safety") == safety]
    if not matching:
        return None

    ref_x, ref_y = reference.get("pixel", (0, 0))
    return min(
        matching,
        key=lambda candidate: (
            candidate["pixel"][0] - ref_x
        ) ** 2 + (candidate["pixel"][1] - ref_y) ** 2,
    )


def _candidate_by_id(candidates, candidate_id):
    return next((candidate for candidate in candidates if candidate["id"] == candidate_id), None)


def _center_candidate(candidates):
    return next((candidate for candidate in candidates if candidate["id"] == "P8"), candidates[7])


def _format_depth(depth):
    if depth is None or not np.isfinite(depth):
        return "unknown"
    return f"{depth:.2f}"


def _log_depth_range(depth_image):
    try:
        depth_array = np.asarray(depth_image)
        finite = depth_array[np.isfinite(depth_array)]
        if finite.size == 0:
            print("Depth image range: no finite values")
            return
        print(
            f"Depth image range: min={float(np.min(finite)):.2f}, "
            f"max={float(np.max(finite)):.2f}"
        )
    except Exception as e:
        print(f"Depth image range: unavailable ({e})")


class AirSimActionProjector(ActionProjector):
    def __init__(
        self,
        image_width=1920,
        image_height=1080,
        adaptive_mode=False,
        config_path="config_airsim.yaml",
    ):
        super().__init__(image_width, image_height, config_path)

        self.adaptive_mode = adaptive_mode

        self.action_space = AirSimDroneActionSpace(n_samples=8)
        self.ieve_lite = SPFIEVELite()
        self.search_memory = SearchMemory()
        self._pending_search_weak_guidance = None

        print(
            f"[AirSimActionProjector] Initialized with {self.api_provider} provider using model: {self.model_name}"
        )

    def _determine_model_name(self):
        if self.custom_model:
            return self.custom_model

        if self.api_provider == "openai":
            return "google/gemini-2.5-flash"
        else:
            return "gemini-2.5-flash"

    def get_vlm_points(
        self, image: np.ndarray, instruction: str, **kwargs
    ) -> List[ActionPoint]:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        try:
            actions = [self._get_single_action(image, instruction, **kwargs)]

            if actions and actions[0] is not None:
                viz_image = image.copy()

                for i, action in enumerate(actions, 1):
                    cv2.circle(
                        viz_image,
                        (int(action.screen_x), int(action.screen_y)),
                        10,
                        (0, 255, 0),
                        -1,
                    )

                    cv2.putText(
                        viz_image,
                        f"{i}: ({action.dx:.1f}, {action.dy:.1f}, {action.dz:.1f})",
                        (int(action.screen_x) + 15, int(action.screen_y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                save_path = f"{self.output_dir}/decision_{timestamp}.jpg"
                cv2.imwrite(save_path, viz_image)

                decision_data = {
                    "timestamp": timestamp,
                    "mode": "airsim",
                    "adaptive_mode": self.adaptive_mode,
                    "instruction": instruction,
                    "actions": [],
                }

                for action in actions:
                    action_data = {
                        "dx": action.dx,
                        "dy": action.dy,
                        "dz": action.dz,
                        "screen_x": action.screen_x,
                        "screen_y": action.screen_y,
                    }

                    if (
                        hasattr(action, "adaptive_depth")
                        and action.adaptive_depth is not None
                    ):
                        action_data["adaptive_depth"] = action.adaptive_depth
                    if hasattr(action, "vlm_depth") and action.vlm_depth is not None:
                        action_data["vlm_depth"] = action.vlm_depth

                    decision_data["actions"].append(action_data)

                with open(f"{self.output_dir}/decision_{timestamp}.json", "w") as f:
                    json.dump(decision_data, f, indent=2)

            return actions

        except Exception as e:
            print(f"Error getting points: {e}")
            return []

    def _get_single_action(
        self, image: np.ndarray, instruction: str, **kwargs
    ) -> ActionPoint:
        candidates = generate_candidate_points(image)
        attach_candidate_safety(candidates, kwargs.get("depth_image"))
        vlm_image = draw_candidate_points(image, candidates)
        debug_path = f"{self.output_dir}/debug_vlm_input.jpg"
        cv2.imwrite(debug_path, vlm_image)
        candidate_lines = "\n".join(
            f'- {candidate["id"]}: [y, x] = {candidate["point"]}, '
            f'depth={_format_depth(candidate.get("depth"))}, '
            f'safety={candidate.get("safety", "unknown")}'
            for candidate in candidates
        )
        search_weak_guidance = self._consume_search_weak_guidance(candidates)

        prompt = f"""no thought process, no explanations, only JSON output with the chosen candidate and its details.
You are a drone navigation expert analyzing a drone camera view.

Task: {instruction}

The image contains labeled candidate waypoints.
Only choose one point from the labeled candidates.
Do not output free coordinates.

Return in this exact JSON format:
{{"choice": "P8", "target_visible": false, "confidence": 0.0, "reason": "brief reason"}}

Valid choices and original [y, x] coordinates:
{candidate_lines}
{search_weak_guidance}

IMPORTANT:
- choice must match one labeled candidate
- Prefer candidates marked safety=safe.
- Avoid candidates marked safety=blocked unless no other direction is reasonable.
- Candidates marked safety=unknown are allowed, but safe is preferred.
- target_visible must be true only when the described target is visible in the current image
- confidence must be a number from 0.0 to 1.0 for the target visibility/choice
- If the target is visible near the image boundary, choose the candidate closest to the target direction, even if it is not exactly on the target.
- For search tasks like find/locate/look for, choose the candidate that moves the drone/camera toward the target direction.
- If the target is not visible, set target_visible to false, confidence to 0.0, and choose the best exploratory direction.
- Keep reason under 12 words.
- Output only JSON, without markdown or extra text"""

        try:
            response_text = self.vlm_client.generate_response(prompt, vlm_image)
            print("\n=== VLM RAW OUTPUT ===")
            print(response_text)
            print("=== END VLM RAW OUTPUT ===\n")

            candidate, parsed_selection = parse_candidate_choice(response_text, candidates)
            vlm_choice = parsed_selection["choice"]
            target_visible = bool(parsed_selection["target_visible"])
            confidence = float(parsed_selection["confidence"])
            reason = parsed_selection["reason"]
            used_fallback = bool(parsed_selection["fallback_used"])

            decision = self.ieve_lite.update_and_select(
                candidates=candidates,
                current_candidate=candidate,
                target_visible=target_visible,
                confidence=confidence,
                target_world_position=kwargs.get("target_world_position"),
                image_shape=image.shape[:2],
            )
            intended_candidate = decision.selected_candidate
            final_candidate, safety_before, safety_after, blocked_replaced = (
                replace_blocked_choice(intended_candidate, candidates)
            )
            candidate = final_candidate
            after_depth_safety_id = candidate["id"]
            used_fallback = used_fallback or decision.used_center_fallback

            vlm_depth = 4
            if self.adaptive_mode:
                adaptive_depth = self._calculate_adaptive_depth(vlm_depth)
                depth_for_projection = adaptive_depth
                velocity_multiplier = adaptive_depth
            else:
                adaptive_depth = None
                depth_for_projection = vlm_depth / 10.0 * 2.0
                velocity_multiplier = 1.0

            (
                candidate,
                ground_risk_before,
                ground_risk_after,
                ground_risk_replaced,
                ground_risk_vector,
                ground_risk_vz,
            ) = replace_ground_risk_choice(
                self, candidate, candidates, depth_for_projection, velocity_multiplier
            )
            ground_risk = ground_risk_vz > GROUND_RISK_VZ_THRESHOLD_MPS
            _final_risk, final_vector, _final_vz = _candidate_ground_risk(
                self, candidate, depth_for_projection, velocity_multiplier
            )

            selected_debug_path = f"{self.output_dir}/debug_vlm_selected.jpg"
            final_vs_vlm_reasons = []
            if intended_candidate["id"] != vlm_choice:
                final_vs_vlm_reasons.append("ieve_override")
            if blocked_replaced:
                final_vs_vlm_reasons.append("blocked_replaced")
            if ground_risk_replaced:
                final_vs_vlm_reasons.append("ground_risk_replaced")
            if used_fallback:
                final_vs_vlm_reasons.append("fallback_used")
            if candidate["id"] == vlm_choice:
                final_vs_vlm_reason = "same_as_vlm"
            elif final_vs_vlm_reasons:
                final_vs_vlm_reason = ",".join(final_vs_vlm_reasons)
            else:
                final_vs_vlm_reason = "final_diff_unknown"
            search_memory_snapshot = self.search_memory.record(
                vlm_choice=vlm_choice,
                ieve_intended_choice=intended_candidate["id"],
                final_action_choice=candidate["id"],
                eve_state=decision.eve_state.value,
                target_visible=target_visible,
                confidence=confidence,
                stable_count=self.ieve_lite.target_memory.stable_count,
                lost_count=self.ieve_lite.target_memory.lost_count,
            )
            self._update_search_weak_guidance(search_memory_snapshot)
            status_lines = [
                f"EVE: {decision.eve_state.value}",
                f"visible: {target_visible} conf: {confidence:.2f}",
                f"VLM selected: {vlm_choice}",
                f"IEVE intended: {intended_candidate['id']}",
                f"IEVE source: {decision.ieve_output_source}",
                f"Memory last: {self.ieve_lite.target_memory.last_seen_choice}",
                f"After depth safety: {after_depth_safety_id}",
                f"Final executed: {candidate['id']}",
                f"Final vs VLM: {final_vs_vlm_reason}",
                f"fallback: {used_fallback} memory: {decision.using_memory_target}",
                f"SM failed top: {search_memory_snapshot['failed_top_candidates']}",
                f"SM repeated: {search_memory_snapshot['repeated_search_detected']}",
                f"SM suggested: {search_memory_snapshot['suggested_alternative']}",
            ]
            if blocked_replaced:
                status_lines.append("blocked replaced")
            if ground_risk_replaced:
                status_lines.append("ground risk replaced")
            selected_debug = draw_selected_candidate(
                image,
                candidates,
                candidate["id"],
                used_fallback,
                status_lines,
                selected_pixel=candidate.get("pixel"),
                vlm_selected_id=vlm_choice,
            )
            cv2.imwrite(selected_debug_path, selected_debug)
            self.ieve_lite.log_decision(
                vlm_choice=vlm_choice,
                target_visible=target_visible,
                confidence=confidence,
                decision=decision,
            )
            print(f"IEVE intended choice: {intended_candidate['id']}")
            print(f"Safety check before: {safety_before}")
            print(f"Safety check after: {safety_after}")
            print(f"Blocked choice replaced: {blocked_replaced}")
            print(f"Ground risk check before: {ground_risk_before}")
            print(f"candidate dz={ground_risk_vector['z3d']:.3f}")
            print(f"candidate vz={ground_risk_vz:.3f}")
            print(f"Ground risk check: {ground_risk}")
            print("Ground risk replacement: disabled")
            print(f"Final action choice: {candidate['id']}")
            print(f"Final vs VLM reason: {final_vs_vlm_reason}")
            print(f"Fallback used: {used_fallback}")
            self.search_memory.log(search_memory_snapshot)
            print("Saved debug_vlm_selected.jpg")

            y, x = candidate["point"]
            pixel_x = final_vector["pixel_x"]
            pixel_y = final_vector["pixel_y"]
            x3d = final_vector["x3d"]
            y3d = final_vector["y3d"]
            z3d = final_vector["z3d"]
            vertical_velocity_before_clamp = _final_vz
            if ground_risk and vertical_velocity_before_clamp > 0:
                z3d = 0.0
            distance_xy = float(np.sqrt(x3d**2 + y3d**2))
            if distance_xy > 0.01:
                vertical_velocity_after_clamp = (
                    (-z3d / distance_xy)
                    * getattr(self.action_space, "base_velocity", 2.0)
                    * velocity_multiplier
                )
            else:
                vertical_velocity_after_clamp = 0.0
            print(
                f"Vertical velocity before clamp: {vertical_velocity_before_clamp:.3f}"
            )
            print(
                f"Vertical velocity after clamp: {vertical_velocity_after_clamp:.3f}"
            )
            x3d, y3d, raw_target_angle, clamped_yaw_angle, yaw_was_clamped = (
                _clamp_yaw_motion_vector(x3d, y3d)
            )
            print(f"[YAW DEBUG] raw target_angle={raw_target_angle:.1f}")
            print(f"[YAW DEBUG] clamped_yaw_angle={clamped_yaw_angle:.1f}")
            print(f"[YAW DEBUG] max_yaw_step_deg={MAX_YAW_STEP_DEG:.1f}")
            print(f"[YAW DEBUG] yaw_was_clamped={yaw_was_clamped}")

            action = ActionPoint(
                dx=x3d,
                dy=y3d,
                dz=z3d,
                action_type="move",
                screen_x=pixel_x,
                screen_y=pixel_y,
            )

            if self.adaptive_mode:
                action.adaptive_depth = adaptive_depth
                action.vlm_depth = vlm_depth

            action.candidate_id = candidate["id"]
            action.vlm_choice = vlm_choice
            action.vlm_reason = reason
            action.target_visible = target_visible
            action.vlm_confidence = confidence
            action.eve_state = decision.eve_state.value
            action.using_memory_target = decision.using_memory_target
            action.ieve_intended_choice = intended_candidate["id"]
            action.blocked_choice_replaced = blocked_replaced
            action.selected_point = candidate["point"]

            print(f"\nIdentified single action: {candidate['id']} - {reason}")
            print(f"2D Normalized: ({x}, {y})")
            print(f"2D Pixels: ({pixel_x}, {pixel_y})")
            print(f"VLM Depth: {vlm_depth}/10")
            if self.adaptive_mode:
                print(f"Adaptive Depth: {adaptive_depth:.2f}")
            print(f"3D Vector: ({x3d:.2f}, {y3d:.2f}, {z3d:.2f})")

            return action

        except Exception as e:
            print(f"Error in single action mode: {e}")
            print("Full response:")
            if "response_text" in locals():
                print(response_text)
            return None

    def _consume_search_weak_guidance(self, candidates):
        hint = self._pending_search_weak_guidance
        self._pending_search_weak_guidance = None
        if not hint:
            return ""

        valid_choices = {str(candidate.get("id", "")).upper() for candidate in candidates}
        suggested = str(hint.get("suggested_alternative", "")).upper()
        if suggested not in valid_choices:
            print(
                "[SearchMemory WeakGuidance] skipped invalid suggestion: "
                f"{suggested}"
            )
            return ""

        print(
            "[SearchMemory WeakGuidance] advisory only: "
            f"previous_failed={hint.get('failed_choice')} "
            f"suggested_alternative={suggested} "
            f"failed_count={hint.get('failed_count')} "
            f"threshold={hint.get('failure_threshold')} "
            f"recent_fail={hint.get('recent_failure_count')} "
            f"consecutive_fail={hint.get('consecutive_failure_count')}"
        )
        return (
            "\nOptional weak search-memory hint, advisory only:\n"
            f"- Previous EXPLORE search repeatedly failed at {hint.get('failed_choice')}.\n"
            f"- If the target is still not visible, consider {suggested} as an alternative.\n"
            "- Do not use this hint when the target is visible, and do not ignore safety labels.\n"
        )

    def _update_search_weak_guidance(self, snapshot):
        suggested = snapshot.get("suggested_alternative")
        if not snapshot.get("repeated_search_detected") or not suggested:
            self._pending_search_weak_guidance = None
            print("[SearchMemory WeakGuidance] none")
            return

        failed_choice = snapshot.get("repeated_failed_choice")
        failed_counts = snapshot.get("failed_search_count") or {}
        failed_count = failed_counts.get(failed_choice, 0)
        threshold = snapshot.get("failure_threshold", self.search_memory.failure_threshold)
        if failed_count < threshold:
            self._pending_search_weak_guidance = None
            print(
                "[SearchMemory WeakGuidance] none: "
                f"failed_choice={failed_choice} "
                f"failed_count={failed_count} threshold={threshold}"
            )
            return

        self._pending_search_weak_guidance = {
            "failed_choice": failed_choice,
            "suggested_alternative": str(suggested).upper(),
            "failed_count": failed_count,
            "failure_threshold": threshold,
            "recent_failure_count": snapshot.get("recent_failure_count", 0),
            "consecutive_failure_count": snapshot.get("consecutive_failure_count", 0),
        }
        print(
            "[SearchMemory WeakGuidance] queued for next frame: "
            f"failed_choice={failed_choice} "
            f"suggested_alternative={suggested} "
            f"failed_count={failed_count} "
            f"threshold={threshold} "
            f"recent_fail={snapshot.get('recent_failure_count')} "
            f"consecutive_fail={snapshot.get('consecutive_failure_count')} "
            f"failed_search_count={failed_counts}"
        )

    def _calculate_adaptive_depth(self, vlm_depth):
        if vlm_depth <= 2:
            adaptive_depth = 0
            print(
                f"AirSim: VLM depth {vlm_depth}/10 → Adaptive depth {adaptive_depth} (No movement - too close)"
            )
        elif vlm_depth <= 5:
            adaptive_depth = (vlm_depth / 10.0) * 2
            print(
                f"AirSim: VLM depth {vlm_depth}/10 → Adaptive depth {adaptive_depth:.2f} (Careful movement)"
            )
        else:
            adaptive_depth = 1.0 + (vlm_depth - 5) / 5.0
            print(
                f"AirSim: VLM depth {vlm_depth}/10 → Adaptive depth {adaptive_depth:.2f} (Normal movement)"
            )

        return adaptive_depth
