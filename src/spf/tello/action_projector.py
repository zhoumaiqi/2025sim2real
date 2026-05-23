import cv2
import numpy as np
from ..base.action_projector import ActionProjector
from ..base.drone_space import ActionPoint
from ..clients import RemoteDepthProClient
from .drone_space import TelloDroneActionSpace
from typing import List, Tuple
import time
import json
import re
from .depth_safety import (
    UNKNOWN,
    apply_front_risk_to_motion,
    attach_depth_safety,
    format_depth_prompt_hint,
    replace_blocked_candidate,
)


TELLO_CANDIDATE_RATIOS = [
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
TELLO_DEFAULT_CANDIDATE_CHOICE = "P8"
TELLO_DEFAULT_CANDIDATE_DEPTH = 6


def _coerce_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    if value is None:
        return default
    return bool(value)


def _coerce_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def generate_tello_candidate_points(image):
    height, width = image.shape[:2]
    candidates = []
    for idx, (x_ratio, y_ratio) in enumerate(TELLO_CANDIDATE_RATIOS, 1):
        candidates.append(
            {
                "id": f"P{idx}",
                "point": [int(round(y_ratio * 1000)), int(round(x_ratio * 1000))],
                "pixel": (int(round(x_ratio * width)), int(round(y_ratio * height))),
            }
        )
    return candidates


def draw_tello_candidate_points(image, candidates, selected_id=None, status_lines=None):
    annotated = image.copy()
    for candidate in candidates:
        x, y = candidate["pixel"]
        safety = candidate.get("safety", UNKNOWN)
        if candidate["id"] == selected_id:
            color = (0, 255, 0)
        elif safety == "blocked":
            color = (80, 80, 255)
        elif safety == "safe":
            color = (0, 255, 255)
        else:
            color = (180, 180, 180)

        cv2.circle(annotated, (x, y), 14, color, -1)
        cv2.circle(annotated, (x, y), 14, (0, 0, 0), 2)
        cv2.putText(
            annotated,
            candidate["id"],
            (x + 18, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            safety,
            (x + 18, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )
    if status_lines:
        for index, line in enumerate(status_lines):
            y = 28 + index * 24
            cv2.putText(
                annotated,
                line,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                annotated,
                line,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
    return annotated


def parse_tello_candidate_selection(response_text, candidates):
    valid_choices = {candidate["id"] for candidate in candidates}
    parsed_choice = TELLO_DEFAULT_CANDIDATE_CHOICE
    target_visible = False
    confidence = 0.0
    reason = "fallback"
    fallback_used = False

    try:
        response_data = json.loads(response_text)
        if isinstance(response_data, list):
            response_data = response_data[0] if response_data else {}
        if not isinstance(response_data, dict):
            response_data = {}
    except json.JSONDecodeError:
        response_data = {}
        choice_match = re.search(
            r'"choice"\s*:\s*"(P\d+)"', response_text, re.IGNORECASE
        )
        visible_match = re.search(
            r'"target_visible"\s*:\s*(true|false)', response_text, re.IGNORECASE
        )
        confidence_match = re.search(
            r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', response_text
        )
        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', response_text)
        if choice_match:
            response_data["choice"] = choice_match.group(1).upper()
        if visible_match:
            response_data["target_visible"] = visible_match.group(1).lower() == "true"
        if confidence_match:
            response_data["confidence"] = confidence_match.group(1)
        if reason_match:
            response_data["reason"] = reason_match.group(1)
        fallback_used = True

    raw_choice = str(
        response_data.get("choice", TELLO_DEFAULT_CANDIDATE_CHOICE)
    ).strip().upper()
    if raw_choice in valid_choices:
        parsed_choice = raw_choice
    else:
        parsed_choice = TELLO_DEFAULT_CANDIDATE_CHOICE
        fallback_used = True

    target_visible = _coerce_bool(response_data.get("target_visible"), default=False)
    confidence = max(
        0.0, min(1.0, _coerce_float(response_data.get("confidence"), default=0.0))
    )
    reason = str(response_data.get("reason", "fallback")).strip() or "fallback"

    candidate = next(
        (item for item in candidates if item["id"] == parsed_choice),
        next(item for item in candidates if item["id"] == TELLO_DEFAULT_CANDIDATE_CHOICE),
    )
    return {
        "candidate": candidate,
        "choice": parsed_choice,
        "target_visible": target_visible,
        "confidence": confidence,
        "reason": reason,
        "fallback_used": fallback_used,
    }


class TelloActionProjector(ActionProjector):
    """
    Tello-specific action projector with mode-specific processing
    """

    def __init__(
        self,
        image_width=960,
        image_height=720,
        mode="adaptive_mode",
        config_path="config_tello.yaml",
    ):
        """
        Initialize the Tello projector with mode-specific settings

        Args:
            image_width (int): Width of the input image
            image_height (int): Height of the input image
            mode (str): Operational mode ("adaptive_mode" or "obstacle_mode")
            config_path (str): Path to configuration file
        """
        # Store operational mode FIRST (needed by parent's _determine_model_name)
        self.operational_mode = mode

        super().__init__(image_width, image_height, config_path)

        # Use Tello-specific action space
        self.action_space = TelloDroneActionSpace(n_samples=8)
        self.depth_client = RemoteDepthProClient.from_config(
            self.config.get("depth_pro")
        )

        print(
            f"[TelloActionProjector] Initialized in {mode} with {self.api_provider} provider using model: {self.model_name}"
        )
        if self.depth_client is not None:
            print(
                "[TelloActionProjector] Remote Depth Pro enabled: "
                f"{self.depth_client.endpoint}"
            )

    def _determine_model_name(self):
        """Determine model name based on provider, mode, and custom setting"""
        if self.custom_model:
            return self.custom_model

        # Default models based on provider and mode
        if self.api_provider == "openai":
            if self.operational_mode == "obstacle_mode":
                return "google/gemini-2.5-pro"
            else:
                return "google/gemini-2.5-flash"
        else:  # gemini provider
            if self.operational_mode == "obstacle_mode":
                return "gemini-2.5-pro"
            else:
                return "gemini-2.0-flash"

    def reverse_project_point(
        self, point_2d: Tuple[int, int], depth: float = 2
    ) -> Tuple[float, float, float]:
        """Project 2D image point back to 3D space with Tello-specific parameters"""
        # Set reference point at 35% from top of frame
        reference_y = self.image_height * 0.35

        # Center and normalize coordinates
        x_normalized = (point_2d[0] - self.image_width / 2) / (self.image_width / 2)
        y_normalized = (reference_y - point_2d[1]) / (self.image_height / 2)

        # Adjust depth based on vertical position (closer if lower in image)
        depth_factor = 1.0 + (y_normalized * 0.5)
        depth = depth * depth_factor

        # Calculate 3D coordinates with optimized depth
        x = depth * x_normalized * np.tan(np.radians(self.fov_horizontal / 2))
        z = depth * y_normalized * np.tan(np.radians(self.fov_vertical / 2))
        y = depth

        return (x, y, z)

    def calculate_adjusted_depth(self, vlm_depth):
        """
        Convert a nominal 1-10 depth score into a movement depth.
        """
        if vlm_depth <= 2:
            adjusted_depth = 0.5
            print(f"Tello: candidate depth {vlm_depth}/10 -> Adjusted depth {adjusted_depth}")
            return adjusted_depth

        base = (vlm_depth / 10.0) ** 2.0 * 8
        adjusted_depth = base
        print(
            f"Tello: candidate depth {vlm_depth}/10 -> Adjusted depth {adjusted_depth:.2f}"
        )
        return adjusted_depth

    def get_vlm_points(
        self, image: np.ndarray, instruction: str, tello_controller=None
    ) -> List[ActionPoint]:
        """Use VLM to identify points based on current mode and API provider"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        try:
            # Get single action from VLM with mode-specific processing
            if self.operational_mode == "obstacle_mode":
                print("\nin obstacle mode")
                actions = [self._get_single_action(image, instruction, tello_controller)]
            else:
                actions = [self._get_single_action(image, instruction)]

            actions = [action for action in actions if action is not None]

            if actions:
                print("\n actions in visualization part:")
                print("/n", actions)

                # Save visualization
                viz_image = image.copy()

                # Draw points on image
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

                    if (
                        self.operational_mode == "obstacle_mode"
                        and hasattr(action, "detected_obstacles")
                        and action.detected_obstacles
                    ):
                        for obstacle in action.detected_obstacles:
                            if "bounding_box" in obstacle:
                                ymin, xmin, ymax, xmax = obstacle["bounding_box"]
                                cv2.rectangle(
                                    viz_image,
                                    (int(xmin), int(ymin)),
                                    (int(xmax), int(ymax)),
                                    (0, 0, 255),
                                    2,
                                )
                                label = obstacle.get("label", "obstacle")
                                cv2.putText(
                                    viz_image,
                                    label,
                                    (int(xmin), int(ymin) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 0, 255),
                                    2,
                                )

                save_path = f"{self.output_dir}/decision_{timestamp}.jpg"
                cv2.imwrite(save_path, viz_image)

                decision_data = {
                    "timestamp": timestamp,
                    "mode": self.operational_mode,
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

                    if hasattr(action, "vlm_choice"):
                        action_data["choice"] = action.vlm_choice
                    if hasattr(action, "target_visible"):
                        action_data["target_visible"] = action.target_visible
                    if hasattr(action, "confidence"):
                        action_data["confidence"] = action.confidence
                    if hasattr(action, "fallback_used"):
                        action_data["fallback_used"] = action.fallback_used
                    if hasattr(action, "depth_front_risk"):
                        action_data["depth_front_risk"] = action.depth_front_risk
                    if hasattr(action, "depth_safety_source"):
                        action_data["depth_safety_source"] = action.depth_safety_source
                    if hasattr(action, "depth_original_choice"):
                        action_data["depth_original_choice"] = action.depth_original_choice
                    if hasattr(action, "depth_final_choice"):
                        action_data["depth_final_choice"] = action.depth_final_choice
                    if hasattr(action, "depth_choice_replaced"):
                        action_data["depth_choice_replaced"] = action.depth_choice_replaced
                    if hasattr(action, "depth_adjusted_m"):
                        action_data["depth_adjusted_m"] = action.depth_adjusted_m

                    if (
                        self.operational_mode == "obstacle_mode"
                        and hasattr(action, "detected_obstacles")
                        and action.detected_obstacles
                    ):
                        action_data["obstacles"] = action.detected_obstacles

                    decision_data["actions"].append(action_data)

                with open(f"{self.output_dir}/decision_{timestamp}.json", "w") as f:
                    json.dump(decision_data, f, indent=2)

            return actions

        except Exception as e:
            print(f"Error getting points: {e}")
            return []

    def _get_single_action(
        self, image: np.ndarray, instruction: str, tello_controller=None
    ) -> ActionPoint:
        """Get single next best action with mode-specific processing"""

        # Mode-specific processing
        if self.operational_mode == "obstacle_mode":
            print("\nFinished encoding image")
            print(
                f"[{self.api_provider.upper()}] Preparing API call at {time.strftime('%H:%M:%S')}"
            )
            api_start_time = time.time()

            if tello_controller:
                print(
                    f"[{self.api_provider.upper()}] Confirming intensive keepalive before API call"
                )
                tello_controller.start_intensive_keepalive()

            prompt = f"""You are a drone navigation expert analyzing a drone camera view.

        Task: {instruction}

        main task:
        1. Identify objects in the image that match the description "{instruction}".
        2. Then, select the MOST RELEVANT target object and place a "target point" DIRECTLY ON that object.
        sub task:
        3. Identify obstacles in the path, if necessary, "slighty" adjust the point.

        Return in this JSON format:
        {{
            "point": [y, x],
            "label": "action description",
            "obstacles": [
                    {{"bounding_box": [ymin, xmin, ymax, xmax], "label": "obstacle_description"}}
            ]
        }}

        Coordinate system:
        - x: 0-1000 scale (500=center, >500=right, <500=left)
        - y: 0-1000 scale (lower values=higher in image/sky)

        Notes:
        - "Pointing on the target" is the most important thing.
        - Prioritize the closest/largest matching object if multiple exist
        - Consider immediate obstacles and choose a safe path.
        - Aim for target's center.
        """
        else:
            candidates = generate_tello_candidate_points(image)
            depth_result = None
            depth_state = {
                "available": False,
                "front_risk": False,
                "roi_median_m": None,
            }
            if self.depth_client is not None:
                depth_result = self.depth_client.infer(
                    image,
                    candidate_ratios=TELLO_CANDIDATE_RATIOS,
                )
                depth_state = attach_depth_safety(candidates, depth_result)
                self._log_depth_state(candidates, depth_state)
            else:
                attach_depth_safety(candidates, None)
            vlm_image = draw_tello_candidate_points(image, candidates)
            debug_input_path = f"{self.output_dir}/debug_vlm_input.jpg"
            cv2.imwrite(debug_input_path, vlm_image)
            if self.depth_client is not None and self.depth_client.inject_depth_safety_into_prompt:
                candidate_lines = format_depth_prompt_hint(
                    candidates,
                    front_risk=depth_state["front_risk"],
                )
            else:
                candidate_lines = "\n".join(
                    f'- {candidate["id"]}: [y, x] = {candidate["point"]}'
                    for candidate in candidates
                )

            prompt = f"""no thought process, no explanations, only JSON output with the chosen candidate and its details.
You are a drone navigation expert analyzing a drone camera view.

Task: {instruction}

The image contains labeled candidate waypoints.
Only choose one point from the labeled candidates.
Do not output free coordinates.

Return in this exact JSON format:
{{"choice":"P8","target_visible":true,"confidence":0.9,"reason":"brief reason"}}

Valid choices and original [y, x] coordinates:
{candidate_lines}

IMPORTANT:
- choice must be exactly one of P1 to P15
- Never output free coordinates
- If the target is visible, choose the candidate closest to the target direction
- If the target is not visible, choose the most reasonable exploratory/search direction candidate
- target_visible must be true only when the target is visible in the current image
- confidence must be a number from 0.0 to 1.0
- Keep reason under 12 words
- Output only JSON, without markdown or extra text"""

        try:
            if self.operational_mode == "obstacle_mode":
                print(
                    f"[{self.api_provider.upper()}] Sending API request at {time.strftime('%H:%M:%S')}"
                )
                response_text = self.vlm_client.generate_response(prompt, image)
            else:
                response_text = self.vlm_client.generate_response(prompt, vlm_image)

            if self.operational_mode == "obstacle_mode":
                api_duration = time.time() - api_start_time
                print(
                    f"[{self.api_provider.upper()}] Response received in {api_duration:.2f} seconds"
                )

                if tello_controller:
                    tello_controller.stop_intensive_keepalive()

            from ..clients.vlm_client import VLMClient

            response_text = VLMClient.clean_response_text(response_text)

            print("\nraw_vlm_response:")
            print(response_text)

            if self.operational_mode == "obstacle_mode":
                try:
                    response_data = json.loads(response_text)
                    if not response_data:
                        raise ValueError("No data returned from VLM")

                    y, x = response_data["point"]
                    pixel_x = int((x / 1000.0) * self.image_width)
                    pixel_y = int((y / 1000.0) * self.image_height)

                    x3d, y3d, z3d = self.reverse_project_point(
                        (pixel_x, pixel_y), depth=1.1
                    )

                    action = ActionPoint(
                        dx=x3d,
                        dy=y3d,
                        dz=z3d,
                        action_type="move",
                        screen_x=pixel_x,
                        screen_y=pixel_y,
                    )

                    if "obstacles" in response_data:
                        obstacles = []
                        for obstacle in response_data["obstacles"]:
                            if "bounding_box" in obstacle:
                                ymin, xmin, ymax, xmax = obstacle["bounding_box"]
                                if max(obstacle["bounding_box"]) <= 1000:
                                    xmin = int((xmin / 1000.0) * self.image_width)
                                    ymin = int((ymin / 1000.0) * self.image_height)
                                    xmax = int((xmax / 1000.0) * self.image_width)
                                    ymax = int((ymax / 1000.0) * self.image_height)
                                obstacle["bounding_box"] = [ymin, xmin, ymax, xmax]
                            obstacles.append(obstacle)
                        action.detected_obstacles = obstacles

                    print(f"\nIdentified single action: {response_data.get('label')}")
                    print(f"2D Normalized: ({x}, {y})")
                    print(f"2D Pixels: ({pixel_x}, {pixel_y})")
                    print(f"3D Vector: ({x3d:.2f}, {y3d:.2f}, {z3d:.2f})")
                    if (
                        hasattr(action, "detected_obstacles")
                        and action.detected_obstacles
                    ):
                        print(f"Detected {len(action.detected_obstacles)} obstacles")

                    return action

                except json.JSONDecodeError as json_error:
                    print(
                        f"[{self.api_provider.upper()}] Error parsing JSON: {json_error}"
                    )
                    print(
                        f"[{self.api_provider.upper()}] Raw response text: {response_text}"
                    )

                    point_match = re.search(
                        r'"point":\s*\[(\d+),\s*(\d+)\]', response_text
                    )
                    if point_match:
                        print(
                            f"[{self.api_provider.upper()}] Attempting fallback point extraction with regex"
                        )
                        y, x = int(point_match.group(1)), int(point_match.group(2))
                        pixel_x = int((x / 1000.0) * self.image_width)
                        pixel_y = int((y / 1000.0) * self.image_height)
                        x3d, y3d, z3d = self.reverse_project_point(
                            (pixel_x, pixel_y), depth=1.1
                        )

                        action = ActionPoint(
                            dx=x3d,
                            dy=y3d,
                            dz=z3d,
                            action_type="move",
                            screen_x=pixel_x,
                            screen_y=pixel_y,
                        )
                        print(
                            f"[{self.api_provider.upper()}] Fallback action created: ({x3d:.2f}, {y3d:.2f}, {z3d:.2f})"
                        )
                        return action

                    raise
            else:
                parsed_selection = parse_tello_candidate_selection(response_text, candidates)
                selected_candidate = parsed_selection["candidate"]
                parsed_choice = parsed_selection["choice"]
                target_visible = parsed_selection["target_visible"]
                confidence = parsed_selection["confidence"]
                reason = parsed_selection["reason"]
                fallback_used = parsed_selection["fallback_used"]

                original_choice = selected_candidate["id"]
                depth_choice_replaced = False
                depth_replacement_reason = "depth_disabled"
                if self.depth_client is not None:
                    (
                        selected_candidate,
                        depth_choice_replaced,
                        depth_replacement_reason,
                    ) = replace_blocked_candidate(selected_candidate, candidates)

                selected_debug_path = f"{self.output_dir}/debug_vlm_selected.jpg"
                status_lines = [
                    f"Depth available: {depth_state['available']}",
                    f"Front risk: {depth_state['front_risk']}",
                    f"Original choice: {original_choice}",
                    f"Final choice: {selected_candidate['id']}",
                    f"Depth replace: {depth_choice_replaced}",
                    f"Reason: {depth_replacement_reason}",
                ]
                cv2.imwrite(
                    selected_debug_path,
                    draw_tello_candidate_points(
                        image,
                        candidates,
                        selected_id=selected_candidate["id"],
                        status_lines=status_lines,
                    ),
                )

                y, x = selected_candidate["point"]
                pixel_x, pixel_y = selected_candidate["pixel"]

                adjusted_depth = self.calculate_adjusted_depth(
                    TELLO_DEFAULT_CANDIDATE_DEPTH
                )
                x3d, y3d, z3d = self.reverse_project_point(
                    (pixel_x, pixel_y), depth=adjusted_depth
                )
                front_risk_forward_reduced = False
                front_risk_downward_clamped = False
                if self.depth_client is not None:
                    adjusted_depth, z3d, front_risk_forward_reduced, front_risk_downward_clamped = (
                        apply_front_risk_to_motion(
                            adjusted_depth=adjusted_depth,
                            z3d=z3d,
                            front_risk=depth_state["front_risk"],
                            reduce_forward_on_front_risk=self.depth_client.reduce_forward_on_front_risk,
                            clamp_downward_on_front_risk=self.depth_client.clamp_downward_on_front_risk,
                            min_motion_depth_m=self.depth_client.min_motion_depth_m,
                        )
                    )
                    x3d, y3d, z3d = self.reverse_project_point(
                        (pixel_x, pixel_y), depth=adjusted_depth
                    )
                    if front_risk_downward_clamped and z3d < 0:
                        z3d = 0.0

                action = ActionPoint(
                    dx=x3d,
                    dy=y3d,
                    dz=z3d,
                    action_type="move",
                    screen_x=pixel_x,
                    screen_y=pixel_y,
                )
                action.vlm_choice = parsed_choice
                action.target_visible = target_visible
                action.confidence = confidence
                action.fallback_used = fallback_used
                action.depth_front_risk = depth_state["front_risk"]
                action.depth_safety_source = "remote_depth_pro" if depth_state["available"] else "none"
                action.depth_original_choice = original_choice
                action.depth_final_choice = selected_candidate["id"]
                action.depth_choice_replaced = depth_choice_replaced
                action.depth_replacement_reason = depth_replacement_reason
                action.depth_adjusted_m = adjusted_depth

                print(f"parsed_choice: {parsed_choice}")
                print(f"target_visible: {target_visible}")
                print(f"confidence: {confidence:.2f}")
                print(f"selected_pixel: ({pixel_x}, {pixel_y})")
                print(f"fallback_used: {fallback_used}")
                print(f"selected_reason: {reason}")
                print(f"depth_front_risk: {depth_state['front_risk']}")
                print(f"depth_choice_replaced: {depth_choice_replaced}")
                print(f"depth_replacement_reason: {depth_replacement_reason}")
                print(
                    f"front_risk_forward_reduced: {front_risk_forward_reduced}"
                )
                print(
                    f"front_risk_downward_clamped: {front_risk_downward_clamped}"
                )
                print(f"Saved candidate input image: {debug_input_path}")
                print(f"Saved candidate selection image: {selected_debug_path}")

                print(f"\nIdentified single action: {selected_candidate['id']} - {reason}")
                print(f"2D Normalized: ({x}, {y})")
                print(f"2D Pixels: ({pixel_x}, {pixel_y})")
                print(
                    f"Depth estimation: {TELLO_DEFAULT_CANDIDATE_DEPTH}/10 (adjusted to {adjusted_depth:.2f})"
                )
                print(f"3D Vector: ({x3d:.2f}, {y3d:.2f}, {z3d:.2f})")

                return action

        except Exception as e:
            if self.operational_mode == "obstacle_mode":
                print(f"[{self.api_provider.upper()}] Error in API call: {e}")
                if "response_text" in locals():
                    print(f"[{self.api_provider.upper()}] Full response:")
                    print(response_text)
                else:
                    print(f"[{self.api_provider.upper()}] No response received from API")
            else:
                print(f"Error in single action mode: {e}")
                print("Full response:")
                if "response_text" in locals():
                    print(response_text)
            return None

    def _log_depth_state(self, candidates, depth_state):
        print("[DepthPro] candidate safety:")
        for candidate in candidates:
            print(
                f"  {candidate['id']}: "
                f"depth={candidate.get('depth')} "
                f"safety={candidate.get('safety')}"
            )
        print(
            "[DepthPro] summary: "
            f"available={depth_state['available']} "
            f"front_risk={depth_state['front_risk']} "
            f"roi_median_m={depth_state['roi_median_m']}"
        )
