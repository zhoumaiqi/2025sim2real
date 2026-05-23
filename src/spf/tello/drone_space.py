import math
from typing import List, Tuple
from ..base.drone_space import DroneActionSpace, ActionPoint


class TelloDroneActionSpace(DroneActionSpace):
    def __init__(self, n_samples: int = 8):
        super().__init__(n_samples)
        # self.rotate_time = 3750 # time to rotate 360 degree (7500->50, 4166->90, 3750->100)
        self.rotate_time = 3400  # time to rotate 360 degree (7500->50, 4166->90, 3750->100)
        self.move_time = 500  # time to move 1 unit (1000->50, 555->90, 500->100)
        self.forward_duration_scale = 0.25
        self.max_forward_duration_ms = 350
        self.min_forward_duration_ms = 120
        self.yaw_duration_scale = 0.6
        self.max_yaw_duration_ms = 180
        self.vertical_duration_scale = 0.4
        self.max_vertical_duration_ms = 250
        self.enable_vertical_commands = False

    def _scale_duration(
        self,
        command_name: str,
        raw_duration_ms: int,
        scale_type: str,
    ) -> int:
        """Scale and clamp command duration for smaller Tello steps."""
        if scale_type == "forward":
            scaled_duration_ms = int(round(raw_duration_ms * self.forward_duration_scale))
            scaled_duration_ms = max(
                self.min_forward_duration_ms,
                min(scaled_duration_ms, self.max_forward_duration_ms),
            )
        elif scale_type == "yaw":
            scaled_duration_ms = int(round(raw_duration_ms * self.yaw_duration_scale))
            scaled_duration_ms = max(
                1,
                min(scaled_duration_ms, self.max_yaw_duration_ms),
            )
        elif scale_type == "vertical":
            scaled_duration_ms = int(round(raw_duration_ms * self.vertical_duration_scale))
            scaled_duration_ms = max(
                1,
                min(scaled_duration_ms, self.max_vertical_duration_ms),
            )
        else:
            scaled_duration_ms = max(1, raw_duration_ms)

        print(
            f"[TELLO STEP] {command_name} "
            f"raw={raw_duration_ms}ms "
            f"scaled={scaled_duration_ms}ms "
            f"scale_type={scale_type}"
        )
        return scaled_duration_ms

    def action_to_commands(self, action: ActionPoint) -> List[Tuple[str, int]]:
        """Convert a relative movement action into drone commands"""
        commands = []

        # 1. Calculate yaw angle needed
        target_angle = math.degrees(math.atan2(action.dx, action.dy)) % 360

        # 2. Add yaw command if needed (if there's horizontal movement)
        if abs(action.dx) > 0.01 or abs(action.dy) > 0.01:
            if target_angle > 180:
                raw_yaw_duration_ms = int(
                    abs(360 - target_angle) * (self.rotate_time / 360)
                )
                scaled_yaw_duration_ms = self._scale_duration(
                    "yaw_left", raw_yaw_duration_ms, "yaw"
                )
                commands.append(
                    ("yaw_left", scaled_yaw_duration_ms)
                )
            else:
                raw_yaw_duration_ms = int(target_angle * (self.rotate_time / 360))
                scaled_yaw_duration_ms = self._scale_duration(
                    "yaw_right", raw_yaw_duration_ms, "yaw"
                )
                commands.append(
                    ("yaw_right", scaled_yaw_duration_ms)
                )

        # 3. Add forward movement if needed
        distance_xy = math.sqrt(action.dx**2 + action.dy**2)
        if distance_xy > 0.01:
            raw_forward_duration_ms = int(distance_xy * self.move_time)
            scaled_forward_duration_ms = self._scale_duration(
                "pitch_forward", raw_forward_duration_ms, "forward"
            )
            commands.append(("pitch_forward", scaled_forward_duration_ms))

        # 4. Add vertical movement if needed
        if abs(action.dz) > 0.01:
            raw_vertical_duration_ms = int(abs(action.dz) * self.move_time)
            if not self.enable_vertical_commands:
                vertical_command = (
                    "increase_throttle" if action.dz > 0 else "decrease_throttle"
                )
                print(
                    f"[TELLO STEP] vertical disabled: {vertical_command} "
                    f"raw={raw_vertical_duration_ms}ms skipped"
                )
                return commands

            if action.dz > 0:
                scaled_vertical_duration_ms = self._scale_duration(
                    "increase_throttle", raw_vertical_duration_ms, "vertical"
                )
                commands.append(("increase_throttle", scaled_vertical_duration_ms))
            else:
                scaled_vertical_duration_ms = self._scale_duration(
                    "decrease_throttle", raw_vertical_duration_ms, "vertical"
                )
                commands.append(("decrease_throttle", scaled_vertical_duration_ms))

        return commands
