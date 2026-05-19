import math
from typing import List, Tuple
from ..base.drone_space import DroneActionSpace, ActionPoint


class TelloDroneActionSpace(DroneActionSpace):
    def __init__(self, n_samples: int = 8):
        super().__init__(n_samples)
        # self.rotate_time = 3750 # time to rotate 360 degree (7500->50, 4166->90, 3750->100)
        self.rotate_time = 3400  # time to rotate 360 degree (7500->50, 4166->90, 3750->100)
        self.move_time = 500  # time to move 1 unit (1000->50, 555->90, 500->100)
        self.yaw_rc_speed = 100
        self.max_yaw_duration_ms = 150

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
                clamped_yaw_duration_ms = min(
                    raw_yaw_duration_ms, self.max_yaw_duration_ms
                )
                print(
                    "[TELLO YAW] "
                    f"yaw_direction=yaw_left "
                    f"yaw_rc_speed={self.yaw_rc_speed} "
                    f"raw_yaw_duration_ms={raw_yaw_duration_ms} "
                    f"clamped_yaw_duration_ms={clamped_yaw_duration_ms}"
                )
                commands.append(
                    ("yaw_left", clamped_yaw_duration_ms)
                )
            else:
                raw_yaw_duration_ms = int(target_angle * (self.rotate_time / 360))
                clamped_yaw_duration_ms = min(
                    raw_yaw_duration_ms, self.max_yaw_duration_ms
                )
                print(
                    "[TELLO YAW] "
                    f"yaw_direction=yaw_right "
                    f"yaw_rc_speed={self.yaw_rc_speed} "
                    f"raw_yaw_duration_ms={raw_yaw_duration_ms} "
                    f"clamped_yaw_duration_ms={clamped_yaw_duration_ms}"
                )
                commands.append(
                    ("yaw_right", clamped_yaw_duration_ms)
                )

        # 3. Add forward movement if needed
        distance_xy = math.sqrt(action.dx**2 + action.dy**2)
        if distance_xy > 0.01:
            commands.append(("pitch_forward", int(distance_xy * self.move_time)))

        # 4. Add vertical movement if needed
        if abs(action.dz) > 0.01:
            if action.dz > 0:
                commands.append(("increase_throttle", int(abs(action.dz) * self.move_time)))
            else:
                commands.append(("decrease_throttle", int(abs(action.dz) * self.move_time)))

        return commands
