"""Simple viewer for a World."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import time

try:
    import pygame

    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False


class Viewer:
    def __init__(
        self,
        world,
        width: int = 800,
        height: int = 600,
        scale: float = 200.0,
        clock_provider: Callable[[], Mapping[str, float]] | None = None,
        controls_provider: Callable[[], list[Mapping[str, object]]] | None = None,
        control_setter: Callable[[str, float], None] | None = None,
    ):
        self.world = world
        self.width = width
        self.height = height
        self.scale = scale
        self.clock_provider = clock_provider
        self.controls_provider = controls_provider
        self.control_setter = control_setter
        self.active_knob_key: str | None = None
        self.active_knob_start_y = 0
        self.active_knob_start_value = 0.0
        self.default_offset_x = width // 2
        self.offset_x = self.default_offset_x
        self.offset_y = height // 8
        self.camera_x = 0.0
        self.camera_target_screen_x = width * 0.42

        if PYGAME_AVAILABLE:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Triad Simulation")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 22)
            self.small_font = pygame.font.Font(None, 17)
            self.background = self._build_background()

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        sx = int(self.offset_x + x * self.scale)
        sy = int(self.height - (self.offset_y + y * self.scale))
        return sx, sy

    def _creature_center_x(self) -> float:
        if not self.world.particles:
            return 0.0
        return sum(p.x for p in self.world.particles) / len(self.world.particles)

    def _update_camera(self) -> None:
        target_x = self._creature_center_x()
        self.camera_x += (target_x - self.camera_x) * 0.12
        self.offset_x = int(self.camera_target_screen_x - self.camera_x * self.scale)

    @staticmethod
    def _mix_color(
        start: tuple[int, int, int], end: tuple[int, int, int], amount: float
    ) -> tuple[int, int, int]:
        amount = max(0.0, min(1.0, amount))
        return tuple(int(a + (b - a) * amount) for a, b in zip(start, end))

    @staticmethod
    def _phase_color(phase: float) -> tuple[int, int, int]:
        blend = 0.5 + 0.5 * math.sin(2.0 * math.pi * phase)
        return Viewer._mix_color((74, 216, 202), (255, 180, 88), blend)

    @staticmethod
    def _error_intensity(state: Mapping[str, float] | None) -> float:
        if state is None:
            return 0.0
        return Viewer._clamp(abs(state.get("error_deg", 0.0)) / 45.0, 0.0, 1.0)

    @staticmethod
    def _error_color(state: Mapping[str, float] | None) -> tuple[int, int, int]:
        return Viewer._mix_color((89, 210, 198), (255, 92, 72), Viewer._error_intensity(state))

    def _build_background(self):
        surface = pygame.Surface((self.width, self.height))
        top = (15, 19, 30)
        middle = (29, 49, 63)
        bottom = (66, 77, 76)

        for y in range(self.height):
            ratio = y / max(1, self.height - 1)
            if ratio < 0.68:
                color = self._mix_color(top, middle, ratio / 0.68)
            else:
                color = self._mix_color(middle, bottom, (ratio - 0.68) / 0.32)
            pygame.draw.line(surface, color, (0, y), (self.width, y))

        horizon_y = min(self.height - 1, self.world_to_screen(0, 0)[1])
        pygame.draw.line(surface, (93, 120, 118), (0, horizon_y), (self.width, horizon_y), 1)
        for i in range(8):
            x = int((i / 7) * self.width)
            peak = horizon_y - 48 - (i % 3) * 18
            pygame.draw.polygon(
                surface,
                (20, 34, 43),
                [(x - 170, horizon_y), (x, peak), (x + 180, horizon_y)],
            )

        return surface

    def _clock_state(self) -> Mapping[str, float] | None:
        if self.clock_provider is None:
            return None
        return self.clock_provider()

    def _control_specs(self) -> list[Mapping[str, object]]:
        if self.controls_provider is None:
            return []
        return self.controls_provider()

    @staticmethod
    def _knob_specs(specs: list[Mapping[str, object]]) -> list[Mapping[str, object]]:
        return [spec for spec in specs if spec.get("type", "knob") != "button"]

    @staticmethod
    def _button_specs(specs: list[Mapping[str, object]]) -> list[Mapping[str, object]]:
        return [spec for spec in specs if spec.get("type", "knob") == "button"]

    def draw(self) -> None:
        clock_state = self._clock_state()
        control_specs = self._control_specs()
        self._update_camera()
        if not PYGAME_AVAILABLE:
            pts = [(p.x, p.y) for p in self.world.particles]
            cons = [
                ((c.p1.x, c.p1.y), (c.p2.x, c.p2.y)) for c in self.world.constraints
            ]
            print("Particles:")
            for i, (x, y) in enumerate(pts):
                print(f"  {i}: ({x:.2f}, {y:.2f})")
            print("Constraints:")
            for i, ((x1, y1), (x2, y2)) in enumerate(cons):
                length = math.hypot(x2 - x1, y2 - y1)
                print(
                    f"  {i + 1}: "
                    f"({x1:.2f},{y1:.2f}) - ({x2:.2f},{y2:.2f}) "
                    f"len={length:.2f}"
                )
            if clock_state is not None:
                print(
                    "Clock: "
                    f"t={clock_state.get('time', 0.0):.2f}s "
                    f"phase={clock_state.get('phase', 0.0) % 1.0:.2f}"
                )
            for spec in control_specs:
                print(f"{spec.get('label', spec.get('key'))}: {spec.get('value')}")
            print("---")
            return

        self.screen.blit(self.background, (0, 0))
        self._draw_ground()
        self._draw_target_ghost(clock_state)

        segment_labels = []
        for segment_index, c in enumerate(self.world.constraints, start=1):
            link_state = clock_state
            if clock_state is not None and clock_state.get("mode") == "pivot_angles":
                if segment_index == 1:
                    link_state = {"error_deg": clock_state.get("angle1_error_deg", 0.0)}
                elif segment_index == 2:
                    link_state = {
                        "error_deg": max(
                            abs(clock_state.get("angle1_error_deg", 0.0)),
                            abs(clock_state.get("angle2_error_deg", 0.0)),
                        )
                    }
                elif segment_index == 3:
                    link_state = {"error_deg": clock_state.get("angle2_error_deg", 0.0)}
            elif (
                clock_state is not None
                and "branch_error_deg" in clock_state
                and len(self.world.particles) >= 4
                and (
                    (c.p1 is self.world.particles[1] and c.p2 is self.world.particles[3])
                    or (c.p1 is self.world.particles[3] and c.p2 is self.world.particles[1])
                )
            ):
                link_state = {"error_deg": clock_state["branch_error_deg"]}
            error_color = self._error_color(link_state)
            error_width = 3 + int(self._error_intensity(link_state) * 4)
            a = self.world_to_screen(c.p1.x, c.p1.y)
            b = self.world_to_screen(c.p2.x, c.p2.y)
            pygame.draw.line(self.screen, (235, 238, 226), a, b, error_width + 1)
            pygame.draw.line(self.screen, error_color, a, b, error_width)
            segment_labels.append((segment_index, a, b, error_color))

        for p in self.world.particles:
            radius = max(3, int(5 + (0 if p.inv_mass == 0 else (1.0 / p.inv_mass)) * 0.5))
            color = (96, 104, 110) if p.inv_mass == 0 else (255, 246, 214)
            pygame.draw.circle(
                self.screen,
                (25, 30, 32),
                self.world_to_screen(p.x, p.y),
                radius + 3,
            )
            pygame.draw.circle(self.screen, color, self.world_to_screen(p.x, p.y), radius)
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                self.world_to_screen(p.x, p.y),
                max(1, radius // 3),
            )

        for segment_index, start, end, color in segment_labels:
            self._draw_segment_label(segment_index, start, end, color)

        self._draw_clock(clock_state)
        self._draw_controls(control_specs)

        pygame.display.flip()

    def _draw_ground(self) -> None:
        ground_y = self.world_to_screen(0, 0)[1]
        pygame.draw.rect(
            self.screen,
            (38, 72, 48),
            (0, ground_y, self.width, self.height - ground_y),
        )
        pygame.draw.line(self.screen, (126, 187, 113), (0, ground_y), (self.width, ground_y), 2)
        self._draw_ground_graduations(ground_y)

        vanishing_x = self.offset_x
        vanishing_y = ground_y
        for x in range(-self.width, self.width * 2, 72):
            end_x = int(vanishing_x + (x - vanishing_x) * 0.18)
            pygame.draw.line(
                self.screen,
                (47, 95, 59),
                (x, self.height),
                (end_x, vanishing_y),
                1,
            )

        y = ground_y + 22
        spacing = 18
        while y < self.height:
            pygame.draw.line(self.screen, (50, 100, 62), (0, y), (self.width, y), 1)
            y += spacing
            spacing = min(42, int(spacing * 1.16))

    def _draw_ground_graduations(self, ground_y: int) -> None:
        visible_min_x = (0 - self.offset_x) / self.scale
        visible_max_x = (self.width - self.offset_x) / self.scale
        start_meter = math.floor(visible_min_x) - 1
        end_meter = math.ceil(visible_max_x) + 1
        zero_screen_x = self.world_to_screen(0.0, 0.0)[0]

        for meter in range(start_meter, end_meter + 1):
            x = self.world_to_screen(float(meter), 0.0)[0]
            if x < -40 or x > self.width + 40:
                continue

            is_zero = meter == 0
            tick_top = ground_y - (18 if is_zero or meter % 5 == 0 else 12)
            line_color = (255, 233, 154) if is_zero else (103, 153, 122)
            pygame.draw.line(self.screen, line_color, (x, tick_top), (x, ground_y + 10), 2)

            if meter % 5 == 0 or is_zero:
                label = self.small_font.render(f"{meter} m", True, (196, 215, 214))
                self.screen.blit(label, (x - label.get_width() // 2, ground_y + 14))

        for half_meter in range(math.floor(visible_min_x * 2) - 1, math.ceil(visible_max_x * 2) + 2):
            if half_meter % 2 == 0:
                continue
            world_x = half_meter * 0.5
            x = self.world_to_screen(world_x, 0.0)[0]
            if -20 <= x <= self.width + 20:
                pygame.draw.line(
                    self.screen,
                    (69, 118, 84),
                    (x, ground_y - 7),
                    (x, ground_y + 5),
                    1,
                )

        if -30 <= zero_screen_x <= self.width + 30:
            pygame.draw.line(
                self.screen,
                (255, 233, 154),
                (zero_screen_x, ground_y - 48),
                (zero_screen_x, self.height),
                1,
            )

    def _draw_target_ghost(self, state: Mapping[str, float] | None) -> None:
        if state is None or len(self.world.particles) < 3:
            return

        left = self.world.particles[0]
        joint = self.world.particles[1]
        right = self.world.particles[2]
        left_len = math.hypot(left.x - joint.x, left.y - joint.y)
        right_len = math.hypot(right.x - joint.x, right.y - joint.y)
        target_angle = math.radians(
            abs(state.get("angle1_target_deg", state.get("target_angle_deg", 0.0)))
        )
        half_angle = self._clamp(target_angle * 0.5, 0.0, math.pi)

        ghost_left = (
            joint.x - math.sin(half_angle) * left_len,
            joint.y - math.cos(half_angle) * left_len,
        )
        ghost_right = (
            joint.x + math.sin(half_angle) * right_len,
            joint.y - math.cos(half_angle) * right_len,
        )

        ghost = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        joint_screen = self.world_to_screen(joint.x, joint.y)
        left_screen = self.world_to_screen(*ghost_left)
        right_screen = self.world_to_screen(*ghost_right)
        color = (*self._phase_color(state.get("phase", 0.0)), 138)

        self._draw_dashed_line(ghost, color, joint_screen, left_screen, 9, 7, 3)
        self._draw_dashed_line(ghost, color, joint_screen, right_screen, 9, 7, 3)
        pygame.draw.circle(ghost, (255, 233, 154, 96), joint_screen, 8, 2)
        pygame.draw.circle(ghost, color, left_screen, 7, 2)
        pygame.draw.circle(ghost, color, right_screen, 7, 2)

        arc_radius = 34
        arc_rect = pygame.Rect(
            joint_screen[0] - arc_radius,
            joint_screen[1] - arc_radius,
            arc_radius * 2,
            arc_radius * 2,
        )
        pygame.draw.arc(
            ghost,
            (255, 233, 154, 118),
            arc_rect,
            math.pi / 2.0 - half_angle,
            math.pi / 2.0 + half_angle,
            3,
        )

        if state.get("mode") == "pivot_angles" and len(self.world.particles) >= 4:
            branch = self.world.particles[3]
            branch_len = math.hypot(branch.x - joint.x, branch.y - joint.y)
            right_angle = math.atan2(ghost_right[1] - joint.y, ghost_right[0] - joint.x)
            branch_angle = right_angle + math.radians(state.get("angle2_target_deg", 0.0))
            branch_target = (
                joint.x + math.cos(branch_angle) * branch_len,
                joint.y + math.sin(branch_angle) * branch_len,
            )
            branch_target_screen = self.world_to_screen(*branch_target)
            branch_color = (255, 233, 154, 155)
            self._draw_dashed_line(
                ghost,
                branch_color,
                right_screen,
                branch_target_screen,
                8,
                6,
                3,
            )
            self._draw_dashed_line(
                ghost,
                branch_color,
                joint_screen,
                branch_target_screen,
                8,
                6,
                2,
            )
            pygame.draw.circle(ghost, branch_color, branch_target_screen, 9, 2)
        elif "branch_target_angle_deg" in state and len(self.world.particles) >= 4:
            branch = self.world.particles[3]
            branch_len = math.hypot(branch.x - joint.x, branch.y - joint.y)
            branch_angle = math.radians(90.0 - state.get("branch_target_angle_deg", 0.0))
            branch_target = (
                joint.x + math.cos(branch_angle) * branch_len,
                joint.y + math.sin(branch_angle) * branch_len,
            )
            branch_target_screen = self.world_to_screen(*branch_target)
            branch_color = (255, 233, 154, 155)
            self._draw_dashed_line(
                ghost,
                branch_color,
                joint_screen,
                branch_target_screen,
                8,
                6,
                3,
            )
            pygame.draw.circle(ghost, branch_color, branch_target_screen, 9, 2)

        self.screen.blit(ghost, (0, 0))

    @staticmethod
    def _draw_dashed_line(
        surface,
        color: tuple[int, int, int, int],
        start: tuple[int, int],
        end: tuple[int, int],
        dash: int,
        gap: int,
        width: int,
    ) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance == 0:
            return

        step = dash + gap
        ux = dx / distance
        uy = dy / distance
        progress = 0.0
        while progress < distance:
            segment_end = min(progress + dash, distance)
            a = (int(start[0] + ux * progress), int(start[1] + uy * progress))
            b = (int(start[0] + ux * segment_end), int(start[1] + uy * segment_end))
            pygame.draw.line(surface, color, a, b, width)
            progress += step

    def _draw_segment_label(
        self,
        segment_index: int,
        start: tuple[int, int],
        end: tuple[int, int],
        accent: tuple[int, int, int],
    ) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return

        nx = -dy / length
        ny = dx / length
        if ny > 0:
            nx = -nx
            ny = -ny

        label = self.small_font.render(str(segment_index), True, (255, 246, 214))
        padding_x = 7
        padding_y = 4
        width = label.get_width() + padding_x * 2
        height = label.get_height() + padding_y * 2
        x = int((start[0] + end[0]) * 0.5 + nx * 17 - width * 0.5)
        y = int((start[1] + end[1]) * 0.5 + ny * 17 - height * 0.5)
        x = max(2, min(self.width - width - 2, x))
        y = max(2, min(self.height - height - 2, y))

        badge = pygame.Surface((width, height), pygame.SRCALPHA)
        badge.fill((0, 0, 0, 0))
        pygame.draw.rect(badge, (7, 13, 19, 214), badge.get_rect(), border_radius=7)
        pygame.draw.rect(badge, (*accent, 210), badge.get_rect(), 1, border_radius=7)
        badge.blit(label, (padding_x, padding_y))
        self.screen.blit(badge, (x, y))

    def _draw_clock(self, state: Mapping[str, float] | None) -> None:
        if state is None:
            return
        if state.get("mode") == "pivot_angles":
            self._draw_pivot_angle_hud(state)
            return

        phase = state.get("phase", 0.0) % 1.0
        sim_time = state.get("time", 0.0)
        target = state.get("target_angle_deg", 0.0)
        current = state.get("current_angle_deg", 0.0)
        error = state.get("error_deg", 0.0)
        torque = state.get("torque", 0.0)
        max_torque = state.get("max_torque", 0.0)
        color = self._phase_color(phase)
        has_branch = "branch_target_angle_deg" in state

        panel_height = 154 if has_branch else 132
        panel = pygame.Surface((340, panel_height), pygame.SRCALPHA)
        panel.fill((7, 13, 19, 188))
        pygame.draw.rect(panel, (82, 112, 120, 150), panel.get_rect(), 1, border_radius=8)

        center = (68, 72)
        radius = 42
        pygame.draw.circle(panel, (16, 27, 36), center, radius + 7)
        pygame.draw.circle(panel, (60, 92, 100), center, radius + 7, 1)
        pygame.draw.circle(panel, (12, 18, 24), center, radius)

        for tick in range(60):
            angle = -math.pi / 2.0 + (tick / 60.0) * 2.0 * math.pi
            tick_len = 11 if tick % 5 == 0 else 5
            tick_color = (235, 238, 226) if tick % 5 == 0 else (100, 128, 132)
            outer = (
                int(center[0] + math.cos(angle) * (radius - 2)),
                int(center[1] + math.sin(angle) * (radius - 2)),
            )
            inner = (
                int(center[0] + math.cos(angle) * (radius - tick_len)),
                int(center[1] + math.sin(angle) * (radius - tick_len)),
            )
            pygame.draw.line(panel, tick_color, inner, outer, 1)

        rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
        pygame.draw.arc(
            panel,
            (255, 233, 154),
            rect,
            -math.pi / 2.0,
            -math.pi / 2.0 + phase * 2.0 * math.pi,
            4,
        )
        hand_angle = -math.pi / 2.0 + phase * 2.0 * math.pi
        hand = (
            int(center[0] + math.cos(hand_angle) * (radius - 12)),
            int(center[1] + math.sin(hand_angle) * (radius - 12)),
        )
        pygame.draw.line(panel, color, center, hand, 3)
        pygame.draw.circle(panel, color, center, 5)

        title = self.font.render("triad clock", True, (235, 238, 226))
        panel.blit(title, (132, 18))
        lines = [
            f"t {sim_time:05.2f}s   phase {phase * 100:05.1f}%",
            f"target {target:06.1f} deg",
            f"angle  {current:06.1f} deg",
            f"err {error:06.1f} deg   tau {torque:05.1f}/{max_torque:.0f}",
        ]
        if has_branch:
            lines.append(
                "branch "
                f"{state.get('branch_current_angle_deg', 0.0):05.1f}->"
                f"{state.get('branch_target_angle_deg', 0.0):05.1f} deg"
            )
        for index, text in enumerate(lines):
            label = self.small_font.render(text, True, (196, 215, 214))
            panel.blit(label, (132, 43 + index * 18))

        wave_left = 132
        wave_top = 128 if has_branch else 106
        wave_width = 154
        wave_height = 18
        baseline = wave_top + wave_height // 2
        points = []
        for x in range(wave_width):
            local_phase = (x / max(1, wave_width - 1) + phase) % 1.0
            y = baseline - int(math.sin(local_phase * 2.0 * math.pi) * 7)
            points.append((wave_left + x, y))
        if len(points) > 1:
            pygame.draw.lines(panel, (89, 210, 198), False, points, 2)
        marker_x = wave_left + int(phase * wave_width)
        pygame.draw.line(
            panel,
            (255, 233, 154),
            (marker_x, wave_top),
            (marker_x, wave_top + wave_height),
            1,
        )

        self.screen.blit(panel, (18, 18))

    def _draw_pivot_angle_hud(self, state: Mapping[str, float]) -> None:
        panel = pygame.Surface((390, 222), pygame.SRCALPHA)
        panel.fill((7, 13, 19, 188))
        pygame.draw.rect(panel, (82, 112, 120, 150), panel.get_rect(), 1, border_radius=8)

        title = self.font.render("pivot angles", True, (235, 238, 226))
        panel.blit(title, (16, 14))
        position_text = (
            f"pos {int(state.get('position_index', 0)) + 1}/"
            f"{int(state.get('position_count', 1))}"
            f"  {state.get('clock_hz', 0.0):.2f}Hz"
            f"  {'playing' if state.get('playing', False) else 'editing'}"
        )
        position_label = self.small_font.render(position_text, True, (196, 215, 214))
        panel.blit(position_label, (224, 18))

        rows = [
            (
                "angle 1",
                state.get("angle1_current_unsigned_deg", state.get("angle1_current_deg", 0.0)),
                state.get("angle1_target_deg", 0.0),
                state.get("angle1_applied_setpoint_deg", state.get("angle1_target_deg", 0.0)),
                state.get("angle1_applied_target_deg", state.get("angle1_target_deg", 0.0)),
                state.get("angle1_error_deg", 0.0),
                state.get("angle1_torque", 0.0),
            ),
            (
                "angle 2",
                state.get("angle2_current_unsigned_deg", state.get("angle2_current_deg", 0.0)),
                state.get("angle2_target_deg", 0.0),
                state.get("angle2_applied_setpoint_deg", state.get("angle2_target_deg", 0.0)),
                state.get("angle2_applied_target_deg", state.get("angle2_target_deg", 0.0)),
                state.get("angle2_error_deg", 0.0),
                state.get("angle2_torque", 0.0),
            ),
        ]

        for index, (
            label,
            current,
            target,
            applied_setpoint,
            applied,
            error,
            torque,
        ) in enumerate(rows):
            y = 46 + index * 50
            intensity = self._clamp(abs(error) / 45.0, 0.0, 1.0)
            color = self._mix_color((89, 210, 198), (255, 92, 72), intensity)
            pygame.draw.circle(panel, color, (24, y + 8), 6)
            text = (
                f"{label}: set {target:06.1f} deg  apply {applied_setpoint:06.1f} deg"
            )
            row = self.small_font.render(text, True, (196, 215, 214))
            panel.blit(row, (40, y))
            measured = (
                f"cmd {applied:07.1f} deg  measured {current:06.1f} deg  "
                f"err {error:07.1f} tau {torque:06.1f}"
            )
            measured_row = self.small_font.render(measured, True, (196, 215, 214))
            panel.blit(measured_row, (40, y + 16))

            bar_x = 40
            bar_y = y + 34
            bar_w = 210
            pygame.draw.rect(panel, (43, 60, 65), (bar_x, bar_y, bar_w, 5), border_radius=3)
            fill_w = int(bar_w * intensity)
            if fill_w > 0:
                pygame.draw.rect(panel, color, (bar_x, bar_y, fill_w, 5), border_radius=3)

        energy_text = (
            f"energy {state.get('energy_total_j', 0.0):08.2f} J  "
            f"power {state.get('power_total_w', 0.0):07.2f} W"
        )
        energy_label = self.small_font.render(energy_text, True, (255, 233, 154))
        panel.blit(energy_label, (40, 148))
        pivot_energy_text = (
            f"pivots {state.get('energy_pivot1_j', 0.0):.2f} / "
            f"{state.get('energy_pivot2_j', 0.0):.2f} J"
        )
        pivot_energy_label = self.small_font.render(
            pivot_energy_text, True, (196, 215, 214)
        )
        panel.blit(pivot_energy_label, (40, 164))

        score_text = (
            f"score {state.get('score', 0.0):08.2f}  "
            f"dx {state.get('distance_x_m', 0.0):+.3f} m"
        )
        score_label = self.small_font.render(score_text, True, (235, 238, 226))
        panel.blit(score_label, (40, 184))
        score_detail_text = (
            f"move {state.get('score_distance_points', 0.0):+.2f}  "
            f"energy -{state.get('score_energy_penalty_points', 0.0):.2f}"
        )
        score_detail_label = self.small_font.render(
            score_detail_text, True, (196, 215, 214)
        )
        panel.blit(score_detail_label, (40, 200))

        self.screen.blit(panel, (18, 18))

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    def _control_panel_rect(self, specs: list[Mapping[str, object]]) -> pygame.Rect:
        knobs = self._knob_specs(specs)
        buttons = self._button_specs(specs)
        columns = min(3, max(1, len(knobs)))
        rows = max(1, math.ceil(len(knobs) / columns)) if knobs else 0
        panel_width = 28 + columns * 72
        panel_height = 42 + rows * 102 + (44 if buttons else 0)
        return pygame.Rect(self.width - panel_width - 18, 18, panel_width, panel_height)

    def _control_knob_centers(
        self, specs: list[Mapping[str, object]]
    ) -> list[tuple[Mapping[str, object], tuple[int, int]]]:
        panel = self._control_panel_rect(specs)
        centers = []
        for index, spec in enumerate(self._knob_specs(specs)):
            column = index % 3
            row = index // 3
            centers.append((spec, (panel.x + 42 + column * 72, panel.y + 70 + row * 102)))
        return centers

    def _control_button_rects(
        self, specs: list[Mapping[str, object]]
    ) -> list[tuple[Mapping[str, object], pygame.Rect]]:
        buttons = self._button_specs(specs)
        if not buttons:
            return []

        panel = self._control_panel_rect(specs)
        knob_rows = max(1, math.ceil(len(self._knob_specs(specs)) / 3))
        y = panel.y + 34 + knob_rows * 102
        rects = []
        for index, spec in enumerate(buttons):
            rects.append((spec, pygame.Rect(panel.x + 14, y + index * 38, panel.width - 28, 30)))
        return rects

    def _draw_controls(self, specs: list[Mapping[str, object]]) -> None:
        if not specs:
            return

        panel_rect = self._control_panel_rect(specs)
        panel = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        panel.fill((7, 13, 19, 188))
        pygame.draw.rect(panel, (82, 112, 120, 150), panel.get_rect(), 1, border_radius=8)

        title = self.font.render("controls", True, (235, 238, 226))
        panel.blit(title, (14, 12))

        for spec, center_screen in self._control_knob_centers(specs):
            center = (center_screen[0] - panel_rect.x, center_screen[1] - panel_rect.y)
            key = str(spec.get("key", ""))
            label = str(spec.get("label", key))
            unit = str(spec.get("unit", ""))
            value = float(spec.get("value", 0.0))
            min_value = float(spec.get("min", 0.0))
            max_value = float(spec.get("max", 1.0))
            value = self._clamp(value, min_value, max_value)
            span = max(0.0001, max_value - min_value)
            ratio = (value - min_value) / span
            color = self._phase_color(ratio)
            radius = 25

            pygame.draw.circle(panel, (16, 27, 36), center, radius + 7)
            pygame.draw.circle(panel, (60, 92, 100), center, radius + 7, 1)
            pygame.draw.circle(panel, (12, 18, 24), center, radius)

            arc_rect = pygame.Rect(
                center[0] - radius,
                center[1] - radius,
                radius * 2,
                radius * 2,
            )
            pygame.draw.arc(
                panel,
                (75, 95, 102),
                arc_rect,
                math.radians(135),
                math.radians(405),
                4,
            )
            pygame.draw.arc(
                panel,
                color,
                arc_rect,
                math.radians(135),
                math.radians(135 + ratio * 270),
                4,
            )

            hand_angle = math.radians(135 + ratio * 270)
            hand = (
                int(center[0] + math.cos(hand_angle) * (radius - 7)),
                int(center[1] + math.sin(hand_angle) * (radius - 7)),
            )
            pygame.draw.line(panel, color, center, hand, 3)
            pygame.draw.circle(panel, color, center, 4)

            if self.active_knob_key == key:
                pygame.draw.circle(panel, (255, 233, 154), center, radius + 10, 2)

            label_surface = self.small_font.render(label, True, (196, 215, 214))
            if unit == "Hz":
                value_text = f"{value:.2f}{unit}"
            elif unit == "deg":
                value_text = f"{value:.0f}{unit}"
            else:
                value_text = f"{value:.1f}"
            value_surface = self.small_font.render(value_text, True, (235, 238, 226))
            panel.blit(label_surface, (center[0] - label_surface.get_width() // 2, center[1] + 30))
            panel.blit(value_surface, (center[0] - value_surface.get_width() // 2, center[1] + 45))

        for spec, rect_screen in self._control_button_rects(specs):
            rect = rect_screen.move(-panel_rect.x, -panel_rect.y)
            active = bool(spec.get("active", False))
            fill = (29, 92, 82, 230) if active else (24, 35, 43, 230)
            border = (89, 210, 198, 220) if active else (82, 112, 120, 180)
            pygame.draw.rect(panel, fill, rect, border_radius=8)
            pygame.draw.rect(panel, border, rect, 1, border_radius=8)
            label = self.small_font.render(str(spec.get("label", "")), True, (235, 238, 226))
            panel.blit(
                label,
                (
                    rect.centerx - label.get_width() // 2,
                    rect.centery - label.get_height() // 2,
                ),
            )

        self.screen.blit(panel, panel_rect)

    def _find_control_spec(
        self, key: str, specs: list[Mapping[str, object]]
    ) -> Mapping[str, object] | None:
        for spec in specs:
            if str(spec.get("key", "")) == key:
                return spec
        return None

    def _knob_key_at(
        self, pos: tuple[int, int], specs: list[Mapping[str, object]]
    ) -> str | None:
        for spec, center in self._control_knob_centers(specs):
            if math.hypot(pos[0] - center[0], pos[1] - center[1]) <= 34:
                return str(spec.get("key", ""))
        return None

    def _button_key_at(
        self, pos: tuple[int, int], specs: list[Mapping[str, object]]
    ) -> str | None:
        for spec, rect in self._control_button_rects(specs):
            if rect.collidepoint(pos):
                return str(spec.get("key", ""))
        return None

    def _handle_control_event(self, event) -> bool:
        if self.control_setter is None:
            return False

        specs = self._control_specs()
        if not specs:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            button_key = self._button_key_at(event.pos, specs)
            if button_key is not None:
                self.control_setter(button_key, 1.0)
                return True

            key = self._knob_key_at(event.pos, specs)
            if key is None:
                return False
            spec = self._find_control_spec(key, specs)
            if spec is None:
                return False
            self.active_knob_key = key
            self.active_knob_start_y = event.pos[1]
            self.active_knob_start_value = float(spec.get("value", 0.0))
            return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.active_knob_key is None:
                return False
            self.active_knob_key = None
            return True

        if event.type == pygame.MOUSEMOTION and self.active_knob_key is not None:
            spec = self._find_control_spec(self.active_knob_key, specs)
            if spec is None:
                return False
            min_value = float(spec.get("min", 0.0))
            max_value = float(spec.get("max", 1.0))
            span = max_value - min_value
            precision = 0.35 if pygame.key.get_mods() & pygame.KMOD_SHIFT else 1.0
            delta = (self.active_knob_start_y - event.pos[1]) * span / 130.0
            value = self.active_knob_start_value + delta * precision
            self.control_setter(self.active_knob_key, value)
            return True

        return False

    def run_loop(self, step_callback, fps: int = 60, max_time: float | None = None) -> None:
        if not PYGAME_AVAILABLE:
            steps = 10 if max_time is None else max(1, int(max_time * fps))
            print(
                "Pygame not available; using console fallback. "
                f"Running {steps} steps then exit."
            )
            for _ in range(steps):
                if step_callback() is False:
                    break
                self.draw()
            return

        running = True
        start = time.perf_counter()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif self._handle_control_event(event):
                    continue
            if step_callback() is False:
                running = False
            self.draw()
            if max_time is not None and (time.perf_counter() - start) >= max_time:
                running = False
            self.clock.tick(fps)
        pygame.quit()
