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
            self.visual_time = 0.0

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

    def _parallax_x(self, world_x: float, parallax: float) -> int:
        return int(self.camera_target_screen_x + (world_x - self.camera_x * parallax) * self.scale)

    def _visible_decor_indices(
        self,
        spacing_m: float,
        parallax: float,
        padding_m: float = 1.0,
    ) -> range:
        left = self.camera_x * parallax - self.camera_target_screen_x / self.scale - padding_m
        right = (
            self.camera_x * parallax
            + (self.width - self.camera_target_screen_x) / self.scale
            + padding_m
        )
        start = math.floor(left / spacing_m)
        end = math.ceil(right / spacing_m)
        return range(start, end + 1)

    @staticmethod
    def _mix_color(
        start: tuple[int, int, int], end: tuple[int, int, int], amount: float
    ) -> tuple[int, int, int]:
        amount = max(0.0, min(1.0, amount))
        return tuple(int(a + (b - a) * amount) for a, b in zip(start, end))

    @staticmethod
    def _phase_color(phase: float) -> tuple[int, int, int]:
        blend = 0.5 + 0.5 * math.sin(2.0 * math.pi * phase)
        return Viewer._mix_color((38, 188, 180), (255, 184, 82), blend)

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
        top = (108, 205, 238)
        middle = (166, 226, 217)
        bottom = (254, 226, 156)

        for y in range(self.height):
            ratio = y / max(1, self.height - 1)
            if ratio < 0.68:
                color = self._mix_color(top, middle, ratio / 0.68)
            else:
                color = self._mix_color(middle, bottom, (ratio - 0.68) / 0.32)
            pygame.draw.line(surface, color, (0, y), (self.width, y))

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
        if clock_state is not None:
            self.visual_time = float(clock_state.get("time", self.visual_time))
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
        self._draw_parallax_decor()
        self._draw_ground()
        self._draw_low_poly_plants(self.world_to_screen(0, 0)[1])

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
            error_width = 10 + int(self._error_intensity(link_state) * 5)
            a = self.world_to_screen(c.p1.x, c.p1.y)
            b = self.world_to_screen(c.p2.x, c.p2.y)
            self._draw_organic_segment(a, b, error_color, error_width, segment_index)

        for i, p in enumerate(self.world.particles):
            radius = max(8, int(9 + (0 if p.inv_mass == 0 else (1.0 / p.inv_mass)) * 0.5))
            self._draw_organic_joint(self.world_to_screen(p.x, p.y), radius, i, p.inv_mass == 0)

        self._draw_clock(clock_state)

        pygame.display.flip()

    def _draw_parallax_decor(self) -> None:
        horizon = self.world_to_screen(0, 0)[1]
        self._draw_sky_items()
        self._draw_low_poly_hills(horizon + 8, 0.13, (106, 162, 166), 150, 0, True)
        self._draw_low_poly_hills(horizon + 24, 0.24, (88, 171, 130), 105, 31, True)
        self._draw_low_poly_hills(horizon + 44, 0.46, (61, 150, 90), 62, 67, False)
        self._draw_mountain_mist(horizon)
        self._draw_cloud_band(84, 0.08, (255, 247, 217), 1.08)
        self._draw_cloud_band(184, 0.18, (249, 236, 198))
        self._draw_tree_layer(horizon, 0.38, 92, (74, 151, 95), (121, 91, 65))
        self._draw_tree_layer(horizon, 0.62, 58, (49, 132, 83), (102, 74, 52))
        self._draw_tree_layer(horizon, 0.88, 112, (67, 158, 86), (114, 78, 50), 1.28)

    def _draw_sky_items(self) -> None:
        self._draw_zeppelin_layer()
        self._draw_balloon_layer()
        self._draw_bird_layer()

    def _draw_zeppelin_layer(self) -> None:
        spacing_m = 7.2
        parallax = 0.10
        for i in self._visible_decor_indices(spacing_m, parallax, 3.0):
            if i % 3 != 0:
                continue
            x = self._parallax_x(i * spacing_m + 1.8, parallax)
            y = 82 + (abs(i) % 3) * 24
            hull = [
                (x - 78, y),
                (x - 60, y - 20),
                (x - 16, y - 29),
                (x + 43, y - 22),
                (x + 77, y - 2),
                (x + 48, y + 21),
                (x - 18, y + 28),
                (x - 62, y + 17),
            ]
            pygame.draw.polygon(
                self.screen,
                (129, 83, 128),
                [(px + 3, py + 4) for px, py in hull],
            )
            pygame.draw.polygon(
                self.screen,
                (240, 117, 100),
                hull,
            )
            pygame.draw.polygon(
                self.screen,
                (255, 183, 103),
                [(x - 70, y - 1), (x - 52, y - 17), (x - 13, y - 25), (x - 24, y - 1)],
            )
            pygame.draw.polygon(
                self.screen,
                (255, 215, 117),
                [(x - 24, y - 1), (x - 13, y - 25), (x + 35, y - 19), (x + 17, y - 1)],
            )
            pygame.draw.polygon(
                self.screen,
                (214, 85, 112),
                [(x + 17, y - 1), (x + 35, y - 19), (x + 70, y - 2), (x + 43, y + 5)],
            )
            pygame.draw.polygon(
                self.screen,
                (149, 91, 159),
                [(x - 70, y), (x - 24, y), (x - 18, y + 22), (x - 62, y + 16)],
            )
            pygame.draw.polygon(
                self.screen,
                (95, 174, 191),
                [(x - 24, y), (x + 17, y), (x + 48, y + 18), (x - 18, y + 25)],
            )
            pygame.draw.polygon(
                self.screen,
                (194, 77, 118),
                [(x + 17, y), (x + 70, y - 1), (x + 48, y + 18)],
            )
            pygame.draw.polygon(
                self.screen,
                (103, 181, 196),
                [(x + 54, y - 8), (x + 91, y - 26), (x + 78, y + 1)],
            )
            pygame.draw.polygon(
                self.screen,
                (79, 139, 162),
                [(x + 54, y + 8), (x + 91, y + 26), (x + 78, y + 1)],
            )
            pygame.draw.line(self.screen, (112, 82, 105), (x - 48, y + 5), (x + 43, y + 5), 2)
            gondola = [
                (x - 28, y + 28),
                (x + 26, y + 28),
                (x + 17, y + 42),
                (x - 18, y + 42),
            ]
            pygame.draw.line(self.screen, (104, 80, 86), (x - 22, y + 22), (x - 20, y + 30), 1)
            pygame.draw.line(self.screen, (104, 80, 86), (x + 20, y + 22), (x + 18, y + 30), 1)
            pygame.draw.polygon(self.screen, (82, 103, 119), [(px + 2, py + 3) for px, py in gondola])
            pygame.draw.polygon(self.screen, (246, 194, 103), gondola)
            pygame.draw.polygon(
                self.screen,
                (91, 168, 185),
                [(x - 24, y + 30), (x + 24, y + 30), (x + 17, y + 37), (x - 18, y + 37)],
            )
            for window_x in (x - 14, x + 2, x + 18):
                pygame.draw.circle(self.screen, (255, 246, 190), (window_x, y + 4), 3)

    def _draw_balloon_layer(self) -> None:
        spacing_m = 4.1
        parallax = 0.20
        palette = [
            ((230, 91, 92), (255, 196, 111)),
            ((82, 159, 190), (255, 224, 128)),
            ((166, 116, 205), (247, 183, 111)),
        ]
        for i in self._visible_decor_indices(spacing_m, parallax, 2.0):
            if i % 2 == 0:
                continue
            x = self._parallax_x(i * spacing_m + 0.6, parallax)
            y = 158 + (abs(i) % 4) * 22
            primary, stripe = palette[abs(i) % len(palette)]
            balloon = pygame.Rect(x - 20, y - 32, 40, 48)
            pygame.draw.ellipse(self.screen, self._mix_color(primary, (103, 80, 91), 0.18), balloon.move(2, 3))
            pygame.draw.ellipse(self.screen, primary, balloon)
            pygame.draw.polygon(
                self.screen,
                stripe,
                [(x, y - 31), (x - 8, y - 20), (x - 7, y + 9), (x, y + 15), (x + 7, y + 9), (x + 8, y - 20)],
            )
            pygame.draw.line(self.screen, (116, 91, 70), (x - 12, y + 10), (x - 8, y + 25), 1)
            pygame.draw.line(self.screen, (116, 91, 70), (x + 12, y + 10), (x + 8, y + 25), 1)
            pygame.draw.rect(self.screen, (132, 94, 62), (x - 9, y + 24, 18, 10))
            pygame.draw.rect(self.screen, (206, 154, 82), (x - 7, y + 22, 14, 8))

    def _draw_bird_layer(self) -> None:
        spacing_m = 1.25
        parallax = 0.33
        wing_phase = math.sin(self.visual_time * 2.3) * 3
        for i in self._visible_decor_indices(spacing_m, parallax, 1.0):
            if i % 4 == 0:
                continue
            x = self._parallax_x(i * spacing_m, parallax)
            y = 64 + (abs(i * 17) % 95)
            scale = 0.7 + (abs(i) % 3) * 0.18
            wing = int((7 + wing_phase + (i % 2) * 2) * scale)
            body = (x, y)
            color = (73, 89, 96)
            pygame.draw.line(self.screen, color, body, (x - int(15 * scale), y - wing), 2)
            pygame.draw.line(self.screen, color, body, (x + int(15 * scale), y - wing), 2)
            pygame.draw.circle(self.screen, color, body, max(2, int(2 * scale)))

    def _draw_tree_layer(
        self,
        horizon: int,
        parallax: float,
        spacing: int,
        leaf_color: tuple[int, int, int],
        trunk_color: tuple[int, int, int],
        size: float = 1.0,
    ) -> None:
        spacing_m = spacing / self.scale
        for i in self._visible_decor_indices(spacing_m, parallax, 1.4):
            world_x = i * spacing_m
            x = self._parallax_x(world_x, parallax)
            base = horizon + int(2 + (abs(i) % 3) * 5 * size)
            height = int((42 + (abs(i + 1) % 4) * 8) * size)
            trunk_w = max(6, int(9 * size))
            crown_w = int(35 * size)
            side_w = int(26 * size)
            notch_w = int(14 * size)
            pygame.draw.rect(
                self.screen,
                trunk_color,
                (x - trunk_w // 2, base - height // 2, trunk_w, height // 2 + 16),
            )
            crown = [
                (x, base - height),
                (x - side_w, base - height // 2),
                (x - notch_w, base - height // 2),
                (x - crown_w, base - int(14 * size)),
                (x + crown_w, base - int(14 * size)),
                (x + notch_w, base - height // 2),
                (x + side_w, base - height // 2),
            ]
            pygame.draw.polygon(self.screen, leaf_color, crown)
            pygame.draw.polygon(
                self.screen,
                self._mix_color(leaf_color, (236, 228, 147), 0.22),
                [(x, base - height), (x - side_w, base - height // 2), (x, base - int(18 * size))],
            )
            pygame.draw.polygon(
                self.screen,
                self._mix_color(leaf_color, (31, 91, 68), 0.25),
                [(x, base - height), (x + side_w, base - height // 2), (x, base - int(18 * size))],
            )

    def _draw_low_poly_hills(
        self,
        base_y: int,
        parallax: float,
        color: tuple[int, int, int],
        height: int,
        seed_offset: int,
        snow: bool,
    ) -> None:
        darker = self._mix_color(color, (27, 76, 84), 0.30)
        lighter = self._mix_color(color, (247, 239, 184), 0.24)
        spacing_m = 270 / self.scale
        for i in self._visible_decor_indices(spacing_m, parallax, 2.0):
            x = self._parallax_x(i * spacing_m, parallax)
            peak_x = x + 42 + ((abs(i + seed_offset) % 3) - 1) * 24
            peak = base_y - height - ((abs(i) + seed_offset) % 3) * 32
            left_foot = x - 210
            right_foot = x + 280
            mid = x + 92
            points = [(left_foot, base_y), (peak_x, peak), (right_foot, base_y)]
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(
                self.screen,
                lighter,
                [(peak_x, peak), (mid, base_y), (left_foot, base_y)],
            )
            pygame.draw.polygon(
                self.screen,
                darker,
                [(peak_x, peak), (right_foot, base_y), (mid, base_y)],
            )
            pygame.draw.polygon(
                self.screen,
                self._mix_color(color, (19, 65, 78), 0.18),
                [(peak_x, peak), (peak_x + 34, peak + height // 2), (mid - 14, base_y)],
            )
            if snow:
                snow_w = 34 + (height // 5)
                snow_h = 32 + (height // 7)
                snow_color = (250, 250, 232)
                snow_shadow = (211, 231, 225)
                cap = [
                    (peak_x, peak),
                    (peak_x - snow_w, peak + snow_h),
                    (peak_x - snow_w // 3, peak + snow_h - 8),
                    (peak_x, peak + snow_h + 8),
                    (peak_x + snow_w // 3, peak + snow_h - 10),
                    (peak_x + snow_w, peak + snow_h),
                ]
                pygame.draw.polygon(self.screen, snow_color, cap)
                pygame.draw.polygon(
                    self.screen,
                    snow_shadow,
                    [(peak_x, peak), (peak_x + snow_w, peak + snow_h), (peak_x, peak + snow_h + 8)],
                )

    def _draw_cloud_band(
        self,
        y: int,
        parallax: float,
        color: tuple[int, int, int],
        size: float = 1.0,
    ) -> None:
        spacing_m = 310 / self.scale
        for i in self._visible_decor_indices(spacing_m, parallax, 2.0):
            x = self._parallax_x(i * spacing_m, parallax)
            w = int(178 * size)
            h = int(50 * size)
            y_offset = (abs(i) % 3) * int(7 * size)
            top = y + y_offset
            points = [
                (x, top + int(h * 0.56)),
                (x + int(w * 0.07), top + int(h * 0.38)),
                (x + int(w * 0.16), top + int(h * 0.25)),
                (x + int(w * 0.27), top + int(h * 0.15)),
                (x + int(w * 0.39), top + int(h * 0.03)),
                (x + int(w * 0.51), top + int(h * 0.10)),
                (x + int(w * 0.62), top + int(h * 0.18)),
                (x + int(w * 0.76), top + int(h * 0.10)),
                (x + int(w * 0.90), top + int(h * 0.29)),
                (x + w, top + int(h * 0.52)),
                (x + int(w * 0.93), top + int(h * 0.70)),
                (x + int(w * 0.80), top + int(h * 0.83)),
                (x + int(w * 0.64), top + int(h * 0.90)),
                (x + int(w * 0.45), top + h),
                (x + int(w * 0.28), top + int(h * 0.92)),
                (x + int(w * 0.12), top + int(h * 0.80)),
            ]
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(
                self.screen,
                self._mix_color(color, (255, 255, 255), 0.30),
                [
                    points[1],
                    points[2],
                    points[3],
                    points[4],
                    points[5],
                    (x + int(w * 0.47), top + int(h * 0.42)),
                    (x + int(w * 0.16), top + int(h * 0.58)),
                ],
            )
            pygame.draw.polygon(
                self.screen,
                self._mix_color(color, (207, 199, 174), 0.20),
                [
                    (x + int(w * 0.47), top + int(h * 0.42)),
                    points[8],
                    points[9],
                    points[10],
                    points[12],
                    (x + int(w * 0.30), top + int(h * 0.78)),
                ],
            )

    def _draw_mountain_mist(self, horizon: int) -> None:
        mist = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        spacing_m = 2.75
        parallax = 0.30
        for i in self._visible_decor_indices(spacing_m, parallax, 1.6):
            if i % 3 == 1:
                continue
            x = self._parallax_x(i * spacing_m + 0.45, parallax)
            y = horizon - 140 + (abs(i) % 3) * 22
            points = [
                (x - 118, y + 18),
                (x - 70, y - 8),
                (x + 12, y - 3),
                (x + 88, y + 11),
                (x + 126, y + 32),
                (x + 54, y + 48),
                (x - 76, y + 42),
            ]
            pygame.draw.polygon(mist, (250, 247, 224, 68), points)
            pygame.draw.polygon(
                mist,
                (255, 255, 245, 46),
                [(x - 82, y + 15), (x - 31, y - 6), (x + 55, y + 9), (x + 10, y + 31)],
            )
        self.screen.blit(mist, (0, 0))

    def _draw_low_poly_plants(self, horizon: int) -> None:
        plant_spacing_m = 0.72
        for i in self._visible_decor_indices(plant_spacing_m, 1.0, 0.8):
            x = self.world_to_screen(i * plant_spacing_m + 0.18, 0.0)[0]
            base = horizon + 26 + (abs(i) % 3) * 9
            color = (61, 151, 91) if i % 2 else (79, 173, 102)
            self._draw_grass_clump(x + 18, base + 32, color, i)
            if i % 2 == 0:
                self._draw_stylized_flower(x + 18, base + 9, i)

        rock_spacing_m = 1.05
        for i in self._visible_decor_indices(rock_spacing_m, 1.0, 1.0):
            x = self.world_to_screen(i * rock_spacing_m + 0.52, 0.0)[0]
            y = horizon + 35 + (abs(i + 1) % 2) * 13
            width = 54 + (abs(i) % 3) * 10
            height = 24 + (abs(i + 2) % 3) * 5
            shadow = pygame.Rect(x - 4, y + height - 5, width + 10, 9)
            pygame.draw.ellipse(self.screen, (55, 113, 70), shadow)
            outline = [
                (x, y + height),
                (x + 8, y + height // 3),
                (x + width // 3, y),
                (x + width - 12, y + 4),
                (x + width, y + height - 3),
                (x + width * 2 // 3, y + height + 4),
            ]
            pygame.draw.polygon(
                self.screen,
                (82, 116, 111),
                outline,
            )
            pygame.draw.polygon(
                self.screen,
                (151, 184, 158),
                [(x + 8, y + height // 3), (x + width // 3, y), (x + width // 2, y + height - 2), (x + 12, y + height)],
            )
            pygame.draw.polygon(
                self.screen,
                (62, 91, 96),
                [(x + width // 2, y + height - 2), (x + width - 12, y + 4), (x + width, y + height - 3), (x + width * 2 // 3, y + height + 4)],
            )
            pygame.draw.line(
                self.screen,
                (59, 85, 84),
                (x + width // 2 + 3, y + height // 3),
                (x + width // 2 - 5, y + height - 2),
                2,
            )

    def _draw_grass_clump(
        self,
        root_x: int,
        root_y: int,
        color: tuple[int, int, int],
        seed: int,
    ) -> None:
        shadow = pygame.Rect(root_x - 17, root_y - 4, 35, 8)
        pygame.draw.ellipse(self.screen, (50, 127, 73), shadow)

        blades = [
            (-15, -18, -7, 0),
            (-9, -28, -4, -2),
            (-2, -22, 0, 0),
            (5, -31, 4, -2),
            (12, -20, 7, 0),
        ]
        for index, (dx, height, bend, foot_offset) in enumerate(blades):
            blade_color = self._mix_color(color, (37, 108, 67), 0.12 * (index % 2))
            tip = (root_x + dx + bend, root_y + height)
            left = (root_x + dx - 3 + foot_offset, root_y)
            right = (root_x + dx + 4 + foot_offset, root_y)
            mid = (root_x + dx + bend // 2, root_y + height // 2)
            pygame.draw.polygon(self.screen, blade_color, [left, tip, mid, right])
            pygame.draw.line(
                self.screen,
                self._mix_color(blade_color, (228, 229, 131), 0.22),
                (root_x + dx + foot_offset, root_y - 2),
                tip,
                1,
            )

        if seed % 3 == 0:
            small_leaf = [
                (root_x - 7, root_y - 6),
                (root_x - 22, root_y - 12),
                (root_x - 29, root_y - 5),
                (root_x - 13, root_y + 1),
            ]
            pygame.draw.polygon(self.screen, self._mix_color(color, (232, 222, 127), 0.14), small_leaf)

    def _draw_stylized_flower(self, root_x: int, root_y: int, seed: int) -> None:
        sway = -5 if seed % 4 == 0 else 5
        center = (root_x + sway, root_y - 31 - (abs(seed) % 2) * 4)
        stem_mid = (root_x + sway // 2, root_y - 16)
        stem_points = [(root_x, root_y), stem_mid, (center[0], center[1] + 8)]
        pygame.draw.lines(self.screen, (36, 104, 63), False, stem_points, 3)
        pygame.draw.lines(self.screen, (92, 177, 95), False, stem_points, 1)

        leaf_color = (72, 157, 83)
        leaf_shadow = (44, 117, 68)
        left_leaf = [
            (stem_mid[0], stem_mid[1] + 3),
            (stem_mid[0] - 15, stem_mid[1] - 4),
            (stem_mid[0] - 22, stem_mid[1] + 3),
            (stem_mid[0] - 9, stem_mid[1] + 8),
        ]
        right_leaf = [
            (stem_mid[0] + 2, stem_mid[1] - 2),
            (stem_mid[0] + 15, stem_mid[1] - 10),
            (stem_mid[0] + 22, stem_mid[1] - 4),
            (stem_mid[0] + 10, stem_mid[1] + 3),
        ]
        pygame.draw.polygon(self.screen, leaf_shadow, [(x + 1, y + 2) for x, y in left_leaf])
        pygame.draw.polygon(self.screen, leaf_color, left_leaf)
        if seed % 3 == 0:
            pygame.draw.polygon(self.screen, leaf_shadow, [(x + 1, y + 2) for x, y in right_leaf])
            pygame.draw.polygon(self.screen, self._mix_color(leaf_color, (176, 213, 101), 0.18), right_leaf)

        palette = [
            ((255, 111, 118), (255, 164, 132)),
            ((255, 205, 73), (255, 234, 125)),
            ((214, 123, 232), (246, 170, 235)),
        ]
        petal_color, petal_light = palette[abs(seed) % len(palette)]
        petal_dark = self._mix_color(petal_color, (114, 63, 91), 0.28)
        petal_count = 7
        for petal_index in range(petal_count):
            angle = -math.pi / 2.0 + petal_index * 2.0 * math.pi / petal_count
            outer = (
                center[0] + int(math.cos(angle) * 16),
                center[1] + int(math.sin(angle) * 14),
            )
            side_a = (
                center[0] + int(math.cos(angle - 0.34) * 7),
                center[1] + int(math.sin(angle - 0.34) * 6),
            )
            side_b = (
                center[0] + int(math.cos(angle + 0.34) * 7),
                center[1] + int(math.sin(angle + 0.34) * 6),
            )
            crease = (
                center[0] + int(math.cos(angle) * 7),
                center[1] + int(math.sin(angle) * 6),
            )
            petal = [side_a, outer, side_b, crease]
            pygame.draw.polygon(self.screen, petal_dark, [(x + 1, y + 2) for x, y in petal])
            pygame.draw.polygon(self.screen, petal_color, petal)
            pygame.draw.polygon(self.screen, petal_light, [side_a, outer, crease])

        pygame.draw.circle(self.screen, (107, 75, 56), (center[0] + 1, center[1] + 2), 6)
        pygame.draw.circle(self.screen, (255, 221, 92), center, 6)
        pygame.draw.circle(self.screen, (255, 246, 172), (center[0] - 2, center[1] - 2), 2)
        pygame.draw.circle(self.screen, (136, 93, 58), center, 6, 1)


    def _draw_organic_segment(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        accent: tuple[int, int, int],
        width: int,
        segment_index: int,
    ) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return

        base = self._mix_color((255, 195, 118), accent, 0.22)
        shadow = self._mix_color(base, (76, 63, 56), 0.38)
        highlight = self._mix_color(base, (255, 250, 214), 0.42)
        pygame.draw.line(self.screen, shadow, start, end, width + 7)
        pygame.draw.line(self.screen, base, start, end, width + 3)
        pygame.draw.line(self.screen, highlight, start, end, max(2, width // 3))

        ux = dx / length
        uy = dy / length
        nx = -uy
        ny = ux
        stripe_count = max(2, int(length // 34))
        stripe = self._mix_color(base, (130, 86, 64), 0.36)
        for i in range(1, stripe_count + 1):
            t = i / (stripe_count + 1)
            wobble = math.sin(t * math.pi * 2.0 + segment_index) * width * 0.18
            cx = start[0] + dx * t + nx * wobble
            cy = start[1] + dy * t + ny * wobble
            half = width * 0.42
            a = (int(cx + nx * half), int(cy + ny * half))
            b = (int(cx - nx * half), int(cy - ny * half))
            pygame.draw.line(self.screen, stripe, a, b, 2)

    def _draw_organic_joint(
        self,
        center: tuple[int, int],
        radius: int,
        index: int,
        fixed: bool,
    ) -> None:
        palette = [
            (255, 158, 113),
            (255, 199, 101),
            (112, 205, 161),
            (97, 190, 221),
            (197, 159, 232),
        ]
        color = (126, 131, 127) if fixed else palette[index % len(palette)]
        shadow = self._mix_color(color, (63, 50, 48), 0.42)
        highlight = self._mix_color(color, (255, 251, 220), 0.45)
        pygame.draw.circle(self.screen, shadow, (center[0] + 2, center[1] + 3), radius + 4)
        pygame.draw.circle(self.screen, color, center, radius + 2)
        pygame.draw.circle(self.screen, highlight, (center[0] - radius // 3, center[1] - radius // 3), max(2, radius // 3))
        pygame.draw.circle(self.screen, self._mix_color(color, (90, 58, 58), 0.35), center, radius + 2, 2)

    def _draw_ground(self) -> None:
        ground_y = self.world_to_screen(0, 0)[1]
        pygame.draw.rect(
            self.screen,
            (87, 180, 91),
            (0, ground_y, self.width, self.height - ground_y),
        )
        pygame.draw.rect(
            self.screen,
            (122, 206, 98),
            (0, ground_y, self.width, 18),
        )
        pygame.draw.line(self.screen, (247, 229, 128), (0, ground_y), (self.width, ground_y), 3)
        self._draw_ground_graduations(ground_y)

        for y in range(ground_y + 42, self.height, 42):
            color = (79, 165, 82) if ((y - ground_y) // 42) % 2 == 0 else (84, 172, 86)
            pygame.draw.rect(self.screen, color, (0, y, self.width, 18))

    def _draw_ground_graduations(self, ground_y: int) -> None:
        visible_min_x = (0 - self.offset_x) / self.scale
        visible_max_x = (self.width - self.offset_x) / self.scale
        start_meter = math.floor(visible_min_x) - 1
        end_meter = math.ceil(visible_max_x) + 1
        zero_screen_x = self.world_to_screen(0.0, 0.0)[0]

        rail_y = ground_y + 22
        pygame.draw.line(
            self.screen,
            (61, 124, 76),
            (0, rail_y + 2),
            (self.width, rail_y + 2),
            5,
        )
        pygame.draw.line(
            self.screen,
            (255, 236, 150),
            (0, rail_y),
            (self.width, rail_y),
            3,
        )

        for half_meter in range(
            math.floor(visible_min_x * 2) - 1, math.ceil(visible_max_x * 2) + 2
        ):
            if half_meter % 2 == 0:
                continue
            world_x = half_meter * 0.5
            x = self.world_to_screen(world_x, 0.0)[0]
            if -20 <= x <= self.width + 20:
                pygame.draw.line(
                    self.screen,
                    (52, 121, 70),
                    (x, rail_y - 8),
                    (x, rail_y + 9),
                    2,
                )

        for meter in range(start_meter, end_meter + 1):
            x = self.world_to_screen(float(meter), 0.0)[0]
            if x < -40 or x > self.width + 40:
                continue

            is_zero = meter == 0
            is_major = meter % 5 == 0
            tick_top = rail_y - (28 if is_zero or is_major else 20)
            tick_bottom = rail_y + (18 if is_zero or is_major else 13)
            line_color = (255, 248, 205) if is_zero else (40, 106, 66)
            shadow_color = (42, 103, 62)
            pygame.draw.line(self.screen, shadow_color, (x + 2, tick_top + 2), (x + 2, tick_bottom + 2), 4)
            pygame.draw.line(self.screen, line_color, (x, tick_top), (x, tick_bottom), 3)

            if is_major or is_zero:
                label = self.small_font.render(f"{meter} m", True, (74, 82, 68))
                pad_x = 7
                pad_y = 3
                label_rect = pygame.Rect(
                    x - label.get_width() // 2 - pad_x,
                    rail_y - 48,
                    label.get_width() + pad_x * 2,
                    label.get_height() + pad_y * 2,
                )
                pole_top = label_rect.bottom - 2
                pole_bottom = rail_y + 18
                pygame.draw.line(
                    self.screen,
                    (120, 91, 66),
                    (x + 2, pole_top),
                    (x + 2, pole_bottom),
                    3,
                )
                flag_points = [
                    (label_rect.left, label_rect.top),
                    (label_rect.right + 8, label_rect.top + 5),
                    (label_rect.right, label_rect.bottom),
                    (label_rect.left, label_rect.bottom),
                ]
                shadow_points = [(px + 2, py + 3) for px, py in flag_points]
                pygame.draw.polygon(self.screen, (69, 136, 79), shadow_points)
                pygame.draw.polygon(self.screen, (255, 247, 199), flag_points)
                pygame.draw.lines(
                    self.screen,
                    (232, 179, 80),
                    True,
                    flag_points,
                    2,
                )
                self.screen.blit(label, (label_rect.x + pad_x, label_rect.y + pad_y))

        if -30 <= zero_screen_x <= self.width + 30:
            pygame.draw.line(
                self.screen,
                (255, 246, 200),
                (zero_screen_x, rail_y - 44),
                (zero_screen_x, rail_y + 36),
                5,
            )
            pygame.draw.line(
                self.screen,
                (238, 139, 74),
                (zero_screen_x, rail_y - 44),
                (zero_screen_x, rail_y + 36),
                2,
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
        panel = pygame.Surface((320, 138), pygame.SRCALPHA)
        panel.fill((255, 250, 220, 218))
        pygame.draw.rect(panel, (242, 176, 86, 210), panel.get_rect(), 2, border_radius=8)

        sim_time = state.get("time", 0.0)
        cycle_hz = state.get("clock_hz", 0.0)
        phase = (sim_time * cycle_hz) % 1.0
        color = self._phase_color(phase)

        bar_x = 18
        bar_y = 27
        bar_w = 118
        bar_h = 14
        pygame.draw.rect(panel, (255, 226, 151), (bar_x, bar_y, bar_w, bar_h), border_radius=7)
        fill_w = max(4, int(bar_w * phase))
        pygame.draw.rect(panel, color, (bar_x, bar_y, fill_w, bar_h), border_radius=7)
        pygame.draw.circle(panel, (75, 91, 83), (bar_x + fill_w, bar_y + bar_h // 2), 7)

        energy_units = int(round(state.get("energy_total_j", 0.0) / 100.0))
        distance_value = state.get("distance_x_m", 0.0)
        score_value = state.get("score", 0.0)
        distance_text = f"{distance_value:.2f} m" if distance_value >= 0 else f"{distance_value:.2f} m"
        score_text = f"{score_value:.0f}" if score_value >= 0 else f"{score_value:.0f}"
        energy = self.font.render(f"{energy_units}", True, (70, 84, 78))
        cycle = self.font.render(f"{cycle_hz:.2f} Hz", True, (70, 84, 78))
        distance = self.font.render(distance_text, True, (70, 84, 78))
        score = self.font.render(score_text, True, (70, 84, 78))
        label_cycle = self.small_font.render("cycle", True, (118, 134, 116))
        label_energy = self.small_font.render("energy", True, (118, 134, 116))
        label_distance = self.small_font.render("distance", True, (118, 134, 116))
        label_score = self.small_font.render("score", True, (118, 134, 116))
        panel.blit(label_cycle, (18, 10))
        panel.blit(cycle, (18, 51))
        panel.blit(label_energy, (158, 10))
        panel.blit(energy, (158, 27))
        panel.blit(label_distance, (158, 62))
        panel.blit(distance, (158, 78))
        panel.blit(label_score, (18, 88))
        panel.blit(score, (18, 104))

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
