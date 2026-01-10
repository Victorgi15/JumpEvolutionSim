"""Simple viewer for the World. Uses pygame if available, otherwise prints positions to console."""

from typing import Optional
import math

try:
    import pygame

    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False


class Viewer:
    def __init__(self, world, creature=None, width=800, height=600, scale=200.0):
        self.world = world
        self.creature = creature
        self.width = width
        self.height = height
        self.scale = scale  # meters -> pixels
        self.offset_x = width // 2
        self.offset_y = height // 8
        if PYGAME_AVAILABLE:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Sim Viewer")
            self.clock = pygame.time.Clock()

    def world_to_screen(self, x, y):
        sx = int(self.offset_x + x * self.scale)
        sy = int(self.height - (self.offset_y + y * self.scale))
        return sx, sy

    def draw(self):
        if not PYGAME_AVAILABLE:
            # console fallback
            pts = [(p.x, p.y) for p in self.world.particles]
            cons = [
                ((c.p1.x, c.p1.y), (c.p2.x, c.p2.y)) for c in self.world.constraints
            ]
            print("Particles:")
            for i, (x, y) in enumerate(pts):
                print(f"  {i}: ({x:.2f}, {y:.2f})")
            print("Constraints:")
            for i, ((x1, y1), (x2, y2)) in enumerate(cons):
                print(
                    f"  {i}: ({x1:.2f},{y1:.2f}) - ({x2:.2f},{y2:.2f}) len={math.hypot(x2-x1,y2-y1):.2f}"
                )
            print("---")
            return

        self.screen.fill((30, 30, 30))
        # draw ground
        gy = self.world_to_screen(0, 0)[1]
        pygame.draw.rect(
            self.screen, (50, 120, 50), (0, gy, self.width, self.height - gy)
        )

        # constraints as lines
        for c in self.world.constraints:
            a = self.world_to_screen(c.p1.x, c.p1.y)
            b = self.world_to_screen(c.p2.x, c.p2.y)
            pygame.draw.line(self.screen, (200, 200, 200), a, b, 3)
        # if a creature is provided and has muscle activation data, draw muscles
        if self.creature is not None and hasattr(self.creature, "last_activations"):
            for m in self.creature.last_activations:
                p1 = m["p1"]
                p2 = m["p2"]
                act = m["activation"]
                force = m["force"]
                a = self.world_to_screen(p1.x, p1.y)
                b = self.world_to_screen(p2.x, p2.y)
                # activation color: low=grey, high=red
                col = (
                    int(200 * act + 55 * (1 - act)),
                    int(50 * (1 - act)),
                    int(50 * (1 - act)),
                )
                width = max(1, int(1 + force / 100.0))
                pygame.draw.line(self.screen, col, a, b, width)
        # particles as circles
        for p in self.world.particles:
            s = max(3, int(5 + (0 if p.inv_mass == 0 else (1.0 / p.inv_mass)) * 0.5))
            col = (200, 50, 50) if p.inv_mass != 0 else (100, 100, 100)
            pygame.draw.circle(self.screen, col, self.world_to_screen(p.x, p.y), s)

        pygame.display.flip()

    def run_loop(self, step_callback, fps=60):
        if not PYGAME_AVAILABLE:
            print(
                "Pygame not available — using console fallback. Running 10 steps then exit."
            )
            for _ in range(10):
                step_callback()
                self.draw()
            return

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            step_callback()
            self.draw()
            self.clock.tick(fps)
        pygame.quit()
