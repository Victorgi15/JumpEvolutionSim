"""Visualize thrown particles to verify gravity."""

import random
import sys

sys.path.insert(0, ".")
from physics.engine import World, Particle
from render.viewer import Viewer


def demo(n=10):
    world = World()
    # spawn particles above ground with upward velocities
    for i in range(n):
        x = (i - n / 2) * 0.2
        y = 1.0 + random.uniform(0.0, 0.5)
        p = Particle(x, y, random.uniform(-0.5, 0.5), random.uniform(2.0, 5.0), 1.0)
        world.add_particle(p)

    viewer = Viewer(world)

    def step():
        world.step(world.dt)

    viewer.run_loop(step)


if __name__ == "__main__":
    demo()
