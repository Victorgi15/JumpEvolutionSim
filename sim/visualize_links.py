"""Visualize pairs of points connected by links to verify link constraint."""

import sys
import random

sys.path.insert(0, ".")
from physics.engine import World, Particle, DistanceConstraint
from render.viewer import Viewer


def demo(pairs=5):
    world = World()
    spacing = 0.6
    for i in range(pairs):
        x = (i - pairs / 2) * spacing
        # two particles connected by a distance constraint
        # give non-zero initial velocities to test gravity and link behavior
        p1 = Particle(x, 1.5, random.uniform(-0.5, 0.5), random.uniform(0.0, 1.5), 1.0)
        p2 = Particle(x, 2.0, random.uniform(-0.5, 0.5), random.uniform(0.0, 1.0), 1.0)
        world.add_particle(p1)
        world.add_particle(p2)
        c = DistanceConstraint(p1, p2, 0.5)
        world.add_constraint(c)

    viewer = Viewer(world)

    def step():
        # small horizontal impulse to test stability
        for p in world.particles:
            if p.y > 1.4:
                p.apply_impulse(0.01, 0.0)
        world.step(world.dt)

    viewer.run_loop(step)


if __name__ == "__main__":
    demo()
