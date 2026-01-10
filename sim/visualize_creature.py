"""Visualize a creature with muscles highlighted (activation/force).

Runs the creature simulation and shows muscles as colored lines (red ~ high activation).
"""

import sys
import time

sys.path.insert(0, ".")
from physics.engine import World
from creature.genome import default_genome
from creature.creature import Creature
from render.viewer import Viewer


def demo(duration=10.0, dt=1 / 240.0):
    world = World()
    world.dt = dt
    genome = default_genome()
    creature = Creature(genome, world, base_x=0.0)

    viewer = Viewer(world, creature=creature)

    t = 0.0

    def step():
        nonlocal t
        creature.step_controller(t, dt)
        creature.step_actuators(t, dt)
        world.step(dt)
        t += dt

    # run viewer loop (close window to stop)
    viewer.run_loop(step)


if __name__ == "__main__":
    demo()
