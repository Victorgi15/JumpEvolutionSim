"""Run a simple simulation for 5 seconds with a default genome creature and report max COM height."""

import time
from physics.engine import World, Particle
from creature.genome import default_genome
from creature.creature import Creature


def run(duration=5.0, dt=1 / 240.0):
    world = World()
    world.dt = dt
    # build creature at x=0
    genome = default_genome()
    c = Creature(genome, world, base_x=0.0)

    t = 0.0
    steps = int(duration / dt)
    max_y = -1e9
    energy = 0.0
    for i in range(steps):
        # controller update
        c.step_controller(t, dt)
        # step world
        world.step(dt)
        com = world.center_of_mass()
        max_y = max(max_y, com[1])
        t += dt

    print(f"Max COM height over {duration}s: {max_y:.4f} m")
    return max_y


if __name__ == "__main__":
    run()
