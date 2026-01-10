"""Visualize a creature with joint actuators highlighted (activation/torque)."""

import sys
sys.path.insert(0, ".")
from physics.engine import World
from creature.genome import default_genome
from creature.creature import Creature
from creature.humanoid import HumanoidCreature
from render.viewer import Viewer
import argparse


def demo(creature_type="humanoid", duration=10.0, dt=1 / 240.0):
    world = World()
    world.dt = dt
    if creature_type == "humanoid":
        creature = HumanoidCreature({}, world, base_x=0.0)
    else:
        genome = default_genome()
        creature = Creature(genome, world, base_x=0.0)
    print(f"Visualizing creature type: {creature_type}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["chain", "humanoid"], default="humanoid")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=1 / 240.0)
    args = parser.parse_args()
    demo(creature_type=args.type, duration=args.duration, dt=args.dt)
