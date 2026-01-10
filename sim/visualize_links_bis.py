"""Visualize triads: groups of 3 points (two bones) to inspect joint behavior.

Usage:
    python sim/visualize_links_bis.py --groups 6 --spacing 0.8 --seed 1

This script creates several independent 3-point chains (A - B - C) and draws them with the
existing Viewer so you can observe links, constraints and simple dynamics.
"""

import sys
import random

sys.path.insert(0, ".")
from physics.engine import World, Particle, DistanceConstraint
from render.viewer import Viewer


def demo(groups=6, spacing=0.8, seed=None, pinned_top=False):
    rng = random.Random(seed)
    world = World()
    for i in range(groups):
        cx = (i - groups / 2) * spacing
        # create a short chain: pelvis-like base -> mid -> tip
        p0 = Particle(cx, 1.0, rng.uniform(-0.2, 0.2), rng.uniform(0.0, 0.5), 1.0)
        p1 = Particle(cx, 1.4, rng.uniform(-0.2, 0.2), rng.uniform(0.0, 0.5), 1.0)
        p2 = Particle(
            cx + rng.uniform(-0.2, 0.2),
            1.8,
            rng.uniform(-0.2, 0.2),
            rng.uniform(0.0, 0.4),
            1.0,
        )
        world.add_particle(p0)
        world.add_particle(p1)
        world.add_particle(p2)
        # link lengths (approx)
        L01 = ((p0.x - p1.x) ** 2 + (p0.y - p1.y) ** 2) ** 0.5
        L12 = ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5
        # add constraints (bones)
        c01 = DistanceConstraint(p0, p1, L01, stiffness=1.0, compliance=0.0)
        c12 = DistanceConstraint(p1, p2, L12, stiffness=1.0, compliance=0.0)
        world.add_constraint(c01)
        world.add_constraint(c12)
        # optionally pin top particle to see joint rotation
        if pinned_top:
            p2.inv_mass = 0.0

    viewer = Viewer(world)

    def step():
        # small horizontal nudges to activate motion
        for p in world.particles:
            if p.y > 1.1 and p.y < 1.9:
                p.apply_impulse(0.005 * (rng.random() - 0.5), 0.0)
        world.step(world.dt)

    viewer.run_loop(step)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--groups", type=int, default=6, help="Number of triads to display")
    p.add_argument(
        "--spacing", type=float, default=0.8, help="Horizontal spacing between groups"
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument(
        "--pinned-top",
        dest="pinned_top",
        action="store_true",
        help="Pin the top particle of each triad",
    )

    args = p.parse_args()

    demo(
        groups=args.groups,
        spacing=args.spacing,
        seed=args.seed,
        pinned_top=args.pinned_top,
    )
