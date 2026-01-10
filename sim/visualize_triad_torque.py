"""Visualize a single triad (3 particles, 2 bones) with a joint angle controller.

Usage:
    python sim/visualize_triad_torque.py --target-angle 60 --amp 10 --cycle-freq 0.5

This demo creates three particles A-B-C connected by distance constraints (A-B and B-C).
It applies a spring-damper torque at the joint to drive the angle toward a target.
"""

import sys
import math
import time

sys.path.insert(0, ".")
from physics.engine import World, Particle, DistanceConstraint
from render.viewer import Viewer


def demo(
    target_angle_deg: float = 60.0,
    amp_deg: float = 0.0,
    cycle_freq: float = 0.5,
    stiffness: float = 6.0,
    damping: float = 1.2,
    gravity: float = 0.0,
    fps: int = 60,
    time_scale: float = 1.0,
):
    world = World()
    # set gravity to vertical value (0 = no gravity)
    world.gravity = (0.0, -abs(gravity))
    world.dt = 1 / 240.0

    # build triad: A -- B -- C
    a = Particle(-0.2, 0.0, 0.0, 0.0, 1.0)
    b = Particle(0.2, 0.0, 0.0, 0.0, 1.0)
    c = Particle(0.2, 0.4, 0.0, 0.0, 1.0)
    world.add_particle(a)
    world.add_particle(b)
    world.add_particle(c)

    L_ab = math.hypot(b.x - a.x, b.y - a.y)
    L_bc = math.hypot(c.x - b.x, c.y - b.y)
    cab = DistanceConstraint(a, b, L_ab)
    cbc = DistanceConstraint(b, c, L_bc)
    world.add_constraint(cab)
    world.add_constraint(cbc)

    viewer = Viewer(world)

    start = time.time()
    base_angle = math.radians(target_angle_deg)
    amp = math.radians(amp_deg)

    def step():
        t = time.time() - start
        dt = world.dt * time_scale
        target = base_angle + amp * math.sin(2.0 * math.pi * cycle_freq * t)
        world.apply_joint_angle_pd(b, a, c, target, stiffness, damping, dt)
        world.step(dt)

    viewer.run_loop(step, fps=fps)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--target-angle", type=float, default=60.0, help="Target angle (deg)")
    p.add_argument("--amp", type=float, default=0.0, help="Angle oscillation amplitude (deg)")
    p.add_argument("--cycle-freq", type=float, default=0.5, help="Oscillation frequency (Hz)")
    p.add_argument("--stiffness", type=float, default=6.0, help="Angle spring stiffness")
    p.add_argument("--damping", type=float, default=1.2, help="Angle damping")
    p.add_argument("--gravity", type=float, default=0.0, help="Gravity magnitude (set 0 to disable)")
    p.add_argument("--fps", type=int, default=60, help="Viewer FPS (lower = slower wall-clock)")
    p.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Scale physics timestep (e.g. 0.25 = 4x slower)",
    )

    args = p.parse_args()
    demo(
        target_angle_deg=args.target_angle,
        amp_deg=args.amp,
        cycle_freq=args.cycle_freq,
        stiffness=args.stiffness,
        damping=args.damping,
        gravity=args.gravity,
        fps=args.fps,
        time_scale=args.time_scale,
    )
