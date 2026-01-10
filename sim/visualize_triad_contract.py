"""Visualize a single triad and softly contract the joint angle toward a target."""

import sys
import math
sys.path.insert(0, ".")
from physics.engine import World, Particle, DistanceConstraint
from render.viewer import Viewer


def demo(
    target_angle_deg: float = 60.0,
    stiffness: float = 6.0,
    damping: float = 1.2,
    gravity: float = 0.0,
    start_y: float = 0.6,
    fps: int = 60,
    time_scale: float = 1.0,
):
    world = World()
    world.gravity = (0.0, -abs(gravity))
    world.dt = 1 / 240.0

    # build triad: A -- B -- C (joint at B), start above ground
    a = Particle(-0.2, start_y, 0.0, 0.0, 1.0)
    b = Particle(0.2, start_y, 0.0, 0.0, 1.0)
    c = Particle(0.2, start_y + 0.4, 0.0, 0.0, 1.0)
    world.add_particle(a)
    world.add_particle(b)
    world.add_particle(c)

    L_ab = math.hypot(b.x - a.x, b.y - a.y)
    L_bc = math.hypot(c.x - b.x, c.y - b.y)
    world.add_constraint(DistanceConstraint(a, b, L_ab))
    world.add_constraint(DistanceConstraint(b, c, L_bc))

    viewer = Viewer(world)
    target_angle = math.radians(target_angle_deg)
    def step():
        dt = world.dt * time_scale
        world.apply_joint_angle_pd(b, a, c, target_angle, stiffness, damping, dt)
        world.step(dt)

    viewer.run_loop(step, fps=fps)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--target-angle", type=float, default=60.0, help="Target angle (deg)")
    p.add_argument("--stiffness", type=float, default=6.0, help="Angle spring stiffness")
    p.add_argument("--damping", type=float, default=1.2, help="Angle damping")
    p.add_argument("--gravity", type=float, default=0.0, help="Gravity magnitude (0 disables)")
    p.add_argument("--start-y", type=float, default=0.6, help="Initial height of the joint")
    p.add_argument("--fps", type=int, default=60, help="Viewer FPS")
    p.add_argument("--time-scale", type=float, default=1.0, help="Scale physics timestep")

    args = p.parse_args()
    demo(
        target_angle_deg=args.target_angle,
        stiffness=args.stiffness,
        damping=args.damping,
        gravity=args.gravity,
        start_y=args.start_y,
        fps=args.fps,
        time_scale=args.time_scale,
    )
